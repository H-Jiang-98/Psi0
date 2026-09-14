"""Copy a LeRobot pack, re-encoding its videos with the bottom padding band removed.

The g1_sonic_lerobot_0810_merged_{train,val} head videos are 672x384 in which only the
top 376 rows are image: rows 376-383 are a black band, present in all 631 videos with no
exception. It is limited-range black (Y=16) plus codec noise, not zeros -- measured over
every video, rows 378-383 hold Y in [13,19] with ~93% exactly 16, and row 376 reaches
Y=43 because 4:2:0 chroma subsampling bleeds the last real row (375) into it. Row 375
itself is clean (mean Y 96.25 vs 96.32 at row 374), so exactly --pad-rows=8 comes off.

Left in, the band is 2.1% of frame height and survives into training: after the
--data.transform.model.resize.size 270 480 step it is ~5.6 black rows on every image.

--mode picks what happens to the crop:
    stretch  scale the height back to the original, width untouched (DEFAULT)
             672x384 -> crop 672x376 -> 672x384. The pack's declared geometry does not
             change, so meta/info.json and every consumer stay as they are; the cost is
             that the image is 384/376 = 2.1% taller than reality.
    uniform  scale both axes so the height returns to the original: 672x376 -> 686x384.
             Undistorted, but the width -- and info.json's shape -- change.
    crop     no rescale at all: the pack becomes 672x376. No resampling blur; 376 is not
             a multiple of 32.
info.json's video shape is rewritten from what actually comes out, whichever mode runs.

Re-encoding is a second lossy generation over the existing H.264; --crf 18 with the same
codec/pix_fmt/GOP defaults the originals used keeps that visually negligible at a similar
file size. Videos are the only thing rebuilt -- meta/ and data/ are copied verbatim, and
none of the stats files carry image statistics, so nothing needs recomputing.

Usage:
    python scripts/data/strip_video_padding.py SRC DST [--pad-rows 8] [--mode stretch]
        [--jobs N] [--crf 18] [--preset medium] [--limit N] [--overwrite] [--dry-run]

    python scripts/data/strip_video_padding.py \
        .data/g1_sonic_lerobot_0810_merged_train \
        .data/g1_sonic_lerobot_0810_merged_train_nopad
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

VIDEO_DIR = "videos"


def probe(path: Path) -> dict:
    """width/height/nb_frames of a video's first stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_frames,r_frame_rate,pix_fmt", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(out.stdout)["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        # nb_frames is absent for some containers; fall back to counting packets.
        "frames": int(stream["nb_frames"]) if stream.get("nb_frames") else count_frames(path),
        "fps": stream.get("r_frame_rate"),
        "pix_fmt": stream.get("pix_fmt"),
    }


def count_frames(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip())


def out_size(width: int, height: int, pad: int, mode: str) -> tuple[int, int]:
    """Size after cropping `pad` rows and applying `mode`."""
    cropped = height - pad
    if mode == "crop":
        return width, cropped
    if mode == "stretch":
        return width, height
    if mode == "uniform":
        # Keep the crop's aspect ratio and put the height back; even width for yuv420p.
        return int(round(width * height / cropped / 2)) * 2, height
    raise ValueError(mode)


def convert(job: tuple[str, str, int, str, int, str, int]) -> dict:
    src, dst, pad, mode, crf, preset, threads = job
    src_path, dst_path = Path(src), Path(dst)
    info = probe(src_path)
    w, h = info["width"], info["height"]
    if h <= pad:
        return {"src": src, "ok": False, "error": f"height {h} <= --pad-rows {pad}"}
    ow, oh = out_size(w, h, pad, mode)

    # Encode beside the target and rename, so an interrupted run leaves no half video
    # that a later --overwrite-less run would mistake for finished work.
    tmp_path = dst_path.with_suffix(".partial.mp4")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src_path),
        "-vf", f"crop={w}:{h - pad}:0:0,scale={ow}:{oh}",
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
        "-pix_fmt", info["pix_fmt"] or "yuv420p",
        "-vsync", "0",           # keep every source frame; parquet rows must still line up
        "-an", "-threads", str(threads), str(tmp_path),
    ]
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        return {"src": src, "ok": False, "error": proc.stderr.strip()[-400:]}

    got = probe(tmp_path)
    if (got["width"], got["height"]) != (ow, oh):
        tmp_path.unlink(missing_ok=True)
        return {"src": src, "ok": False,
                "error": f"got {got['width']}x{got['height']}, want {ow}x{oh}"}
    if got["frames"] != info["frames"]:
        tmp_path.unlink(missing_ok=True)
        return {"src": src, "ok": False,
                "error": f"frame count {got['frames']} != source {info['frames']}"}
    tmp_path.replace(dst_path)
    return {"src": src, "ok": True, "size": (ow, oh), "frames": got["frames"],
            "bytes": dst_path.stat().st_size, "secs": time.time() - started}


def copy_non_video(src: Path, dst: Path) -> None:
    """Everything but videos/ -- meta/ and data/ are unaffected by the crop."""
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(VIDEO_DIR), dirs_exist_ok=True)


def update_info(dst: Path, sizes: dict[str, tuple[int, int]]) -> list[str]:
    """Rewrite features[<video key>]['shape'] to what was actually written."""
    info_path = dst / "meta/info.json"
    if not info_path.exists() or not sizes:
        return []
    info = json.loads(info_path.read_text())
    changed = []
    for key, (w, h) in sizes.items():
        feature = info.get("features", {}).get(key)
        if not isinstance(feature, dict) or feature.get("dtype") != "video":
            continue
        shape = feature.get("shape")
        want = [h, w, shape[2] if isinstance(shape, list) and len(shape) == 3 else 3]
        if shape != want:
            feature["shape"] = want
            changed.append(f"{key}: {shape} -> {want}")
    if changed:
        shutil.copy2(info_path, info_path.with_suffix(".json.pre_nopad.bak"))
        info_path.write_text(json.dumps(info, indent=4) + "\n")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--pad-rows", type=int, default=8, help="rows to drop from the bottom")
    ap.add_argument("--mode", choices=["stretch", "uniform", "crop"], default="stretch")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--jobs", type=int, default=min(32, (os.cpu_count() or 8) // 4))
    ap.add_argument("--threads", type=int, default=2, help="x264 threads per job")
    ap.add_argument("--limit", type=int, default=0, help="convert only the first N videos")
    ap.add_argument("--overwrite", action="store_true", help="redo videos already in dst")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not (args.src / "meta/info.json").exists():
        raise SystemExit(f"{args.src} is not a LeRobot pack (no meta/info.json)")
    videos = sorted((args.src / VIDEO_DIR).rglob("*.mp4"))
    if not videos:
        raise SystemExit(f"no videos under {args.src / VIDEO_DIR}")
    if args.limit:
        videos = videos[:args.limit]

    first = probe(videos[0])
    ow, oh = out_size(first["width"], first["height"], args.pad_rows, args.mode)
    print(f"{len(videos)} videos  {first['width']}x{first['height']} "
          f"-> crop {first['width']}x{first['height'] - args.pad_rows} "
          f"-> {args.mode} {ow}x{oh}   crf={args.crf} preset={args.preset} "
          f"jobs={args.jobs}x{args.threads}t")
    if args.dry_run:
        print("dry run, nothing written")
        return

    copy_non_video(args.src, args.dst)
    jobs = []
    for src_path in videos:
        dst_path = args.dst / src_path.relative_to(args.src)
        if dst_path.exists() and not args.overwrite:
            continue
        jobs.append((str(src_path), str(dst_path), args.pad_rows, args.mode,
                     args.crf, args.preset, args.threads))
    skipped = len(videos) - len(jobs)
    if skipped:
        print(f"{skipped} already present, skipped (use --overwrite to redo)")

    done, failed, total_bytes, sizes = 0, [], 0, {}
    started = time.time()
    with ProcessPoolExecutor(args.jobs) as ex:
        for future in as_completed([ex.submit(convert, j) for j in jobs]):
            r = future.result()
            if not r["ok"]:
                failed.append(r)
                print(f"  FAIL {Path(r['src']).name}: {r['error']}", file=sys.stderr)
                continue
            done += 1
            total_bytes += r["bytes"]
            # video key is the directory the file sits in: videos/chunk-XXX/<key>/ep.mp4
            sizes[Path(r["src"]).parent.name] = r["size"]
            if done % 50 == 0 or done == len(jobs):
                rate = done / max(time.time() - started, 1e-9)
                print(f"  {done}/{len(jobs)}  {rate:.1f} vid/s  "
                      f"{total_bytes / 1e6:.0f} MB written")

    for note in update_info(args.dst, sizes):
        print(f"info.json {note}")
    print(f"converted {done}/{len(jobs)} in {time.time() - started:.0f}s"
          + (f", {len(failed)} FAILED" if failed else ""))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
