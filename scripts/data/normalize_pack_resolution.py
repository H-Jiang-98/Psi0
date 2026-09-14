#!/usr/bin/env python
"""Force every video in a LeRobot pack to ONE resolution, and fix meta/info.json.

Why this exists
---------------
merge_sonic_v1.py copies source mp4s byte-for-byte, so a merged pack inherits
whatever resolutions its sources had. The sonic-v1 sources are not uniform:

    uni / he_g1 / he_h1 / psi0   480x640  (H x W, aspect 1.333)   16,825 eps
    zmsonic train+val            384x672  (aspect 1.750)             712 eps

That matters because the training transform is
`v2.Resize((270, 480))` with an explicit 2-tuple, which does NOT preserve aspect
ratio -- it squashes whatever comes in to exactly 270x480. So a 480x640 frame is
stretched ~33% horizontally while a 384x672 frame is left essentially untouched
(1.750 -> 1.778). Since zmsonic is 100% of the VAL split and only 4% of TRAIN, the
model saw val geometry that almost none of its training data shared.

Normalising every video to a single resolution makes the downstream resize apply
the *same* distortion to every sample, so train and val agree again.

Frames are re-encoded (h264/yuv420p, CRF) -- this is a lossy pass over already-lossy
source, so only run it on videos that actually differ. Writes to a .tmp file and
renames, so an interrupted run never leaves a truncated mp4 in place.

Usage
-----
    python scripts/data/normalize_pack_resolution.py .data/psix_sonic_v1_train \
        --height 480 --width 640 --workers 16
    # add --dry-run to just report what differs
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def probe_dims(path: Path) -> tuple[int, int] | None:
    """(height, width) of a video's first video stream, or None if unreadable."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=height,width", "-of", "csv=p=0:s=x", str(path)],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return None
    try:
        w, h = (int(x) for x in r.stdout.strip().split("x")[:2])
        return h, w
    except ValueError:
        return None


def reencode(job: dict) -> dict:
    src = Path(job["path"])
    h, w, crf = job["height"], job["width"], job["crf"]
    tmp = src.with_suffix(".tmp.mp4")
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-i", str(src),
        "-vf", f"scale={w}:{h}:flags=bicubic",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-an",                      # these packs carry no audio
        "-movflags", "+faststart",
        str(tmp),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        return {"path": str(src), "ok": False, "err": r.stderr.strip()[-400:]}
    got = probe_dims(tmp)
    if got != (h, w):
        tmp.unlink(missing_ok=True)
        return {"path": str(src), "ok": False, "err": f"post-encode dims {got} != {(h, w)}"}
    os.replace(tmp, src)            # atomic within the same directory
    return {"path": str(src), "ok": True}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pack", help="LeRobot pack root (contains meta/ and videos/)")
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    pack = Path(a.pack)
    target = (a.height, a.width)
    vids = sorted((pack / "videos").rglob("*.mp4"))
    if not vids:
        print(f"FATAL: no mp4 under {pack}/videos", file=sys.stderr)
        return 1
    print(f"{pack.name}: scanning {len(vids)} videos for != {a.height}x{a.width} ...", flush=True)

    todo, bad, counts = [], [], {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for path, dims in zip(vids, ex.map(probe_dims, vids, chunksize=16)):
            if dims is None:
                bad.append(str(path)); continue
            counts[dims] = counts.get(dims, 0) + 1
            if dims != target:
                todo.append(str(path))
    for dims, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"   {dims[0]}x{dims[1]}: {n} videos{'  <- target' if dims == target else ''}")
    if bad:
        print(f"   UNREADABLE: {len(bad)} (e.g. {bad[0]})")
    print(f"   to re-encode: {len(todo)}")
    if a.dry_run or not todo:
        return 0 if not bad else 1

    done = failed = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(reencode, dict(path=p, height=a.height, width=a.width, crf=a.crf))
                for p in todo]
        for fu in as_completed(futs):
            r = fu.result()
            done += 1
            if not r["ok"]:
                failed += 1
                print(f"   FAIL {r['path']}: {r['err']}", flush=True)
            if done % 100 == 0 or done == len(todo):
                print(f"   {done}/{len(todo)} re-encoded ({failed} failed)", flush=True)
    if failed:
        print(f"FATAL: {failed} re-encodes failed; info.json NOT updated", file=sys.stderr)
        return 1

    # info.json declares one shape per video key; it is now true for every episode.
    ip = pack / "meta/info.json"
    info = json.load(open(ip))
    changed = []
    for key, feat in info.get("features", {}).items():
        if feat.get("dtype") == "video" and list(feat.get("shape", []))[:2] != [a.height, a.width]:
            feat["shape"] = [a.height, a.width, 3]
            changed.append(key)
    if changed:
        shutil.copy2(ip, ip.with_suffix(".json.bak"))
        json.dump(info, open(ip, "w"), indent=2)
        print(f"   info.json: set shape {[a.height, a.width, 3]} for {changed} (backup .json.bak)")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
