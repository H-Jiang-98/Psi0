# DreamZero G1 Neck Server/Client Rewrite Plan

Last updated: 2026-06-22

## Goal

Rewrite the DreamZero G1 neck baseline serving path so it loads the existing
`.runs/dreamzero/g1_neck_0617` model and speaks the same flat observation /
chunked action format used by `/home/weiduo/GR00T-WholeBodyControl/dz_client.py`.
The immediate target is open-loop server/client testing; the follow-up target is
smooth integration with the real robot `dz_client.py` wrapper.

## Scope

- Canonical baseline is neck-only.
- Default serving port is `48014`.
- Use the openpi/msgpack websocket protocol so the server is compatible with
  `WebsocketClientPolicy`, `g1_sonic_client.py`, and `dz_client.py`.
- Keep the existing DreamZero LoRA loading compatibility override:
  `model_dtype=float32`.
- Keep Torch Inductor cudagraphs disabled to avoid `beginAllocateToPool`
  allocation failures.
- Run one unconditional startup warmup, then reset DreamZero cache before
  accepting websocket clients.

## Non-goals

- Do not support the old PSI-X RTC JSON endpoint in the rewritten path.
- Do not support `action_only_inference`.
- Do not support fast-mode.
- Do not support non-neck 78D actions.
- Do not support requests missing `observation/neck`.
- Do not modify `/home/weiduo/GR00T-WholeBodyControl/g1_sonic_client.py`.

## Wire Protocol

The client sends one flat observation dictionary per query:

- `observation/head`: RGB image, shape `(H, W, 3)` or `(T, H, W, 3)`.
- `observation/hand_joints`: 14D hand joints.
- `observation/qpos`: 29D joints, layout `leg/base15 + arm14`.
- `observation/neck`: 2D neck joints.
- `prompt`: instruction string.
- `session_id`: session string.

The server converts each request to the DreamZero training layout:

- `video.egocentric`
- `state.joint_positions = hand14 + arm14 + leg15 + neck2`
- `annotation.human_instruction`

DreamZero raw action layout is:

- `hand14 + body_token64 + neck2`

The public server response layout must be:

- `hand14 + neck2 + token64`
- shape `(24, 80)`

The server truncates DreamZero's 32-step action chunk to the first 24 steps to
match `g1_sonic_client.py`'s `ACTION_HORIZON=24`.

## Implementation Checklist

1. Replace the current DreamZero serving entry with one canonical G1 neck server.
2. Load `GrootSimPolicy(EmbodimentTag.PSIX_G1_SONIC_NECK)`.
3. Serve with openpi/msgpack websocket protocol on port `48014` by default.
4. Validate the flat observation keys and dimensions at the server boundary.
5. Reorder incoming qpos from `leg/base15 + arm14` to DreamZero state order:
   `hand14 + arm14 + leg15 + neck2`.
6. Reorder outgoing DreamZero raw action from `hand14 + token64 + neck2` to
   `hand14 + neck2 + token64`.
7. Reset frame buffer and `current_start_frame` on reconnect or new
   `session_id`.
8. Rewrite `baselines/dreamzero/openloop_eval_g1_neck.py` to use the same
   websocket wire format as `dz_client.py`.
9. In open-loop eval, read `data/g1_neck_0617/g1`.
10. Reconstruct `observation/hand_joints`, `observation/qpos`, and
    `observation/neck` from dataset `observation.state`.
11. Compare predicted `hand14 + neck2 + token64` chunks against ground truth
    rearranged into the same layout.
12. In open-loop eval, compute action losses only. Do not add image, state, or
    auxiliary observation losses.
13. Report action losses separately for Sonic body tokens, neck, and hand/head
    action slices:
    - `sonic_token_loss`: `token64`
    - `neck_loss`: `neck2`
    - `hand_loss`: `hand14`
14. Update `/home/weiduo/GR00T-WholeBodyControl/dz_client.py` only.
15. Preserve the high-level flow of `g1_sonic_client.py --include-neck`.
16. Remove or ignore action-only behavior in the DreamZero wrapper.
17. Keep neck publishing and token quantization consistent with
    `g1_sonic_client.py`.

## Verification Plan

Static checks:

- `py_compile` the changed server.
- `py_compile baselines/dreamzero/openloop_eval_g1_neck.py`.
- `py_compile /home/weiduo/GR00T-WholeBodyControl/dz_client.py`.
- Verify `--help` exposes only the necessary baseline arguments.

Protocol smoke:

- Start a dummy policy server.
- Send one request containing `observation/head`, `observation/hand_joints`,
  `observation/qpos`, `observation/neck`, `prompt`, and `session_id`.
- Confirm the client receives an action array shaped `(24, 80)` in
  `hand14 + neck2 + token64` layout.

Model smoke:

- Start the server with `.runs/dreamzero/g1_neck_0617`.
- Run open-loop eval with `--episode-index 0 --num-queries 1`.
- Assert returned action shape is `(24, 80)`.
- Write a metrics CSV containing action-only losses split into
  `sonic_token_loss`, `neck_loss`, and `hand_loss`.

Regression checks:

- Confirm startup warmup completes before listening for clients.
- Confirm reconnect or new `session_id` resets DreamZero frame buffer and
  `current_start_frame`.

## Current Status

Planning is now recorded here. Implementation has not started in this pass.
