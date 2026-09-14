# Release Note: Psi-0 tailored for the SONIC wholebody controller

Release note for the Psi-0 SONIC checkpoints (`psi0/postpre.sonic1.0.unifolm.2609092156.40k` and
`psi0/sonic-checkpoints/multi-task.psi-dream.2609092156`) and their recipes:
[post-train](../scripts/train/psi0/posttrain-psix-unifolm-g1-sonic1.0.sh) on 50h UniFolM and
[fine-tune](../scripts/train/psi0/finetune-sonic-psi-dream-baseline.sh) on 5h Psi-Dream data.

## Motivation

Earlier Psi-0 checkpoints is trained on 36-dim `decouped wholebody control` using AMO. The checkpoints are ***sub-optimal*** to be 
finetuned on 80-dim (ie., 64-dim body-arm token + 14 hand joints + 2 dof neck) `SONIC wholebody controller` due to the action header needs predict significant different action distributions.

We first post-trained on 50H Unitree real g1 data UniFolm after we retarget its action to SONIC tokens. Then, we fine-tuned Psi-0 on 
5h in-domain data consists of 8 tasks.

Beyond the data, we made a series of improvements to the model architecture and state attention design. We also fixed mixed-precision training and add data augmentation. With all combined, we significantly boosted Psi-0's generalizability, which is comparable to the SOTA world-action-models. Stay tuned, I might write a blog [Psi-0-SONIC: how I improve the generalizability of Psi-0](#)! 

## Changed options

We heavily tuned the following knobs in the training recipes:

1. **`--model.tune-vlm`** on (default off) with **`--model.lang-backbone-lr=1e-6`** (default 1e-7),
   **`--model.vision-tower-lr=1e-5`** (default 1e-6), **`--model.mm-projector-lr=1e-4`** (default 1e-5).
   Unfreezes the Qwen3-VL backbone with per-component learning rates so vision and projector adapt to the
   G1 neck camera while the language model moves slowly.

2. **`--model.combined-temb`** on (default off), with `--model.pooled-text-encoder=clip`,
   `--model.pooled-projection-dim=768`, `--model.pooled-cache-path=clip_pooled_cache.pt`.
   Adds a frozen CLIP pooled instruction embedding to the timestep embedding so the AdaLN of every action
   block is conditioned on the **task** globally, not only through cross-attention.

3. **`--model.vlm-layer-indices 3 5 8 10 12 14 17 19 21 23 26 28`** (default none) and
   **`--model.num-blocks=12`** (default 6).
   Twelve action blocks, each cross-attending to its own VLM layer, so early blocks see low-level visual
   features and late blocks see semantic ones instead of every block reading the last layer.

4. **`--model.qk-norm=rms_norm`** (default none).
   RMS-normalises queries and keys inside the action-head attention to keep attention logits bounded during
   multi-task fine-tuning with a tuned VLM.

5. **`--model.final-layer-norm`** on (default on; set `--model.no-final-layer-norm` only to load old checkpoints).
   Final head is LayerNorm with a zero-initialised `(1+scale)` modulation instead of the legacy unbounded
   `x * scale` gate, which caused gradient-norm blowups in multi-task fine-tuning.

6. **`--model.state-as-action-token`** on (default off) and **`--model.state-null-token`** on (default off).
   The proprio state becomes token 0 of the action stream (joining action self-attention and AdaLN in every
   block), and a dropped state is replaced by a learned null token so "state missing" no longer aliases the
   zero vector's "neutral pose".

7. **`--model.state-drop-prob`** 0.5 in post-train, 0.1 in fine-tune (default 0.0).
   Per-sample probability of dropping the state during training, forcing the policy to learn a vision-only
   branch and to treat the state as a refinement rather than the step trigger.

8. **`--data.transform.model.view-aug`** on (default off), `--data.transform.model.view-aug-min-scale=0.85`,
   `--data.transform.model.view-aug-prob=1.0` (defaults 0.85, 1.0).
   Random crop of 85 to 100 percent of the frame resized back to training resolution, so a slightly shifted or
   closer camera at deploy time is in-distribution.

9. **`--data.transform.repack.state-temporal-jitter=10`** (default 0),
   **`--data.transform.repack.state-temporal-jitter-prob=0.5`** (default 1.0),
   **`--data.transform.field.state-noise-std=0.05`** (default 0.0).
   Pairs the image with a state up to 10 frames earlier or later half of the time and adds N(0, 0.05) noise
   on the normalized state, so the policy leans on vision when state and image disagree (latency, pose drift).
   All three are skipped under `no_aug`, which the val split and the deploy server force.

10. **`--model.no-rtc`** (default off) with `--model.max-delay=8`.
    Real-time chunking moved from training time to test time: the checkpoint is trained without action history. We found that `training-time rtc` tends to follow training motion blindly. After switching to `test-time rtc` the model demonstrates much better failure-recovery ability.


