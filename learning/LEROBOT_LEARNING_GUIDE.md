# LeRobot learning guide: imitation learning, transfer learning & the models involved

Written 2026-10-02 against LeRobot commit `1e3a158e` (2026-08-03); the same commit `env/setup_env.sh` installs.
All paths below are relative to the LeRobot repo unless they start with `learning/` (= this folder).

---

## 0. Vocabulary: what kind of learning we're doing

| Term | What it means here | Do we use it? |
|---|---|---|
| **Imitation learning / behavior cloning (BC)** | Supervised learning: given (camera images + joint state) predict the action the human took. The "labels" are the leader-arm positions you record during teleop. | **Yes, this is the main plan (ACT).** |
| **Transfer learning** | Start from weights trained on *other* data, then train further on yours. | **Yes, in two places:** (a) ACT's image encoder starts from ImageNet; (b) optionally fine-tune a pretrained robot foundation model (SmolVLA). |
| **Reinforcement learning (RL)** | The robot tries things, gets a reward, improves by trial and error. | Not in the current plan. LeRobot has it (HIL-SERL / SAC, §2.4) as a possible later step. |
| **VLA (Vision-Language-Action model)** | A big pretrained vision-language model (VLM) with an "action head" attached; takes images + a text instruction, outputs robot actions. | Optional (SmolVLA fine-tuning measured at 2.4 GB here; π0-class models need a cloud GPU). |

---

## 1. The pipeline at a glance

```
teleop (leader -> follower)                 lerobot-record
   │  records each frame at 30 fps:             src/lerobot/scripts/lerobot_record.py
   │    observation.state  (6 joint positions of follower)
   │    observation.images.<cam>  (one per camera)
   │    action  (6 joint positions of leader = the "label")
   ▼
LeRobotDataset (parquet + mp4 on disk / HF Hub)   src/lerobot/datasets/
   │
   ▼
preprocessor: normalize w/ dataset stats   src/lerobot/policies/act/processor_act.py
   │
   ▼
policy.forward(batch) -> loss              src/lerobot/policies/act/modeling_act.py
   │   (training loop: src/lerobot/scripts/lerobot_train.py)
   ▼
checkpoint (weights + config + normalizer stats)
   │
   ▼
policy.select_action(obs) on the real arm  lerobot-record --policy.path=...
```

---

## 2. What models are used: names & pointers

### 2.1 The one we plan to train: **ACT** (Action Chunking with Transformers)

- **Paper:** Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (RSS 2023), the ALOHA paper. Docs: `docs/source/act.mdx`.
- **Code:** `src/lerobot/policies/act/`
- **Size:** 51.6 M parameters (measured, see §4), of which **11.2 M is the vision encoder**.

| Component | Model | Where | Pretrained? |
|---|---|---|---|
| **Vision encoder** | **ResNet18** (torchvision), `layer4` feature map | `modeling_act.py:326-334`, config `configuration_act.py:98-99` | **Yes: `ResNet18_Weights.IMAGENET1K_V1`** (ImageNet-1k classification). ← *transfer learning* |
| Image → token projection | 1×1 conv, 512 → `dim_model` (512) | `modeling_act.py:~353` | No |
| Fusion / reasoning | **Transformer encoder** (4 layers) over [latent, joint-state token, all camera feature-map tokens] + 2D sinusoidal position embeddings | `ACTEncoder` `:515`, `ACTSinusoidalPositionEmbedding2d` `:686` | No |
| Action generation | **Transformer decoder** (1 layer by default; DETR-style) with `chunk_size`=100 learned queries → 100 future actions | `ACTDecoder` `:573` | No |
| Handling multimodal demos | **CVAE** (conditional variational autoencoder): a separate small transformer encodes the true action chunk into a latent `z` during training only | `use_vae`, `kl_weight=10` in config; KL loss in `ACTPolicy.forward` `:137` | No |
| Inference smoothing (optional) | **Temporal ensembling** of overlapping chunks | `ACTTemporalEnsembler` `:167` | n/a |

Loss = L1(predicted actions, demo actions) + 10 × KL divergence (`modeling_act.py:137-165`).
**Not RL:** there is no reward anywhere; it is pure supervised imitation.

### 2.2 Other "train from scratch" imitation policies (small, fit on the 4070)

| Policy | Vision encoder | Action model | Code / docs |
|---|---|---|---|
| **Diffusion Policy** (Chi et al., RSS 2023) | ResNet18 (ImageNet) + SpatialSoftmax keypoints | 1-D conditional **U-Net** denoiser, DDPM/DDIM noise schedule | `policies/diffusion/`, `docs/source/policy_diffusion_README.md` |
| **VQ-BeT** (Lee et al., ICML 2024) | ResNet18 (ImageNet) | **VQ-VAE** action tokenizer + **GPT**-style transformer | `policies/vqbet/`, `docs/source/policy_vqbet_README.md` |
| **Multi-task DiT** (Jones, 2025) | **CLIP ViT-B/16** (`openai/clip-vit-base-patch16`), 0.1× LR | Diffusion **Transformer** | `policies/multi_task_dit/`, `docs/source/multi_task_dit.mdx` |

### 2.3 Pretrained robot foundation models (VLAs): transfer learning by fine-tuning

These ship with weights pretrained on large multi-robot datasets; you **fine-tune** on your demos
(`lerobot-train --policy.path=<checkpoint>` instead of `--policy.type=...`).

| Policy | Vision / language backbone | Action head | Base checkpoint | Fits 8 GB? |
|---|---|---|---|---|
| **SmolVLA** (Hugging Face) | **SmolVLM2-500M** (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`): **SigLIP** vision encoder + SmolLM2 language model | Flow-matching transformer "action expert" | `lerobot/smolvla_base` (pretrained on community SO-100/101 data) | **Yes (measured: 2.4 GB at batch 4, 5.4 it/s).** 450M params total, only 100M trained: by default it freezes vision (`freeze_vision_encoder=True`) and trains only the action expert (`train_expert_only=True`), `configuration_smolvla.py:69-70`. Needs `--rename_map` for camera names (see §5). |
| **π0** (Physical Intelligence) | **PaliGemma** = SigLIP + **Gemma 2B** | Flow-matching action expert (Gemma 300M) | `lerobot/pi0_base` | No (cloud GPU) |
| **π0.5** | PaliGemma | Flow matching, better open-world generalization | `lerobot/pi05_base` | No |
| **π0-FAST** | PaliGemma | Autoregressive tokens via **FAST** action tokenizer (`lerobot/fast-action-tokenizer`) | `lerobot/pi0fast-base` | No |
| **GR00T N1.7** (NVIDIA) | **Cosmos-Reason2-2B** (`nvidia/Cosmos-Reason2-2B`, Qwen3-VL based) | Diffusion-transformer action head | see `docs/source/groot.mdx` | No |
| **X-VLA** | **Florence-2** (DaViT vision encoder) + learnable "soft prompts" per robot | Flow matching | `lerobot/xvla-base` | No |
| EO-1 | Qwen2.5-VL-3B | Flow matching | `docs/source/eo1.mdx` | No |
| Evo-1 | InternVL3-1B (`OpenGVLab/InternVL3-1B-hf`) | Flow matching | `docs/source/evo1.mdx` | Maybe |
| VLA-JEPA | Qwen3-VL-2B + V-JEPA2 video world model | Flow-matching DiT | `docs/source/vla_jepa.mdx` | No |
| WALL-OSS | Qwen2.5-VL (`x-square-robot/wall-oss-flow`) | Flow matching / FAST | `docs/source/walloss.mdx` | No |
| MolmoAct2 (AI2) | Molmo (`allenai/MolmoAct2`) | FAST-tokenized actions | `docs/source/molmoact2.mdx` | No |
| LingBot-VA, Fast-WAM | **Video world models** (Wan2.2 video diffusion) predicting future video + actions | | `docs/source/lingbot_va.mdx`, `fastwam.mdx` | No |

Inference add-on: **RTC (Real-Time Chunking)**, `policies/rtc/`, smooths chunk transitions for the flow-matching VLAs (π0, π0.5, SmolVLA). Not a model itself.

**PEFT / LoRA** fine-tuning (train small adapter matrices instead of the whole network) is built in:
`--peft.*` flags, `src/lerobot/configs/default.py:130`, wiring in `lerobot_train.py:331`. This is the main lever for fitting bigger VLAs into limited GPU memory.

### 2.4 Reinforcement learning in LeRobot

| Component | Model / algorithm | Where |
|---|---|---|
| **HIL-SERL** workflow (Human-in-the-Loop Sample-Efficient RL): demos + human interventions + online RL on the real robot | Actor/learner split over gRPC | `src/lerobot/rl/` (`actor.py`, `learner.py`, `gym_manipulator.py`), `docs/source/hilserl.mdx`, `hilserl_sim.mdx` |
| RL algorithm | **SAC** (Soft Actor-Critic) | `src/lerobot/rl/algorithms/sac/` |
| Policy network for SAC | `gaussian_actor`; vision encoder optional, typically **`lerobot/resnet10`** (pretrained, frozen) | `src/lerobot/policies/gaussian_actor/` |
| Reward (success) classifier | CNN classifier on **`lerobot/resnet10`**, trained on a few labeled frames | `src/lerobot/rewards/classifier/` |
| **TD-MPC** (Hansen et al., ICML 2022) | Model-based RL: learned latent dynamics + MPC planning (sim use mostly) | `src/lerobot/policies/tdmpc/` |

### 2.5 Reward models (used for RL, or to *weight* imitation data)

| Model | Idea | Where |
|---|---|---|
| **SARM** (Stage-Aware Reward Modeling) | Video-based progress/stage reward; can re-weight BC data ("RA-BC") | `src/lerobot/rewards/sarm/`, `docs/source/sarm.mdx` |
| **TOPReward** | Zero-shot reward from a VLM's token probabilities (default `Qwen/Qwen3-VL-8B-Instruct`) | `src/lerobot/rewards/topreward/`, `docs/source/topreward.mdx` |
| Robometer | Qwen-VLM based reward/progress model | `src/lerobot/rewards/robometer/` |

### 2.6 What's actually in *our* plan

```
ACT  =  ResNet18 (ImageNet-pretrained, fine-tuned)  ->  Transformer enc/dec + CVAE (from scratch)
        trained by supervised imitation on ~50 of our own SO-101 demos. No RL, no language.
```
Later options: SmolVLA fine-tune (transfer from many SO-arm datasets, adds language conditioning), or HIL-SERL (RL to polish a single task).

---

## 3. Code reading order

1. **Tutorial scripts (short, ~30 min)**
   - `examples/tutorial/act/act_training_example.py`: dataset → policy → loop → save, in 100 lines.
   - `examples/tutorial/act/act_using_example.py`: load & run a trained policy.
2. **Data**: `src/lerobot/datasets/lerobot_dataset.py` (`LeRobotDataset`), and
   `src/lerobot/datasets/factory.py:34` `resolve_delta_timestamps`: how one sample gets *the next 100 actions* (action chunking).
3. **ACT config**: `src/lerobot/policies/act/configuration_act.py`, every knob with docstrings: `chunk_size`, `n_action_steps`, `use_vae`, `kl_weight`, `vision_backbone`, `pretrained_backbone_weights`, LRs.
4. **ACT model**: `src/lerobot/policies/act/modeling_act.py`
   - `ACT.forward` `:380`: the architecture
   - `ACTPolicy.forward` `:137`: the loss
   - `ACTPolicy.select_action` `:101`: action queue at inference
   - `:73-90`: separate optimizer param group for the backbone (`optimizer_lr_backbone`), a classic transfer-learning knob
5. **Normalization**: `src/lerobot/policies/act/processor_act.py` → `make_default_pre_post_processors`; stats come from the dataset.
6. **Real training script**: `src/lerobot/scripts/lerobot_train.py`, `train()` `:212`: dataset → `make_policy` `:325` → PEFT `:331` → processors `:394` → optimizer `:403` → loop/`update_policy` `:105,:610` → checkpoint `:689`. CLI flags map 1:1 onto config dataclasses (draccus): `--policy.chunk_size=50` sets the field in `ACTConfig`.
7. **Fine-tuning a VLA**: `src/lerobot/policies/smolvla/configuration_smolvla.py` (freeze flags) and `docs/source/smolvla.mdx`.

---

## 4. Hands-on exercises (no robot needed)

Scripts live in `learning/`. Run them from this repo's root with `conda activate so_arm101`.
Dataset used: **`lerobot/svla_so101_pickplace`**: 50 episodes, 11,939 frames, 30 fps, task *"pink lego brick into the transparent box"*, two cameras (`up`, `side`) at 640×480, same 6 joint names as our arm.

### Exercise 1: what a training sample looks like
```bash
python learning/explore_dataset.py              # optional: --frame 500, --repo-id <other dataset>
```
Prints features, normalization stats, one sample as ACT sees it, and saves the camera images to `learning/outputs/frames/`.
Things to notice (from a verified run):
- `observation.state` is shape `(6,)` but `action` is `(100, 6)`: that's the action chunk.
- `action_is_pad`: near the end of an episode the chunk runs past the last frame and is padded; the loss masks those out (`modeling_act.py:145`).
- `action[0]` ≈ `observation.state`: the leader leads the follower by a few degrees.
- Joint values are in degrees-ish units from calibration (gripper 0–100).

### Exercise 2: train ACT for a few hundred steps
```bash
python learning/train_act_mini.py --steps 500
```
Verified on this laptop: **~2.9 it/s at batch 8, peak ~4.1 GB GPU** (this simple loop; the real `lerobot-train` is faster, see §5). Batch 16 fits (7.6 GB) but runs at 1.4 it/s, i.e. the same samples/sec, so stay at batch 8.
Watch: total loss drops fast in the first ~100 steps (mostly the KL term collapsing), while `l1` falls slowly; that slow L1 curve is the real learning.

### Exercise 3: what does the ImageNet pretraining buy us?
```bash
python learning/train_act_mini.py --steps 1500                  # pretrained ResNet18
python learning/train_act_mini.py --steps 1500 --no-pretrained  # random-init ResNet18
python learning/plot_losses.py                                  # -> learning/outputs/loss_compare.png
```
(~9 min each.) Expect the pretrained run's L1 loss to drop faster. Training loss is only a proxy; the real test of transfer is success rate on the arm with few demos, which we'll see later.

### Measured results (2026-10-02)
`learning/outputs/loss_compare.png`, 1500 steps each: the pretrained backbone drops faster early (L1 ≈0.67 vs ≈0.82 at step 150), the curves cross around steps 550-750, and pretrained ends slightly lower (≈0.31 vs ≈0.36). This is a single short run each, so treat it as a small edge, not a clear win.

### Next exercises (ideas)
- Change `chunk_size` (e.g. 20 vs 100) and see how it affects loss and, later, motion smoothness.
- Freeze the backbone entirely (`requires_grad=False` on `policy.model.backbone`) and compare: that's "feature extraction" vs "fine-tuning" transfer.
- Run the real trainer on the public dataset for a short run: `lerobot-train --policy.type=act --dataset.repo_id=lerobot/svla_so101_pickplace --policy.device=cuda --batch_size=8 --steps=2000 --policy.push_to_hub=false`.

---

## 5. Test results on this laptop (2026-10-02, RTX 4070 Laptop 8 GB)

Run with `bash learning/run_all_tests.sh`; logs are in `learning/outputs/test_logs/` (`summary.txt` = one line per test).
**Prerequisite:** install the training extras into `so_arm101`. They weren't installed before 2026-10-02, and `lerobot-train` fails without `accelerate`:
`pip install -e ".[training,smolvla,diffusion,peft]" matplotlib` (run from your LeRobot clone).

| Test | Result | Time | Peak GPU | Notes |
|---|---|---|---|---|
| Ex1 explore dataset | PASS | 5 s | n/a | |
| Ex3 ACT pretrained, 1500 steps (mini loop) | PASS | 8.6 min | 4.1 GB | 2.9 it/s |
| Ex3 ACT scratch backbone, 1500 steps | PASS | 8.6 min | 4.1 GB | |
| Inference, mini checkpoint | PASS | 9 s | 0.5 GB | 18.6 ms per model call (every 100th frame), 2.6 ms otherwise; budget is 33 ms/frame. Mean action error 14.0 |
| ACT batch 16 | PASS | 1.8 min | **7.6 GB** | No throughput gain; use batch 8 |
| **`lerobot-train` ACT, 2000 steps** | PASS | 10 min | 4.6 GB | **3.4 it/s → 50k steps ≈ 4.1 h, 100k ≈ 8.2 h.** Checkpoints at 1000/2000 |
| `lerobot-train --resume=true` to 2300 | PASS | 1.7 min | 4.6 GB | Resumes data order and step; a crashed overnight run can be continued |
| Inference, `lerobot-train` checkpoint | PASS | 9 s | 0.5 GB | 17.7 ms per model call; mean action error 9.7 |
| Diffusion Policy, batch 8 | **FAIL (out of GPU memory)** | | >7.6 GB | Uses 2 frames per camera at 640×480 |
| Diffusion Policy, batch 4 | PASS | 3 min | 7.4 GB | 3.0 it/s; near the memory limit, fragile |
| SmolVLA fine-tune from `lerobot/smolvla_base`, batch 4 | PASS (2nd try) | 1.4 min | 2.4 GB | 5.4 it/s; 100M of 450M params trained |

**SmolVLA camera names:** the base model expects `observation.images.camera1/2/3`. Map your cameras with
`--rename_map='{"observation.images.up": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}'`.
For our rig that's `front` → camera1 and `wrist` → camera2. Fewer cameras than the model expects is OK (`policies/utils.py:226`).

**Not testable without the arm:** recording (`lerobot-record`), replay, and running a policy on the real robot (`lerobot-record --policy.path=...`). The real-time control loop with cameras was already exercised by the teleop test.

**Takeaways for our plan:**
- ACT at batch 8: about 4 h for 50k steps, with plenty of memory headroom.
- SmolVLA is a realistic second option on this laptop.
- Diffusion Policy only fits at batch 4, or with smaller images or cropping.
