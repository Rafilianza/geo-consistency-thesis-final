# Enhancing the Geometric Consistency of Video Generation for Interactive Environments



This repository contains the code needed to inspect the thesis approach: train a semantic codec, predict semantic maps alongside Wan 2.1 video generation, and compare RGB correction with and without semantic guidance. The three RL methods are **RLOO, Flow-GRPO and Flash-GRPO**. Each has an RGB correction head, a semantic-guided correction head and a separate rank-4 LoRA arm.

This repository contains the source code used for the thesis. 

## What is included

| Stage | Main code | Purpose |
|---|---|---|
| Dataset preparation | `src/datasets/` | Generate Wan RGB videos, save final RGB latents and obtain teacher masks. |
| Semantic codec | `src/codec_retraining/` | Train the selected nine-class mask autoencoder. |
| Joint semantic prediction | `src/joint_semantic_mixed/` | Train semantic flow on saved RGB–mask pairs and evaluate joint generation. |
| Wan features and RGB correction | `src/middle_feature_semantic/` | Capture frozen Wan features and define the correction head. |
| RLOO and shared generation | `src/rgb_geometry_campaign/` | Generate sibling videos, compute leave-one-out advantages and replay actions. |
| Flow-GRPO and Flash-GRPO | `src/rgb_geometry_rl_comparison/`, `src/rgb_geometry_rl_methods/` | Apply the local GRPO objectives and replay schedules. |
| Label persistence | `src/rgb_geometry_evaluation/`, `src/cotracker_evaluation/` | Score teacher labels along tracked points. |
| Saved-video study | `src/rgb_geometry_qualitative/`, `src/rgb_geometry_qualitative_lora/` | Train the nine adaptation arms and retain checkpoints and matched videos. |
| Difficult-scene evaluation | `src/rgb_geometry_challenge/` | Generate new videos from existing weights; no retraining. |

Supporting modules from earlier experiments are included where required by imports. Shared objective and protocol files also contain AWM/NFT branches, but their standalone experiment launchers are not included. The semantic workflow uses the semantic-flow predictor evaluated in the thesis.

## How the stages connect

1. **Prepare saved pairs.** Generate street videos with Wan 2.1 and save their final RGB latents and teacher masks.
2. **Train the codec.** Encode the masks into semantic latents, then reconstruct the masks with a separate decoder. Freeze the codec afterwards.
3. **Train semantic prediction.** Independently noise the paired RGB and semantic latents at the same noise level. Use frozen Wan block-16 features to predict semantic flow. Update only the semantic predictor.
4. **Train RGB adaptation.** Generate fresh sibling videos and use tracking-based rewards to update an RGB head, semantic-guided head or LoRA parameters. These are separate arms.
5. **Evaluate matched outputs.** Compare variants using common prompts and seeds, numerical proxies and visual inspection.

Segmentation agreement and tracked label persistence are **proxies**, not measurements of 3D geometry accuracy. A higher reward alone does not demonstrate less object deformation. These local RL implementations use a common thesis experiment setup; they do not reproduce the original papers' full benchmark recipes.

## Start here

- Read [REPRODUCING.md](REPRODUCING.md) for external files, environment requirements and workflow commands.
- See [ATTRIBUTION.md](ATTRIBUTION.md) for models, software and algorithm sources.
- Run the checks below to verify source files and preview the study plans.

```bash
# Run from this repository's root. Requires Python 3 and Bash.
python3 -B tools/verify_package.py

# These print static plans only: no Slurm calls, model loading or output files.
python3 -B jobs/rgb_geometry_qualitative/submit.py --plan
python3 -B jobs/rgb_geometry_qualitative_lora/submit.py \
  --reference-root /absolute/path/to/completed_head_study --plan
python3 -B jobs/rgb_geometry_challenge/submit.py \
  --reference-root /absolute/path/to/completed_head_study \
  --lora-root /absolute/path/to/completed_lora_study --plan
```

## Retained study outputs

| Workflow | Variants | Videos | Other retained outputs |
|---|---:|---:|---|
| Qualitative head study | Baseline + RGB/semantic for three methods | 140 | Round/recovery checkpoints, rewards, tracking records and JSON/CSV metadata |
| LoRA extension | LoRA for three methods | 60 new; 200 combined | Adapter/recovery checkpoints, rewards, tracking records and JSON/CSV metadata |
| Challenge evaluation | All ten variants | 800 | Rewards, raw tracking records and provenance; no new weights |

The first two studies use 20 matched cases per variant. The challenge uses 40 prompts and two seeds per prompt, giving 80 cases per variant. Numerical scores use decoded RGB pixels; exported MP4 videos are lossy copies for inspection. No output files are bundled here.

## Verification and limitations

Local checks cover Python/Bash syntax, JSON validity, copied-file hashes and static submission plans. **34 CPU tests** cover tracking and persistence calculations. GPU execution and a fresh environment installation have not been validated for this source package.

The scripts retain M3 paths and input/source checks. This selected file inventory differs from the full research repository, so exact historical replay or resume requires the original audited checkout. See the reproduction guide before using saved runs.
