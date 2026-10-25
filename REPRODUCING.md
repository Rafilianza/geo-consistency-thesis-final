# Reproduction guide

## Environment

The training and generation scripts were written for Linux, CUDA and Monash M3 Slurm. The recorded Wan environment used Python 3.9 and PyTorch 2.7.0; wrappers load the M3 `pytorch/2.7.0` module. The dependency list in `requirements-core.txt` is a **curated subset of the recorded environment**, not a tested installation lockfile.

Use a CUDA-compatible environment for Wan and a separate environment for CoTracker. Installing the core list alone is insufficient: the exact semantic YOLO teacher and compatible Ultralytics implementation, the CoTracker checkout and the model assets below are also needed. Their complete environment versions are not supplied by the original Wan package snapshot. Do not replace the semantic teacher with ordinary YOLO instance-segmentation weights.

The code uses `PYTHONPATH=<checkout>/src`; there is no Python package installation step. For example:

```bash
# From this repository's root, inside the intended environment:
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
```

## External inputs

| Input | What must be available |
|---|---|
| Wan | Wan 2.1 T2V 1.3B Diffusers model directory, including text encoder, transformer, VAE and scheduler assets. |
| Semantic teacher | The research `yolo26x-sem.pt` checkpoint and its compatible Ultralytics semantic interface. |
| CoTracker | External CoTracker3 checkout, its Python environment and `scaled_offline.pth`. |
| Dataset | Original and mixed dataset roots with manifests, RGB exports, frame labels, metadata and `final_latent.pt` files. |
| Codec reference | Original frozen **11-class** codec checkpoint. The selected nine-class training script loads it for reference evaluation even for a fresh run. |
| Completed codec | `prepared.json`, `review.csv`, `config.json`, `complete.json`, `best_weights.pt` and the source clips referenced by that run. Semantic preparation needs more than the checkpoint alone. |
| Frozen semantic model for RL | The audited semantic run and its normalisation/configuration/checkpoint files. The retained RL recipe binds the selected update-4600 checkpoint and exact artifact hashes. |
| LoRA/challenge inputs | Completed compatible head and LoRA study folders, including their metadata and checkpoints. MP4 files alone are insufficient. |

The selected mixed dataset has **100 clips: 40 retained original clips and 60 newly generated clips**. Keep both roots: its manifest refers to retained clips in the original dataset. Prompt-level splitting gives 80 training clips and 20 validation clips. Each clip has 81 frames at 480 × 832 pixels and 16 frames/s.

Saved metadata contains absolute paths and hashes. Copying a checkpoint to another folder without its companion files is not a supported reconstruction of the original run.

## Path and Slurm settings

The wrappers retain the research account (`gt69`), partitions, GPU requests, log paths and environment defaults. Dataset/codec/semantic GPU wrappers request an L40S; codec preparation uses the CPU `comp` partition. The retained RL and challenge GPU wrappers request A100-80G resources. Training/generation guards require a suitable single-task Slurm CUDA allocation.

`WAN_REPO_ROOT`, `WAN_VENV` and `WAN_CACHE` can change the checkout, Wan environment and cache in wrappers that support them. They **do not replace every model or data path**. In particular, RL runtime setup uses constants in `src/rgb_geometry_pilot/protocol.py`:

| Constant | Recorded default |
|---|---|
| `DEFAULT_SEMANTIC` | `/home/dhiazr/joint_semantic_mixed_v1_epoch60` |
| `DEFAULT_MODEL` | `/home/dhiazr/gt69_scratch/dhiazgeo/wanti2v/models/Wan2.1-T2V-1.3B-Diffusers` |
| `DEFAULT_YOLO` | `/home/dhiazr/gt69_scratch/dhiazgeo/wanti2v/models/Yolo26/yolo26x-sem.pt` |
| `DEFAULT_TRACK_REPO` | `/home/dhiazr/cotracker3/co-tracker` |
| `DEFAULT_TRACK_PYTHON` | `/home/dhiazr/cotracker3/envcotrack/bin/python` |
| `DEFAULT_TRACK_WEIGHTS` | `/home/dhiazr/cotracker3/checkpoints/scaled_offline.pth` |

The fixed RL configs check model inventories, hashes and experiment settings; editing JSON paths does not override all these constraints. Submission helpers remove `SBATCH_*` defaults, so those variables are not an account/resource override. For another machine, adapt a separate working copy, record the changes and revalidate it before use.

## Dataset, codec and semantic workflow

The following are **M3 command templates**, not commands to run on a Mac. Replace `/absolute/...` paths, prepare the required environments and inspect resource/log settings first. Run each stage only after its prerequisites finish.

```bash
# 1. Original saved RGB/teacher-mask pairs.
sbatch jobs/datasets/create_walking_pov_100.sh \
  --out-root /absolute/data/original \
  --model-dir /absolute/models/Wan2.1-T2V-1.3B-Diffusers \
  --yolo-model /absolute/models/yolo26x-sem.pt

# 2. Selected mixed dataset: retain 40 original clips and generate 60 more.
sbatch jobs/datasets/create_walking_pov_mixed_100.sh \
  --source-root /absolute/data/original --out-root /absolute/data/mixed \
  --model-dir /absolute/models/Wan2.1-T2V-1.3B-Diffusers \
  --yolo-model /absolute/models/yolo26x-sem.pt

# 3. Prepare the selected codec data and inspect the generated review.csv.
sbatch jobs/codec_retraining/prepare.sh --dataset-kind mixed \
  --data-root /absolute/data/mixed --run-dir /absolute/runs/codec

# 4. Train only after completing the required data review.
sbatch jobs/codec_retraining/train.sh --dataset-kind mixed \
  --data-root /absolute/data/mixed --run-dir /absolute/runs/codec \
  --baseline-checkpoint /absolute/checkpoints/original_11class_codec.pt

# 5. Prepare latents, train semantic flow for 30 epochs and render examples.
sbatch jobs/joint_semantic_mixed/run.sh all \
  --codec-run /absolute/runs/codec \
  --weights-dir /absolute/runs/semantic30_weights \
  --outputs-dir /absolute/runs/semantic30_outputs \
  --model-dir /absolute/models/Wan2.1-T2V-1.3B-Diffusers \
  --yolo-model /absolute/models/yolo26x-sem.pt

# 6. Continue to 60 total epochs (4,800 total updates), then render examples.
sbatch jobs/joint_semantic_mixed/run.sh extend \
  --continue-from /absolute/runs/semantic30_weights \
  --codec-run /absolute/runs/codec \
  --weights-dir /absolute/runs/semantic60_weights \
  --outputs-dir /absolute/runs/semantic60_outputs \
  --model-dir /absolute/models/Wan2.1-T2V-1.3B-Diffusers \
  --yolo-model /absolute/models/yolo26x-sem.pt --updates 4800 --resume
```

The codec script defaults to the older original dataset, so the explicit `--dataset-kind mixed` matters. The epoch counts above assume the reported 80 accepted training clips: 2400 updates correspond to 30 epochs and 4800 to 60 epochs. Excluding clips during review changes this relationship. Semantic continuation retains references to the parent latent cache; keep that parent run. The historical RL model was selected at update 4600 (57.5 epochs), rather than assuming the final update was best. A newly trained codec/predictor will have new hashes and is **not automatically eligible** for the fixed historical RL configuration.

## RL and evaluation workflow

First use the three `--plan` commands in the README. They do not inspect model assets or contact Slurm. Actual submission performs additional input checks.

Only after the external frozen assets and M3 settings are prepared, the workflow is:

```bash
# A. Baseline plus RGB/semantic heads for Flash-GRPO, RLOO and Flow-GRPO.
bash jobs/rgb_geometry_qualitative/submit.sh

# B. After A finishes, use the fresh run path printed by its submission.
reference_run='/absolute/path/to/completed_head_study'
bash jobs/rgb_geometry_qualitative_lora/submit.sh \
  --reference-root "$reference_run"

# C. After B finishes, evaluate those existing weights on harder prompts.
lora_run='/absolute/path/to/completed_lora_study'
bash jobs/rgb_geometry_challenge/submit.sh \
  --reference-root "$reference_run" --lora-root "$lora_run"
```

The head study waits for its baseline and then runs at most four FIFO GPU workers. LoRA uses three independent jobs. The challenge uses four FIFO workers with no baseline barrier. The CPU report jobs wait for their associated GPU jobs. These scripts are submitted by the researcher; they do not install a scheduler-polling loop.

The RLOO, Flow-GRPO and Flash-GRPO arms retain their original learning rates and update counts. Treat those budgets as part of the reported comparison, not as identical optimisation settings. `R`, `S` and `L` denote RGB head, semantic-guided head and LoRA-only adaptation respectively.

## Historical source identities

The original protocol hashes selected source and job directories. This reduced package omits unrelated files from those directories, so its source dictionary differs from historical receipts even though each copied file is byte-identical.

- Use the **full original checkout at the recorded run revision** for exact historical replay/resume. Individual run receipts determine that revision.
- Do not disable source or input hash checks to make an old checkpoint load.
- New runs made from one unchanged reduced checkout can record their own consistent source identity, provided the required frozen assets satisfy the fixed protocol. This route has not been executed on a GPU for this package.
- Keep a compatible head study and LoRA extension together when using the challenge entry point.

## Local checks

The package verifier checks copied-file hashes, parses Python/JSON and runs Bash syntax checks. It uses only the standard library and Bash; it does not import model code.

```bash
python3 -B tools/verify_package.py

# The two retained test modules need NumPy, but no PyTorch, models or GPU.
PYTHONPATH="$PWD/src" python3 -B -m unittest discover \
  -s tests/cotracker_evaluation -p 'test_tracking_metrics.py' -v
PYTHONPATH="$PWD/src" python3 -B -m unittest discover \
  -s tests/rgb_geometry_evaluation -p 'test_metrics.py' -v
```

The tests cover synthetic tracking cases, visibility, class changes, query exclusion and missing support. The older tracking helper exposes several diagnostic metrics; the final study's label-persistence calculation is in `src/rgb_geometry_evaluation/metrics.py`. Passing these tests validates those small contracts, not teacher/tracker accuracy or geometric correctness.
