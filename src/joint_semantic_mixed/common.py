"""Shared paths and small file/model helpers for the mixed semantic experiment."""

import csv
import importlib.metadata
import os
from pathlib import Path
import re
import uuid

import torch

from codec_retraining.data import CLASS_NAMES, LABEL_MAPPING, file_hash, fingerprint, read_json, write_json
from codec_retraining.model import LargeClassCodec

REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_CODEC = "/home/dhiazr/codec_walking_mixed100_v1"
DEFAULT_WEIGHTS = "/home/dhiazr/joint_semantic_mixed_v1"
DEFAULT_OUTPUTS = "/home/dhiazr/gt69_scratch2/dhiazgeo/joint_semantic_mixed_v1"
DEFAULT_WAN = "/home/dhiazr/gt69_scratch/dhiazgeo/wanti2v/models/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_YOLO = "/home/dhiazr/gt69_scratch/dhiazgeo/wanti2v/models/Yolo26/yolo26x-sem.pt"
SEMANTIC_SHAPE = (1, 8, 21, 60, 104)
RGB_SHAPE = (1, 16, 21, 60, 104)
FORMAT = "joint_semantic_mixed_v1"
NEW_PROMPT = (
    "First-person POV of a person walking steadily forward along a canal-side sidewalk at natural walking speed. "
    "The camera faces ahead at eye level with gentle walking movement. A red car follows a white delivery truck "
    "away along the road with a clear gap, while a blue bus approaches in the opposite lane. All three vehicles "
    "keep moving. Brick studios and planted trees line the street, with water beyond a grassy bank. "
    "Nearby pavement and buildings shift naturally as the camera advances. Realistic daylight and natural perspective. One continuous shot."
)
MONITOR_PROMPT = (
    "First-person POV of a person walking steadily forward along a broad sidewalk beside a museum road. "
    "The camera faces ahead at eye level with gentle walking movement. A yellow bus and a red car travel "
    "away in the near lane with a clear gap, while a white delivery truck approaches in the opposite lane. "
    "All three vehicles keep moving. Stone museum buildings, trees and a grass verge pass beside the walker. "
    "Natural perspective and realistic daylight. One continuous shot."
)


def require_gpu():
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_ARRAY_JOB_ID"):
        raise RuntimeError("Submit the wrapper as one GPU job without an array.")
    if os.environ.get("SLURM_NTASKS", "1") != "1" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU and one task are required.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This experiment uses bfloat16; select an L40S.")
    return torch.device("cuda")


def validate_paths(args):
    paths = [args.codec_run, args.weights_dir, args.outputs_dir, args.model_dir]
    for path in paths:
        if not path.is_absolute() or re.search(r"(?:seed|_s)0*47(?:\D|$)", str(path), re.I):
            raise ValueError("Use absolute paths outside reserved seed-47 data.")
        if re.search(r"(?:seed|_s)0*47(?:\D|$)", str(path.resolve()), re.I):
            raise ValueError("Resolved path enters reserved data.")
    for source in (args.codec_run.resolve(), args.model_dir.resolve()):
        for output in (args.weights_dir.resolve(), args.outputs_dir.resolve()):
            if output == source or source in output.parents or output in source.parents:
                raise ValueError("Keep new output folders separate from source codec/models.")
    weights, outputs = args.weights_dir.resolve(), args.outputs_dir.resolve()
    if weights == outputs or weights in outputs.parents or outputs in weights.parents:
        raise ValueError("Weights and outputs need separate, non-nested folders.")


def source_hashes():
    names = [
        "src/codec_retraining/data.py", "src/codec_retraining/model.py", "src/phase3/semantic_codec.py",
        "src/datasets/create_walking_pov_100.py", "src/datasets/create_walking_pov_mixed_100.py",
        "configs/datasets/walking_pov_100_v1.json", "configs/datasets/walking_pov_mixed_100_v1.json",
        "src/middle_feature_semantic/phase1_semantic_models.py",
        "src/middle_feature_semantic/wan_middle_feature_capture.py",
        "src/video_generation/my_wan_pipeline.py", "src/video_generation/gen_dataset.py",
        "src/yolo/yolo_labelmap_concepts.py",
    ]
    names += [str(path.relative_to(REPOSITORY)) for path in sorted((REPOSITORY / "src/joint_semantic_mixed").glob("*.py"))]
    return {name: file_hash(REPOSITORY / name) for name in names}


def atomic_torch_save(path, payload):
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex[:12]}.partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def check_tensor(tensor, shape):
    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"Expected tensor shape {shape}.")
    if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
        raise ValueError("Expected a finite floating-point tensor.")
    return tensor


def load_codec(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["class_names"] != CLASS_NAMES or payload["label_mapping"] != LABEL_MAPPING:
        raise ValueError("Expected the new nine-class codec.")
    codec = LargeClassCodec().to(device)
    codec.load_state_dict(payload["model_state_dict"], strict=True)
    codec.requires_grad_(False).eval()
    return codec


def load_prepared(args):
    manifest = read_json(args.weights_dir / "prepared.json")
    if manifest["format"] != FORMAT or manifest["sources"] != source_hashes():
        raise ValueError("Prepared inputs belong to different code; use the original checkout or a new run.")
    if manifest["outputs_dir"] != str(args.outputs_dir.resolve()):
        raise ValueError("Outputs path differs from the prepared run.")
    for name, expected in manifest["weight_files"].items():
        if file_hash(args.weights_dir / name) != expected:
            raise ValueError(f"Prepared codec file changed: {name}")
    for record in manifest["records"]:
        for key in ("semantic_latent", "rgb_latent"):
            path = Path(record[key])
            if file_hash(path) != record[key + "_sha256"]:
                raise ValueError(f"Changed latent: {path}")
    return manifest


def save_history(outputs, rows):
    write_json(outputs / "history.json", rows)
    temporary = outputs / "history.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["update", "training_loss", "validation_loss", "seconds"])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(outputs / "history.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 4))
    for key in ("training_loss", "validation_loss"):
        measured = [row for row in rows if row[key] is not None]
        axis.plot([row["update"] for row in measured], [row[key] for row in measured], label=key.replace("_", " "))
    axis.set(xlabel="CNN updates", ylabel="Semantic flow MSE", title="Training and fixed-noise validation loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(outputs / "loss.png", dpi=140)
    plt.close(figure)


def wan_inventory(model_dir):
    import importlib.util
    specification = importlib.util.spec_from_file_location(
        "joint_wan_inventory", REPOSITORY / "src/datasets/create_walking_pov_100.py")
    dataset = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(dataset)
    return dataset.model_inventory(model_dir.resolve(strict=True))


def load_wan(model_dir, device):
    from diffusers import AutoencoderKLWan
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from video_generation.my_wan_pipeline import MyWanPipeline

    class JointSemanticWanPipeline(MyWanPipeline):
        # Diffusers 0.36 may infer CPU from an offloaded text encoder or VAE.
        # Denoising must create inputs on the frozen transformer's device.
        @property
        def _execution_device(self):
            return next(self.transformer.parameters()).device

    vae = AutoencoderKLWan.from_pretrained(model_dir, subfolder="vae", torch_dtype=torch.float32, local_files_only=True)
    pipeline = JointSemanticWanPipeline.from_pretrained(model_dir, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True)
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, flow_shift=3.0)
    if not pipeline.scheduler.config.use_flow_sigmas or pipeline.config.expand_timesteps:
        raise ValueError("Expected Wan2.1 T2V1.3B scalar timesteps and a flow-sigma scheduler.")
    if len(pipeline.transformer.blocks) != 30 or tuple(pipeline.transformer.config.patch_size) != (1, 2, 2):
        raise ValueError("Unexpected Wan transformer architecture.")
    for component in (pipeline.transformer, pipeline.text_encoder, pipeline.vae):
        component.requires_grad_(False).eval()
    pipeline.set_progress_bar_config(disable=True)
    pipeline.to(device)
    return pipeline


def runtime():
    return {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers", "numpy")}
