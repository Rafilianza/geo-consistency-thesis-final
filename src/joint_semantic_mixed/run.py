#!/usr/bin/env python3
"""Prepare, train or sample the new nine-class joint semantic branch."""

import argparse
import fcntl
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_semantic_mixed.common import (
    DEFAULT_CODEC, DEFAULT_OUTPUTS, DEFAULT_WAN, DEFAULT_WEIGHTS, DEFAULT_YOLO, NEW_PROMPT,
    load_prepared, require_gpu, validate_paths,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "extend", "examples", "check"))
    parser.add_argument("--continue-from", type=Path, help="Completed parent weights folder for the extend stage.")
    parser.add_argument("--codec-run", type=Path, default=Path(DEFAULT_CODEC))
    parser.add_argument("--weights-dir", type=Path, default=Path(DEFAULT_WEIGHTS))
    parser.add_argument("--outputs-dir", type=Path, default=Path(DEFAULT_OUTPUTS))
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_WAN))
    parser.add_argument("--updates", type=int, default=2400)
    parser.add_argument("--evaluate-every", type=int, default=200)
    parser.add_argument("--preview-every", type=int, default=400,
                        help="Prompt/YOLO check interval in updates (400 = five epochs with 80 clips).")
    parser.add_argument("--yolo-model", type=Path, default=Path(DEFAULT_YOLO))
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--new-prompt", default=NEW_PROMPT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "extend" and args.continue_from is None:
        parser.error("extend requires --continue-from and separate destination folders.")
    validate_paths(args)
    if min(args.updates, args.evaluate_every, args.preview_every) < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Use positive training settings.")
    if args.stage == "check":
        manifest = load_prepared(args)
        print(f"Verified {len(manifest['records'])} cached semantic and RGB latent pairs.")
        return 0
    device = require_gpu()
    # The sibling lock avoids modifying an existing source/output directory.
    args.weights_dir.parent.mkdir(parents=True, exist_ok=True)
    with args.weights_dir.with_name(args.weights_dir.name + ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "prepare":
            from joint_semantic_mixed.prepare import prepare
            prepare(args, device)
        elif args.stage in ("train", "extend"):
            if args.stage == "extend":
                from joint_semantic_mixed.continuation import prepare_continuation
                prepare_continuation(args)
            from joint_semantic_mixed.training import train
            train(args, device)
        else:
            from joint_semantic_mixed.examples import examples
            examples(args, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
