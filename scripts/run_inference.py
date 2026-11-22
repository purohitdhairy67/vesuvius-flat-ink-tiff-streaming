"""Run Villa's CLI and record timing, scratch use, and output identity."""
import argparse
from collections import defaultdict
from functools import wraps
import hashlib
import json
from pathlib import Path
import resource
import tempfile
import time

import numpy as np
import tifffile
import torch
from vesuvius.ink_detection.inference import infer

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surface")
    parser.add_argument("checkpoint")
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("ram", "spill"), default="spill")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--direction", default="both")
    parser.add_argument("--roi", type=int, nargs=4)
    parser.add_argument("--layer-start", type=int)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    timers = defaultdict(float)
    staging = []
    original_infer = infer.run_block_inference
    original_flush = infer.ChunkAccumulator._flush
    original_write = infer.write_output_tiff

    def timed(original, label):
        @wraps(original)
        def measure(*a, **kw):
            start = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                timers[label] += time.perf_counter() - start
        return measure

    def observe_scratch(*a, **kw):
        files = [p for p in scratch.glob("ink_flat_infer_*/*/**") if p.is_file()]
        staging.append({"bytes": sum(p.stat().st_size for p in files), "files": len(files)})
        return original_write(*a, **kw)

    infer.run_block_inference = timed(original_infer, "model_and_loader_with_accumulator_seconds")
    infer.ChunkAccumulator._flush = timed(original_flush, "accumulator_flush_seconds")
    infer.write_output_tiff = timed(observe_scratch, "final_tiff_and_scratch_stat_seconds")
    params = [
        args.surface, args.checkpoint, str(args.output), "--direction", args.direction,
        "--device", args.device, "--amp-dtype", "default", "--no-compile", "--num-workers", "0",
        "--batch-size", "4", "--overlap", "0.5", "--blend-mode", "hann", "--chunk-cache-size", "64",
        "--max-ram-accumulation-gb", "0" if args.mode == "spill" else "2",
    ]
    if args.roi:
        params += ["--roi", *map(str, args.roi)]
    if args.layer_start is not None:
        params += ["--layer-start", str(args.layer_start), "--layer-end", str(args.layer_start + 17)]
    with tempfile.TemporaryDirectory(prefix="ink-benchmark-") as scratch_dir:
        scratch = Path(scratch_dir)
        tempfile.tempdir = scratch_dir
        start = time.perf_counter()
        infer.main(params)
        elapsed = time.perf_counter() - start

    outputs = [args.output]
    if args.direction == "both":
        outputs.append(args.output.with_stem(args.output.stem + "_reverse"))
    images = []
    for path in outputs:
        a = tifffile.imread(path)
        images.append({
            "file": path.name, "shape": list(a.shape),
            "decoded_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "tiff_sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
            "tiff_bytes": path.stat().st_size, "nonzero_pixels": int(np.count_nonzero(a)),
            "min": int(a.min()), "max": int(a.max()),
        })
    evidence = {
        "cli": params, "torch": torch.__version__, "seconds": elapsed, **timers,
        "scratch_at_finalization": staging,
        "peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "outputs": images,
    }
    args.output.with_suffix(".json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))

if __name__ == "__main__":
    main()
