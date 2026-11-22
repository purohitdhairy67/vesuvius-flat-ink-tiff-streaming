"""Benchmark Villa's output stage using the released PHerc. 1447 w058 map.

The input tiles replay a published map. This measures only stitching and TIFF
writing. It does not run a checkpoint or estimate model throughput.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from PIL import Image
import tifffile
from vesuvius.ink_detection.inference import infer as current

def import_baseline(path):
    spec = importlib.util.spec_from_file_location("villa_baseline_infer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map_png", type=Path)
    parser.add_argument("output_tiff", type=Path)
    parser.add_argument("--baseline", type=Path, help="Path to unmodified infer.py")
    args = parser.parse_args()
    inference = import_baseline(args.baseline) if args.baseline else current
    source = np.asarray(Image.open(args.map_png).convert("L"))
    tile_side = 384
    tile_shape = (tile_side, tile_side)
    weight = inference.compute_importance_map_2d(patch_size=tile_shape, mode="hann").numpy()
    blocks = inference.iter_blocks(source.shape, patch_size=tile_side, stride=192)
    tasks = []
    for b in blocks:
        y, x, h, w = b.y0, b.x0, b.valid_h, b.valid_w
        values = source[y:y+h, x:x+w]
        tasks.append((b, values if values.any() else None))

    def consume(accumulator):
        for b, p in tasks:
            if p is None:
                accumulator.skip_tile(y0=b.y0, x0=b.x0, valid_h=b.valid_h, valid_w=b.valid_w)
            else:
                accumulator.add_tile(y0=b.y0, x0=b.x0, tile=p.astype(np.float32) / np.float32(255), tile_weights=weight[:b.valid_h, :b.valid_w])

    args.output_tiff.parent.mkdir(parents=True, exist_ok=True)
    peak_bytes = files = 0
    start = perf_counter()
    if args.baseline:
        with TemporaryDirectory(prefix="ink-replay-") as tmp:
            path = Path(tmp)
            prob = inference.open_temp_zarr_array(path / "probability.zarr", shape=source.shape, chunks=tile_shape)
            weights = inference.open_temp_zarr_array(path / "weight.zarr", shape=source.shape, chunks=tile_shape)
            accumulator = inference.ChunkAccumulator(
                shape=source.shape, chunk_shape=tile_shape, prob_sum_store=prob, weight_sum_store=weights,
                contribution_counts=inference.compute_chunk_contribution_counts(blocks, chunk_shape=tile_shape),
            )
            consume(accumulator)
            accumulator.flush_remaining()
            scratch = [p for p in path.rglob("*") if p.is_file()]
            peak_bytes = sum(p.stat().st_size for p in scratch)
            files = len(scratch)
            inference.write_output_tiff(prob, weights, args.output_tiff, tile_shape)
    else:
        inference.write_flat_output(
            output_path=args.output_tiff, output_shape=source.shape, tile_shape=tile_shape,
            blocks=blocks, max_ram_accumulation_gb=0, consume=consume,
        )
    seconds = perf_counter() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = tifffile.imread(args.output_tiff)
    print(json.dumps({
        "mode": "baseline" if args.baseline else "direct TIFF",
        "shape": list(source.shape), "scheduled_tiles": len(tasks),
        "nonempty_tiles": sum(p is not None for _, p in tasks),
        "seconds": seconds, "float_scratch_payload_bytes": peak_bytes,
        "float_scratch_files": files, "tiff_bytes": args.output_tiff.stat().st_size,
        "peak_rss_bytes_macos": rss,
        "decoded_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
        "decoded_nonzero_pixels": int(np.count_nonzero(result)),
    }))

if __name__ == "__main__":
    main()
