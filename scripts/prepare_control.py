"""Prepare aligned held-out PHerc. 1447 control crop from published w058 TIFFs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image
import tifffile
import zarr

ROOT = ("https://huggingface.co/datasets/YoussefMoNader/ink-8um-pherc1447-surfaces/"
        "resolve/7e4d918712a8d8642a6fdeee45b80217c1e98317/w058/")
ROWS = slice(2880, 4416)
COLS = slice(1152, 2688)

def prepare(out: Path, download=True):
    def fetch(rel):
        path = out / ("layers/" + Path(rel).name if rel.startswith("layers/") else Path(rel).name)
        if not path.is_file():
            path.parent.mkdir(exist_ok=True, parents=True)
            tmp = path.with_suffix(path.suffix + ".part")
            subprocess.run(["curl", "-fsSL", "--retry", "2", ROOT + rel, "-o", str(tmp)], check=True)
            tmp.replace(path)
        return path

    paths = [f"layers/{i:02}.tif" for i in range(3, 20)]
    labels = ["labels/inklabels.png", "labels/mask.png", "predictions/v8in.png"]
    if download:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(fetch, paths + labels))
    for rel in labels:
        path = out / Path(rel).name
        crop = Image.open(path).convert("L").crop((COLS.start, ROWS.start, COLS.stop, ROWS.stop))
        crop.save(out / ("crop-" + path.name))
    output = out / "w058.zarr"
    if output.exists():
        return
    temp = out / "w058-building.zarr"
    if temp.exists():
        shutil.rmtree(temp)
    volume = zarr.open_array(
        temp, mode="w", shape=(17, ROWS.stop - ROWS.start, COLS.stop - COLS.start),
        chunks=(17, 128, 128), dtype="u1", zarr_format=2, compressor=None,
    )
    layers = np.stack([tifffile.imread(out / rel)[ROWS, COLS].copy() for rel in paths])
    volume[:] = layers
    temp.rename(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    prepare(args.out, download=not args.skip_download)
