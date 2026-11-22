"""Mirror selected public OME-Zarr levels, mesh and model maps via S3."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

HOST = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
SEGMENTS = {
    "20251028222030": "20251028222030940",
    "20251028225813": "20251028225813045",
    "20251029010146": "20251029010146642",
}

def objects(prefix):
    query = {"prefix": prefix, "list-type": 2}
    while True:
        with urllib.request.urlopen(HOST + "?" + urllib.parse.urlencode(query), timeout=45) as response:
            tree = ET.fromstring(response.read())
        for node in tree.findall("{*}Contents"):
            yield node.findtext("{*}Key"), int(node.findtext("{*}Size"))
        token = tree.findtext("{*}NextContinuationToken")
        if token is None:
            return
        query["continuation-token"] = token

def fetch(item):
    key, size, dest = item
    if dest.is_file() and dest.stat().st_size == size:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(HOST + urllib.parse.quote(key), timeout=60) as response, tmp.open("wb") as file:
                shutil.copyfileobj(response, file)
            if tmp.stat().st_size != size:
                raise OSError(f"Unexpected object length for {key}")
            tmp.replace(dest)
            return size
        except OSError:
            tmp.unlink(missing_ok=True)
            if attempt == 2:
                raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("segments", nargs="+", choices=SEGMENTS)
    args = parser.parse_args()
    for segment in args.segments:
        prefix = f"PHerc0800/segments/{segment}-auto_grown_{SEGMENTS[segment]}/"
        surface = "surface-volumes/8.64um-1.2m-116keV-volume-20250521135224.zarr/"
        full = list(objects(prefix + surface + "0/")) + list(objects(prefix + surface + "3/"))
        ancillary = list(objects(prefix + "mesh/" + segment + "-on-20250521135224-8.64um.tifxyz/"))
        ancillary += [(key, size) for key, size in objects(prefix + "ink-detection/")
                      if "/downsampled/" not in key and key.endswith(".tif")]
        # Root .zgroup and .zattrs are selected without listing every pyramid chunk.
        for name in [".zgroup", ".zattrs"]:
            ancillary.extend(objects(prefix + surface + name))
        selected = sorted(set(full + ancillary))
        print(f"{segment}: {len(selected)} objects, {sum(size for _, size in selected):,} bytes", flush=True)
        local = args.out / segment
        with ThreadPoolExecutor(max_workers=18) as pool:
            fetched = sum(pool.map(fetch, [(key, size, local / key.removeprefix(prefix)) for key, size in selected]))
        listing = [{"key": key, "bytes": size} for key, size in selected]
        (local / "objects.json").write_text(json.dumps(listing, indent=2) + "\n")
        print(f"{segment}: fetched {fetched:,} bytes", flush=True)

if __name__ == "__main__":
    main()
