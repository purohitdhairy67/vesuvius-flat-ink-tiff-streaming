# Stream completed flat ink tiles to TIFF

Villa stitches model predictions as overlapping float32 tiles. For outputs over its accumulation RAM limit, upstream writes two temporary Zarr arrays, reads them back for normalization, and finally produces a tiled uint8 TIFF. This patch sends a completed chunk directly to Villa's existing atomic LZW TIFF writer. It retains float32 overlap-add and the original clipping and truncation. The writer fills untouched tiles with zero. The existing RAM limit remains useful, and setting it to zero selects direct output on a small machine.

## Results on public scroll data

| Workload | Upstream | Direct TIFF | Decoded output |
|---|---:|---:|---|
| PHerc. 1447 w058, full 4,512 × 8,184 released map replay | 0.914 s median; 131,326,995 bytes of temporary Zarr files | 0.449 s median; 0 temporary Zarr bytes | Exact across all ten runs |
| PHerc. 800 segment 20251028222030, full 2,580 × 2,580 surface, Reader v3 both orders | 58.04 s; up to 25,477,135 bytes of temporary Zarr files per order | 52.46 s; 0 temporary Zarr bytes | Exact in both orders |
| PHerc. 800 segment 20251028225813, full 2,300 × 2,260 surface, Reader v3 both orders | 42.66 s; up to 20,092,066 bytes of temporary Zarr files per order | 39.80 s; 0 temporary Zarr bytes | Exact in both orders |
| PHerc. 1447 w058, 1,536 × 1,536 labeled CT crop, Reader v3 both orders | 20.48 s; up to 10,869,646 bytes of temporary Zarr files per order | 21.02 s; 0 temporary Zarr bytes | Exact in both orders |

The replay measures the output stage with published map values passed through a deterministic patch schedule, Hann weights and production TIFF code. It covers 36,926,208 pixels, with 903 nonempty tiles of 966 scheduled. Five runs per arm alternated order. Mean/range were 0.930 s / 0.888–1.021 s for upstream and 0.446 s / 0.434–0.456 s for direct TIFF. Upstream generated 511 scratch files. The output-stage median fell 51%, or 2.04× throughput for this replay. One profiler run attributed 0.399 s to upstream chunk flushes and 0.331 s to the later TIFF pass, of which 0.102 s was reading and encoding probability tiles. The patched output-stage helper took 0.451 s in its separate profile run.

The three complete Reader v3 comparisons are single runs per arm. They establish decoded parity and real inference scratch use. Their wall times do not establish an end-to-end speedup or RSS improvement. Measurements used one 16 GB Apple M4 Mac, macOS 27.0.1, Apple Clang release build of vc_reader, Python 3.14.6 and PyTorch 2.15.0 MPS. The tests force the out-of-core branch using `--max-ram-accumulation-gb 0`. The output writer still creates its ordinary same-directory temporary TIFF for atomic publication. TIFF file hashes may differ because the physical order of compressed tiles changes. Linux, CUDA and Windows performance are unmeasured. The patch leaves the model, its predictions and its interpretation alone.

[Before/after w058 image](evidence/w058-parity.png) · [All ten timings](evidence/w058-output-replay.jsonl) · [Automated tests](evidence/patched-tests.txt) · [Setup](SETUP.md)

## Reproduce

Use the setup commands in [SETUP.md](SETUP.md). The paths below assume you are inside this package and that `REPO` resolves to the pinned Villa checkout.

```bash
export PYTHONPATH="$REPO/vesuvius/src"
BASE=4980fd6b0f7bf15f6b63faafbc466409e3da231b
mkdir -p data evidence-local
git -C "$REPO" show "$BASE:vesuvius/src/vesuvius/ink_detection/inference/infer.py" > evidence-local/upstream-infer.py
git -C "$REPO" apply "$PWD/stream-ink-tiles.patch"
curl -fsSL 'https://huggingface.co/datasets/YoussefMoNader/ink-8um-pherc1447-surfaces/resolve/7e4d918712a8d8642a6fdeee45b80217c1e98317/w058/predictions/v8in.png' -o data/w058-v8in.png
python scripts/replay_output.py data/w058-v8in.png evidence-local/upstream.tif --baseline evidence-local/upstream-infer.py
python scripts/replay_output.py data/w058-v8in.png evidence-local/streamed.tif
```

Run the replay five times per arm, alternating order, and compare each `decoded_sha256`. The expected source SHA-256 is `976cb7a3e5f23ece2145689c729d7a57f342d1a54b00f2a02a2ae05fcb6826b1`. The expected decoded SHA-256 is `f8866172a1f95157e948150fa736f019ad1cfbb01eaf4e56624aaef94b3fa629`.

For the complete inference comparison, download the checkpoint and CT data with these commands. The control downloader fetches roughly 380 MB of public TIFF layers. The PHerc. 800 downloader fetches roughly 232 MB of named public Zarr objects and ancillary files. The checkpoint is 138 MB.

```bash
curl -fsSL 'https://huggingface.co/domenicor046/reader-v3/resolve/aee1d64bced4eb35546348f3d276b9e2a0b99c46/reader-v3.pth' -o data/reader-v3.pth
printf '%s  %s\n' ea405417450abd8ace1b8ff79ebce3adb646c5e48ab95da46844d77efe984d8a data/reader-v3.pth | shasum -a 256 -c
python scripts/fetch_scroll.py data 20251028222030 20251028225813
python scripts/prepare_control.py data/w058
export PYTORCH_ENABLE_MPS_FALLBACK=1
RENDER=8.64um-1.2m-116keV-volume-20250521135224.zarr
for segment in 20251028222030 20251028225813; do
  python scripts/run_inference.py "data/$segment/surface-volumes/$RENDER" data/reader-v3.pth "evidence-local/$segment-streamed.tif"
done
python scripts/run_inference.py data/w058/w058.zarr data/reader-v3.pth evidence-local/w058-streamed.tif
git -C "$REPO" apply -R "$PWD/stream-ink-tiles.patch"
for segment in 20251028222030 20251028225813; do
  python scripts/run_inference.py "data/$segment/surface-volumes/$RENDER" data/reader-v3.pth "evidence-local/$segment-upstream.tif"
done
python scripts/run_inference.py data/w058/w058.zarr data/reader-v3.pth evidence-local/w058-upstream.tif
git -C "$REPO" apply "$PWD/stream-ink-tiles.patch"
```

Both commands record exact CLI arguments, output shapes, hashes, scratch totals for the upstream spill path, and times. `--device cpu` is available for correctness replay without an Apple GPU. Exact parity compares builds on the same hardware and settings.

Run focused tests with the patch applied.

```bash
python -m pytest -q "$REPO"/vesuvius/tests/ink_detection/{test_flat_inference.py,test_flat_inference_empty_chunks.py,test_roi_inference.py,test_roi_output_metadata.py,test_flat_inference_ensemble.py,test_inference_boundaries.py,test_inference_mps_copy.py,test_volume_io.py}
```

## Attribution

The patch targets MIT-licensed [ScrollPrize/villa at 4980fd6](https://github.com/ScrollPrize/villa/tree/4980fd6b0f7bf15f6b63faafbc466409e3da231b). The proof uses Domenico Russo's MIT-licensed [Reader v3](https://huggingface.co/domenicor046/reader-v3/tree/aee1d64bced4eb35546348f3d276b9e2a0b99c46), Youssef Nader's [PHerc. 1447 w058](https://huggingface.co/datasets/YoussefMoNader/ink-8um-pherc1447-surfaces/tree/7e4d918712a8d8642a6fdeee45b80217c1e98317), and [PHerc. 800](https://scrollprize.org/data_browser/PHerc0800). Scroll CT data and the w058 figure retain CC BY-NC 4.0. Code and figure licenses apply independently. The checkpoint and raw CT data are fetched from their original sources.

Dataset citation for the newer scans: Giorgio Angelotti, Stephen Parsons, Sean Johnson, Elian Rafael Dal Prà, Johannes Rudolph, Paul Tafforeau, Alessandro Mirone, Paul Henderson, Hendrik Schilling, Forrest McDonald, David Josey, Youssef Nader, C. Seth Parker, W. Brent Seales. *Vesuvius Challenge – CT Scans of Herculaneum Papyri*. Vesuvius Challenge. [Dataset citation and license](https://scrollprize.org/data#how-to-cite).

OpenAI Codex developed the patch, ran the measurements, and prepared the release evidence. No community usage, maintainer review or merge is claimed.
