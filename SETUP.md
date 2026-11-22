# Reproduction environment

This is the build used on macOS arm64. Run it from the extracted package in a new working directory. The progress package adds one patch to Villa and uses its native vc_reader TIFF and Zarr implementation.

```bash
BASE=4980fd6b0f7bf15f6b63faafbc466409e3da231b
git clone --filter=blob:none --no-checkout --depth 1 https://github.com/ScrollPrize/villa.git villa
git -C villa fetch --depth 1 origin "$BASE"
git -C villa sparse-checkout init --cone
git -C villa sparse-checkout set vc_reader vc_object_store vesuvius
git -C villa checkout --detach "$BASE"
export REPO="$PWD/villa"
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-macos-arm64.txt
```

The native build uses Apple Clang, CMake, Ninja, pkg-config, Boost, concurrentqueue, nlohmann-json, c-blosc, c-blosc2, zstd, lz4, libtiff, libpng, jpeg-turbo and the system curl, zlib and SQLite. With Homebrew these packages can be installed as needed. Match your existing dependency prefix when setting `CMAKE_ARGS`. The explicit nlohmann-json include resolves the header lookup in this pinned Villa revision on macOS.

```bash
AGENTS_AGENT_MODE=1 AGENTS_ALLOW_INSTALL=1 \
  CC=/usr/bin/clang CXX=/usr/bin/clang++ CMAKE_BUILD_PARALLEL_LEVEL=6 \
  CMAKE_ARGS='-DVC_READER_ENABLE_CUDA=OFF -DCMAKE_PREFIX_PATH=/opt/homebrew -DBOOST_ROOT=/opt/homebrew/opt/boost -DCMAKE_CXX_FLAGS=-I/opt/homebrew/opt/nlohmann-json/include' \
  python -m pip install --no-deps "$REPO/vc_reader"
export PYTHONPATH="$REPO/vesuvius/src"
python -m vesuvius.ink_detection.inference.infer --help
```

For another OS, start with the [Villa installation instructions](https://github.com/ScrollPrize/villa/blob/4980fd6b0f7bf15f6b63faafbc466409e3da231b/CONTRIBUTING.md) and [vc_reader native requirements](https://github.com/ScrollPrize/villa/tree/4980fd6b0f7bf15f6b63faafbc466409e3da231b/vc_reader#building). The tested timings apply to the environment above.
