# Host-side native test suite

Runs in this repository (`sempermechanics/semper-dic-engine`). The private Android
app consumes a pinned submodule and does **not** re-run these host tiers.

```bash
# From the engine repo root
git submodule update --init --recursive
cmake -S tests -B build/tests -DCMAKE_BUILD_TYPE=Release -DDIC_REQUIRE_OPENCV=ON
cmake --build build/tests
./build/tests/dic_tests
```

> **OpenCV 4.8+ required, and the submodule is not optional.**
> `include/semper/simd.hpp` uses the `cv::v_add` / `v_sub` / `v_mul` /
> `VTraits` universal-intrinsic API introduced in 4.8; on an older OpenCV the
> build fails with `'v_mul' is not a member of 'cv'` (Ubuntu 24.04 ships
> 4.6.0). This tree puts the **vendored** OpenCV core headers ahead of the
> system ones and links the system **libraries**, so checking out
> `third_party/opencv` is enough to build against an otherwise-too-old system
> OpenCV. Skip the submodule and the build falls through to the system
> headers and fails.

Run a subset by passing a name prefix:

```bash
./build/tests/dic_tests Perf              # throughput only
./build/tests/dic_tests GoldenCorpus      # subset-level regression fixture
./build/tests/dic_tests FullFieldGolden   # end-to-end + determinism
```

CI: `.github/workflows/ci.yml` (host tests, sanitizers, cross-ABI determinism,
C SDK smoke).

See [docs/TESTING.md](../docs/TESTING.md) for the catalog.

## Golden fixtures

Two committed binary fixtures pin numerical output. Both are captured by
running the suite with an environment variable set, and both are compared
automatically otherwise.

| Fixture | Covers | Capture with |
|---|---|---|
| `fixtures/golden_corpus.bin.{bicubic,keys6x6}` | The subset solver (`precompute_subset` + `calculate_deformation`) | `SEMPER_GOLDEN_CAPTURE=1` |
| `fixtures/full_field_golden.bin` | `run_full_field` end to end: AKAZE seeding, mesh guess field, Path A, Path B, strain | `SEMPER_FF_GOLDEN_CAPTURE=1` |

```bash
# Recapture after an intentional numerical change (and say so in the commit).
SEMPER_GOLDEN_CAPTURE=1    ./build/tests/dic_tests GoldenCorpus
SEMPER_FF_GOLDEN_CAPTURE=1 ./build/tests/dic_tests FullFieldGolden.CaptureOrCompare
```

`SEMPER_GOLDEN_FILE` and `SEMPER_FF_GOLDEN_FILE` redirect either fixture to
another path — useful for capturing two builds and diffing them without
touching the committed files.

## Determinism

The engine is bitwise reproducible, and two tiers of test hold it that way.

**Run to run**, within one build:

```bash
./build/tests/dic_tests FullFieldGolden.RepeatSolve
```

**Across target ISAs** — the real gate, and what the `determinism` CI job
runs. Same source, two instruction sets, byte-compared:

```bash
for arch in x86-64 x86-64-v3; do
  cmake -S tests -B "build/$arch" -DCMAKE_BUILD_TYPE=Release \
    -DDIC_REQUIRE_OPENCV=ON -DCMAKE_CXX_FLAGS="-march=$arch"
  cmake --build "build/$arch" -j"$(nproc)"
  SEMPER_GOLDEN_CAPTURE=1 SEMPER_GOLDEN_FILE="corpus_$arch.bin" \
    "./build/$arch/dic_tests" GoldenCorpus
  SEMPER_FF_GOLDEN_CAPTURE=1 SEMPER_FF_GOLDEN_FILE="ff_$arch.bin" \
    "./build/$arch/dic_tests" FullFieldGolden.CaptureOrCompare
done

cmp corpus_x86-64.bin.bicubic corpus_x86-64-v3.bin.bicubic
cmp corpus_x86-64.bin.keys6x6 corpus_x86-64-v3.bin.keys6x6
cmp ff_x86-64.bin             ff_x86-64-v3.bin
```

All three must be silent. If one is not, something in the floating-point
contract has been broken — see [docs/DETERMINISM.md](../docs/DETERMINISM.md)
for the checklist. **Do not widen a tolerance to make it pass.**

## Throughput

`Perf.SubsetSolveThroughput` prints a rate; it is not a gate (CI runners are
noisy and the same binary is built under sanitizers). Compare **on one
machine, before and after a change** — never against another machine's
absolute number.

```bash
for i in $(seq 1 15); do
  ./build/tests/dic_tests Perf 2>&1 | grep -oE '[0-9]+ solves/s' | grep -oE '^[0-9]+'
done | sort -n | awk '{a[NR]=$1} END{printf "median=%d max=%d\n", a[int((NR+1)/2)], a[NR]}'
```

## OpenCL backend

Off by default. Build it in with `-DSEMPER_OPENCL=ON`; the backend is then
runtime-probed, so the binary still runs correctly on a machine with no GPU
and no ICD.

```bash
cmake -S tests -B build/gpu -DCMAKE_BUILD_TYPE=Release \
  -DDIC_REQUIRE_OPENCV=ON -DSEMPER_OPENCL=ON
cmake --build build/gpu -j"$(nproc)"
./build/gpu/dic_tests ClRuntime      # prints what was detected, and why not
```

`ClRuntime` passes with or without a device; the device-dependent test
reports itself as skipped rather than passing vacuously. To exercise the
with-device path locally without a GPU, install a conformant CPU device:

```bash
sudo apt-get install -y pocl-opencl-icd    # POCL 5.0
./build/gpu/dic_tests ClRuntime
```

POCL is IEEE-conformant, which is the point: it catches FP-contract and
reduction-order mistakes that a lenient vendor driver would paper over.

```bash
SEMPER_OPENCL_DISABLE=1 ./build/gpu/dic_tests   # force the CPU path
```

Full roadmap and device bring-up: [docs/GPU_ACCELERATION.md](../docs/GPU_ACCELERATION.md).

## Python bindings

`bindings/python/tests` is **not** run by `ci.yml` — only by the
manually-dispatched `wheels.yml`. Run it by hand after touching the bindings:

```bash
pip install ./bindings/python && pytest bindings/python/tests
```
