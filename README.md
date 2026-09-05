# Semper DIC Engine

Hybrid Delaunay Digital Image Correlation (DIC) library — portable C++ core,
stable **C ABI**, **Android JNI** adapter, and **Python** bindings.

This repository contains **only the engine**. It has no network stack, auth, or
secrets. Downstream apps link it as a git submodule (or via `find_package(Semper)`).

## Layout

```
├── include/semper/       # public C++ / C headers
│   └── kernels/          # canonical math shared by CPU and OpenCL kernels
├── src/                  # math, seeding, strain, pipeline (Internal)
├── adapters/
│   ├── android/          # JNI → libsemper_core.so
│   └── c/                # stable C ABI → libsemper_c
├── bindings/python/      # pybind11 + scikit-build-core
├── cmake/                # options, OpenCV helpers, package config
├── examples/             # beginner C++ / Python demos + sample images
├── tests/                # host suite (no NDK) + C smoke
├── docs/                 # CONTRACT, ARCHITECTURE, MATHEMATICS, TESTING,
│                         # EXAMPLES, DETERMINISM, GPU_ACCELERATION, PERF_BASELINE
└── third_party/{eigen,opencv}
```

## Quick start (examples)

DICe-published translation pair with a verified **0.4 px** displacement:

```bash
# C++ — exact DICe 4-subset contract (|u-0.4|<=0.1)
cmake -S . -B build/sdk -DSEMPER_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/sdk --target run_translation
./build/sdk/examples/cpp/run_translation \
  examples/samples/translation/ref.tif \
  examples/samples/translation/def.tif

# Python — synthetic full-field smoke, then the same DICe TIFFs
pip install ./bindings/python
python examples/python/run_synthetic.py
python examples/python/run_translation.py
```

Details: [examples/README.md](examples/README.md) · [docs/EXAMPLES.md](docs/EXAMPLES.md).

## Build

### Host tests

Host correctness (+ sanitizers + C SDK smoke) runs in this repo's GitHub Actions
(`.github/workflows/ci.yml`). Locally:

```bash
# Both submodules are required even for the host suite -- see the note below.
git submodule update --init --recursive
# C SDK additionally builds vendored OpenCV from source:
#   ./scripts/sparse-opencv.sh   # or scripts/sparse-opencv.ps1 on Windows

cmake -S tests -B build/tests -DCMAKE_BUILD_TYPE=Release -DDIC_REQUIRE_OPENCV=ON
cmake --build build/tests
./build/tests/dic_tests
```

> **OpenCV 4.8 or newer is required.** `include/semper/simd.hpp` uses the
> `cv::v_add` / `v_sub` / `v_mul` / `VTraits` universal-intrinsic API, which
> does not exist before 4.8 — on an older OpenCV the build fails with
> `'v_mul' is not a member of 'cv'`. Debian/Ubuntu's `libopencv-dev` is
> often too old (Ubuntu 24.04 ships 4.6.0).
>
> The host suite is built to cope with this: it puts the **vendored** OpenCV
> core headers ahead of the system ones and links the system **libraries**,
> so `git submodule update --init third_party/opencv` is enough to build and
> run `dic_tests` against a system OpenCV that would otherwise be too old.
> If you skip the submodule, the build falls through to the system headers
> and fails.

### C SDK

```bash
cmake -S . -B build/sdk -DSEMPER_BUILD_C_SDK=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/sdk --target semper_c semper_c_smoke
./build/sdk/adapters/c/semper_c_smoke   # path may vary by generator
```

Install exports `find_package(Semper)` → `Semper::semper_c`.

### Python

```bash
pip install ./bindings/python
pytest bindings/python/tests
```

Wheels are built with `cibuildwheel` (see `.github/workflows/wheels.yml`).

### Android JNI

Configure with `-DSEMPER_ANDROID=ON`. The shared library `OUTPUT_NAME` is
`semper_core` (`System.loadLibrary("semper_core")`). JNI symbols target
`com.indicvision.semper.SemperNativeLib`.

## CMake targets

| Target | Role |
|---|---|
| `semper_math` | Portable math (no JNI / Android) |
| `semper_pipeline` | Seeding + full-field (OpenCV + OpenMP) |
| `semper` | INTERFACE alias → pipeline |
| `semper_android` | JNI adapter (`OUTPUT_NAME semper_core`) |
| `semper_c` | Stable C ABI shared library |
| `_semper` | Python extension module |

## Options

| Option | Default | Meaning |
|---|---|---|
| `SEMPER_ANDROID` | OFF | Build JNI shared library |
| `SEMPER_BUILD_C_SDK` | OFF | Build `semper_c` + smoke |
| `SEMPER_BUILD_PYTHON` | OFF | Build pybind11 module |
| `SEMPER_BUILD_EXAMPLES` | OFF | Beginner demos (needs C SDK) |
| `SEMPER_BUILD_TESTS` | OFF | Build `dic_tests` via parent project |
| `SEMPER_OPENCL` | OFF | Build the OpenCL backend (forced OFF for Android) |

## Documentation

| Doc | Contents |
|---|---|
| [docs/CONTRACT.md](docs/CONTRACT.md) | Frozen / Stable API surface (semver tiers) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module map, data flow, dependencies |
| [docs/MATHEMATICS.md](docs/MATHEMATICS.md) | Formal DIC / ICGN / strain derivations |
| [docs/TESTING.md](docs/TESTING.md) | Host / C / Python test catalog |
| [docs/EXAMPLES.md](docs/EXAMPLES.md) | Beginner samples + verified results |
| [docs/PERF_BASELINE_bd44af0.md](docs/PERF_BASELINE_bd44af0.md) | Non-regression speed/quality floor |
| [docs/DETERMINISM.md](docs/DETERMINISM.md) | Bit-exactness contract: FP flags, canonical reduction order |
| [docs/GPU_ACCELERATION.md](docs/GPU_ACCELERATION.md) | OpenCL roadmap + device bring-up and verification runbook |

Output packing (**8 floats/point**), metrics layout (**17 floats**), and return
codes are **Frozen**. Touching `include/semper/*` requires stating the semver
tier in the PR.

The engine is **bitwise reproducible**: identical results run-to-run and across
target ISAs (SSE2 / AVX2 / NEON) for a given toolchain. That is a contract, not
an accident — the floating-point flags, the reduction order, and Path B's
propagation order are all pinned, and CI compares byte-for-byte across two
`-march` levels. Read [docs/DETERMINISM.md](docs/DETERMINISM.md) before changing
anything under `src/math/`, `src/strain/`, or `include/semper/kernels/`.

## License

BSD 3-Clause — see [LICENSE](LICENSE). Derivative of [DICe](https://github.com/dicengine/dice)
(Sandia / NTESS).
