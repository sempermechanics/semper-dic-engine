# Semper DIC Engine

Semper measures how a surface moves and deforms between two images. Given a
reference image and a deformed image, it returns the displacement (u, v) and
strain (exx, eyy, exy) at each point of a grid. This is Digital Image
Correlation (DIC), derived from [DICe](https://github.com/dicengine/dice).

The engine is a C++17 library with three ways in: a stable **C ABI**, an
**Android JNI** adapter and **Python** bindings. It contains no network, auth
or app code; apps link it as a git submodule or through `find_package(Semper)`.

## Quick start

Get the source with its submodules (OpenCV and Eigen are built from source):

```bash
git clone --recursive https://github.com/sempermechanics/semper-dic-engine.git
cd semper-dic-engine
./scripts/sparse-opencv.sh          # Windows: scripts/sparse-opencv.ps1
```

**Python**

```bash
pip install ./bindings/python
python examples/python/run_synthetic.py    # ends with "OK — synthetic translation recovered"
```

```python
import semper

eng = semper.Engine()
eng.set_reference(ref)                     # 2-D uint8 array, or PNG/JPEG/TIFF bytes
res = eng.run(deformed, rect=(0, 0, 512, 512), step=10, subset=21, strain_window=21)
res.points                                 # (N, 8): x, y, u, v, exx, eyy, exy, corr
```

**C++** — the DICe sample pair, shifted 0.4 px:

```bash
cmake -S . -B build/sdk -DCMAKE_BUILD_TYPE=Release -DSEMPER_BUILD_EXAMPLES=ON
cmake --build build/sdk --target run_translation
./build/sdk/examples/cpp/run_translation \
    examples/samples/translation/ref.tif examples/samples/translation/def.tif
# ends with: OK — DICe custom_app contract: 4/4 subsets |u-0.4| <= 0.1 px
```

More samples and their verified results: [examples/README.md](examples/README.md).

## What a solve takes and returns

| | |
|---|---|
| **Inputs** | Reference image, deformed image, region `rect = (x, y, w, h)`, grid `step`, `subset` size (odd), `strain_window`, optional mask |
| **Output** | 8 floats per grid point: `x, y, u, v, exx, eyy, exy, corr` |
| **Metrics** | 24 floats of telemetry per solve |
| **Errors** | Negative return codes (Python raises `semper.DicError`) |

The output layout, metrics slots and return codes are Frozen: they do not
change within a major version. See [docs/CONTRACT.md](docs/CONTRACT.md).

### Fixed or incremental reference

By default every frame is compared with the first image. Rotation at any
angle and moderate stretch are handled this way.

For large strain, where the texture of the first image no longer matches,
use `IncrementalSession`. It moves the reference forward every
`update_interval` frames and still reports total displacement from the first
image, in the same layout:

```python
s = semper.IncrementalSession(frames[0], rect=(0, 0, 512, 512), step=10,
                              subset=21, strain_window=21, update_interval=1)
for frame in frames[1:]:
    res = s.process(frame)        # res.points: totals from frames[0]
```

On small or moderate deformation the fixed reference is more accurate, since
each update adds a little interpolation error. Measurements:
[docs/INCREMENTAL_BENCHMARK.md](docs/INCREMENTAL_BENCHMARK.md).

## Entry points

| Language | API | Where |
|---|---|---|
| C++ | `Semper::pipeline::run_full_field`, `ReferenceCache`, `IncrementalSession` | `include/semper/pipeline.hpp`, `incremental.hpp` |
| C | `semper_create`, `semper_set_reference`, `semper_run`, `semper_cancel`, `semper_incremental_*` | `include/semper/semper_c.h`; link `Semper::semper_c` |
| Python | `semper.Engine`, `semper.IncrementalSession` | [bindings/python](bindings/python/README.md) |
| Android | `com.sempermechanics.semper.SemperNativeLib` (the old `com.indicvision.semper` names are still exported) | `System.loadLibrary("semper_core")` |

Every API can cancel a solve in progress from another thread.

## Build and test

```bash
# Host test suite (uses the system OpenCV: apt install libopencv-dev, or brew)
cmake -S tests -B build/tests -DCMAKE_BUILD_TYPE=Release -DDIC_REQUIRE_OPENCV=ON
cmake --build build/tests
ctest --test-dir build/tests --output-on-failure

# C SDK (vendored OpenCV), with its smoke and contract tests
cmake -S . -B build/sdk -DCMAKE_BUILD_TYPE=Release -DSEMPER_BUILD_C_SDK=ON
cmake --build build/sdk
ctest --test-dir build/sdk --output-on-failure

# Python bindings
pip install ./bindings/python pytest
pytest bindings/python/tests
```

Android: configure with `-DSEMPER_ANDROID=ON` and the NDK toolchain; the
library is `libsemper_core.so`.

Sanitizer builds, the test catalog and how to re-capture golden data:
[docs/TESTING.md](docs/TESTING.md). CI runs all of the above on every pull
request ([.github/workflows/ci.yml](.github/workflows/ci.yml)); wheels are
built by the manual `wheels` workflow.

### CMake options

| Option | Default | Effect |
|---|---|---|
| `SEMPER_BUILD_C_SDK` | OFF | Build the `semper_c` shared library and its tests |
| `SEMPER_BUILD_PYTHON` | OFF | Build the `_semper` Python module |
| `SEMPER_BUILD_EXAMPLES` | OFF | Build the C++ example |
| `SEMPER_BUILD_TESTS` | OFF | Build `dic_tests` from the root project |
| `SEMPER_ANDROID` | OFF | Build the JNI library |
| `SEMPER_REQUIRE_OPENMP` | OFF | Fail configure instead of building a single-threaded pipeline |
| `SEMPER_SANITIZER` | empty | Sanitizers for the engine libraries, e.g. `address,undefined` |
| `SEMPER_VERBOSE_LOG` | OFF | Print the solve trace to stderr on host builds |
| `SEMPER_FORCE_RELEASE` | ON | Use Release flags even when the Android build passes Debug |

### CMake targets

| Target | What it is |
|---|---|
| `semper_math` | Portable math (no OpenCV, no OpenMP) |
| `semper_pipeline` | Seeding and full-field solve (OpenCV, OpenMP) |
| `semper` | Interface alias for `semper_pipeline` |
| `semper_c` | C ABI shared library |
| `semper_android` | JNI library, output name `semper_core` |
| `_semper` | Python extension module |

## Repository layout

```
include/semper/     public headers (C++ and C)
src/                engine sources: math, seeding, strain, pipeline, I/O
adapters/c/         C ABI
adapters/android/   JNI adapter
bindings/python/    Python package (pybind11, scikit-build-core)
examples/           C++ and Python examples with sample images
tests/              host test suite, C ABI tests, fixtures
docs/               reference documentation
third_party/        OpenCV and Eigen (git submodules)
```

## Documentation

| Document | Read it to |
|---|---|
| [CONTRACT.md](docs/CONTRACT.md) | Know which APIs and formats are stable, and how to change them |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Find your way around the engine's modules and data flow |
| [MATHEMATICS.md](docs/MATHEMATICS.md) | Follow the correlation, optimisation and strain derivations |
| [TESTING.md](docs/TESTING.md) | Run the tests, sanitizers and golden checks |
| [EXAMPLES.md](docs/EXAMPLES.md) | See what the sample images verify |
| [SEEDING_BENCHMARK.md](docs/SEEDING_BENCHMARK.md) | See why the solve is seeded the way it is |
| [INCREMENTAL_BENCHMARK.md](docs/INCREMENTAL_BENCHMARK.md) | Choose between fixed and incremental reference |
| [PERF_BASELINE_bd44af0.md](docs/PERF_BASELINE_bd44af0.md) | Check speed and quality against the baseline |
| [CHANGELOG.md](CHANGELOG.md) | See what changed in each release |

## Contributing

Build and test as above. A change to `include/semper/` must state its
semantic-versioning tier (Frozen, Stable, Additive-only or Internal) in the
pull request; [CONTRACT.md](docs/CONTRACT.md) lists the rules.

## License

BSD 3-Clause; see [LICENSE](LICENSE). Derived from
[DICe](https://github.com/dicengine/dice) (Sandia / NTESS). The sample images
are redistributed under [examples/samples/LICENSE.DICe](examples/samples/LICENSE.DICe).
