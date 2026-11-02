# Testing

This page covers how to build and run the tests, what CI runs, and what each
suite proves. Every C++ test is in one host binary, `dic_tests`, built from
`tests/`. The C ABI and the Python bindings have their own test programs.
Internals are in [ARCHITECTURE.md](ARCHITECTURE.md); the API surface the tests
pin is in [CONTRACT.md](CONTRACT.md).

## Quick start

From the repository root, with `libopencv-dev` installed:

```bash
git submodule update --init --recursive
cmake -S tests -B build/tests -DCMAKE_BUILD_TYPE=Release -DDIC_REQUIRE_OPENCV=ON
cmake --build build/tests -j"$(nproc)"
./build/tests/dic_tests                                   # every suite
./build/tests/dic_tests Engine.PureTranslation_Subpixel   # one test
```

Without OpenCV, drop `-DDIC_REQUIRE_OPENCV=ON`. The build then holds only the
OpenCV-free suites (see the [Test catalog](#test-catalog)).
`ctest --test-dir build/tests --output-on-failure` runs the same binary, as CI
does.

## Running the tests

### Host suite

`cmake -S tests` builds `dic_tests` out of tree, so `tests/` holds only
sources. The engine sources compile straight into the test binary with the
production flags (`-O3 -ffast-math -fno-finite-math-only -ffp-contract=off`;
`/fp:precise` on MSVC), so numerical regressions show up here first.

Eigen and OpenCV's universal-intrinsics headers come from the submodules
(`third_party/eigen`, `third_party/opencv`). OpenCV's generated
`opencv_modules.hpp` and `cvconfig.h` are committed under
`tests/shim/opencv2/`, so the host build needs no OpenCV configure.
`scripts/sparse-opencv.sh` optionally trims the OpenCV worktree.

The full-field pipeline, image decoding, the image-backed DICe suites and the
JNI adapter need a system OpenCV with `core`, `imgcodecs`, `imgproc`,
`features2d`, `calib3d` and `flann`. Without it, configure prints a message and
leaves them out. `-DDIC_REQUIRE_OPENCV=ON` turns that into a configure error;
CI always sets it.

**Filters.** `dic_tests [filter]` matches substrings of `Suite.Name`. `A:B` runs
tests matching A or B, and anything after `-` excludes. A filter that matches
nothing exits 1.

```bash
./build/tests/dic_tests Engine.                        # the Engine suite only
./build/tests/dic_tests Engine.:Strain.                # two suites
./build/tests/dic_tests Incremental.-DiceIncremental   # Incremental, not DiceIncremental
```

Keep the trailing dot when you mean one suite: `Engine` alone also matches
`PathC.UsesSameEngineConfig` and `CancelToken.ConcurrentEnginesCancelIndependently`.

### Build options

Options of `tests/CMakeLists.txt`:

| Option | Default | Effect |
|---|---|---|
| `DIC_REQUIRE_OPENCV` | OFF | Fail configure if OpenCV is missing. |
| `DIC_REQUIRE_JNI` | OFF | Fail configure if the NDK's `jni.h` is missing (see `JniAdapter`). |
| `SEMPER_NDK_JNI_H` | searched | Path to the NDK's `jni.h`. Found from `ANDROID_NDK_LATEST_HOME`, `ANDROID_NDK_HOME`, `ANDROID_NDK_ROOT` or `ANDROID_NDK` when unset. |
| `DIC_OPENMP` | ON | OFF builds the pipeline without OpenMP. The suite must pass either way. |
| `DIC_SANITIZER` | empty | `address,undefined` or `thread`. See [Sanitizers](#sanitizers). |
| `SEMPER_COVERAGE` | OFF | gcov instrumentation at `-O1`. See [Coverage](#coverage-report-only). |
| `SEMPER_WERROR` | OFF | Warnings in first-party code fail the build. |

The root project adds `SEMPER_BUILD_TESTS`, `SEMPER_BUILD_C_SDK`,
`SEMPER_BUILD_PYTHON`, `SEMPER_BUILD_EXAMPLES` and `SEMPER_SANITIZER`; see
[README.md](../README.md). The test build always defines `SEMPER_TEST_HOOKS`,
which compiles in `internal::test_hooks` (a Path B fault and a worker-count
override). The shipped libraries are built without it.

### Against the vendored OpenCV

Without `libopencv-dev`, configure from the repository root. The vendored
OpenCV submodule is then built and linked, which is also the configuration that
ships.

```bash
cmake -S . -B build/full -DCMAKE_BUILD_TYPE=Release \
      -DSEMPER_BUILD_TESTS=ON -DDIC_REQUIRE_OPENCV=ON
cmake --build build/full -j"$(nproc)" --target dic_tests
./build/full/bin/dic_tests
```

A root build puts executables in `build/<dir>/bin/`. The `tests/` options apply
here too, for example `-DDIC_SANITIZER=address,undefined`.

### C ABI smoke and contract

```bash
cmake -S . -B build/sdk -DCMAKE_BUILD_TYPE=Release -DSEMPER_BUILD_C_SDK=ON
cmake --build build/sdk -j"$(nproc)" --target semper_c semper_c_smoke semper_c_contract
ctest --test-dir build/sdk --output-on-failure -R '^semper_c_'
```

`tests/c/smoke.c` (`semper_c_smoke`) runs a 3 px rigid shift through
`semper_run`, then two 3 px frames through `semper_incremental_*`. It proves the
C path runs end to end, not solve quality.

- The fixed-reference run passes with zero returned points, provided metrics
  slot 0 (attempted) is at least 1. The post-filter can reject every point on
  this synthetic pair, depending on fast-math reassociation. Returned points
  must have a median u within 0.75 px of 3.
- Each incremental frame must return points, with median u near 3 and then
  6 px, and must move the reference.
- Solve quality is gated in `dic_tests` (`DiceTranslation*`). Do not tighten the
  smoke to require points (see commit `5aca384`).

`tests/c/contract.c` (`semper_c_contract`) pins the Frozen C surface without
depending on solve quality. It returns non-zero on any breach.

- Compile time: `_Static_assert` on every Frozen constant (`SEMPER_ERR_*`,
  `SEMPER_FLOATS_PER_POINT` = 8, `SEMPER_METRICS_LEN` = 24) and on
  `sizeof`/`offsetof` of all eight `semper_params` fields and all six
  `semper_incremental_info` fields. A reordered field fails the build.
- Run time: null-argument guards and a run before `set_reference` give
  `SEMPER_ERR_INIT`. A degenerate or off-image ROI gives `SEMPER_ERR_ROI`. An even
  or zero subset and an `INT_MAX` strain window give `SEMPER_ERR_INIT`; the
  largest accepted window still solves. A strain window of one step gives
  `SEMPER_ERR_STRAIN_WINDOW`. An `INT_MAX` rect is clipped to the image.
- Also pinned: an undecodable mask is `SEMPER_ERR_INIT` on every entry point; a
  call made from inside the progress callback is refused; a negative capacity is
  an argument error; the capacity-drop and `metrics_len` rules; per-engine cancel
  independence; the `MAJOR.MINOR.PATCH` version string; incremental-session
  guards and codes.

`tests/c/abi_symbols.txt` is the golden list of exported symbols. CI diffs
`nm -D --defined-only` of `libsemper_c.so` against it, so an accidental export
or a dropped `SEMPER_C_API` fails. For an intended change, update the list and
the version as [CONTRACT.md](CONTRACT.md) §A.1 describes.

### Python tests

```bash
pip install -r bindings/python/requirements-ci.txt   # pinned pybind11, numpy, pytest
pip install ./bindings/python                        # builds with the vendored OpenCV
python -m pytest -v bindings/python/tests
```

Run pytest from the repository root or from outside the tree. Inside
`bindings/python`, `import semper` finds the source package, which has no
compiled extension. `test_smoke.py` and `test_incremental.py` cover
marshalling: images, parameters, results, error codes, masks, progress
callbacks and re-entrant calls. `.github/workflows/wheels.yml` runs the same
tests against every built wheel.

### Windows (MinGW)

Git Bash puts Git for Windows' own `mingw64\bin`, with its own
`libstdc++-6.dll`, on `PATH`. A `dic_tests.exe` built with another MinGW-w64
toolchain then loads that libstdc++ and crashes, for example in
`ImageCodec.EncodedPng_RoundTripsExactly` inside OpenCV's WebP decoder. It
looks like a codec bug and is not.

Put your toolchain's `mingw64/bin` ahead of Git's on `PATH`. For the C SDK
binaries, also add `build/<dir>/adapters/c`, which holds `libsemper_c.dll`.

## Sanitizers

The sanitizer jobs configure with `-DDIC_REQUIRE_OPENCV=ON` and
`-DDIC_REQUIRE_JNI=ON`. They therefore cover the full-field pipeline (its
OpenMP regions and Path B's `std::thread` workers) and the JNI adapter, not
only the math suites. Without OpenCV the pipeline is silently left out.
`DIC_SANITIZER` builds with `-fno-sanitize-recover=all`, so the first report
aborts the run, and adds `float-cast-overflow` to `undefined`.

### ASan + UBSan (GCC)

```bash
cmake -S tests -B build/asan -DCMAKE_BUILD_TYPE=Release \
      -DDIC_REQUIRE_OPENCV=ON -DDIC_SANITIZER=address,undefined
cmake --build build/asan -j"$(nproc)"
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ./build/asan/dic_tests
```

Without `libopencv-dev`, use the [root configure](#against-the-vendored-opencv)
with `-DDIC_SANITIZER=address,undefined` and run `./build/<dir>/bin/dic_tests`.

### TSan (Clang 18, libomp and Archer)

Use Clang with LLVM's `libomp` and the Archer OMPT tool. Do not use GCC: its
`libgomp` has no ThreadSanitizer instrumentation, so TSan misses its fork/join
edges and reports hundreds of false races on reused OpenMP stack frames. Do not
add a GCC TSan job.

```bash
sudo apt-get install -y clang-18 libomp-18-dev libclang-rt-18-dev libopencv-dev

cmake -S tests -B build/tsan -DCMAKE_BUILD_TYPE=Release \
      -DDIC_REQUIRE_OPENCV=ON -DDIC_SANITIZER=thread \
      -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18
cmake --build build/tsan -j"$(nproc)"

OMP_TOOL_LIBRARIES=/usr/lib/llvm-18/lib/libarcher.so \
TSAN_OPTIONS=ignore_noninstrumented_modules=1 \
  ./build/tsan/dic_tests
```

`ignore_noninstrumented_modules=1` suppresses one report from libomp's own lazy
lock-pool initialization. Both of its accesses are inside the runtime, with no
engine frame on either stack.

**Check that Archer loaded.** Set `OMP_TOOL_VERBOSE_INIT=stdout` and look for
`Tool was started and is using the OMPT interface.` Any other line, such as
`Found but not using the OMPT interface.` or
`Archer detected OpenMP application without TSan`, means TSan has no fork/join
edges. A failed registration shows up as hundreds of race reports, not as an
error.

**Without root, or on Ubuntu 22.04.** Ubuntu 22.04 has no clang-18 without the
LLVM apt repository, and its clang-14 `libarcher.so` did not register. The LLVM release
tarball `clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04.tar.xz` needs no root
and matches CI. Point `CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER` and
`OMP_TOOL_LIBRARIES` at it. It needs `libtinfo.so.5`: unpack the `libtinfo5`
.deb with `dpkg -x` onto `LD_LIBRARY_PATH`. A `libtinfo.so.6` symlink does not
work (versioned symbols).

**WSL.** TSan's shadow mapping collides with WSL's address-space layout, and the
process dies before `main`. Run it as `setarch "$(uname -m)" -R ./build/tsan/dic_tests`.

### C ABI and Python under ASan + UBSan

`dic_tests` links neither adapter, so the "C ABI + Python (ASan+UBSan)" CI job
builds them with the root option `SEMPER_SANITIZER`:

```bash
pip install -r bindings/python/requirements-ci.txt
cmake -S . -B build/san -DCMAKE_BUILD_TYPE=Release \
      -DSEMPER_SANITIZER=address,undefined \
      -DSEMPER_BUILD_C_SDK=ON -DSEMPER_BUILD_PYTHON=ON \
      -DPython_EXECUTABLE="$(command -v python)" \
      -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build/san -j"$(nproc)" --target semper_c semper_c_smoke semper_c_contract _semper
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
ctest --test-dir build/san --output-on-failure -R '^semper_c_'
```

For Python, assemble a scratch `semper` package from the sources and the built
extension. The interpreter is not instrumented, so preload the sanitizer
runtimes. Leak detection is off because CPython's own allocations are not
engine leaks.

```bash
REPO="$PWD"; PKG="$(mktemp -d)"; mkdir -p "$PKG/semper"
cp bindings/python/semper/*.py "$PKG/semper/"
cp "$(find build/san -name '_semper*.so' | head -n1)" "$PKG/semper/"
cd "$PKG"
LD_PRELOAD="$(gcc -print-file-name=libasan.so) $(gcc -print-file-name=libubsan.so)" \
ASAN_OPTIONS=detect_leaks=0 PYTHONPATH="$PKG" \
  python -m pytest -v "$REPO/bindings/python/tests"
```

### What instrumented builds skip

Sanitizer and coverage builds define `SEMPER_SANITIZER_BUILD` or
`SEMPER_COVERAGE_BUILD`, and their floating-point results differ from Release.
`GoldenCorpus.*` and `FullFieldGolden.CaptureOrCompare` still solve every
scenario but skip the numeric comparison. `Incremental.LongHorizon_160Frames`
skips its memory-growth check, because sanitizer allocators hold freed memory.
Everything else runs as in Release.

## Coverage (report-only)

```bash
cmake -S tests -B build/cov -DCMAKE_BUILD_TYPE=Debug -DSEMPER_COVERAGE=ON -DDIC_REQUIRE_OPENCV=ON
cmake --build build/cov -j"$(nproc)"
./build/cov/dic_tests
gcovr --root . build/cov --filter 'src/' --txt
```

There is no threshold. The report shows when a `src/` file drops out of the
exercised set. The build is `-O1`, because at `-O0` the suite outran a
90-minute job. Keep gcc's default counter update (atomic under `-fopenmp`):
`-fprofile-update=single` corrupts solver results across OpenMP threads.

## CI jobs

`.github/workflows/ci.yml` runs on pushes and pull requests to `main`.

| Job (id) | Build | Runs |
|---|---|---|
| Host tests (`host-tests`) | `cmake -S tests`, Release, system OpenCV, `DIC_REQUIRE_OPENCV`, `DIC_REQUIRE_JNI`, `SEMPER_WERROR` | `ctest` (`dic_tests`) |
| Host tests (no OpenMP) (`serial-build`) | as above, plus `DIC_OPENMP=OFF` | checks that no compile line has `-fopenmp` and `dic_tests` calls no OpenMP runtime; `ctest` |
| Sanitized (`sanitizers`) | `DIC_SANITIZER=address,undefined` (GCC) or `thread` (clang-18, Archer) | `dic_tests` in two shards per sanitizer |
| Vendored OpenCV (C SDK, host tests, examples) (`c-sdk-smoke`) | root build, vendored OpenCV 4.14, `SEMPER_BUILD_C_SDK`, `SEMPER_BUILD_TESTS`, `SEMPER_BUILD_EXAMPLES` | `ctest`: `dic_tests`, `semper_c_smoke`, `semper_c_contract`; exported symbols against `tests/c/abi_symbols.txt`; builds `run_translation` |
| C ABI + Python (ASan+UBSan) (`adapters-sanitized`) | root build, `SEMPER_SANITIZER=address,undefined`, C SDK and Python | `ctest -R '^semper_c_'`; pytest with the sanitizer runtimes preloaded |
| Python bindings (`python-bindings`) | `pip install ./bindings/python` | pytest from outside the checkout |
| Android NDK (arm64-v8a) (`android-ndk`) | NDK toolchain, `SEMPER_ANDROID=ON` | builds `semper_android`; JNI exports against `tests/jni/jni_symbols.txt`. Nothing runs on a device. |
| Coverage (`coverage`, `coverage-report`) | `cmake -S tests`, Debug, `SEMPER_COVERAGE=ON` | `dic_tests` in two shards; one merged gcovr report |

The sharded jobs split on the `SHARD_1` filter in `ci.yml`: shard 1 runs the
long tests it names, shard 2 runs everything else, so each test runs once.

## Numerical reproducibility contract

What "the same result" means, and when a difference is a bug.

**Same build, same inputs: bit-identical output, at any thread count.** Any
difference is a defect, such as a data race or uninitialized memory. The worker
count cannot reach the field: Path B claims cells in fixed rounds of
`tuning::kPathBBatchNodes`, and every other parallel region writes only its own
nodes. These tests guard it:

- `Engine.RepeatSolve_BitIdentical`: one ICGN solve, repeated.
- `Robustness.ConcurrentSolves_BitIdenticalToSingleThread`: per-thread engines
  on shared read-only data.
- `FullField.RepeatSolve_BitIdenticalField`, `FullFieldDeterminism.*_BitIdentical`
  and `FullFieldGolden.RepeatSolve_IsDeterministic`: the parallel
  `run_full_field`, repeated.
- `FullFieldDeterminism.*_ThreadCountInvariant` (1, 2, 3 and 7 workers) and
  `Incremental.ThreadCount_BitIdentical` (1 and 3 workers): other worker counts
  give the bytes of the host's count. `internal::test_hooks::thread_count()`
  sets the count.

They run in every build, including the plain host job, so a race fails them
without a sanitizer. The `serial-build` job (`DIC_OPENMP=OFF`) must pass the
same suite. No test compares the serial and OpenMP binaries, so between them
the cross-build tolerances below apply.

**Different builds: small drift is expected.** A change of compiler, NDK,
OpenCV build (Carotene on ARM or the generic path), Eigen version or summation
order can change rounding, because `-ffast-math` lets the compiler reassociate.
Contraction is off (`-ffp-contract=off`, and the SIMD kernels multiply then add,
never `v_fma`), so FMA is not a source of drift. ICGN is iterative, so a
last-bit difference can shift the convergence path.

Typical agreement, measured across the prebuilt-SDK to from-source OpenCV
migration and the ARM NEON to portable-SIMD change:

| Quantity | Expected cross-build agreement |
|---|---|
| Displacements (U, V) | ≤ ~1×10⁻⁴ px (typically identical to 5 decimals) |
| Strains (Exx, Eyy, Exy) | ≤ ~1 µε (0.001 mε) |
| Solver stats (solved/dead counts, convergence %) | identical |
| Report max/min **locations** | may hop between near-tied grid points |

An individual point can differ by one final IC-GN step. Between Windows MinGW
and Ubuntu GCC the worst `FullFieldGolden` point moved 1.5×10⁻³ px, so the
golden suites use wider tolerances (see [Golden fixtures](#golden-fixtures)).

Anything beyond this is a regression: values off in the first or second
significant digit, extrema in unrelated regions, or changed dead-point counts
on the same input. Bisect it with the `Engine` suite per ABI. `-ffast-math`
makes results build-specific by design; if bit-reproducibility across builds
becomes a requirement, build `semper_math` and `semper_pipeline` with
`-fno-fast-math` and re-benchmark.

## The synthetic-deformation methodology

The engine suites use ground truth that is exact by construction:

1. A continuous speckle function `g(x, y)`, a seeded sum of 400 Gaussian blobs
   (`dictest::SpeckleField`, `framework/synthetic.h`), is sampled to make the
   reference image.
2. The deformed image samples `g` at the analytic inverse of the affine warp:
   `def(q) = g(c + A⁻¹(q − c − t))`.

No image is resampled, so the true 6-DOF parameters are known to machine
precision and every reported error is the engine's own. The warp matches the
engine's shape function (ARCHITECTURE.md, "Warp convention"). Sequence and
full-field tests render frames the same way with `framework/sequence.h`.

Engine tolerances follow the literature for ICGN with bicubic interpolation on
smooth speckle: translation ≤ 0.02 px, displacement gradients ≤ 2×10⁻³.

## Test catalog

Run one suite with `./build/tests/dic_tests <Suite>.` (see
[filters](#host-suite)). Every test has a comment in its source that states what
it proves, what a failure means, and the measured values its bounds come from.
Suites marked *OpenCV* compile only when OpenCV is found.

### Unit suites (`tests/unit/`)

| Suite | File | What it proves | Notable cases |
|---|---|---|---|
| `SimdKernels` | `test_simd_kernels.cpp` | Each SIMD kernel matches a double-precision scalar oracle: the vector path on SSE or NEON, the scalar fallback elsewhere. | `SumSqDiff_AllSizes` (n = 1…1000, ragged tails), `FusedErrorEqualsStandaloneZnssd` |
| `Image` | `test_image.cpp` | Interpolators, gradients and blur satisfy exact identities: pixel values at integer coordinates, linear ramps at sub-pixel positions, a constant image through the blur. | `BoundaryDemotionLadderIsContinuousInRange`, `BilinearOutOfBoundsReturnsZero`, `PixelCountOverIntThrowsBeforeReading` |
| `SubsetPrecomputer`, `Finite` | `test_subset_precomputer.cpp` | Subset statistics, steepest-descent images and the Hessian match their definitions. The SoA `sdi_planes` equal the AoS vectors element-wise. Ill-conditioned, off-image and even-sized subsets are rejected. | `SdiPlanesMirrorSteepestDescentImages`, `FastPathMatchesFullPrecompute`, `IllConditioned_Rejected` (`tuning::kHessianMinRcond`), `Finite.FiniteBitsCatchesInfAndNan` |
| `Strain`, `Median`, `StrainCalculator` | `test_strain_calculator.cpp` | VSG recovers linear fields exactly (float precision) and keeps the `−1000` sentinel where its window is unsupported. The displacement median test rejects spikes and clusters and keeps smooth noise and straight jumps. | `VsgRigidBodyTranslationGivesZeroStrain`, `OutlierTestKeepsAStraightJump`, `Median.NaNsAreLeftOut`, `StrainCalculator.ZeroStepOrShortVectors_KeepTheSentinel` |
| `CancelToken`, `Metrics` | `test_cancel_token.cpp` | The cancel flag is sticky, crosses threads and is per instance. The cancel window (`CancelScope`) keeps a request made after the caller began but before the solve entered. `kCancelled` is −99. Needs no OpenCV. | `CrossesThreads`, `WindowKeepsRequestsSinceBegin`, `Metrics.PrefixLenTable` (`metrics_prefix_len`) |
| `ImageCodec` *OpenCV* | `test_image_codec.cpp` | `Semper::io::decode_gray`, `decode_bgr`, `image_dimensions` and `to_gray8`: raw versus encoded detection, colour, 16-bit and float conversion, masks read as booleans, bad input returning empty without a crash. | `EncodedWithRawLength_StillDecoded`, `Encoded16Bit_PreservesContrast`, `Mask16BitAndFloat_AreBoolean`, `OversizedHeader_RefusedBeforeDecode` |
| `IncrementalMath` *OpenCV* | `test_incremental_math.cpp` | Incremental-session helpers: sampling and plane fits of the increment field, the anchor drop/refresh/keep decision, anchor matching, the forward mask warp. | `Fit_ReproducesLinearFieldAcrossAHole`, `MatchAnchor_RejectsUnconvergedSolve`, `MaskWarp_LargeRotation_IsTheRotatedMask` |

### Engine suites (`tests/integration/`)

| Suite | File | What it proves | Notable cases |
|---|---|---|---|
| `Engine` | `test_optimization_engine.cpp` | `OptimizationEngine` end to end on exact synthetic deformations: ICGN, Simplex, auto-search and the failure guards. Translation ≤ 0.02 px, gradients ≤ 2×10⁻³. | `PureTranslation_Subpixel` (the canonical benchmark, (2.25, −1.50) px), `General6DOF_Recovered`, `AutoSearch_FindsLargeTranslation`, `ZnssdInvariantToBrightnessAndContrast`, `SubsetOffImage_ReportsFailureStatus`, `RepeatSolve_BitIdentical` |
| `Robustness` | `test_robustness.cpp` | Safe on bad input and correct under concurrency: ±3 grey levels of noise (0.05 px), 6% contrast (0.03 px), a textureless subset rejected at precompute, per-thread engines bit-identical to one thread. | `ConcurrentSolves_BitIdenticalToSingleThread` (a race detector under TSan) |
| `GoldenCorpus` | `test_golden_corpus.cpp` | The subset solver against captured fixtures. See [Golden fixtures](#golden-fixtures). | `CaptureOrCompare_4x4Bicubic`, `CaptureOrCompare_6x6Keys` |

### Full-field pipeline suites (`tests/integration/`, OpenCV)

| Suite | File | What it proves | Notable cases |
|---|---|---|---|
| `FullField`, `DebugExport`, `PathC`, `Telemetry` | `test_full_field_contracts.cpp` | `run_full_field` contracts without depending on solve quality: Frozen return codes, parameter validation, the capacity-drop rule, the metrics layout and point accounting, reference-cache reuse, cancel and progress. Also holds the three `CancelToken` cases that need a real solve. | `UndersizedBuffer_TruncatesWithoutOverflow`, `MetricsPointCountsAccountForEverySolvedPoint`, `StrainWindowTooSmallForStep_IsRejected` (−4), `UnseedableField_ReturnsDocumentedCode` (−5), `RefCache_SequenceOnOneCache_BitIdenticalToFresh`, `ProgressFinalTickOnCallingThread`, `CancelToken.ConcurrentEnginesCancelIndependently` |
| `ReferenceCache` | `test_reference_cache.cpp` | ROI-mask sterilization (masked pixels become the `-10.0f` sentinel; a mask of another size is resized with `INTER_NEAREST`) and the cache lifecycle. `adopt()` equals `set_from_gray` bit for bit. `static_assert`s pin the class as non-copyable and non-movable. | `RoiSterilization_MasksLeftHalf`, `EmptyInput_LeavesCacheCleared`, `PixelCountOverInt_LeavesCacheCleared` |
| `FullFieldGolden` | `test_full_field_golden.cpp` | The orchestration layer (anchor lattice, Delaunay mesh guess, Paths A and B, strain stage) against a committed baseline. See [Golden fixtures](#golden-fixtures). | `CaptureOrCompare`, `RepeatSolve_IsDeterministic` |
| `FullFieldDeterminism` | `test_full_field_determinism.cpp` | The app's default grid (subset 21, step 5, strain window 15), solved four times from fresh caches, repeats every byte and metrics slot, also on 1, 2, 3 and 7 workers. It uses synthetic speckle because TIFF decoding crashes in the MinGW host build. | `MaskedHole_BitIdentical`, `NoLockStretch_ThreadCountInvariant` (needs more than one `kPathBBatchNodes` round of Path B) |
| `SeedBench`, `PathB`, `Telemetry` | `test_seeding_contracts.cpp` | The always-on seeding contracts: the `phase_correlate_*` sign convention and log-polar estimate (Internal API, off the solve path), AKAZE-seeded rotation and quality gates, the median test and its publication rule, lattice sizing, the Path C fallback. Bounds are 3–5× the measured values noted in the file. | `AnchorSeedingRotationGate` (5° to 180°), `HalfTurnSolvesTheField`, `StretchSeedsWithoutBlunders`, `MedianTestKeepsNoUntestedAnchor`, `PathB.WorkerExceptionFailsTheSolve` |
| `Metamorphic` | `test_metamorphic.cpp` | Pairs of solves whose inputs differ by a transformation with a known effect: a rigid shift, a whole-pixel shift of the deformed image, swapped images, gain and offset, mirroring. Images are exact; bounds are 3–20× the measured values. | `SwappedImages_InvertTheMap`, `MirroredImages_MirrorTheField` |

### Incremental suite (OpenCV)

| Suite | File | What it proves | Notable cases |
|---|---|---|---|
| `Incremental` | `integration/test_incremental.cpp` | `IncrementalSession` on sequences with exact ground truth (`framework/sequence.h`). `update_interval = 0` and each first frame equal `run_full_field` bit for bit. Also covers the update schedule, the early update and the update floor, failed solves never becoming the reference, masks that travel with the material, coasting across occlusions, and long-horizon error and memory. | `IntervalZero_BitIdenticalToRunFullField`, `Rotation90_200Frames_TracksTheTurningRoi`, `StationaryBand_NodesCoastAcrossAndComeBack`, `FailedFirstFrame_NeverBecomesTheReference`, `LongHorizon_160Frames` |

`SEMPER_INC_TRACE=1` prints per-frame error, an error map and a bias fit. The
long-horizon bounds are wide on purpose: each reference update adds a small
same-sign interpolation bias, so error grows roughly linearly with the number of
updates. See [MATHEMATICS.md](MATHEMATICS.md) §9.5 and
[INCREMENTAL_BENCHMARK.md](INCREMENTAL_BENCHMARK.md).

### DICe comparison suites (`tests/dice/`)

[DICe](https://github.com/dicengine/dice) (BSD 3-Clause) is an established
reference DIC implementation. These suites reuse its published scenarios,
images and solved fields (`tests/fixtures/dice/`,
[LICENSE.DICe](../tests/fixtures/dice/LICENSE.DICe)), not its code. All but
`DiceTranslationSynthetic` read TIFFs and need OpenCV.

| Suite | File | What it proves |
|---|---|---|
| `DiceTranslationSynthetic` | `test_translation_synthetic.cpp` | DICe's `custom_app` contract on analytic images: a 0.4 px x-shift, subset 27, four subsets, each within DICe's 0.1 px `errorTol`. |
| `DiceTranslationReal` | `test_translation_real_image.cpp` | The same contract on DICe's own 512×512 `ref.tif` and `def.tif`. Only U is asserted, because DICe publishes no V truth. |
| `DiceStrainGradients` | `test_strain_gradients.cpp` | Raw ICGN gradients on real speckle with a 1% uniaxial strain (`def_exx.tif`): field mean within 1.5×10⁻³, per-subset rms ≤ 6×10⁻³. |
| `DiceStrainVsg` | `test_strain_vsg.cpp` | The same field through `StrainCalculator`'s VSG (120 px window): mean exx within 1×10⁻³ of 0.01, rms < 2×10⁻³. |
| `DiceFieldAgreement` | `test_field_agreement.cpp` | Agreement with DICe's solved open-hole tension fields (`oht_cfrp`, `DICe_solution_NN.txt`) at DICe's coordinates. Step 01: rms < 0.005 px, max < 0.02 px. Steps 03, 06 and 11 go through the coarse search: rms < 0.05 px. |
| `DiceIncremental` | `test_incremental_oht.cpp` | The ladder `oht_cfrp_00 → 01 → 03 → 06 → 11` through an `IncrementalSession` (N = 1, subset 27, Keys 6×6, step 7) against DICe's fixed-reference fields: ≥ 90% of points compared, rms < 0.1 px. At rung 11 the incremental rms du / dv is 0.058 / 0.026 px, mostly a uniform u bias; the fixed reference on the same grid gives 0.0024 / 0.0029 px. |

### Adapter suite (OpenCV and an NDK `jni.h`)

| Suite | File | What it proves | Notable cases |
|---|---|---|---|
| `JniAdapter` | `jni/test_jni_adapter.cpp` | `adapters/android/jni/SemperJNI.cpp`, compiled into `dic_tests` and driven through a fake `JNIEnv` / `JavaVM`: array pinning and release, the output-capacity bound, the metrics prefix, progress-thread attach and detach, cancel, and exceptions at the boundary. Each result equals the engine called directly. | `FullFieldDirect_MatchesRunFullField`, `FullFieldDirect_SmallBufferStaysInBounds`, `Metrics_CopiesTheLargestPrefix`, `ExceptionInTheSolve_IsMinus3`, `LegacyPackage_Forwards` |

The suite compiles against the NDK's own `jni.h`, copied alone out of the NDK
sysroot so no Bionic header reaches the host include path.
`tests/jni/shim/android/` stands in for `<android/log.h>` and
`<android/bitmap.h>`. It is not built with MSVC. GitHub's Ubuntu runners set
`ANDROID_NDK_LATEST_HOME`, so CI finds the header.

The fake is stricter than a VM. Element pointers are fresh heap copies of
exactly the array's length, released only through the pointer handed out.
Direct buffers are exactly their stated capacity. Global references, thread
attaches and UTF strings must balance after every call, and a JNI function the
harness does not implement is a null pointer.

The C ABI and Python tests are separate programs; see
[C ABI smoke and contract](#c-abi-smoke-and-contract) and
[Python tests](#python-tests).

### Benchmarks and throughput

These measure and print; they gate nothing.

| Suite | File | What it does |
|---|---|---|
| `Perf` | `perf/test_throughput.cpp` | Prints precompute and solve rates. Asserts only a 120 s wall-clock ceiling, which catches a hang. Run it on a quiet machine for real numbers. |
| `SeedBench` sweeps | `integration/test_seeding_bench.cpp` | `CandidateSweep`, `RealSpeckleSweep`, `Challenge5`, `Challenge14Sinusoid`, `LargeMotionSweep` produce the tables in [SEEDING_BENCHMARK.md](SEEDING_BENCHMARK.md). Skipped unless `SEMPER_RUN_SEEDBENCH=1`. `Challenge5` and `Challenge14Sinusoid` also need `SEMPER_DICE_REPO` set to a `dicengine/dice` checkout. |
| `IncBench` | `integration/test_incremental_bench.cpp` | `FixedVsIncremental` runs the fixed reference and N = 1, 2, 5 and 10 in lockstep on scenarios S1–S10 (S10 is DICe's OHT ladder) and produces [INCREMENTAL_BENCHMARK.md](INCREMENTAL_BENCHMARK.md). Skipped unless `SEMPER_RUN_INCBENCH=1`. The full run takes about 40 minutes on a 20-thread laptop. |

```bash
SEMPER_RUN_SEEDBENCH=1 ./build/tests/dic_tests SeedBench
SEMPER_RUN_SEEDBENCH=1 SEMPER_SEEDBENCH_REPEATS=5 SEMPER_SEEDBENCH_CSV=sweep.csv \
  ./build/tests/dic_tests SeedBench.CandidateSweep
SEMPER_RUN_INCBENCH=1 SEMPER_INCBENCH_CSV=incbench.csv ./build/tests/dic_tests IncBench
SEMPER_RUN_INCBENCH=1 SEMPER_INCBENCH_QUICK=1 ./build/tests/dic_tests IncBench   # T = 160, σ = 2, one seed
SEMPER_RUN_INCBENCH=1 SEMPER_INCBENCH_ONLY=S4 ./build/tests/dic_tests IncBench
python tools/plot_incremental_bench.py incbench.csv docs/images/incremental
```

The other sweeps write CSV to `SEMPER_SEEDBENCH_CSV_REAL`, `_C5`, `_C14` and
`_BIG`. `SEMPER_INCBENCH_SEEDS=n` sets the noise seeds per σ > 0 (default 3).
`tools/cache_bench.c` times the fixed-reference cache against any
`libsemper_c.so`: `gcc -O2 -o cache_bench tools/cache_bench.c -ldl -lm`.

## Golden fixtures

Two suites compare against committed binaries. Recapture only after a
deliberate output change, in a commit whose message says why the output moved.
Capture on Ubuntu with GCC in a Release build. Sanitizer and coverage builds
skip the comparison.

### `GoldenCorpus`: the subset solver

An 18×18 grid over two scenarios (pure translation and affine) is solved with
`INIT_NO_SIMPLEX` from a zero guess, once per interpolator. The results are
compared with `tests/fixtures/golden_corpus.bin.bicubic` and `.keys6x6`. The
suite never enters `run_full_field`, so a seeding change can move the full
field without moving this golden.

- **Values:** `1e-2` px for displacement, `1e-3` for gradients. A point the
  fixture puts more than `ON_TRUTH_PX` (0.1 px) off the exact field sits in a
  local minimum and gets a 5× looser bound.
- **Status:** `solve_icgn` returns status 1 both for a hard reject (sentinel
  score `2.0f`) and for an exhausted iteration budget (the real final ZNSSD). A
  point still creeping when the budget runs out lands on either side of
  `delta_p.norm() < 1e-3` from last-bit arithmetic alone. Such a flip is excused
  only when neither side carries the reject sentinel and all six values agree.
- **Flip gate:** host noise flips points both ways, but a shrunken iteration
  budget only moves them to "exhausted". The gate is therefore the imbalance
  `flips_to_exhausted − flips_to_converged`, capped at `MAX_FLIP_IMBALANCE` (2).
  `MAX_SLOW_CONVERGENCE_FLIPS` (6) caps the total. The measurements behind both
  caps are in the file.

```bash
SEMPER_GOLDEN_CAPTURE=1 ./build/tests/dic_tests GoldenCorpus
```

`SEMPER_GOLDEN_FILE` overrides the fixture base path; the `.bicubic` and
`.keys6x6` suffixes are appended to it.

### `FullFieldGolden`: the orchestration layer

Two synthetic scenarios (`translate`, `affine`) go through `run_full_field` and
are compared with `tests/fixtures/full_field_golden.bin`.

- **Exact:** the point count, the grid coordinates, and 13 of the 15 recorded
  metrics slots (`0..8`, `16`, `19..23`). Slot 8 (mean IC-GN iterations) gets
  `0.1` and slot 22 (mesh coverage) gets `1e-4`. The timing slots (`9..14`,
  `17`, `18`) are not recorded.
- **Point values:** `5e-3` px for displacement, `1e-4` for strain and
  correlation. IC-GN stops at `||delta_p|| < 1e-3`, so two toolchains can stop
  one step apart, and the input images depend on the host libm. The bounds sit
  at 3–5× the worst measured cross-toolchain difference.
- Each run prints the worst deviation per slot, so the headroom stays visible.

```bash
SEMPER_FF_GOLDEN_CAPTURE=1 ./build/tests/dic_tests FullFieldGolden
```

`SEMPER_FF_GOLDEN_FILE` overrides the fixture path. The file originated on the
GPU branch (`claude/gpu-parallelization-displacement-strain-asx9w1`). Fixtures
from the two branches are not interchangeable, because their seeding front-ends
differ.

## Test layout

```
tests/
  CMakeLists.txt   builds dic_tests and lists the sources
  test_main.cpp    runner entry point
  framework/       test_framework.h (TEST_CASE, CHECK, ...), synthetic.h,
                   synthetic_cv.h, sequence.h, seed_bench.h, bench_util.h,
                   image_io.{h,cpp}
  unit/            one component against an oracle or an identity
  integration/     the engine and run_full_field end to end, goldens, incremental
  dice/            comparisons with DICe
  perf/            throughput
  jni/             JniAdapter, jni_symbols.txt, shim/android/
  c/               smoke.c, contract.c, abi_symbols.txt (built by adapters/c)
  fixtures/        golden binaries; dice/ images and solved fields
  shim/opencv2/    OpenCV's generated config headers
```

`tests/CMakeLists.txt` always compiles `DIC_UNIT_TESTS`,
`DIC_INTEGRATION_TESTS`, `DIC_DICE_TESTS` and `DIC_PERF_TESTS`.
`DIC_PIPELINE_TESTS` and the pipeline sources attach only when OpenCV is found.
`jni/test_jni_adapter.cpp` and the adapter source attach when OpenCV and an NDK
`jni.h` are both found. Image-backed DICe files sit in the always-on list and
compile their tests out with `#if defined(DIC_HAVE_OPENCV)`.

## Adding a new test

1. Add it to an existing file, or create `unit/test_<module>.cpp` or
   `integration/test_<module>.cpp` and list it in `DIC_UNIT_TESTS` or
   `DIC_INTEGRATION_TESTS`. If it needs OpenCV (`imgcodecs`, `imgproc`,
   `features2d`), list it in `DIC_PIPELINE_TESTS` and wrap it in
   `#if defined(DIC_HAVE_OPENCV)`.
2. Write `TEST_CASE(Suite, Name) { ... }` with `CHECK`, `REQUIRE`, `CHECK_NEAR`,
   `CHECK_REL` or `FAIL_MSG` from `framework/test_framework.h`.
3. Build ground truth analytically: `SpeckleField` and `AffineDeformation`
   (`framework/synthetic.h`), or `framework/sequence.h` for sequences. Never
   resample images, and never use a warp that differs from the engine's
   shape-function convention.
4. Above the test, write what it proves, what a failure would mean, and the
   measured values its bounds come from. A test whose failure nobody can
   interpret is a liability. Add a new suite to the catalog above.

## Host vs JNI responsibility split

- **Host C++ tests** (`dic_tests`) own algorithmic correctness: displacement
  accuracy, strain math, SIMD equivalence, determinism and robustness.
- **The `JniAdapter` suite** owns the adapter's C++ on the host, under
  ASan + UBSan and TSan.
- **The C ABI and Python tests** own marshalling at the public SDK surface.
- **The private app's tests** own the runtime: `System.loadLibrary`, the real
  VM, OpenMP on Android and Kotlin orchestration. They do not re-assert
  displacement accuracy.

## Downstream app tests (informative)

Kotlin JVM unit tests, instrumented JNI smokes and backend contract tests live
in the private application repository. It consumes this engine as a pinned
submodule and does not re-run these host tiers. Its test map is not reachable
from this tree.

## Known limitations / future work

- **Per-ABI drift is not gated.** CI runs the suite on x86-64 only. To compare
  ABIs, build it with the NDK toolchain per ABI and run it on devices; the
  tolerances already absorb fast-math reassociation.
- **No exact cross-toolchain gate.** The `FullFieldGolden` tolerances are the
  only cross-build check. The GPU branch adds a `determinism` CI job and a
  `docs/DETERMINISM.md`; link them here when that lands.
- **Silent output truncation** (`FullField.UndersizedBuffer_TruncatesWithoutOverflow`,
  `contract.c`). When `out_capacity` is below the solved-point count,
  `run_full_field` drops the overflow and returns the reduced count without
  signalling it. This is the Frozen capacity-drop rule
  ([CONTRACT.md](CONTRACT.md) §A.4); metrics-only (`capacity == 0`) and
  partial-buffer callers rely on it.
- **`metrics_len` below 16 writes nothing** (`FullField.MetricsLen16_LeavesSlot16Untouched`,
  `Metrics.PrefixLenTable`, `contract.c`). The Frozen minimum is 16. Python
  passes `kMetricsLen` (24), the JNI adapter copies the largest prefix its array
  holds, and the C ABI passes the caller's value through.
- **`image_dimensions` ignores raw buffers**
  (`ImageCodec.ImageDimensions_RawBufferYieldsZeros_KnownTrap`). A raw RGBA or
  gray blob is not a decodable container, so it reports 0×0. Callers with raw
  buffers already know the size; the test pins the behaviour so nobody relies on
  it.
