# Test suite reference

Complete catalog of the automated tests: what each one proves, why it exists,
and how to run everything. The engine it exercises is described in
[ARCHITECTURE.md](ARCHITECTURE.md); CI wiring is in
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

---

## Sanitizers

Both sanitizer jobs cover the **full-field pipeline**, not just the math suite.
That requires `libopencv-dev` at configure time and `-DDIC_REQUIRE_OPENCV=ON`;
without them `_SEMPER_OPENCV_READY` stays false and `run_full_field`, its three
OpenMP regions and Path B's `std::thread` workers are silently excluded from the
build. (They were, until this was fixed.)

### ASan + UBSan — GCC

```bash
cmake -S tests -B build/asan -DCMAKE_BUILD_TYPE=Release \
      -DDIC_REQUIRE_OPENCV=ON -DDIC_SANITIZER=address,undefined
cmake --build build/asan -j"$(nproc)" && ./build/asan/dic_tests
```

Clean: **97/97, exit 0, zero findings** (re-run on Ubuntu 22.04, GCC 11.4,
`ASAN_OPTIONS=detect_leaks=1`, `UBSAN_OPTIONS=print_stacktrace=1`). Three of
those 97 are the numeric goldens, which skip themselves under
`SEMPER_SANITIZER_BUILD` and pass without comparing anything.

`cmake -S tests` needs `libopencv-dev` installed. Without it, configure from the
repo root instead and the vendored OpenCV submodule is built and picked up
automatically — same sanitizer flags, since `dic_tests` compiles the engine
sources directly into the test binary rather than linking a separately built
library:

```bash
cmake -S . -B build/asan -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DSEMPER_BUILD_TESTS=ON -DDIC_REQUIRE_OPENCV=ON \
      -DDIC_SANITIZER=address,undefined
cmake --build build/asan -j"$(nproc)" && ./build/asan/bin/dic_tests
```

### TSan — Clang, libomp and Archer (not GCC)

**GCC cannot be used for TSan here.** Its OpenMP runtime, `libgomp`, carries no
ThreadSanitizer instrumentation, so TSan cannot see the join at the end of
`GOMP_parallel`. libgomp then recycles the outlined function's stack frame for
the next parallel region and TSan reports the reused slot as a race — the
"previous write" being, for instance, `compute_hessian_only` storing an Eigen
6x6 on the master's stack, and the "read" carrying no instrumented frame at all
because it is performed inside libgomp itself. Measured on this suite: **576
reports on the `FullField` tests alone, none of them involving engine data.**

LLVM's `libomp` plus the [Archer](https://github.com/PRUNERS/archer) OMPT tool
supplies those edges:

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

**Result: 97/97, exit 0, zero ThreadSanitizer warnings** (Ubuntu 22.04,
clang 18.1.8, `OMP_NUM_THREADS=4`). Three of the 97 are the numeric goldens,
which skip themselves under `SEMPER_SANITIZER_BUILD`. This run covers the
0.3.0 anchor-phase refactor, which changed what the parallel region writes.
`ignore_noninstrumented_modules=1` is
needed for one libomp-internal report — its lazy lock-pool initialisation, where
both accesses are inside the uninstrumented runtime with no engine frames on
either stack.

| configuration | reports (`FullField`) |
|---|---|
| GCC + libgomp | 576 |
| Clang + libomp + Archer | 0 |

The engine has no data races; the earlier count was entirely a property of the
toolchain. Do not re-enable a GCC TSan job — it will be red and the redness will
mean nothing.

**Check that Archer actually registered.** Set `OMP_TOOL_VERBOSE_INIT=stdout`
and require the line `Tool was started and is using the OMPT interface.`
Anything else — `Found but not using the OMPT interface.`, or
`Archer detected OpenMP application without TSan` — means the run is the GCC
row of the table above wearing a Clang badge. A failed registration shows up
as hundreds of reports rather than as an error message, so the symptom you see
is "TSan found races" and the cause is that the tool never loaded.

**Without root, and under WSL.** Two obstacles, both worth writing down
because neither is obvious from the failure:

- CI's packages (`clang-18 libomp-18-dev libclang-rt-18-dev`, with
  `/usr/lib/llvm-18/lib/libarcher.so`) are the supported path. Jammy has no
  clang-18 without adding the LLVM apt repo, and its **clang-14** `libarcher.so`
  would not register for us (`Found but not using the OMPT interface`, with or
  without `-rdynamic`, and with `RunningOnValgrind` exported `T`). The upstream
  LLVM release tarball
  (`clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04.tar.xz`) registers first
  try, needs no root, and matches the compiler CI uses. Point
  `CMAKE_C_COMPILER`/`CMAKE_CXX_COMPILER` and `OMP_TOOL_LIBRARIES` at it. That
  build wants `libtinfo.so.5`, which jammy does not ship; the `libtinfo5` deb
  unpacked with `dpkg -x` onto `LD_LIBRARY_PATH` is enough, and symlinking
  `libtinfo.so.6` is *not* (versioned symbols).
- TSan's shadow mapping collides with WSL's ASLR layout and the process dies
  before `main`. Run it as `setarch "$(uname -m)" -R <exe>`.

### Windows: run the tests with the right MinGW runtime first

Git for Windows ships its own `libstdc++-6.dll` under its own
`mingw64\bin`, and Git Bash puts that directory on `PATH`. A `dic_tests.exe`
built with a different MinGW-w64 (a UCRT WinLibs toolchain, say) then loads
*that* libstdc++ and crashes — for us, reproducibly inside
`ImageCodec.EncodedPng_RoundTripsExactly`, with the fault landing in
`cv::WebPDecoder::WebPDecoder()` constructing a `std::string`. It looks
exactly like a broken image codec and is not: the same binary passes 97/97
once the build toolchain's `mingw64/bin` is ahead of Git's on `PATH`.

So: prepend your toolchain's `mingw64/bin`, and add
`build/<dir>/adapters/c` as well for the C SDK binaries, which look for
`libsemper_c.dll` there rather than beside the executable.

### The guarantee that does not need a sanitizer

`FullField.RepeatSolve_BitIdenticalField` solves the same field five times and
requires the packed output to be bit-identical, which is what a race would
break. It runs in every build, including the plain host job.

Thread count is deliberately not varied: `run_full_field` pins every OpenMP
region with `num_threads(safe_cores)`, which overrides `omp_set_num_threads` and
`OMP_NUM_THREADS`, so a test cannot change it without adding a production hook.
Cross-thread-count invariance is not a contract this engine makes — Path B's
reliability-guided propagation is order-dependent by construction.

## Running the tests

### Host engine tests (no device / NDK needed)

Built out of the source tree (from the engine repo root) so `tests/` stays pure
sources — the build dir is regenerated on demand and gitignored:

```bash
git submodule update --init --recursive
./scripts/sparse-opencv.sh   # optional; shrinks OpenCV worktree

cmake -S tests -B build/tests -DCMAKE_BUILD_TYPE=Release  # any C++17 compiler
cmake --build build/tests
./build/tests/dic_tests                 # all suites
./build/tests/dic_tests Engine          # one suite
./build/tests/dic_tests Engine.PureTranslation_Subpixel   # one test
```

Dependencies come from the git submodules (`git submodule update --init`):
Eigen (`third_party/eigen`) and OpenCV's universal-intrinsics headers
(`third_party/opencv/modules/core/include`). OpenCV's generated
`opencv2/opencv_modules.hpp` + `cvconfig.h` are committed under
`tests/shim/opencv2/` so the host build needs no OpenCV configure.
Logging uses portable `src/util/log.hpp` (stderr on host; Android log on device).

### Beginner examples (sample images + verified results)

See [EXAMPLES.md](EXAMPLES.md) and [`examples/README.md`](../examples/README.md).
The C++ translation demo mirrors `DiceTranslationReal` on DICe `ref.tif`/`def.tif`.

```bash
cmake -S . -B build/sdk -DSEMPER_BUILD_EXAMPLES=ON
cmake --build build/sdk --target run_translation
./build/sdk/examples/cpp/run_translation \
  examples/samples/translation/ref.tif \
  examples/samples/translation/def.tif
```

### C ABI smoke

```bash
cmake -S . -B build/sdk -DSEMPER_BUILD_C_SDK=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/sdk --target semper_c semper_c_smoke
# Ensure the shared lib is on the loader path (Windows: adapters/c next to PATH)
./build/sdk/bin/semper_c_smoke   # path may vary by generator
```

`smoke.c` deliberately treats **zero recovered points as a pass** — it asserts the
C ABI *path* runs end-to-end (create → set_reference → run → non-negative return),
not solve *quality*. The post-filter can legitimately reject every point on the
synthetic pair depending on toolchain fast-math reassociation, so requiring `n > 0`
here would be a flaky quality gate at the wrong layer. Solve quality is asserted by
`dic_tests` (`DiceTranslation*`), where inputs and tolerances are controlled. Do not
"tighten" the smoke to demand points — see commit `5aca384`.

### Python smoke

```bash
pip install ./bindings/python
pytest bindings/python/tests
```

> Host math/DICe tests compile with `-ffast-math` deliberately — the same
> floating-point model as the production library, so numerical regressions
> surface here first.

---

## The synthetic-deformation methodology

The engine suite's ground truth is **exact by construction**, not rendered:

1. A continuous speckle function `g(x,y)` (seeded sum of ~400 Gaussian blobs,
   `framework/synthetic.h`) is sampled to create the reference image.
2. The deformed image is created by sampling `g` at the **analytic inverse**
   of the affine warp: `def(q) = g(c + A⁻¹(q − c − t))`.

No image resampling is involved, so the true 6-DOF parameters are known to
machine precision, and every error the engine reports is the engine's own.
The warp convention matches the engine's shape function exactly
(see ARCHITECTURE.md → "Warp convention").

**Tolerances** (literature-standard for ICGN + bicubic on smooth speckle):
translation ≤ 0.02 px, displacement gradients ≤ 2×10⁻³.

---

## Numerical reproducibility contract

What "the same result" means across builds, and when a difference is a bug.

**Same APK, same device, same inputs → bit-identical results.**
Guarded by `Engine.RepeatSolve_BitIdentical` (one ICGN solve) and
`FullFieldDeterminism.*` (the whole parallel `run_full_field`). Any run-to-run
variation on identical inputs is a defect (threading race, uninitialized
memory). The field does not depend on the core count either: the Path B flood
fill claims cells in fixed-size rounds (`tuning::kPathBBatchNodes`), not
per-thread.

**Different builds / ABIs / dependency versions → small drift is expected.**
The engine compiles with `-ffast-math` and uses FMA-based SIMD reductions, so
any change to the compiler, NDK, OpenCV build (e.g. Carotene on ARM vs the
generic path), Eigen version, or kernel summation order legally perturbs
floating-point rounding. Because ICGN is iterative, last-bit differences per
iteration shift the convergence path. Empirically (verified across the
prebuilt-SDK → from-source OpenCV migration, ARM NEON → portable SIMD):

| Quantity | Expected cross-build agreement |
|---|---|
| Displacements (U, V) | ≤ ~1×10⁻⁴ px (typically identical to 5 decimals) |
| Strains (Exx, Eyy, Exy) | ≤ ~1 µε (0.001 mε) |
| Solver stats (solved/dead counts, convergence %) | identical |
| Report max/min **locations** | may hop between near-tied grid points |

Anything beyond this — values off in the first or second significant digit,
extrema in unrelated regions, changed dead-point counts on the same input —
is a real regression: bisect with the `Engine` suite per-ABI.

Trade-off note: `-ffast-math` makes results build-specific by design. If
bit-reproducibility across builds ever becomes a requirement, compile `core/`
and `preprocessing/` with `-fno-fast-math` and re-benchmark; the explicit SIMD
kernels already do the heavy lifting, so the expected cost is small.

---

## Suite: `Engine` — `integration/test_optimization_engine.cpp`

The synthetic deformation regression suite. Covers `OptimizationEngine`
end-to-end (ICGN, Simplex, auto-search, guards).

| Test | Proves | Failure would mean |
|---|---|---|
| `ZeroDeformation_RecoversZero` | Identity warp → (0,0), ZNSSD ≈ 0 | Broken normalization or warp math — nothing else can be trusted |
| `PureTranslation_IntegerPixel` | Integer shift recovered from an exact guess | Interpolation at integer coords broken |
| `PureTranslation_Subpixel` | **The canonical DIC benchmark.** (2.25, −1.50) px recovered from integer guess; gradients ≈ 0 | Sub-pixel interpolation, gradient, or Hessian regression |
| `PureTranslation_SubpixelPhaseSweep` | 6 fractional phases (0.1…0.9) all recovered | Phase-dependent interpolation bias (a single phase can hide it) |
| `UniaxialStrain_Recovered` | 1% exx strain recovered, other DOFs ≈ 0 | Gradient-DOF columns of steepest-descent images broken |
| `SimpleShear_Recovered` | uy shear recovered without cross-talk | uy/vx column swap or sign error |
| `General6DOF_Recovered` | Translation + all four gradients simultaneously | Cross-DOF coupling errors invisible in single-DOF tests |
| `MultiSubsetGrid_ConsistentRigidTranslation` | DICe-style field consistency: a 3×3 grid of POIs all recover the same rigid translation (0.03 px grid tolerance — see the in-test note on per-POI speckle variance) | Position-dependent defect: coordinate-origin mistake, asymmetric boundary handling, or a row-stride bug invisible to the center-only tests |
| `AutoSearch_FindsLargeTranslation` | (6.4, −8.3) px found with NO initial guess | Coarse SSD search / Simplex / hand-off between stages broken |
| `ZnssdInvariantToBrightnessAndContrast` | Solution unchanged under `I' = 0.7·I + 30` | ZNSSD normalization broken — real-world lighting robustness lost |
| `SubsetOffImage_ReportsFailureStatus` | Impossible warp → `status != 0` | Engine fabricates answers instead of failing (silent corruption) |
| `BothInterpolatorsConverge` | Bicubic AND Keys 6×6 paths both solve | One interpolator selector path regressed |
| `LmDampingPreservesWellPosedSolution` | LM damping (α=1e-3) doesn't shift a good solution | LM applied to wrong DOFs or damping leaking into the answer |
| `RepeatSolve_BitIdentical` | The same solve twice is bit-identical | Threading race, uninitialized buffer, or run-to-run nondeterminism |
| `SuccessfulSolve_CorrelationNonNegative` | Successful solves report ZNSSD ≥ 0 | Sentinel contract broken — the JNI layer marks failed/skipped points with `CORR_INVALID = -1`, so a real score must never be negative |
| `SaturatedBlackSpeckle_Converges` | A subset with > 10% pixels saturated to exactly 0 converges with both interpolators (0.05 px: the clip is a kink no cubic reproduces) | Validity is read off the sample value again: before v0.5.0 every sample not > 0 was dropped as if off the image, the 4×4 path failed the 90% guard and the 6×6 one solved on the masked path |
| `MaskedThenFullIteration_MatchesFullOnlyPath` | After iterations that lost pixels, a full iteration steps with the reference Hessian again (made visible by scaling the AoS images only the masked branch reads) | The masked inverse is reused for the rest of the call |
| `DegenerateMaskedHessian_FailsCleanly` | A masked Hessian singular in its strain block (translation block fine) ends in status 1 with finite values near the start | Its inverse is taken anyway and Inf/NaN reaches the warp, the `(int)` pixel guard and the result |

## Suite: `DiceTranslationSynthetic` — `dice/test_translation_synthetic.cpp`

Cross-validation against **DICe** (Digital Image Correlation Engine,
[github.com/dicengine/dice](https://github.com/dicengine/dice), BSD 3-Clause),
an established reference implementation. Reproduces the **input→output
contract** of DICe's `tests/examples/custom_app` — a rigid 0.4 px
X-translation, subset size 27, four subsets, each recovered within DICe's
`errorTol = 0.1 px` — on our engine and our **analytic synthetic** images. No
DICe code is used; only the published scenario and tolerance. It shows our ICGN
solver matches a reference DIC engine's accuracy on the canonical translation
case, in milliseconds on a laptop (no Trilinos/MPI).

| Test | Proves | Failure would mean |
|---|---|---|
| `PureTranslation_0p4px_FourSubsets` | All 4 subsets recover the 0.4 px shift within DICe's 0.1 px tolerance (we actually land ≤ 0.02 px — see `Engine.PureTranslation_Subpixel`) | Our engine no longer agrees with a reference DIC implementation on rigid translation — a correlation/interpolation regression, or accuracy fallen below the field's accepted bar |

## Suite: `DiceTranslationReal` — `dice/test_translation_real_image.cpp`

The real-image companion to `DiceTranslationSynthetic`: same 0.4 px / 0.1 px contract, but on
DICe's **actual** 512×512 speckle images (`fixtures/dice/ref.tif`, `def.tif` —
their `custom_app` `ref.tif`/`def.tif`, BSD-3, see
[`fixtures/dice/LICENSE.DICe`](../tests/fixtures/dice/LICENSE.DICe)).
Where `DiceTranslationSynthetic` proves accuracy on math-perfect synthetic texture, this adds
**real-speckle robustness** and **independence** — an image we did not generate,
a target we did not compute. The fixtures path is injected by CMake as
`DICE_FIXTURES_DIR`; a minimal P5 reader loads the PGMs (the host build has no
image codec).

| Test | Proves | Failure would mean |
|---|---|---|
| `CustomApp_0p4px_RealSpeckle` | All 4 subsets recover the 0.4 px X-shift on DICe's real images, within DICe's 0.1 px tolerance | Our engine disagrees with DICe on their own experimental data — a real-texture/robustness gap the synthetic tests don't expose |

## Suite: `SimdKernels` — `unit/test_simd_kernels.cpp`

Every kernel is checked against a **double-precision scalar oracle**. On an
SSE/NEON host this validates the vectorized path; on other compilers, the
scalar fallback.

| Test | Proves |
|---|---|
| `SumSqDiff_AllSizes` | Exact for n = 1…1000, covering full vector blocks, ragged tails, and n < lane-width |
| `SumSqDiff_ZeroWhenAllEqualMean` | No catastrophic cancellation on the degenerate case |
| `Znssd_AllSizes` | ZNSSD reduction matches oracle across sizes |
| `Znssd_PerfectMatchIsZero` | Perfectly correlated signals score ≈ 0 (the convergence anchor) |
| `ErrorAndGradient_MatchesScalarOracle` | Fused kernel: error AND all 6 SoA gradient projections match |
| `FusedErrorEqualsStandaloneZnssd` | Contract: fused error term ≡ standalone `znssd_sum` (callers assume it) |

## Suite: `Image` — `unit/test_image.cpp`

Interpolation ladder + gradients + blur, validated via mathematical identities
(no golden files).

| Test | Property exploited |
|---|---|
| `InterpolatorsReproducePixelValuesAtIntegerCoords` | All three kernels are *interpolating*: weight 1 at s=0, 0 at other integers |
| `InterpolatorsExactOnLinearRamp` | Catmull-Rom & Keys reproduce degree-1 polynomials exactly at any sub-pixel position |
| `GradientOfLinearRampIsExactSlope` | 5-point central difference is exact for degree ≤ 4 |
| `BlurPreservesConstantImage` | Normalized kernel ⇒ constant in = constant out |
| `BilinearOutOfBoundsReturnsZero` | Bilinear returns `0.0f` outside its support (IC-GN's 4 px guard keeps it from ever being sampled there; validity is not read from values) |
| `BoundaryDemotionLadderIsContinuousInRange` | 6×6→bilinear and 4×4→bilinear demotion never extrapolates outside [0,255] |

## Suite: `SubsetPrecomputer` — `unit/test_subset_precomputer.cpp`

| Test | Proves |
|---|---|
| `MeanAndStdMatchManualComputation` | Stats vs a double-precision manual computation; normalized intensities are zero-mean/unit-RMS |
| `SdiPlanesMirrorSteepestDescentImages` | **The SIMD-refactor regression guard**: SoA `sdi_planes` ≡ AoS `steepest_descent_images` element-wise. The SIMD fast path reads planes, the masked path reads vectors — drift silently corrupts solutions |
| `SteepestDescentImagesFollowDefinition` | `sd = [gx, gy, gx·x, gx·y, gy·x, gy·y]/σ` at spot-checked pixels |
| `HessianIsSymmetricAndInverseIsValid` | `H = Hᵀ` and `H·H⁻¹ ≈ I` |
| `RejectsSubsetOffImageBoundary` | Boundary subsets → `is_initialized == false` (never solved) |
| `FastPathMatchesFullPrecompute` | `compute_hessian_only` + `precompute_subset_fast` ≡ full `precompute_subset`, byte for byte (v0.5.0: the full path is built from the other two); an invalid pool entry gives an uninitialized subset |
| `LargeSubset_HinvFinite` | A 301 px subset is accepted and `H·H⁻¹ ≈ I` with a finite inverse | The conditioning test overflows with subset size again (the old `\|det H\|` did past ~170 px) |
| `IllConditioned_Rejected` | Vertical stripes (2×2 check) and a pattern symmetric about the subset centre, which cannot see rotation, are rejected at 21 and 201 px. The second had `det H` of +1e9 to +8e16 and was accepted with a meaningless inverse before v0.5.0 | The 6×6 Hessian is no longer checked as a whole (`hessian::well_conditioned`, `tuning::kHessianMinRcond`) |
| `EvenDimRejected` | An even, zero, one or negative `dim` gives an uninitialized subset from all three functions, even with a valid pool entry. Before v0.5.0 an even `dim` wrote `(dim+1)²` values into `dim²` buffers (heap overflow under ASan) |

## Suite: `Strain` — `unit/test_strain_calculator.cpp`

Linear displacement fields have closed-form Green-Lagrange strain, and VSG's
linear least-squares fits them **exactly** — so interior tolerances are float
precision, not "close enough".

| Test | Proves |
|---|---|
| `VsgRecoversUniaxialStrainExactly` | 1% exx from u = 0.01·X, exact at grid center |
| `VsgRecoversGeneralLinearField` | All three strain components for a general 4-coefficient field, at multiple interior points |
| `VsgRigidBodyTranslationGivesZeroStrain` | Constant displacement → zero strain (the classic false-strain bug) |
| `VsgLeavesSentinelWhereWindowUnsupported` | Corner points (<90% window fill) and invalidated points keep the `−1000` sentinel |
| `OutlierTestLeavesSmoothNoisyFieldAlone` | A general linear field with ±0.1 px noise loses no point to the median test |
| `OutlierTestRejectsASingleSpike` | One point 3 px off is the only one rejected |
| `OutlierTestRejectsAWholeCluster` | A 4x4 block 20 px off is rejected whole (its inner points only on a later pass), and strain beside it is the field's again |
| `OutlierTestKeepsAStraightJump` | A 5 px step along a column (a crack) is not an outlier on either side |
| `Median.EvenCountAveragesMiddlePair` | `util::median_inplace` averages the middle pair for an even count (the anchor lattice's own median took the upper one) | The two medians diverge again |
| `OutlierTestSkipsSparseAndInvalidPoints` | Under 3 neighbours, no test; an already invalid point is neither counted nor used |

## Suite: `CancelToken` — `unit/test_cancel_token.cpp`

The flag `run_full_field` polls to stop a solve mid-frame. The polling itself
needs a real solve (OpenCV, threads) and is covered on the Android side; what is
pinned here is the contract those polls rest on.

| Test | Proves |
|---|---|
| `StartsClear` / `RequestIsObserved` | The flag reads back what was set |
| `StaysSetUntilCleared` | Sticky across a thousand polls — the caller must clear it before a run, which is why both run paths assign `false` on entry |
| `CrossesThreads` | A worker spinning on the flag sees a request made from another thread — the flag is set on the UI thread, read by solver workers |
| `FreeFunctionsActOnTheGlobalTokenOnly` | `request_cancel` / `clear_cancel` set and clear the process-wide token and never touch a caller-owned one |
| `ConcurrentEnginesCancelIndependently` (in `test_full_field_contracts.cpp`) | Two solves in flight on two threads: cancelling A returns `-99` from A, and B finishes with the uncontested field byte for byte. Failed before v0.5.0, when the token was bound into one global pointer and B was cancelled by A's request | A token leaked between concurrent engines again (the SDK / Python case) |
| `Metrics.PrefixLenTable` | `metrics_prefix_len`: nothing below 16, 18 → 17, 19..24 as given, 24 above | The one prefix table drifts from CONTRACT §A.4 |
| `CancelledCodeIsDistinctFromEngineErrors` | `kCancelled` is −99, matching `AnalysisRunCodes.ERROR_CANCELLED`, and is neither the −2 ROI nor the −3 init failure |

`cancel.cpp` is deliberately its own translation unit with no OpenCV in it, so
the host suite can link it without the rest of the pipeline.

## Suite: `ImageCodec` — `unit/test_image_codec.cpp`

The single point where every external image (C ABI, Python, JNI) enters the
engine: `Semper::io::decode_gray` / `decode_bgr` / `image_dimensions`
(`src/io/image_codec.cpp`). Before this suite the file was in **no** test target,
so a pixel-corrupting decode bug was invisible to the whole pipeline. OpenCV-gated
(needs `imgcodecs`/`imgproc`), so it lives in `DIC_PIPELINE_TESTS`.

| Test | Proves | Failure would mean |
|---|---|---|
| `RawRgba_MatchesOpenCvGray` | `len == w*h*4` path decodes to `w×h` gray matching OpenCV's `RGBA2GRAY` on spot pixels | Wrong raw stride/channel order — every RGBA frame silently corrupted |
| `RawGray_IsExactClone` | `len == w*h` path returns a byte-identical, independent clone | Raw grayscale mangled, or an alias of the caller's buffer (use-after-free) |
| `EncodedPng_RoundTripsExactly` | Encoded PNG decodes losslessly to the original | Encoded-image decode regressed |
| `EncodedColor_ConvertsToGray` | Encoded 3-channel image collapses to one channel | Colour images enter the solver as multi-channel garbage |
| `EncodedNon8U_ConvertsTo8U` | 16-bit PNG is converted to `CV_8U` | Non-8U depth reaches the solver, which assumes 8-bit |
| `Encoded16Bit_PreservesContrast` | A 12-bit ramp in a 16-bit PNG decodes to 0..255, strictly increasing | The depth conversion saturates again (before v0.5.0 the ramp came out nearly all 255) |
| `FloatAndSigned_ScaledToGray8` | `to_gray8`: a [0, 1] float image spans 0..255, a flat one is 128, 8-bit gray passes through, colour collapses, empty stays empty | The one conversion every non-8-bit input goes through (decode, Python arrays) changed |
| `EncodedWithRawLength_StillDecoded` | A PNG whose byte length equals `w*h` for the expected size is decoded as a PNG; a raw buffer that merely starts like a JPEG is still raw | Raw-vs-encoded is decided by length alone again |
| `MismatchedLength_FallsThroughToEncoded` | `expected_w/h` set but `len` matching neither raw size → encoded path | A wrong-length raw buffer is misread as raw instead of decoded |
| `NullAndEmpty_ReturnEmpty` | `nullptr` / `len == 0` → empty `cv::Mat`, no crash | Null-deref on a bad caller argument |
| `GarbageBytes_ReturnEmptyNoCrash` | Undecodable bytes → empty, no crash | Decoder trusts attacker-controlled bytes |
| `ImageDimensions_EncodedBuffer` | Correct w/h from an encoded buffer | Dimension probe wrong — downstream allocations mis-sized |
| `ImageDimensions_ZerosOnFailureAndNull` | Zeros on undecodable/null input | Garbage dimensions treated as valid |
| `ImageDimensions_RawBufferYieldsZeros_KnownTrap` | Documents that `image_dimensions` has **no** raw-buffer awareness (raw blob → 0×0) | (Pins a known limitation, not a bug — see Known limitations) |

## Suite: `ReferenceCache` — `integration/test_reference_cache.cpp`

Direct characterization of the ROI-mask **"Ghost Wall"** sterilization and the
cache lifecycle in `src/pipeline/reference_cache.cpp`, previously only exercised
incidentally. Masked pixels are overwritten with the `-10.0f` sentinel so the
solver skips them. OpenCV-gated.

| Test | Proves | Failure would mean |
|---|---|---|
| `RoiSterilization_MasksLeftHalf` | Masked pixels become exactly `-10.0f`; unmasked stay real | ROI mask ignored, or the wrong region sterilized — masked material enters the solve |
| `MaskResize_NearestKeepsHalfSplit` | A differently-sized mask is `INTER_NEAREST`-resized to the reference before sterilizing | Mask/reference misalignment silently masks the wrong pixels |
| `NoMask_LeavesAllPixelsReal` | Empty mask → no sentinels written | Phantom masking with no ROI supplied |
| `Relifecycle_UpdatesDims` | Re-`set_from_gray` with a new size updates the cached dimensions | Stale dimension state leaks across references |
| `EmptyInput_LeavesCacheCleared` | Empty gray clears the cache (`width == 0`, `ref_img == nullptr`) instead of retaining the old reference | A failed re-init silently solves against the previous frame |
| `Reset_ClearsEverything` | `reset()` frees and zeroes all state | Double-free or leak on teardown |
| `Adopt_EqualsSetFromGray_NoMask` / `_WithMask` | `adopt()` of an image prepared by the solver gives the same intensities and gradients, bit for bit, as `set_from_gray` of the same pixels, sentinels included | An incremental reference update would solve against a different image than a fresh reference would |
| `Adopt_NullImage_LeavesCacheCleared` | Adopting nothing clears the cache rather than keeping the old reference | A failed hand-off silently solves against a stale reference |

Compile-time `static_assert`s pin `ReferenceCache` as non-copyable/non-movable
(§A.2 of the contract) regardless of OpenCV.

## Suite: `FullField` — `integration/test_full_field_contracts.cpp`

Host characterization of `run_full_field` — the Frozen return codes, the
capacity-drop rule, and the metrics layout — without depending on solve
quality. OpenCV-gated.

| Test | Proves | Failure would mean |
|---|---|---|
| `DegenerateRoi_ReturnsRoiError` | `rect_w < step` → `-2` | ROI validation regressed |
| `EmptyDeformed_ReturnsInitError` | Empty deformed image → `-3` | Init-guard regressed |
| `DeformedSizeMismatch_ReturnsInitError` / `DeformedWrongType_ReturnsInitError` | A deformed frame of another size than the reference, or not `CV_8UC1` (float, BGR), and a non-`CV_8UC1` mask → `-3` | Before v0.5.0 a smaller frame solved into a plausible field of nonsense and a float frame was read as bytes |
| `DebugExport.RectBeyondImage_NoOOB` | The debug export (maps + CSV) runs on a rect hanging off the image with a mask; clean under ASan | The export's mask lookup reads outside the mask again |
| `EvenOrZeroSubsetSize_ReturnsInitError` | `subset_size` 20, 0, −21, 1, 2 → `-3`; 3 solves | `validate_params` stopped guarding the subset buffers (crashes under ASan without it) |
| `HugeRect_ClippedToImage` | A rect of `INT_MAX` sides, and one hanging off the top-left by whole steps, give the image-sized rect's field byte for byte | The rect sizes the grid again (`std::bad_alloc` through the C/JNI boundary), or clipping moved the lattice |
| `HugeStrainWindow_ReturnsInitErrorPromptly` | `strain_window` of `INT_MAX`, 2²⁰ and one past `2·(min(gridW,gridH)+1)·step` → `-3` within 20 s (on a worker thread, so a regression fails instead of hanging), and the largest accepted window still solves | The window bound is gone: `INT_MAX` spun in `vsg_window_node_count` for hours before v0.5.0 |
| `RectOffImage_ReturnsRoiError` | Rect wholly off the image, `rect_x + rect_w` overflowing `int`, or a clipped rect narrower than a step → `-2` | Clipping overflowed or let an empty grid through |
| `StrainWindowTooSmallForStep_IsRejected` | A `strain_window` spanning fewer than 3 grid nodes → `-4`, with the corrective value logged | The VSG plane fit is rank-deficient at every point, the strain post-filter discards the whole field, and the caller gets zero points and a success code — the silent-empty-field defect |
| `StaleCancelDoesNotAbortNextSolve` | A leftover cancel is cleared on entry | A prior cancel poisons the next solve |
| `UndersizedBuffer_TruncatesWithoutOverflow` | Small `out_capacity` drops points, never overflows, returns `≤ capacity/8` | Buffer overflow on a caller-sized buffer |
| `MetricsLayout_Contract` | attempted ≥ solved ≥ 0; `rejected == attempted − solved`; convergence % ∈ [0,100]; seed flag ∈ {0,1,2} | A downstream telemetry reader mis-parses the Frozen slots |
| `MetricsLen16_LeavesSlot16Untouched` | `metrics_len == 16` fills 0..15 and never writes slot 16 | Write past a 16-float caller buffer |
| `NullMetrics_DoesNotCrash` | `metrics == nullptr` is legal | Null-deref when a caller wants points only |
| `RefCache_SequenceOnOneCache_BitIdenticalToFresh` | Five frames solved against one cache give the same bytes as five fresh caches, on a DFT-optimal ROI with the 6×6 interpolator | The per-reference cache (grid scan, Hessian pool, phase-correlation reference) changed a result. This test caught `cv::phaseCorrelate` windowing a cached input in place |
| `RefCache_InvalidatesOnParamsOrMask` | Changing `step`, `subset_size`, `rect` or the mask between solves gives the same result as a fresh cache | A stale grid scan or Hessian pool reused for a different grid |
| `RefCache_CancelledSolveDoesNotPoisonCache` | A solve cancelled during the pre-pass, then repeated, matches a fresh solve | A partly filled Hessian pool cached as complete |
| `MetricsIdentityHoldsWithOutliers` | A 44 px block moved 4 px further than the rest: the median test rejects its inner nodes and `3 + 4 + 19 == 1 + 20 + 23` still holds, with slot 23 > 0 | Points the median test drops fall out of the accounting again (before v0.5.0 there was no slot 23 and the identity failed whenever the test fired) |
| `UnseedableField_ReturnsDocumentedCode` | A blank deformed frame → `kSeedFailed` (`-5`) | Path C's failure leaks an undocumented code again (it was `-1`), or a blank crop "locks" phase correlation again (it did, at zero shift, and the solve returned 0 points with success) |
| `PathC.UsesSameEngineConfig` | `make_engine` sets LM damping (`kLmAlpha`) and the interpolator; every pipeline solve builds its engine through it (Path C used to leave LM off) | The engine configurations drift apart again |
| `Telemetry.RescueClassifiedByTriggeringRun` | A Simplex rescue after an IC-GN run that used its whole budget is counted as a timeout, even when the IC-GN after the rescue is short | The crash/timeout split reads the post-rescue run again (it did until v0.5.0) |
| `MetricsPointCountsAccountForEverySolvedPoint` | `metrics[3] + metrics[4] + metrics[19] == metrics[1] + metrics[20]`, anchors > 0, mean ICGN iters ≥ 1, slot 18 > 0 | A solving path or a post-filter drop stops being counted — the failure mode that hid the anchor lattice from slots 3/4 and the strain post-filter from everything |

## Suite: `GoldenCorpus` — `integration/test_golden_corpus.cpp`

Relative equivalence for the **subset** solver: an 18×18 grid over two
scenarios (pure translation and affine) is solved with `INIT_NO_SIMPLEX` from a
zero guess, once per interpolator. 554 of the 648 nodes survive
`precompute_subset` and are recorded (578 before v0.5.0: the 6×6 conditioning
test now rejects 12 per scenario, every one of which was a status-1 record
with diverged values); they are compared against
`tests/fixtures/golden_corpus.bin.bicubic` and `.keys6x6`.

| Test | Proves | Failure would mean |
|---|---|---|
| `CaptureOrCompare_4x4Bicubic` | Every point's convergence status and converged u/v/strains match the capture | The bicubic path moved |
| `CaptureOrCompare_6x6Keys` | The same for the Keys 4th-order path | The 6×6 path moved |

**The fixtures are captured on Ubuntu GCC Release** (re-captured for v0.5.0 on
Ubuntu 24.04, GCC 13.3; first captured at `244231e`), and host tests
build with `-ffast-math`, so bit-exact reproduction is not on offer even for an
identical binary rerun. Values get tolerances (`1e-2` px, `1e-3` strain). Status
gets one specific, bounded excuse:

`solve_icgn` returns status 1 for two unrelated things. A **hard reject** —
the 90%-valid-pixel guard, the deactivation check, the minimum-gradient check
— returns the sentinel score `2.0f`. **Exhausting the iteration budget**
returns the real final ZNSSD. A point still creeping when the budget runs out
lands on either side of `delta_p.norm() < 0.001f` on last-bit arithmetic alone,
so it reports "budget exhausted" on one host and "converged" on another while
producing the same answer. Both hosts measured show it: Ubuntu GCC 11.4 flips
`(368,192)` and `(170,434)` in *opposite* directions, Windows MinGW flips
`(258,126)`, and every one of those agrees with the fixture on all six values.

A flip is excused only when **neither side carries the reject sentinel and all
six values still agree**. What is then gated is not the count but the
**imbalance**, `flips_to_exhausted − flips_to_converged`:

> Host arithmetic pushes a point across `delta_p.norm() < 1e-3` in whichever
> direction its last bits fall, so it flips both ways with no bias. A shrunken
> iteration budget can only ever move a point from converged to exhausted.
> The signed difference separates the two; the raw count does not.

| configuration | 4×4 flips (net) | 6×6 flips (net) |
|---|---|---|
| Ubuntu GCC 11.4 | 0 (0) | 2 (0) |
| Windows MinGW GCC | 0 (0) | 1 (+1) |
| Ubuntu, `kIcgnMaxIter` − 2 | 2 (+2) | 3 (+3) |
| Windows, `kIcgnMaxIter` − 2 | 2 (+2) | 4 (+4) |
| either host, `kIcgnMaxIter` − 8 | 3 hard mismatches | 6 hard mismatches |

`MAX_FLIP_IMBALANCE` is 2 — one clear of the worst clean host, and still
failing the −2 perturbation through the 6×6 case on both. A raw cap
cannot do both jobs: Ubuntu already sits at 2 clean and goes to 3 perturbed.
`MAX_SLOW_CONVERGENCE_FLIPS` (6) remains as a backstop against churn in both
directions at once. Every perturbed flip measured was `0 -> 1`, on both hosts.

Sanitizer and coverage builds skip this suite — instrumentation and the
coverage build's `-O1` change which subsets initialize.

```bash
# recapture both fixtures, on Ubuntu GCC Release only
SEMPER_GOLDEN_CAPTURE=1 ./build/full/bin/dic_tests GoldenCorpus
```

## Suite: `FullFieldGolden` — `integration/test_full_field_golden.cpp`

`GoldenCorpus` pins the **subset** solver; it calls `precompute_subset` /
`calculate_deformation` directly and never enters `run_full_field`. This suite
pins the **orchestration layer** — anchor-lattice seeding, the Delaunay mesh
guess field, Path A, Path B's propagation order, and the strain stage —
against a committed binary baseline, `tests/fixtures/full_field_golden.bin`.

| Test | Proves | Failure would mean |
|---|---|---|
| `CaptureOrCompare` | Two synthetic scenarios (translate, affine) reproduce the recorded point count, the recorded counting metrics exactly, and every point value within a portable tolerance | Any stage of the pipeline changed its output — intended or not |
| `RepeatSolve_IsDeterministic` | Five solves of the same field agree bit-for-bit | A data race or an order-dependent reduction entered the parallel region |

Fifteen metrics slots are recorded alongside the field: `0..8`, `16`, and this
branch's `19..23` (anchor points, post-filter drops, phase lock, mesh coverage,
and from v0.5.0 median-test drops).
The timing slots (`9..14`, `17`, `18`) are excluded — they differ every run by
construction. Thirteen of the fifteen are integer counts or a 0/1 flag and are
compared **exactly**; that exactness is what makes this suite a gate. The two
exceptions are averages: slot 8 (mean IC-GN iterations) gets `0.1` and slot 22
(mesh coverage, a convex-hull area ratio) gets `1e-4`.

**Point values carry toolchain-portable tolerances**, not bit-tight ones:
`5e-3` px displacement, `1e-4` strain and correlation, and exactly `0` for the
grid coordinates. That is deliberate and measured. IC-GN is a fixed-point
iteration stopped at `||delta_p|| < 1e-3`, so two toolchains differing only in
FMA contraction take slightly different paths into the same basin and stop one
step apart: the disagreement is the size of a final IC-GN step, not of a
rounding error. Between the Windows MinGW capture and Ubuntu GCC 11.4, over 772
points in two scenarios, the worst were `|du|` 9.7e-4 px, `|dv|` 1.5e-3 px,
strain 1.8e-5, corr 1.9e-5 — and slot 8 moved 7.96579 to 7.94737. The bounds
sit at roughly 3× that. `synthetic.h` also builds its images from several
hundred `std::exp` terms per pixel, so even the *input* depends on the host
libm. **There is no exact cross-ABI gate on this branch** — these tolerances
are it. The GPU branch
(`claude/gpu-parallelization-displacement-strain-asx9w1`) adds a
`determinism` CI job and a `docs/DETERMINISM.md`; point this paragraph at them
when that lands.

The comparison prints the worst deviation it saw per slot on every run, passing
or failing, so the headroom is visible rather than assumed. Perturbing the
anchor stride by one still fails 14 assertions on Ubuntu — metrics slots 3, 8,
19 and 22 on the translate scenario and 3 through 8, 19 and 22 on the affine
one, plus 6 and 52 value mismatches whose worst `dv` is 0.101 px, 67×
the tolerance.

```bash
# recapture the baseline after a deliberate output change
SEMPER_FF_GOLDEN_CAPTURE=1 ./build/full/bin/dic_tests FullFieldGolden
```

`SEMPER_FF_GOLDEN_FILE` overrides the fixture path. **Recapturing is a
deliberate act**: the diff to the `.bin` belongs in a commit whose message says
why the field moved. The file originated on the GPU branch; the two branches'
fixtures are not interchangeable, because their seeding front-ends differ.

## Suite: `SeedBench` — `integration/test_seeding_bench.cpp` + `integration/test_seeding_contracts.cpp`

Split in v0.5.0 (the one file had passed 1200 lines): the sweeps are in
`test_seeding_bench.cpp`, the always-on cases below in
`test_seeding_contracts.cpp`, and the harness both use (ground-truth fields,
scenarios, per-row measurement, table/CSV writers) in
`framework/seed_bench.h`. The suite name is unchanged.

Mostly a **measurement harness**, not a gate: `CandidateSweep`,
`RealSpeckleSweep`, `Challenge5`, `Challenge14Sinusoid` and `LargeMotionSweep`
all early-return unless `SEMPER_RUN_SEEDBENCH=1`, and the last three also need
`SEMPER_DICE_REPO` pointed at a `dicengine/dice` checkout. They produce the
tables in `docs/SEEDING_BENCHMARK.md`. Seven cases are always on:

| Test | Proves | Failure would mean |
|---|---|---|
| `PhaseCorrelateSignConvention` | `phase_correlate_roi` returns `(u, v)` such that reference point `p` appears at `p + (u, v)` | Every anchor is seeded with the sign of the motion inverted |
| `LogPolarRotationScale` | The Fourier-Mellin estimate reports rotation (sign and size, within 0.15°) and scale (within 0.005) in the engine's shape-function convention, at −7° to +25° and 0.97–1.04 with translation; the similarity it returns carries the right centre displacement and `sR − I` gradients; and below `kSimilarityMinCornerPx` — including FullFieldGolden's affine scenario — the result is the translation-only one, bit for bit | Rotated frames are seeded with the turn inverted, or small motion silently leaves the translation-only path and every golden field moves |
| `AnchorSeedingRotationGate` | At 5°, 15°, and 10° + 5% zoom + 20 px shift, the lattice locks, seeds with the similarity, routes `FULL`, keeps ≥ 90% hull coverage and ≥ 95% mesh fraction, and stays under 0.05 px vertex error | Handheld-capture rotation is back to the 0.45 / 0.15 coverage of the translation-only seeder (`SEEDING_BENCHMARK.md` §4.5) |
| `AnchorSeedingHalfTurn` | At a pixel-exact quarter turn (`cv::rotate`), ±90°, 100°, 135°, 179° and −120°, L0 picks the right half-turn (≤ 0.15°, centre ≤ 0.25 px) and the solve returns ≥ 95% of the zero-motion nodes within 0.5 px (3600/3600 measured at each) | The log-polar angle is taken modulo 180° again: 0.5.0 returned `kSeedFailed` (−5) from 90.5° to 180° (`SEEDING_BENCHMARK.md` §4.5.2) |
| `SimilarityFallsBackUnderStretch` | On the IncBench frame where the log-polar estimate misreads a 45% uniaxial stretch as −5.8° × 0.92, the similarity-seeded lattice is rejected for the translation-only one, and the field stays at ≤ 0.05 px rms (0.0137 measured; 0.2150 with the fallback disabled) | A misfired similarity publishes a self-consistent cluster of wrong anchors and Path B floods from it |
| `AnchorSeedingQualityGate` | Three of the seven sweep scenarios — 0.4 px translation, 0.5° rotation, 12 px translation — hold a phase lock, route `FULL`, cover ≥ 90% of the ROI, and stay inside loose vertex- and gradient-error bounds | Seeding stopped working, rather than merely got slower |
| `PublishAnchorResultsRespectsMedianTest` | `publish_anchor_results` writes only anchors that cleared the result gate *and* survived the universal median test, consumes one compute-order number per published point, and credits the thread that ran the IC-GN | A periodic-speckle blunder reaches the field and Path B floods from it — the failure the median test exists to prevent |
| `AnchorLatticeSizing` | `plan_anchor_lattice` keeps the isotropic stride on a square ROI (64×64 → 289 anchors, the count the sweep reports), holds a 3-node floor on the short axis of a 1000×6 ROI without exceeding 2×`kAnchorTarget`, is symmetric under transpose, and returns an empty lattice for a degenerate grid | A long thin ROI — a beam, a weld seam — silently costs several times the intended seeding budget |

The gate's bounds are 3–5× the measured medians, each annotated in-file
with the value it was set from. It is deliberately not a benchmark: it should
fire when the lattice stops locking or stops converging, and stay silent when a
host is 20% slower or noisier. Read exact numbers from the sweep.

```bash
SEMPER_RUN_SEEDBENCH=1 ./build/full/bin/dic_tests SeedBench
SEMPER_RUN_SEEDBENCH=1 SEMPER_SEEDBENCH_REPEATS=5 \
  SEMPER_SEEDBENCH_CSV=sweep.csv ./build/full/bin/dic_tests SeedBench.CandidateSweep
```

## Suite: `FullFieldDeterminism` — `integration/test_full_field_determinism.cpp`

Solves the same inputs four times in one process, each time from a fresh
reference cache as the app does, and compares the packed output byte for byte.
App default grid (subset 21, step 5, strain window 15). OpenCV-gated.
Synthetic speckle, because TIFF decoding crashes in the MinGW host build.

| Test | Proves | Failure would mean |
|---|---|---|
| `TwoFrameBatch_BitIdentical` | Two load steps against one cache: every output byte and the Path A / Path B point counts repeat | A parallel phase (pre-pass, Path A, Path B flood) depends on thread timing |
| `MaskedHole_BitIdentical` | Same under a ROI mask with a hole, where Path B solves most of the field | As above; this case exposed the Path B claim race (app TD-65) |

## Suite: `Incremental` — `integration/test_incremental.cpp`

`IncrementalSession` on sequences with exact ground truth
(`framework/sequence.h`): every frame is rendered from one analytic speckle
(`SmoothSpeckleField`, Gaussian blobs truncated at 4σ so the texture is smooth)
under a known motion, so total displacement and Green-Lagrange strain at each
frame-0 node are known in closed form. `SEMPER_INC_TRACE=1` prints per-frame
error, an error map and a bias fit. OpenCV-gated.

| Test | Proves | Failure would mean |
|---|---|---|
| `IntervalZero_BitIdenticalToRunFullField` | `update_interval = 0` equals `run_full_field` per frame, byte for byte, with and without a mask | The "off" setting changes results |
| `FirstFrame_BitIdenticalToRunFullField` | With N = 1 the first frame equals `run_full_field` | Turning incremental on changes frames that never used a moved reference |
| `UpdateSchedule_FollowsInterval` | `reference_frame` / `reference_updated` follow 0,0,0,3,3,3,… for N = 3 | The reference moves on the wrong frame |
| `StretchRotation_TracksWhereFixedReferenceDegrades` | 448², 20 frames to 40% stretch and 30° rotation: N = 1 keeps ≥ 98% of frame-0 nodes, rms U ≤ 0.10 px, rms strain ≤ 5e-3, and more nodes within 0.2 px than the fixed reference. Measured: 196/196 nodes, rms 0.047 px | Composition or the mask warp broke, or incremental no longer helps where it should |
| `Rotation90_200Frames_TracksTheTurningRoi` | 256², steady rotation at 0.45° per frame, no stretch. N = 5 (6×6) and N = 10 (4×4) to 90°, N = 1 (6×6) to 45°. Every frame solves, ≥ 98% of frame-0 nodes are still tracked, and ≥ 95% / 95% / 90% of frame 1's points are within 0.5 px. Rms U ≤ 0.08 / 0.20 / 0.40 px; the rms carries one weakly textured corner of this speckle that drifts over 100+ updates. Measured: 482/484 tracked, 391/397 within 0.5 px, rms 0.047 (N = 5); 483/484, 392/397, 0.134 (N = 10); 483/484, 390/397, 0.292 (N = 1, 45°) | Edge nodes of a turning ROI fall out of the ref-k scan (material padded by `half` instead of `half·√2`), or the mask warp blanks bands of rows (the old per-pixel Newton inverse followed false branches). Before the fixes, every mode lost the chain between 50° and 85° |
| `MaskedSequence_TracksMaterialInsideMask` | Ring mask with a hole, 10 frames to 15% stretch and 8°: every reported point is a frame-0 node inside the mask, ≥ 85% of frame-1's points remain, rms U ≤ 0.15 px | The mask is not carried with the material, so subsets cross the hole edge after an update |
| `LostNodesNeverReturn` | A 70 px drift carries part of the ROI past the image-edge margin: `tracked_points` never increases and ends below the frame-0 count | A lost node comes back with a stale base |
| `StationaryBand_NodesCoastAcrossAndComeBack` | 320×480, 36 frames moving 108 px down under a stationary 9 px band that shows fresh speckle every frame (glare, a background edge under translucent film). N = 1. ≥ 98% of the nodes tracked after frame 1 still tracked. Nodes whose path stayed clear of the band: ≥ 90% reported, worst error ≤ 0.6 px. Nodes that crossed it: ≥ 45% reported again by the last frame, median error ≤ 0.08 px, worst ≤ 0.4 px. Measured: 806 tracked (frame 1: 816; frame 1 cannot measure the nodes in the band, and they are not followed); clear 168/180, worst 0.46 px; crossed 146/285, median 0.029 px, worst 0.20 px (the rest still wait for a clean subset) | Coasting stops carrying nodes across a gap (before it, every node that reached the band was lost), or re-anchoring stops removing the coast's drift (without it, crossed nodes came back up to 2.3 px off) |
| `EarlyUpdate_WhenReportedPointsDrop` | The band sequence with N = 100: the reference moves early, and only on frames reporting < 95% of the first frame on the current reference; N = 0 never moves | The early update fires on noise, or not at all |
| `RepeatRun_BitIdentical` | The same sequence twice gives the same bytes | Run-to-run nondeterminism in composition or the mask warp |
| `Contracts_ErrorCodes` | `-3` before `init`, for a wrong-size frame and for a null buffer; `-2` / `-4` / `-3` from `init`, including an even subset (`-3`), a strain window far beyond the grid (`-3`) and a rect off the image (`-2`) | The documented codes drifted |
| `Cancel_LeavesSessionUnchanged` | A frame cancelled mid-solve, then resubmitted, gives the same bytes as an uncancelled run | A cancelled frame half-updates the session |
| `LongHorizon_160Frames` | 256², 160 frames of 0.1% stretch (16% total), noise σ = 1, N = 1 and N = 5: ≥ 95% of nodes tracked, rms U ≤ 0.45 / 0.35 px, rms strain ≤ 3.5e-2, RSS growth < 16 MB. Measured: rms 0.40 (N = 1), 0.19 (N = 5), fixed 0.05 | Error growth over a long horizon got worse, or memory leaks per reference update |

The long-horizon bounds are wide on purpose. On slow loading each increment is
a small sub-pixel step, and the per-solve interpolation bias at that step has
one sign, so it adds up roughly linearly with the number of updates. That is
the known cost of incremental DIC, not a defect; see MATHEMATICS.md §9.5 and
INCREMENTAL_BENCHMARK.md.

## Suite: `IncBench` — `integration/test_incremental_bench.cpp`

A **measurement harness**, not a gate: `FixedVsIncremental` early-returns unless
`SEMPER_RUN_INCBENCH=1`, prints per-scenario tables, optionally writes a
per-frame CSV, and asserts nothing about which mode wins. It produces
[INCREMENTAL_BENCHMARK.md](INCREMENTAL_BENCHMARK.md). Scenarios S1–S9 are
synthetic sequences of 200–300 frames (static, slow stretch, translation,
rotation, large stretch, combined, non-affine, increment size, ring mask); S10
is DICe's OHT ladder. Fixed, N = 1, 2, 5 and 10 run in lockstep on the same
frames.

```bash
SEMPER_RUN_INCBENCH=1 SEMPER_INCBENCH_CSV=incbench.csv ./build/full/bin/dic_tests IncBench
SEMPER_RUN_INCBENCH=1 SEMPER_INCBENCH_QUICK=1 ./build/full/bin/dic_tests IncBench   # T = 160, σ = 2, one seed
SEMPER_RUN_INCBENCH=1 SEMPER_INCBENCH_ONLY=S4 ./build/full/bin/dic_tests IncBench
python tools/plot_incremental_bench.py incbench.csv docs/images/incremental
```

The full run takes about 40 minutes on a 20-thread laptop (peak RSS 58 MB, on S10). `tools/cache_bench.c`
times the fixed-reference cache (first frame vs the rest) against any
`libsemper_c.so` build: `gcc -O2 -o cache_bench tools/cache_bench.c -ldl -lm`.

## Suite: `IncrementalMath` — `unit/test_incremental_math.cpp`

| Test | Proves |
|---|---|
| `Sample_OnNode_ReturnsNodeExactly` | Sampling at a node returns its value, not an interpolation |
| `Sample_Bilinear_ReproducesLinearField` / `Sample_TaylorFallback_ReproducesLinearField` | Both sampling paths are exact for a linear increment field |
| `Sample_TooFarFromAnyNode_IsInvalid` | Beyond 1.5 cells from any valid node the sample fails |
| `Sample_CorrIsWorstContributor` | The reported ZNSSD is the max of the contributing nodes |
| `NodeField_DropsUnsolvedAndOutliers` | Unsolved nodes, negative ZNSSD and median-test outliers are not sampled |
| `Fit_ReproducesLinearFieldAcrossAHole` | Across a band 4 nodes wide that sampling cannot reach, the coasting plane fit reproduces a linear field |
| `Fit_TooFewOrCollinearNodes_IsInvalid` | One row of nodes (no spread across it), five nodes (one short), or none within the radius: no fit; six spread nodes: exact |
| `Fit_CorrIsWorstUsed` | The fit's ZNSSD is the worst of the nodes inside the radius, not of the field |
| `AnchorNeed_DropRefreshKeep` | Far from any gap: drop the anchor; clean with a gap or the grid edge within reach: cut a new one; a hole inside its own subset: keep the old one |
| `Anchor_MatchesTheNodeWhereItMoved` | An anchor cut at a sub-pixel node position and matched 1 px off from where it moved lands within 0.05 px; a match further from the guess than allowed, or on another texture, fails; a patch leaving the image is not cut |
| `MatchAnchor_RejectsUnconvergedSolve` | A match whose IC-GN hard-rejects at its starting guess (status 1, score 2.0) does not confirm the node, however close to the guess it stopped; a real match still does. Before v0.5.0 it confirmed with corr 2.0 |
| `MaskWarp_ZeroDisplacement_IsIdentity` / `_Translation_ShiftsMask` / `_Stretch_ScalesArea` | The forward mask warp is exact for identity and translation, and the area of a stretched mask is within 1% |
| `MaskWarp_LargeRotation_IsTheRotatedMask` | An 84° turn, with the material reaching 16 px past the outer nodes: area within 2%, no pixel set more than 1 px outside the analytically rotated square, none missing more than 1 px inside it |
| `MaskWarp_NoValidNode_IsEmpty` | No tracked node, no material |

## Suite: `DiceIncremental` — `dice/test_incremental_oht.cpp`

DICe's open-hole tension images `oht_cfrp_00 → 01 → 03 → 06 → 11` through an
`IncrementalSession` (N = 1, subset 27, Keys 6×6, step 7 so every DICe point is a
grid node), compared with DICe's fixed-reference solutions. The fixed-reference
full field on the same grid is logged next to it.

| Rung | Incremental rms du / dv (px) | Fixed rms du / dv (px) |
|---|---|---|
| 01 | 0.0006 / 0.0006 | 0.0006 / 0.0006 |
| 03 | 0.0124 / 0.0186 | 0.0011 / 0.0010 |
| 06 | 0.0442 / 0.0230 | 0.0014 / 0.0017 |
| 11 | 0.0584 / 0.0260 | 0.0024 / 0.0029 |

All 230 points are compared on every rung. Most of the incremental error is a
uniform u bias (mean du −0.054 px at rung 11), built up from u increments of
about −0.2 px each. Bounds: ≥ 90% compared, rms < 0.1 px.

### C ABI contract — `tests/c/contract.c` (built with `SEMPER_BUILD_C_SDK`)

Where `c/smoke.c` proves the happy path runs, `c/contract.c` pins the **Frozen
ABI surface** the private downstream app and any external C/C#/Rust caller depend
on. It runs in the `c-sdk-smoke` CI job and returns non-zero on any breach:

- Compile-time `_Static_assert` on every Frozen constant (`SEMPER_ERR_*` incl. `SEMPER_ERR_SEED`,
  `SEMPER_FLOATS_PER_POINT`, `SEMPER_METRICS_LEN`) and on `sizeof`/`offsetof` of
  all eight `semper_params` fields — a reordered field is a build failure, not a
  silent parameter mis-read.
- Runtime: null-argument guards → `INIT`, run-before-`set_reference` → `INIT`,
  degenerate ROI → `ROI`, even/zero subset → `INIT`, an `INT_MAX` strain
  window → `INIT` (the largest accepted one still solves), and an `INT_MAX` rect
  clipped to the image rather than aborting the process (v0.5.0), the capacity-drop rule, the `metrics_len` rule,
  per-engine cancel independence, and the `MAJOR.MINOR.PATCH` version format.
- Incremental session (v0.4.0): `sizeof`/`offsetof` of the six
  `semper_incremental_info` fields, null guards, process-before-init → `INIT`.
  `c/smoke.c` runs two 3 px frames and checks a 6 px total.
- **Golden symbols** (`tests/c/abi_symbols.txt`): CI diffs `nm -D` of
  `libsemper_c.so` against the documented exports, so an accidental export or
  a dropped `SEMPER_C_API` annotation fails the build.

## Coverage (report-only)

The `coverage` CI job configures `-DSEMPER_COVERAGE=ON`, runs `dic_tests`, and
publishes a `gcovr` report over `src/` as an artifact. **There is no threshold or
gate** — it exists to make it visible when a source file (like `image_codec.cpp`,
which was at ~0%) drifts out of the exercised set, not to block a PR on a number.
The build is `-O1`: at `-O0` the incremental suites alone outran a 90-minute
job. Leave gcc's counter update at its default (atomic under `-fopenmp`);
`-fprofile-update=single` corrupted solver results across OpenMP threads.
Locally: `cmake -S tests -B build/cov -DSEMPER_COVERAGE=ON -DDIC_REQUIRE_OPENCV=ON
&& cmake --build build/cov -j && ./build/cov/dic_tests && gcovr --root . build/cov
--filter 'src/' --txt`.

## Downstream app tests (informative)

Kotlin JVM unit tests, instrumented JNI smokes, and backend contract tests live
in the **private** application repository that consumes this engine as a
submodule. They are not part of this tree. This engine repo owns algorithmic
correctness; the app owns bridge/orchestration contracts.

---

## Test layout

Host tests live under `tests/`, grouped by scope:

```
tests/
  test_main.cpp           micro-framework runner entry point
  framework/              the test harness — synthetic.h, test_framework.h
  shim/                   host stand-ins for OpenCV configs
  c/smoke.c               C ABI smoke (built with SEMPER_BUILD_C_SDK)
  c/contract.c            C ABI Frozen-contract binary (semper_c_contract)
  c/abi_symbols.txt       golden list of exported semper_* symbols
  unit/                   one component vs. an oracle / mathematical identity
                            test_simd_kernels, test_image,
                            test_subset_precomputer, test_strain_calculator,
                            test_cancel_token, test_image_codec (OpenCV-gated)
  integration/            the assembled engine end-to-end + robustness
                            test_optimization_engine, test_robustness,
                            test_full_field_contracts,
                            test_full_field_golden, test_seeding_contracts,
                            test_seeding_bench, test_full_field_determinism,
                            test_reference_cache (the last six OpenCV-gated)
  dice/                   DICe golden comparisons
  perf/                   throughput gates
```

`tests/CMakeLists.txt` lists sources under `DIC_UNIT_TESTS` /
`DIC_INTEGRATION_TESTS` / `DIC_DICE_TESTS` / `DIC_PERF_TESTS`, plus
`DIC_PIPELINE_TESTS` for the OpenCV-gated suites (`test_full_field_contracts`,
`test_full_field_golden`, `test_seeding_contracts`, `test_seeding_bench`, `test_full_field_determinism`,
`test_image_codec`, `test_reference_cache`) — attached only when OpenCV is present.

## Adding a new test

1. Pick the suite file, or create `unit/test_<module>.cpp` (component) or
   `integration/test_<module>.cpp` (end-to-end) and add it to the matching
   `DIC_UNIT_TESTS` / `DIC_INTEGRATION_TESTS` list in `CMakeLists.txt` — or
   `DIC_PIPELINE_TESTS` if it needs OpenCV (imgcodecs/imgproc/features2d).
2. `TEST_CASE(Suite, Name) { ... }` — use `CHECK`, `REQUIRE`, `CHECK_NEAR`,
   `CHECK_REL` (see `framework/test_framework.h`). The runner filters by a
   **substring** of `Suite.Name`, so `dic_tests ImageCodec` runs the whole suite.
3. For engine tests, build ground truth with `SpeckleField` +
   `AffineDeformation` — never by resampling images, and never with warps that
   don't match the engine's shape-function convention.
4. Document the test's *reason to exist* in this file. A test whose failure
   nobody can interpret is a liability.

## Host vs JNI responsibility split

- **Host C++ tests** (this suite) own algorithmic correctness: displacement
  accuracy, strain math, SIMD equivalence, determinism, and robustness.
- **C ABI / Python smokes** own binding marshalling at the public SDK surface.
- **Downstream Android JNI / JVM tests** (private app) own runtime/bridge
  correctness: `System.loadLibrary`, OpenMP on the Android runtime, and Kotlin
  orchestration. They do not re-assert displacement accuracy here.

The downstream app's own test map lives in that private repository (see
["Downstream app tests"](#downstream-app-tests-informative) above); it is not
reachable from this tree.

## Known limitations / future work

- **Per-ABI numerical drift**: the host suite runs on x86 SSE. To compare ABIs,
  build the same suite with the NDK toolchain per-ABI and run on devices —
  tolerances are already set to absorb fast-math reassociation differences.
- **Silent output truncation** (`FullField.UndersizedBuffer`): when `out_capacity`
  is smaller than the solved-point count, `run_full_field` drops the overflow and
  returns the reduced count — it does **not** signal that truncation happened. This
  is the Frozen capacity-drop rule (docs/CONTRACT.md §A.4); changing it to an error
  would break metrics-only (`capacity == 0`) and deliberate partial-buffer callers,
  so it is a documented behavior, not a bug. `contract.c` pins it as-is.
- **`metrics_len` in 1..15 writes nothing** (`FullField.MetricsLen16_...`,
  `contract.c`): the metrics writer is gated on `metrics_len >= 16` and silently
  writes nothing below that. All in-repo callers pass ≥ 16 (JNI self-gates at 16,
  Python and the C ABI pass 17), so this is pinned as current behavior rather than
  changed — the Frozen minimum is 16.
- **`image_dimensions` has no raw-buffer awareness**
  (`ImageCodec.ImageDimensions_RawBufferYieldsZeros_KnownTrap`): a raw RGBA/gray
  blob is not a decodable container, so it reports 0×0. Callers holding raw buffers
  already know the dimensions; the test pins the trap so nobody relies on it.
