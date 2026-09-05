# Test suite reference

Complete catalog of the automated tests: what each one proves, why it exists,
and how to run everything. The engine it exercises is described in
[ARCHITECTURE.md](ARCHITECTURE.md); CI wiring is in
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

---

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

> The whole test tree compiles with `-fno-fast-math -ffp-contract=off`, the
> same floating-point model as `semper_math` in the shipped library, so
> numerical regressions surface here first. This applies to the harness too,
> not just the engine sources: `framework/synthetic.h` sums hundreds of
> `std::exp` terms per pixel, and under `-ffast-math` GCC vectorizes that
> onto libmvec — whose AVX2 `exp` disagrees with its SSE2 one, so the
> generated **images** differed between `-march` levels. A determinism gate
> whose own inputs are non-deterministic proves nothing.

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

What "the same result" means, and when a difference is a bug.

> This section changed substantially. The engine used to compile with
> `-ffast-math` and reduce over hardware-width SIMD lanes, which made results
> legitimately build-specific; the tolerances below were sized for that. Both
> are now pinned. See [DETERMINISM.md](DETERMINISM.md) for the full contract.

**Same build, same inputs → bit-identical.** Two tiers guard this:

| Scope | Test |
|---|---|
| One subset solve, repeated | `Engine.RepeatSolve_BitIdentical` |
| Concurrent solves vs single-threaded | `Robustness.ConcurrentSolves_BitIdenticalToSingleThread` |
| A whole `run_full_field` solve, repeated | `FullFieldGolden.RepeatSolve_IsDeterministic` |

The third is the one that matters most and is newest. Path B used to drain a
shared priority queue from racing threads, so each point's guess came from
whichever parent won a `compare_exchange`: measured at **0 of 15 runs
reproducing**, with 10-230 of ~3100 output floats differing between two
consecutive solves of the same binary. Nothing caught it, because the other
two tests are subset-level and the golden corpus never enters
`run_full_field`. It is now a level-synchronous wavefront and reproduces
15/15.

**Different target ISA, same toolchain → also bit-identical.** This is the
part that used to be "small drift is expected". SSE2, AVX2 and NEON now
produce byte-identical output, enforced by the `determinism` CI job, which
builds the same source at `-march=x86-64` and `-march=x86-64-v3` and
byte-compares both golden fixtures. Reproduce it locally with the commands in
[tests/README.md](../tests/README.md#determinism).

**Different toolchain or libm → small drift possible, in the fixtures only.**
The remaining dependency is not in the engine: `tests/framework/synthetic.h`
builds its ground-truth images from several hundred `std::exp` terms per
pixel, so a different libm can move the last ulp of the *input*. This is why
`test_golden_corpus.cpp` compares at 1e-6 px / 1e-7 strain rather than
exactly — tight enough to catch any real change in engine arithmetic, loose
enough to survive a glibc bump. The exact gate is the CI job, not the
tolerance.

| Quantity | Expected agreement across ISAs (same toolchain) |
|---|---|
| Displacements (U, V) | **exact** |
| Strains (Exx, Eyy, Exy) | **exact** |
| Solver stats (solved/dead counts, convergence %) | **exact** |
| Report max/min locations | **exact** |

Any difference at all across ISAs is now a regression, not noise. Do not
widen a tolerance to absorb one — work the checklist in
[DETERMINISM.md](DETERMINISM.md).

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

The tests above prove the kernels are *accurate*. These prove something
stricter, and for the OpenCL work more important — that they are
**bit-identical** to the canonical reference the device kernels will
reproduce. Exact `==`, never a tolerance: if the CPU vector path drifts from
the reference by one ulp there is no single answer for the GPU to match.

| Test | Proves |
|---|---|
| `SumSqDiff_BitIdenticalToCanonical` | Vector path ≡ `semper_canon_sum_sq_diff`, every size |
| `Znssd_BitIdenticalToCanonical` | Vector path ≡ `semper_canon_znssd_sum` |
| `ErrorAndGradient_BitIdenticalToCanonical` | Fused kernel ≡ canonical, error and all 6 gradients |
| `VectorWidthIsPinnedTo4Lanes` | `CV__SIMD_FORCE_WIDTH` took effect. The lane count **is** the summation order, so an 8-lane register silently changes every reduction's association |

## Suite: `ClRuntime` — `unit/test_cl_runtime.cpp`

The OpenCL backend's probe and build plumbing. Compiled only under
`-DSEMPER_OPENCL=ON`; otherwise the file reduces to one test asserting the
backend really is absent, so the suite count does not shift silently with the
flag.

Every test must pass on a machine with **no** OpenCL, because that is the CI
default and the common developer case — the backend's contract is that it
degrades to the CPU, not that it is present.

| Test | Proves |
|---|---|
| `ProbeIsSafeAndIdempotent` | Probing never throws whatever is installed, and the result is cached rather than re-probed |
| `UnavailableAlwaysExplainsItself` | `unavailable_reason` is never empty when the backend is unused. A GPU that goes unused silently is indistinguishable from one that works — this is the assertion that makes that visible |
| `CapabilitiesImplyAvailability` | `fp64` / `exact_fp32` are never claimed while the backend is unusable |
| `BuildOptionsContainNoAccuracyDestroyingFlag` | No `-cl-fast-relaxed-math`, `-cl-mad-enable`, `-cl-unsafe-math` or `-cl-no-signed-zeros`, and `-cl-fp32-correctly-rounded-divide-sqrt` **is** present. These are contract, not tuning — exactly the flags someone adds while chasing a benchmark |
| `EmbeddedKernelIsSelfContained` | `scripts/embed_cl_kernels.py` expanded the first-party include: the `_g` address-space variant exists (proving `canonical_reductions.inc` expanded twice), `FP_CONTRACT OFF` survived, and no unresolved `#include <semper/...>` remains |
| `EnvDisableForcesCpuPath` | `SEMPER_OPENCL_DISABLE=1` forces the CPU path, for bisecting a suspected device-side difference |
| `DeviceReproducesCanonicalReductionExactly` | **Device-gated.** A real OpenCL device returns bit-identical results to `semper_canon_sum_sq_diff` at n = 0…729, tail included. Skips with a printed reason rather than passing vacuously when no device is present |

## Suite: `CanonicalReduce` — `unit/test_canonical_reduce.cpp`

Pins the reduction **order**, not merely the value, in
`include/semper/kernels/canonical_math.h`. Every long summation accumulates
into four stride-4 accumulators combined as `((a0+a1)+(a2+a3))`, with the
tail added to the combined result in index order. A reduction that produces
the mathematically-correct sum by a different association fails here, because
the GPU would then have no fixed target.

| Test | Proves |
|---|---|
| `SumSqDiff_MatchesSpecExactly` | Matches an independently-written restatement of the documented order, across every tail residue and n = 729 (a 27 px subset) |
| `ZnssdSum_MatchesSpecExactly` | Same, for the ZNSSD residual |
| `FusedGradient_ResidualMatchesZnssdSum` | Fused kernel's residual ≡ the standalone one, and each of the six gradient planes independently follows the same association |
| `BlockedBeatsNaiveOnAverageAgainstDoubleOracle` | Pinning the order costs no accuracy. Stated across 300 trials, not per-sample: pairwise summation wins on error *growth*, but on any single input a naive accumulator can land closer by luck of rounding |
| `EmptyInputReturnsZero` | n = 0 (a fully masked subset) returns exactly zero rather than NaN |

## Suite: `CanonicalInverse` — `unit/test_canonical_inverse.cpp`

`semper_inv3x3` (fp64, strain normal equations) and `semper_inv6x6` (fp32,
ICGN Hessian) replace Eigen's `inverse()` on **both** sides of the CPU/GPU
boundary — Eigen's blocked pivoting LU cannot be called from a kernel, and
having the device chase it is not a contract anyone can hold. Verified
structurally (`A·A⁻¹ = I`, known analytic inverses) rather than by diffing
against Eigen, because matching Eigen is explicitly not the goal.

| Test | Proves |
|---|---|
| `Inv3x3_KnownAnalyticInverse` | Exactly-representable diagonal case is exact |
| `Inv3x3_RoundTripsToIdentity` | 200 well-conditioned matrices, residual ≤ 1e-12 |
| `Inv3x3_SingularIsRejected` | Structurally rank-deficient input returns failure, not garbage |
| `Inv6x6_IdentityIsItsOwnInverse` | Trivial case exact, determinant 1 |
| `Inv6x6_RoundTripsToIdentityOnSpdHessians` | 50 SPD matrices shaped like real ICGN Hessians (built as JᵀJ) |
| `Inv6x6_RequiresRowPivoting` | An anti-diagonal matrix — every leading pivot is zero without row swaps |
| `Inv6x6_SingularIsRejected` | Duplicate row detected |
| `Inv6x6_PivotTieBreakIsLowestRow` | Equal-magnitude pivots resolve deterministically. An unspecified tie-break is a bit-exactness hole, and ties happen routinely on structured matrices |

## Suite: `FullFieldGolden` — `integration/test_full_field_golden.cpp`

The end-to-end regression and determinism gate. `GoldenCorpus` pins the
*subset* solver by calling `precompute_subset` / `calculate_deformation`
directly; it never enters `run_full_field`, which left the entire
orchestration layer — AKAZE seeding, the Delaunay mesh guess field, Path A,
Path B propagation, and strain — with no golden reference at all. Those are
exactly the stages moving to the GPU.

| Test | Proves | Failure would mean |
|---|---|---|
| `CaptureOrCompare` | Packed 8-float output and the counting metrics match `fixtures/full_field_golden.bin` for a translation and an affine scenario | Orchestration changed: seeding, mesh, path split, or strain |
| `RepeatSolve_IsDeterministic` | Two solves in one process agree **exactly** | An order-dependent step is back in the pipeline — a guess read mid-round, or a tie-break resolved by completion order rather than flat index |

Values compare at 1e-6 px / 1e-7 strain and grid coordinates exactly; the
counting metrics compare exactly, and timing slots are excluded because they
differ every run by construction.

> Its `STRAIN_WIN` is 48 against `step` 12, not the 15 the contract test
> uses. `strain_window` is a diameter in pixels and must be at least
> `2 * step`, or the VSG window contains only its own centre point, fails
> `valid_pts >= 3`, and the strain filter drops the **entire** field
> (484 attempted → 0 output).

## Suite: `Image` — `unit/test_image.cpp`

Interpolation ladder + gradients + blur, validated via mathematical identities
(no golden files).

| Test | Property exploited |
|---|---|
| `InterpolatorsReproducePixelValuesAtIntegerCoords` | All three kernels are *interpolating*: weight 1 at s=0, 0 at other integers |
| `InterpolatorsExactOnLinearRamp` | Catmull-Rom & Keys reproduce degree-1 polynomials exactly at any sub-pixel position |
| `GradientOfLinearRampIsExactSlope` | 5-point central difference is exact for degree ≤ 4 |
| `BlurPreservesConstantImage` | Normalized kernel ⇒ constant in = constant out |
| `BilinearOutOfBoundsReturnsZero` | The `0.0f` dead-pixel sentinel contract that ICGN's `val > 0` guard relies on |
| `BoundaryDemotionLadderIsContinuousInRange` | 6×6→bilinear and 4×4→bilinear demotion never extrapolates outside [0,255] |

## Suite: `SubsetPrecomputer` — `unit/test_subset_precomputer.cpp`

| Test | Proves |
|---|---|
| `MeanAndStdMatchManualComputation` | Stats vs a double-precision manual computation; normalized intensities are zero-mean/unit-RMS |
| `SdiPlanesMirrorSteepestDescentImages` | **The SIMD-refactor regression guard**: SoA `sdi_planes` ≡ AoS `steepest_descent_images` element-wise. The SIMD fast path reads planes, the masked path reads vectors — drift silently corrupts solutions |
| `SteepestDescentImagesFollowDefinition` | `sd = [gx, gy, gx·x, gx·y, gy·x, gy·y]/σ` at spot-checked pixels |
| `HessianIsSymmetricAndInverseIsValid` | `H = Hᵀ` and `H·H⁻¹ ≈ I` |
| `RejectsSubsetOffImageBoundary` | Boundary subsets → `is_initialized == false` (never solved) |
| `FastPathMatchesFullPrecompute` | `compute_hessian_only` + `precompute_subset_fast` ≡ full `precompute_subset` — the batch pipeline treats them as interchangeable |

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

## Suite: `CancelToken` — `unit/test_cancel_token.cpp`

The flag `run_full_field` polls to stop a solve mid-frame. The polling itself
needs a real solve (OpenCV, threads) and is covered on the Android side; what is
pinned here is the contract those polls rest on.

| Test | Proves |
|---|---|
| `StartsClear` / `RequestIsObserved` | The flag reads back what was set |
| `StaysSetUntilCleared` | Sticky across a thousand polls — the caller must clear it before a run, which is why both run paths assign `false` on entry |
| `CrossesThreads` | A worker spinning on the flag sees a request made from another thread — the flag is set on the UI thread, read by solver workers |
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
| `Relifecycle_UpdatesDimsAndClearsAkaze` | Re-`set_from_gray` with a new size updates dims and clears AKAZE state (`akaze_scale == 0.25`) | Stale seeding/dimension state leaks across references |
| `EmptyInput_LeavesCacheCleared` | Empty gray clears the cache (`width == 0`, `ref_img == nullptr`) instead of retaining the old reference | A failed re-init silently solves against the previous frame |
| `Reset_ClearsEverything` | `reset()` frees and zeroes all state | Double-free or leak on teardown |

Compile-time `static_assert`s pin `ReferenceCache` as non-copyable/non-movable
(§A.2 of the contract) regardless of OpenCV.

## Suite: `FullField` — `integration/test_full_field_contracts.cpp`

Host characterization of `run_full_field` — the Frozen return codes, the
capacity-drop rule, and the 17-float metrics layout — without depending on solve
quality. OpenCV-gated.

| Test | Proves | Failure would mean |
|---|---|---|
| `DegenerateRoi_ReturnsRoiError` | `rect_w < step` → `-2` | ROI validation regressed |
| `EmptyDeformed_ReturnsInitError` | Empty deformed image → `-3` | Init-guard regressed |
| `StaleCancelDoesNotAbortNextSolve` | A leftover cancel is cleared on entry | A prior cancel poisons the next solve |
| `UndersizedBuffer_TruncatesWithoutOverflow` | Small `out_capacity` drops points, never overflows, returns `≤ capacity/8` | Buffer overflow on a caller-sized buffer |
| `MetricsLayout_Contract` | attempted ≥ solved ≥ 0; `rejected == attempted − solved`; convergence % ∈ [0,100]; seed flag ∈ {0,1,2} | A downstream telemetry reader mis-parses the Frozen slots |
| `MetricsLen16_LeavesSlot16Untouched` | `metrics_len == 16` fills 0..15 and never writes slot 16 | Write past a 16-float caller buffer |
| `NullMetrics_DoesNotCrash` | `metrics == nullptr` is legal | Null-deref when a caller wants points only |

### C ABI contract — `tests/c/contract.c` (built with `SEMPER_BUILD_C_SDK`)

Where `c/smoke.c` proves the happy path runs, `c/contract.c` pins the **Frozen
ABI surface** the private downstream app and any external C/C#/Rust caller depend
on. It runs in the `c-sdk-smoke` CI job and returns non-zero on any breach:

- Compile-time `_Static_assert` on every Frozen constant (`SEMPER_ERR_*`,
  `SEMPER_FLOATS_PER_POINT`, `SEMPER_METRICS_LEN`) and on `sizeof`/`offsetof` of
  all eight `semper_params` fields — a reordered field is a build failure, not a
  silent parameter mis-read.
- Runtime: null-argument guards → `INIT`, run-before-`set_reference` → `INIT`,
  degenerate ROI → `ROI`, the capacity-drop rule, the `metrics_len` rule,
  per-engine cancel independence, and the `MAJOR.MINOR.PATCH` version format.
- **Golden symbols** (`tests/c/abi_symbols.txt`): CI diffs `nm -D` of
  `libsemper_c.so` against the six documented exports, so an accidental export or
  a dropped `SEMPER_C_API` annotation fails the build.

## Coverage (report-only)

The `coverage` CI job configures `-DSEMPER_COVERAGE=ON`, runs `dic_tests`, and
publishes a `gcovr` report over `src/` as an artifact. **There is no threshold or
gate** — it exists to make it visible when a source file (like `image_codec.cpp`,
which was at ~0%) drifts out of the exercised set, not to block a PR on a number.
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
                            test_full_field_contracts, test_reference_cache
                            (the last two OpenCV-gated)
  dice/                   DICe golden comparisons
  perf/                   throughput gates
```

`tests/CMakeLists.txt` lists sources under `DIC_UNIT_TESTS` /
`DIC_INTEGRATION_TESTS` / `DIC_DICE_TESTS` / `DIC_PERF_TESTS`, plus
`DIC_PIPELINE_TESTS` for the OpenCV-gated suites (`test_full_field_contracts`,
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
