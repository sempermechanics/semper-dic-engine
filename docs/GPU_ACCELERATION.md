# GPU acceleration — roadmap and device runbook

Plan of record for moving the displacement and strain solvers onto the GPU
via OpenCL, **without any loss of accuracy**, and the procedure for
verifying that on real hardware.

Companion to [DETERMINISM.md](DETERMINISM.md): that document explains *why*
the CPU numerics are pinned and what the contract is. This one covers what
is being built on top, and how to check it.

**Scope:** desktop and laptop GPUs. Android is deliberately out of scope —
`SEMPER_OPENCL` is forced OFF when `SEMPER_ANDROID` is on, because OpenCL
is not part of the NDK (it requires dlopening a vendor `libOpenCL.so`) and
Mali/Adreno parts largely do not expose `cl_khr_fp64`, which the strain
kernel needs. Android keeps the CPU path unchanged. Revisit only with a
real device matrix.

---

## 1. Roadmap

### The accuracy contract

The requirement is bit-exactness against a **defined CPU reference**, not
merely "close enough". Concretely:

- The reference is `semper_math` built with `-fno-fast-math
  -ffp-contract=off`, using the canonical 4-accumulator reduction and the
  shared kernels in `include/semper/kernels/canonical_math.h`.
- That header is written in the common subset of C99 and OpenCL C and
  compiles **as the same text** into both the host library and the device
  kernels. There is no second copy to drift.
- Device programs are built with `-cl-std=CL1.2
  -cl-fp32-correctly-rounded-divide-sqrt`, and **never** with
  `-cl-fast-relaxed-math` or `-cl-mad-enable`. Every `.cl` opens with
  `#pragma OPENCL FP_CONTRACT OFF`.
- Only `+ - * /` and `sqrt` appear in mirrored kernels. No transcendentals,
  no `native_*`, no `fma()` — the reference rounds twice where an FMA would
  round once.

A device that does not report correctly-rounded fp32 divide/sqrt **cannot**
run the bit-exact ICGN path and is refused for that stage.

### Phases

| # | Scope | Status |
|---|---|---|
| **0** | Make the CPU reference reproducible. Strict FP, canonical reduction order, canonical linear algebra replacing Eigen in the mirrored paths, deterministic Path B. **No GPU code.** | **Complete** |
| **1** | OpenCL runtime + build plumbing: `SEMPER_OPENCL`, dlopen loader, device capability gate, kernel embedding. **No compute kernels.** | **Complete** |
| **2** | Strain VSG on GPU. One work-item per grid point, fp64. Currently the only fully serial numerical stage. | **Complete** |
| **3** | Hessian pre-pass on GPU. One work-item per grid point. | **Complete** |
| **4** | Path A ICGN on GPU. **One work-item per subset**, so the reduction keeps the canonical order rather than becoming a cross-lane tree. | **Landed, off by default** — bit-exact, but slower than the threaded CPU on the measured host; gated behind `kGpuIcgnPathAEnabled = false` |
| **5** | Path B wavefront on GPU. Reuses the Phase 4 kernel; one launch per round. Only possible because Phase 0 made Path B round-based. | **Evaluated — stays on CPU.** The one round big enough to pay for a launch is 34-47% Nelder-Mead rescues, which the device path cannot take |
| **6** | Image prep (gradients, optional blur). AKAZE stays on CPU — randomized RANSAC, poor return. | **Kernel written and bit-exact; not wired in.** The stencil is bus-bound at 12 bytes per pixel, so the device loses at every image size with no crossover |

### Phase 0 results (measured)

| Property | Before | After |
|---|---|---|
| Golden corpus, SSE2 vs AVX2 | differed (~6900 bytes) | **bit-identical** |
| Full-field golden, SSE2 vs AVX2 | no such test existed | **bit-identical** |
| Run-to-run reproducibility | **0 / 15** runs | **15 / 15** |
| Host tests | 83 | **102** |
| Throughput (median / max) | 3212 / 3232 | 3205 / 3273 |
| DICe field agreement | rms 0.0006 px | rms 0.0006 px, 230/230 |

Three independent causes of cross-ISA divergence were found and fixed:
`-ffast-math`, a SIMD reduction order that followed the hardware vector
width, and Eigen's per-ISA packing of fixed-size products. Path B was
additionally non-deterministic run-to-run because its worker threads raced
a shared priority queue. See [DETERMINISM.md](DETERMINISM.md).

### Phase 1 results (measured)

The backend compiles and probes; no DIC work runs on the device yet. What
was verified:

| Check | Result |
|---|---|
| `SEMPER_OPENCL=OFF` (default) | 102 tests pass; `nm -D` finds **zero** OpenCL symbols |
| `SEMPER_OPENCL=ON`, no ICD installed | 109 tests pass; probe reports *"loader present but reports zero platforms"* and every stage falls back to CPU |
| `SEMPER_OPENCL=ON`, POCL 5.0 CPU device | 109 tests pass; kernel builds on the device; `fp64=1 exact_fp32=1` |
| **Device reproduces the canonical reduction** | **Exactly**, at n = 0, 1, 3, 4, 5, 8 and 729 — including the scalar tail (729 = 4·182 + 1) |
| Throughput, backend ON vs OFF back-to-back | 2900 vs 2910 median — 0.3%, within noise |
| Cross-ABI determinism | Subset corpus and full-field golden both still byte-identical |

The reduction result is the one that matters for everything after this. The
POCL device vectorizes internally on `skylake-avx512` and still reproduces
the host's pinned 4-lane summation order bit-for-bit, which is the premise
Phases 2-5 are built on.

> The throughput numbers above are lower than Phase 0's 3205/3273 because the
> host was busier, not because anything regressed — the *unchanged*
> `SEMPER_OPENCL=OFF` build measured 2910 in the same session. This is
> precisely why §4a says to compare ON against OFF back-to-back rather than
> against a number recorded on another day.

### Phase 2 results (measured)

The VSG plane fit runs on the device, bit-identical to the CPU. Measured on
a Windows 11 host, MinGW-w64 GCC (UCRT), Release, against an NVIDIA GeForce
RTX 3060 Laptop GPU on the CUDA ICD (OpenCL 3.0, `fp64=1 exact_fp32=1`).

| Check | Result |
|---|---|
| `SEMPER_OPENCL=OFF` | 71 tests pass; binary contains **zero** OpenCL loader references |
| `SEMPER_OPENCL=ON`, RTX 3060 | 84 tests pass (`ClParity` 7/7, `ClRuntime` 7/7) |
| **CPU/GPU parity, values** | **Exact** over all 723 points of four grid geometries, 419 of them solved — `==`, not a tolerance |
| **CPU/GPU parity, rejections** | **Exact** — 166 of 195 points refused identically by both paths (edge clipping, sub-90% fill, rank deficiency) |
| Run-to-run reproducibility, device | Identical bytes across repeat dispatches |
| Cross-ABI determinism | Subset corpus still byte-identical, `x86-64` vs `x86-64-v3` |
| Eigen removal, before → after | **Byte-identical** — see below |

**Throughput does not have a single number, and that is the finding.**
Dispatch costs a roughly fixed 0.45 ms in buffer traffic and launch latency,
so the device loses on small grids and wins on large ones:

| Grid points | CPU | GPU | |
|---|---|---|---|
| 2 209 (47×47) | 0.324 ms | 0.484 ms | 0.67× — **CPU wins** |
| 3 600 (60×60) | 0.551 ms | 0.524 ms | 1.05× — break-even |
| 6 084 (78×78) | 0.921 ms | 0.508 ms | 1.81× |
| 9 216 (96×96) | 1.357 ms | 0.608 ms | 2.23× |
| 36 864 (192×192) | 5.727 ms | 1.076 ms | **5.32×** |

Real grids straddle that break-even: the DICe fixtures are a few hundred
points, a full frame at a small step is six figures. So the caller
(`src/pipeline/full_field_solver_stats.cpp`) applies a **3600-point
threshold** and stays on the CPU below it. The threshold is the measured
break-even, not a guess; `dic_tests Perf` re-derives it, and the constant
moves if the kernel or the transfers change. `compute_vsg_strain_gpu` itself
stays policy-free so the parity tests can drive it at any size.

Making the six buffer transfers non-blocking around a single `clFinish` --
rather than six separate host round-trips -- moved the break-even down from
about 6400 points to 3600 and cut large-grid time by roughly 15%.

**On dropping Eigen (step one of the phase).** The kernel had to mirror a
CPU reference that does not use Eigen, since Eigen packs fixed-size products
per ISA. Rewriting `compute_vsg_strain` onto `semper_inv3x3` /
`semper_mat3_vec3` / `semper_mat3_l1_norm` turned out to produce a
**byte-identical golden corpus** on this host, and the Eigen version was
*also* `x86-64`/`x86-64-v3` stable here. So no golden recapture was needed
and this is not evidence of a bug that was fixed. It is evidence that the
hazard did not happen to manifest in this ISA pair -- which is exactly why
the reference is pinned rather than left to a library's codegen.

### Phase 3 results (measured)

The static Hessian pre-pass runs on the device, bit-identical to the CPU.
Same host as Phase 2: Windows 11, MinGW-w64 GCC (UCRT), Release, NVIDIA
GeForce RTX 3060 Laptop GPU on the CUDA ICD. Unlike the strain kernel this
one is fp32 from end to end, so it is gated on **`exact_fp32`, not `fp64`**,
and it builds on a device with no `cl_khr_fp64` at all.

| Check | Result |
|---|---|
| `SEMPER_OPENCL=OFF` | 72 tests run, 70 pass |
| `SEMPER_OPENCL=ON`, RTX 3060 | 93 tests run, 91 pass (`ClParity` 15/15, `ClRuntime` 7/7) |
| `SEMPER_OPENCL=ON` + OpenCV, RTX 3060 | 127 tests run, 124 pass — the same three fail with the device disabled |
| Linux, GCC 11.4, OpenCV, `OFF` → `ON` | 106 → 127 tests; same single pre-existing failure in both, and `ClPipelineParity` skips itself with no ICD |
| **CPU/GPU parity, values** | **Exact** over all 547 points of four grid geometries, 442 of them accepted — `==`, not a tolerance, on `H`, `H_inv`, `mean_intensity` and `std_dev` |
| **CPU/GPU parity, rejections** | **Exact** — 53 of 380 points refused identically on a grid stamped with a ghost wall and a flat (rank-deficient) patch, and 88 of 100 on a grid deliberately run off the image edge |
| Unrequested pool slots | Untouched, byte for byte — the pre-pass skips points the solver has already solved |
| Run-to-run reproducibility, device | Identical bytes across repeat dispatches |
| Cross-ABI determinism | Subset corpus byte-identical, `x86-64` vs `x86-64-v3` |
| CPU fallback unchanged | Full-field output identical between the `OFF` and `ON` Linux builds apart from timing lines; Path A/B split still 420/58 |

The two `SEMPER_OPENCL=OFF` failures on Windows are `GoldenCorpus`, and they
are the known MinGW-vs-glibc `libm` gap on 13 of 580 points — the fixture is
pinned to Linux GCC Release (see [DETERMINISM.md](DETERMINISM.md)). The one
Linux failure is `FullFieldGolden.CaptureOrCompare`, which fails identically
on the **unmodified** baseline in that environment because the vendored
OpenCV 4.13.0 seeds AKAZE differently from CI's packaged build.

**Throughput.** Measured at a fixed 1024×1024 reference image, subset 21, ten
repetitions, sweeping the step to vary the grid:

| Grid points | CPU (serial) | GPU | |
|---|---|---|---|
| 2 500 (step 19) | 11.27 ms | 3.25 ms | 3.5× |
| 5 329 (step 13) | 23.99 ms | 3.61 ms | 6.6× |
| 11 236 (step 9) | 50.41 ms | 4.13 ms | 12.2× |
| 25 600 (step 6) | 115.06 ms | 6.28 ms | 18.3× |
| 57 600 (step 4) | 257.95 ms | 10.04 ms | **25.7×** |

**Read that CPU column carefully: it is serial.** The test binary does not
link OpenMP, while the pipeline pre-pass is an `#pragma omp parallel for` over
`safe_cores`. The number the caller actually races against is this column
divided by the core count.

The other half of the cost is fixed, and it scales with the **image**, not the
grid — the three float planes of the reference image cross the bus on every
call:

| Reference image | Dispatch floor | Per megapixel |
|---|---|---|
| 512×512 | 1.14 ms | 4.35 ms/MP |
| 1024×1024 | 2.65 ms | 2.52 ms/MP |
| 2048×2048 | 7.54 ms | 1.80 ms/MP |

That is about 0.6 ms constant plus 2 ms per megapixel. So the shape that loses
on the device is a **large image with a coarse grid**, which a flat point-count
threshold would miss entirely. `src/pipeline/full_field_solver.cpp` therefore
gates on points *per megapixel of reference image*: 12 000, roughly four times
the 6-core crossover, plus a 4 000-point floor. `compute_hessian_pool_gpu`
itself stays policy-free so the parity tests can drive tiny grids.

**On dropping Eigen (step one of the phase, commit `e9f6e7a`).** Unlike
Phase 2, this one moved the golden corpus. Eigen's `PartialPivLU` inverse and
`semper_inv6x6`'s Gauss-Jordan are both correct and differ in the last few
digits, which shifts the ICGN iterate path: 394 of 580 points (4×4) and 399 of
580 (6×6) moved by roughly 1e-5, with **zero** status mismatches. The fixture
was recaptured deliberately, on the Linux GCC Release platform it is pinned
to, and the recapture is recorded in [DETERMINISM.md](DETERMINISM.md).
`full_field_golden.bin` was verified unchanged and was not recaptured.

**End to end, on the device.** The dispatch functions are policy-free and the
size thresholds live in their callers, so `ClParity` — which drives the
dispatch directly — never executes the wired caller. `ClPipelineParity`
(`tests/integration/test_full_field_gpu_prepass.cpp`) closes that: it runs
`run_full_field` three times on one 512x512 pair at step 7, a geometry chosen
to clear both thresholds (5 329 grid points, over the 4 000-point floor and
over 12 000 x 0.262 MP), once with `SEMPER_OPENCL_DISABLE=1` and twice with the
RTX 3060 engaged. On the RTX 3060, MinGW GCC 16.1, OpenCV 4.13.0:

| | Result |
|---|---|
| Packed output, CPU vs GPU | **0 float mismatches** of 42 632 — `==`, all 8 slots of all 5 329 points |
| Valid points | 3 265 on both |
| Path A / Path B split | 3 470 / 489 on both |
| Cold vs warm device solve | Identical bytes |

Running the whole suite on that host needs the OpenCV the tests link against
to come from the same toolchain, so it was built for MinGW from the vendored
submodule; that build is a local prerequisite, not a repo change.

**The end-to-end timing is the honest half of this.** At that geometry the
pre-pass costs **4.05 ms on the CPU and 2.99 ms on the device** — 1.35x, far
from the 6.6x the same point count shows in the throughput table above,
because there the CPU column is serial and here it is OpenMP across six cores.
The first device solve in a process costs **62.14 ms** instead, all of it the
one-time backend bring-up (`dlopen`, platform and device probe,
`clBuildProgram` for every kernel) landing inside that frame's pre-pass
timing. A single-frame solve therefore loses; a session amortises it over the
first frame. That cost is not new to this phase — Phase 2 already pays it —
but it had not been measured until now.

Re-deriving the threshold from these numbers rather than the serial ones: the
CPU costs 0.76 us/point (6 cores) and the warm device 2.33 ms fixed plus
0.123 us/point at 512x512, so the crossover is about 3 700 points. The shipped
4 000-point floor clears it, with about 9% of margin — thin, and it is the
floor, not the per-megapixel term, that binds below 0.33 MP. Above that the
12 000 pts/MP term binds and is roughly 3x conservative. The exposure is a
sub-millisecond loss on a small image with a dense grid, which is why the
floor is left as it is rather than tuned to the edge.

**Not run, and not claimed.** Still deferred from Phase 3a: the determinism
job's Linux cross-`-march` compare (subsequently **run in Phase 4** — see
below), and a `FullFieldGolden` run on CI's OpenCV. On the Windows host the full suite is 127 tests, 124 passing; the
three failures are `FullFieldGolden.CaptureOrCompare` and the two
`GoldenCorpus` cases, and **the same three fail from the same binary with
`SEMPER_OPENCL_DISABLE=1`** — they are the vendored-OpenCV AKAZE difference
and the MinGW `libm` gap described above, not the GPU path.

### Phase 4 results (measured)

**Phase 4 clears two of its three gates and fails the third.** The ICGN
kernel is bit-exact against the CPU, and it is slower than the CPU here, so
it ships behind `kGpuIcgnPathAEnabled = false` in
`src/pipeline/full_field_path_a.cpp`. The kernel, the dispatch and the parity
tests are in the tree because Phase 5 reuses them and because the parity
result is worth keeping; what is not in the tree is a claim that this is
faster.

Same host as Phases 2 and 3: Windows 11, MinGW-w64 GCC (UCRT), Release,
NVIDIA GeForce RTX 3060 Laptop GPU on the CUDA ICD. fp32 end to end, so like
Phase 3 it is gated on **`exact_fp32`**, not `fp64`.

| Check | Result |
|---|---|
| `SEMPER_OPENCL=ON` + OpenCV, RTX 3060 | 139 tests run, 136 pass (`ClParity` 24/24, `ClRuntime` 7/7) |
| `SEMPER_OPENCL=OFF` + OpenCV, Windows | 108 tests run, 105 pass — **the same three fail**, from a binary with no GPU code in it |
| Linux, GCC 11.4, no OpenCV, `OFF` → `ON` | 74 → 104 tests, **all passing in both**; the device cases skip themselves with no ICD |
| `SEMPER_OPENCL=OFF`, `nm -D \| grep -i opencl` | Silent, and no `libOpenCL` string in the binary |
| **CPU/GPU parity, values** | **Exact** over **3 235 points** across seven geometries — `==`, not a tolerance, on all six warp parameters, the correlation score, the iteration count and the invalid-pixel count |
| **CPU/GPU parity, rejections** | **Exact** — 12 of 132 points refused identically on a grid deliberately run off the image edge; 240 of 400 taken by the partial path identically on a ghost-wall grid |
| **CPU/GPU parity, iteration counts** | **Identical per point.** An equal answer reached in a different number of iterations would mean the convergence test diverged; it does not |
| Both interpolators | Bicubic and Keys 6×6 both exact |
| Multi-tile launch | A 2 209-point batch split across scratch-budget tiles: 0 mismatches, so the tiling is not observable in the answer |
| **Throughput vs the threaded CPU** | **FAILED — 1.12× to 1.44× slower on 20 threads, noise-level on 8 and 4** |
| Cross-ABI determinism | Subset corpus byte-identical, `x86-64` vs `x86-64-v3`, on Linux GCC — see below |

The seven parity geometries in detail, all `==`:

| Geometry | Compared | Converged | Refused | Mismatch |
|---|---|---|---|---|
| 12×10 step 9, subset 21 | 120 | 120 | 0 | **0** |
| 14×13 step 7, subset 31 | 182 | 182 | 0 | **0** |
| 9×8 step 11, subset 15 | 72 | 72 | 0 | **0** |
| 12×12 step 13, subset 25 (edge overrun) | 132 | 130 | 12 | **0** |
| 12×10 step 9, Keys 6×6 | 120 | 120 | 0 | **0** |
| Ghost wall, 20×20 | 400 | 160 (240 partial) | 0 | **0** |
| Tiled, 2 209 points | 2 209 | — | 0 | **0** |

**End to end, on the device.** With `kGpuIcgnPathAEnabled` temporarily
flipped to `true`, `ClPipelineParity` runs the real `run_full_field` on one
512×512 pair at step 7 — 5 329 grid points — once with
`SEMPER_OPENCL_DISABLE=1` and once with the RTX 3060 solving Path A:

| | Result |
|---|---|
| Packed output, CPU vs GPU | **0 float mismatches** of 42 632 — `==`, all 8 slots of all 5 329 points |
| Valid points | 3 265 on both |
| Path A / Path B split | 3 470 / 489 on both |
| Simplex rescues | 262 on both |
| Mean ICGN iterations | 9.398162 on both |

That the rescue tally and the mean iteration count are identical is the
substantive part. The device answer is accepted **only** when it would not
have tripped the CPU rescue test — `status == 0 && score <=
kCorrSimplexTrigger`, the condition `optimization_engine.cpp:54` uses — and
every other point, including every `kIcgnHostRequired` point, falls through
to the untouched CPU path. The substitution is therefore equivalent by
construction rather than by approximation, and these two counters are what
would expose it if it were not.

**Throughput, against a serial CPU loop.** 1024×1024 reference, subset 21,
best of three, sweeping the step:

| Grid points | CPU (serial) | GPU | | GPU µs/pt |
|---|---|---|---|---|
| 2 500 (step 19) | 180.16 ms | 32.72 ms | 4.8× | 13.09 |
| 5 329 (step 13) | 385.01 ms | 55.65 ms | 6.9× | 8.93 |
| 11 236 (step 9) | 811.78 ms\* | 67.75 ms | 10.2× | 6.03 |
| 25 600 (step 6) | 1 849.56 ms\* | 160.66 ms | 11.5× | 5.36 |
| 57 600 (step 4) | 4 161.51 ms\* | 332.63 ms | **12.5×** | 4.94 |

\* extrapolated at the measured flat serial rate; that rate varies by under
2% across the sweep, so the two smallest grids are timed and the rest are
not, to keep the suite from spending half a minute on solves that tell us
nothing new.

**That table is not the gate, and it is why the gate exists.** The CPU column
is one core. Path A runs on `hardware_concurrency()` threads. The number that
decides whether to engage the device is the whole-solve wall clock with and
without it, which `Perf.PathAPipelineThroughputCpuVsGpu` measures on the real
`run_full_field`, best of three after a warm-up, with every other stage —
image prep, AKAZE, RANSAC, Delaunay, Path B, strain — running identically in
both arms:

| Path A points | 20 threads | 8 threads | 4 threads |
|---|---|---|---|
| 3 470 | 1.27× **CPU** | 1.09× **CPU** | 1.07× **CPU** |
| 10 562 | 1.44× **CPU** | 1.05× **CPU** | 1.01× **CPU** |
| 17 110 | 1.21× **CPU** | 1.00× | 1.06× GPU |
| 32 645 | 1.12× **CPU** | 1.02× GPU | 1.04× GPU |

The thread counts below 20 were produced by masking the process affinity
(`Start-Process -PassThru`, then `$p.ProcessorAffinity`), because `safe_cores`
comes from `std::thread::hardware_concurrency()` and there is no knob for it.

**Why, measured rather than guessed.** Capping the iteration count separates
the fixed per-point setup from the per-iteration work. At 11 236 points:

| Iteration cap | Time | µs/pt |
|---|---|---|
| 1 | 10.2 ms | 0.91 |
| 2 | 11.8 ms | 1.05 |
| 4 | 14.5 ms | 1.29 |
| 8 | 20.4 ms | 1.82 |
| 50 (the real cap) | **68.0 ms** | 6.05 |

The mean is **~9.4 iterations per point**, which by the table above should
cost about 22 ms. It costs 68. One work-item per subset — the constraint that
keeps the reduction bit-exact, and which is not negotiable — means a warp
runs until its *slowest* lane converges, and a full-field solve puts roughly
137 points that time out at 50 iterations among 3 470. At 32 lanes per warp
nearly every warp contains one, so the launch effectively pays 50 iterations
for every point. Memory bandwidth and the interpolator were ruled out
arithmetically: the per-point scratch is 8n floats and `canonical_interp.inc`
contains no divisions.

The fixed cost, isolated the same way as in Phase 3 — it scales with the
image, not the grid:

| Reference image | Dispatch floor |
|---|---|
| 512×512 | 2.12 ms |
| 1024×1024 | 4.87 ms |
| 2048×2048 | 10.84 ms |

**The remedy, and why it is not in this commit.** It is the one the roadmap
already names: **round-based launches over a compacted active-point list**,
not a different reduction shape. Stage the reference planes once, run a
bounded number of iterations per launch, read back which points are still
running, relaunch over just those. Resuming from the stored warp matrix is
bit-identical — every per-iteration input is either recomputed or carried in
full precision — so it costs no accuracy. It is a real restructure of the
kernel, it was not attempted here, and shipping a 1.1×-to-1.4× regression
while it is pending would be worse than shipping the flag off. The two
thresholds in the caller (`kGpuIcgnPointsPerMegapixel = 16000`,
`kGpuIcgnMinPoints = 16000`) are where the 4- and 8-thread measurements put
the crossover **today**; whoever lands the compaction must re-measure them,
because the shape of the curve changes.

**Newly run this phase, previously deferred.** The determinism job Linux
cross-`-march` compare, deferred since Phase 3a, was executed: two Release
builds under WSL Ubuntu 22.04 / GCC 11.4, `-march=x86-64` (SSE2 baseline)
against `-march=x86-64-v3` (AVX2/FMA), golden corpus captured from each. Both
`cmp`s silent — `corpus.bin.bicubic` and `corpus.bin.keys6x6` byte-identical
across the two instruction sets. The third `cmp` in §4b, `ff_*.bin`, needs
`FullFieldGolden`, which needs OpenCV; WSL has none and the vendored
submodule was not built there, so **that one `cmp` remains deferred**.

The same Linux run also settles the standing Windows failures: `GoldenCorpus`
passes there with **0 value mismatches on all 580 points, both
interpolators**, confirming the 13-point Windows discrepancy is the
MinGW-vs-glibc `libm` gap the pinning of the fixture already describes, and
not a defect.

**Not run, and not claimed.**

- The Linux/POCL parity run. WSL has no OpenCL ICD, installing
  `pocl-opencl-icd` needs `sudo` and the session is non-interactive, so
  `ClParity` and `ClPipelineParity` **skip themselves** there. Every parity
  number above is from the RTX 3060 on Windows and from nowhere else.
- The `ff_*.bin` leg of the cross-`-march` compare, and a `FullFieldGolden`
  run on CI's OpenCV — both still blocked on an OpenCV build, as in Phase 3.
- On Windows the suite is 139 tests, 136 passing; the three failures are
  `FullFieldGolden.CaptureOrCompare` and the two `GoldenCorpus` cases, and
  **the same three fail from the `SEMPER_OPENCL=OFF` binary** — vendored-OpenCV
  AKAZE and the MinGW `libm` gap, not the GPU path.

### Phase 5 results (measured)

**Path B stays on the CPU.** No kernel was written, and the reason is a
measurement rather than a difficulty: the one round large enough to pay for a
kernel launch is expensive for precisely the reason the device cannot help
with. This is the outcome the roadmap allowed for, and it is recorded here as
a result, not deferred.

The evaluation is cheap to re-run on other hardware, and it should be, because
the conclusion is hardware-dependent: `SEMPER_PATHB_ROUNDS=1` prints the round
structure, and `Perf.IcgnPathAThroughputCpuVsGpu` prints the small-batch launch
cost the round structure has to beat.

**What the wavefront actually looks like.** Path B is round-serial by
construction — round *k*+1's guesses come from round *k*'s answers, so the
rounds cannot be merged or reordered without giving up the determinism Phase 0
bought. Measured on the RTX 3060 host, 20 threads, at the four geometries
`Perf.PathAPipelineThroughputCpuVsGpu` uses:

| Geometry | Path B pts | Rounds | Round 0 | Round 0 share of Path B time |
|---|---|---|---|---|
| 512×512 step 7 | 622 | 11 | 347 pts, 17.7 ms | 72% |
| 1024×1024 step 9 | 881 | 15 | 599 pts, 33.2 ms | 63% |
| 1024×1024 step 7 | 1 921 | 21 | 950 pts, 49.2 ms | 63% |
| 1024×1024 step 5 | 4 533 | 29 | 1 532 pts, 83.1 ms | 53% |

Every geometry is one big round and a long tail. At step 5 the tail is 28
rounds averaging 107 points, ending in rounds of 4, 3, 2 and 1.

**Why round 0 is expensive, and why that rules the device out.** It is the
boundary between Path A's solved region and the unsolved interior, so it
carries almost all of the hard points:

| Geometry | Round 0 rescue rate | iters/pt | Round 1 rescue rate | Round 1 iters/pt |
|---|---|---|---|---|
| 512×512 step 7 | **34.3%** | 17.5 | 4.5% | 6.0 |
| 1024×1024 step 9 | **40.2%** | 18.8 | 5.6% | 6.9 |
| 1024×1024 step 7 | **41.2%** | 19.7 | 3.8% | 6.1 |
| 1024×1024 step 5 | **47.3%** | 22.0 | 6.1% | 6.5 |

A third to a half of round 0 needs the Nelder-Mead rescue — and the Phase 4
device path accepts a device answer **only** where it would not have tripped
that rescue test, so those points run on the CPU whether or not a kernel is
launched. What is left for the device is the cheap majority.

The arithmetic, at 1024×1024 step 5. Round 1 is 691 points at 6.1% rescues and
16.3 µs/pt, which is close to a pure-ICGN rate; round 0's 1 532 points at that
rate would be about **25 ms** of its measured **83 ms**, leaving ~58 ms in the
rescue tail. The measured device cost of a 1 532-point launch is **~24 ms**
(interpolated on the batch sweep below). So the device would replace 25 ms of
CPU work with a 24 ms launch, leave the other 58 ms exactly where it was, and
add a host round-trip per round. There is no version of that which wins.

**Launch cost at the sizes that matter.** The Phase 4 sweep starts at 2 500
points because that is Path A's scale; Path B rounds are two orders of
magnitude smaller, so `Perf.IcgnPathAThroughputCpuVsGpu` now also sweeps the
small end (1024×1024 reference, subset 21, best of three):

| Batch | Time | µs/pt |
|---|---|---|
| 8 | 10.53 ms | 1316 |
| 32 | 14.38 ms | 449 |
| 64 | 14.69 ms | 229 |
| 128 | 15.10 ms | 118 |
| 256 | 17.36 ms | 67.8 |
| 512 | 18.80 ms | 36.7 |
| 1 024 | 23.14 ms | 22.6 |
| 2 048 | 24.84 ms | 12.1 |

The floor is ~14.5 ms for anything under a few hundred points. Every Path B
round after the first is under 700 points and already runs at 10–17 µs/pt on
the CPU — round 2 at step 5 is 672 points in 7.0 ms, against a ~20 ms launch.
A launch-per-round design projects to **2.5×–5.3× slower** than the CPU across
the four geometries; restricting it to rounds of 256 or more still loses,
because those rounds are either round 0 (rescue-bound) or already cheap.

**What would change this.** Not a better kernel — the same kernel is already
bit-exact and already fails Phase 4's throughput gate on larger, friendlier
batches. It would take a device path for the Nelder-Mead rescue itself, which
is a different kernel with a different reduction shape and its own
bit-exactness problem, or a machine where the launch floor is far below
14.5 ms. Both are out of scope here, and neither is implied by the roadmap.

**Not run, and not claimed.**

- No Path B kernel exists, so there is no Path B parity result. `ClParity` and
  `ClPipelineParity` are unchanged and still cover Phases 2–4 only.
- The conclusion above is a **projection** from two sets of measured numbers —
  the round structure and the launch-cost curve — not an end-to-end wall-clock
  measurement of a Path B device path, because no such path was built. The
  projected margin is 2.5× or worse at every geometry measured, which is why
  it was not built.
- `FullFieldGolden.CaptureOrCompare` with the GPU forced on, which §5c of the
  roadmap asks for from Phase 5, is moot for Path B specifically: nothing in
  Path B changed. It still fails on this Windows host for the pre-existing
  vendored-OpenCV AKAZE reason, identically with `SEMPER_OPENCL_DISABLE=1`.
  `FullFieldGolden.RepeatSolve_IsDeterministic` passes.

### Phase 6 results (measured)

**The kernel exists, is bit-exact, and the engine does not call it.** Unlike
Phase 5 there was no doubt about the arithmetic — the stencil is three adds
and a divide — and none about whether it could be written. The throughput
gate failed anyway, for a reason no kernel change can address, and the
measurement below is the reason the stage is not wired into the pipeline.

**What was built.** `src/gpu/kernels/image_grad.cl`, one work-item per pixel,
and `compute_image_gradients_gpu` in `src/gpu/image_prep_dispatch.cpp`. It
fills `Image::grad_x` and `Image::grad_y` with exactly what
`Image::prepare_data(false)` writes — the border band included, which is
compared too. It does **not** apply the DICe 7-tap blur: `prepare_data`'s blur
flag is false at every call site in the engine and true only in
`tests/unit/test_image.cpp`, so a device blur would be a kernel with no
production caller. AKAZE seeding was not touched and is still ruled out.

**The CPU reference moved first, and was gated on its own.** The gradient
expression left `src/math/image_processor.cpp` and became
`SEMPER_CANON_DERIV5` in `include/semper/kernels/canonical_math.h`, so the
host and the kernel read the same parenthesisation rather than two copies of
it. Before any kernel was written, the refactored host path was compared
against the literal pre-refactor expression over four geometries with the
`-10.0f` ghost-wall and `-5.0f` void sentinels stirred in: **2 630 210
floats, 0 bit-level mismatches**.

**Parity.** `ClParity` gained seven cases (`tests/unit/test_cl_image_grad.cpp`).
The device matched the host on every float of both planes at 64×64, 128×96,
96×128, 257×129 and 131×67 — the last two chosen so that an odd dimension and
a prime pixel count exercise a launch that does not divide evenly into
work-groups, and the non-square ones so that a transposed row/column stride
cannot pass. Images at or below the 2-pixel border in one or both dimensions
(1×1 through 64×3) come out entirely zero on both paths. Run-to-run
reproducible.

**Throughput — the gate that failed.** `Perf.ImageGradThroughputCpuVsGpu`,
RTX 3060 Laptop, best of 20, both columns carrying the whole cost the caller
pays:

| Image | CPU | GPU | Ratio | Effective bandwidth |
|---|---|---|---|---|
| 256×256 (0.07 MP) | 0.098 ms | 0.521 ms | **0.19×** | 1.51 GB/s |
| 512×512 (0.26 MP) | 0.430 ms | 1.224 ms | **0.35×** | 2.57 GB/s |
| 1024×1024 (1.05 MP) | 1.747 ms | 3.805 ms | **0.46×** | 3.31 GB/s |
| 2048×2048 (4.19 MP) | 7.393 ms | 12.945 ms | **0.57×** | 3.89 GB/s |
| 4096×4096 (16.78 MP) | 30.568 ms | 51.106 ms | **0.60×** | 3.94 GB/s |

The device loses at every size, and — unlike Phases 2 and 3, where a small
grid lost and a large one won — **there is no crossover**. The ratio
asymptotes at 0.60× and the bandwidth column says why: from 2048² to 4096²
the pixel count quadruples and the GPU time goes up 3.95×, while effective
bandwidth flattens at 3.94 GB/s. The stage is pure bus traffic. Twelve bytes
cross per pixel — four up, eight back — against four float operations, an
arithmetic intensity of 0.33 flop/byte, so essentially none of the measured
time is the kernel. Making the kernel faster changes nothing; there is
nothing else on the curve.

Note that this ratio, unlike the Phase 3 and Phase 4 sweeps, is **not** an
upper bound. Those compare against a serial CPU column while the engine runs
the stage on all cores. `Image::prepare_data` carries no OpenMP pragma — the
gradient loops really are serial in the engine — so 0.19×–0.60× is what the
pipeline would actually see.

**What would change this, and what would not.** Not a better kernel, and not
a different launch shape. Two things would:

- **A device with no bus to cross.** On an integrated GPU or a
  unified-memory part the 12 bytes per pixel never leave host memory, and a
  parallel stencil against a serial CPU loop is a straightforward win. The
  sweep above is the test; re-run it before concluding anything about this
  stage on such a host.
- **Device residency, which is where the real number is.** The reference
  image's gradients are computed once per reference set, but
  `compute_hessian_pool_gpu` re-uploads `intensities`, `grad_x` and `grad_y`
  on **every frame** — three float planes, and the measured Phase 3 dispatch
  floor of ~0.6 ms plus ~2 ms per megapixel is almost entirely those planes.
  Gradients produced on the device and left there would cut that upload from
  three planes to one, saving roughly two thirds of a fixed cost that is
  ~5 ms per frame at 1024². That is a larger number than this whole stage is
  worth on its own, and it is the sense in which the roadmap's "keeping the
  image resident on the device across frames matters more than the kernel
  itself" is correct.

  It was **not built**, and the obstacles are structural rather than
  numerical. `detail::ProgramScope` holds the runtime's global mutex for its
  whole lifetime and the mutex is not recursive, so one stage's dispatch
  cannot invoke another stage's program; a resident buffer also has no owner
  today, since `ReferenceCache` lives in the public `semper/pipeline.hpp`
  and cannot hold a `cl_mem`. Both are real work with real risk to buffer
  lifetime, and neither belongs in a commit whose subject is a stencil.

**Not run, and not claimed.**

- The stage is **not wired into the pipeline**. There is no caller-side
  constant to flip, unlike Phase 4's `kGpuIcgnPathAEnabled`: Phase 4's flag
  guards an integration that a less divergent workload could justify turning
  on, whereas here no geometry on the curve would justify it. Wiring
  `reference_cache.cpp` and `full_field_solver.cpp` to a path that loses
  everywhere would be dead code pretending to be a policy.
- Because nothing in the pipeline calls it, the full-field golden is
  unchanged **by construction** rather than by measurement of a device path:
  the only production code this phase touched is the host gradient loop, and
  that was gated bit-exactly before the kernel was written. `ClPipelineParity`
  still covers Phases 2–4 only.
- No POCL/Linux parity run for this kernel, for the same reason as Phases 3–5:
  no OpenCL ICD is installable in this WSL environment without an interactive
  `sudo`. The kernel's OFF-build behaviour (`ClParity.ImageGradKernelAbsentWithoutOpenCLBuild`)
  is covered.

### Per-phase gate

Every phase must clear all three before the next begins.

| Gate | Rule |
|---|---|
| **Unit tests** | New kernels and code paths ship with tests in the same commit |
| **Performance** | Throughput ≥ the previous phase's measured number **on the same machine** |
| **Bit-exactness** | Exact float equality vs the CPU canonical path — `==`, never a tolerance |

---

## 2. Bring-up

### Prerequisites

| Requirement | Check | Notes |
|---|---|---|
| OpenCL ICD | `ls /etc/OpenCL/vendors/` | Vendor driver, or `pocl-opencl-icd` for a conformant CPU device |
| `clinfo` | `clinfo --version` | Only for inspection; the engine does not use it |
| OpenCV **≥ 4.8** | `pkg-config --modversion opencv4` | See [tests/README.md](../tests/README.md) — 4.6 does not compile |
| Submodules | `git submodule status` | Both must be checked out, no leading `-` |

The engine itself needs no OpenCL SDK, headers, or link-time library — it
declares the entry points it uses and `dlopen`s the loader at runtime. A
`SEMPER_OPENCL=ON` build runs correctly on a machine with no GPU at all.

### Does this device qualify?

```bash
clinfo | grep -Ei 'Device Name|Device Version|Device OpenCL C Version|Correctly-rounded|cl_khr_fp64'
```

Capabilities are evaluated **per stage**, not as one pass/fail — a device
may legitimately run one and not the other:

| Field | Needed for | If absent |
|---|---|---|
| `Device Version` ≥ OpenCL 1.2 | everything | Whole backend falls back to CPU |
| `cl_khr_fp64` in extensions | strain (Phase 2) | Strain stays on CPU |
| `Correctly-rounded divide/sqrt` in single-precision FP config | ICGN and the Hessian pre-pass (Phases 3–5) | Those stages stay on CPU — they **cannot** be bit-exact without it |

`clinfo` reporting no platforms means no ICD is registered. Note that
`libOpenCL.so.1` being present proves nothing: it is only the loader, and
it resolves happily with zero drivers behind it.

---

## 3. Build and run

```bash
git submodule update --init --recursive

# GPU backend compiled in (default is OFF)
cmake -S tests -B build/gpu \
  -DCMAKE_BUILD_TYPE=Release -DDIC_REQUIRE_OPENCV=ON -DSEMPER_OPENCL=ON
cmake --build build/gpu -j"$(nproc)"
./build/gpu/dic_tests
```

The `ClRuntime` suite prints what was detected. When the backend is not
used it reports a `unavailable_reason` naming the step that failed — an
empty reason alongside an unused GPU is itself a bug.

```bash
# Force the CPU path without rebuilding — use this to bisect a suspected
# GPU-side difference.
SEMPER_OPENCL_DISABLE=1 ./build/gpu/dic_tests
```

Path B's round structure is what decided Phase 5, and it is hardware- and
image-dependent, so it is worth re-measuring before concluding anything about
Path B on a new device:

```bash
# Print per-round point counts, solve times, rescue rates and iteration
# counts for the Path B wavefront. Off by default; costs one getenv when off.
SEMPER_PATHB_ROUNDS=1 ./build/gpu/dic_tests ClPipelineParity
```

Compare the round sizes it prints against the small-batch launch costs from
`Perf.IcgnPathAThroughputCpuVsGpu`. A device path for Path B only makes sense
if some round is both large enough to clear the launch floor **and** not
dominated by simplex rescues — on the RTX 3060 no round is both.

With `SEMPER_OPENCL=OFF` (the default) no OpenCL code is compiled and the
binary carries no OpenCL symbols:

```bash
nm -D build/tests/dic_tests 2>/dev/null | grep -i opencl   # expect no output
```

---

## 4. Verification

Run all three at every phase. Record the numbers in §5.

### 4a. Throughput — must not decrease

```bash
for i in $(seq 1 15); do
  ./build/gpu/dic_tests Perf 2>&1 | grep -oE '[0-9]+ solves/s' | grep -oE '^[0-9]+'
done | sort -n | awk '{a[NR]=$1; s+=$1}
  END{printf "min=%d median=%d max=%d mean=%.0f\n", a[1], a[int((NR+1)/2)], a[NR], s/NR}'
```

Reference on the development container (4-core Xeon @ 2.1 GHz): **median
3205, max 3273**.

> **These absolute numbers are machine-specific and are not a target for
> your hardware.** `docs/PERF_BASELINE_bd44af0.md` quotes 4797 solves/s from
> a quieter machine; the container cannot reach it. The gate is
> **relative** — measure on your box before a change and after it, and
> compare those two. On a shared or thermally throttled machine prefer the
> **max** over the median: it is the least noise-contaminated estimate of
> true speed.

Two per-stage sweeps are worth running on their own on any new device,
because the conclusions they produced are hardware-dependent and both ended
in a *negative* result that a different machine could overturn:

```bash
# Phase 4/5: per-batch launch cost, from 8 points upward. A Path B device
# path needs a round that clears this floor and is not rescue-bound.
./build/gpu/dic_tests Perf.IcgnPathAThroughputCpuVsGpu

# Phase 6: the gradient stencil, with effective bus bandwidth printed. On a
# discrete GPU this flattens at the link rate and the ratio never reaches
# 1.0; on an integrated or unified-memory device it should not.
./build/gpu/dic_tests Perf.ImageGradThroughputCpuVsGpu
```

### 4b. Cross-ABI determinism — must stay byte-identical

The real bit-exactness gate. Same source, two instruction sets.

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

cmp corpus_x86-64.bin.bicubic  corpus_x86-64-v3.bin.bicubic
cmp corpus_x86-64.bin.keys6x6  corpus_x86-64-v3.bin.keys6x6
cmp ff_x86-64.bin              ff_x86-64-v3.bin
```

All three `cmp`s must be silent. This is what the `determinism` CI job runs.

### 4c. Run-to-run reproducibility

```bash
./build/gpu/dic_tests FullFieldGolden.RepeatSolve
```

Two solves in one process must agree exactly. Before Phase 0 this failed
every time.

### 4d. CPU/GPU parity (Phase 2 onward)

```bash
./build/gpu/dic_tests ClParity
```

Exact float equality per kernel, `==` rather than `CHECK_NEAR`. A single
differing bit is a failure.

`ClParity` drives each dispatch entry point directly, deliberately at small
sizes the production caller would route to the CPU: the size threshold is a
performance policy and must not become a hole in the parity coverage. The
device-dependent cases report themselves skipped, rather than passing
vacuously, when no suitable device is present.

---

## 5. Device results

Fill in as hardware is tried, so results accumulate here rather than in a
chat log.

| Date | GPU | Driver / ICD | CL ver | fp64 | exact fp32 | Throughput before → after | Parity | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-09 | *(none — CPU only)* | loader present, no ICD | — | — | — | 3205 / 3273 (median/max) | n/a | Development container, 4-core Xeon. Phase 0 reference |
| 2026-09 | POCL CPU device | pocl-opencl-icd 5.0 | 3.0 (CL C 1.2) | yes | yes | 2900 ON / 2910 OFF (median, back-to-back) | pass | `cpu-skylake-avx512`. Canonical reduction exact vs host at every size |
| 2026-09 | NVIDIA GeForce RTX 3060 Laptop | CUDA ICD | 3.0 | yes | yes | strain 0.67× at 2.2k pts → 5.32× at 37k pts | pass | Windows 11 / MinGW-w64. Phase 2. Fixed ~0.45 ms dispatch cost sets a 3600-point break-even; below it the caller stays on CPU |
| 2026-09 | NVIDIA GeForce RTX 3060 Laptop | CUDA ICD | 3.0 | yes | yes | Hessian pre-pass 3.5× at 2.5k pts → 25.7× at 58k pts (vs serial CPU) | pass | Windows 11 / MinGW-w64. Phase 3. fp32 only, so gated on `exact_fp32`. Dispatch floor ~0.6 ms + ~2 ms per megapixel of reference image, which is why the caller's threshold is per megapixel rather than a flat point count |
| 2026-09 | NVIDIA GeForce RTX 3060 Laptop | CUDA ICD | 3.0 | yes | yes | **Path A ICGN 1.12×–1.44× SLOWER than 20 threads; 1.00×–1.06× at 8 and 4 threads** (4.8×–12.5× vs *serial* CPU) | pass — 3 235 points, 0 mismatches | Windows 11 / MinGW-w64. Phase 4. **Throughput gate failed, so the stage ships off** (`kGpuIcgnPathAEnabled = false`). Cause is warp divergence on iteration count: one work-item per subset makes a warp run until its slowest lane converges, and ~137 of 3 470 points hit the 50-iteration cap. Remedy is round-based launches over a compacted active list |
| 2026-09 | NVIDIA GeForce RTX 3060 Laptop | CUDA ICD | 3.0 | yes | yes | **Path B not ported** — launch-per-round projects 2.5x-5.3x slower | n/a — no kernel written | Windows 11 / MinGW-w64. Phase 5. Path B is one big round plus a long tail; round 0 is 34-47% simplex rescues, which the Phase 4 acceptance rule sends to the CPU anyway, and every later round is under 700 points against a ~14.5 ms launch floor. Re-check with `SEMPER_PATHB_ROUNDS=1` |
| 2026-09 | NVIDIA GeForce RTX 3060 Laptop | CUDA ICD | 3.0 | yes | yes | **image gradients 0.19x at 0.26 MP -> 0.60x at 16.8 MP, no crossover** | pass - 5 geometries + degenerate sizes, 0 mismatches | Windows 11 / MinGW-w64. Phase 6. **Throughput gate failed, so the stage is not wired into the pipeline.** Bus-bound: 12 bytes/pixel against 4 flops/pixel, effective bandwidth flat at 3.94 GB/s. Unlike Phases 3-4 the CPU column is serial in the engine too, so this ratio is not an upper bound. Re-run `Perf.ImageGradThroughputCpuVsGpu` on an integrated or unified-memory device |
|  |  |  |  |  |  |  |  |  |

---

## 6. Triage

| Symptom | Likely cause | Action |
|---|---|---|
| Probe reports zero platforms | No ICD registered. `/etc/OpenCL/vendors/` missing or empty — `libOpenCL.so.1` alone is just the loader | Install a vendor driver, or `apt install pocl-opencl-icd` for a CPU device |
| Probe reports no device | ICD present but exposes no matching device | Check `clinfo`; the probe tries GPU first, then any device type |
| `exact_fp32 == false` | Device does not advertise correctly-rounded divide/sqrt | Expected on some hardware. ICGN correctly stays on CPU; strain may still run |
| `fp64 == false` | No `cl_khr_fp64` | Strain stays on CPU. Common on mobile and some integrated parts |
| Kernel build failure | Include expansion or a syntax error in the embedded source | The reason string carries the device compiler log; check the generated kernel header |
| GPU unused, empty reason | Bug — the probe must always explain itself | Report it; a silent fallback is the failure mode the reason string exists to prevent |
| Parity mismatch | A forbidden build flag, a reintroduced FMA, or a changed reduction order | See [DETERMINISM.md](DETERMINISM.md) §"If the determinism job fails". **Never** widen a tolerance to make it pass |
| Throughput dropped | Something linked or initialised on a hot path | Bisect with `SEMPER_OPENCL_DISABLE=1`; compare against the *same machine's* previous number |
