# Determinism and the bit-exactness contract

This document defines what "the same answer" means for this engine, and
which build settings and code shapes are load-bearing in delivering it.

It exists because the OpenCL backend must reproduce the CPU result
bit-for-bit. That is only a meaningful requirement if the CPU result is
itself fixed — and until the canonical-math work it was not.

## What was wrong before

The engine produced *different bits on different machines*, in three
independent ways:

1. **`-ffast-math` was applied to the math sources.** It licenses
   reassociation and implicit FMA contraction, so the summation order was
   whatever the compiler chose for that build.

2. **The SIMD reduction order followed the hardware vector width.**
   `simd.hpp` accumulated into `VTraits<v_float32>::vlanes()` partial
   sums — 4 on SSE/NEON, 8 on AVX2. Lane *k* holds the partial sum of
   elements *k, k+W, k+2W, …*, so the register width **is** the
   summation order.

3. **Eigen packs its fixed-size products per ISA.** A `6x6 * 6x1`
   product associates its dot products differently under SSE2 than under
   AVX2. This one survived the first two fixes: with images, gradients,
   subset precompute, `H_inv` and the SIMD kernels all bit-identical
   across ISA levels, the full ICGN solve still diverged — same status,
   same iteration count, different values.

Symptom of all this: the golden-corpus tolerances had been widened twice
(`ce28937`, `6872cd7`) to absorb what the commit messages called "runner
FP noise". That noise was not the runner.

4. **Path B was not reproducible even on one machine.** The flood fill
   drained a shared `std::priority_queue` from several worker threads, and
   each point's initial guess is extrapolated from whichever parent won the
   `compare_exchange` for that cell. Measured on the host suite before the
   rewrite: **0 of 15 runs reproduced**, with 10-230 of ~3100 output floats
   differing between two consecutive solves of the same binary on identical
   input.

   Nothing caught this, because the only bit-identity tests in the repo were
   subset-level (`Engine.RepeatSolve_BitIdentical`,
   `Robustness.ConcurrentSolves_BitIdenticalToSingleThread`) and the golden
   corpus never entered `run_full_field` at all.

## The contract

### Compile flags

| Target | Flags | Why |
|---|---|---|
| `semper_math` (root `CMakeLists.txt`) | `-O3 -fno-fast-math -ffp-contract=off` | The four mirrored translation units |
| whole `tests/` tree | `-O2 -fno-fast-math -ffp-contract=off` | See "test inputs" below |
| `semper_pipeline` | `-O3 -ffast-math` (unchanged) | Orchestration only; performs no mirrored arithmetic |
| OpenCL kernels | no `-cl-fast-relaxed-math`, no `-cl-mad-enable`, plus `-cl-fp32-correctly-rounded-divide-sqrt` and `#pragma OPENCL FP_CONTRACT OFF` | Device-side equivalent |

`semper_math` contains exactly the four mirrored TUs — `image_processor`,
`subset_precomputer`, `optimization_engine`, `strain_calculator` — so the
target boundary and the contract boundary coincide.

**Measured cost of strict FP: none.** Throughput was 3212 solves/s before
and 3192 median / 3232 max after, on the same host. The hot reductions
were already hand-written with explicit intrinsics and never depended on
`-ffast-math` to vectorize.

### The canonical reduction

Every long summation accumulates into exactly **four** accumulators,
stride-4 interleaved, combined pairwise:

```
acc[k] += term(4*i + k)                k = 0..3
result  = (acc0 + acc1) + (acc2 + acc3)
result += term(j)                      j = 4*(n/4) .. n-1, in index order
```

Four, because that is reproducible everywhere that matters: one
SSE/NEON register, half an AVX2 register, and four scalar registers
inside an OpenCL work-item. `include/semper/simd.hpp` pins OpenCV to
128-bit vectors via `CV__SIMD_FORCE_WIDTH` and finishes with an explicit
lane combine rather than `v_reduce_sum`, whose association is
unspecified.

### No FMA

The kernels use `v_mul` + `v_add`, never `v_fma`. A fused multiply-add
rounds once; a separate multiply and add round twice. The canonical
reference is written `acc += d * d` under `-ffp-contract=off`, i.e. two
roundings, so the vector path must not fuse either. This also keeps the
kernels correct on baseline x86-64, where hardware FMA is not guaranteed.

### One definition of the math

`include/semper/kernels/canonical_math.h` is written in the common subset
of C99 and OpenCL C and is compiled **as the same text** into both the
host library and the `.cl` kernels. It holds the reductions, the Keys 6x6
and bicubic weights, the three image samplers behind
`Image::interpolate_*`, and the small dense linear algebra (3x3 and 6x6
inverses, the 6x6 matvec, the 6-norm).

Those replace `Eigen`'s equivalents *on the host as well*, not only on
the device. Eigen's blocked, pivoting LU cannot be called from a kernel;
rather than have the GPU chase Eigen, both sides use one definition.

Because OpenCL C 1.2 has no generic address space, the three reduction
bodies live in `canonical_reductions.inc` and the samplers in
`canonical_interp.inc`, each included once per address space (unsuffixed
for private, `_g` for `__global`) rather than being copy-pasted.

The samplers moved out of `image_processor.cpp` when the ICGN kernel
needed them. An interpolator is the most parity-critical routine here:
every ICGN iteration of every point goes through it, and a one-ulp
disagreement does not stay one ulp — it changes the Newton step, which
changes the next warp, which changes which pixels are sampled at all.
The one sequence still written twice is the batch-of-four row sum in
`interpolate_bicubic_x4` / `interpolate_keys_fourth_x4`, which exists so
the compiler can pipeline four independent points;
`Image.BatchOfFourMatchesScalarExactly` pins it to the scalar sampler
with `==`.

The header **fails the build** with `#error` if `__FAST_MATH__` is
defined. Forgetting to put an including TU on the strict-FP list is a
compile error, not a silent loss of the guarantee.

### Test inputs must be deterministic too

`tests/framework/synthetic.h` builds the ground-truth images by summing
several hundred `std::exp` terms per pixel. Under `-ffast-math` GCC
vectorizes that loop onto libmvec, whose AVX2 `exp` does not agree
bit-for-bit with its SSE2 one — so the generated **images** differed
between `-march` levels, and every downstream comparison differed with
them. The whole `tests/` tree is therefore strict too. A determinism gate
whose own inputs are non-deterministic proves nothing.

### Path B is level-synchronous

`full_field_path_b.cpp` no longer uses a priority queue. It advances in
rounds: each round takes the whole current frontier, resolves every
child's parent **before** any solving happens, and only then solves that
round's children in parallel.

The tie-break is explicit — best parent correlation score, ties to the
lowest parent flat index — and the frontier is kept sorted by flat index
so that rule is well defined. Because every guess is fixed before the
parallel section opens, thread interleaving can no longer change the
answer. `compute_order` is assigned after the round in winner order for
the same reason.

This is also the shape the GPU needs: one kernel launch per round.

Measured effect of the rewrite: determinism went from 0/15 to 15/15
reproducible, with **identical coverage** — the same 392 and 380 points,
and the same Path A / Path B split (413/70 and 420/58), as the racing
queue produced. DICe field agreement is unchanged at rms 0.0006 px
against a 0.005 tolerance, 230/230 points compared, 100% convergence
across every load step of the ladder.

## What is guaranteed, and what is not

**Guaranteed:** for one toolchain and one libm, the engine produces
bit-identical output regardless of the target ISA — SSE2, AVX2, NEON.
This is enforced by the `determinism` CI job, which builds the same
source at `-march=x86-64` and `-march=x86-64-v3` and byte-compares both
the subset corpus and the full-field golden. Run-to-run reproducibility
within a single build is separately gated by
`FullFieldGolden.RepeatSolve_IsDeterministic`.

**Not guaranteed:** identical output across different *libm*
implementations. That dependency is confined to the test image
synthesis, not to the engine — but it does mean the committed golden
fixtures are pinned to the toolchain that captured them. This is why
`test_golden_corpus.cpp` compares at 1e-6 px / 1e-7 strain rather than
exactly: tight enough to catch any real change in engine arithmetic,
loose enough to survive a glibc bump. The exact gate is the CI job.

### Recaptures of the committed fixtures

A golden fixture is recaptured only when a deliberate change to engine
arithmetic moves it, never to make a red test green. Each one is recorded
here with what moved and by how much.

| When | Fixture | Why |
|---|---|---|
| GPU Phase 3a | `golden_corpus.bin.{bicubic,keys6x6}` | The static Hessian moved off Eigen's `PartialPivLU` inverse onto `semper_inv6x6` (Gauss-Jordan, partial pivoting), so the Phase 3 pre-pass kernel has a reference it can reproduce. Both are correct inverses of the same matrix and differ only in the last few digits, which shifts the ICGN iterate path: 394 of 580 points (4x4) and 399 of 580 (6x6) moved by roughly 1e-5, with **0 status mismatches** — no point changed whether it converged. Recaptured on Linux GCC Release, the platform the fixture is pinned to; the accompanying accumulation change was separately shown to be bit-identical, and `full_field_golden.bin` was verified **unchanged** and was not recaptured. |

## If the determinism job fails

Something in the list above was broken. In rough order of likelihood:

1. A new TU doing mirrored arithmetic was added without strict FP flags.
   (`canonical_math.h`'s `#error` catches this if it includes the header.)
2. `CV__SIMD_FORCE_WIDTH` stopped taking effect because some other header
   pulled in `intrin.hpp` first — `simd.hpp` has a `static_assert` for
   exactly this.
3. An Eigen fixed-size expression was reintroduced into a mirrored TU.
4. A `v_fma` was reintroduced into `simd.hpp`.
5. Path B regained an order-dependent step — a guess read from a
   neighbour mid-round, or a tie-break that depends on completion order
   rather than on flat index.

Do **not** widen a tolerance to make it pass.
