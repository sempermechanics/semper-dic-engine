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
| **1** | OpenCL runtime + build plumbing: `SEMPER_OPENCL`, dlopen loader, device capability gate, kernel embedding. **No kernels.** | In progress |
| **2** | Strain VSG on GPU. One work-item per grid point, fp64. Currently the only fully serial numerical stage. | Pending |
| **3** | Hessian pre-pass on GPU. One work-item per grid point. | Pending |
| **4** | Path A ICGN on GPU. **One work-item per subset**, so the reduction keeps the canonical order rather than becoming a cross-lane tree. | Pending |
| **5** | Path B wavefront on GPU. Reuses the Phase 4 kernel; one launch per round. Only possible because Phase 0 made Path B round-based. | Pending |
| **6** | Image prep (gradients, optional blur). AKAZE stays on CPU — randomized RANSAC, poor return. | Pending |

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
| `Correctly-rounded divide/sqrt` in single-precision FP config | ICGN (Phases 3–5) | ICGN stays on CPU — **cannot** be bit-exact without it |

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

---

## 5. Device results

Fill in as hardware is tried, so results accumulate here rather than in a
chat log.

| Date | GPU | Driver / ICD | CL ver | fp64 | exact fp32 | Throughput before → after | Parity | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-09 | *(none — CPU only)* | loader present, no ICD | — | — | — | 3205 / 3273 (median/max) | n/a | Development container, 4-core Xeon. Phase 0 reference |
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
