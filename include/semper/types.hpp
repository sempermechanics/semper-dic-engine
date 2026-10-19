#ifndef SEMPER_TYPES_HPP
#define SEMPER_TYPES_HPP

#include <vector>
#include <Eigen/Dense>
#include <Eigen/StdVector> // 🚀 Added for safe Eigen STL vector alignment

namespace Semper {
    // 🚀 THE BIG SHIFT: All core math is now 32-bit float
    using scalar_t = float;
    using int_t = int;

    // Initialization modes
    enum InitializationMode {
        INIT_AUTO_SEARCH = 0,
        INIT_NO_SEARCH = 1,
        INIT_NO_SIMPLEX = 2
    };

    // DIC analysis result (6-DOF + status). Default-initialized so a result
    // that is read before every field was written can never carry indeterminate
    // memory (the same class of bug the SubsetData::dim comment below records).
    struct AnalysisResult {
        float u = 0.0f;
        float v = 0.0f;
        float ux = 0.0f;
        float uy = 0.0f;
        float vx = 0.0f;
        float vy = 0.0f;

        int status = 0;
        float correlation_score = 0.0f;
        int iters = 0;
        int invalid_ref_pixels = 0; // 🚀 NEW: Tracks Ghost Wall overlap
        // Iterations of the IC-GN run that decided on a Simplex rescue: the
        // first run when one followed, otherwise `iters` itself. The rescue
        // telemetry classifies on this; `iters` is the run after the rescue.
        int trigger_iters = 0;
    };

    // Seed node for reliability-guided propagation
    struct SeedNode {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        int x_idx, y_idx;
        float u, v, ux, uy, vx, vy;
        float correlation_score;

        SeedNode() : x_idx(0), y_idx(0), u(0.0f), v(0.0f), ux(0.0f), uy(0.0f), vx(0.0f), vy(0.0f), correlation_score(0.0f) {}

        SeedNode(int x, int y, float _u, float _v, float _ux, float _uy, float _vx, float _vy, float c)
                : x_idx(x), y_idx(y), u(_u), v(_v), ux(_ux), uy(_uy), vx(_vx), vy(_vy), correlation_score(c) {}

        bool operator<(const SeedNode& other) const {
            return correlation_score > other.correlation_score;
        }
    };

    // Replaces the old "Subset" class. Now just a pure data struct.
    struct SubsetData {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        // dim MUST default to 0 (never a real subset dim): precompute_subset
        // only allocates its vectors when data.dim != dim, so an indeterminate
        // dim that happens to equal the request (e.g. stale stack memory from
        // a previous SubsetData at the same address) skips allocation and
        // null-derefs. Caught by TSan in CI.
        int_t cx = 0, cy = 0, dim = 0;
        std::vector<int_t> x_offsets, y_offsets;
        // 🚀 OPTIMIZATION T1.1: Pre-converted float offsets to eliminate 4-cycle SCVTF latency in the ICGN hot loop
        std::vector<float> x_offsets_f, y_offsets_f;
        std::vector<scalar_t> ref_intensities;
        std::vector<scalar_t> norm_ref_intensities;
        std::vector<scalar_t> gx_vec, gy_vec; // Pre-sampled gradients
        scalar_t mean_intensity = 0.0f;
        scalar_t std_dev = 0.0f;

        // 🚀 Matrix memory footprint cut in half!
        Eigen::Matrix<float, 6, 6> H_inv;

        // ── LM ADDITION ─────────────────────────────────────────────────────────
        // Raw (undamped) Hessian. Stored so solve_icgn can apply
        // α-damping to H(0,0) and H(1,1) once before the iteration loop
        // without re-accumulating from steepest_descent_images each time.
        // Memory cost: 144 bytes per SubsetData (one per OMP thread — negligible).
        Eigen::Matrix<float, 6, 6> H;
        // ────────────────────────────────────────────────────────────────────────

        std::vector<Eigen::Matrix<float, 6, 1>,
                Eigen::aligned_allocator<Eigen::Matrix<float, 6, 1>>>
                steepest_descent_images;

        // 🚀 SIMD SoA MIRROR: steepest_descent_images transposed into 6
        // contiguous planes (plane k at [k*n, (k+1)*n)). The Structure-of-Arrays
        // layout lets the portable SIMD kernel (simd.hpp) vectorize the
        // dp_sum gradient accumulation, which the strided AoS layout blocks.
        // Kept in sync with steepest_descent_images by SubsetPrecomputer.
        std::vector<float> sdi_planes;

        bool is_initialized = false;
    };
}
#endif