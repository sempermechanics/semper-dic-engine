#ifndef SEMPER_SUBSET_HPP
#define SEMPER_SUBSET_HPP

#include <semper/types.hpp>
#include <semper/image.hpp>

namespace Semper {

    // =========================================================
    // 🚀 GLOBAL HESSIAN CACHE ENTRY
    //    Memory footprint: ~153 bytes × 77K points = ~11.8 MB total.
    //    Stores ONLY the result of the expensive H accumulation,
    //    so precompute_subset_fast can skip it entirely.
    // =========================================================
    struct CachedHessianData {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Matrix<float, 6, 6> H_inv;
        float mean_intensity = 0.0f;
        float std_dev = 1.0f;
        bool valid = false;
        // ── LM ADDITION ─────────────────────────────────────────────────────────
        // Raw Hessian before inversion. Stored so the hessian pool entry
        // can supply both H_inv (for standard ICGN) and H (for LM ICGN)
        // to precompute_subset_fast without any recomputation.
        // Memory cost: 144 bytes × 78k pool entries ≈ 11 MB. One-time allocation.
        Eigen::Matrix<float, 6, 6> H;
        // ────────────────────────────────────────────────────────────────────────
    };

    class SubsetPrecomputer {
    public:
        // Full precompute: compute_hessian_only, then precompute_subset_fast
        // from it. For callers without a Hessian pool (tests, re-anchoring).
        static void precompute_subset(SubsetData& data, const Image& ref_img,
                                      int_t cx, int_t cy, int_t dim);

        // 🚀 NEW: Compute ONLY the H_inv + stats for caching.
        //    Zero heap allocation, OMP-safe, no SubsetData needed.
        static CachedHessianData compute_hessian_only(const Image& ref_img,
                                                      int_t cx, int_t cy, int_t dim);

        // 🚀 NEW: Fast precompute that skips H re-accumulation using the pool.
        //    Saves ~65% of precompute FP work per call site. An invalid
        //    `cached` gives an uninitialized subset.
        static void precompute_subset_fast(SubsetData& data, const Image& ref_img,
                                           int_t cx, int_t cy, int_t dim,
                                           const CachedHessianData& cached);
    };

} // namespace Semper

#endif // SEMPER_SUBSET_HPP