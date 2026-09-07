#ifndef SEMPER_SOLVER_HPP
#define SEMPER_SOLVER_HPP

#include <semper/image.hpp>
#include <semper/types.hpp>

namespace Semper {

    class OptimizationEngine {
    public:
        // Profiling trackers (kept as double for timing precision)
        double time_icgn_ms = 0.0;
        double time_simplex_ms = 0.0;
        int count_simplex = 0;
        // ── LM ADDITION ─────────────────────────────────────────────────────────
        // Matches DICe schema parameters:
        //   use_objective_regularization()           → lm_enabled
        //   levenberg_marquardt_regularization_factor() → lm_alpha
        //
        // DICe adds lm_alpha to H(0,0) and H(1,1) only — the two translation
        // DOFs — leaving the four strain-gradient diagonal entries unmodified.
        // Default off (lm_alpha = 0.0f) preserves existing behaviour exactly.
        //
        // Tuning: start at lm_alpha = 1e-3f and increase until timeouts drop.
        // Typical H diagonal values for a 41×41 subset with good speckle are
        // in the range [1e3, 1e5], so alpha should be 1–3 orders of magnitude
        // smaller than H(0,0) to regularise without over-damping.
        bool  lm_enabled = false;
        float lm_alpha   = 0.0f;
        // ────────────────────────────────────────────────────────────────────────
        bool use_6x6_interpolator = false;
        // 🟡 BUG 3 FIX: Separated Buffers
        // Prevents memory aliasing and silent data corruption between the two solvers
        std::vector<float> icgn_buffer;
        std::vector<float> simplex_buffer;

        // 🚀 UPDATED SIGNATURE: Now accepts full 6-DOF guess
        AnalysisResult calculate_deformation(const SubsetData &subset,
                                             const Image &def_img, float guess_u,
                                             float guess_v, float guess_ux,
                                             float guess_uy, float guess_vx, float guess_vy,
                                             InitializationMode init_mode);

    private:
        void estimate_initial_guess(const SubsetData &subset, const Image &def_img,
                                    scalar_t &best_u, scalar_t &best_v);

        // 🚀 UPDATED SIGNATURE: ICGN now initializes with full strain tensor
        AnalysisResult solve_icgn(const SubsetData &subset, const Image &def_img,
                                  float init_u, float init_v, float init_ux,
                                  float init_uy, float init_vx, float init_vy);

        AnalysisResult solve_simplex(const SubsetData &subset, const Image &def_img,
                                     AnalysisResult start, bool translation_only);

        float evaluate_znssd(const SubsetData &subset, const Image &def_img, float u,
                             float v, float ux, float uy, float vx, float vy,
                             std::vector<float> &buffer, const std::vector<bool> *ref_valid_mask = nullptr);
        // 🚀 NEW: Centralized mask builder
        // build_ref_valid stays vector<bool> until a measured A/B proves a
        // packed bitmask (or similar) still clears ≥95% of bd44af0 throughput.
        // Do not rewrite as a drive-by cleanup.
        std::vector<bool> build_ref_valid(const SubsetData& subset) const {
            const size_t n = subset.x_offsets.size();
            std::vector<bool> rv(n, true);
            for (size_t i = 0; i < n; ++i) {
                if (subset.ref_intensities[i] < -5.0f) rv[i] = false;
            }
            return rv;
        }
    };

} // namespace Semper
#endif