// Post-solve helpers for the full-field pipeline: strain + output packing,
// per-thread telemetry aggregation, the profiling log, and the metrics array
// handed back to Kotlin. Lifted verbatim out of full_field_solver.cpp.

#include "full_field_internal.hpp"

#include <semper/strain.hpp>
#include <semper/tuning.hpp>
#include "util/log.hpp"

#include <chrono>
#include <vector>

#undef LOG_TAG
#define LOG_TAG "SemperPipeline"

namespace Semper {
namespace pipeline {
namespace internal {

PackedFieldResult pack_full_field_output(
        ResultGrid& resultGrid,
        int gridW,
        int gridH,
        int step,
        int strain_window,
        float* output_ptr,
        int output_capacity) {
    PackedFieldResult out;
    auto t_strain_start = std::chrono::high_resolution_clock::now();
    DisplacementField dispField;
    dispField.width = gridW;
    dispField.height = gridH;
    dispField.step = step;
    dispField.u.resize(static_cast<size_t>(gridW * gridH), 0.0f);
    dispField.v.resize(static_cast<size_t>(gridW * gridH), 0.0f);
    dispField.valid.resize(static_cast<size_t>(gridW * gridH), false);

    for (int y = 0; y < gridH; ++y) {
        for (int x = 0; x < gridW; ++x) {
            int idx = y * gridW + x;
            if (resultGrid[y][x].solved && resultGrid[y][x].corr >= 0.0f) {
                dispField.u[idx] = resultGrid[y][x].u;
                dispField.v[idx] = resultGrid[y][x].v;
                dispField.valid[idx] = true;
            }
        }
    }

    out.strain = StrainCalculator::compute_vsg_strain(dispField, strain_window);

    for (int y = 0; y < gridH && !out.output_truncated; ++y) {
        for (int x = 0; x < gridW; ++x) {
            int idx = y * gridW + x;
            if (!dispField.valid[idx]) continue;
            if ((out.valid_count + 1) * 8 > output_capacity) {
                out.output_truncated = true;
                break;
            }
            if (out.strain.exx[idx] <= tuning::kStrainFailSentinel) {
                out.dropped_by_post_filter++;
                resultGrid[y][x].solved = false;
                continue;
            }
            int out_idx = out.valid_count * 8;
            output_ptr[out_idx + 0] = resultGrid[y][x].x;
            output_ptr[out_idx + 1] = resultGrid[y][x].y;
            output_ptr[out_idx + 2] = dispField.u[idx];
            output_ptr[out_idx + 3] = dispField.v[idx];
            output_ptr[out_idx + 4] = out.strain.exx[idx];
            output_ptr[out_idx + 5] = out.strain.eyy[idx];
            output_ptr[out_idx + 6] = out.strain.exy[idx];
            output_ptr[out_idx + 7] = resultGrid[y][x].corr;
            out.valid_count++;
        }
    }
    out.time_strain_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_strain_start).count();
    return out;
}

SolveSummary aggregate_thread_stats(
        const std::vector<ThreadStats>& stats_pathA,
        const std::vector<ThreadStats>& stats_pathB,
        const std::vector<ThreadStats>& stats_anchors,
        int safe_cores) {
    SolveSummary s;

    for (int i = 0; i < safe_cores; i++) {
        s.a_icgn += stats_pathA[i].icgn_time_ms; s.a_simp += stats_pathA[i].simplex_time_ms;
        s.pathA_pts += stats_pathA[i].points_solved; s.total_hessian += stats_pathA[i].hessian_time_ms; s.total_icgn_iters += stats_pathA[i].icgn_iters;
        s.a_simp_calls += stats_pathA[i].simplex_calls; s.a_simp_saved += stats_pathA[i].simplex_saved; s.a_simp_dead += stats_pathA[i].simplex_dead;

        s.b_icgn += stats_pathB[i].icgn_time_ms; s.b_simp += stats_pathB[i].simplex_time_ms;
        s.b_wait += stats_pathB[i].queue_wait_time_ms; s.pathB_pts += stats_pathB[i].points_solved; s.total_hessian += stats_pathB[i].hessian_time_ms;
        s.total_icgn_iters += stats_pathB[i].icgn_iters;
        s.b_simp_calls += stats_pathB[i].simplex_calls; s.b_simp_saved += stats_pathB[i].simplex_saved; s.b_simp_dead += stats_pathB[i].simplex_dead;

        s.a_simp_crash += stats_pathA[i].simplex_from_crash;
        s.a_simp_timeout += stats_pathA[i].simplex_from_timeout;

        s.b_simp_crash += stats_pathB[i].simplex_from_crash;
        s.b_simp_timeout += stats_pathB[i].simplex_from_timeout;

        s.n_icgn += stats_anchors[i].icgn_time_ms; s.n_simp += stats_anchors[i].simplex_time_ms;
        s.anchor_pts += stats_anchors[i].points_solved; s.total_hessian += stats_anchors[i].hessian_time_ms;
        s.total_icgn_iters += stats_anchors[i].icgn_iters;
        s.n_simp_calls += stats_anchors[i].simplex_calls; s.n_simp_saved += stats_anchors[i].simplex_saved; s.n_simp_dead += stats_anchors[i].simplex_dead;
        s.n_simp_crash += stats_anchors[i].simplex_from_crash;
        s.n_simp_timeout += stats_anchors[i].simplex_from_timeout;
    }

    return s;
}

void log_profiling_summary(
        const PhaseTimings& t,
        const SolveSummary& s,
        const MeshSeedResult& seeds,
        int valid_count) {
    LOGD("=== ⏱️ ADVANCED PERFORMANCE PROFILING ===");
    LOGD("Image Prep & Masking: %.2f ms", t.img_prep);
    LOGD("Seeding (phase+anchors): %.2f ms", t.phase_corr + t.anchors);
    LOGD("  ↳ Phase lock: %s | Mesh coverage: %.1f%% of ROI | anchors %d/%d accepted, %d kept",
         seeds.phase_locked ? "YES" : "NO", seeds.coverage * 100.0f,
         seeds.accepted, seeds.attempted, seeds.solved);
    if (s.anchor_pts > 0 || s.n_simp_calls > 0) {
        LOGD("  ↳ Anchor ICGN Math: %.2f ms | Simplex: %.2f ms (%d pts kept as results)",
             s.n_icgn, s.n_simp, s.anchor_pts);
        LOGD("      ↳ %d Calls (%d Crashes, %d Timeouts) -> %d Saved, %d Dead",
             s.n_simp_calls, s.n_simp_crash, s.n_simp_timeout, s.n_simp_saved, s.n_simp_dead);
    }
    LOGD("Hessian Pre-pass:     %.2f ms (One-time Global Math)", t.prepass);

    LOGD("Delaunay Mesh Setup:  %.2f ms", t.delaunay);
    LOGD("Contour Assignment:   %.2f ms", t.contour_assign);
    LOGD("Extrapolation Pass:   %.2f ms", t.extrapolate);
    LOGD("Spatial Smoothing:    %.2f ms", t.smoothing);

    LOGD("Path A (Mesh Eval):   %.2f ms (Throughput: %.1f pts/ms)", t.pathA, (t.pathA > 0) ? s.pathA_pts / t.pathA : 0.0);
    if (s.pathA_pts > 0) {
        LOGD("  ↳ Path A ICGN Math: %.2f ms | Simplex: %.2f ms", s.a_icgn, s.a_simp);
        LOGD("      ↳ %d Calls (%d Crashes, %d Timeouts) -> %d Saved, %d Dead",
             s.a_simp_calls, s.a_simp_crash, s.a_simp_timeout, s.a_simp_saved, s.a_simp_dead);
    }

    if (s.pathB_pts > 0) {
        LOGD("  ↳ Path B ICGN Math: %.2f ms | Simplex: %.2f ms", s.b_icgn, s.b_simp);
        LOGD("      ↳ %d Calls (%d Crashes, %d Timeouts) -> %d Saved, %d Dead",
             s.b_simp_calls, s.b_simp_crash, s.b_simp_timeout, s.b_simp_saved, s.b_simp_dead);
        LOGD("  ↳ Path B Wait Time: %.2f ms (Thread idle/lock contention)", s.b_wait);
    }

    LOGD("Strain Calculation:   %.2f ms", t.strain);
    LOGD("Total JNI Execution:  %.2f ms", t.total);
    LOGD("--- ENGINE MATH & HARDWARE EFFICIENCY ---");
    LOGD("Total Points Solved:  %d (Anchors: %d, A: %d, B: %d)",
         valid_count, s.anchor_pts, s.pathA_pts, s.pathB_pts);
    LOGD("Hessian Precompute:   %.2f ms (both paths)", s.total_hessian);
    LOGD("Average ICGN Speed:   %.4f iterations / point", (valid_count > 0) ? (float)s.total_icgn_iters / valid_count : 0.0f);
    int tot_simp_calls = s.a_simp_calls + s.b_simp_calls + s.n_simp_calls;
    int tot_simp_saved = s.a_simp_saved + s.b_simp_saved + s.n_simp_saved;
    int tot_simp_dead = s.a_simp_dead + s.b_simp_dead + s.n_simp_dead;
    LOGD("Total Simplex Rescue: %.2f ms (%d calls -> %d saved, %d dead)", (s.a_simp + s.b_simp + s.n_simp), tot_simp_calls, tot_simp_saved, tot_simp_dead);
    LOGD("=======================================");
}

void fill_engine_metrics(
        float* metrics,
        int metrics_len,
        const PhaseTimings& t,
        const SolveSummary& s,
        int total_valid_points,
        int valid_count,
        int dropped_by_post_filter,
        const MeshSeedResult& seeds,
        MeshQuality mesh_quality) {
    float avg_iters = 0.0f;
    if (valid_count > 0) {
        avg_iters = (float)s.total_icgn_iters / (float)valid_count;
    }

    if (metrics != nullptr && metrics_len >= 16) {
        const int tot_simp_calls = s.a_simp_calls + s.b_simp_calls + s.n_simp_calls;
        const int tot_simp_saved = s.a_simp_saved + s.b_simp_saved + s.n_simp_saved;
        const int tot_simp_dead = s.a_simp_dead + s.b_simp_dead + s.n_simp_dead;

        float metrics_data[23];

            // 0-4: Point Counts
            metrics_data[0] = (float)total_valid_points;      // Total Attempted
            metrics_data[1] = (float)valid_count;             // Total Solved
            metrics_data[2] = (float)(total_valid_points - valid_count); // Total Rejected
            metrics_data[3] = (float)s.pathA_pts;             // Solved via Mesh
            metrics_data[4] = (float)s.pathB_pts;             // Solved via Flood Fill

            // 5-8: Simplex Stats
            metrics_data[5] = (float)tot_simp_calls;          // Total Simplex Rescues
            metrics_data[6] = (float)tot_simp_saved;          // Rescues that Succeeded
            metrics_data[7] = (float)tot_simp_dead;           // Rescues that Failed
            metrics_data[8] = avg_iters;                      // Mean ICGN Iterations

            // 9-13: Timestamps & Performance (ms)
            metrics_data[9]  = (float)t.total;                // Total Wall Time
            metrics_data[10] = (float)(t.phase_corr + t.anchors); // Seeding time (Frozen slot)
            metrics_data[11] = (float)t.prepass;              // Hessian Pre-Pass Time
            metrics_data[12] = (float)t.delaunay;             // Mesh Setup Time
            metrics_data[13] = (float)t.strain;               // Strain Calc Time

            // 14-15: Ratios
            // Throughput over the whole solve, not just Path A — when the
            // mesh phase is skipped (pathA_pts == 0) the old formula
            // reported 0.00 pts/ms even though RGDIC solved every point.
            metrics_data[14] = (t.total > 0) ? (float)(valid_count / t.total) : 0.0f; // Throughput Pts/ms
            metrics_data[15] = (total_valid_points > 0) ? ((float)valid_count / total_valid_points) * 100.0f : 0.0f; // Convergence %

            // 16: How the solve was seeded. Surfaces silently-skipped mesh
            // phases (too few anchors kept → Path C, RGDIC-only) in the report
            // instead of just "0.0 ms / 0 points".
            //   2 = full mesh, 1 = sparse mesh, 0 = Path C fallback
            metrics_data[16] = (mesh_quality == MeshQuality::FULL) ? 2.0f
                             : (mesh_quality == MeshQuality::SPARSE) ? 1.0f : 0.0f;

            // 17-18: Simplex-rescue cost (Phase 3.2 — "measure before touching").
            // Already computed and aggregated above (log_profiling_summary logs
            // the same s.a_simp+s.b_simp / s.a_icgn+s.b_icgn via LOGD) but never
            // reached the app before this — these two slots are the only change,
            // no solve behaviour is touched.
            metrics_data[17] = (float)(s.a_simp + s.b_simp + s.n_simp);  // Total simplex-rescue time (ms)
            metrics_data[18] = (float)(s.a_icgn + s.b_icgn + s.n_icgn);  // Total ICGN time (ms) — the main solve

            // 19: points the anchor lattice solved outright. Slots 3 and 4 are
            // Frozen as "via mesh" (Path A) and "via flood fill" (Path B), and
            // neither describes an anchor — so the third bucket is appended
            // rather than folded into one of them. 3 + 4 + 19 == slot 1.
            metrics_data[19] = (float)s.anchor_pts;           // Solved via Anchor Lattice

            // 20: points that converged and were then discarded because their
            // VSG strain fit was rank-deficient. Logged since forever, never
            // reported: a caller seeing slot 1 fall short of slots 3+4+19 had
            // no way to tell a dropped point from one that never solved. With
            // this slot the accounting closes exactly:
            //   slot 3 + slot 4 + slot 19 == slot 1 + slot 20
            metrics_data[20] = (float)dropped_by_post_filter;  // Dropped by strain post-filter

            // 21/22: the two seeding health signals. The anchor lattice is a
            // translation-only front end -- past roughly 5 degrees of rotation
            // the phase lock fails and mesh coverage collapses (0.45 at 5 deg,
            // 0.15 at 15 deg; docs/SEEDING_BENCHMARK.md 4.5), and without these
            // slots that field is reported exactly like a healthy one.
            metrics_data[21] = seeds.phase_locked ? 1.0f : 0.0f;  // Phase-correlation lock
            metrics_data[22] = seeds.coverage;                    // Mesh hull / ROI area, 0..1

            int ncopy = (metrics_len >= 23) ? 23 : (metrics_len >= 22) ? 22
                      : (metrics_len >= 21) ? 21 : (metrics_len >= 20) ? 20
                      : (metrics_len >= 19) ? 19 : (metrics_len >= 17) ? 17 : 16;
        for (int mi = 0; mi < ncopy; ++mi) metrics[mi] = metrics_data[mi];
    }
}

} // namespace internal
} // namespace pipeline
} // namespace Semper
