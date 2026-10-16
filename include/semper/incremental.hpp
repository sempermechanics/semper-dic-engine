#ifndef SEMPER_INCREMENTAL_HPP
#define SEMPER_INCREMENTAL_HPP

#include <semper/cancel.hpp>
#include <semper/pipeline.hpp>
#include <opencv2/core.hpp>

#include <memory>
#include <string>

namespace Semper {
namespace pipeline {

/**
 * Settings for an incremental (updated-reference) DIC sequence.
 *
 * `field` describes the frame-0 grid exactly as for run_full_field: results
 * are always reported at those grid points, as total displacement from frame
 * 0. `update_interval` is how many solved frames pass before the reference
 * moves to the latest solved frame:
 *   0   never — a fixed reference, identical to calling run_full_field per frame
 *   1   every frame (classic incremental DIC)
 *   N   every N frames, or as soon as a frame reports 5% fewer points than the
 *       first frame on the current reference
 * A node the solve cannot reach on a frame coasts, unreported, on the plane
 * through its neighbours' increments, and is reported again once measured
 * (after a long coast, once its earlier subset confirms it).
 */
struct IncrementalParams {
    FullFieldParams field;
    int update_interval = 1;
};

/** Per-frame bookkeeping for an incremental solve (frame-0 node counts). */
struct IncrementalFrameInfo {
    int reference_frame = 0;      // frame index this frame was solved against (0 = first image)
    bool reference_updated = false; // this frame became the new reference after solving
    int tracked_points = 0;       // frame-0 nodes still tracked after this frame, reported or coasting
    int frame0_points = 0;        // frame-0 nodes that passed the boundary + mask scan
    int taylor_fallbacks = 0;     // nodes sampled by first-order expansion, not bilinear
    int lost_this_frame = 0;      // nodes that stopped being tracked on this frame
};

/**
 * Multi-frame DIC that can move its reference forward (incremental DIC).
 *
 * Each frame is correlated against the current reference on that reference's
 * own grid, and the increment is composed onto the running total:
 *
 *     U(X0, t) = U(X0, k) + u_k( X0 + U(X0, k) )
 *
 * where k is the reference frame and u_k is sampled at the material point's
 * position in frame k (bilinear between solved nodes, first-order expansion
 * near edges). Output keeps run_full_field's frozen layout — 8 floats per
 * point, [X0, Y0, U, V, exx, eyy, exy, corr] — at the frame-0 grid, with
 * Green-Lagrange strain of the total displacement. A frame-0 node that cannot
 * be followed is dropped and never comes back.
 *
 * When the reference moves, the frame just solved is taken over as-is (its
 * image is already prepared for correlation), and the ROI mask is warped into
 * it from frame 0.
 *
 * The metrics array describes the increment solve (the call to the solver on
 * the current reference); frame-0 bookkeeping is in IncrementalFrameInfo.
 *
 * Not thread-safe: one session per sequence, one call at a time. Cancellation
 * via the CancelToken overload (or the process-global flag) leaves the session
 * exactly as it was before the cancelled call.
 */
class IncrementalSession {
public:
    IncrementalSession();
    ~IncrementalSession();
    IncrementalSession(const IncrementalSession&) = delete;
    IncrementalSession& operator=(const IncrementalSession&) = delete;
    IncrementalSession(IncrementalSession&&) noexcept;
    IncrementalSession& operator=(IncrementalSession&&) noexcept;

    /**
     * Start a sequence at `ref_gray` (frame 0). `roi_mask` may be empty.
     * Returns 0, or -2 (step <= 0 / empty grid), -3 (empty image, bad
     * interval), kBadStrainWindow (strain_window too small for step).
     */
    int init(const cv::Mat& ref_gray, const cv::Mat& roi_mask, const IncrementalParams& params);

    /**
     * Solve the next frame. Returns the number of points written (>= 0) or a
     * negative code as run_full_field (-3 also for "not initialized" and for a
     * frame whose size differs from frame 0).
     */
    int process_frame(const cv::Mat& def_gray, float* output, int output_capacity,
                      float* metrics, int metrics_len,
                      ProgressCallback on_progress = nullptr,
                      IncrementalFrameInfo* info = nullptr);

    /** As above, cancelled through `cancel` instead of the process-global flag. */
    int process_frame(const cv::Mat& def_gray, float* output, int output_capacity,
                      float* metrics, int metrics_len, CancelToken& cancel,
                      ProgressCallback on_progress = nullptr,
                      IncrementalFrameInfo* info = nullptr);

    void reset();
    bool initialized() const;
    /** Frames solved so far (frame 0 not counted). */
    int frames_processed() const;
    /** Upper bound on points one frame can return (frame-0 grid node count). */
    int max_points() const;
    void set_debug_dir(std::string dir);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace pipeline
} // namespace Semper

#endif
