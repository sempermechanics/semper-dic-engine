#ifndef SEMPER_PIPELINE_HPP
#define SEMPER_PIPELINE_HPP

#include <semper/cancel.hpp>
#include <semper/image.hpp>
#include <opencv2/core.hpp>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace Semper {
namespace pipeline {

using ProgressCallback = std::function<void(int percentage)>;

/**
 * Cached reference for multi-frame solves (decoded gray + Image).
 *
 * Owns a raw Image* (freed in reset()), so it is non-copyable/non-movable to
 * avoid a double-free — there is one process-wide instance (see the JNI layer).
 * The embedded `mutex` does NOT make the struct's methods thread-safe; it is
 * held by the JNI caller around a whole solve so a cancel can still interrupt.
 */
struct ReferenceCache {
    Image* ref_img = nullptr;
    int width = 0;
    int height = 0;
    cv::Mat gray;
    std::mutex mutex;
    std::string debug_dir;

    ReferenceCache() = default;
    ~ReferenceCache() { reset(); }
    ReferenceCache(const ReferenceCache&) = delete;
    ReferenceCache& operator=(const ReferenceCache&) = delete;
    ReferenceCache(ReferenceCache&&) = delete;
    ReferenceCache& operator=(ReferenceCache&&) = delete;

    void reset();
    void set_from_gray(const cv::Mat& gray_in, const cv::Mat& roi_mask);
};

struct FullFieldParams {
    int rect_x = 0, rect_y = 0, rect_w = 0, rect_h = 0;
    int step = 0;
    int subset_size = 0;
    int strain_window = 0;
    bool use_6x6_interpolator = false;
};

/**
 * Run the hybrid full-field DIC pipeline.
 * @return number of valid output points, or negative error code
 *   (-2 ROI, -3 init, [kBadStrainWindow] if strain_window is too small for
 *   step, [kCancelled] if cancelled mid-solve).
 * Writes packed points to output_ptr (8 floats each: x,y,u,v,exx,eyy,exy,corr).
 * output_capacity is the number of floats output_ptr can hold; the solver never
 * writes past it (points beyond the capacity are dropped rather than overflowing).
 * If metrics != nullptr and metrics_len >= 16, fills engine telemetry (23 floats preferred).
 */
int run_full_field(
    ReferenceCache& cache,
    const cv::Mat& def_gray,
    const cv::Mat& roi_mask,
    const FullFieldParams& params,
    float* output_ptr,
    int output_capacity,
    float* metrics,
    int metrics_len,
    ProgressCallback on_progress = nullptr);

/**
 * Re-entrant overload: cancellation is driven by the caller-owned [cancel] token
 * instead of the process-global flag, so an embedded engine instance (SDK / Python)
 * can stop its own solve without touching any other. Behaviour is otherwise
 * identical to the overload above; the token is bound as the active cancel target
 * for the duration of the call (and cleared on entry, like the global flag).
 */
int run_full_field(
    ReferenceCache& cache,
    const cv::Mat& def_gray,
    const cv::Mat& roi_mask,
    const FullFieldParams& params,
    float* output_ptr,
    int output_capacity,
    float* metrics,
    int metrics_len,
    CancelToken& cancel,
    ProgressCallback on_progress = nullptr);

} // namespace pipeline
} // namespace Semper

#endif
