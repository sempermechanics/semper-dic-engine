// C ABI implementation — a thin translation over the C++ core. No algorithm here.
#include <semper/semper_c.h>

#include <semper/guarded_call.hpp>
#include <semper/incremental.hpp>
#include <semper/io.hpp>
#include <semper/pipeline.hpp>
#include <semper/version.hpp>
#include <opencv2/core.hpp>


using Semper::guarded_call;
using Semper::pipeline::CancelToken;
using Semper::pipeline::FullFieldParams;
using Semper::pipeline::IncrementalFrameInfo;
using Semper::pipeline::IncrementalParams;
using Semper::pipeline::IncrementalSession;
using Semper::pipeline::ReferenceCache;

// One engine instance owns a reference cache and its own cancel token, so several
// instances in a process cancel independently (see docs/CONTRACT.md §A.2).
struct semper_engine {
    ReferenceCache cache;
    CancelToken cancel;
};

struct semper_incremental {
    IncrementalSession session;
    CancelToken cancel;
    int width = 0, height = 0;   // frames are decoded against the first image's size
};

namespace {

FullFieldParams to_params(const semper_params* params) {
    FullFieldParams p;
    p.rect_x = params->rect_x;
    p.rect_y = params->rect_y;
    p.rect_w = params->rect_w;
    p.rect_h = params->rect_h;
    p.step = params->step;
    p.subset_size = params->subset_size;
    p.strain_window = params->strain_window;
    p.use_6x6_interpolator = params->use_6x6_interpolator != 0;
    return p;
}

// No mask bytes is no mask; mask bytes that do not decode are an error. Solving
// without them would report a field over the regions the caller excluded.
bool decode_mask(const uint8_t* mask, size_t mask_len, int w, int h, cv::Mat& roi) {
    if (mask == nullptr || mask_len == 0) return true;
    roi = Semper::io::decode_gray(mask, mask_len, w, h);
    return !roi.empty();
}

Semper::pipeline::ProgressCallback to_progress(semper_progress_cb cb, void* user) {
    Semper::pipeline::ProgressCallback progress;
    if (cb != nullptr) {
        progress = [cb, user](int pct) { cb(pct, user); };
    }
    return progress;
}

} // namespace

// Every entry point below runs its body through guarded_call: an exception
// that reached this extern "C" frame would terminate the caller's process,
// so it becomes SEMPER_ERR_INIT (or NULL from a constructor) instead.

extern "C" {

SEMPER_C_API semper_engine* semper_create(void) {
    return guarded_call<semper_engine*>(nullptr, [] { return new semper_engine(); });
}

SEMPER_C_API void semper_destroy(semper_engine* eng) {
    delete eng;
}

SEMPER_C_API int semper_set_reference(semper_engine* eng,
                         const uint8_t* img, size_t img_len,
                         const uint8_t* mask, size_t mask_len,
                         int expected_w, int expected_h) {
    if (eng == nullptr || img == nullptr || img_len == 0) return SEMPER_ERR_INIT;
    return guarded_call(SEMPER_ERR_INIT, [&] {
        cv::Mat ref = Semper::io::decode_gray(img, img_len, expected_w, expected_h);
        if (ref.empty()) return SEMPER_ERR_INIT;
        cv::Mat roi;
        if (!decode_mask(mask, mask_len, ref.cols, ref.rows, roi)) return SEMPER_ERR_INIT;
        eng->cache.set_from_gray(ref, roi);
        return 0;
    });
}

SEMPER_C_API int semper_run(semper_engine* eng,
               const uint8_t* def, size_t def_len,
               const uint8_t* mask, size_t mask_len,
               const semper_params* params,
               float* out, int out_capacity,
               float* metrics, int metrics_len,
               semper_progress_cb cb, void* user) {
    if (eng == nullptr || def == nullptr || def_len == 0 || params == nullptr || out == nullptr)
        return SEMPER_ERR_INIT;
    return guarded_call(SEMPER_ERR_INIT, [&] {
        // Decode against the cached reference's dimensions, matching the Android path.
        cv::Mat def_gray = Semper::io::decode_gray(def, def_len, eng->cache.width, eng->cache.height);
        if (def_gray.empty()) return SEMPER_ERR_INIT;
        cv::Mat roi;
        if (!decode_mask(mask, mask_len, eng->cache.width, eng->cache.height, roi))
            return SEMPER_ERR_INIT;

        return Semper::pipeline::run_full_field(
            eng->cache, def_gray, roi, to_params(params), out, out_capacity,
            metrics, metrics_len, eng->cancel, to_progress(cb, user));
    });
}

SEMPER_C_API void semper_cancel(semper_engine* eng) {
    if (eng != nullptr) eng->cancel.request();
}

SEMPER_C_API semper_incremental* semper_incremental_create(void) {
    // Not new (std::nothrow): that covers the allocation only, and the
    // session's constructor allocates too.
    return guarded_call<semper_incremental*>(nullptr, [] { return new semper_incremental(); });
}

SEMPER_C_API void semper_incremental_destroy(semper_incremental* inc) {
    delete inc;
}

SEMPER_C_API int semper_incremental_init(semper_incremental* inc,
                         const uint8_t* ref, size_t ref_len,
                         const uint8_t* mask, size_t mask_len,
                         int expected_w, int expected_h,
                         const semper_params* params,
                         int32_t update_interval) {
    if (inc == nullptr || ref == nullptr || ref_len == 0 || params == nullptr) return SEMPER_ERR_INIT;
    return guarded_call(SEMPER_ERR_INIT, [&] {
        cv::Mat ref_gray = Semper::io::decode_gray(ref, ref_len, expected_w, expected_h);
        if (ref_gray.empty()) return SEMPER_ERR_INIT;
        cv::Mat roi;
        if (!decode_mask(mask, mask_len, ref_gray.cols, ref_gray.rows, roi)) return SEMPER_ERR_INIT;
        IncrementalParams ip;
        ip.field = to_params(params);
        ip.update_interval = update_interval;
        const int rc = inc->session.init(ref_gray, roi, ip);
        inc->width = rc == 0 ? ref_gray.cols : 0;
        inc->height = rc == 0 ? ref_gray.rows : 0;
        return rc;
    });
}

SEMPER_C_API int semper_incremental_process(semper_incremental* inc,
                         const uint8_t* def, size_t def_len,
                         float* out, int out_capacity,
                         float* metrics, int metrics_len,
                         semper_incremental_info* info,
                         semper_progress_cb cb, void* user) {
    if (inc == nullptr || def == nullptr || def_len == 0 || out == nullptr) return SEMPER_ERR_INIT;
    if (!inc->session.initialized()) return SEMPER_ERR_INIT;
    return guarded_call(SEMPER_ERR_INIT, [&] {
        cv::Mat def_gray = Semper::io::decode_gray(def, def_len, inc->width, inc->height);
        if (def_gray.empty()) return SEMPER_ERR_INIT;

        IncrementalFrameInfo fi;
        const int rc = inc->session.process_frame(def_gray, out, out_capacity, metrics, metrics_len,
                                                  inc->cancel, to_progress(cb, user), &fi);
        if (info != nullptr) {
            info->reference_frame = fi.reference_frame;
            info->reference_updated = fi.reference_updated ? 1 : 0;
            info->tracked_points = fi.tracked_points;
            info->frame0_points = fi.frame0_points;
            info->taylor_fallbacks = fi.taylor_fallbacks;
            info->lost_this_frame = fi.lost_this_frame;
        }
        return rc;
    });
}

SEMPER_C_API void semper_incremental_cancel(semper_incremental* inc) {
    if (inc != nullptr) inc->cancel.request();
}

SEMPER_C_API const char* semper_version(void) {
    return SEMPER_VERSION_STRING;
}

} // extern "C"
