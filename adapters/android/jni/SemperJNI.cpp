#include <android/bitmap.h>
#include <jni.h>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <semper/incremental.hpp>
#include <semper/io.hpp>
#include <semper/pipeline.hpp>
#include "jni_log.hpp"

#undef LOG_TAG
#define LOG_TAG "SemperJNI"

using Semper::pipeline::FullFieldParams;
using Semper::pipeline::IncrementalFrameInfo;
using Semper::pipeline::IncrementalParams;
using Semper::pipeline::IncrementalSession;
using Semper::pipeline::ReferenceCache;

static ReferenceCache g_cache;

// The incremental session, when one is open. Guarded by g_cache.mutex like the
// cache: one solve at a time, and setCancelRequested (lock-free, process-global
// flag) still reaches it. Opening a session releases the fixed reference, so
// only one reference image is held at a time.
static std::unique_ptr<IncrementalSession> g_session;
static int g_session_w = 0, g_session_h = 0;

namespace {

// Bridges the solver's progress callback to SemperNativeLib.ProgressCallback.
// The solver calls it from its own progress thread, which attaches itself.
struct ProgressBridge {
    JavaVM* jvm = nullptr;
    jobject globalCallback = nullptr;
    jmethodID methodId = nullptr;

    ProgressBridge(JNIEnv* env, jobject callbackObj) {
        env->GetJavaVM(&jvm);
        if (callbackObj != nullptr) {
            globalCallback = env->NewGlobalRef(callbackObj);
            jclass cls = env->GetObjectClass(callbackObj);
            methodId = env->GetMethodID(cls, "onProgressUpdate", "(I)V");
            env->DeleteLocalRef(cls);
        }
    }
    ProgressBridge(const ProgressBridge&) = delete;
    ProgressBridge& operator=(const ProgressBridge&) = delete;

    // Called on the creating thread once the solve has returned.
    void release(JNIEnv* env) {
        if (globalCallback != nullptr) {
            env->DeleteGlobalRef(globalCallback);
            globalCallback = nullptr;
        }
    }

    void operator()(int percentage) const {
        if (jvm == nullptr || globalCallback == nullptr || methodId == nullptr)
            return;
        JNIEnv* pEnv = nullptr;
        int getEnvStat = jvm->GetEnv((void**)&pEnv, JNI_VERSION_1_6);
        bool attached = false;
        if (getEnvStat == JNI_EDETACHED) {
            if (jvm->AttachCurrentThread(&pEnv, nullptr) != JNI_OK) return;
            attached = true;
        } else if (getEnvStat != JNI_OK) {
            return;
        }
        if (pEnv && !pEnv->ExceptionCheck()) {
            pEnv->CallVoidMethod(globalCallback, methodId, (jint)percentage);
            if (pEnv->ExceptionCheck()) pEnv->ExceptionClear();
        }
        // Do not detach if we were already on a JNI thread (main compute thread).
        // Progress worker attaches itself via GetEnv==DETACHED path; detach only then.
        if (attached) jvm->DetachCurrentThread();
    }
};

// Decode an optional byte array (encoded image, or raw pixels at w x h).
// Returns false only when the array exists but its elements cannot be read.
bool decode_optional(JNIEnv* env, jbyteArray bytes, int w, int h, cv::Mat& out) {
    out.release();
    if (bytes == nullptr || env->GetArrayLength(bytes) <= 0) return true;
    jsize len = env->GetArrayLength(bytes);
    jbyte* buf = env->GetByteArrayElements(bytes, nullptr);
    if (buf == nullptr) return false;
    out = Semper::io::decode_gray(reinterpret_cast<uint8_t*>(buf), (size_t)len, w, h);
    env->ReleaseByteArrayElements(bytes, buf, JNI_ABORT);
    return true;
}

// Metrics slots handed to the solver: all 23 (CONTRACT.md "Metrics buffer").
constexpr int kJniMetricsLen = 23;

int metrics_capacity(JNIEnv* env, jfloatArray out_metrics) {
    if (out_metrics == nullptr) return 0;
    return env->GetArrayLength(out_metrics) >= 16 ? kJniMetricsLen : 0;
}

// Copies the largest prefix the Kotlin array holds: 16, 17, 19, 20, 21, 22 or
// 23 slots. 18 is not a prefix the contract offers (slots 17 and 18 came in
// together), so an 18-float array gets 17.
void copy_metrics(JNIEnv* env, jfloatArray out_metrics, const float* metrics_buf) {
    const int metrics_len = env->GetArrayLength(out_metrics);
    int ncopy = metrics_len >= kJniMetricsLen ? kJniMetricsLen : metrics_len;
    if (ncopy == 18) ncopy = 17;
    env->SetFloatArrayRegion(out_metrics, 0, ncopy, metrics_buf);
}

// Float capacity of a direct ByteBuffer, or -1 if it is not usable.
int output_capacity_floats(JNIEnv* env, jobject outputBuffer, float*& output_ptr) {
    output_ptr = (float*)env->GetDirectBufferAddress(outputBuffer);
    if (!output_ptr) return -1;
    // Bound the solver's write. Each point packs 8 floats; without this the
    // solver would trust output_ptr blindly and a ROI/step combination larger
    // than the Kotlin-side allocation would overflow the direct buffer.
    jlong output_cap_bytes = env->GetDirectBufferCapacity(outputBuffer);
    if (output_cap_bytes < static_cast<jlong>(sizeof(float))) return -1;
    return static_cast<int>(output_cap_bytes / static_cast<jlong>(sizeof(float)));
}

FullFieldParams make_params(jint rectX, jint rectY, jint rectWidth, jint rectHeight,
                            jint step, jint subsetSize, jint strainWindow,
                            jboolean use6x6Interpolator) {
    FullFieldParams params;
    params.rect_x = rectX;
    params.rect_y = rectY;
    params.rect_w = rectWidth;
    params.rect_h = rectHeight;
    params.step = step;
    params.subset_size = subsetSize;
    params.strain_window = strainWindow;
    params.use_6x6_interpolator = use6x6Interpolator == JNI_TRUE;
    return params;
}

} // namespace

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    (void)vm;
    (void)reserved;
    LOGD("Semper Native Library Loaded.");
    cv::setNumThreads(1);
    return JNI_VERSION_1_6;
}

extern "C" {

JNIEXPORT void JNICALL
Java_com_sempermechanics_semper_SemperNativeLib_setDebugOutputDir(
        JNIEnv* env, jobject, jstring debugDir) {
    std::lock_guard<std::mutex> lock(g_cache.mutex);
    if (debugDir == nullptr) {
        g_cache.debug_dir.clear();
        return;
    }
    const char* dir = env->GetStringUTFChars(debugDir, nullptr);
    g_cache.debug_dir = std::string(dir);
    env->ReleaseStringUTFChars(debugDir, dir);
}

/**
 * Ask the running solve to stop, or clear the flag before starting one.
 *
 * Deliberately takes no lock: computeFullFieldDirect holds g_cache.mutex for the
 * whole solve, so a cancel that waited for it could never arrive in time — which
 * is the entire point of the call.
 */
JNIEXPORT void JNICALL
Java_com_sempermechanics_semper_SemperNativeLib_setCancelRequested(
        JNIEnv*, jobject, jboolean cancel) {
    if (cancel == JNI_TRUE) {
        Semper::pipeline::request_cancel();
    } else {
        Semper::pipeline::clear_cancel();
    }
}

JNIEXPORT jobject JNICALL
Java_com_sempermechanics_semper_SemperNativeLib_getPreviewFromBytes(
        JNIEnv* env, jobject, jbyteArray fileData, jint targetWidth) {
    if (fileData == nullptr) return nullptr;
    jsize len = env->GetArrayLength(fileData);
    jbyte* buf = env->GetByteArrayElements(fileData, nullptr);
    if (buf == nullptr) return nullptr;
    cv::Mat fullImg = Semper::io::decode_bgr(reinterpret_cast<uint8_t*>(buf), (size_t)len);
    env->ReleaseByteArrayElements(fileData, buf, JNI_ABORT);
    if (fullImg.empty()) return nullptr;

    float ratio = (float)targetWidth / fullImg.cols;
    int targetHeight = (int)(fullImg.rows * ratio);
    cv::Mat resizedImg;
    cv::resize(fullImg, resizedImg, cv::Size(targetWidth, targetHeight));

    jclass bitmapCls = env->FindClass("android/graphics/Bitmap");
    jmethodID createBitmapMethod = env->GetStaticMethodID(
            bitmapCls, "createBitmap",
            "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");
    jclass configCls = env->FindClass("android/graphics/Bitmap$Config");
    jfieldID argb8888Field = env->GetStaticFieldID(
            configCls, "ARGB_8888", "Landroid/graphics/Bitmap$Config;");
    jobject config = env->GetStaticObjectField(configCls, argb8888Field);
    jobject jBitmap = env->CallStaticObjectMethod(
            bitmapCls, createBitmapMethod, targetWidth, targetHeight, config);

    void* pixels;
    if (AndroidBitmap_lockPixels(env, jBitmap, &pixels) < 0) return nullptr;
    cv::cvtColor(resizedImg, resizedImg, cv::COLOR_BGR2RGBA);
    memcpy(pixels, resizedImg.data, resizedImg.total() * resizedImg.elemSize());
    AndroidBitmap_unlockPixels(env, jBitmap);
    return jBitmap;
}

JNIEXPORT jintArray JNICALL
Java_com_sempermechanics_semper_SemperNativeLib_getImageDimensions(
        JNIEnv* env, jobject, jbyteArray fileData) {
    int w = 0, h = 0;
    if (fileData != nullptr) {
        jsize len = env->GetArrayLength(fileData);
        jbyte* buf = env->GetByteArrayElements(fileData, nullptr);
        if (buf != nullptr) {
            Semper::io::image_dimensions(reinterpret_cast<uint8_t*>(buf), (size_t)len, w, h);
            env->ReleaseByteArrayElements(fileData, buf, JNI_ABORT);
        }
    }
    jintArray result = env->NewIntArray(2);
    jint temp[] = {w, h};
    env->SetIntArrayRegion(result, 0, 2, temp);
    return result;
}

JNIEXPORT void JNICALL
Java_com_sempermechanics_semper_SemperNativeLib_initializeReference(
        JNIEnv* env, jobject, jbyteArray refBytes, jbyteArray maskBytes,
        jint width, jint height) {
    std::lock_guard<std::mutex> lock(g_cache.mutex);
    // Back to the fixed-reference path: an incremental session left open
    // would hold its two images until the next initializeIncremental.
    g_session.reset();
    g_session_w = g_session_h = 0;
    if (refBytes == nullptr) {
        g_cache.reset();
        return;
    }
    jsize len = env->GetArrayLength(refBytes);
    jbyte* buf = env->GetByteArrayElements(refBytes, nullptr);
    if (buf == nullptr) { g_cache.reset(); return; }
    cv::Mat refMat = Semper::io::decode_gray(
            reinterpret_cast<uint8_t*>(buf), (size_t)len, width, height);
    env->ReleaseByteArrayElements(refBytes, buf, JNI_ABORT);

    cv::Mat roiMask;
    if (maskBytes != nullptr && env->GetArrayLength(maskBytes) > 0) {
        jsize mlen = env->GetArrayLength(maskBytes);
        jbyte* mbuf = env->GetByteArrayElements(maskBytes, nullptr);
        if (mbuf == nullptr) { g_cache.reset(); return; }
        roiMask = Semper::io::decode_gray(
                reinterpret_cast<uint8_t*>(mbuf), (size_t)mlen,
                refMat.cols, refMat.rows);
        env->ReleaseByteArrayElements(maskBytes, mbuf, JNI_ABORT);
    }
    g_cache.set_from_gray(refMat, roiMask);
}

JNIEXPORT jint JNICALL
Java_com_sempermechanics_semper_SemperNativeLib_computeFullFieldDirect(
        JNIEnv* env, jobject, jbyteArray refBytes, jbyteArray defBytes,
        jbyteArray maskBytes, jint rectX, jint rectY, jint rectWidth,
        jint rectHeight, jint step, jint subsetSize, jint strainWindow,
        jboolean use6x6Interpolator, jobject outputBuffer, jobject callbackObj,
        jfloatArray out_metrics) {
    (void)refBytes;  // reference comes from initializeReference cache
    std::lock_guard<std::mutex> lock(g_cache.mutex);

    if (env == nullptr || defBytes == nullptr || outputBuffer == nullptr)
        return -3;
    float* output_ptr = nullptr;
    const int output_capacity = output_capacity_floats(env, outputBuffer, output_ptr);
    if (output_capacity < 0) return -3;

    cv::Mat defMat;
    if (!decode_optional(env, defBytes, g_cache.width, g_cache.height, defMat)) return -3;
    if (defMat.empty()) return -3;

    cv::Mat roiMask;
    if (!decode_optional(env, maskBytes, g_cache.width, g_cache.height, roiMask)) return -3;

    const FullFieldParams params = make_params(rectX, rectY, rectWidth, rectHeight,
                                               step, subsetSize, strainWindow, use6x6Interpolator);

    ProgressBridge progress(env, callbackObj);
    float metrics_buf[kJniMetricsLen] = {0};
    const int metrics_len = metrics_capacity(env, out_metrics);

    int result = Semper::pipeline::run_full_field(
            g_cache, defMat, roiMask, params, output_ptr, output_capacity,
            metrics_len ? metrics_buf : nullptr, metrics_len,
            [&progress](int pct) { progress(pct); });

    progress.release(env);
    if (metrics_len) copy_metrics(env, out_metrics, metrics_buf);
    return result;
}

/**
 * Open an incremental (updated-reference) DIC session at the first image.
 * `updateInterval`: 0 = fixed reference, N >= 1 = move the reference to the
 * latest solved frame every N frames. Releases the fixed-reference cache.
 * Returns 0 or a negative code (-2 ROI, -3 init, -4 strain window).
 */
JNIEXPORT jint JNICALL
Java_com_sempermechanics_semper_SemperNativeLib_initializeIncremental(
        JNIEnv* env, jobject, jbyteArray refBytes, jbyteArray maskBytes,
        jint width, jint height, jint rectX, jint rectY, jint rectWidth,
        jint rectHeight, jint step, jint subsetSize, jint strainWindow,
        jboolean use6x6Interpolator, jint updateInterval) {
    std::lock_guard<std::mutex> lock(g_cache.mutex);
    g_cache.reset();
    g_session.reset();
    g_session_w = g_session_h = 0;
    if (refBytes == nullptr) return -3;

    cv::Mat refMat;
    if (!decode_optional(env, refBytes, width, height, refMat) || refMat.empty()) return -3;
    cv::Mat roiMask;
    if (!decode_optional(env, maskBytes, refMat.cols, refMat.rows, roiMask)) return -3;

    IncrementalParams ip;
    ip.field = make_params(rectX, rectY, rectWidth, rectHeight, step, subsetSize,
                           strainWindow, use6x6Interpolator);
    ip.update_interval = updateInterval;
    auto session = std::make_unique<IncrementalSession>();
    const int rc = session->init(refMat, roiMask, ip);
    if (rc != 0) return rc;
    g_session = std::move(session);
    g_session_w = refMat.cols;
    g_session_h = refMat.rows;
    return 0;
}

/**
 * Solve the next frame of the open incremental session. Output, progress and
 * metrics as computeFullFieldDirect; results are the total displacement from
 * the first image at its grid points. `outInfo` (may be null, length >= 6)
 * receives IncrementalFrameInfo: reference frame, reference updated (0/1),
 * tracked points, frame-0 points, Taylor fallbacks, points lost this frame.
 */
JNIEXPORT jint JNICALL
Java_com_sempermechanics_semper_SemperNativeLib_computeIncrementalFrame(
        JNIEnv* env, jobject, jbyteArray defBytes, jobject outputBuffer,
        jobject callbackObj, jfloatArray out_metrics, jintArray out_info) {
    std::lock_guard<std::mutex> lock(g_cache.mutex);
    if (env == nullptr || defBytes == nullptr || outputBuffer == nullptr || !g_session)
        return -3;
    float* output_ptr = nullptr;
    const int output_capacity = output_capacity_floats(env, outputBuffer, output_ptr);
    if (output_capacity < 0) return -3;

    cv::Mat defMat;
    if (!decode_optional(env, defBytes, g_session_w, g_session_h, defMat)) return -3;
    if (defMat.empty()) return -3;

    // setDebugOutputDir arms one solve, as it does for computeFullFieldDirect.
    if (!g_cache.debug_dir.empty()) {
        g_session->set_debug_dir(g_cache.debug_dir);
        g_cache.debug_dir.clear();
    }

    ProgressBridge progress(env, callbackObj);
    float metrics_buf[kJniMetricsLen] = {0};
    const int metrics_len = metrics_capacity(env, out_metrics);
    IncrementalFrameInfo info;

    int result = g_session->process_frame(
            defMat, output_ptr, output_capacity,
            metrics_len ? metrics_buf : nullptr, metrics_len,
            [&progress](int pct) { progress(pct); }, &info);
    g_session->set_debug_dir(std::string());

    progress.release(env);
    if (metrics_len) copy_metrics(env, out_metrics, metrics_buf);
    if (out_info != nullptr && env->GetArrayLength(out_info) >= 6) {
        const jint values[6] = {info.reference_frame, info.reference_updated ? 1 : 0,
                                info.tracked_points, info.frame0_points,
                                info.taylor_fallbacks, info.lost_this_frame};
        env->SetIntArrayRegion(out_info, 0, 6, values);
    }
    return result;
}

/** Close the incremental session and free its reference. */
JNIEXPORT void JNICALL
Java_com_sempermechanics_semper_SemperNativeLib_releaseIncremental(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lock(g_cache.mutex);
    g_session.reset();
    g_session_w = g_session_h = 0;
}

// Legacy package (com.indicvision.semper). The app is moving to
// com.sempermechanics.semper; these forwarders let a build under either package
// link against this engine during the move. Remove them once no app or fork
// ships SemperNativeLib under the old package.

JNIEXPORT void JNICALL
Java_com_indicvision_semper_SemperNativeLib_setDebugOutputDir(
        JNIEnv* env, jobject self, jstring debugDir) {
    Java_com_sempermechanics_semper_SemperNativeLib_setDebugOutputDir(env, self, debugDir);
}

JNIEXPORT void JNICALL
Java_com_indicvision_semper_SemperNativeLib_setCancelRequested(
        JNIEnv* env, jobject self, jboolean cancel) {
    Java_com_sempermechanics_semper_SemperNativeLib_setCancelRequested(env, self, cancel);
}

JNIEXPORT jobject JNICALL
Java_com_indicvision_semper_SemperNativeLib_getPreviewFromBytes(
        JNIEnv* env, jobject self, jbyteArray fileData, jint targetWidth) {
    return Java_com_sempermechanics_semper_SemperNativeLib_getPreviewFromBytes(
            env, self, fileData, targetWidth);
}

JNIEXPORT jintArray JNICALL
Java_com_indicvision_semper_SemperNativeLib_getImageDimensions(
        JNIEnv* env, jobject self, jbyteArray fileData) {
    return Java_com_sempermechanics_semper_SemperNativeLib_getImageDimensions(env, self, fileData);
}

JNIEXPORT void JNICALL
Java_com_indicvision_semper_SemperNativeLib_initializeReference(
        JNIEnv* env, jobject self, jbyteArray refBytes, jbyteArray maskBytes,
        jint width, jint height) {
    Java_com_sempermechanics_semper_SemperNativeLib_initializeReference(
            env, self, refBytes, maskBytes, width, height);
}

JNIEXPORT jint JNICALL
Java_com_indicvision_semper_SemperNativeLib_computeFullFieldDirect(
        JNIEnv* env, jobject self, jbyteArray refBytes, jbyteArray defBytes,
        jbyteArray maskBytes, jint rectX, jint rectY, jint rectWidth,
        jint rectHeight, jint step, jint subsetSize, jint strainWindow,
        jboolean use6x6Interpolator, jobject outputBuffer, jobject callbackObj,
        jfloatArray out_metrics) {
    return Java_com_sempermechanics_semper_SemperNativeLib_computeFullFieldDirect(
            env, self, refBytes, defBytes, maskBytes, rectX, rectY, rectWidth, rectHeight,
            step, subsetSize, strainWindow, use6x6Interpolator, outputBuffer, callbackObj,
            out_metrics);
}

}  // extern "C"
