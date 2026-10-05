#include <android/bitmap.h>
#include <jni.h>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <semper/io.hpp>
#include <semper/pipeline.hpp>
#include "jni_log.hpp"

#undef LOG_TAG
#define LOG_TAG "SemperJNI"

using Semper::pipeline::FullFieldParams;
using Semper::pipeline::ReferenceCache;

static ReferenceCache g_cache;

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
    float* output_ptr = (float*)env->GetDirectBufferAddress(outputBuffer);
    if (!output_ptr) return -3;
    // Bound the solver's write. Each point packs 8 floats; without this the
    // solver would trust output_ptr blindly and a ROI/step combination larger
    // than the Kotlin-side allocation would overflow the direct buffer.
    jlong output_cap_bytes = env->GetDirectBufferCapacity(outputBuffer);
    if (output_cap_bytes < static_cast<jlong>(sizeof(float))) return -3;
    int output_capacity = static_cast<int>(output_cap_bytes / static_cast<jlong>(sizeof(float)));

    jsize dlen = env->GetArrayLength(defBytes);
    jbyte* dbuf = env->GetByteArrayElements(defBytes, nullptr);
    if (dbuf == nullptr) return -3;
    cv::Mat defMat = Semper::io::decode_gray(
            reinterpret_cast<uint8_t*>(dbuf), (size_t)dlen,
            g_cache.width, g_cache.height);
    env->ReleaseByteArrayElements(defBytes, dbuf, JNI_ABORT);
    if (defMat.empty()) return -3;

    cv::Mat roiMask;
    if (maskBytes != nullptr && env->GetArrayLength(maskBytes) > 0) {
        jsize mlen = env->GetArrayLength(maskBytes);
        jbyte* mbuf = env->GetByteArrayElements(maskBytes, nullptr);
        if (mbuf == nullptr) return -3;
        roiMask = Semper::io::decode_gray(
                reinterpret_cast<uint8_t*>(mbuf), (size_t)mlen,
                g_cache.width, g_cache.height);
        env->ReleaseByteArrayElements(maskBytes, mbuf, JNI_ABORT);
    }

    FullFieldParams params;
    params.rect_x = rectX;
    params.rect_y = rectY;
    params.rect_w = rectWidth;
    params.rect_h = rectHeight;
    params.step = step;
    params.subset_size = subsetSize;
    params.strain_window = strainWindow;
    params.use_6x6_interpolator = use6x6Interpolator == JNI_TRUE;

    JavaVM* jvm = nullptr;
    env->GetJavaVM(&jvm);
    jobject globalCallback = nullptr;
    jmethodID methodId = nullptr;
    if (callbackObj != nullptr) {
        globalCallback = env->NewGlobalRef(callbackObj);
        jclass cls = env->GetObjectClass(callbackObj);
        methodId = env->GetMethodID(cls, "onProgressUpdate", "(I)V");
        env->DeleteLocalRef(cls);
    }

    auto on_progress = [&](int percentage) {
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
    };

    float metrics_buf[19] = {0};
    float* metrics_ptr = nullptr;
    int metrics_len = 0;
    if (out_metrics != nullptr) {
        metrics_len = env->GetArrayLength(out_metrics);
        if (metrics_len >= 16) metrics_ptr = metrics_buf;
    }

    int result = Semper::pipeline::run_full_field(
            g_cache, defMat, roiMask, params, output_ptr, output_capacity,
            metrics_ptr, metrics_ptr ? 19 : 0, on_progress);

    if (globalCallback != nullptr) {
        env->DeleteGlobalRef(globalCallback);
    }
    if (metrics_ptr != nullptr && out_metrics != nullptr) {
        int ncopy = (metrics_len >= 19) ? 19 : (metrics_len >= 17) ? 17 : 16;
        env->SetFloatArrayRegion(out_metrics, 0, ncopy, metrics_buf);
    }
    return result;
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
