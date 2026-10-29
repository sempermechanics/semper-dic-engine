// Host stand-in for the NDK's <android/bitmap.h>, for the JNI adapter harness
// (tests/jni/test_jni_adapter.cpp) only: the struct, constants and three calls
// adapters/android/jni/SemperJNI.cpp uses, with the NDK's layout and values.
// The harness defines the calls over its fake Bitmap objects.
#ifndef SEMPER_TEST_SHIM_ANDROID_BITMAP_H
#define SEMPER_TEST_SHIM_ANDROID_BITMAP_H

#include <jni.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    ANDROID_BITMAP_RESULT_SUCCESS = 0,
    ANDROID_BITMAP_RESULT_BAD_PARAMETER = -1,
    ANDROID_BITMAP_RESULT_JNI_EXCEPTION = -2,
    ANDROID_BITMAP_RESULT_ALLOCATION_FAILED = -3,
};

enum AndroidBitmapFormat {
    ANDROID_BITMAP_FORMAT_NONE = 0,
    ANDROID_BITMAP_FORMAT_RGBA_8888 = 1,
    ANDROID_BITMAP_FORMAT_RGB_565 = 4,
    ANDROID_BITMAP_FORMAT_RGBA_4444 = 7,
    ANDROID_BITMAP_FORMAT_A_8 = 8,
};

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    int32_t format;
    uint32_t flags;
} AndroidBitmapInfo;

int AndroidBitmap_getInfo(JNIEnv* env, jobject jbitmap, AndroidBitmapInfo* info);
int AndroidBitmap_lockPixels(JNIEnv* env, jobject jbitmap, void** addrPtr);
int AndroidBitmap_unlockPixels(JNIEnv* env, jobject jbitmap);

#ifdef __cplusplus
}
#endif

#endif  // SEMPER_TEST_SHIM_ANDROID_BITMAP_H
