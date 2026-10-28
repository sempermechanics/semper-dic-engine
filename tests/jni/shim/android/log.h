// Host stand-in for the NDK's <android/log.h>, for the JNI adapter harness
// (tests/jni/test_jni_adapter.cpp) only. Declares the one call
// adapters/android/jni/jni_log.hpp makes; the harness defines it (stderr).
#ifndef SEMPER_TEST_SHIM_ANDROID_LOG_H
#define SEMPER_TEST_SHIM_ANDROID_LOG_H

#ifdef __cplusplus
extern "C" {
#endif

// The NDK's values.
enum android_LogPriority {
    ANDROID_LOG_UNKNOWN = 0,
    ANDROID_LOG_DEFAULT,
    ANDROID_LOG_VERBOSE,
    ANDROID_LOG_DEBUG,
    ANDROID_LOG_INFO,
    ANDROID_LOG_WARN,
    ANDROID_LOG_ERROR,
    ANDROID_LOG_FATAL,
    ANDROID_LOG_SILENT,
};

int __android_log_print(int prio, const char* tag, const char* fmt, ...)
#if defined(__GNUC__)
        __attribute__((format(printf, 3, 4)))
#endif
        ;

#ifdef __cplusplus
}
#endif

#endif  // SEMPER_TEST_SHIM_ANDROID_LOG_H
