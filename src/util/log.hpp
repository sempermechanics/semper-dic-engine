#ifndef SEMPER_LOG_HPP
#define SEMPER_LOG_HPP

// Portable logging for the DIC engine.
// Android adapter builds define SEMPER_ANDROID_LOG; host builds get stderr,
// LOGE always and LOGD only with SEMPER_HOST_VERBOSE (CMake
// SEMPER_VERBOSE_LOG; dic_tests turn it on).

#ifndef LOG_TAG
#define LOG_TAG "Semper"
#endif

#if defined(SEMPER_ANDROID_LOG)
#include <android/log.h>
#ifndef LOGD
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#endif
#ifndef LOGE
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#endif
#elif defined(SEMPER_HOST_LOG)
#include <cstdio>
#ifndef LOGD
#if defined(SEMPER_HOST_VERBOSE)
#define LOGD(...) do { std::fprintf(stderr, "[%s] ", LOG_TAG); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)
#else
// Never runs, but still uses its arguments: a value kept only for the trace
// must not turn into an unused-variable warning (-Werror builds).
#define LOGD(...) do { if (false) std::fprintf(stderr, __VA_ARGS__); } while (0)
#endif
#endif
#ifndef LOGE
#define LOGE(...) do { std::fprintf(stderr, "[%s] ", LOG_TAG); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)
#endif
#else
#ifndef LOGD
#define LOGD(...) ((void)0)
#endif
#ifndef LOGE
#define LOGE(...) ((void)0)
#endif
#endif

#endif // SEMPER_LOG_HPP
