// The Android JNI adapter (adapters/android/jni/SemperJNI.cpp) on the host.
//
// The adapter is the C++ between Kotlin and the engine: it pins and releases
// Java arrays, bounds the solver's write by the direct buffer's capacity,
// copies a metrics prefix, bridges progress onto a thread it attaches, and
// turns exceptions into error codes. None of it ran under a sanitizer, since
// it only built with the NDK. Here it is compiled into dic_tests against the
// NDK's own jni.h (copied out of the sysroot by tests/CMakeLists.txt) and
// driven through a fake JNIEnv / JavaVM, so the ASan + UBSan and TSan jobs
// see it. Stand-ins for <android/log.h> and <android/bitmap.h> are in
// tests/jni/shim/android.
//
// The fake is strict where a real VM would be: element pointers are fresh
// heap copies of exactly the array's length (so ASan bounds any over-read)
// that must be released with the pointer that was handed out, the direct
// buffer is exactly its stated capacity, global refs and thread attachments
// are counted, and an out-of-range array region is recorded as an error.
// Every result is compared with the engine called directly.
#include "framework/test_framework.h"

#if defined(DIC_HAVE_OPENCV) && defined(SEMPER_JNI_HARNESS)

#include "framework/sequence.h"
#include "pipeline/full_field_internal.hpp"

#include <android/bitmap.h>
#include <jni.h>
#include <semper/cancel.hpp>
#include <semper/incremental.hpp>
#include <semper/pipeline.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// The adapter's exports. It has no header: SemperNativeLib.kt declares them.
#define SNL(name) Java_com_sempermechanics_semper_SemperNativeLib_##name
extern "C" {
void SNL(setDebugOutputDir)(JNIEnv*, jobject, jstring);
void SNL(setCancelRequested)(JNIEnv*, jobject, jboolean);
jobject SNL(getPreviewFromBytes)(JNIEnv*, jobject, jbyteArray, jint);
jintArray SNL(getImageDimensions)(JNIEnv*, jobject, jbyteArray);
void SNL(initializeReference)(JNIEnv*, jobject, jbyteArray, jbyteArray, jint, jint);
jint SNL(computeFullFieldDirect)(JNIEnv*, jobject, jbyteArray, jbyteArray, jbyteArray, jint,
                                 jint, jint, jint, jint, jint, jint, jboolean, jobject, jobject,
                                 jfloatArray);
jint SNL(initializeIncremental)(JNIEnv*, jobject, jbyteArray, jbyteArray, jint, jint, jint, jint,
                                jint, jint, jint, jint, jint, jboolean, jint);
jint SNL(computeIncrementalFrame)(JNIEnv*, jobject, jbyteArray, jobject, jobject, jfloatArray,
                                  jintArray);
void SNL(releaseIncremental)(JNIEnv*, jobject);
void Java_com_indicvision_semper_SemperNativeLib_initializeReference(JNIEnv*, jobject, jbyteArray,
                                                                     jbyteArray, jint, jint);
jint Java_com_indicvision_semper_SemperNativeLib_computeFullFieldDirect(
        JNIEnv*, jobject, jbyteArray, jbyteArray, jbyteArray, jint, jint, jint, jint, jint, jint,
        jint, jboolean, jobject, jobject, jfloatArray);
}

using Semper::pipeline::FullFieldParams;
using Semper::pipeline::IncrementalFrameInfo;
using Semper::pipeline::IncrementalParams;
using Semper::pipeline::IncrementalSession;
using Semper::pipeline::ReferenceCache;
using Semper::pipeline::kMetricsLen;

// ---------------------------------------------------------------------------
// The fake VM
// ---------------------------------------------------------------------------
namespace fakejni {

enum class Kind { ByteArray, FloatArray, IntArray, Buffer, String, Callback, Bitmap, Other };

struct ByteArray : _jbyteArray {
    std::vector<jbyte> data;
    bool refuse = false;              // GetByteArrayElements returns null (OOM)
    std::vector<jbyte*> pinned;       // handed out, not yet released
};
struct FloatArray : _jfloatArray {
    std::vector<jfloat> data;
};
struct IntArray : _jintArray {
    std::vector<jint> data;
};
struct Buffer : _jobject {
    std::vector<float> data;          // exactly capacity_bytes / 4 floats
    jlong capacity_bytes = 0;
    bool direct = true;
};
struct String : _jstring {
    std::string text;
};
struct Callback : _jobject {
    bool has_method = true;           // onProgressUpdate(I)V exists
    std::function<void(int)> on_call; // run on the calling thread, unlocked
    std::mutex m;
    std::vector<int> calls;
};
struct Bitmap : _jobject {
    int w = 0, h = 0;
    uint32_t stride = 0;
    int32_t format = ANDROID_BITMAP_FORMAT_RGBA_8888;
    std::vector<uint8_t> px;
    int locks = 0;
};
struct Klass : _jclass {};
struct Plain : _jobject {};

constexpr uint8_t kBitmapFill = 0xA5;   // unwritten bitmap bytes, row padding included
constexpr int kBitmapPadding = 12;      // bytes past width * 4 in every row

struct State {
    std::mutex m;
    std::map<const _jobject*, Kind> kinds;
    std::atomic<int> global_refs{0};
    std::atomic<int> attaches{0}, detaches{0};
    std::atomic<int> utf_held{0};
    std::atomic<int> region_errors{0};
    std::atomic<bool> pending_exception{false};
    int32_t bitmap_format = ANDROID_BITMAP_FORMAT_RGBA_8888;
    std::deque<std::unique_ptr<IntArray>> new_int_arrays;
    std::deque<std::unique_ptr<Bitmap>> bitmaps;
    Klass bitmap_class, config_class, callback_class, bare_class;
    Plain argb8888;
};

State &st() {
    static State s;
    return s;
}

thread_local bool t_attached = false;

char kOnProgress, kCreateBitmap, kArgbField;

void track(const _jobject *o, Kind k) {
    std::lock_guard<std::mutex> lk(st().m);
    st().kinds[o] = k;
}
void untrack(const _jobject *o) {
    std::lock_guard<std::mutex> lk(st().m);
    st().kinds.erase(o);
}
Kind kind_of(const _jobject *o) {
    std::lock_guard<std::mutex> lk(st().m);
    auto it = st().kinds.find(o);
    if (it == st().kinds.end()) {
        std::fprintf(stderr, "fake JNI: unknown object %p\n", (const void *)o);
        std::abort();
    }
    return it->second;
}

// Owned fakes register themselves, so every call can check what it was given.
template <class T, Kind K>
struct Owned {
    T obj;
    Owned() { track(&obj, K); }
    ~Owned() { untrack(&obj); }
    Owned(const Owned &) = delete;
    Owned &operator=(const Owned &) = delete;
    T *get() { return &obj; }
    T *operator->() { return &obj; }
};

using JByteArray = Owned<ByteArray, Kind::ByteArray>;
using JFloatArray = Owned<FloatArray, Kind::FloatArray>;
using JIntArray = Owned<IntArray, Kind::IntArray>;
using JBuffer = Owned<Buffer, Kind::Buffer>;
using JString = Owned<String, Kind::String>;
using JCallback = Owned<Callback, Kind::Callback>;

// --- JNINativeInterface -----------------------------------------------------

jclass FindClass(JNIEnv *, const char *name) {
    if (std::strcmp(name, "android/graphics/Bitmap") == 0) return &st().bitmap_class;
    if (std::strcmp(name, "android/graphics/Bitmap$Config") == 0) return &st().config_class;
    st().pending_exception = true;   // NoClassDefFoundError
    return nullptr;
}
jclass GetObjectClass(JNIEnv *, jobject obj) {
    if (kind_of(obj) != Kind::Callback) std::abort();
    return static_cast<Callback *>(obj)->has_method ? &st().callback_class : &st().bare_class;
}
jmethodID GetMethodID(JNIEnv *, jclass cls, const char *name, const char *sig) {
    if (cls == &st().callback_class && std::strcmp(name, "onProgressUpdate") == 0 &&
        std::strcmp(sig, "(I)V") == 0) {
        return reinterpret_cast<jmethodID>(&kOnProgress);
    }
    st().pending_exception = true;   // NoSuchMethodError
    return nullptr;
}
void CallVoidMethodV(JNIEnv *, jobject obj, jmethodID mid, va_list args) {
    if (mid != reinterpret_cast<jmethodID>(&kOnProgress) || kind_of(obj) != Kind::Callback ||
        !t_attached) {
        std::abort();
    }
    const jint pct = va_arg(args, jint);
    auto *cb = static_cast<Callback *>(obj);
    {
        std::lock_guard<std::mutex> lk(cb->m);
        cb->calls.push_back(pct);
    }
    if (cb->on_call) cb->on_call(pct);
}
jmethodID GetStaticMethodID(JNIEnv *, jclass cls, const char *name, const char *sig) {
    if (cls == &st().bitmap_class && std::strcmp(name, "createBitmap") == 0 &&
        std::strcmp(sig, "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;") == 0) {
        return reinterpret_cast<jmethodID>(&kCreateBitmap);
    }
    st().pending_exception = true;
    return nullptr;
}
jobject CallStaticObjectMethodV(JNIEnv *, jclass cls, jmethodID mid, va_list args) {
    if (cls != &st().bitmap_class || mid != reinterpret_cast<jmethodID>(&kCreateBitmap)) std::abort();
    const jint w = va_arg(args, jint);
    const jint h = va_arg(args, jint);
    const jobject config = va_arg(args, jobject);
    if (config != &st().argb8888 || w <= 0 || h <= 0) {
        st().pending_exception = true;   // IllegalArgumentException
        return nullptr;
    }
    auto b = std::make_unique<Bitmap>();
    b->w = w;
    b->h = h;
    b->stride = (uint32_t)w * 4 + kBitmapPadding;
    b->format = st().bitmap_format;
    b->px.assign((size_t)b->stride * h, kBitmapFill);
    track(b.get(), Kind::Bitmap);
    st().bitmaps.push_back(std::move(b));
    return st().bitmaps.back().get();
}
jfieldID GetStaticFieldID(JNIEnv *, jclass cls, const char *name, const char *sig) {
    if (cls == &st().config_class && std::strcmp(name, "ARGB_8888") == 0 &&
        std::strcmp(sig, "Landroid/graphics/Bitmap$Config;") == 0) {
        return reinterpret_cast<jfieldID>(&kArgbField);
    }
    st().pending_exception = true;
    return nullptr;
}
jobject GetStaticObjectField(JNIEnv *, jclass cls, jfieldID fid) {
    if (cls != &st().config_class || fid != reinterpret_cast<jfieldID>(&kArgbField)) std::abort();
    return &st().argb8888;
}
jboolean ExceptionCheck(JNIEnv *) { return st().pending_exception ? JNI_TRUE : JNI_FALSE; }
void ExceptionClear(JNIEnv *) { st().pending_exception = false; }
jobject NewGlobalRef(JNIEnv *, jobject obj) {
    ++st().global_refs;
    return obj;
}
void DeleteGlobalRef(JNIEnv *, jobject) { --st().global_refs; }
void DeleteLocalRef(JNIEnv *, jobject) {}
JavaVM *the_vm();
jint GetJavaVM(JNIEnv *, JavaVM **vm) {
    *vm = the_vm();
    return JNI_OK;
}
jsize GetArrayLength(JNIEnv *, jarray arr) {
    switch (kind_of(arr)) {
        case Kind::ByteArray: return (jsize)static_cast<ByteArray *>(arr)->data.size();
        case Kind::FloatArray: return (jsize)static_cast<FloatArray *>(arr)->data.size();
        case Kind::IntArray: return (jsize)static_cast<IntArray *>(arr)->data.size();
        default: std::abort();
    }
}
jbyte *GetByteArrayElements(JNIEnv *, jbyteArray arr, jboolean *is_copy) {
    if (kind_of(arr) != Kind::ByteArray) std::abort();
    auto *a = static_cast<ByteArray *>(arr);
    if (a->refuse) {
        st().pending_exception = true;   // OutOfMemoryError
        return nullptr;
    }
    // A copy of exactly the array's length: ASan bounds any over-read.
    auto *copy = static_cast<jbyte *>(std::malloc(std::max<size_t>(a->data.size(), 1)));
    if (!a->data.empty()) std::memcpy(copy, a->data.data(), a->data.size());
    a->pinned.push_back(copy);
    if (is_copy) *is_copy = JNI_TRUE;
    return copy;
}
void ReleaseByteArrayElements(JNIEnv *, jbyteArray arr, jbyte *elems, jint mode) {
    if (kind_of(arr) != Kind::ByteArray) std::abort();
    auto *a = static_cast<ByteArray *>(arr);
    auto it = std::find(a->pinned.begin(), a->pinned.end(), elems);
    if (it == a->pinned.end()) std::abort();   // not a pointer this array handed out
    if (mode != JNI_ABORT && !a->data.empty()) std::memcpy(a->data.data(), elems, a->data.size());
    a->pinned.erase(it);
    std::free(elems);
}
const char *GetStringUTFChars(JNIEnv *, jstring s, jboolean *is_copy) {
    if (kind_of(s) != Kind::String) std::abort();
    ++st().utf_held;
    if (is_copy) *is_copy = JNI_TRUE;
    const std::string &t = static_cast<String *>(s)->text;
    char *copy = static_cast<char *>(std::malloc(t.size() + 1));
    std::memcpy(copy, t.c_str(), t.size() + 1);
    return copy;
}
void ReleaseStringUTFChars(JNIEnv *, jstring, const char *chars) {
    --st().utf_held;
    std::free(const_cast<char *>(chars));
}
jintArray NewIntArray(JNIEnv *, jsize n) {
    auto a = std::make_unique<IntArray>();
    a->data.assign((size_t)n, 0);
    track(a.get(), Kind::IntArray);
    st().new_int_arrays.push_back(std::move(a));
    return st().new_int_arrays.back().get();
}
template <class A, class T>
void set_region(A *a, jsize start, jsize len, const T *buf) {
    if (start < 0 || len < 0 || (size_t)start + (size_t)len > a->data.size()) {
        ++st().region_errors;   // ArrayIndexOutOfBoundsException
        st().pending_exception = true;
        return;
    }
    if (len > 0) std::memcpy(a->data.data() + start, buf, (size_t)len * sizeof(T));
}
void SetIntArrayRegion(JNIEnv *, jintArray arr, jsize start, jsize len, const jint *buf) {
    if (kind_of(arr) != Kind::IntArray) std::abort();
    set_region(static_cast<IntArray *>(arr), start, len, buf);
}
void SetFloatArrayRegion(JNIEnv *, jfloatArray arr, jsize start, jsize len, const jfloat *buf) {
    if (kind_of(arr) != Kind::FloatArray) std::abort();
    set_region(static_cast<FloatArray *>(arr), start, len, buf);
}
void *GetDirectBufferAddress(JNIEnv *, jobject obj) {
    if (kind_of(obj) != Kind::Buffer) std::abort();
    auto *b = static_cast<Buffer *>(obj);
    return b->direct ? b->data.data() : nullptr;
}
jlong GetDirectBufferCapacity(JNIEnv *, jobject obj) {
    if (kind_of(obj) != Kind::Buffer) std::abort();
    auto *b = static_cast<Buffer *>(obj);
    return b->direct ? b->capacity_bytes : -1;
}

// --- JNIInvokeInterface -----------------------------------------------------

JNIEnv *the_env();
jint GetEnv(JavaVM *, void **penv, jint version) {
    if (version != JNI_VERSION_1_6) return JNI_EVERSION;
    if (!t_attached) {
        *penv = nullptr;
        return JNI_EDETACHED;
    }
    *penv = the_env();
    return JNI_OK;
}
jint AttachCurrentThread(JavaVM *, JNIEnv **penv, void *) {
    if (t_attached) std::abort();   // the adapter attaches only detached threads
    t_attached = true;
    ++st().attaches;
    *penv = the_env();
    return JNI_OK;
}
jint DetachCurrentThread(JavaVM *) {
    if (!t_attached) std::abort();
    t_attached = false;
    ++st().detaches;
    return JNI_OK;
}

JNINativeInterface make_functions() {
    JNINativeInterface f;
    std::memset(&f, 0, sizeof f);   // an unexpected call is a null-pointer crash
    f.FindClass = FindClass;
    f.GetObjectClass = GetObjectClass;
    f.GetMethodID = GetMethodID;
    f.CallVoidMethodV = CallVoidMethodV;
    f.GetStaticMethodID = GetStaticMethodID;
    f.CallStaticObjectMethodV = CallStaticObjectMethodV;
    f.GetStaticFieldID = GetStaticFieldID;
    f.GetStaticObjectField = GetStaticObjectField;
    f.ExceptionCheck = ExceptionCheck;
    f.ExceptionClear = ExceptionClear;
    f.NewGlobalRef = NewGlobalRef;
    f.DeleteGlobalRef = DeleteGlobalRef;
    f.DeleteLocalRef = DeleteLocalRef;
    f.GetJavaVM = GetJavaVM;
    f.GetArrayLength = GetArrayLength;
    f.GetByteArrayElements = GetByteArrayElements;
    f.ReleaseByteArrayElements = ReleaseByteArrayElements;
    f.GetStringUTFChars = GetStringUTFChars;
    f.ReleaseStringUTFChars = ReleaseStringUTFChars;
    f.NewIntArray = NewIntArray;
    f.SetIntArrayRegion = SetIntArrayRegion;
    f.SetFloatArrayRegion = SetFloatArrayRegion;
    f.GetDirectBufferAddress = GetDirectBufferAddress;
    f.GetDirectBufferCapacity = GetDirectBufferCapacity;
    return f;
}
JNIInvokeInterface make_invoke() {
    JNIInvokeInterface f;
    std::memset(&f, 0, sizeof f);
    f.AttachCurrentThread = AttachCurrentThread;
    f.DetachCurrentThread = DetachCurrentThread;
    f.GetEnv = GetEnv;
    return f;
}

JNIEnv *the_env() {
    static const JNINativeInterface fns = make_functions();
    static _JNIEnv env{&fns};
    return &env;
}
JavaVM *the_vm() {
    static const JNIInvokeInterface fns = make_invoke();
    static _JavaVM vm{&fns};
    return &vm;
}

} // namespace fakejni

// The NDK's libjnigraphics, over the fake Bitmap objects.
extern "C" int AndroidBitmap_getInfo(JNIEnv *, jobject jbitmap, AndroidBitmapInfo *info) {
    if (fakejni::kind_of(jbitmap) != fakejni::Kind::Bitmap || info == nullptr)
        return ANDROID_BITMAP_RESULT_BAD_PARAMETER;
    const auto *b = static_cast<fakejni::Bitmap *>(jbitmap);
    info->width = (uint32_t)b->w;
    info->height = (uint32_t)b->h;
    info->stride = b->stride;
    info->format = b->format;
    info->flags = 0;
    return ANDROID_BITMAP_RESULT_SUCCESS;
}
extern "C" int AndroidBitmap_lockPixels(JNIEnv *, jobject jbitmap, void **addr) {
    if (fakejni::kind_of(jbitmap) != fakejni::Kind::Bitmap) return ANDROID_BITMAP_RESULT_BAD_PARAMETER;
    auto *b = static_cast<fakejni::Bitmap *>(jbitmap);
    ++b->locks;
    *addr = b->px.data();
    return ANDROID_BITMAP_RESULT_SUCCESS;
}
extern "C" int AndroidBitmap_unlockPixels(JNIEnv *, jobject jbitmap) {
    if (fakejni::kind_of(jbitmap) != fakejni::Kind::Bitmap) return ANDROID_BITMAP_RESULT_BAD_PARAMETER;
    --static_cast<fakejni::Bitmap *>(jbitmap)->locks;
    return ANDROID_BITMAP_RESULT_SUCCESS;
}
// The NDK's liblog: to stderr, as the engine's host log does.
extern "C" int __android_log_print(int prio, const char *tag, const char *fmt, ...) {
    std::fprintf(stderr, "[%s/%d] ", tag, prio);
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fputc('\n', stderr);
    return 1;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
namespace {

using fakejni::JBuffer;
using fakejni::JByteArray;
using fakejni::JCallback;
using fakejni::JFloatArray;
using fakejni::JIntArray;
using fakejni::JString;

// A JNI call's world: this thread attached, as a Kotlin caller's is, and the
// bookkeeping zeroed; the destructor checks nothing leaked.
struct Jni {
    JNIEnv *env = fakejni::the_env();
    jobject self = nullptr;   // SemperNativeLib: the adapter never reads it
    Jni() {
        fakejni::t_attached = true;
        auto &s = fakejni::st();
        s.global_refs = 0;
        s.attaches = 0;
        s.detaches = 0;
        s.utf_held = 0;
        s.region_errors = 0;
        s.pending_exception = false;
        s.bitmap_format = ANDROID_BITMAP_FORMAT_RGBA_8888;
    }
    ~Jni() { fakejni::t_attached = false; }
    // What a well-behaved adapter leaves behind after any call.
    void check_balanced() const {
        auto &s = fakejni::st();
        CHECK(s.global_refs == 0);
        CHECK(s.attaches == s.detaches);
        CHECK(s.utf_held == 0);
        CHECK(s.region_errors == 0);
        CHECK(!s.pending_exception);
    }
};

void fill(JByteArray &a, const std::vector<uchar> &bytes) {
    a->data.assign(bytes.begin(), bytes.end());
}
void fill_raw(JByteArray &a, const cv::Mat &gray) {
    REQUIRE(gray.type() == CV_8UC1 && gray.isContinuous());
    a->data.assign(gray.datastart, gray.dataend);
}
std::vector<uchar> png(const cv::Mat &img) {
    std::vector<uchar> out;
    REQUIRE(cv::imencode(".png", img, out));
    return out;
}
void size_buffer(JBuffer &b, size_t floats) {
    b->data.assign(floats, -12345.0f);
    b->capacity_bytes = (jlong)(floats * sizeof(float));
}
bool unpinned(JByteArray &a) { return a->pinned.empty(); }

struct Scene {
    cv::Mat ref, def;
    FullFieldParams p;
    int cap_floats() const { return 8 * ((p.rect_w / p.step) * (p.rect_h / p.step) + 8); }
};

// A 192 px speckle pair, translated and slightly stretched and turned, solved
// on the app's default grid (subset 21, step 5, strain window 15).
Scene speckle_scene() {
    Scene s;
    const int w = 192, h = 192;
    const dictest::SmoothSpeckleField field(31, w, h);
    s.ref = dictest::render_frame(field, w, h, dictest::identity_motion());
    s.def = dictest::render_frame(
            field, w, h,
            dictest::stretch_rotate(0.01, 0.3, 0.02, dictest::Vec2{1.3, -0.7}, dictest::Vec2{96, 96}));
    s.p.rect_x = 26;
    s.p.rect_y = 26;
    s.p.rect_w = 140;
    s.p.rect_h = 140;
    s.p.step = 5;
    s.p.subset_size = 21;
    s.p.strain_window = 15;
    s.p.use_6x6_interpolator = false;
    return s;
}

struct Direct {
    int rc = 0;
    std::vector<float> out;
    float metrics[kMetricsLen] = {};
};

Direct solve_direct(const Scene &s, const cv::Mat &mask, int cap_floats) {
    ReferenceCache cache;
    cache.set_from_gray(s.ref, mask);
    Direct d;
    d.out.assign((size_t)cap_floats, -12345.0f);
    d.rc = Semper::pipeline::run_full_field(cache, s.def, mask, s.p, d.out.data(), cap_floats,
                                            d.metrics, kMetricsLen, nullptr);
    return d;
}

jint compute(Jni &j, JByteArray *def, JByteArray *mask, const FullFieldParams &p, JBuffer &buf,
             JCallback *cb, JFloatArray *metrics) {
    return SNL(computeFullFieldDirect)(
            j.env, j.self, nullptr, def ? def->get() : nullptr, mask ? mask->get() : nullptr,
            p.rect_x, p.rect_y, p.rect_w, p.rect_h, p.step, p.subset_size, p.strain_window,
            p.use_6x6_interpolator ? JNI_TRUE : JNI_FALSE, buf.get(), cb ? cb->get() : nullptr,
            metrics ? metrics->get() : nullptr);
}

bool same_floats(const float *a, const float *b, size_t n) {
    return std::memcmp(a, b, n * sizeof(float)) == 0;
}

// Result slots (not the timings 9..14, 17, 18) agree over the first n.
bool same_result_slots(const float *a, const float *b, int n) {
    for (int i = 0; i < n; ++i) {
        if ((i >= 9 && i <= 14) || i == 17 || i == 18) continue;
        if (a[i] != b[i]) {
            std::printf("    metrics[%d] %.9g vs %.9g\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_CASE(JniAdapter, OnLoad_ReturnsVersionAndPinsOpenCvToOneThread) {
    const int before = cv::getNumThreads();
    const jint v = JNI_OnLoad(fakejni::the_vm(), nullptr);
    const int after = cv::getNumThreads();
    cv::setNumThreads(before);   // the rest of dic_tests runs as it did
    CHECK(v == JNI_VERSION_1_6);
    CHECK(after == 1);
}

// The app's batch path: initializeReference (PNG) then computeFullFieldDirect
// (raw pixels), with and without a mask, equals run_full_field byte for byte;
// progress arrives on an attached thread and ends at 100; nothing leaks.
TEST_CASE(JniAdapter, FullFieldDirect_MatchesRunFullField) {
    const Scene s = speckle_scene();
    cv::Mat mask(s.ref.size(), CV_8UC1, cv::Scalar(255));
    cv::circle(mask, cv::Point(96, 96), 25, cv::Scalar(0), -1);

    for (const bool masked : {false, true}) {
        Jni j;
        JByteArray ref, def, m;
        fill(ref, png(s.ref));
        fill_raw(def, s.def);
        fill(m, png(mask));
        SNL(initializeReference)(j.env, j.self, ref.get(), masked ? m.get() : nullptr, s.ref.cols,
                                 s.ref.rows);
        JBuffer buf;
        size_buffer(buf, (size_t)s.cap_floats());
        JCallback cb;
        JFloatArray metrics;
        metrics->data.assign(kMetricsLen, -7.0f);
        const jint rc = compute(j, &def, masked ? &m : nullptr, s.p, buf, &cb, &metrics);

        const Direct d = solve_direct(s, masked ? mask : cv::Mat(), s.cap_floats());
        std::printf("    %s: jni %d direct %d, %zu progress calls, %d thread attaches\n",
                    masked ? "masked" : "plain", (int)rc, d.rc, cb->calls.size(),
                    fakejni::st().attaches.load());
        REQUIRE(d.rc > 0);
        // The solver's progress thread reports first: the adapter attached it.
        CHECK(fakejni::st().attaches >= 1);
        CHECK(rc == d.rc);
        CHECK(same_floats(buf->data.data(), d.out.data(), (size_t)d.rc * 8));
        CHECK(same_result_slots(metrics->data.data(), d.metrics, kMetricsLen));
        REQUIRE(!cb->calls.empty());
        CHECK(cb->calls.back() == 100);
        CHECK(std::is_sorted(cb->calls.begin(), cb->calls.end()));
        CHECK(unpinned(ref) && unpinned(def) && unpinned(m));
        j.check_balanced();
    }
}

// Every input the adapter refuses returns -3 and leaves no array pinned.
TEST_CASE(JniAdapter, FullFieldDirect_RefusesBadInput) {
    const Scene s = speckle_scene();
    Jni j;
    JByteArray ref, def, garbage, refused;
    fill(ref, png(s.ref));
    fill_raw(def, s.def);
    // A PNG signature over bytes that do not decode, and not w * h long.
    garbage->data = {(jbyte)0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n', 1, 2, 3, 4, 5};
    fill_raw(refused, s.def);
    refused->refuse = true;
    SNL(initializeReference)(j.env, j.self, ref.get(), nullptr, s.ref.cols, s.ref.rows);

    JBuffer buf;
    size_buffer(buf, (size_t)s.cap_floats());
    CHECK(compute(j, nullptr, nullptr, s.p, buf, nullptr, nullptr) == -3);
    CHECK(compute(j, &garbage, nullptr, s.p, buf, nullptr, nullptr) == -3);
    CHECK(compute(j, &def, &garbage, s.p, buf, nullptr, nullptr) == -3);
    CHECK(compute(j, &refused, nullptr, s.p, buf, nullptr, nullptr) == -3);
    fakejni::st().pending_exception = false;   // the refused pin's OutOfMemoryError

    JBuffer heap;   // a heap ByteBuffer: no direct address
    size_buffer(heap, (size_t)s.cap_floats());
    heap->direct = false;
    CHECK(compute(j, &def, nullptr, s.p, heap, nullptr, nullptr) == -3);
    JBuffer tiny;   // under one float
    tiny->data.assign(1, 0.0f);
    tiny->capacity_bytes = 3;
    CHECK(compute(j, &def, nullptr, s.p, tiny, nullptr, nullptr) == -3);
    CHECK(SNL(computeFullFieldDirect)(nullptr, nullptr, nullptr, def.get(), nullptr, 0, 0, 1, 1, 1,
                                      1, 1, JNI_FALSE, buf.get(), nullptr, nullptr) == -3);
    CHECK(unpinned(ref) && unpinned(def) && unpinned(garbage) && unpinned(refused));
    j.check_balanced();

    // A reference that does not decode empties the cache: the next solve fails.
    SNL(initializeReference)(j.env, j.self, garbage.get(), nullptr, s.ref.cols, s.ref.rows);
    const jint rc = compute(j, &def, nullptr, s.p, buf, nullptr, nullptr);
    std::printf("    solve after a bad reference: %d\n", (int)rc);
    CHECK(rc < 0);
    CHECK(unpinned(garbage) && unpinned(def));
    j.check_balanced();
}

// A direct buffer smaller than the field: the adapter passes its capacity, the
// solver stays inside it (the fake buffer is exactly that long, so ASan sees
// any write past it), and the result is the engine's for that capacity.
TEST_CASE(JniAdapter, FullFieldDirect_SmallBufferStaysInBounds) {
    const Scene s = speckle_scene();
    Jni j;
    JByteArray ref, def;
    fill(ref, png(s.ref));
    fill_raw(def, s.def);
    SNL(initializeReference)(j.env, j.self, ref.get(), nullptr, s.ref.cols, s.ref.rows);
    for (const int cap : {8 * 10 + 3, 8}) {
        JBuffer buf;
        size_buffer(buf, (size_t)cap);
        const jint rc = compute(j, &def, nullptr, s.p, buf, nullptr, nullptr);
        const Direct d = solve_direct(s, cv::Mat(), cap);
        std::printf("    capacity %d floats: jni %d direct %d\n", cap, (int)rc, d.rc);
        CHECK(rc == d.rc);
        CHECK(same_floats(buf->data.data(), d.out.data(), (size_t)cap));
    }
    j.check_balanced();
}

// The metrics array gets the largest prefix of the engine's table it can hold
// (16, 17, 19 ... 24 slots); under 16, none.
TEST_CASE(JniAdapter, Metrics_CopiesTheLargestPrefix) {
    const Scene s = speckle_scene();
    const Direct d = solve_direct(s, cv::Mat(), s.cap_floats());
    REQUIRE(d.rc > 0);
    Jni j;
    JByteArray ref, def;
    fill(ref, png(s.ref));
    fill_raw(def, s.def);
    SNL(initializeReference)(j.env, j.self, ref.get(), nullptr, s.ref.cols, s.ref.rows);
    for (const int len : {8, 16, 18, 23, 24, 30}) {
        JBuffer buf;
        size_buffer(buf, (size_t)s.cap_floats());
        JFloatArray metrics;
        metrics->data.assign((size_t)len, -7.0f);
        CHECK(compute(j, &def, nullptr, s.p, buf, nullptr, &metrics) == d.rc);
        const int expect = len < 16 ? 0 : Semper::pipeline::metrics_prefix_len(len);
        int written = 0;
        while (written < len && metrics->data[(size_t)written] != -7.0f) ++written;
        std::printf("    array of %d: %d slots written, prefix table says %d\n", len, written, expect);
        CHECK(same_result_slots(metrics->data.data(), d.metrics, expect));
        for (int i = expect; i < len; ++i) CHECK(metrics->data[(size_t)i] == -7.0f);
    }
    j.check_balanced();
}

// initializeIncremental + computeIncrementalFrame equal an IncrementalSession
// driven directly, frame info included; error codes pass through; after
// releaseIncremental a frame is -3.
TEST_CASE(JniAdapter, Incremental_MatchesSession) {
    const int w = 192, h = 192;
    const dictest::SmoothSpeckleField field(43, w, h);
    const cv::Mat ref = dictest::render_frame(field, w, h, dictest::identity_motion());
    std::vector<cv::Mat> frames;
    for (int k = 1; k <= 4; ++k) {
        frames.push_back(dictest::render_frame(
                field, w, h,
                dictest::stretch_rotate(0.01 * k, 0.3, 0.01 * k, dictest::Vec2{0.6 * k, -0.4 * k},
                                        dictest::Vec2{96, 96})));
    }
    FullFieldParams p = speckle_scene().p;
    const int cap = 8 * ((p.rect_w / p.step) * (p.rect_h / p.step) + 8);

    IncrementalSession session;
    REQUIRE(session.init(ref, cv::Mat(), IncrementalParams{p, 2}) == 0);

    Jni j;
    JByteArray jref;
    fill(jref, png(ref));
    auto init = [&](const FullFieldParams &q, int interval, JByteArray *r) {
        return SNL(initializeIncremental)(j.env, j.self, r ? r->get() : nullptr, nullptr, w, h,
                                          q.rect_x, q.rect_y, q.rect_w, q.rect_h, q.step,
                                          q.subset_size, q.strain_window, JNI_FALSE, interval);
    };
    FullFieldParams bad = p;
    bad.subset_size = 20;
    CHECK(init(bad, 2, &jref) == -3);
    bad = p;
    bad.rect_x = 1000;
    CHECK(init(bad, 2, &jref) == -2);
    CHECK(init(p, 2, nullptr) == -3);
    REQUIRE(init(p, 2, &jref) == 0);

    for (size_t t = 0; t < frames.size(); ++t) {
        JByteArray def;
        fill_raw(def, frames[t]);
        JBuffer buf;
        size_buffer(buf, (size_t)cap);
        JCallback cb;
        JFloatArray metrics;
        metrics->data.assign(kMetricsLen, -7.0f);
        JIntArray info;
        info->data.assign(6, -1);
        const jint rc = SNL(computeIncrementalFrame)(j.env, j.self, def.get(), buf.get(), cb.get(),
                                                     metrics.get(), info.get());

        std::vector<float> out((size_t)cap, -12345.0f);
        float m[kMetricsLen] = {};
        IncrementalFrameInfo fi;
        const int drc = session.process_frame(frames[t], out.data(), cap, m, kMetricsLen, nullptr, &fi);
        std::printf("    frame %zu: jni %d direct %d, reference %d\n", t + 1, (int)rc, drc,
                    info->data[0]);
        REQUIRE(drc > 0);
        CHECK(rc == drc);
        CHECK(same_floats(buf->data.data(), out.data(), (size_t)drc * 8));
        CHECK(same_result_slots(metrics->data.data(), m, kMetricsLen));
        CHECK(info->data[0] == fi.reference_frame);
        CHECK(info->data[1] == (fi.reference_updated ? 1 : 0));
        CHECK(info->data[2] == fi.tracked_points);
        CHECK(info->data[3] == fi.frame0_points);
        CHECK(info->data[4] == fi.taylor_fallbacks);
        CHECK(info->data[5] == fi.lost_this_frame);
        CHECK(!cb->calls.empty() && cb->calls.back() == 100);
        CHECK(unpinned(def));
    }

    // An info array shorter than 6 is left alone.
    {
        JByteArray def;
        fill_raw(def, frames.back());
        JBuffer buf;
        size_buffer(buf, (size_t)cap);
        JIntArray info;
        info->data.assign(5, -1);
        CHECK(SNL(computeIncrementalFrame)(j.env, j.self, def.get(), buf.get(), nullptr, nullptr,
                                           info.get()) > 0);
        for (jint v : info->data) CHECK(v == -1);
    }

    SNL(releaseIncremental)(j.env, j.self);
    JByteArray def;
    fill_raw(def, frames[0]);
    JBuffer buf;
    size_buffer(buf, (size_t)cap);
    CHECK(SNL(computeIncrementalFrame)(j.env, j.self, def.get(), buf.get(), nullptr, nullptr,
                                       nullptr) == -3);
    CHECK(unpinned(jref) && unpinned(def));
    j.check_balanced();
}

// setCancelRequested(true) from another thread while the solve holds the
// adapter's lock (here, from the first progress callback, as the app's UI
// thread does) stops it with kCancelled; the next solve is the engine's. A
// flag left set before a solve does not cancel it: run_full_field clears it.
TEST_CASE(JniAdapter, Cancel_StopsTheRunningSolve) {
    const Scene s = speckle_scene();
    Jni j;
    JByteArray ref, def;
    fill(ref, png(s.ref));
    fill_raw(def, s.def);
    SNL(initializeReference)(j.env, j.self, ref.get(), nullptr, s.ref.cols, s.ref.rows);
    JBuffer buf;
    size_buffer(buf, (size_t)s.cap_floats());
    JCallback cb;
    std::atomic<bool> asked(false);
    cb->on_call = [&](int pct) {
        if (pct < 100 && !asked.exchange(true)) {
            std::thread ui([&] { SNL(setCancelRequested)(j.env, j.self, JNI_TRUE); });
            ui.join();
        }
    };
    const jint cancelled = compute(j, &def, nullptr, s.p, buf, &cb, nullptr);

    SNL(setCancelRequested)(j.env, j.self, JNI_TRUE);   // stale: cleared by the next solve
    const jint rc = compute(j, &def, nullptr, s.p, buf, nullptr, nullptr);
    SNL(setCancelRequested)(j.env, j.self, JNI_FALSE);
    const Direct d = solve_direct(s, cv::Mat(), s.cap_floats());
    std::printf("    asked %d: cancelled %d, then %d (direct %d)\n", (int)asked.load(),
                (int)cancelled, (int)rc, d.rc);
    REQUIRE(asked.load());
    CHECK(cancelled == Semper::pipeline::kCancelled);
    CHECK(rc == d.rc);
    CHECK(same_floats(buf->data.data(), d.out.data(), (size_t)std::max(d.rc, 0) * 8));
    j.check_balanced();
}

// A progress object without onProgressUpdate(I)V: the NoSuchMethodError is
// cleared, nothing is called, and the solve is unaffected.
TEST_CASE(JniAdapter, ProgressWithoutMethod_SolvesSilently) {
    const Scene s = speckle_scene();
    Jni j;
    JByteArray ref, def;
    fill(ref, png(s.ref));
    fill_raw(def, s.def);
    SNL(initializeReference)(j.env, j.self, ref.get(), nullptr, s.ref.cols, s.ref.rows);
    JBuffer buf;
    size_buffer(buf, (size_t)s.cap_floats());
    JCallback cb;
    cb->has_method = false;
    const jint rc = compute(j, &def, nullptr, s.p, buf, &cb, nullptr);
    const Direct d = solve_direct(s, cv::Mat(), s.cap_floats());
    CHECK(rc == d.rc);
    CHECK(cb->calls.empty());
    j.check_balanced();
}

// A Path B worker that throws (the test build's hook; nothing an input can
// do makes one) comes back through guarded_call as -3, not a crash; the next
// solve is the engine's. The scene is test_seeding_contracts.cpp's
// PathCRouteScene, where Path B floods the row on its workers.
TEST_CASE(JniAdapter, ExceptionInTheSolve_IsMinus3) {
    Scene s;
    const int w = 384, h = 256;
    const dictest::SmoothSpeckleField field(101, w, h);
    s.ref = dictest::render_frame(field, w, h, dictest::identity_motion(), 0.0f, 7919u);
    s.def = dictest::render_frame(
            field, w, h, dictest::stretch_rotate(0.45, 0.3, 0.0, dictest::Vec2{}, dictest::Vec2{192, 128}),
            0.0f, 7919u + 150u);
    s.p.rect_x = 172;
    s.p.rect_y = 128;
    s.p.rect_w = 8 * 5;
    s.p.rect_h = 5;
    s.p.step = 5;
    s.p.subset_size = 21;
    s.p.strain_window = 11;
    s.p.use_6x6_interpolator = false;

    Jni j;
    JByteArray ref, def;
    fill(ref, png(s.ref));
    fill_raw(def, s.def);
    SNL(initializeReference)(j.env, j.self, ref.get(), nullptr, w, h);
    JBuffer buf;
    size_buffer(buf, 8 * 8);
    JCallback cb;

    std::atomic<int> cells(0);
    Semper::pipeline::internal::test_hooks::path_b_worker_cell() = [&](int) {
        if (cells.fetch_add(1) == 0) throw std::runtime_error("injected Path B fault");
    };
    const jint failed = compute(j, &def, nullptr, s.p, buf, &cb, nullptr);
    Semper::pipeline::internal::test_hooks::path_b_worker_cell() = nullptr;
    const jint rc = compute(j, &def, nullptr, s.p, buf, nullptr, nullptr);
    const Direct d = solve_direct(s, cv::Mat(), 8 * 8);
    std::printf("    worker cells %d: %d, then %d (direct %d)\n", cells.load(), (int)failed,
                (int)rc, d.rc);
    REQUIRE(cells.load() >= 1);
    CHECK(failed == -3);
    CHECK(rc == d.rc);
    CHECK(unpinned(ref) && unpinned(def));
    j.check_balanced();
}

// getPreviewFromBytes writes the scaled RGBA image row by row into a Bitmap
// whose rows are padded, and leaves the padding alone.
TEST_CASE(JniAdapter, Preview_ScalesIntoAPaddedBitmap) {
    cv::Mat bgr(40, 64, CV_8UC3);
    cv::RNG rng(5);
    rng.fill(bgr, cv::RNG::UNIFORM, 0, 256);
    Jni j;
    JByteArray bytes;
    fill(bytes, png(bgr));

    jobject obj = SNL(getPreviewFromBytes)(j.env, j.self, bytes.get(), 32);
    REQUIRE(obj != nullptr);
    const auto *bmp = static_cast<fakejni::Bitmap *>(obj);
    cv::Mat expect;
    cv::resize(bgr, expect, cv::Size(32, 20));
    cv::cvtColor(expect, expect, cv::COLOR_BGR2RGBA);
    CHECK(bmp->w == 32 && bmp->h == 20);
    CHECK(bmp->locks == 0);
    bool pixels = true, padding = true;
    for (int y = 0; y < bmp->h; ++y) {
        const uint8_t *row = bmp->px.data() + (size_t)y * bmp->stride;
        pixels = pixels && std::memcmp(row, expect.ptr(y), 32 * 4) == 0;
        for (int k = 0; k < fakejni::kBitmapPadding; ++k) {
            padding = padding && row[32 * 4 + k] == fakejni::kBitmapFill;
        }
    }
    CHECK(pixels);
    CHECK(padding);

    // A very wide image keeps a height of one row.
    JByteArray wide;
    fill(wide, png(cv::Mat(3, 400, CV_8UC3, cv::Scalar(10, 20, 30))));
    jobject strip = SNL(getPreviewFromBytes)(j.env, j.self, wide.get(), 10);
    REQUIRE(strip != nullptr);
    CHECK(static_cast<fakejni::Bitmap *>(strip)->h == 1);

    // Refusals: no width, bytes that do not decode, a Bitmap not RGBA_8888.
    JByteArray garbage;
    garbage->data = {1, 2, 3, 4};
    CHECK(SNL(getPreviewFromBytes)(j.env, j.self, bytes.get(), 0) == nullptr);
    CHECK(SNL(getPreviewFromBytes)(j.env, j.self, garbage.get(), 32) == nullptr);
    CHECK(SNL(getPreviewFromBytes)(j.env, j.self, nullptr, 32) == nullptr);
    fakejni::st().bitmap_format = ANDROID_BITMAP_FORMAT_RGB_565;
    CHECK(SNL(getPreviewFromBytes)(j.env, j.self, bytes.get(), 32) == nullptr);
    for (const auto &b : fakejni::st().bitmaps) CHECK(b->locks == 0);
    CHECK(unpinned(bytes) && unpinned(wide) && unpinned(garbage));
    j.check_balanced();
}

TEST_CASE(JniAdapter, ImageDimensions_OrZeros) {
    Jni j;
    JByteArray bytes, garbage, refused;
    fill(bytes, png(cv::Mat(40, 64, CV_8UC1, cv::Scalar(7))));
    garbage->data = {1, 2, 3};
    fill(refused, png(cv::Mat(40, 64, CV_8UC1, cv::Scalar(7))));
    refused->refuse = true;
    auto dims = [&](jbyteArray a) {
        auto *r = static_cast<fakejni::IntArray *>(SNL(getImageDimensions)(j.env, j.self, a));
        REQUIRE(r != nullptr && r->data.size() == 2);
        return std::make_pair(r->data[0], r->data[1]);
    };
    CHECK(dims(bytes.get()) == std::make_pair(64, 40));
    CHECK(dims(garbage.get()) == std::make_pair(0, 0));
    CHECK(dims(nullptr) == std::make_pair(0, 0));
    CHECK(dims(refused.get()) == std::make_pair(0, 0));
    fakejni::st().pending_exception = false;
    CHECK(unpinned(bytes) && unpinned(garbage) && unpinned(refused));
    j.check_balanced();
}

// setDebugOutputDir copies the string and releases it; null clears it.
TEST_CASE(JniAdapter, DebugOutputDir_ReleasesTheString) {
    Jni j;
    JString dir;
    dir->text = "/nonexistent/semper-debug";
    SNL(setDebugOutputDir)(j.env, j.self, dir.get());
    SNL(setDebugOutputDir)(j.env, j.self, nullptr);
    j.check_balanced();
}

// The com.indicvision.semper forwarders reach the same code.
TEST_CASE(JniAdapter, LegacyPackage_Forwards) {
    const Scene s = speckle_scene();
    Jni j;
    JByteArray ref, def;
    fill(ref, png(s.ref));
    fill_raw(def, s.def);
    Java_com_indicvision_semper_SemperNativeLib_initializeReference(j.env, j.self, ref.get(),
                                                                    nullptr, s.ref.cols, s.ref.rows);
    JBuffer buf;
    size_buffer(buf, (size_t)s.cap_floats());
    const jint rc = Java_com_indicvision_semper_SemperNativeLib_computeFullFieldDirect(
            j.env, j.self, nullptr, def.get(), nullptr, s.p.rect_x, s.p.rect_y, s.p.rect_w,
            s.p.rect_h, s.p.step, s.p.subset_size, s.p.strain_window, JNI_FALSE, buf.get(), nullptr,
            nullptr);
    const Direct d = solve_direct(s, cv::Mat(), s.cap_floats());
    CHECK(rc == d.rc);
    CHECK(same_floats(buf->data.data(), d.out.data(), (size_t)std::max(d.rc, 0) * 8));
    j.check_balanced();
}

#endif // DIC_HAVE_OPENCV && SEMPER_JNI_HARNESS
