#include "gpu/cl_runtime.hpp"
#include "gpu/cl_api.hpp"
#include "util/log.hpp"

#include <semper/gpu/embedded_kernels.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

#undef LOG_TAG
#define LOG_TAG "SemperCL"

namespace Semper {
namespace gpu {
namespace {

// Build options. Every entry is load-bearing for bit-exactness, and the
// omissions matter as much as the inclusions:
//
//   -cl-std=CL1.2                            the version we target
//   -cl-fp32-correctly-rounded-divide-sqrt   makes / and sqrt reproducible;
//                                            without it a device cannot match
//                                            the CPU reference at all
//
// Deliberately ABSENT, and never to be added:
//   -cl-fast-relaxed-math   licenses reassociation, exactly what
//                           canonical_math.h exists to prevent
//   -cl-mad-enable          licenses a*b+c -> fma, one rounding where the
//                           reference does two
//
// tests/unit/test_cl_runtime.cpp asserts both absences.
constexpr char kBuildOptions[] = "-cl-std=CL1.2 -cl-fp32-correctly-rounded-divide-sqrt";

struct Loader {
    void *handle = nullptr;
    ClApi api;

    bool open(std::string &reason) {
        // Try the versioned soname first: an unversioned libOpenCL.so usually
        // only exists when a -dev package is installed, which end users have
        // no reason to have.
        static const char *kNames[] = {
#if defined(_WIN32)
            "OpenCL.dll",
#elif defined(__APPLE__)
            "/System/Library/Frameworks/OpenCL.framework/OpenCL",
            "libOpenCL.dylib",
#else
            "libOpenCL.so.1",
            "libOpenCL.so",
#endif
        };
        for (const char *n : kNames) {
            handle = dl_open(n);
            if (handle) break;
        }
        if (!handle) {
            reason = "no OpenCL loader found (tried libOpenCL.so.1 / .so). "
                     "Install a vendor driver, or pocl-opencl-icd for a CPU device.";
            return false;
        }
        // A missing symbol here means the loader is present but not an
        // OpenCL 1.2 one -- report it as such rather than crashing later.
        const char *missing = resolve_all();
        if (missing) {
            reason = std::string("OpenCL loader is missing entry point ") + missing +
                     " (not an OpenCL 1.2 implementation?)";
            return false;
        }
        return true;
    }

private:
    static void *dl_open(const char *name) {
#if defined(_WIN32)
        return reinterpret_cast<void *>(LoadLibraryA(name));
#else
        return dlopen(name, RTLD_LAZY | RTLD_LOCAL);
#endif
    }

    void *sym(const char *name) const {
#if defined(_WIN32)
        return reinterpret_cast<void *>(
                GetProcAddress(reinterpret_cast<HMODULE>(handle), name));
#else
        return dlsym(handle, name);
#endif
    }

    // Returns the name of the first symbol that failed to resolve, or null.
    const char *resolve_all() {
#define SEMPER_RESOLVE(field, name)                                        \
        do {                                                               \
            void *p = sym(name);                                           \
            if (!p) return name;                                           \
            api.field = reinterpret_cast<decltype(api.field)>(p);          \
        } while (0)

        SEMPER_RESOLVE(GetPlatformIDs, "clGetPlatformIDs");
        SEMPER_RESOLVE(GetDeviceIDs, "clGetDeviceIDs");
        SEMPER_RESOLVE(GetDeviceInfo, "clGetDeviceInfo");
        SEMPER_RESOLVE(CreateContext, "clCreateContext");
        SEMPER_RESOLVE(CreateCommandQueue, "clCreateCommandQueue");
        SEMPER_RESOLVE(CreateProgramWithSource, "clCreateProgramWithSource");
        SEMPER_RESOLVE(BuildProgram, "clBuildProgram");
        SEMPER_RESOLVE(GetProgramBuildInfo, "clGetProgramBuildInfo");
        SEMPER_RESOLVE(CreateKernel, "clCreateKernel");
        SEMPER_RESOLVE(SetKernelArg, "clSetKernelArg");
        SEMPER_RESOLVE(CreateBuffer, "clCreateBuffer");
        SEMPER_RESOLVE(EnqueueNDRangeKernel, "clEnqueueNDRangeKernel");
        SEMPER_RESOLVE(EnqueueReadBuffer, "clEnqueueReadBuffer");
        SEMPER_RESOLVE(EnqueueWriteBuffer, "clEnqueueWriteBuffer");
        SEMPER_RESOLVE(Finish, "clFinish");
        SEMPER_RESOLVE(ReleaseMemObject, "clReleaseMemObject");
        SEMPER_RESOLVE(ReleaseKernel, "clReleaseKernel");
        SEMPER_RESOLVE(ReleaseProgram, "clReleaseProgram");
        SEMPER_RESOLVE(ReleaseCommandQueue, "clReleaseCommandQueue");
        SEMPER_RESOLVE(ReleaseContext, "clReleaseContext");
#undef SEMPER_RESOLVE
        return nullptr;
    }
};

// Everything the backend holds open once a device is accepted.
struct Runtime {
    Loader loader;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    ClCaps caps;

    void release() {
        // Order matters: children before parents.
        if (program) loader.api.ReleaseProgram(program), program = nullptr;
        if (queue) loader.api.ReleaseCommandQueue(queue), queue = nullptr;
        if (context) loader.api.ReleaseContext(context), context = nullptr;
        device = nullptr;
        // The library handle is intentionally NOT closed. Some ICDs register
        // atexit handlers or thread-local state that a dlclose invalidates,
        // turning shutdown into a crash. Leaking one handle for process
        // lifetime is the conventional trade.
    }
};

std::mutex g_mutex;
Runtime *g_rt = nullptr;
bool g_probed = false;

std::string device_string(const ClApi &api, cl_device_id d, cl_device_info info) {
    size_t n = 0;
    if (api.GetDeviceInfo(d, info, 0, nullptr, &n) != SEMPER_CL_SUCCESS || n == 0)
        return std::string();
    std::vector<char> buf(n + 1, '\0');
    if (api.GetDeviceInfo(d, info, n, buf.data(), nullptr) != SEMPER_CL_SUCCESS)
        return std::string();
    return std::string(buf.data());
}

bool env_disabled() {
    const char *e = std::getenv("SEMPER_OPENCL_DISABLE");
    return e && e[0] == '1' && e[1] == '\0';
}

// Parse the major/minor out of "OpenCL <major>.<minor> <vendor text>".
bool version_at_least_1_2(const std::string &v) {
    const char *p = std::strstr(v.c_str(), "OpenCL ");
    if (!p) return false;
    p += 7;
    int major = 0, minor = 0;
    if (std::sscanf(p, "%d.%d", &major, &minor) != 2) return false;
    return major > 1 || (major == 1 && minor >= 2);
}

// Populate g_rt. Sets caps.unavailable_reason on every failure path.
void probe_locked() {
    g_rt = new Runtime();
    ClCaps &c = g_rt->caps;

    if (env_disabled()) {
        c.unavailable_reason = "disabled by SEMPER_OPENCL_DISABLE=1";
        return;
    }

    std::string reason;
    if (!g_rt->loader.open(reason)) {
        c.unavailable_reason = reason;
        return;
    }
    const ClApi &api = g_rt->loader.api;

    cl_uint num_platforms = 0;
    if (api.GetPlatformIDs(0, nullptr, &num_platforms) != SEMPER_CL_SUCCESS ||
        num_platforms == 0) {
        // The single most common state on a machine that has the loader
        // package but no driver -- name it precisely, because "no GPU" and
        // "no ICD registered" call for different fixes.
        c.unavailable_reason =
                "OpenCL loader present but reports zero platforms "
                "(no ICD registered in /etc/OpenCL/vendors)";
        return;
    }
    std::vector<cl_platform_id> platforms(num_platforms);
    if (api.GetPlatformIDs(num_platforms, platforms.data(), nullptr) != SEMPER_CL_SUCCESS) {
        c.unavailable_reason = "clGetPlatformIDs failed on the second call";
        return;
    }

    // Prefer a GPU; fall back to any device type so a CPU ICD (POCL) works
    // for development and CI.
    for (cl_device_type want : {(cl_device_type) SEMPER_CL_DEVICE_TYPE_GPU,
                                (cl_device_type) SEMPER_CL_DEVICE_TYPE_ALL}) {
        for (cl_platform_id p : platforms) {
            cl_uint n = 0;
            if (api.GetDeviceIDs(p, want, 0, nullptr, &n) != SEMPER_CL_SUCCESS || n == 0)
                continue;
            std::vector<cl_device_id> devs(n);
            if (api.GetDeviceIDs(p, want, n, devs.data(), nullptr) != SEMPER_CL_SUCCESS)
                continue;
            g_rt->device = devs[0];
            break;
        }
        if (g_rt->device) break;
    }
    if (!g_rt->device) {
        c.unavailable_reason = "OpenCL platforms exist but expose no usable device";
        return;
    }

    c.device_name = device_string(api, g_rt->device, SEMPER_CL_DEVICE_NAME);
    c.driver_version = device_string(api, g_rt->device, SEMPER_CL_DRIVER_VERSION);
    c.device_version = device_string(api, g_rt->device, SEMPER_CL_DEVICE_VERSION);

    if (!version_at_least_1_2(c.device_version)) {
        c.unavailable_reason = "device reports '" + c.device_version +
                               "', but OpenCL 1.2 or newer is required";
        return;
    }

    const std::string ext = device_string(api, g_rt->device, SEMPER_CL_DEVICE_EXTENSIONS);
    c.fp64 = ext.find("cl_khr_fp64") != std::string::npos;

    cl_device_fp_config fp = 0;
    if (api.GetDeviceInfo(g_rt->device, SEMPER_CL_DEVICE_SINGLE_FP_CONFIG,
                          sizeof(fp), &fp, nullptr) == SEMPER_CL_SUCCESS) {
        c.exact_fp32 = (fp & SEMPER_CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT) != 0;
    }

    cl_int err = SEMPER_CL_SUCCESS;
    g_rt->context = api.CreateContext(nullptr, 1, &g_rt->device, nullptr, nullptr, &err);
    if (!g_rt->context || err != SEMPER_CL_SUCCESS) {
        c.unavailable_reason = "clCreateContext failed (" + std::to_string(err) + ")";
        return;
    }
    g_rt->queue = api.CreateCommandQueue(g_rt->context, g_rt->device, 0, &err);
    if (!g_rt->queue || err != SEMPER_CL_SUCCESS) {
        c.unavailable_reason = "clCreateCommandQueue failed (" + std::to_string(err) + ")";
        return;
    }

    const char *src = kernels::PROBE;
    const size_t len = std::strlen(src);
    g_rt->program = api.CreateProgramWithSource(g_rt->context, 1, &src, &len, &err);
    if (!g_rt->program || err != SEMPER_CL_SUCCESS) {
        c.unavailable_reason = "clCreateProgramWithSource failed (" + std::to_string(err) + ")";
        return;
    }
    err = api.BuildProgram(g_rt->program, 1, &g_rt->device, kBuildOptions, nullptr, nullptr);
    if (err != SEMPER_CL_SUCCESS) {
        // The device compiler's log is the only useful diagnostic here, so
        // carry it rather than just the error number.
        std::string log;
        size_t log_len = 0;
        if (api.GetProgramBuildInfo(g_rt->program, g_rt->device, SEMPER_CL_PROGRAM_BUILD_LOG,
                                    0, nullptr, &log_len) == SEMPER_CL_SUCCESS && log_len) {
            std::vector<char> buf(log_len + 1, '\0');
            if (api.GetProgramBuildInfo(g_rt->program, g_rt->device, SEMPER_CL_PROGRAM_BUILD_LOG,
                                        log_len, buf.data(), nullptr) == SEMPER_CL_SUCCESS)
                log.assign(buf.data());
        }
        c.unavailable_reason = "kernel build failed (" + std::to_string(err) + "): " + log;
        return;
    }

    c.available = true;
    LOGD("OpenCL ready: %s (%s) fp64=%d exact_fp32=%d", c.device_name.c_str(),
         c.device_version.c_str(), (int) c.fp64, (int) c.exact_fp32);
}

Runtime &runtime_locked() {
    if (!g_probed) {
        g_probed = true;
        probe_locked();
    }
    return *g_rt;
}

} // namespace

const ClCaps &caps() {
    std::lock_guard<std::mutex> lk(g_mutex);
    return runtime_locked().caps;
}

const char *build_options() { return kBuildOptions; }

const char *probe_kernel_source() { return kernels::PROBE; }

void reset_for_testing() {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (g_rt) {
        g_rt->release();
        delete g_rt;
        g_rt = nullptr;
    }
    g_probed = false;
}

bool run_probe_reduce(const float *vals, int n, float mean, float *out) {
    std::lock_guard<std::mutex> lk(g_mutex);
    Runtime &rt = runtime_locked();
    if (!rt.caps.available || n < 0 || (n > 0 && !vals) || !out) return false;

    const ClApi &api = rt.loader.api;
    cl_int err = SEMPER_CL_SUCCESS;
    bool ok = false;

    // n == 0 is legal (a fully masked subset) but a zero-size buffer is not,
    // so round the allocation up while still passing n through unchanged.
    const size_t in_bytes = sizeof(float) * (size_t) (n > 0 ? n : 1);
    cl_mem d_in = api.CreateBuffer(rt.context, SEMPER_CL_MEM_READ_ONLY, in_bytes, nullptr, &err);
    cl_mem d_out = nullptr;
    cl_kernel k = nullptr;
    if (!d_in || err != SEMPER_CL_SUCCESS) goto cleanup;

    d_out = api.CreateBuffer(rt.context, SEMPER_CL_MEM_WRITE_ONLY, sizeof(float), nullptr, &err);
    if (!d_out || err != SEMPER_CL_SUCCESS) goto cleanup;

    if (n > 0 && api.EnqueueWriteBuffer(rt.queue, d_in, SEMPER_CL_TRUE, 0,
                                        sizeof(float) * (size_t) n, vals,
                                        0, nullptr, nullptr) != SEMPER_CL_SUCCESS)
        goto cleanup;

    k = api.CreateKernel(rt.program, "semper_probe_reduce", &err);
    if (!k || err != SEMPER_CL_SUCCESS) goto cleanup;

    if (api.SetKernelArg(k, 0, sizeof(cl_mem), &d_in) != SEMPER_CL_SUCCESS ||
        api.SetKernelArg(k, 1, sizeof(int), &n) != SEMPER_CL_SUCCESS ||
        api.SetKernelArg(k, 2, sizeof(float), &mean) != SEMPER_CL_SUCCESS ||
        api.SetKernelArg(k, 3, sizeof(cl_mem), &d_out) != SEMPER_CL_SUCCESS)
        goto cleanup;

    {
        const size_t global = 1;
        if (api.EnqueueNDRangeKernel(rt.queue, k, 1, nullptr, &global, nullptr,
                                     0, nullptr, nullptr) != SEMPER_CL_SUCCESS)
            goto cleanup;
    }
    if (api.Finish(rt.queue) != SEMPER_CL_SUCCESS) goto cleanup;
    if (api.EnqueueReadBuffer(rt.queue, d_out, SEMPER_CL_TRUE, 0, sizeof(float), out,
                              0, nullptr, nullptr) != SEMPER_CL_SUCCESS)
        goto cleanup;
    ok = true;

cleanup:
    if (k) api.ReleaseKernel(k);
    if (d_out) api.ReleaseMemObject(d_out);
    if (d_in) api.ReleaseMemObject(d_in);
    return ok;
}

} // namespace gpu
} // namespace Semper
