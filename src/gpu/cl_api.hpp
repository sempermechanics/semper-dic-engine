#ifndef SEMPER_GPU_CL_API_HPP
#define SEMPER_GPU_CL_API_HPP

// =====================================================================
// MINIMAL OpenCL 1.2 DECLARATIONS
//
// Just the entry points and constants this engine uses, declared here so
// that building the OpenCL backend needs no Khronos headers, no
// find_package, and no link-time libOpenCL. cl_runtime.cpp dlopen()s the
// ICD loader and resolves these at runtime, which means:
//
//   - a SEMPER_OPENCL=ON build links and RUNS on a machine with no
//     OpenCL installed at all (the probe simply reports unavailable);
//   - there is no new third_party submodule to vendor and pin;
//   - CI needs no OpenCL packages to compile the backend.
//
// Values are from the OpenCL 1.2 specification (cl.h). They are ABI, not
// implementation detail: they have been stable since 1.0 and cannot
// change without breaking every existing binary. Do not "tidy" them.
//
// Only what is used is declared. Adding a call means adding its typedef
// here and resolving it in cl_runtime.cpp -- deliberately a little
// friction, so the surface stays small enough to audit.
// =====================================================================

#include <cstddef>
#include <cstdint>

extern "C" {

typedef int32_t  cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef uint64_t cl_bitfield;
typedef cl_uint  cl_bool;

typedef cl_bitfield cl_device_type;
typedef cl_uint     cl_platform_info;
typedef cl_uint     cl_device_info;
typedef cl_uint     cl_program_build_info;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_command_queue_properties;
typedef cl_bitfield cl_device_fp_config;
typedef intptr_t    cl_context_properties;

typedef struct _cl_platform_id     *cl_platform_id;
typedef struct _cl_device_id       *cl_device_id;
typedef struct _cl_context         *cl_context;
typedef struct _cl_command_queue   *cl_command_queue;
typedef struct _cl_program         *cl_program;
typedef struct _cl_kernel          *cl_kernel;
typedef struct _cl_mem             *cl_mem;
typedef struct _cl_event           *cl_event;

} // extern "C"

// Windows uses __stdcall for the OpenCL ABI; everything else uses the
// platform default. Getting this wrong corrupts the stack on 32-bit
// Windows rather than failing to link, so it is spelled out.
#if defined(_WIN32) && !defined(_WIN64)
#  define SEMPER_CL_CALL __stdcall
#else
#  define SEMPER_CL_CALL
#endif

// --- Error codes (only those we surface) ---
#define SEMPER_CL_SUCCESS                       0
#define SEMPER_CL_DEVICE_NOT_FOUND             -1
#define SEMPER_CL_BUILD_PROGRAM_FAILURE       -11

// --- Device types ---
#define SEMPER_CL_DEVICE_TYPE_GPU        (1 << 2)
#define SEMPER_CL_DEVICE_TYPE_ALL        0xFFFFFFFFu

// --- clGetDeviceInfo selectors ---
#define SEMPER_CL_DEVICE_SINGLE_FP_CONFIG   0x101B
#define SEMPER_CL_DEVICE_NAME               0x102B
#define SEMPER_CL_DRIVER_VERSION            0x102D
#define SEMPER_CL_DEVICE_VERSION            0x102F
#define SEMPER_CL_DEVICE_EXTENSIONS         0x1030
#define SEMPER_CL_DEVICE_OPENCL_C_VERSION   0x103D

// --- Floating-point capability bits (CL_DEVICE_SINGLE_FP_CONFIG) ---
// The last one is the whole reason this gate exists: without correctly
// rounded fp32 divide and sqrt, a device cannot reproduce the CPU
// reference bit-for-bit, so the ICGN kernels must not run on it.
#define SEMPER_CL_FP_DENORM                          (1 << 0)
#define SEMPER_CL_FP_INF_NAN                         (1 << 1)
#define SEMPER_CL_FP_ROUND_TO_NEAREST                (1 << 2)
#define SEMPER_CL_FP_FMA                             (1 << 5)
#define SEMPER_CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT   (1 << 7)

// --- Program build info ---
#define SEMPER_CL_PROGRAM_BUILD_LOG    0x1183

// --- Memory flags ---
#define SEMPER_CL_MEM_READ_WRITE     (1 << 0)
#define SEMPER_CL_MEM_WRITE_ONLY     (1 << 1)
#define SEMPER_CL_MEM_READ_ONLY      (1 << 2)
#define SEMPER_CL_MEM_COPY_HOST_PTR  (1 << 5)

#define SEMPER_CL_TRUE   1
#define SEMPER_CL_FALSE  0

namespace Semper {
namespace gpu {

// Function-pointer types for every call resolved from the loader.
typedef cl_int (SEMPER_CL_CALL *PFN_clGetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
typedef cl_int (SEMPER_CL_CALL *PFN_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
typedef cl_int (SEMPER_CL_CALL *PFN_clGetDeviceInfo)(cl_device_id, cl_device_info, size_t, void *, size_t *);
typedef cl_context (SEMPER_CL_CALL *PFN_clCreateContext)(const cl_context_properties *, cl_uint, const cl_device_id *,
                                                         void (SEMPER_CL_CALL *)(const char *, const void *, size_t, void *),
                                                         void *, cl_int *);
typedef cl_command_queue (SEMPER_CL_CALL *PFN_clCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int *);
typedef cl_program (SEMPER_CL_CALL *PFN_clCreateProgramWithSource)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
typedef cl_int (SEMPER_CL_CALL *PFN_clBuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *,
                                                    void (SEMPER_CL_CALL *)(cl_program, void *), void *);
typedef cl_int (SEMPER_CL_CALL *PFN_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *);
typedef cl_kernel (SEMPER_CL_CALL *PFN_clCreateKernel)(cl_program, const char *, cl_int *);
typedef cl_int (SEMPER_CL_CALL *PFN_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
typedef cl_mem (SEMPER_CL_CALL *PFN_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
typedef cl_int (SEMPER_CL_CALL *PFN_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *,
                                                            const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
typedef cl_int (SEMPER_CL_CALL *PFN_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *,
                                                          cl_uint, const cl_event *, cl_event *);
typedef cl_int (SEMPER_CL_CALL *PFN_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *,
                                                           cl_uint, const cl_event *, cl_event *);
typedef cl_int (SEMPER_CL_CALL *PFN_clFinish)(cl_command_queue);
typedef cl_int (SEMPER_CL_CALL *PFN_clReleaseMemObject)(cl_mem);
typedef cl_int (SEMPER_CL_CALL *PFN_clReleaseKernel)(cl_kernel);
typedef cl_int (SEMPER_CL_CALL *PFN_clReleaseProgram)(cl_program);
typedef cl_int (SEMPER_CL_CALL *PFN_clReleaseCommandQueue)(cl_command_queue);
typedef cl_int (SEMPER_CL_CALL *PFN_clReleaseContext)(cl_context);

// Resolved entry points. Every member is null until load_icd() succeeds.
struct ClApi {
    PFN_clGetPlatformIDs          GetPlatformIDs          = nullptr;
    PFN_clGetDeviceIDs            GetDeviceIDs            = nullptr;
    PFN_clGetDeviceInfo           GetDeviceInfo           = nullptr;
    PFN_clCreateContext           CreateContext           = nullptr;
    PFN_clCreateCommandQueue      CreateCommandQueue      = nullptr;
    PFN_clCreateProgramWithSource CreateProgramWithSource = nullptr;
    PFN_clBuildProgram            BuildProgram            = nullptr;
    PFN_clGetProgramBuildInfo     GetProgramBuildInfo     = nullptr;
    PFN_clCreateKernel            CreateKernel            = nullptr;
    PFN_clSetKernelArg            SetKernelArg            = nullptr;
    PFN_clCreateBuffer            CreateBuffer            = nullptr;
    PFN_clEnqueueNDRangeKernel    EnqueueNDRangeKernel    = nullptr;
    PFN_clEnqueueReadBuffer       EnqueueReadBuffer       = nullptr;
    PFN_clEnqueueWriteBuffer      EnqueueWriteBuffer      = nullptr;
    PFN_clFinish                  Finish                  = nullptr;
    PFN_clReleaseMemObject        ReleaseMemObject        = nullptr;
    PFN_clReleaseKernel           ReleaseKernel           = nullptr;
    PFN_clReleaseProgram          ReleaseProgram          = nullptr;
    PFN_clReleaseCommandQueue     ReleaseCommandQueue     = nullptr;
    PFN_clReleaseContext          ReleaseContext          = nullptr;
};

} // namespace gpu
} // namespace Semper

#endif // SEMPER_GPU_CL_API_HPP
