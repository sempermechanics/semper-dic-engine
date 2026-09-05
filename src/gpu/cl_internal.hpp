#ifndef SEMPER_GPU_CL_INTERNAL_HPP
#define SEMPER_GPU_CL_INTERNAL_HPP

#include "gpu/cl_api.hpp"

namespace Semper {
namespace gpu {
namespace detail {

// Serialised access to the shared context and queue, plus a lazily built,
// process-cached program.
//
// Every dispatch stage from Phase 2 onward needs the same four things --
// resolved entry points, the context, the queue, and a built program -- and
// all of them sit behind the one mutex guarding the cached runtime. Handing
// them over as a scope guard means a stage cannot forget the lock, and means
// each .cl is compiled once per process instead of once per call. (Program
// build is not cheap: on POCL it invokes a full LLVM pipeline.)
//
// Construction never throws and never fails loudly. ok() is false when the
// backend is unavailable, or when this particular program did not build, and
// the caller falls back to the CPU. A build failure here does NOT flip
// caps().available: one stage failing to compile must not disable the
// others, which is the same per-stage rule the fp64 / exact_fp32 split
// encodes. Check the capability your stage needs BEFORE constructing this --
// a fp64 kernel on a device without cl_khr_fp64 compiles away to nothing and
// would fail at clCreateKernel with a less obvious error.
class ProgramScope {
public:
    // `key` identifies the cache entry and must be a distinct literal per
    // .cl source; `source` is the embedded, include-expanded text.
    ProgramScope(const char *key, const char *source);
    ~ProgramScope();

    ProgramScope(const ProgramScope &) = delete;
    ProgramScope &operator=(const ProgramScope &) = delete;

    bool ok() const { return ok_; }
    const ClApi &api() const { return *api_; }
    cl_context context() const { return context_; }
    cl_command_queue queue() const { return queue_; }
    cl_program program() const { return program_; }

private:
    bool ok_ = false;
    bool locked_ = false;
    const ClApi *api_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program program_ = nullptr;
};

} // namespace detail
} // namespace gpu
} // namespace Semper

#endif // SEMPER_GPU_CL_INTERNAL_HPP
