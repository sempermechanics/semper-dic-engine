#ifndef SEMPER_UTIL_OMP_HPP
#define SEMPER_UTIL_OMP_HPP

// OpenMP is optional (CMake links it when it finds it; MSVC builds without).
// Without it every `#pragma omp` region runs on the calling thread, and the
// per-thread slots the regions index are slot 0.

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace Semper {
namespace util {

inline int omp_thread() {
#if defined(_OPENMP)
    return omp_get_thread_num();
#else
    return 0;
#endif
}

} // namespace util
} // namespace Semper

#endif // SEMPER_UTIL_OMP_HPP
