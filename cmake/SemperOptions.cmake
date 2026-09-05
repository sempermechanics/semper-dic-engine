option(SEMPER_ANDROID "Build Android shared library (JNI adapter)" OFF)
option(SEMPER_BUILD_TESTS "Build host-side dic_tests" OFF)
option(SEMPER_BUILD_C_SDK "Build the stable C ABI shared library (semper_c)" OFF)
option(SEMPER_BUILD_PYTHON "Build the pybind11 Python extension" OFF)
option(SEMPER_BUILD_EXAMPLES "Build beginner C++ examples (links semper_pipeline)" OFF)
option(SEMPER_FORCE_RELEASE "Force Release flags even when AGP passes Debug" ON)

# OpenCL backend. OFF by default: with it off no OpenCL code is compiled and
# the binary carries no OpenCL symbols. With it on, the backend is still
# runtime-probed and silently falls back to the CPU path when no conforming
# device is present, so an ON build is safe to ship. Forced OFF for Android --
# OpenCL is not in the NDK and Mali/Adreno largely lack cl_khr_fp64.
# See docs/GPU_ACCELERATION.md.
option(SEMPER_OPENCL "Build the OpenCL backend (displacement + strain)" OFF)
if(SEMPER_ANDROID AND SEMPER_OPENCL)
    message(STATUS "SEMPER_OPENCL forced OFF: not supported on Android")
    set(SEMPER_OPENCL OFF CACHE BOOL "" FORCE)
endif()

# Optional sanitizers for first-party native libs (math/pipeline). OFF by default
# and must stay off for Release throughput measurement — never the default for
# production or perf runs. Example: -DSEMPER_SANITIZER=address,undefined
set(SEMPER_SANITIZER "" CACHE STRING "Comma-separated sanitizers for semper_* libs (empty = off)")
