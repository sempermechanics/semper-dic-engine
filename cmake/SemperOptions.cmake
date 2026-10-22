option(SEMPER_ANDROID "Build Android shared library (JNI adapter)" OFF)
option(SEMPER_BUILD_TESTS "Build host-side dic_tests" OFF)
option(SEMPER_BUILD_C_SDK "Build the stable C ABI shared library (semper_c)" OFF)
option(SEMPER_BUILD_PYTHON "Build the pybind11 Python extension" OFF)
option(SEMPER_BUILD_EXAMPLES "Build beginner C++ examples (links semper_pipeline)" OFF)
option(SEMPER_FORCE_RELEASE "Force Release flags even when AGP passes Debug" ON)
# Host builds always print errors (LOGE) to stderr. The per-solve profiling
# and routing trace (LOGD, ~30 lines a solve) only with this on, so the C SDK
# and the Python wheel stay quiet. Android logs through logcat either way.
option(SEMPER_VERBOSE_LOG "Print the LOGD solve trace to stderr on host builds" OFF)

# Optional sanitizers for first-party native libs (math/pipeline). OFF by default
# and must stay off for Release throughput measurement — never the default for
# production or perf runs. Example: -DSEMPER_SANITIZER=address,undefined
set(SEMPER_SANITIZER "" CACHE STRING "Comma-separated sanitizers for semper_* libs (empty = off)")
