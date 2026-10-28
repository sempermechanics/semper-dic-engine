# =====================================================================
# HOST-SIDE NATIVE TEST SUITE
#
# Builds the portable DIC math library against the micro test framework.
# No Android NDK required.
#
# From the engine repo root (sempermechanics/semper-dic-engine):
#   cmake -S tests -B build/tests -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/tests
#   ./build/tests/dic_tests
# =====================================================================
cmake_minimum_required(VERSION 3.16)
project(semper_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(NATIVE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Treat any first-party warning as an error under -DSEMPER_WERROR=ON.
# Off by default for local builds; CI host job enables it (-DSEMPER_WERROR=ON).
option(SEMPER_WERROR "Fail the build on compiler warnings in first-party code" OFF)
# OFF builds the pipeline without OpenMP, as on a toolchain that has none
# (CI's serial-build job). The whole suite must pass either way.
option(DIC_OPENMP "Build the pipeline sources with OpenMP" ON)

if (NOT MSVC)
    add_compile_options(-O2 -ffast-math -Wall -Wextra)
    if (SEMPER_WERROR)
        # GCC's -Waggressive-loop-optimizations is a known false positive on
        # SIMD residual tails (reports absurd iteration counts). Keep it as a
        # warning so real -Wall/-Wextra issues still fail the build.
        add_compile_options(-Werror -Wno-error=aggressive-loop-optimizations)
        # GCC 15 reports copying a std::vector<double> and letting the copy go
        # out of scope as deleting a pointer it did not allocate
        # (-Wfree-nonheap-object), inlined from libstdc++ into test bodies.
        # Same treatment; GCC only, as Clang has no such warning.
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            add_compile_options(-Wno-error=free-nonheap-object)
        endif ()
    endif ()
else ()
    add_compile_options(/O2 /fp:fast /W3)
    if (SEMPER_WERROR)
        add_compile_options(/WX)
    endif ()
endif ()

set(DIC_SANITIZER "" CACHE STRING "Comma-separated sanitizers (address,undefined,thread) or empty")
if (DIC_SANITIZER AND NOT MSVC)
    add_compile_options(-fsanitize=${DIC_SANITIZER} -g -fno-omit-frame-pointer)
    add_link_options(-fsanitize=${DIC_SANITIZER})
    # Golden-corpus numbers are a Release-host gate. Sanitizers change
    # floating-point results enough to trip that tight relative check.
    add_compile_definitions(SEMPER_SANITIZER_BUILD=1)
endif ()

# Report-only gcov/llvm-cov instrumentation: build with -DSEMPER_COVERAGE=ON,
# run dic_tests, then gcovr/llvm-cov over the build tree. No gate.
option(SEMPER_COVERAGE "Instrument the host tests for coverage (gcov)" OFF)
if (SEMPER_COVERAGE AND NOT MSVC)
    # -O1, not -O0: at -O0 the suite ran over 90 min on CI (an incremental
    # test that takes 3 s in Release took over 9 min); at -O1 the whole suite
    # takes about 27 min on 4 cores.
    # Keep gcc's default counter update (atomic under -fopenmp): with
    # -fprofile-update=single the racing counter writes corrupted solver
    # results and broke the bit-identity tests.
    add_compile_options(--coverage -O1 -g)
    add_link_options(--coverage)
    # Golden-corpus numbers are a Release-host gate. Coverage builds are
    # -O1 without -O2's flags, which changes how many subsets initialize and
    # their values.
    add_compile_definitions(SEMPER_COVERAGE_BUILD=1)
endif ()

set(DIC_UNIT_TESTS
        unit/test_simd_kernels.cpp
        unit/test_image.cpp
        unit/test_subset_precomputer.cpp
        unit/test_strain_calculator.cpp
        unit/test_cancel_token.cpp
)
set(DIC_INTEGRATION_TESTS
        integration/test_optimization_engine.cpp
        integration/test_robustness.cpp
        integration/test_golden_corpus.cpp
)
set(DIC_DICE_TESTS
        dice/test_translation_synthetic.cpp
        dice/test_translation_real_image.cpp
        dice/test_strain_gradients.cpp
        dice/test_strain_vsg.cpp
        dice/test_field_agreement.cpp
        dice/test_incremental_oht.cpp
)
set(DIC_PERF_TESTS
        perf/test_throughput.cpp
)

set(DIC_ENGINE_SOURCES
        ${NATIVE_ROOT}/src/math/image_processor.cpp
        ${NATIVE_ROOT}/src/math/subset_precomputer.cpp
        ${NATIVE_ROOT}/src/math/optimization_engine.cpp
        ${NATIVE_ROOT}/src/strain/strain_calculator.cpp
        ${NATIVE_ROOT}/src/pipeline/cancel.cpp
)

# Full-field solver + seeding (OpenCV). Linked only when OpenCV is present so
# hosts without it still build the math suite; CI sets DIC_REQUIRE_OPENCV=ON.
set(DIC_PIPELINE_SOURCES
        ${NATIVE_ROOT}/src/io/image_codec.cpp
        ${NATIVE_ROOT}/src/pipeline/akaze_features.cpp
        ${NATIVE_ROOT}/src/pipeline/full_field_solver.cpp
        ${NATIVE_ROOT}/src/pipeline/full_field_anchors.cpp
        ${NATIVE_ROOT}/src/pipeline/full_field_mesh.cpp
        ${NATIVE_ROOT}/src/pipeline/full_field_path_a.cpp
        ${NATIVE_ROOT}/src/pipeline/full_field_path_b.cpp
        ${NATIVE_ROOT}/src/pipeline/full_field_path_c.cpp
        ${NATIVE_ROOT}/src/pipeline/full_field_solver_stats.cpp
        ${NATIVE_ROOT}/src/pipeline/full_field_debug_export.cpp
        ${NATIVE_ROOT}/src/pipeline/reference_cache.cpp
        ${NATIVE_ROOT}/src/pipeline/phase_correlate.cpp
        ${NATIVE_ROOT}/src/pipeline/incremental_math.cpp
        ${NATIVE_ROOT}/src/pipeline/incremental_session.cpp
)

# OpenCV-gated tests. test_image_codec / test_reference_cache exercise the image
# boundary and ROI-mask sterilization and need imgcodecs/imgproc, so they live
# here (attached via target_sources under _SEMPER_OPENCV_READY) rather than in the
# always-on DIC_UNIT_TESTS / DIC_INTEGRATION_TESTS lists.
set(DIC_PIPELINE_TESTS
        integration/test_full_field_contracts.cpp
        integration/test_full_field_golden.cpp
        integration/test_seeding_contracts.cpp
        integration/test_seeding_bench.cpp
        integration/test_incremental_bench.cpp
        integration/test_full_field_determinism.cpp
        integration/test_metamorphic.cpp
        unit/test_image_codec.cpp
        integration/test_reference_cache.cpp
        integration/test_incremental.cpp
        unit/test_incremental_math.cpp
)

add_executable(dic_tests
        test_main.cpp
        ${DIC_UNIT_TESTS}
        ${DIC_INTEGRATION_TESTS}
        ${DIC_DICE_TESTS}
        ${DIC_PERF_TESTS}
        ${DIC_ENGINE_SOURCES}
)

target_include_directories(dic_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/shim
        ${NATIVE_ROOT}/include
        ${NATIVE_ROOT}/src
)
# Vendored headers are not ours to warn about — SYSTEM silences -Wall/-Wextra there.
target_include_directories(dic_tests SYSTEM PRIVATE
        ${NATIVE_ROOT}/third_party/eigen
        ${NATIVE_ROOT}/third_party/opencv/modules/core/include
)

target_compile_definitions(dic_tests PRIVATE
        DICE_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/dice"
        NATIVE_TESTS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        SEMPER_HOST_LOG=1
        SEMPER_HOST_VERBOSE=1
        # Compiles in the fault-injection points in src/pipeline
        # (internal::test_hooks); the libraries are built without them.
        SEMPER_TEST_HOOKS=1
)

option(DIC_REQUIRE_OPENCV "Fail configure if OpenCV is missing (CI sets this)" OFF)
option(DIC_REQUIRE_JNI "Fail configure if the JNI adapter harness cannot be built (CI sets this)" OFF)

# Prefer OpenCV targets already built by the parent `native/` package (vendored
# third_party). Fall back to find_package for the standalone host-tests tree (CI).
set(_SEMPER_OPENCV_READY FALSE)
if (TARGET opencv_core AND TARGET opencv_imgcodecs AND TARGET opencv_imgproc
        AND TARGET opencv_features2d AND TARGET opencv_calib3d)
    set(_SEMPER_OPENCV_READY TRUE)
    set(OpenCV_LIBS opencv_core opencv_imgproc opencv_imgcodecs
            opencv_features2d opencv_calib3d opencv_flann)
    message(STATUS "Using parent-built OpenCV targets — full_field_solver linked")
else ()
    find_package(OpenCV QUIET COMPONENTS core imgcodecs imgproc features2d calib3d flann)
    if (OpenCV_FOUND)
        set(_SEMPER_OPENCV_READY TRUE)
        message(STATUS "OpenCV ${OpenCV_VERSION} found — image-backed DICe + full_field_solver linked")
    endif ()
endif ()

if (_SEMPER_OPENCV_READY)
    if (NOT TARGET dic_test_imageio)
        add_library(dic_test_imageio STATIC framework/image_io.cpp)
        target_include_directories(dic_test_imageio PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(dic_test_imageio PRIVATE ${OpenCV_LIBS})
        if (OpenCV_INCLUDE_DIRS)
            target_include_directories(dic_test_imageio SYSTEM PRIVATE
                    ${OpenCV_INCLUDE_DIRS})
        endif ()
        # Parent native/ package exposes SEMPER_OPENCV_INCLUDE_DIRS for the
        # vendored OpenCV tree (find_package leaves these empty).
        if (DEFINED SEMPER_OPENCV_INCLUDE_DIRS)
            target_include_directories(dic_test_imageio SYSTEM PRIVATE
                    ${SEMPER_OPENCV_INCLUDE_DIRS})
        endif ()
    endif ()
    target_compile_definitions(dic_tests PRIVATE DIC_HAVE_OPENCV=1)
    if (OpenCV_INCLUDE_DIRS)
        target_include_directories(dic_tests SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
    endif ()
    if (DEFINED SEMPER_OPENCV_INCLUDE_DIRS)
        target_include_directories(dic_tests SYSTEM PRIVATE ${SEMPER_OPENCV_INCLUDE_DIRS})
    endif ()
    target_link_libraries(dic_tests PRIVATE dic_test_imageio ${OpenCV_LIBS})

    # N1: host-link the production full-field solver so characterization can
    # pin contracts without the Android emulator. Additive — no algorithm change.
    target_sources(dic_tests PRIVATE
            ${DIC_PIPELINE_SOURCES}
            ${DIC_PIPELINE_TESTS})
    target_include_directories(dic_tests PRIVATE ${NATIVE_ROOT}/src)
    if (NOT DIC_OPENMP)
        # The engine as built where OpenMP is missing: every parallel region
        # runs on the calling thread and its #pragma omp is ignored.
        message(STATUS "dic_tests: OpenMP off (DIC_OPENMP=OFF), pipeline runs serially")
        if (NOT MSVC)
            target_compile_options(dic_tests PRIVATE -Wno-unknown-pragmas)
        endif ()
    else ()
        find_package(OpenMP)
        if (OpenMP_CXX_FOUND)
            target_link_libraries(dic_tests PRIVATE OpenMP::OpenMP_CXX)
        elseif (NOT MSVC)
            # Match production pipeline flags when CMake's OpenMP package is absent.
            target_compile_options(dic_tests PRIVATE -fopenmp)
            target_link_options(dic_tests PRIVATE -fopenmp)
        endif ()
    endif ()
    # JNI adapter harness (tests/jni/test_jni_adapter.cpp): the Android
    # adapter's C++ (adapters/android/jni/SemperJNI.cpp) compiled into
    # dic_tests and driven through a fake JNIEnv, so the sanitizer jobs cover
    # it. It needs the NDK's own jni.h (the adapter is written against it; the
    # JDK's differs in C++), which is copied alone out of the NDK sysroot so
    # none of Bionic's headers reach the host include path. Set
    # SEMPER_NDK_JNI_H to a jni.h, or let it be found from ANDROID_NDK_*.
    find_file(SEMPER_NDK_JNI_H jni.h
            PATHS $ENV{ANDROID_NDK_LATEST_HOME} $ENV{ANDROID_NDK_HOME}
                  $ENV{ANDROID_NDK_ROOT} $ENV{ANDROID_NDK}
            PATH_SUFFIXES toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include
                          toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/include
                          toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/include
            NO_DEFAULT_PATH)
    if (SEMPER_NDK_JNI_H AND NOT MSVC)
        configure_file(${SEMPER_NDK_JNI_H} ${CMAKE_CURRENT_BINARY_DIR}/jni_include/jni.h COPYONLY)
        message(STATUS "dic_tests: JNI adapter harness on (${SEMPER_NDK_JNI_H})")
        target_sources(dic_tests PRIVATE
                ${NATIVE_ROOT}/adapters/android/jni/SemperJNI.cpp
                jni/test_jni_adapter.cpp)
        target_include_directories(dic_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/jni/shim)
        target_include_directories(dic_tests SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/jni_include)
        target_compile_definitions(dic_tests PRIVATE SEMPER_JNI_HARNESS=1)
    elseif (DIC_REQUIRE_JNI)
        message(FATAL_ERROR
                "DIC_REQUIRE_JNI=ON but the NDK's jni.h was not found. Set ANDROID_NDK_HOME "
                "(or ANDROID_NDK_LATEST_HOME / ANDROID_NDK_ROOT) to an NDK, or pass "
                "-DSEMPER_NDK_JNI_H=<ndk>/toolchains/llvm/prebuilt/<host>/sysroot/usr/include/jni.h.")
    else ()
        message(STATUS "dic_tests: no NDK jni.h, JNI adapter harness skipped")
    endif ()
elseif (DIC_REQUIRE_OPENCV)
    message(FATAL_ERROR
            "DIC_REQUIRE_OPENCV=ON but OpenCV (core, imgcodecs, imgproc, features2d, "
            "calib3d, flann) was not found. "
            "Install it (Debian/Ubuntu: libopencv-dev) or build via native/ with "
            "SEMPER_BUILD_TESTS=ON.")
else ()
    message(STATUS "OpenCV not found — image-backed DICe and full_field contracts skipped")
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(dic_tests PRIVATE -Wno-macro-redefined)
endif ()
