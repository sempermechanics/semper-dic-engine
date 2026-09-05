cmake_minimum_required(VERSION 3.22)
# Version tracks include/semper/version.hpp (keep the two in sync).
# C is required: vendored OpenCV and the C ABI smoke binary compile C sources.
project(semper VERSION 0.2.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(SemperOptions)

set(SEMPER_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(OPENCV_SRC_DIR "${SEMPER_ROOT}/third_party/opencv")
set(EIGEN_SRC_DIR "${SEMPER_ROOT}/third_party/eigen")

if(SEMPER_ANDROID AND SEMPER_FORCE_RELEASE)
    # AGP passes -DCMAKE_BUILD_TYPE=Debug for debug APKs, which would compile
    # OpenCV at -O0 (~10x slower AKAZE). Force Release like the prebuilt SDK.
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

include(SemperOpenCV)
add_subdirectory(${OPENCV_SRC_DIR} ${CMAKE_BINARY_DIR}/opencv_build EXCLUDE_FROM_ALL)

set(SEMPER_OPENCV_INCLUDE_DIRS
        ${OPENCV_SRC_DIR}/include
        ${OPENCV_SRC_DIR}/modules/core/include
        ${OPENCV_SRC_DIR}/modules/imgproc/include
        ${OPENCV_SRC_DIR}/modules/imgcodecs/include
        ${OPENCV_SRC_DIR}/modules/features2d/include
        ${OPENCV_SRC_DIR}/modules/calib3d/include
        ${OPENCV_SRC_DIR}/modules/flann/include
        ${CMAKE_BINARY_DIR}
)

set(SEMPER_PUBLIC_INCLUDE ${SEMPER_ROOT}/include)
set(SEMPER_SRC_INCLUDE ${SEMPER_ROOT}/src)

# ---------------------------------------------------------------------------
# semper_math — portable DIC math (no Android / JNI / OpenMP)
# ---------------------------------------------------------------------------
add_library(semper_math STATIC
        src/math/image_processor.cpp
        src/math/subset_precomputer.cpp
        src/math/optimization_engine.cpp
        src/strain/strain_calculator.cpp
)
target_include_directories(semper_math
        PUBLIC
            ${SEMPER_PUBLIC_INCLUDE}
        PRIVATE
            ${SEMPER_SRC_INCLUDE}
)
# Vendored headers marked SYSTEM so -Wall/-Wextra below flag only our code.
target_include_directories(semper_math SYSTEM PUBLIC
            ${EIGEN_SRC_DIR}
            ${OPENCV_SRC_DIR}/modules/core/include
            ${CMAKE_BINARY_DIR}
)
target_compile_definitions(semper_math PRIVATE
        $<$<BOOL:${SEMPER_ANDROID}>:SEMPER_ANDROID_LOG=1>
        $<$<NOT:$<BOOL:${SEMPER_ANDROID}>>:SEMPER_HOST_LOG=1>
)
if(NOT MSVC)
    target_compile_options(semper_math PRIVATE -O3 -ffast-math -fPIC -fvisibility=hidden -Wall -Wextra)
endif()
if(SEMPER_SANITIZER AND NOT MSVC)
    target_compile_options(semper_math PRIVATE -fsanitize=${SEMPER_SANITIZER} -g -fno-omit-frame-pointer)
    target_link_options(semper_math PUBLIC -fsanitize=${SEMPER_SANITIZER})
endif()

# ---------------------------------------------------------------------------
# semper_pipeline — seeding + full-field orchestration (OpenCV + OpenMP)
# ---------------------------------------------------------------------------
add_library(semper_pipeline STATIC
        src/io/image_codec.cpp
        src/seeding/akaze_seed.cpp
        src/seeding/anchor_seed.cpp
        src/pipeline/reference_cache.cpp
        src/pipeline/cancel.cpp
        src/pipeline/full_field_solver.cpp
        src/pipeline/full_field_akaze.cpp
        src/pipeline/full_field_mesh.cpp
        src/pipeline/full_field_path_a.cpp
        src/pipeline/full_field_path_b.cpp
        src/pipeline/full_field_path_c.cpp
        src/pipeline/full_field_solver_stats.cpp
        src/pipeline/full_field_debug_export.cpp
)
target_include_directories(semper_pipeline
        PUBLIC
            ${SEMPER_PUBLIC_INCLUDE}
        PRIVATE
            ${SEMPER_SRC_INCLUDE}
)
target_include_directories(semper_pipeline SYSTEM PUBLIC
            ${SEMPER_OPENCV_INCLUDE_DIRS}
            ${EIGEN_SRC_DIR}
)
target_link_libraries(semper_pipeline
        PUBLIC
            semper_math
            opencv_core
            opencv_imgproc
            opencv_imgcodecs
            opencv_features2d
            opencv_calib3d
            opencv_flann
)
target_compile_definitions(semper_pipeline PRIVATE
        $<$<BOOL:${SEMPER_ANDROID}>:SEMPER_ANDROID_LOG=1>
        $<$<NOT:$<BOOL:${SEMPER_ANDROID}>>:SEMPER_HOST_LOG=1>
)
if(NOT MSVC)
    # Apply aggressive flags only to our TUs, not OpenCV (added after add_subdirectory).
    # -flto is Android-only: MinGW host shared libs (semper_c) fail at runtime with
    # "32 bit pseudo relocation ... out of range" when the static archive was LTO-built.
    set(_semper_pipeline_opts -O3 -ffast-math -fPIC -fvisibility=hidden -fexceptions -Wall -Wextra)
    if(SEMPER_ANDROID)
        list(APPEND _semper_pipeline_opts -flto -fopenmp)
        target_compile_options(semper_pipeline PRIVATE ${_semper_pipeline_opts})
        target_link_options(semper_pipeline PUBLIC -fopenmp)
    else()
        # Apple Clang rejects bare -fopenmp; use CMake's OpenMP package when present.
        find_package(OpenMP)
        target_compile_options(semper_pipeline PRIVATE ${_semper_pipeline_opts})
        if(OpenMP_CXX_FOUND)
            target_link_libraries(semper_pipeline PUBLIC OpenMP::OpenMP_CXX)
        endif()
    endif()
endif()
if(SEMPER_SANITIZER AND NOT MSVC)
    # Sanitizers and -flto fight; drop LTO when instrumenting.
    target_compile_options(semper_pipeline PRIVATE -fsanitize=${SEMPER_SANITIZER} -g -fno-omit-frame-pointer -fno-lto)
    target_link_options(semper_pipeline PUBLIC -fsanitize=${SEMPER_SANITIZER})
endif()

add_library(semper INTERFACE)
target_link_libraries(semper INTERFACE semper_pipeline)
target_include_directories(semper INTERFACE ${SEMPER_PUBLIC_INCLUDE})

# ---------------------------------------------------------------------------
# Public distribution adapters (off by default)
# ---------------------------------------------------------------------------
if(SEMPER_BUILD_C_SDK)
    add_subdirectory(adapters/c)
endif()
if(SEMPER_BUILD_PYTHON)
    add_subdirectory(bindings/python)
endif()
if(SEMPER_BUILD_EXAMPLES)
    add_subdirectory(examples/cpp)
endif()

# ---------------------------------------------------------------------------
# Android shared library (keeps OUTPUT_NAME semper_core for System.loadLibrary)
# ---------------------------------------------------------------------------
if(SEMPER_ANDROID)
    add_subdirectory(adapters/android)
endif()

if(SEMPER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
