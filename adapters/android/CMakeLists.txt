# Thin Android/JNI adapter. Included from native/CMakeLists.txt when
# SEMPER_ANDROID=ON. Produces libsemper_core.so for System.loadLibrary.

add_library(semper_android SHARED
        jni/SemperJNI.cpp
)
set_target_properties(semper_android PROPERTIES OUTPUT_NAME semper_core)

# Public engine headers only (+ the adapter's own dir for jni_log.hpp). The adapter
# deliberately does NOT include ${SEMPER_ROOT}/src, so it can be lifted into the app
# repo and link just the public engine — see docs/CONTRACT.md.
target_include_directories(semper_android PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/jni
        ${SEMPER_ROOT}/include
)
# Vendored OpenCV headers SYSTEM so -Wall/-Wextra flag only the JNI adapter.
target_include_directories(semper_android SYSTEM PRIVATE
        ${SEMPER_OPENCV_INCLUDE_DIRS}
)

target_compile_definitions(semper_android PRIVATE SEMPER_ANDROID_LOG=1)

if(NOT MSVC)
    target_compile_options(semper_android PRIVATE
            -O3 -flto -ffast-math -fno-finite-math-only -ffp-contract=off -fopenmp -fPIC -fvisibility=hidden -fexceptions -Wall -Wextra
    )
    target_link_options(semper_android PRIVATE
            -s -flto -Wl,-z,max-page-size=16384
            -fopenmp -static-openmp -static-libstdc++
    )
endif()

target_link_libraries(semper_android
        PRIVATE
            semper_pipeline
            log
            android
            jnigraphics
)
