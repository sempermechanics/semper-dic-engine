cmake_minimum_required(VERSION 3.22)
# Built via scikit-build-core, which points CMake at the engine root (native/)
# with -DSEMPER_BUILD_PYTHON=ON; native/CMakeLists.txt then add_subdirectory()s
# this file. The semper_pipeline target + SEMPER_PUBLIC_INCLUDE are already defined
# by the parent, so here we only add the extension module — no recursion into native.
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_semper src/module.cpp)
target_link_libraries(_semper PRIVATE semper_pipeline)
# CI's sanitized job builds the module this way and runs pytest with the
# sanitizer runtime preloaded, so module.cpp is instrumented as well.
if(SEMPER_SANITIZER AND NOT MSVC)
    target_compile_options(_semper PRIVATE -fsanitize=${SEMPER_SANITIZER} -fno-sanitize-recover=all
                           -g -fno-omit-frame-pointer)
    target_link_options(_semper PRIVATE -fsanitize=${SEMPER_SANITIZER} -fno-sanitize-recover=all)
endif()

# Installed into the `semper` package dir; scikit-build-core packs it into the wheel.
install(TARGETS _semper LIBRARY DESTINATION semper)
