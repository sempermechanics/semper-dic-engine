# OpenCL backend wiring, shared by the root package and the host test tree so
# both compile it identically. See docs/GPU_ACCELERATION.md.
#
# There is no find_package(OpenCL) and no link against libOpenCL on purpose:
# src/gpu/cl_api.hpp declares the entry points and cl_runtime.cpp dlopen()s the
# loader at runtime. A SEMPER_OPENCL=ON build therefore compiles with no OpenCL
# SDK installed and RUNS on a machine with no OpenCL at all -- the probe simply
# reports unavailable and every stage falls back to the CPU.

set(SEMPER_GPU_KERNELS
        ${SEMPER_ROOT_FOR_GPU}/src/gpu/kernels/probe.cl
)

# Bake the .cl sources into a header, resolving first-party includes. An
# OpenCL runtime compiler has no filesystem to include from, so
# <semper/kernels/canonical_math.h> has to be expanded at build time --
# recursively, and without deduplicating, because canonical_math.h includes
# canonical_reductions.inc twice (once per address space).
function(semper_add_opencl target)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    set(_gen_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
    set(_gen_hdr "${_gen_dir}/semper/gpu/embedded_kernels.hpp")

    add_custom_command(
            OUTPUT "${_gen_hdr}"
            COMMAND ${Python3_EXECUTABLE}
                    "${SEMPER_ROOT_FOR_GPU}/scripts/embed_cl_kernels.py"
                    --output "${_gen_hdr}"
                    --include-dir "${SEMPER_ROOT_FOR_GPU}/include"
                    ${SEMPER_GPU_KERNELS}
            DEPENDS ${SEMPER_GPU_KERNELS}
                    "${SEMPER_ROOT_FOR_GPU}/scripts/embed_cl_kernels.py"
                    "${SEMPER_ROOT_FOR_GPU}/include/semper/kernels/canonical_math.h"
                    "${SEMPER_ROOT_FOR_GPU}/include/semper/kernels/canonical_reductions.inc"
            COMMENT "Embedding OpenCL kernels (expanding first-party includes)"
            VERBATIM)

    add_custom_target(${target}_cl_kernels DEPENDS "${_gen_hdr}")
    add_dependencies(${target} ${target}_cl_kernels)

    target_sources(${target} PRIVATE ${SEMPER_ROOT_FOR_GPU}/src/gpu/cl_runtime.cpp)
    target_include_directories(${target} PRIVATE
            "${_gen_dir}"
            "${SEMPER_ROOT_FOR_GPU}/src")
    target_compile_definitions(${target} PRIVATE SEMPER_OPENCL=1)

    # dlopen/dlsym live in libdl on older glibc; harmless when already in libc.
    if(UNIX AND NOT APPLE)
        target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
    endif()
endfunction()
