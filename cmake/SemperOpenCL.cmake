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
        ${SEMPER_ROOT_FOR_GPU}/src/gpu/kernels/strain_vsg.cl
        ${SEMPER_ROOT_FOR_GPU}/src/gpu/kernels/hessian_prepass.cl
)

# Host-side dispatch. One file per stage; each is the CPU-side twin of a
# kernel above and is bound by the same bit-exactness contract.
set(SEMPER_GPU_SOURCES
        ${SEMPER_ROOT_FOR_GPU}/src/gpu/cl_runtime.cpp
        ${SEMPER_ROOT_FOR_GPU}/src/gpu/strain_dispatch.cpp
        ${SEMPER_ROOT_FOR_GPU}/src/gpu/hessian_dispatch.cpp
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

    target_sources(${target} PRIVATE ${SEMPER_GPU_SOURCES})

    # --- Why these files opt out of the target's -ffast-math ------------
    #
    # semper_pipeline is built -O3 -ffast-math (see the root CMakeLists), and
    # that is fine for orchestration code. It is NOT fine here. Dispatch
    # computes the scalars the kernel is handed -- radius_sq, grid_rad,
    # d_radius_sq, the expected-window point count -- and those must come out
    # bit-identical to the values the CPU reference computes in semper_math,
    # which is built -fno-fast-math -ffp-contract=off. Under -ffast-math the
    # compiler may reassociate or contract them, and the GPU result would then
    # differ from the CPU result for reasons that have nothing to do with the
    # device.
    #
    # <semper/kernels/canonical_math.h> makes this a hard error rather than a
    # silent drift: it #errors under __FAST_MATH__. Any dispatch file that
    # includes it -- Phases 3 through 6 all will -- would simply fail to
    # build without this.
    #
    # Chosen over the alternative of keeping the canonical header out of
    # dispatch entirely: that rule is invisible, unenforceable, and would have
    # to hold for every future stage. A per-source flag is checked by the
    # compiler on every build.
    if(NOT MSVC)
        # Directory-scoped, which is correct here: semper_add_opencl is
        # always called from the CMakeLists that defines ${target}.
        set_source_files_properties(${SEMPER_GPU_SOURCES}
                PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
    endif()
    target_include_directories(${target} PRIVATE
            "${_gen_dir}"
            "${SEMPER_ROOT_FOR_GPU}/src")
    target_compile_definitions(${target} PRIVATE SEMPER_OPENCL=1)

    # dlopen/dlsym live in libdl on older glibc; harmless when already in libc.
    if(UNIX AND NOT APPLE)
        target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
    endif()
endfunction()
