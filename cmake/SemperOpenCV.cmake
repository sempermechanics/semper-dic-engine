# Shared OpenCV-from-source options for the semper native package.
# Expects OPENCV_SRC_DIR to be set by the caller.

# Sparse OpenCV checkouts omit doc/ and data/ (~600 MB). OpenCV's CMakeLists
# always add_subdirectory() those paths, so plant no-op stubs when missing.
foreach(_stub doc data)
    if(NOT EXISTS "${OPENCV_SRC_DIR}/${_stub}/CMakeLists.txt")
        file(MAKE_DIRECTORY "${OPENCV_SRC_DIR}/${_stub}")
        file(WRITE "${OPENCV_SRC_DIR}/${_stub}/CMakeLists.txt"
             "# Auto-generated stub for sparse OpenCV checkout (see scripts/sparse-opencv.*)\n")
    endif()
endforeach()

# Modules semper_pipeline links (keep in sync with CMakeLists.txt).
# features2d (AKAZE, BFMatcher), calib3d (findHomography) and flann (a
# features2d dependency) serve the AKAZE global-motion front end; core, imgproc
# and imgcodecs the rest. v0.5.0 built without the first three, ~29% smaller
# on aarch64.
set(SEMPER_OPENCV_REQUIRED_MODULES core imgproc imgcodecs features2d calib3d flann)

# SEMPER_OPENCV_BUILD_LIST adds modules on top of the required ones; it can't
# remove them. Until v0.6.0 it was the whole list, and build trees configured
# by v0.3.0-v0.5.x cache it as core,imgproc,imgcodecs: replacing BUILD_LIST
# with it dropped features2d/calib3d/flann and failed the link.
set(SEMPER_OPENCV_BUILD_LIST ""
    CACHE STRING "Extra OpenCV modules to compile (comma-separated), beyond the ones the engine links")
string(REPLACE "," ";" _semper_cv_modules "${SEMPER_OPENCV_BUILD_LIST}")
list(TRANSFORM _semper_cv_modules STRIP)
list(PREPEND _semper_cv_modules ${SEMPER_OPENCV_REQUIRED_MODULES})
list(FILTER _semper_cv_modules EXCLUDE REGEX "^$")
list(REMOVE_DUPLICATES _semper_cv_modules)
string(REPLACE ";" "," _semper_cv_modules "${_semper_cv_modules}")
set(BUILD_LIST "${_semper_cv_modules}" CACHE STRING "" FORCE)
message(STATUS "Semper: OpenCV BUILD_LIST=${BUILD_LIST}")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Call after add_subdirectory(OpenCV): a required module that OpenCV still
# skipped (e.g. a cached BUILD_opencv_<module>=OFF) fails here, not at link.
function(semper_check_opencv_modules)
    foreach(_m IN LISTS SEMPER_OPENCV_REQUIRED_MODULES)
        if(NOT TARGET opencv_${_m})
            message(FATAL_ERROR
                "OpenCV module '${_m}' was not configured, but the engine links "
                "opencv_${_m}. BUILD_LIST=${BUILD_LIST}. Check the cache for "
                "BUILD_opencv_${_m}=OFF, or configure a fresh build directory.")
        endif()
    endforeach()
endfunction()

# Python extensions (and most host toolchains) use the dynamic CRT (/MD). OpenCV's
# default static CRT (/MT) on MSVC then fails with LNK2038 at link time.
if(MSVC AND NOT SEMPER_ANDROID)
    set(BUILD_WITH_STATIC_CRT OFF CACHE BOOL "" FORCE)
endif()

set(WITH_PNG  ON CACHE BOOL "" FORCE)
set(WITH_JPEG ON CACHE BOOL "" FORCE)
set(WITH_TIFF ON CACHE BOOL "" FORCE)
set(WITH_WEBP ON CACHE BOOL "" FORCE)
set(BUILD_ZLIB ON CACHE BOOL "" FORCE)
set(BUILD_PNG  ON CACHE BOOL "" FORCE)
set(BUILD_JPEG ON CACHE BOOL "" FORCE)
set(BUILD_TIFF ON CACHE BOOL "" FORCE)
set(BUILD_WEBP ON CACHE BOOL "" FORCE)

set(OPENCV_ENABLE_NONFREE OFF CACHE BOOL "" FORCE)
foreach(_flag
        BUILD_opencv_apps BUILD_TESTS BUILD_PERF_TESTS BUILD_EXAMPLES
        BUILD_ANDROID_EXAMPLES BUILD_DOCS BUILD_JAVA BUILD_opencv_java
        BUILD_opencv_python3 BUILD_opencv_python2
        BUILD_opencv_python_bindings_generator BUILD_opencv_gapi
        BUILD_opencv_dnn BUILD_opencv_ml BUILD_opencv_photo
        BUILD_opencv_video BUILD_opencv_objdetect BUILD_ITT
        WITH_PROTOBUF WITH_ADE WITH_IPP WITH_OPENCL
        WITH_QUIRC WITH_FFMPEG WITH_GSTREAMER WITH_OPENEXR
        WITH_OPENJPEG WITH_JASPER WITH_CUDA WITH_EIGEN WITH_1394)
    set(${_flag} OFF CACHE BOOL "" FORCE)
endforeach()

set(WITH_CAROTENE ON CACHE BOOL "" FORCE)
set(WITH_KLEIDICV OFF CACHE BOOL "" FORCE)
set(OPENCV_PYTHON_SKIP_DETECTION ON CACHE BOOL "" FORCE)

# APK zip path breaks cv::glob plugin discovery — disable all plugins.
set(PARALLEL_ENABLE_PLUGINS OFF CACHE BOOL "" FORCE)
set(VIDEOIO_ENABLE_PLUGINS OFF CACHE BOOL "" FORCE)
set(IMGCODECS_ENABLE_PLUGINS OFF CACHE BOOL "" FORCE)
set(HIGHGUI_ENABLE_PLUGINS OFF CACHE BOOL "" FORCE)
