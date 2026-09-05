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

# Overridable so the binary-size cost of the descriptor seeding front-end can be
# measured: features2d and flann serve only AKAZE + BFMatcher, and calib3d is
# pulled in by a single cv::findHomography call in src/seeding/akaze_seed.cpp.
# Default reproduces the previous hard-coded list exactly.
set(SEMPER_OPENCV_BUILD_LIST "core,imgproc,imgcodecs,features2d,calib3d,flann"
    CACHE STRING "OpenCV modules compiled from source")
set(BUILD_LIST "${SEMPER_OPENCV_BUILD_LIST}" CACHE STRING "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

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
