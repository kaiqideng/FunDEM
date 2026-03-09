cmake_minimum_required(VERSION 3.22)
project(DEMSPH LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================
# Options
# ============================================================
set(DEMSPH_CUDA_ARCH "native" CACHE STRING "CUDA arch: native or e.g. 75;86")
option(DEMSPH_ENABLE_RDC "Enable CUDA separable compilation (RDC)" ON)
option(DEMSPH_ENABLE_PIC "Build position independent code (PIC)" ON)

# ============================================================
# CUDA arch
# ============================================================
if (DEMSPH_CUDA_ARCH STREQUAL "native")
    set(CMAKE_CUDA_ARCHITECTURES native)
else()
    set(CMAKE_CUDA_ARCHITECTURES ${DEMSPH_CUDA_ARCH})
endif()
message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")

# ============================================================
# Project root
# ============================================================
set(DEMSPH_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# ============================================================
# Bring in kernel/ (defines target: DEMSPH_kernels)
# ============================================================
add_subdirectory(kernel)

# ============================================================
# App sources (top-level only)
#   - One executable per file that contains main()
#   - Other top-level sources are built into a common static library
# ============================================================
file(GLOB APP_SRCS CONFIGURE_DEPENDS
    ${DEMSPH_ROOT}/*.cpp
    ${DEMSPH_ROOT}/*.cxx
    ${DEMSPH_ROOT}/*.cu
)

set(APP_MAIN_SRCS "")
set(APP_COMMON_SRCS "")

foreach(src ${APP_SRCS})
    # Heuristic: detect a translation unit that contains main()
    file(STRINGS "${src}" _main_hits
        REGEX "[^A-Za-z0-9_]main[ \t\r\n]*\\("
        LIMIT_COUNT 2000
    )
    if (_main_hits)
        list(APPEND APP_MAIN_SRCS "${src}")
    else()
        list(APPEND APP_COMMON_SRCS "${src}")
    endif()
endforeach()

# ============================================================
# Common library for non-main top-level sources
# ============================================================
if (APP_COMMON_SRCS)
    add_library(DEMSPH_app_common STATIC ${APP_COMMON_SRCS})

    # Link against kernels so app_common inherits CUDA/C++ settings and include paths
    target_link_libraries(DEMSPH_app_common PUBLIC DEMSPH_kernels)

    # Optional: allow app_common to include headers from the project root
    target_include_directories(DEMSPH_app_common PUBLIC ${DEMSPH_ROOT})

    set_target_properties(DEMSPH_app_common PROPERTIES
        POSITION_INDEPENDENT_CODE ${DEMSPH_ENABLE_PIC}
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
    )
endif()

# ============================================================
# Create executables for every top-level file that has main()
# ============================================================
if (NOT APP_MAIN_SRCS)
    message(FATAL_ERROR "No top-level main() was found under ${DEMSPH_ROOT}")
endif()

foreach(main_src ${APP_MAIN_SRCS})
    get_filename_component(exe_name "${main_src}" NAME_WE)

    add_executable(${exe_name} ${main_src})

    # Inherit include dirs from DEMSPH_kernels via PUBLIC usage requirements
    target_link_libraries(${exe_name} PRIVATE DEMSPH_kernels)

    if (TARGET DEMSPH_app_common)
        target_link_libraries(${exe_name} PRIVATE DEMSPH_app_common)
    endif()

    set_target_properties(${exe_name} PROPERTIES
        POSITION_INDEPENDENT_CODE ${DEMSPH_ENABLE_PIC}
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
    )
endforeach()