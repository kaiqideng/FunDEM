# FunDEM/CMakeLists.txt
cmake_minimum_required(VERSION 3.22)

project(FunDEM LANGUAGES CXX CUDA)

# Make CUDA headers available to ALL targets (including pure C++ files)
find_package(CUDAToolkit REQUIRED)

# ------------------------------------------------------------
# C++ standard
# ------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate compile_commands.json (useful for clangd)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------------------------------------
# Options
# ------------------------------------------------------------
option(FUNDEM_ENABLE_RDC "Enable CUDA separable compilation (RDC)" ON)
option(FUNDEM_ENABLE_PIC "Build position independent code (PIC)" ON)

# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------
set(FUNDEM_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")

# ------------------------------------------------------------
# Add kernel subdirectory (defines target FunDEM_kernels)
# ------------------------------------------------------------
add_subdirectory(kernel)

# ------------------------------------------------------------
# Collect top-level app sources (ONLY files directly under FunDEM/)
# - Build one executable per top-level .cpp/.cxx containing main()
# - Optional: build other top-level sources (no main) into FunDEM_app_common
# ------------------------------------------------------------
file(GLOB FUNDEM_TOP_SRCS CONFIGURE_DEPENDS
    ${FUNDEM_ROOT}/*.cpp
    ${FUNDEM_ROOT}/*.cxx
    ${FUNDEM_ROOT}/*.cu
)

set(FUNDEM_MAIN_SRCS "")
set(FUNDEM_COMMON_SRCS "")

foreach(src ${FUNDEM_TOP_SRCS})
    # Heuristic: detect "main(" in the file
    # This is not a full parser, but works well in practice.
    file(STRINGS "${src}" _main_hits
        REGEX "[^A-Za-z0-9_]main[ \t\r\n]*\\("
        LIMIT_COUNT 2000
    )

    if (_main_hits)
        list(APPEND FUNDEM_MAIN_SRCS "${src}")
    else()
        list(APPEND FUNDEM_COMMON_SRCS "${src}")
    endif()
endforeach()

# ------------------------------------------------------------
# Common library for non-main top-level sources (optional)
# ------------------------------------------------------------
if (FUNDEM_COMMON_SRCS)
    add_library(FunDEM_app_common STATIC ${FUNDEM_COMMON_SRCS})

    # Link kernels so this target inherits CUDA device-link settings if needed
    target_link_libraries(FunDEM_app_common PUBLIC FunDEM_kernels)

    target_include_directories(FunDEM_app_common PUBLIC
        ${FUNDEM_ROOT}
    )

    set_target_properties(FunDEM_app_common PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        POSITION_INDEPENDENT_CODE ${FUNDEM_ENABLE_PIC}
    )

    if (FUNDEM_ENABLE_RDC)
        set_target_properties(FunDEM_app_common PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endif()
endif()

# ------------------------------------------------------------
# Create one executable for each top-level file that has main()
# ------------------------------------------------------------
if (NOT FUNDEM_MAIN_SRCS)
    message(FATAL_ERROR "No top-level *.cpp/*.cxx/*.cu containing main() found in ${FUNDEM_ROOT}")
endif()

foreach(main_src ${FUNDEM_MAIN_SRCS})
    get_filename_component(exe_name "${main_src}" NAME_WE)

    add_executable(${exe_name} ${main_src})

    # Root include dir (for headers like buildLevelSetParticle.h, etc.)
    target_include_directories(${exe_name} PRIVATE
        ${FUNDEM_ROOT}
    )

    # Link kernels (PUBLIC includes from kernels propagate)
    target_link_libraries(${exe_name} PRIVATE FunDEM_kernels)

    # Link common non-main code if it exists
    if (TARGET FunDEM_app_common)
        target_link_libraries(${exe_name} PRIVATE FunDEM_app_common)
    endif()

    set_target_properties(${exe_name} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        POSITION_INDEPENDENT_CODE ${FUNDEM_ENABLE_PIC}
    )

    if (FUNDEM_ENABLE_RDC)
        set_target_properties(${exe_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endif()
endforeach()