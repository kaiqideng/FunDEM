# FunDEM/kernel/CMakeLists.txt
cmake_minimum_required(VERSION 3.22)

# NOTE:
# This CMakeLists builds a kernel static library that contains all CUDA/C++ sources
# under FunDEM/kernel/ (including CUDAKernelFunction/ and myUtility/).
# The root CMakeLists will add_subdirectory(kernel) and link executables to this target.

# ------------------------------------------------------------
# Collect sources (recursively) under this kernel directory
# ------------------------------------------------------------
set(FUNDEM_KERNEL_DIR ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE FUNDEM_KERNEL_SRCS CONFIGURE_DEPENDS
    ${FUNDEM_KERNEL_DIR}/*.cu
    ${FUNDEM_KERNEL_DIR}/*.cpp
    ${FUNDEM_KERNEL_DIR}/*.cxx
)

# Headers are not required for compilation, but listing them helps IDEs.
file(GLOB_RECURSE FUNDEM_KERNEL_HDRS CONFIGURE_DEPENDS
    ${FUNDEM_KERNEL_DIR}/*.h
    ${FUNDEM_KERNEL_DIR}/*.hpp
    ${FUNDEM_KERNEL_DIR}/*.cuh
)

add_library(FunDEM_kernels STATIC
    ${FUNDEM_KERNEL_SRCS}
    ${FUNDEM_KERNEL_HDRS}
)

# ------------------------------------------------------------
# Include directories (PUBLIC so executables inherit them)
# ------------------------------------------------------------
target_include_directories(FunDEM_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}                          # kernel/
    ${CMAKE_CURRENT_SOURCE_DIR}/CUDAKernelFunction       # kernel/CUDAKernelFunction/
    ${CMAKE_CURRENT_SOURCE_DIR}/CUDAKernelFunction/myUtility
)

target_include_directories(FunDEM_kernels PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(FunDEM_kernels PUBLIC CUDA::cudart)

# ------------------------------------------------------------
# Language standards (use properties instead of target_compile_features)
# ------------------------------------------------------------
set_target_properties(FunDEM_kernels PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

# ------------------------------------------------------------
# Optional RDC / PIC toggles inherited from root options if defined
# (Root CMakeLists defines FUNDEM_ENABLE_RDC / FUNDEM_ENABLE_PIC.)
# ------------------------------------------------------------
if (DEFINED FUNDEM_ENABLE_RDC AND FUNDEM_ENABLE_RDC)
    set_target_properties(FunDEM_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if (DEFINED FUNDEM_ENABLE_PIC)
    set_target_properties(FunDEM_kernels PROPERTIES POSITION_INDEPENDENT_CODE ${FUNDEM_ENABLE_PIC})
endif()