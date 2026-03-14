# ============================================================
# Kernel library
#   - kernel/ has:
#       * CUDAKernelFunction/ (contains .cu/.h/.cuh/.cpp...)
#       * some extra headers directly under kernel/
# ============================================================

set(KERNEL_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(CUDAK_DIR   ${KERNEL_ROOT}/CUDAKernelFunction)
set(UTIL_DIR    ${CUDAK_DIR}/myUtility)

# Collect all compilation units under CUDAKernelFunction/
file(GLOB_RECURSE KERNEL_COMPILE_SRCS CONFIGURE_DEPENDS
    ${CUDAK_DIR}/*.cu
    ${CUDAK_DIR}/*.cpp
    ${CUDAK_DIR}/*.cxx
)

# Also track headers (not required for building, but nice for IDE)
file(GLOB_RECURSE KERNEL_HEADERS CONFIGURE_DEPENDS
    ${KERNEL_ROOT}/*.h
    ${KERNEL_ROOT}/*.hpp
    ${KERNEL_ROOT}/*.cuh
    ${CUDAK_DIR}/*.h
    ${CUDAK_DIR}/*.hpp
    ${CUDAK_DIR}/*.cuh
)

add_library(FunDEM_kernels STATIC
    ${KERNEL_COMPILE_SRCS}
    ${KERNEL_HEADERS}
)

# ------------------------------------------------------------
# CUDA Toolkit (provides cuda_runtime.h, etc.)
# ------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
target_link_libraries(FunDEM_kernels PUBLIC CUDA::cudart)

# Export include paths to apps
target_include_directories(FunDEM_kernels PUBLIC
    ${FUNDEM_ROOT}     # project root
    ${KERNEL_ROOT}     # kernel/ (for those extra .h files)
    ${CUDAK_DIR}       # kernel/CUDAKernelFunction/
    ${UTIL_DIR}        # kernel/CUDAKernelFunction/myUtility/
)

set_target_properties(FunDEM_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ${FUNDEM_ENABLE_RDC}
    POSITION_INDEPENDENT_CODE  ${FUNDEM_ENABLE_PIC}
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)