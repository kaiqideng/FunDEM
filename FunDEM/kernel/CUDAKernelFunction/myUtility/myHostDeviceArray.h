#pragma once
#include <iomanip>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#ifndef NDEBUG
#include <stdexcept>
#endif

inline void check_cuda_error(cudaError_t result, const char* func, const char* file, int line)
{
#ifndef NDEBUG
    if (result != cudaSuccess) 
    {
        std::string msg = std::string("CUDA Error at ") + file + ":" + std::to_string(line) +
        " / " + func + ": " + cudaGetErrorString(result);
        // Throw exception to allow destructors (RAII) to clean up
        throw std::runtime_error(msg);
    }
#endif
}

#define CUDA_CHECK(val) check_cuda_error((val), #val, __FILE__, __LINE__)

template <typename T>
void debug_dump_device_array(const T* d_ptr,
std::size_t n,
const char* name,
cudaStream_t stream = 0)
{
    if (n == 0) return;

    if (stream == 0) {
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<T> h_buf(n);

    CUDA_CHECK(cudaMemcpy(h_buf.data(), d_ptr,
    n * sizeof(T),
    cudaMemcpyDeviceToHost));

    std::cout << "[DEBUG] " << name << " (first " << n << " values):\n";
    for (std::size_t i = 0; i < n; ++i) {
        std::cout << "  [" << i << "] = " << h_buf[i] << "\n";
    }
}

template <>
inline void debug_dump_device_array<double3>(const double3* d_ptr, 
std::size_t n,
const char* name,
cudaStream_t stream)
{
    if (n == 0) return;

    if (stream == 0) {
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<double3> h(n);
    cudaMemcpyAsync(h.data(), d_ptr, n * sizeof(double3), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::cout << name << ":\n";

    std::ios old_state(nullptr);
    old_state.copyfmt(std::cout);

    std::cout << std::scientific << std::setprecision(3);

    for (std::size_t i = 0; i < n; ++i)
    {
        const auto& v = h[i];
        std::cout << "  [" << i << "] = ("
                  << v.x << ", "
                  << v.y << ", "
                  << v.z << ")\n";
    }

    std::cout.copyfmt(old_state);
}

template <typename T>
struct HostDeviceArray1D
{
private:
    std::vector<T> h_data;     // host data
    size_t d_size {0};         // number of elements on device

public:
    T* d_ptr {nullptr};        // device pointer

    // ---------------------------------------------------------------------
    // Rule of Five
    // ---------------------------------------------------------------------
    HostDeviceArray1D() = default;

    ~HostDeviceArray1D()
    {
        // Destructor cannot rely on a user-provided stream; use sync free.
        releaseDeviceSync();
    }

    HostDeviceArray1D(const HostDeviceArray1D&) = delete;
    HostDeviceArray1D& operator=(const HostDeviceArray1D&) = delete;

    HostDeviceArray1D(HostDeviceArray1D&& other) noexcept
    {
        *this = std::move(other);
    }

    HostDeviceArray1D& operator=(HostDeviceArray1D&& other) noexcept
    {
        if (this != &other)
        {
            // We cannot safely cudaFreeAsync without a stream here; use sync free.
            releaseDeviceSync();

            h_data = std::move(other.h_data);

            d_ptr  = std::exchange(other.d_ptr,  nullptr);
            d_size = std::exchange(other.d_size, 0);
        }
        return *this;
    }

    // ---------------------------------------------------------------------
    // Sizes
    // ---------------------------------------------------------------------
    size_t hostSize() const { return h_data.size(); }
    size_t deviceSize() const { return d_size; }

    // ---------------------------------------------------------------------
    // Device memory management
    // ---------------------------------------------------------------------

    // Async free (preferred when you have a stream).
    void releaseDevice(cudaStream_t stream)
    {
        if (d_ptr)
        {
            CUDA_CHECK(cudaFreeAsync(d_ptr, stream));
            d_ptr = nullptr;
        }
        d_size = 0;
    }

    // Sync free (safe for destructor / move assignment).
    void releaseDeviceSync()
    {
        if (d_ptr)
        {
            CUDA_CHECK(cudaFree(d_ptr));
            d_ptr = nullptr;
        }
        d_size = 0;
    }

    // Allocate device array of size n.
    // - Frees old device memory first.
    // - Uses cudaMallocAsync/cudaFreeAsync on the provided stream.
    // - If n==0, releases device memory and returns.
    void allocateDevice(size_t n, cudaStream_t stream, bool zeroFill = true)
    {
        // Release old allocation if any.
        if (d_ptr) releaseDevice(stream);

        d_size = n;
        if (d_size == 0)
        {
            d_ptr = nullptr;
            return;
        }

        CUDA_CHECK(cudaMallocAsync((void**)&d_ptr, d_size * sizeof(T), stream));
        if (zeroFill)
        {
            CUDA_CHECK(cudaMemsetAsync(d_ptr, 0, d_size * sizeof(T), stream));
        }
    }

    // ---------------------------------------------------------------------
    // Host-side modifications
    // ---------------------------------------------------------------------
    void pushHost(const T& value)
    {
        h_data.push_back(value);
    }

    void insertHost(size_t index, const T& value)
    {
        if (index >= hostSize()) { h_data.push_back(value); return; }
        h_data.insert(h_data.begin() + static_cast<std::ptrdiff_t>(index), value);
    }

    void eraseHost(size_t index)
    {
        if (index >= h_data.size()) return;
        h_data.erase(h_data.begin() + static_cast<std::ptrdiff_t>(index));
    }

    void clearHost()
    {
        h_data.clear();
    }

    void reserveHost(size_t n)
    {
        h_data.reserve(n);
    }

    void resizeHost(size_t n)
    {
        h_data.resize(n);
    }

    const std::vector<T>& hostRef() const { return h_data; }

    void setHost(const std::vector<T>& newData) { h_data = newData; }

    // ---------------------------------------------------------------------
    // Host <-> Device transfer
    // ---------------------------------------------------------------------

    // Host -> Device (async).
    // Allocates device memory to match hostSize().
    void copyHostToDevice(cudaStream_t stream)
    {
        const size_t n = hostSize();

        if (n != d_size || d_ptr == nullptr)
        {
            // No need to zero-fill because memcpy overwrites the buffer.
            allocateDevice(n, stream, /*zeroFill=*/false);
        }

        if (n > 0)
        {
            CUDA_CHECK(cudaMemcpyAsync(d_ptr,
                                       h_data.data(),
                                       n * sizeof(T),
                                       cudaMemcpyHostToDevice,
                                       stream));
        }
    }

    // Device -> Host (async).
    // After calling this, synchronize the stream before reading h_data.
    void copyDeviceToHost(cudaStream_t stream)
    {
        if (d_size == 0 || d_ptr == nullptr) return;

        if (d_size != hostSize())
        {
            h_data.resize(d_size);
        }

        CUDA_CHECK(cudaMemcpyAsync(h_data.data(),
                                   d_ptr,
                                   d_size * sizeof(T),
                                   cudaMemcpyDeviceToHost,
                                   stream));
    }

    // Device -> Host (sync).
    std::vector<T> getHostCopy()
    {
        if (d_size > 0 && d_ptr)
        {
            if (d_size != hostSize())
            {
                h_data.resize(d_size);
            }
            CUDA_CHECK(cudaMemcpy(h_data.data(),
                                  d_ptr,
                                  d_size * sizeof(T),
                                  cudaMemcpyDeviceToHost));
        }
        return h_data;
    }
};