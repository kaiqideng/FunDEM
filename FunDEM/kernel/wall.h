#pragma once
#include "myHostDeviceArray.h"
#include <cassert>

struct triangle
{
private:
    HostDeviceArray1D<int> index0_;
    HostDeviceArray1D<int> index1_;
    HostDeviceArray1D<int> index2_;
    HostDeviceArray1D<int> wallIndex_;
    HostDeviceArray1D<double3> circumcenter_;

    HostDeviceArray1D<int> hashIndex_;
    HostDeviceArray1D<int> hashValue_;

    size_t hostSize_ {0};
    size_t deviceSize_ {0};
    size_t blockDim_ {1};
    size_t gridDim_ {0};

    void assertAligned_() const
    {
#ifndef NDEBUG
        const size_t n = hostSize_;
        bool ok = true;

        ok = ok && (index0_.hostSize() == n);
        ok = ok && (index1_.hostSize() == n);
        ok = ok && (index2_.hostSize() == n);
        ok = ok && (wallIndex_.hostSize() == n);
        ok = ok && (circumcenter_.hostSize() == n);

        ok = ok && (hashIndex_.hostSize() == n);
        ok = ok && (hashValue_.hostSize() == n);

        assert(ok && "triangle: field host sizes mismatch!");
#endif
    }

    void updateGridDim_()
    {
        if (blockDim_ == 0) { gridDim_ = 0; return; }
        gridDim_ = (deviceSize_ + blockDim_ - 1) / blockDim_;
    }

public:
    triangle() = default;
    ~triangle() = default;

    triangle(const triangle&) = delete;
    triangle& operator=(const triangle&) = delete;

    triangle(triangle&&) noexcept = default;
    triangle& operator=(triangle&&) noexcept = default;

    size_t hostSize() const { return hostSize_; }
    size_t deviceSize() const { return deviceSize_; }

    void setBlockDim(const size_t blockDim)
    {
        if (blockDim == 0) return;
        blockDim_ = blockDim;
        updateGridDim_();
    }

    size_t blockDim() const { return blockDim_; }
    size_t gridDim() const { return gridDim_; }

    void eraseHost(const size_t index)
    {
        if (index < hostSize_)
        {
            index0_.eraseHost(index);
            index1_.eraseHost(index);
            index2_.eraseHost(index);
            wallIndex_.eraseHost(index);
            circumcenter_.eraseHost(index);

            hashIndex_.eraseHost(index);
            hashValue_.eraseHost(index);

            hostSize_--;
        }
    }

    void addHost(const int i0,
    const int i1,
    const int i2,
    const int w,
    const double3 c,
    const int hI,
    const int hV)
    {
        index0_.pushHost(i0);
        index1_.pushHost(i1);
        index2_.pushHost(i2);
        wallIndex_.pushHost(w);
        circumcenter_.pushHost(c);

        hashIndex_.pushHost(hI);
        hashValue_.pushHost(hV);

        hostSize_++;
        assertAligned_();
    }

    void copyFromHost(const triangle& other)
    {
        hostSize_ = other.hostSize_;
        blockDim_ = other.blockDim_;

        index0_.setHost(other.index0_.hostRef());
        index1_.setHost(other.index1_.hostRef());
        index2_.setHost(other.index2_.hostRef());
        wallIndex_.setHost(other.wallIndex_.hostRef());
        circumcenter_.setHost(other.circumcenter_.hostRef());

        hashIndex_.setHost(other.hashIndex_.hostRef());
        hashValue_.setHost(other.hashValue_.hostRef());

        assertAligned_();
    }

    void copyHostToDevice(cudaStream_t stream)
    {
        assertAligned_();

        index0_.copyHostToDevice(stream);
        index1_.copyHostToDevice(stream);
        index2_.copyHostToDevice(stream);
        wallIndex_.copyHostToDevice(stream);
        circumcenter_.copyHostToDevice(stream);

        hashIndex_.copyHostToDevice(stream);
        hashValue_.copyHostToDevice(stream);

        const size_t n = hostSize_;
        if (n > 0)
        {
            CUDA_CHECK(cudaMemsetAsync(hashIndex_.d_ptr, 0xFF, n * sizeof(int), stream));
            CUDA_CHECK(cudaMemsetAsync(hashValue_.d_ptr, 0xFF, n * sizeof(int), stream));
        }

        deviceSize_ = hostSize_;
        updateGridDim_();
    }

    void copyDeviceToHost(cudaStream_t stream)
    {
        index0_.copyDeviceToHost(stream);
        index1_.copyDeviceToHost(stream);
        index2_.copyDeviceToHost(stream);
        wallIndex_.copyDeviceToHost(stream);
        circumcenter_.copyDeviceToHost(stream);

        hashIndex_.copyDeviceToHost(stream);
        hashValue_.copyDeviceToHost(stream);

        hostSize_ = deviceSize_;
    }

    int* index0() { return index0_.d_ptr; }
    int* index1() { return index1_.d_ptr; }
    int* index2() { return index2_.d_ptr; }
    int* wallIndex() { return wallIndex_.d_ptr; }
    double3* circumcenter() { return circumcenter_.d_ptr; }

    int* hashIndex() { return hashIndex_.d_ptr; }
    int* hashValue() { return hashValue_.d_ptr; }

    const std::vector<int>& index0HostRef() { return index0_.hostRef(); }
    const std::vector<int>& index1HostRef() { return index1_.hostRef(); }
    const std::vector<int>& index2HostRef() { return index2_.hostRef(); }
    const std::vector<int>& wallIndexHostRef() { return wallIndex_.hostRef(); }

    std::vector<double3> circumcenterHostCopy() { return circumcenter_.getHostCopy(); }
    std::vector<int> hashIndexHostCopy() { return hashIndex_.getHostCopy(); }
    std::vector<int> hashValueHostCopy() { return hashValue_.getHostCopy(); }
};

struct vertex
{
private:
    HostDeviceArray1D<double3> localPosition_;
    HostDeviceArray1D<double3> globalPosition_;
    HostDeviceArray1D<int> wallIndex_;

    size_t hostSize_ {0};
    size_t deviceSize_ {0};
    size_t blockDim_ {1};
    size_t gridDim_ {0};

    void assertAligned_() const
    {
#ifndef NDEBUG
        const size_t n = hostSize_;
        bool ok = true;

        ok = ok && (localPosition_.hostSize() == n);
        ok = ok && (globalPosition_.hostSize() == n);
        ok = ok && (wallIndex_.hostSize() == n);

        assert(ok && "vertex: field host sizes mismatch!");
#endif
    }

    void updateGridDim_()
    {
        if (blockDim_ == 0) { gridDim_ = 0; return; }
        gridDim_ = (deviceSize_ + blockDim_ - 1) / blockDim_;
    }

public:
    vertex() = default;
    ~vertex() = default;

    vertex(const vertex&) = delete;
    vertex& operator=(const vertex&) = delete;

    vertex(vertex&&) noexcept = default;
    vertex& operator=(vertex&&) noexcept = default;

    size_t hostSize() const { return hostSize_; }
    size_t deviceSize() const { return deviceSize_; }

    void setBlockDim(const size_t blockDim)
    {
        if (blockDim == 0) return;
        blockDim_ = blockDim;
        updateGridDim_();
    }

    size_t blockDim() const { return blockDim_; }
    size_t gridDim() const { return gridDim_; }

    void eraseHost(const size_t index)
    {
        if (index < hostSize_)
        {
            localPosition_.eraseHost(index);
            globalPosition_.eraseHost(index);
            wallIndex_.eraseHost(index);

            hostSize_--;
        }
    }

    void addHost(const double3 localPos,
    const double3 globalPos,
    const int wallID)
    {
        localPosition_.pushHost(localPos);
        globalPosition_.pushHost(globalPos);
        wallIndex_.pushHost(wallID);

        hostSize_++;
        assertAligned_();
    }

    void copyFromHost(const vertex& other)
    {
        hostSize_ = other.hostSize_;
        blockDim_ = other.blockDim_;

        localPosition_.setHost(other.localPosition_.hostRef());
        globalPosition_.setHost(other.globalPosition_.hostRef());
        wallIndex_.setHost(other.wallIndex_.hostRef());

        assertAligned_();
    }

    void copyHostToDevice(cudaStream_t stream)
    {
        assertAligned_();

        localPosition_.copyHostToDevice(stream);
        globalPosition_.copyHostToDevice(stream);
        wallIndex_.copyHostToDevice(stream);

        deviceSize_ = hostSize_;
        updateGridDim_();
    }

    void copyDeviceToHost(cudaStream_t stream)
    {
        localPosition_.copyDeviceToHost(stream);
        globalPosition_.copyDeviceToHost(stream);
        wallIndex_.copyDeviceToHost(stream);

        hostSize_ = deviceSize_;
    }

    double3* localPosition() { return localPosition_.d_ptr; }
    double3* globalPosition() { return globalPosition_.d_ptr; }
    int* wallIndex() { return wallIndex_.d_ptr; }

    std::vector<double3> globalPositionHostCopy() { return globalPosition_.getHostCopy(); }

    const std::vector<double3>& localPositionHostRef() { return localPosition_.hostRef(); }
    const std::vector<int>& wallIndexHostRef() { return wallIndex_.hostRef(); }
};