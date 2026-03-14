#pragma once
#include "myHostDeviceArray.h"

struct pair
{
private:
    // ---------------------------------------------------------------------
    // Data
    // ---------------------------------------------------------------------
    HostDeviceArray1D<int> objectPointed_;
    HostDeviceArray1D<int> objectPointing_;
    HostDeviceArray1D<int> hashValue_;
    HostDeviceArray1D<int> hashIndex_;
    HostDeviceArray1D<int> cancelFlag_;

    size_t deviceSize_ {0};

public:
    // ---------------------------------------------------------------------
    // Rule of Five
    // ---------------------------------------------------------------------
    pair() = default;
    ~pair() = default;

    pair(const pair&) = delete;
    pair& operator=(const pair&) = delete;

    pair(pair&&) noexcept = default;
    pair& operator=(pair&&) noexcept = default;

public:
    // ---------------------------------------------------------------------
    // Sizes
    // ---------------------------------------------------------------------
    size_t deviceSize() const { return deviceSize_; }

public:
    // ---------------------------------------------------------------------
    // Device buffer allocation (empty buffer)
    // ---------------------------------------------------------------------
    void allocateDevice(const size_t n, cudaStream_t stream, bool zeroFill = true)
    {
        objectPointed_.allocateDevice(n, stream, zeroFill);
        objectPointing_.allocateDevice(n, stream, zeroFill);
        hashValue_.allocateDevice(n, stream, zeroFill);
        hashIndex_.allocateDevice(n, stream, zeroFill);
        cancelFlag_.allocateDevice(n, stream, zeroFill);

        if (n > 0)
        {
            CUDA_CHECK(cudaMemsetAsync(objectPointed_.d_ptr, 0xFF, n * sizeof(int), stream));
            CUDA_CHECK(cudaMemsetAsync(objectPointing_.d_ptr, 0xFF, n * sizeof(int), stream));
            CUDA_CHECK(cudaMemsetAsync(hashValue_.d_ptr, 0xFF, n * sizeof(int), stream));
            CUDA_CHECK(cudaMemsetAsync(hashIndex_.d_ptr, 0xFF, n * sizeof(int), stream));
        }

        deviceSize_ = n;
    }

public:
    // ---------------------------------------------------------------------
    // Host operations
    // ---------------------------------------------------------------------
    void addHost(const int obj0,
    const int obj1)
    {
        objectPointed_.pushHost(obj0);
        objectPointing_.pushHost(obj1);
        hashValue_.pushHost(-1);
        hashIndex_.pushHost(-1);
        cancelFlag_.pushHost(0);
    }

public:
    // ---------------------------------------------------------------------
    // Transfers
    // ---------------------------------------------------------------------
    void copyHostToDevice(cudaStream_t stream)
    {
        objectPointed_.copyHostToDevice(stream);
        objectPointing_.copyHostToDevice(stream);
        hashValue_.copyHostToDevice(stream);
        hashIndex_.copyHostToDevice(stream);
        cancelFlag_.copyHostToDevice(stream);

        deviceSize_ = objectPointed_.hostSize();
    }

public:
    // ---------------------------------------------------------------------
    // Device pointers
    // ---------------------------------------------------------------------
    int* objectPointed() { return objectPointed_.d_ptr; }
    int* objectPointing() { return objectPointing_.d_ptr; }
    int* hashValue() { return hashValue_.d_ptr; }
    int* hashIndex() { return hashIndex_.d_ptr; }
    int* cancelFlag() { return cancelFlag_.d_ptr; }

public:
    // ---------------------------------------------------------------------
    // Host copies (sync D2H inside HostDeviceArray1D::getHostCopy)
    // ---------------------------------------------------------------------
    std::vector<int> objectPointedHostCopy() { return objectPointed_.getHostCopy(); }
    std::vector<int> objectPointingHostCopy() { return objectPointing_.getHostCopy(); }
    std::vector<int> hashIndexHostCopy() { return hashIndex_.getHostCopy(); }
    std::vector<int> cancelFlagHostCopy() { return cancelFlag_.getHostCopy(); }
};

struct objectPointed
{
private:
    // ---------------------------------------------------------------------
    // Data
    // ---------------------------------------------------------------------
    HostDeviceArray1D<int> neighborCount_;
    HostDeviceArray1D<int> neighborPrefixSum_;

    size_t deviceSize_ {0};

public:
    // ---------------------------------------------------------------------
    // Rule of Five
    // ---------------------------------------------------------------------
    objectPointed() = default;
    ~objectPointed() = default;

    objectPointed(const objectPointed&) = delete;
    objectPointed& operator=(const objectPointed&) = delete;

    objectPointed(objectPointed&&) noexcept = default;
    objectPointed& operator=(objectPointed&&) noexcept = default;

public:
    // ---------------------------------------------------------------------
    // Sizes
    // ---------------------------------------------------------------------
    size_t deviceSize() const { return deviceSize_; }

public:
    // ---------------------------------------------------------------------
    // Device buffer allocation (empty buffer)
    // ---------------------------------------------------------------------
    void allocateDevice(const size_t n, cudaStream_t stream, bool zeroFill = true)
    {
        neighborCount_.allocateDevice(n, stream, zeroFill);
        neighborPrefixSum_.allocateDevice(n, stream, zeroFill);
        deviceSize_ = n;
    }

public:
    // ---------------------------------------------------------------------
    // Device pointers
    // ---------------------------------------------------------------------
    int* neighborCount() { return neighborCount_.d_ptr; }
    int* neighborPrefixSum() { return neighborPrefixSum_.d_ptr; }

public:
    // ---------------------------------------------------------------------
    // Host copies (sync D2H inside HostDeviceArray1D::getHostCopy)
    // ---------------------------------------------------------------------
    std::vector<int> neighborCountHostCopy() { return neighborCount_.getHostCopy(); }
    std::vector<int> neighborPrefixSumHostCopy() { return neighborPrefixSum_.getHostCopy(); }

    size_t numNeighborPairs(cudaStream_t stream)
    {
        if (deviceSize_ == 0) return 0;
        cudaStreamSynchronize(stream);
        int last = 0;
        cudaMemcpy(&last, neighborPrefixSum_.d_ptr + (deviceSize_ - 1), sizeof(int), cudaMemcpyDeviceToHost);
        return static_cast<size_t>(last);
    }
};

struct objectPointing
{
private:
    // ---------------------------------------------------------------------
    // Data
    // ---------------------------------------------------------------------
    HostDeviceArray1D<int> interactionStart_;
    HostDeviceArray1D<int> interactionEnd_;

    size_t deviceSize_ {0};

public:
    // ---------------------------------------------------------------------
    // Rule of Five
    // ---------------------------------------------------------------------
    objectPointing() = default;
    ~objectPointing() = default;

    objectPointing(const objectPointing&) = delete;
    objectPointing& operator=(const objectPointing&) = delete;

    objectPointing(objectPointing&&) noexcept = default;
    objectPointing& operator=(objectPointing&&) noexcept = default;

public:
    // ---------------------------------------------------------------------
    // Sizes
    // ---------------------------------------------------------------------
    size_t deviceSize() const { return deviceSize_; }

public:
    // ---------------------------------------------------------------------
    // Device buffer allocation (empty buffer)
    // ---------------------------------------------------------------------
    void allocateDevice(const size_t n, cudaStream_t stream, bool zeroFill = true)
    {
        interactionStart_.allocateDevice(n, stream, zeroFill);
        interactionEnd_.allocateDevice(n, stream, zeroFill);

        if (n > 0)
        {
            CUDA_CHECK(cudaMemsetAsync(interactionStart_.d_ptr, 0xFF, n * sizeof(int), stream));
            CUDA_CHECK(cudaMemsetAsync(interactionEnd_.d_ptr, 0xFF, n * sizeof(int), stream));
        }

        deviceSize_ = n;
    }

public:
    // ---------------------------------------------------------------------
    // Device pointers
    // ---------------------------------------------------------------------
    int* interactionStart() { return interactionStart_.d_ptr; }
    int* interactionEnd() { return interactionEnd_.d_ptr; }

public:
    // ---------------------------------------------------------------------
    // Host copies (sync D2H inside HostDeviceArray1D::getHostCopy)
    // ---------------------------------------------------------------------
    std::vector<int> interactionStartHostCopy() { return interactionStart_.getHostCopy(); }
    std::vector<int> interactionEndHostCopy() { return interactionEnd_.getHostCopy(); }
};

struct spring
{
private:
    // ---------------------------------------------------------------------
    // Data
    // ---------------------------------------------------------------------
    HostDeviceArray1D<double3> sliding_;
    HostDeviceArray1D<double3> rolling_;
    HostDeviceArray1D<double3> torsion_;

    size_t deviceSize_ {0};

public:
    // ---------------------------------------------------------------------
    // Rule of Five
    // ---------------------------------------------------------------------
    spring() = default;
    ~spring() = default;

    spring(const spring&) = delete;
    spring& operator=(const spring&) = delete;

    spring(spring&&) noexcept = default;
    spring& operator=(spring&&) noexcept = default;

public:
    // ---------------------------------------------------------------------
    // Sizes
    // ---------------------------------------------------------------------
    size_t deviceSize() const { return deviceSize_; }

public:
    // ---------------------------------------------------------------------
    // Device buffer allocation (empty buffer)
    // ---------------------------------------------------------------------
    void allocateDevice(const size_t n, cudaStream_t stream, bool zeroFill = true)
    {
        sliding_.allocateDevice(n, stream, zeroFill);
        rolling_.allocateDevice(n, stream, zeroFill);
        torsion_.allocateDevice(n, stream, zeroFill);

        deviceSize_ = n;
    }

public:
    // ---------------------------------------------------------------------
    // Device pointers
    // ---------------------------------------------------------------------
    double3* sliding() { return sliding_.d_ptr; }
    double3* rolling() { return rolling_.d_ptr; }
    double3* torsion() { return torsion_.d_ptr; }

public:
    // ---------------------------------------------------------------------
    // Host copies (sync D2H inside HostDeviceArray1D::getHostCopy)
    // ---------------------------------------------------------------------
    std::vector<double3> slidingHostCopy() { return sliding_.getHostCopy(); }
    std::vector<double3> rollingHostCopy() { return rolling_.getHostCopy(); }
    std::vector<double3> torsionHostCopy() { return torsion_.getHostCopy(); }
};

struct contact
{
private:
    // ---------------------------------------------------------------------
    // Data
    // ---------------------------------------------------------------------
    HostDeviceArray1D<double3> point_;
    HostDeviceArray1D<double3> normal_;
    HostDeviceArray1D<double3> force_;
    HostDeviceArray1D<double3> torque_;
    HostDeviceArray1D<double> overlap_;

    size_t deviceSize_ {0};

public:
    // ---------------------------------------------------------------------
    // Rule of Five
    // ---------------------------------------------------------------------
    contact() = default;
    ~contact() = default;

    contact(const contact&) = delete;
    contact& operator=(const contact&) = delete;

    contact(contact&&) noexcept = default;
    contact& operator=(contact&&) noexcept = default;

public:
    // ---------------------------------------------------------------------
    // Sizes
    // ---------------------------------------------------------------------
    size_t deviceSize() const { return deviceSize_; }

public:
    // ---------------------------------------------------------------------
    // Device buffer allocation (empty buffer)
    // ---------------------------------------------------------------------
    void allocateDevice(const size_t n, cudaStream_t stream, bool zeroFill = true)
    {
        point_.allocateDevice(n, stream, zeroFill);
        normal_.allocateDevice(n, stream, zeroFill);
        force_.allocateDevice(n, stream, zeroFill);
        torque_.allocateDevice(n, stream, zeroFill);
        overlap_.allocateDevice(n, stream, zeroFill);

        deviceSize_ = n;
    }

public:
    // ---------------------------------------------------------------------
    // Device pointers
    // ---------------------------------------------------------------------
    double3* point() { return point_.d_ptr; }
    double3* normal() { return normal_.d_ptr; }
    double3* force() { return force_.d_ptr; }
    double3* torque() { return torque_.d_ptr; }
    double* overlap() { return overlap_.d_ptr; }

public:
    // ---------------------------------------------------------------------
    // Host copies (sync D2H inside HostDeviceArray1D::getHostCopy)
    // ---------------------------------------------------------------------
    std::vector<double3> pointHostCopy() { return point_.getHostCopy(); }
    std::vector<double3> normalHostCopy() { return normal_.getHostCopy(); }
    std::vector<double3> forceHostCopy() { return force_.getHostCopy(); }
    std::vector<double3> torqueHostCopy() { return torque_.getHostCopy(); }
    std::vector<double> overlapHostCopy() { return overlap_.getHostCopy(); }
};

struct bond
{
private:
    // ---------------------------------------------------------------------
    // Data
    // ---------------------------------------------------------------------
    HostDeviceArray1D<int> isBonded_;
    HostDeviceArray1D<double3> point_;
    HostDeviceArray1D<double3> normal_;
    HostDeviceArray1D<double> normalForce_;
    HostDeviceArray1D<double> torsionTorque_;
    HostDeviceArray1D<double3> shearForce_;
    HostDeviceArray1D<double3> bendingTorque_;
    HostDeviceArray1D<double> maxNormalStress_;
    HostDeviceArray1D<double> maxShearStress_;

    size_t hostSize_ {0};
    size_t deviceSize_ {0};
    
public:
    // ---------------------------------------------------------------------
    // Rule of Five
    // ---------------------------------------------------------------------
    bond() = default;
    ~bond() = default;

    bond(const bond&) = delete;
    bond& operator=(const bond&) = delete;

    bond(bond&&) noexcept = default;
    bond& operator=(bond&&) noexcept = default;

public:
    // ---------------------------------------------------------------------
    // Sizes
    // ---------------------------------------------------------------------
    size_t hostSize() const { return hostSize_; }
    size_t deviceSize() const { return deviceSize_; }

public:
    // ---------------------------------------------------------------------
    // Device buffer allocation (empty buffer)
    // ---------------------------------------------------------------------
    void allocateDevice(const size_t n, cudaStream_t stream, bool zeroFill = true)
    {
        isBonded_.allocateDevice(n, stream, zeroFill);
        point_.allocateDevice(n, stream, zeroFill);
        normal_.allocateDevice(n, stream, zeroFill);
        normalForce_.allocateDevice(n, stream, zeroFill);
        torsionTorque_.allocateDevice(n, stream, zeroFill);
        shearForce_.allocateDevice(n, stream, zeroFill);
        bendingTorque_.allocateDevice(n, stream, zeroFill);
        maxNormalStress_.allocateDevice(n, stream, zeroFill);
        maxShearStress_.allocateDevice(n, stream, zeroFill);

        deviceSize_ = n;
    }
    
public:
    // ---------------------------------------------------------------------
    // Host operations
    // ---------------------------------------------------------------------
    void addHost(const double3 p,
    const double3 n,
    const int isBonded = 1,
    const double Fn = 0.,
    const double Tn = 0.,
    const double3 Ft = make_double3(0., 0., 0.),
    const double3 Tt = make_double3(0., 0., 0.),
    const double sigma_max = 0.,
    const double tau_max = 0.)
    {
        isBonded_.pushHost(isBonded);
        point_.pushHost(p);
        normal_.pushHost(n);
        normalForce_.pushHost(Fn);
        torsionTorque_.pushHost(Tn);
        shearForce_.pushHost(Ft);
        bendingTorque_.pushHost(Tt);
        maxNormalStress_.pushHost(sigma_max);
        maxShearStress_.pushHost(tau_max);

        hostSize_++;
    }

public:
    // ---------------------------------------------------------------------
    // Transfers
    // ---------------------------------------------------------------------
    void copyHostToDevice(cudaStream_t stream)
    {
        isBonded_.copyHostToDevice(stream);
        point_.copyHostToDevice(stream);
        normal_.copyHostToDevice(stream);
        normalForce_.copyHostToDevice(stream);
        torsionTorque_.copyHostToDevice(stream);
        shearForce_.copyHostToDevice(stream);
        bendingTorque_.copyHostToDevice(stream);
        maxNormalStress_.copyHostToDevice(stream);
        maxShearStress_.copyHostToDevice(stream);

        deviceSize_ = hostSize_;
    }

    void copyDeviceToHost(cudaStream_t stream)
    {
        isBonded_.copyDeviceToHost(stream);
        point_.copyDeviceToHost(stream);
        normal_.copyDeviceToHost(stream);
        normalForce_.copyDeviceToHost(stream);
        torsionTorque_.copyDeviceToHost(stream);
        shearForce_.copyDeviceToHost(stream);
        bendingTorque_.copyDeviceToHost(stream);
        maxNormalStress_.copyDeviceToHost(stream);
        maxShearStress_.copyDeviceToHost(stream);

        hostSize_ = deviceSize_;
    }

public:
    // ---------------------------------------------------------------------
    // Device pointers
    // ---------------------------------------------------------------------
    int* isBonded() { return isBonded_.d_ptr; }
    double3* point() { return point_.d_ptr; }
    double3* normal() { return normal_.d_ptr; }
    double* normalForce() { return normalForce_.d_ptr; }
    double* torsionTorque() { return torsionTorque_.d_ptr; }
    double3* shearForce() { return shearForce_.d_ptr; }
    double3* bendingTorque() { return bendingTorque_.d_ptr; }
    double* maxNormalStress() { return maxNormalStress_.d_ptr; }
    double* maxShearStress() { return maxShearStress_.d_ptr; }

public:
    // ---------------------------------------------------------------------
    // Host copies (sync D2H inside HostDeviceArray1D::getHostCopy)
    // ---------------------------------------------------------------------
    std::vector<int> isBondedHostCopy() { return isBonded_.getHostCopy(); }
    std::vector<double3> pointHostCopy() { return point_.getHostCopy(); }
    std::vector<double3> normalHostCopy() { return normal_.getHostCopy(); }
    std::vector<double> normalForceHostCopy() { return normalForce_.getHostCopy(); }
    std::vector<double> torsionTorqueHostCopy() { return torsionTorque_.getHostCopy(); }
    std::vector<double3> shearForceHostCopy() { return shearForce_.getHostCopy(); }
    std::vector<double3> bendingTorqueHostCopy() { return bendingTorque_.getHostCopy(); }
    std::vector<double> maxNormalStressHostCopy() { return maxNormalStress_.getHostCopy(); }
    std::vector<double> maxShearStressHostCopy() { return maxShearStress_.getHostCopy(); }
};