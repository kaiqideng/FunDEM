#pragma once
#include "myHostDeviceArray.h"

class ballInteraction
{
public:
    ballInteraction() = default;
    ~ballInteraction() = default;

    ballInteraction(const ballInteraction&) = delete;
    ballInteraction& operator=(const ballInteraction&) = delete;

    ballInteraction(ballInteraction&&) noexcept = default;
    ballInteraction& operator=(ballInteraction&&) noexcept = default;

    void initialize(const size_t numBall, cudaStream_t stream)
    {
        if (numBall == 0) return;
        ballPointedNeighborCount_.allocateDevice(numBall, stream);
        ballPointedNeighborPrefixSum_.allocateDevice(numBall, stream);
        objectPointingInteractionStart_.allocateDevice(numBall, stream);
        objectPointingInteractionEnd_.allocateDevice(numBall, stream);
        cudaMemsetAsync(objectPointingInteractionStart_.d_ptr, 0xFF, numBall * sizeof(int), stream);
        cudaMemsetAsync(objectPointingInteractionEnd_.d_ptr, 0xFF, numBall * sizeof(int), stream);
    }

    void initialize_ballTriangle(const size_t numBall, const size_t numTriangle, cudaStream_t stream)
    {
        if (numBall == 0 || numTriangle == 0) return;
        ballPointedNeighborCount_.allocateDevice(numBall, stream);
        ballPointedNeighborPrefixSum_.allocateDevice(numBall, stream);
        objectPointingInteractionStart_.allocateDevice(numTriangle, stream);
        objectPointingInteractionEnd_.allocateDevice(numTriangle, stream);
        cudaMemsetAsync(objectPointingInteractionStart_.d_ptr, 0xFF, numTriangle * sizeof(int), stream);
        cudaMemsetAsync(objectPointingInteractionEnd_.d_ptr, 0xFF, numTriangle * sizeof(int), stream);
    }

    void build()
    {

    }

private:
    HostDeviceArray1D<double3> contactPoint_;
    HostDeviceArray1D<double3> contactNormal_;
    HostDeviceArray1D<double3> contactForce_;
    HostDeviceArray1D<double3> contactTorque_;
    HostDeviceArray1D<double> contactOverlap_;

    HostDeviceArray1D<double3> slidingSpring_;
    HostDeviceArray1D<double3> rollingSpring_;
    HostDeviceArray1D<double3> torsionSpring_;

    HostDeviceArray1D<int> ballPointed_;
    HostDeviceArray1D<int> ballPointing_;
    HostDeviceArray1D<int> hashValue_;
    HostDeviceArray1D<int> hashIndex_;
    HostDeviceArray1D<int> cancelFlag_;

    HostDeviceArray1D<double3> slidingSpring0_;
    HostDeviceArray1D<double3> rollingSpring0_;
    HostDeviceArray1D<double3> torsionSpring0_;
    HostDeviceArray1D<int> ballPointed0_;

    HostDeviceArray1D<int> ballPointedNeighborCount_;
    HostDeviceArray1D<int> ballPointedNeighborPrefixSum_;
    HostDeviceArray1D<int> objectPointingInteractionStart_;
    HostDeviceArray1D<int> objectPointingInteractionEnd_;

    size_t numPair_ {0};
    size_t numPair0_ {0};
}