#include "particle.h"
#include "wall.h"
#include "interaction.h"
#include "boundary.h"
#include "CUDAKernelFunction/neighborSearchKernel.cuh"
#include "CUDAKernelFunction/ballNeighborSearchKernel.cuh"
#include "CUDAKernelFunction/contactKernel.cuh"
#include "CUDAKernelFunction/ballIntegrationKernel.cuh"
#include "CUDAKernelFunction/wallIntegrationKernel.cuh"
#include "CUDAKernelFunction/myUtility//buildHashStartEnd.cuh"
#include "CUDAKernelFunction/myUtility/myContactParameters.h"
#include "CUDAKernelFunction/myUtility/myFileEdit.h"
#include <fstream>
#include <sstream>

struct solidInteraction
{
    pair pair_;
    spring spring_;
    contact contact_;

    objectPointed objectPointed_;
    objectPointing objectPointing_;

    pair oldPair_;
    spring oldSpring_;

    size_t numActivated_ { 0 };

    void updateOldPairOldSpring(cudaStream_t stream)
    {
        if (numActivated_ == 0) return;
        if (numActivated_ > oldPair_.deviceSize())
        {
            oldPair_.allocateDevice(numActivated_, stream);
            oldSpring_.allocateDevice(numActivated_, stream);
        }
#ifndef NDEBUG
        CUDA_CHECK(cudaMemcpyAsync(oldPair_.objectPointed(), pair_.objectPointed(), numActivated_ * sizeof(int), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(oldPair_.objectPointing(), pair_.objectPointing(), numActivated_ * sizeof(int), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(oldPair_.hashIndex(), pair_.hashIndex(), numActivated_ * sizeof(int), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(oldPair_.hashValue(), pair_.hashValue(), numActivated_ * sizeof(int), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(oldPair_.cancelFlag(), pair_.cancelFlag(), numActivated_ * sizeof(int), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(oldSpring_.sliding(), spring_.sliding(), numActivated_ * sizeof(double3), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(oldSpring_.rolling(), spring_.rolling(), numActivated_ * sizeof(double3), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(oldSpring_.torsion(), spring_.torsion(), numActivated_ * sizeof(double3), cudaMemcpyDeviceToDevice, stream));
#else
        cudaMemcpyAsync(oldPair_.objectPointed(), pair_.objectPointed(), numActivated_ * sizeof(int), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(oldPair_.objectPointing(), pair_.objectPointing(), numActivated_ * sizeof(int), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(oldPair_.hashIndex(), pair_.hashIndex(), numActivated_ * sizeof(int), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(oldPair_.hashValue(), pair_.hashValue(), numActivated_ * sizeof(int), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(oldPair_.cancelFlag(), pair_.cancelFlag(), numActivated_ * sizeof(int), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(oldSpring_.sliding(), spring_.sliding(), numActivated_ * sizeof(double3), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(oldSpring_.rolling(), spring_.rolling(), numActivated_ * sizeof(double3), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(oldSpring_.torsion(), spring_.torsion(), numActivated_ * sizeof(double3), cudaMemcpyDeviceToDevice, stream);
#endif
    }

    void resizePair(cudaStream_t stream)
    {
        numActivated_ = objectPointed_.numNeighborPairs(stream);
        if (numActivated_ > pair_.deviceSize())
        {
            pair_.allocateDevice(numActivated_, stream);
            spring_.allocateDevice(numActivated_, stream);
            contact_.allocateDevice(numActivated_, stream);
        }
    }

    void buildObjectPointingInteractionStartEnd(size_t maxThread, cudaStream_t stream)
    {
        if (numActivated_ == 0) return;

#ifndef NDEBUG
        CUDA_CHECK(cudaMemcpyAsync(pair_.hashValue(), pair_.objectPointing(), pair_.deviceSize() * sizeof(int), cudaMemcpyDeviceToDevice, stream));
#else
        cudaMemcpyAsync(pair_.hashValue(), pair_.objectPointing(), pair_.deviceSize() * sizeof(int), cudaMemcpyDeviceToDevice, stream);
#endif
        size_t blockD = maxThread;
        if (numActivated_ < maxThread) blockD = numActivated_;
        size_t gridD = (numActivated_ + blockD - 1) / blockD;
        buildHashStartEnd(objectPointing_.interactionStart(), 
        objectPointing_.interactionEnd(), 
        pair_.hashIndex(), 
        pair_.hashValue(), 
        objectPointing_.deviceSize(), 
        numActivated_, 
        gridD, 
        blockD, 
        stream);
    }
};

struct periodicBoundary
{
    bool activatedFlag_ { false };
    HostDeviceArray1D<double3> dummyPosition_;
    solidInteraction dummyInteraction_;
};

struct bondedInteraction
{
    pair pair_;
    bond bond_;

    void add(const std::vector<int>& ob0,
    const std::vector<int>& ob1,
    const std::vector<double3>& p)
    {
        if (ob0.size() != ob1.size()) return;

        std::vector<int> existingPointed = pair_.objectPointedHostCopy();
        std::vector<int> existingPointing = pair_.objectPointingHostCopy();

        for (size_t i = 0; i < ob0.size(); ++i)
        {
            int i0 = ob0[i];
            int i1 = ob1[i];

            if (i0 < 0 || i1 < 0) continue;
            if (static_cast<size_t>(i0) >= p.size() || static_cast<size_t>(i1) >= p.size()) continue;
            if (i0 == i1) continue;

            int a = i0;
            int b = i1;
            if (a > b) std::swap(a, b);

            bool found = false;
            for (size_t j = 0; j < existingPointed.size(); ++j)
            {
                if (existingPointed[j] == a && existingPointing[j] == b)
                {
                    found = true;
                    break;
                }
            }
            if (found) continue;

            existingPointed.push_back(a);
            existingPointing.push_back(b);

            pair_.addHost(a, b);
            double3 n = p[a] - p[b];
            double3 n_norm = normalize(n);
            bond_.addHost(0.5 * (p[a] + p[b]), n_norm);
        }
    }
};

struct meshWall
{
    clump body_;
    triangle triangle_;
    vertex vertex_;
};

class DEMSolver
{
public:
    DEMSolver(cudaStream_t s)
    {
        maxThread_ = 256;
        stream_ = s;
    }

    ~DEMSolver() = default;

private:
    void upload()
    {
        int matID_max = *std::max_element(ball_.materialIDHostRef().begin(), ball_.materialIDHostRef().end());
        if (matID_max >= materialTable_.size()) 
        {
            materialRow row;
            row.poissonRatio = 0.;
            row.YoungsModulus = 0.;
            row.restitutionCoefficient = 1.;
            for (size_t i = 0; i < matID_max + 1 - materialTable_.size(); i++)
            {
                materialTable_.push_back(row);
            }
        }
        contactParameters_.buildFromTables(materialTable_, 
        frictionTable_, 
        linearStiffnessTable_, 
        bondTable_,
        stream_);

        ball_.copyHostToDevice(stream_);
        clump_.copyHostToDevice(stream_);
        meshWall_.body_.copyHostToDevice(stream_);
        meshWall_.triangle_.copyHostToDevice(stream_);
        meshWall_.vertex_.copyHostToDevice(stream_);

        bondedInteraction_.pair_.copyHostToDevice(stream_);
        bondedInteraction_.bond_.copyHostToDevice(stream_);

        if (ball_.deviceSize() > 0 && ballInteraction_.numActivated_ == 0)
        {
            ballInteraction_.objectPointed_.allocateDevice(ball_.deviceSize(), stream_);
            ballInteraction_.objectPointing_.allocateDevice(ball_.deviceSize(), stream_);
        }

        if (meshWall_.triangle_.deviceSize() > 0 && ballTriangleInteraction_.numActivated_ == 0)
        {
            ballTriangleInteraction_.objectPointed_.allocateDevice(ball_.deviceSize(), stream_);
            ballTriangleInteraction_.objectPointing_.allocateDevice(meshWall_.triangle_.deviceSize(), stream_);
        }

        if (periodicX_.activatedFlag_ && periodicX_.dummyInteraction_.numActivated_ == 0) 
        {
            periodicX_.dummyPosition_.allocateDevice(ball_.deviceSize(), stream_);
            periodicX_.dummyInteraction_.objectPointed_.allocateDevice(ball_.deviceSize(), stream_);
            periodicX_.dummyInteraction_.objectPointing_.allocateDevice(ball_.deviceSize(), stream_);
        }

        if (periodicY_.activatedFlag_ && periodicY_.dummyInteraction_.numActivated_ == 0) 
        {
            periodicY_.dummyPosition_.allocateDevice(ball_.deviceSize(), stream_);
            periodicY_.dummyInteraction_.objectPointed_.allocateDevice(ball_.deviceSize(), stream_);
            periodicY_.dummyInteraction_.objectPointing_.allocateDevice(ball_.deviceSize(), stream_);
        }

        if (periodicZ_.activatedFlag_ && periodicZ_.dummyInteraction_.numActivated_ == 0) 
        {
            periodicZ_.dummyPosition_.allocateDevice(ball_.deviceSize(), stream_);
            periodicZ_.dummyInteraction_.objectPointed_.allocateDevice(ball_.deviceSize(), stream_);
            periodicZ_.dummyInteraction_.objectPointing_.allocateDevice(ball_.deviceSize(), stream_);
        }

        if (periodicXY_.activatedFlag_ && periodicXY_.dummyInteraction_.numActivated_ == 0) 
        {
            periodicXY_.dummyPosition_.allocateDevice(ball_.deviceSize(), stream_);
            periodicXY_.dummyInteraction_.objectPointed_.allocateDevice(ball_.deviceSize(), stream_);
            periodicXY_.dummyInteraction_.objectPointing_.allocateDevice(ball_.deviceSize(), stream_);
        }

        if (periodicYZ_.activatedFlag_ && periodicYZ_.dummyInteraction_.numActivated_ == 0) 
        {
            periodicYZ_.dummyPosition_.allocateDevice(ball_.deviceSize(), stream_);
            periodicYZ_.dummyInteraction_.objectPointed_.allocateDevice(ball_.deviceSize(), stream_);
            periodicYZ_.dummyInteraction_.objectPointing_.allocateDevice(ball_.deviceSize(), stream_);
        }

        if (periodicXZ_.activatedFlag_ && periodicXZ_.dummyInteraction_.numActivated_ == 0) 
        {
            periodicXZ_.dummyPosition_.allocateDevice(ball_.deviceSize(), stream_);
            periodicXZ_.dummyInteraction_.objectPointed_.allocateDevice(ball_.deviceSize(), stream_);
            periodicXZ_.dummyInteraction_.objectPointing_.allocateDevice(ball_.deviceSize(), stream_);
        }

        if (periodicXYZ_.activatedFlag_ && periodicXYZ_.dummyInteraction_.numActivated_ == 0) 
        {
            periodicXYZ_.dummyPosition_.allocateDevice(ball_.deviceSize(), stream_);
            periodicXYZ_.dummyInteraction_.objectPointed_.allocateDevice(ball_.deviceSize(), stream_);
            periodicXYZ_.dummyInteraction_.objectPointing_.allocateDevice(ball_.deviceSize(), stream_);
        }
    }

    void initializeSpatialGrid(const double3 minBoundary, const double3 maxBoundary)
    {
        double cellSizeOneDim = 0.0;
        const std::vector<double> r = ball_.radiusHostRef();
        if (r.size() > 0) cellSizeOneDim = *std::max_element(r.begin(), r.end()) * 2.0;
        ballSpatialGrid_.set(minBoundary, maxBoundary, cellSizeOneDim, stream_);

        if (meshWall_.triangle_.deviceSize() > 0) 
        {
            for (size_t i = 0; i < meshWall_.triangle_.deviceSize(); i++)
            {
                double3 v0 = meshWall_.vertex_.localPositionHostRef()[meshWall_.triangle_.index0HostRef()[i]];
                double3 v1 = meshWall_.vertex_.localPositionHostRef()[meshWall_.triangle_.index1HostRef()[i]];
                double3 v2 = meshWall_.vertex_.localPositionHostRef()[meshWall_.triangle_.index2HostRef()[i]];
                auto getTriangleCircumcenter = [&](const double3& a, const double3& b, const double3& c)->double3
                {
                    double3 ab = b - a;
                    double3 ac = c - a;
                    double3 n  = cross(ab, ac);
                    double n2  = dot(n, n);
                    if (n2 < 1e-30)
                    {
                        return (a + b + c) / 3.0;
                    }
                    double3 term1 = cross(n,  ab) * dot(ac, ac);
                    double3 term2 = cross(ac, n ) * dot(ab, ab);
                    double invDen = 1.0 / (2.0 * n2);
                    return a + (term1 + term2) * invDen;
                };
                double3 c = getTriangleCircumcenter(v0, v1, v2);
                cellSizeOneDim = std::max(cellSizeOneDim, 2.0 * length(v0 - c));
                cellSizeOneDim = std::max(cellSizeOneDim, 2.0 * length(v1 - c));
                cellSizeOneDim = std::max(cellSizeOneDim, 2.0 * length(v2 - c));
            }
            triangleSpatialGrid_.set(minBoundary, maxBoundary, cellSizeOneDim, stream_);
        }
    }

protected:
    void DEMInitialize(const double3 minBoundary, const double3 maxBoundary)
    {
        ball_.setBlockDim(maxThread_ < ball_.hostSize() ? maxThread_ : ball_.hostSize());
        clump_.setBlockDim(maxThread_ < clump_.hostSize() ? maxThread_ : clump_.hostSize());
        meshWall_.body_.setBlockDim(maxThread_ < meshWall_.body_.hostSize() ? maxThread_ : meshWall_.body_.hostSize());
        meshWall_.triangle_.setBlockDim(maxThread_ < meshWall_.triangle_.hostSize() ? maxThread_ : meshWall_.triangle_.hostSize());
        meshWall_.vertex_.setBlockDim(maxThread_ < meshWall_.vertex_.hostSize() ? maxThread_ : meshWall_.vertex_.hostSize());

        upload();

        initializeSpatialGrid(minBoundary, maxBoundary);
    }

    void addDummyContactForceTorque(periodicBoundary& periodic, const int3 directionFlag, const double timeStep)
    {
        if (periodic.dummyPosition_.deviceSize() == 0) return;

        launchBuildDummyPosition(periodic.dummyPosition_.d_ptr,
        ball_.position(), 
        ballSpatialGrid_.minimumBoundary(), 
        ballSpatialGrid_.maximumBoundary(), 
        ballSpatialGrid_.cellSize(), 
        directionFlag, 
        ball_.deviceSize(), 
        ball_.gridDim(), 
        ball_.blockDim(), 
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif

        launchUpdateDummySpatialGridCellHashStartEnd(periodic.dummyPosition_.d_ptr, 
        ball_.hashIndex(), 
        ball_.hashValue(), 
        ballSpatialGrid_.cellHashStart(), 
        ballSpatialGrid_.cellHashEnd(), 
        ballSpatialGrid_.minimumBoundary(), 
        ballSpatialGrid_.maximumBoundary(), 
        ballSpatialGrid_.cellSize(), 
        ballSpatialGrid_.gridSize(), 
        ballSpatialGrid_.numGrids(),
        ball_.deviceSize(), 
        ball_.gridDim(), 
        ball_.blockDim(), 
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif

        periodic.dummyInteraction_.updateOldPairOldSpring(stream_);

        launchCountBallInteractions(periodic.dummyPosition_.d_ptr, 
        ball_.radius(), 
        ball_.inverseMass(), 
        ball_.clumpID(), 
        ball_.hashIndex(), 
        periodic.dummyInteraction_.objectPointed_.neighborCount(), 
        periodic.dummyInteraction_.objectPointed_.neighborPrefixSum(), 
        ballSpatialGrid_.cellHashStart(), 
        ballSpatialGrid_.cellHashEnd(), 
        ballSpatialGrid_.minimumBoundary(), 
        ballSpatialGrid_.cellSize(), 
        ballSpatialGrid_.gridSize(), 
        ball_.deviceSize(), 
        ball_.gridDim(), 
        ball_.blockDim(), 
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif

        periodic.dummyInteraction_.resizePair(stream_);

        launchWriteBallInteractions(periodic.dummyPosition_.d_ptr, 
        ball_.radius(), 
        ball_.inverseMass(),
        ball_.clumpID(),
        ball_.hashIndex(), 
        periodic.dummyInteraction_.objectPointed_.neighborPrefixSum(),
        periodic.dummyInteraction_.objectPointing_.interactionStart(),
        periodic.dummyInteraction_.objectPointing_.interactionEnd(),
        periodic.dummyInteraction_.spring_.sliding(),
        periodic.dummyInteraction_.spring_.rolling(),
        periodic.dummyInteraction_.spring_.torsion(),
        periodic.dummyInteraction_.pair_.objectPointed(),
        periodic.dummyInteraction_.pair_.objectPointing(),
        periodic.dummyInteraction_.oldSpring_.sliding(),
        periodic.dummyInteraction_.oldSpring_.rolling(),
        periodic.dummyInteraction_.oldSpring_.torsion(),
        periodic.dummyInteraction_.oldPair_.objectPointed(),
        periodic.dummyInteraction_.oldPair_.hashIndex(),
        ballSpatialGrid_.cellHashStart(), 
        ballSpatialGrid_.cellHashEnd(), 
        ballSpatialGrid_.minimumBoundary(),
        ballSpatialGrid_.cellSize(), 
        ballSpatialGrid_.gridSize(), 
        ball_.deviceSize(), 
        ball_.gridDim(), 
        ball_.blockDim(),
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif

        periodic.dummyInteraction_.buildObjectPointingInteractionStartEnd(maxThread_, stream_);

        if (periodic.dummyInteraction_.numActivated_ > 0) 
        {
            size_t blockD = maxThread_;
            if (periodic.dummyInteraction_.numActivated_ < maxThread_) blockD = periodic.dummyInteraction_.numActivated_ ;
            size_t gridD = (periodic.dummyInteraction_.numActivated_ + blockD - 1) / blockD;
            launchCalculateBallContactForceTorque(periodic.dummyPosition_.d_ptr,
            ball_.velocity(),
            ball_.angularVelocity(),
            ball_.radius(),
            ball_.inverseMass(),
            ball_.materialID(),
            periodic.dummyInteraction_.spring_.sliding(),
            periodic.dummyInteraction_.spring_.rolling(),
            periodic.dummyInteraction_.spring_.torsion(),
            periodic.dummyInteraction_.contact_.force(),
            periodic.dummyInteraction_.contact_.torque(),
            periodic.dummyInteraction_.contact_.point(),
            periodic.dummyInteraction_.contact_.normal(),
            periodic.dummyInteraction_.contact_.overlap(),
            periodic.dummyInteraction_.pair_.objectPointed(),
            periodic.dummyInteraction_.pair_.objectPointing(),
            timeStep,
            periodic.dummyInteraction_.numActivated_,
            gridD,
            blockD,
            stream_);
        }

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif

        launchSumContactForceTorque(periodic.dummyPosition_.d_ptr,
        ball_.force(),
        ball_.torque(),
        periodic.dummyInteraction_.objectPointed_.neighborPrefixSum(),
        periodic.dummyInteraction_.objectPointing_.interactionStart(),
        periodic.dummyInteraction_.objectPointing_.interactionEnd(),
        periodic.dummyInteraction_.contact_.force(),
        periodic.dummyInteraction_.contact_.torque(),
        periodic.dummyInteraction_.contact_.point(),
        periodic.dummyInteraction_.pair_.hashIndex(),
        ball_.deviceSize(),
        ball_.gridDim(),
        ball_.blockDim(),
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif
    }

    void addTriangleContactForceTorque(const double timeStep)
    {
        if (meshWall_.triangle_.deviceSize() == 0) return;

        launchUpdateSpatialGridCellHashStartEnd(meshWall_.triangle_.circumcenter(), 
        meshWall_.triangle_.hashIndex(), 
        meshWall_.triangle_.hashValue(), 
        triangleSpatialGrid_.cellHashStart(), 
        triangleSpatialGrid_.cellHashEnd(), 
        triangleSpatialGrid_.minimumBoundary(), 
        triangleSpatialGrid_.maximumBoundary(), 
        triangleSpatialGrid_.cellSize(), 
        triangleSpatialGrid_.gridSize(), 
        triangleSpatialGrid_.numGrids(),
        meshWall_.triangle_.deviceSize(), 
        meshWall_.triangle_.gridDim(), 
        meshWall_.triangle_.blockDim(), 
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif

        ballTriangleInteraction_.updateOldPairOldSpring(stream_);

        launchCountBallTriangleInteractions(ball_.position(), 
        ball_.radius(), 
        ballTriangleInteraction_.objectPointed_.neighborCount(),
        ballTriangleInteraction_.objectPointed_.neighborPrefixSum(),
        meshWall_.triangle_.index0(), 
        meshWall_.triangle_.index1(), 
        meshWall_.triangle_.index2(),
        meshWall_.triangle_.hashIndex(), 
        meshWall_.vertex_.globalPosition(),
        triangleSpatialGrid_.cellHashStart(), 
        triangleSpatialGrid_.cellHashEnd(), 
        triangleSpatialGrid_.minimumBoundary(), 
        triangleSpatialGrid_.cellSize(), 
        triangleSpatialGrid_.gridSize(), 
        ball_.deviceSize(), 
        ball_.gridDim(), 
        ball_.blockDim(), 
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif

        ballTriangleInteraction_.resizePair(stream_);

        launchWriteBallTriangleInteractions(ball_.position(), 
        ball_.radius(), 
        ballTriangleInteraction_.objectPointed_.neighborPrefixSum(),
        meshWall_.triangle_.index0(), 
        meshWall_.triangle_.index1(), 
        meshWall_.triangle_.index2(),
        meshWall_.triangle_.hashIndex(),
        ballTriangleInteraction_.objectPointing_.interactionStart(),
        ballTriangleInteraction_.objectPointing_.interactionEnd(),
        meshWall_.vertex_.globalPosition(),
        ballTriangleInteraction_.spring_.sliding(),
        ballTriangleInteraction_.spring_.rolling(),
        ballTriangleInteraction_.spring_.torsion(),
        ballTriangleInteraction_.pair_.objectPointed(),
        ballTriangleInteraction_.pair_.objectPointing(),
        ballTriangleInteraction_.oldSpring_.sliding(),
        ballTriangleInteraction_.oldSpring_.rolling(),
        ballTriangleInteraction_.oldSpring_.torsion(),
        ballTriangleInteraction_.oldPair_.objectPointed(),
        ballTriangleInteraction_.oldPair_.hashIndex(),
        triangleSpatialGrid_.cellHashStart(), 
        triangleSpatialGrid_.cellHashEnd(),
        triangleSpatialGrid_.minimumBoundary(),
        triangleSpatialGrid_.cellSize(),
        triangleSpatialGrid_.gridSize(), 
        ball_.deviceSize(), 
        ball_.gridDim(),
        ball_.blockDim(),
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif

        ballTriangleInteraction_.buildObjectPointingInteractionStartEnd(maxThread_, stream_);

        if (ballTriangleInteraction_.numActivated_ > 0)
        {            
            launchAddBallWallContactForceTorque(ball_.position(),
            ball_.velocity(),
            ball_.angularVelocity(),
            ball_.force(),
            ball_.torque(),
            ball_.radius(),
            ball_.inverseMass(),
            ball_.materialID(),
            ballTriangleInteraction_.objectPointed_.neighborPrefixSum(),
            meshWall_.body_.position(),
            meshWall_.body_.velocity(),
            meshWall_.body_.angularVelocity(),
            meshWall_.body_.materialID(),
            meshWall_.triangle_.index0(),
            meshWall_.triangle_.index1(),
            meshWall_.triangle_.index2(),
            meshWall_.triangle_.wallIndex(),
            meshWall_.vertex_.globalPosition(),
            ballTriangleInteraction_.spring_.sliding(),
            ballTriangleInteraction_.spring_.rolling(),
            ballTriangleInteraction_.spring_.torsion(),
            ballTriangleInteraction_.contact_.force(),
            ballTriangleInteraction_.contact_.torque(),
            ballTriangleInteraction_.contact_.point(),
            ballTriangleInteraction_.contact_.normal(),
            ballTriangleInteraction_.contact_.overlap(),
            ballTriangleInteraction_.pair_.objectPointed(),
            ballTriangleInteraction_.pair_.objectPointing(),
            ballTriangleInteraction_.pair_.cancelFlag(),
            timeStep,
            ball_.deviceSize(),
            ball_.gridDim(),
            ball_.blockDim(),
            stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif
        }

        launchWallIntegration(meshWall_.body_.position(),
        meshWall_.body_.velocity(),
        meshWall_.body_.angularVelocity(),
        meshWall_.body_.orientation(),
        timeStep,
        meshWall_.body_.deviceSize(),
        meshWall_.body_.gridDim(),
        meshWall_.body_.blockDim(),
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif

        launchUpdateWallVertexGlobalPosition(meshWall_.vertex_.globalPosition(), 
        meshWall_.vertex_.localPosition(), 
        meshWall_.vertex_.wallIndex(), 
        meshWall_.body_.position(), 
        meshWall_.body_.orientation(), 
        meshWall_.vertex_.deviceSize(), 
        meshWall_.vertex_.gridDim(), 
        meshWall_.vertex_.blockDim(), 
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif

        launchUpdateTriangleCircumcenter(meshWall_.triangle_.circumcenter(),
        meshWall_.triangle_.index0(),
        meshWall_.triangle_.index1(),
        meshWall_.triangle_.index2(),
        meshWall_.vertex_.globalPosition(),
        meshWall_.triangle_.deviceSize(),
        meshWall_.triangle_.gridDim(),
        meshWall_.triangle_.blockDim(),
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif
    }

    void calculateBallContactForceTorque(const double timeStep)
    {
#ifndef NDEBUG
CUDA_CHECK(cudaMemsetAsync(ball_.force(), 0, ball_.deviceSize() * sizeof(double3), stream_));
CUDA_CHECK(cudaMemsetAsync(ball_.torque(), 0, ball_.deviceSize() * sizeof(double3), stream_));
#else
        cudaMemsetAsync(ball_.force(), 0, ball_.deviceSize() * sizeof(double3), stream_);
        cudaMemsetAsync(ball_.torque(), 0, ball_.deviceSize() * sizeof(double3), stream_);
#endif

        if (ballInteraction_.numActivated_ > 0) 
        {
            size_t blockD = maxThread_;
            if (ballInteraction_.numActivated_ < maxThread_) blockD = ballInteraction_.numActivated_ ;
            size_t gridD = (ballInteraction_.numActivated_ + blockD - 1) / blockD;
            launchCalculateBallContactForceTorque(ball_.position(),
            ball_.velocity(),
            ball_.angularVelocity(),
            ball_.radius(),
            ball_.inverseMass(),
            ball_.materialID(),
            ballInteraction_.spring_.sliding(),
            ballInteraction_.spring_.rolling(),
            ballInteraction_.spring_.torsion(),
            ballInteraction_.contact_.force(),
            ballInteraction_.contact_.torque(),
            ballInteraction_.contact_.point(),
            ballInteraction_.contact_.normal(),
            ballInteraction_.contact_.overlap(),
            ballInteraction_.pair_.objectPointed(),
            ballInteraction_.pair_.objectPointing(),
            timeStep,
            ballInteraction_.numActivated_ ,
            gridD,
            blockD,
            stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif
        }

        size_t numBondedInteraction = bondedInteraction_.bond_.deviceSize();
        if (numBondedInteraction > 0)
        {
            size_t blockD = maxThread_;
            if (numBondedInteraction < maxThread_) blockD = numBondedInteraction;
            size_t gridD = (numBondedInteraction + blockD - 1) / blockD;
            launchAddBondedForceTorque(ball_.position(),
            ball_.velocity(),
            ball_.angularVelocity(),
            ball_.force(),
            ball_.torque(),
            ball_.radius(),
            ball_.materialID(),
            ballInteraction_.objectPointed_.neighborPrefixSum(),
            ballInteraction_.contact_.force(),
            ballInteraction_.contact_.torque(),
            ballInteraction_.contact_.point(),
            ballInteraction_.contact_.normal(),
            ballInteraction_.pair_.objectPointing(),
            bondedInteraction_.bond_.point(),
            bondedInteraction_.bond_.normal(),
            bondedInteraction_.bond_.shearForce(),
            bondedInteraction_.bond_.bendingTorque(),
            bondedInteraction_.bond_.normalForce(),
            bondedInteraction_.bond_.torsionTorque(),
            bondedInteraction_.bond_.maxNormalStress(),
            bondedInteraction_.bond_.maxShearStress(),
            bondedInteraction_.bond_.isBonded(),
            bondedInteraction_.pair_.objectPointed(),
            bondedInteraction_.pair_.objectPointing(),
            timeStep,
            numBondedInteraction,
            gridD,
            blockD,
            stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif
        }

        launchSumContactForceTorque(ball_.position(),
        ball_.force(),
        ball_.torque(),
        ballInteraction_.objectPointed_.neighborPrefixSum(),
        ballInteraction_.objectPointing_.interactionStart(),
        ballInteraction_.objectPointing_.interactionEnd(),
        ballInteraction_.contact_.force(),
        ballInteraction_.contact_.torque(),
        ballInteraction_.contact_.point(),
        ballInteraction_.pair_.hashIndex(),
        ball_.deviceSize(),
        ball_.gridDim(),
        ball_.blockDim(),
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif
    }

    void ballNeighborSearch()
    {
        launchUpdateSpatialGridCellHashStartEnd(ball_.position(), 
        ball_.hashIndex(), 
        ball_.hashValue(), 
        ballSpatialGrid_.cellHashStart(), 
        ballSpatialGrid_.cellHashEnd(), 
        ballSpatialGrid_.minimumBoundary(), 
        ballSpatialGrid_.maximumBoundary(), 
        ballSpatialGrid_.cellSize(), 
        ballSpatialGrid_.gridSize(), 
        ballSpatialGrid_.numGrids(),
        ball_.deviceSize(), 
        ball_.gridDim(), 
        ball_.blockDim(), 
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif

        ballInteraction_.updateOldPairOldSpring(stream_);

        launchCountBallInteractions(ball_.position(), 
        ball_.radius(), 
        ball_.inverseMass(), 
        ball_.clumpID(), 
        ball_.hashIndex(), 
        ballInteraction_.objectPointed_.neighborCount(), 
        ballInteraction_.objectPointed_.neighborPrefixSum(), 
        ballSpatialGrid_.cellHashStart(), 
        ballSpatialGrid_.cellHashEnd(), 
        ballSpatialGrid_.minimumBoundary(), 
        ballSpatialGrid_.cellSize(), 
        ballSpatialGrid_.gridSize(), 
        ball_.deviceSize(), 
        ball_.gridDim(), 
        ball_.blockDim(), 
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif

        ballInteraction_.resizePair(stream_);

        launchWriteBallInteractions(ball_.position(), 
        ball_.radius(), 
        ball_.inverseMass(),
        ball_.clumpID(),
        ball_.hashIndex(), 
        ballInteraction_.objectPointed_.neighborPrefixSum(),
        ballInteraction_.objectPointing_.interactionStart(),
        ballInteraction_.objectPointing_.interactionEnd(),
        ballInteraction_.spring_.sliding(),
        ballInteraction_.spring_.rolling(),
        ballInteraction_.spring_.torsion(),
        ballInteraction_.pair_.objectPointed(),
        ballInteraction_.pair_.objectPointing(),
        ballInteraction_.oldSpring_.sliding(),
        ballInteraction_.oldSpring_.rolling(),
        ballInteraction_.oldSpring_.torsion(),
        ballInteraction_.oldPair_.objectPointed(),
        ballInteraction_.oldPair_.hashIndex(),
        ballSpatialGrid_.cellHashStart(), 
        ballSpatialGrid_.cellHashEnd(), 
        ballSpatialGrid_.minimumBoundary(),
        ballSpatialGrid_.cellSize(), 
        ballSpatialGrid_.gridSize(), 
        ball_.deviceSize(), 
        ball_.gridDim(), 
        ball_.blockDim(),
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif

        ballInteraction_.buildObjectPointingInteractionStartEnd(maxThread_, stream_);
    }

    void ballClumpIntegration1stHalf(const double3 gravity, const double halfTimeStep)
    {
        if (clump_.deviceSize() > 0)
        {
            launchClump1stHalfIntegration(clump_.position(),
            clump_.velocity(),
            clump_.angularVelocity(),
            clump_.force(),
            clump_.torque(),
            clump_.inverseMass(),
            clump_.orientation(),
            clump_.inverseInertiaTensor(),
            clump_.pebbleStart(),
            clump_.pebbleEnd(),
            ball_.position(),
            ball_.force(),
            ball_.torque(),
            gravity,
            halfTimeStep,
            clump_.deviceSize(),
            clump_.gridDim(),
            clump_.blockDim(),
            stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif

            launchSetPebblePositionVelocityAngularVelocityKernel(ball_.position(),
            ball_.velocity(),
            ball_.angularVelocity(),
            ball_.localPosition(),
            ball_.clumpID(),
            clump_.position(),
            clump_.velocity(),
            clump_.angularVelocity(),
            clump_.orientation(),
            ball_.deviceSize(),
            ball_.gridDim(),
            ball_.blockDim(),
            stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif
        }

        launchBall1stHalfIntegration(ball_.position(),
        ball_.velocity(),
        ball_.angularVelocity(),
        ball_.force(),
        ball_.torque(),
        ball_.radius(),
        ball_.inverseMass(),
        ball_.clumpID(),
        gravity,
        halfTimeStep,
        ball_.deviceSize(),
        ball_.gridDim(),
        ball_.blockDim(),
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif
    }

    void ballClumpIntegration2ndHalf(const double3 gravity, const double halfTimeStep)
    {
        if (clump_.deviceSize() > 0)
        {
            launchClump2ndHalfIntegration(clump_.position(),
            clump_.velocity(),
            clump_.angularVelocity(),
            clump_.force(),
            clump_.torque(),
            clump_.inverseMass(),
            clump_.orientation(),
            clump_.inverseInertiaTensor(),
            clump_.pebbleStart(),
            clump_.pebbleEnd(),
            ball_.position(),
            ball_.force(),
            ball_.torque(),
            gravity,
            halfTimeStep,
            clump_.deviceSize(),
            clump_.gridDim(),
            clump_.blockDim(),
            stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif

            launchSetPebblePositionVelocityAngularVelocityKernel(ball_.position(),
            ball_.velocity(),
            ball_.angularVelocity(),
            ball_.localPosition(),
            ball_.clumpID(),
            clump_.position(),
            clump_.velocity(),
            clump_.angularVelocity(),
            clump_.orientation(),
            ball_.deviceSize(),
            ball_.gridDim(),
            ball_.blockDim(),
            stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif
        }

        launchBall2ndHalfIntegration(ball_.velocity(),
        ball_.angularVelocity(),
        ball_.force(),
        ball_.torque(),
        ball_.radius(),
        ball_.inverseMass(),
        ball_.clumpID(),
        gravity,
        halfTimeStep,
        ball_.deviceSize(),
        ball_.gridDim(),
        ball_.blockDim(),
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif
    }

    void outputBallVTU(const std::string &dir, const size_t iFrame, const size_t iStep, const double time)
    {
        MKDIR(dir.c_str());
        std::ostringstream fname;
        fname << dir << "/ball_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";
        std::ofstream out(fname.str().c_str());
        if (!out) throw std::runtime_error("Cannot open " + fname.str());
        out << std::fixed << std::setprecision(10);

        const size_t N = ball_.deviceSize();
        std::vector<double3> p = ball_.positionHostCopy();
        std::vector<double3> v = ball_.velocityHostCopy();
        std::vector<double3> a = ball_.angularVelocityHostCopy();
        const std::vector<double>& r = ball_.radiusHostRef();
        const std::vector<int>& materialID = ball_.materialIDHostRef();
        const std::vector<int>& clumpID = ball_.clumpIDHostRef();
        
        out << "<?xml version=\"1.0\"?>\n"
            "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
            "  <UnstructuredGrid>\n";

        out << "    <FieldData>\n"
            "      <DataArray type=\"Float32\" Name=\"TIME\"  NumberOfTuples=\"1\" format=\"ascii\"> "
            << time << " </DataArray>\n"
            "      <DataArray type=\"Int32\"   Name=\"STEP\"  NumberOfTuples=\"1\" format=\"ascii\"> "
            << iStep << " </DataArray>\n"
            "    </FieldData>\n";

        out << "    <Piece NumberOfPoints=\"" << N
            << "\" NumberOfCells=\"" << N << "\">\n";

        out << "      <Points>\n"
            "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) {
            out << ' ' << p[i].x << ' ' << p[i].y << ' ' << p[i].z;
        }
        out << "\n        </DataArray>\n"
            "      </Points>\n";

        out << "      <Cells>\n"
            "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << i;
        out << "\n        </DataArray>\n"
            "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
        for (int i = 1; i <= N; ++i) out << ' ' << i;
        out << "\n        </DataArray>\n"
            "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << " 1";          // 1 = VTK_VERTEX
        out << "\n        </DataArray>\n"
            "      </Cells>\n";

        out << "      <PointData Scalars=\"radius\">\n";

        out << "        <DataArray type=\"Float32\" Name=\"radius\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << r[i];
        out << "\n        </DataArray>\n";

        out << "        <DataArray type=\"Int32\" Name=\"material ID\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << materialID[i];
        out << "\n        </DataArray>\n";

        out << "        <DataArray type=\"Int32\" Name=\"clump ID\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << clumpID[i];
        out << "\n        </DataArray>\n";

        const struct {
            const char* name;
            const std::vector<double3>& vec;
        } vec3s[] = {
            { "velocity"       , v     },
            { "angularVelocity", a     }
        };
        for (size_t k = 0; k < sizeof(vec3s) / sizeof(vec3s[0]); ++k) {
            out << "        <DataArray type=\"Float32\" Name=\"" << vec3s[k].name
                << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
            const std::vector<double3>& v = vec3s[k].vec;
            for (size_t i = 0; i < v.size(); ++i)
                out << ' ' << v[i].x << ' ' << v[i].y << ' ' << v[i].z;
            out << "\n        </DataArray>\n";
        }

        out << "      </PointData>\n"
            "    </Piece>\n"
            "  </UnstructuredGrid>\n"
            "</VTKFile>\n";
    }

    void outputMeshWallVTU(const std::string& dir, const size_t iFrame, const size_t iStep, const double time)
    {
        MKDIR(dir.c_str());

        std::ostringstream fname;
        fname << dir << "/triangleWall_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";

        std::ofstream out(fname.str().c_str());
        if (!out) {
            throw std::runtime_error("Cannot open " + fname.str());
        }

        out << std::fixed << std::setprecision(10);

        std::vector<double3> verts = meshWall_.vertex_.globalPositionHostCopy();
        const size_t numPoints = verts.size();

        const std::vector<int>& tri_i0 = meshWall_.triangle_.index0HostRef();
        const std::vector<int>& tri_i1 = meshWall_.triangle_.index1HostRef();
        const std::vector<int>& tri_i2 = meshWall_.triangle_.index2HostRef();
        const size_t numTris = tri_i0.size();

        const std::vector<int> triWallId = meshWall_.triangle_.wallIndexHostRef();

        std::vector<int> triMaterialId(numTris, 0);
        const std::vector<int> wallMaterial = meshWall_.body_.materialIDHostRef();
        if (!wallMaterial.empty() && !triWallId.empty()) {
            for (size_t t = 0; t < numTris; ++t) {
                const int w = triWallId[t];
                if (w >= 0 && static_cast<size_t>(w) < wallMaterial.size()) {
                    triMaterialId[t] = wallMaterial[w];
                }
            }
        }

        out << "<?xml version=\"1.0\"?>\n"
            << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
            << "  <UnstructuredGrid>\n";

        out << "    <FieldData>\n"
            << "      <DataArray type=\"Float32\" Name=\"TIME\"  NumberOfTuples=\"1\" format=\"ascii\"> "
            << time << " </DataArray>\n"
            << "      <DataArray type=\"Int32\"   Name=\"STEP\"  NumberOfTuples=\"1\" format=\"ascii\"> "
            << iStep << " </DataArray>\n"
            << "    </FieldData>\n";

        out << "    <Piece NumberOfPoints=\"" << numPoints
            << "\" NumberOfCells=\"" << numTris << "\">\n";

        out << "      <Points>\n"
            << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (size_t i = 0; i < numPoints; ++i) {
            out << ' ' << verts[i].x
                << ' ' << verts[i].y
                << ' ' << verts[i].z;
        }
        out << "\n        </DataArray>\n"
            << "      </Points>\n";

        out << "      <Cells>\n";

        out << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
        for (size_t t = 0; t < numTris; ++t) {
            out << ' ' << tri_i0[t]
                << ' ' << tri_i1[t]
                << ' ' << tri_i2[t];
        }
        out << "\n        </DataArray>\n";

        out << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
        for (size_t t = 1; t <= numTris; ++t) {
            out << ' ' << (3 * t);
        }
        out << "\n        </DataArray>\n";

        out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
        for (size_t t = 0; t < numTris; ++t) {
            out << " 5";
        }
        out << "\n        </DataArray>\n"
            << "      </Cells>\n";

        out << "      <CellData Scalars=\"material ID\">\n";

        if (!triWallId.empty()) {
            out << "        <DataArray type=\"Int32\" Name=\"wall ID\" format=\"ascii\">\n";
            for (size_t t = 0; t < numTris; ++t) {
                out << ' ' << triWallId[t];
            }
            out << "\n        </DataArray>\n";
        }

        out << "        <DataArray type=\"Int32\" Name=\"material ID\" format=\"ascii\">\n";
        for (size_t t = 0; t < numTris; ++t) {
            out << ' ' << triMaterialId[t];
        }
        out << "\n        </DataArray>\n";

        out << "      </CellData>\n";

        out << "    </Piece>\n"
            << "  </UnstructuredGrid>\n"
            << "</VTKFile>\n";
    }

    void outputBallInteractionVTU(const std::string &dir, const size_t iFrame, const size_t iStep, const double time)
    {
        MKDIR(dir.c_str());
        std::ostringstream fname;
        fname << dir << "/ballInteraction_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";
        std::ofstream out(fname.str().c_str());
        if (!out) throw std::runtime_error("Cannot open " + fname.str());
        out << std::fixed << std::setprecision(10);

        const size_t N = ballInteraction_.numActivated_;
        std::vector<double3> p = ballInteraction_.contact_.pointHostCopy();
        std::vector<double> o = ballInteraction_.contact_.overlapHostCopy();
        std::vector<double3> n = ballInteraction_.contact_.normalHostCopy();
        std::vector<double3> f_c = ballInteraction_.contact_.forceHostCopy();
        std::vector<double3> t_c = ballInteraction_.contact_.torqueHostCopy();
        std::vector<double3> s_s = ballInteraction_.spring_.slidingHostCopy();
        std::vector<double3> s_r = ballInteraction_.spring_.rollingHostCopy();
        std::vector<double3> s_t = ballInteraction_.spring_.torsionHostCopy();
        
        out << "<?xml version=\"1.0\"?>\n"
            "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
            "  <UnstructuredGrid>\n";

        out << "    <FieldData>\n"
            "      <DataArray type=\"Float32\" Name=\"TIME\"  NumberOfTuples=\"1\" format=\"ascii\"> "
            << time << " </DataArray>\n"
            "      <DataArray type=\"Int32\"   Name=\"STEP\"  NumberOfTuples=\"1\" format=\"ascii\"> "
            << iStep << " </DataArray>\n"
            "    </FieldData>\n";

        out << "    <Piece NumberOfPoints=\"" << N
            << "\" NumberOfCells=\"" << N << "\">\n";

        out << "      <Points>\n"
            "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) {
            out << ' ' << p[i].x << ' ' << p[i].y << ' ' << p[i].z;
        }
        out << "\n        </DataArray>\n"
            "      </Points>\n";

        out << "      <Cells>\n"
            "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << i;
        out << "\n        </DataArray>\n"
            "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
        for (int i = 1; i <= N; ++i) out << ' ' << i;
        out << "\n        </DataArray>\n"
            "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << " 1";          // 1 = VTK_VERTEX
        out << "\n        </DataArray>\n"
            "      </Cells>\n";

        out << "      <PointData Scalars=\"overlap\">\n";

        out << "        <DataArray type=\"Float32\" Name=\"overlap\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << o[i];
        out << "\n        </DataArray>\n";

        const struct {
            const char* name;
            const std::vector<double3>& vec;
        } vec3s[] = {
            { "contact normal" , n     },
            { "contact force"  , f_c   },
            { "contact torque" , t_c   },
            { "sliding spring" , s_s   },
            { "rolling spring" , s_r   },
            { "torsion spring" , s_t   }
        };

        for (size_t k = 0; k < sizeof(vec3s) / sizeof(vec3s[0]); ++k) {
            out << "        <DataArray type=\"Float32\" Name=\"" << vec3s[k].name
                << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
            const std::vector<double3>& v = vec3s[k].vec;
            for (size_t i = 0; i < N; ++i)
                out << ' ' << v[i].x << ' ' << v[i].y << ' ' << v[i].z;
            out << "\n        </DataArray>\n";
        }

        out << "      </PointData>\n"
            "    </Piece>\n"
            "  </UnstructuredGrid>\n"
            "</VTKFile>\n";
    }

    void outputBondedInteractionVTU(const std::string &dir, const size_t iFrame, const size_t iStep, const double time)
    {
        MKDIR(dir.c_str());
        std::ostringstream fname;
        fname << dir << "/bondedInteraction_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";
        std::ofstream out(fname.str().c_str());
        if (!out) throw std::runtime_error("Cannot open " + fname.str());
        out << std::fixed << std::setprecision(10);

        const size_t N = bondedInteraction_.pair_.deviceSize();
        std::vector<double3> p = bondedInteraction_.bond_.pointHostCopy();
        std::vector<double3> n = bondedInteraction_.bond_.normalHostCopy();
        std::vector<int> isBonded = bondedInteraction_.bond_.isBondedHostCopy();
        std::vector<double> f_n = bondedInteraction_.bond_.normalForceHostCopy();
        std::vector<double> t_n = bondedInteraction_.bond_.torsionTorqueHostCopy();
        std::vector<double3> f_s = bondedInteraction_.bond_.shearForceHostCopy();
        std::vector<double3> t_s = bondedInteraction_.bond_.bendingTorqueHostCopy();
        std::vector<double> sigma_max = bondedInteraction_.bond_.maxNormalStressHostCopy();
        std::vector<double> tau_max = bondedInteraction_.bond_.maxShearStressHostCopy();
        
        out << "<?xml version=\"1.0\"?>\n"
            "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
            "  <UnstructuredGrid>\n";

        out << "    <FieldData>\n"
            "      <DataArray type=\"Float32\" Name=\"TIME\"  NumberOfTuples=\"1\" format=\"ascii\"> "
            << time << " </DataArray>\n"
            "      <DataArray type=\"Int32\"   Name=\"STEP\"  NumberOfTuples=\"1\" format=\"ascii\"> "
            << iStep << " </DataArray>\n"
            "    </FieldData>\n";

        out << "    <Piece NumberOfPoints=\"" << N
            << "\" NumberOfCells=\"" << N << "\">\n";

        out << "      <Points>\n"
            "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) {
            out << ' ' << p[i].x << ' ' << p[i].y << ' ' << p[i].z;
        }
        out << "\n        </DataArray>\n"
            "      </Points>\n";

        out << "      <Cells>\n"
            "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << i;
        out << "\n        </DataArray>\n"
            "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
        for (int i = 1; i <= N; ++i) out << ' ' << i;
        out << "\n        </DataArray>\n"
            "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << " 1";          // 1 = VTK_VERTEX
        out << "\n        </DataArray>\n"
            "      </Cells>\n";

        out << "      <PointData Scalars=\"is bonded\">\n";

        out << "        <DataArray type=\"Int32\" Name=\"is bonded\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << isBonded[i];
        out << "\n        </DataArray>\n";

        out << "        <DataArray type=\"Float32\" Name=\"normal force\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << f_n[i];
        out << "\n        </DataArray>\n";

        out << "        <DataArray type=\"Float32\" Name=\"torsion torque\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << t_n[i];
        out << "\n        </DataArray>\n";

        out << "        <DataArray type=\"Float32\" Name=\"maximum normal stress\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << sigma_max[i];
        out << "\n        </DataArray>\n";

        out << "        <DataArray type=\"Float32\" Name=\"maximum shear stress\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << tau_max[i];
        out << "\n        </DataArray>\n";

        const struct {
            const char* name;
            const std::vector<double3>& vec;
        } vec3s[] = {
            { "contact normal" , n     },
            { "shearing force" , f_s   },
            { "bending torque" , t_s   }
        };

        for (size_t k = 0; k < sizeof(vec3s) / sizeof(vec3s[0]); ++k) {
            out << "        <DataArray type=\"Float32\" Name=\"" << vec3s[k].name
                << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
            const std::vector<double3>& v = vec3s[k].vec;
            for (size_t i = 0; i < N; ++i)
                out << ' ' << v[i].x << ' ' << v[i].y << ' ' << v[i].z;
            out << "\n        </DataArray>\n";
        }

        out << "      </PointData>\n"
            "    </Piece>\n"
            "  </UnstructuredGrid>\n"
            "</VTKFile>\n";
    }

    void outputBallTriangleInteractionVTU(const std::string &dir, const size_t iFrame, const size_t iStep, const double time)
    {
        MKDIR(dir.c_str());
        std::ostringstream fname;
        fname << dir << "/ballTriangleInteraction_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";
        std::ofstream out(fname.str().c_str());
        if (!out) throw std::runtime_error("Cannot open " + fname.str());
        out << std::fixed << std::setprecision(10);

        const size_t N = ballTriangleInteraction_.numActivated_;
        std::vector<double3> p = ballTriangleInteraction_.contact_.pointHostCopy();
        std::vector<double> o = ballTriangleInteraction_.contact_.overlapHostCopy();
        std::vector<double3> n = ballTriangleInteraction_.contact_.normalHostCopy();
        std::vector<double3> f_c = ballTriangleInteraction_.contact_.forceHostCopy();
        std::vector<double3> t_c = ballTriangleInteraction_.contact_.torqueHostCopy();
        std::vector<double3> s_s = ballTriangleInteraction_.spring_.slidingHostCopy();
        std::vector<double3> s_r = ballTriangleInteraction_.spring_.rollingHostCopy();
        std::vector<double3> s_t = ballTriangleInteraction_.spring_.torsionHostCopy();
        
        out << "<?xml version=\"1.0\"?>\n"
            "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
            "  <UnstructuredGrid>\n";

        out << "    <FieldData>\n"
            "      <DataArray type=\"Float32\" Name=\"TIME\"  NumberOfTuples=\"1\" format=\"ascii\"> "
            << time << " </DataArray>\n"
            "      <DataArray type=\"Int32\"   Name=\"STEP\"  NumberOfTuples=\"1\" format=\"ascii\"> "
            << iStep << " </DataArray>\n"
            "    </FieldData>\n";

        out << "    <Piece NumberOfPoints=\"" << N
            << "\" NumberOfCells=\"" << N << "\">\n";

        out << "      <Points>\n"
            "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) {
            out << ' ' << p[i].x << ' ' << p[i].y << ' ' << p[i].z;
        }
        out << "\n        </DataArray>\n"
            "      </Points>\n";

        out << "      <Cells>\n"
            "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << i;
        out << "\n        </DataArray>\n"
            "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
        for (int i = 1; i <= N; ++i) out << ' ' << i;
        out << "\n        </DataArray>\n"
            "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << " 1";          // 1 = VTK_VERTEX
        out << "\n        </DataArray>\n"
            "      </Cells>\n";

        out << "      <PointData Scalars=\"overlap\">\n";

        out << "        <DataArray type=\"Float32\" Name=\"overlap\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << o[i];
        out << "\n        </DataArray>\n";

        const struct {
            const char* name;
            const std::vector<double3>& vec;
        } vec3s[] = {
            { "contact normal" , n     },
            { "contact force"  , f_c   },
            { "contact torque" , t_c   },
            { "sliding spring" , s_s   },
            { "rolling spring" , s_r   },
            { "torsion spring" , s_t   }
        };

        for (size_t k = 0; k < sizeof(vec3s) / sizeof(vec3s[0]); ++k) {
            out << "        <DataArray type=\"Float32\" Name=\"" << vec3s[k].name
                << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
            const std::vector<double3>& v = vec3s[k].vec;
            for (size_t i = 0; i < N; ++i)
                out << ' ' << v[i].x << ' ' << v[i].y << ' ' << v[i].z;
            out << "\n        </DataArray>\n";
        }

        out << "      </PointData>\n"
            "    </Piece>\n"
            "  </UnstructuredGrid>\n"
            "</VTKFile>\n";
    }

private:
    void loop_ball(double& time, size_t& iStep, size_t& iFrame, 
    const double3 gravity, const double timeStep, const size_t numStep, const size_t frameInterval, const std::string dir)
    {
        outputBallVTU(dir, iFrame, iStep, time);
        outputBondedInteractionVTU(dir, iFrame, iStep, time);
        outputBallInteractionVTU(dir, iFrame, iStep, time);

        const double halfTimeStep = 0.5 * timeStep;
        while (iStep <= numStep)
        {
            iStep++;
            time += timeStep;
            if(iStep % 5 == 0) ballNeighborSearch();

            calculateBallContactForceTorque(halfTimeStep);
            addExternalForceTorque(time - halfTimeStep);

            ballClumpIntegration1stHalf(gravity, halfTimeStep);

            calculateBallContactForceTorque(halfTimeStep);
            addExternalForceTorque(time);

            ballClumpIntegration2ndHalf(gravity, halfTimeStep);

            if (iStep % frameInterval == 0)
            {
                iFrame++;
                std::cout << "Frame " << iFrame << " at Time " << time << std::endl;
                outputBallVTU(dir, iFrame, iStep, time);
                outputBondedInteractionVTU(dir, iFrame, iStep, time);
                outputBallInteractionVTU(dir, iFrame, iStep, time);
            }
        }
    }

    void loop_ball_periodicXYZ(double& time, size_t& iStep, size_t& iFrame, 
    const double3 gravity, const double timeStep, const size_t numStep, const size_t frameInterval, const std::string dir)
    {
        outputBallVTU(dir, iFrame, iStep, time);
        outputBondedInteractionVTU(dir, iFrame, iStep, time);
        outputBallInteractionVTU(dir, iFrame, iStep, time);

        const double halfTimeStep = 0.5 * timeStep;
        while (iStep <= numStep)
        {
            iStep++;
            time += timeStep;
            if(iStep % 5 == 0) ballNeighborSearch();

            calculateBallContactForceTorque(halfTimeStep);
            addDummyContactForceTorque(periodicX_, make_int3(1, 0, 0), halfTimeStep);
            addDummyContactForceTorque(periodicY_, make_int3(0, 1, 0), halfTimeStep);
            addDummyContactForceTorque(periodicZ_, make_int3(0, 0, 1), halfTimeStep);
            addDummyContactForceTorque(periodicXY_, make_int3(1, 1, 0), halfTimeStep);
            addDummyContactForceTorque(periodicYZ_, make_int3(0, 1, 1), halfTimeStep);
            addDummyContactForceTorque(periodicXZ_, make_int3(1, 0, 1), halfTimeStep);
            addDummyContactForceTorque(periodicXYZ_, make_int3(1, 1, 1), halfTimeStep);
            addExternalForceTorque(time - halfTimeStep);

            ballClumpIntegration1stHalf(gravity, halfTimeStep);

            calculateBallContactForceTorque(halfTimeStep);
            addDummyContactForceTorque(periodicX_, make_int3(1, 0, 0), halfTimeStep);
            addDummyContactForceTorque(periodicY_, make_int3(0, 1, 0), halfTimeStep);
            addDummyContactForceTorque(periodicZ_, make_int3(0, 0, 1), halfTimeStep);
            addDummyContactForceTorque(periodicXY_, make_int3(1, 1, 0), halfTimeStep);
            addDummyContactForceTorque(periodicYZ_, make_int3(0, 1, 1), halfTimeStep);
            addDummyContactForceTorque(periodicXZ_, make_int3(1, 0, 1), halfTimeStep);
            addDummyContactForceTorque(periodicXYZ_, make_int3(1, 1, 1), halfTimeStep);
            addExternalForceTorque(time);

            ballClumpIntegration2ndHalf(gravity, halfTimeStep);

            if (iStep % frameInterval == 0)
            {
                iFrame++;
                std::cout << "Frame " << iFrame << " at Time " << time << std::endl;
                outputBallVTU(dir, iFrame, iStep, time);
                outputBondedInteractionVTU(dir, iFrame, iStep, time);
                outputBallInteractionVTU(dir, iFrame, iStep, time);
            }
        }
    }

    void loop_ball_wall(double& time, size_t& iStep, size_t& iFrame, 
    const double3 gravity, const double timeStep, const size_t numStep, const size_t frameInterval, const std::string dir)
    {
        outputBallVTU(dir, iFrame, iStep, time);
        outputMeshWallVTU(dir, iFrame, iStep, time);
        outputBondedInteractionVTU(dir, iFrame, iStep, time);
        outputBallInteractionVTU(dir, iFrame, iStep, time);
        outputBallTriangleInteractionVTU(dir, iFrame, iStep, time);

        const double halfTimeStep = 0.5 * timeStep;
        while (iStep <= numStep)
        {
            iStep++;
            time += timeStep;
            if(iStep % 5 == 0) ballNeighborSearch();

            calculateBallContactForceTorque(halfTimeStep);
            addTriangleContactForceTorque(timeStep);
            addExternalForceTorque(time - halfTimeStep);

            ballClumpIntegration1stHalf(gravity, halfTimeStep);

            calculateBallContactForceTorque(halfTimeStep);
            addTriangleContactForceTorque(timeStep);
            addExternalForceTorque(time);

            ballClumpIntegration2ndHalf(gravity, halfTimeStep);

            if (iStep % frameInterval == 0)
            {
                iFrame++;
                std::cout << "Frame " << iFrame << " at Time " << time << std::endl;
                outputBallVTU(dir, iFrame, iStep, time);
                outputMeshWallVTU(dir, iFrame, iStep, time);
                outputBondedInteractionVTU(dir, iFrame, iStep, time);
                outputBallInteractionVTU(dir, iFrame, iStep, time);
                outputBallTriangleInteractionVTU(dir, iFrame, iStep, time);
            }
        }
    }

    void loop_ball_wall_periodicXYZ(double& time, size_t& iStep, size_t& iFrame, 
    const double3 gravity, const double timeStep, const size_t numStep, const size_t frameInterval, const std::string dir)
    {
        outputBallVTU(dir, iFrame, iStep, time);
        outputMeshWallVTU(dir, iFrame, iStep, time);
        outputBondedInteractionVTU(dir, iFrame, iStep, time);
        outputBallInteractionVTU(dir, iFrame, iStep, time);
        outputBallTriangleInteractionVTU(dir, iFrame, iStep, time);

        const double halfTimeStep = 0.5 * timeStep;
        while (iStep <= numStep)
        {
            iStep++;
            time += timeStep;
            if(iStep % 5 == 0) ballNeighborSearch();

            calculateBallContactForceTorque(halfTimeStep);
            addTriangleContactForceTorque(timeStep);
            addDummyContactForceTorque(periodicX_, make_int3(1, 0, 0), halfTimeStep);
            addDummyContactForceTorque(periodicY_, make_int3(0, 1, 0), halfTimeStep);
            addDummyContactForceTorque(periodicZ_, make_int3(0, 0, 1), halfTimeStep);
            addDummyContactForceTorque(periodicXY_, make_int3(1, 1, 0), halfTimeStep);
            addDummyContactForceTorque(periodicYZ_, make_int3(0, 1, 1), halfTimeStep);
            addDummyContactForceTorque(periodicXZ_, make_int3(1, 0, 1), halfTimeStep);
            addDummyContactForceTorque(periodicXYZ_, make_int3(1, 1, 1), halfTimeStep);
            addExternalForceTorque(time - halfTimeStep);

            ballClumpIntegration1stHalf(gravity, halfTimeStep);

            calculateBallContactForceTorque(halfTimeStep);
            addTriangleContactForceTorque(timeStep);
            addDummyContactForceTorque(periodicX_, make_int3(1, 0, 0), halfTimeStep);
            addDummyContactForceTorque(periodicY_, make_int3(0, 1, 0), halfTimeStep);
            addDummyContactForceTorque(periodicZ_, make_int3(0, 0, 1), halfTimeStep);
            addDummyContactForceTorque(periodicXY_, make_int3(1, 1, 0), halfTimeStep);
            addDummyContactForceTorque(periodicYZ_, make_int3(0, 1, 1), halfTimeStep);
            addDummyContactForceTorque(periodicXZ_, make_int3(1, 0, 1), halfTimeStep);
            addDummyContactForceTorque(periodicXYZ_, make_int3(1, 1, 1), halfTimeStep);
            addExternalForceTorque(time);

            ballClumpIntegration2ndHalf(gravity, halfTimeStep);

            if (iStep % frameInterval == 0)
            {
                iFrame++;
                std::cout << "Frame " << iFrame << " at Time " << time << std::endl;
                outputBallVTU(dir, iFrame, iStep, time);
                outputMeshWallVTU(dir, iFrame, iStep, time);
                outputBondedInteractionVTU(dir, iFrame, iStep, time);
                outputBallInteractionVTU(dir, iFrame, iStep, time);
                outputBallTriangleInteractionVTU(dir, iFrame, iStep, time);
            }
        }
    }

protected:
    void addBondedInteraction(const std::vector<int>& objectPointed, const std::vector<int>& objectPointing)
    {
        bondedInteraction_.bond_.copyDeviceToHost(stream_);
        bondedInteraction_.add(objectPointed, objectPointing, ball_.positionHostCopy());
    }

    std::vector<double3> getBallPosition()
    {
        return ball_.positionHostCopy();
    }

    std::vector<double3> getBallVelocity()
    {
        return ball_.velocityHostCopy();
    }

    std::vector<double3> getBallAngularVelocity()
    {
        return ball_.angularVelocityHostCopy();
    }

    std::vector<int> getBallPairPointed() 
    { 
        std::vector<int> p = ballInteraction_.pair_.objectPointedHostCopy();
        std::vector<int> p1(p.begin(), p.begin() + ballInteraction_.numActivated_);
        return p1;
    }

    std::vector<int> getBallPairPointing() 
    { 
        std::vector<int> p = ballInteraction_.pair_.objectPointingHostCopy();
        std::vector<int> p1(p.begin(), p.begin() + ballInteraction_.numActivated_);
        return p1;
    }

    std::vector<double> getBallPairOverlap() 
    { 
        std::vector<double> o = ballInteraction_.contact_.overlapHostCopy();
        std::vector<double> o1(o.begin(), o.begin() + ballInteraction_.numActivated_);
        return o1;
    }

    std::vector<double3> getBallPairSlidingSpring() 
    { 
        std::vector<double3> s = ballInteraction_.spring_.slidingHostCopy();
        std::vector<double3> s1(s.begin(), s.begin() + ballInteraction_.numActivated_);
        return s1;
    }

    std::vector<double3> getBallPairRollingSpring() 
    { 
        std::vector<double3> s = ballInteraction_.spring_.rollingHostCopy();
        std::vector<double3> s1(s.begin(), s.begin() + ballInteraction_.numActivated_);
        return s1;
    }

    std::vector<double3> getBallPairTorsionSpring() 
    { 
        std::vector<double3> s = ballInteraction_.spring_.torsionHostCopy();
        std::vector<double3> s1(s.begin(), s.begin() + ballInteraction_.numActivated_);
        return s1;
    }

    std::vector<int> getBallTrianglePairPointed() 
    { 
        std::vector<int> p = ballTriangleInteraction_.pair_.objectPointedHostCopy();
        std::vector<int> p1(p.begin(), p.begin() + ballTriangleInteraction_.numActivated_);
        return p1;
    }

    std::vector<int> getBallTrianglePairPointing() 
    { 
        std::vector<int> p = ballTriangleInteraction_.pair_.objectPointingHostCopy();
        std::vector<int> p1(p.begin(), p.begin() + ballTriangleInteraction_.numActivated_);
        return p1;
    }

    std::vector<double> getBallTrianglePairOverlap() 
    { 
        std::vector<double> o = ballInteraction_.contact_.overlapHostCopy();
        std::vector<double> o1(o.begin(), o.begin() + ballInteraction_.numActivated_);
        return o1;
    }

    std::vector<double3> getBallTrianglePairSlidingSpring() 
    { 
        std::vector<double3> s = ballInteraction_.spring_.slidingHostCopy();
        std::vector<double3> s1(s.begin(), s.begin() + ballInteraction_.numActivated_);
        return s1;
    }

    std::vector<double3> getBallTrianglePairRollingSpring() 
    { 
        std::vector<double3> s = ballInteraction_.spring_.rollingHostCopy();
        std::vector<double3> s1(s.begin(), s.begin() + ballInteraction_.numActivated_);
        return s1;
    }

    std::vector<double3> getBallTrianglePairTorsionSpring() 
    { 
        std::vector<double3> s = ballInteraction_.spring_.torsionHostCopy();
        std::vector<double3> s1(s.begin(), s.begin() + ballInteraction_.numActivated_);
        return s1;
    }

    const double3* getBallPositionDevicePtr()
    {
        return ball_.position();
    }

    const double3* getBallVelocityDevicePtr()
    {
        return ball_.velocity();
    }

    const double3* getBallAngularVelocityDevicePtr()
    {
        return ball_.angularVelocity();
    }

    const double* getBallRadiusDevicePtr()
    {
        return ball_.radius();
    }

    const double* getBallInverseMassDevicePtr()
    {
        return ball_.inverseMass();
    }

    const size_t getBallNumber()
    {
        return ball_.deviceSize();
    }

    clump& getClump()
    {
        return clump_;
    }

    double3* getBallForceDevicePtr()
    {
        return ball_.force();
    }

    double3* getBallTorqueDevicePtr()
    {
        return ball_.torque();
    }

    cudaStream_t getDEMCUDAStream()
    {
        return stream_;
    }

    virtual void addExternalForceTorque(const double time) 
    {
    }

    virtual bool addInitialCondition() 
    {
        return false;
    }

public:
    void setMaterial(const int index,
    const double YoungsModulus,
    const double poissonRatio,
    const double restitutionCoefficient)
    {
        if (index >= materialTable_.size())
        {
            for (size_t i = 0; i < index + 1 - materialTable_.size(); i++)
            {
                materialRow row;
                row.YoungsModulus = 0.;
                row.poissonRatio = 0.;
                row.restitutionCoefficient = 1.;
                materialTable_.push_back(row);
            }
        }
        if (YoungsModulus > 0.) materialTable_[index].YoungsModulus = YoungsModulus;
        if (poissonRatio < 2. && poissonRatio > -1.) materialTable_[index].poissonRatio = poissonRatio;
        if (restitutionCoefficient <= 1. && restitutionCoefficient > 0.) materialTable_[index].restitutionCoefficient = restitutionCoefficient;
    }

    void setFriction(const int materialIndexA,
    const int materialIndexB,
    const double slidingFrictionCoefficient,
    const double rollingFrictionCoefficient,
    const double torsionFrictionCoefficient)
    {
        if (materialIndexA < 0) return;
        if (materialIndexB < 0) return;
        frictionRow row;
        row.materialIndexA = materialIndexA;
        row.materialIndexB = materialIndexB;
        row.slidingFrictionCoefficient = 0.;
        row.rollingFrictionCoefficient = 0.;
        row.torsionFrictionCoefficient = 0.;
        if (slidingFrictionCoefficient > 0.) row.slidingFrictionCoefficient = slidingFrictionCoefficient;
        if (rollingFrictionCoefficient > 0.) row.rollingFrictionCoefficient = rollingFrictionCoefficient;
        if (torsionFrictionCoefficient > 0.) row.torsionFrictionCoefficient = torsionFrictionCoefficient;
        frictionTable_.push_back(row);
    }

    void setLinearStiffness(const int materialIndexA,
    const int materialIndexB,
    const double normalStiffness,
    const double slidingStiffness,
    const double rollingStiffness,
    const double torsionStiffness)
    {
        if (materialIndexA < 0) return;
        if (materialIndexB < 0) return;
        linearStiffnessRow row;
        row.materialIndexA = materialIndexA;
        row.materialIndexB = materialIndexB;
        row.normalStiffness = 0.;
        row.slidingStiffness = 0.;
        row.rollingStiffness = 0.;
        row.torsionStiffness = 0.;
        if (normalStiffness > 0.) row.normalStiffness = normalStiffness;
        if (slidingStiffness > 0.) row.slidingStiffness = slidingStiffness;
        if (rollingStiffness > 0.) row.rollingStiffness = rollingStiffness;
        if (torsionStiffness > 0.) row.torsionStiffness = torsionStiffness;
        linearStiffnessTable_.push_back(row);
    }

    void setBond(const int materialIndexA,
    const int materialIndexB,
    const double bondRadiusMultiplier,
    const double bondYoungsModulus,
    const double normalToShearStiffnessRatio,
    const double tensileStrength,
    const double cohesion,
    const double frictionCoefficient)
    {
        if (materialIndexA < 0) return;
        if (materialIndexB < 0) return;
        bondRow row;
        row.materialIndexA = materialIndexA;
        row.materialIndexB = materialIndexB;
        row.bondRadiusMultiplier = 1.;
        row.bondYoungsModulus = 0.;
        row.normalToShearStiffnessRatio = 1.;
        row.tensileStrength = 0.;
        row.cohesion = 0.;
        row.frictionCoefficient = 0.;
        if (bondRadiusMultiplier > 0.) row.bondRadiusMultiplier = bondRadiusMultiplier;
        if (bondYoungsModulus > 0.) row.bondYoungsModulus = bondYoungsModulus;
        if (normalToShearStiffnessRatio > 0.) row.normalToShearStiffnessRatio = normalToShearStiffnessRatio;
        if (tensileStrength > 0.) row.tensileStrength = tensileStrength;
        if (cohesion > 0.) row.cohesion = cohesion;
        if (frictionCoefficient > 0.) row.frictionCoefficient = frictionCoefficient;
        bondTable_.push_back(row);
    }

    void addBall(const double3 position, 
    const double3 velocity, 
    const double3 angularVelocity, 
    const double radius, 
    const double density, 
    const int materialID, 
    const double3 localPosition = make_double3(0., 0., 0.),
    const int clumpID = -1)
    {
        double mass = 4. / 3. * radius * radius * radius * pi() * density;
        double inverseMass = 0.;
        if (mass > 0.0) inverseMass = 1.0 / mass;
        ball_.addHost(position, 
        localPosition, 
        velocity, 
        angularVelocity, 
        make_double3(0., 0., 0.), 
        make_double3(0., 0., 0.), 
        radius, 
        inverseMass, 
        materialID, 
        clumpID, 
        -1, 
        -1);
    }

    void addFixedBall(const double3 position, 
    const double radius, 
    const int materialID, 
    const double3 localPosition = make_double3(0., 0., 0.),
    const int clumpID = -1)
    {
        ball_.addHost(position, 
        localPosition, 
        make_double3(0., 0., 0.), 
        make_double3(0., 0., 0.), 
        make_double3(0., 0., 0.), 
        make_double3(0., 0., 0.), 
        radius, 
        0., 
        materialID, 
        clumpID, 
        -1, 
        -1);
    }

    void eraseBall(const size_t index)
    {
        ball_.eraseHost(index);
        std::vector<int> pebbleStart = clump_.pebbleStartHostRef();
        std::vector<int> pebbleEnd = clump_.pebbleEndHostRef();
        for (size_t i = 0; i < clump_.hostSize(); i++)
        {
            if (pebbleEnd[i] > index)
            {
                for (size_t j = i; j < clump_.hostSize(); j++)
                {
                    if (pebbleStart[j] > index) pebbleStart[j] -= 1;
                    pebbleEnd[j] -= 1;
                }
                break;
            }
        }
        clump_.setPebbleStartHost(pebbleStart);
        clump_.setPebbleEndHost(pebbleEnd);
    }

    void copyBall(const ball& other)
    {
        ball_.copyFromHost(other);
    }

    void addClump(std::vector<double3> globalPoint, 
    std::vector<double> radius, 
    double3 centroidPosition, 
    double3 velocity, 
    double3 angularVelocity, 
    double mass, 
    symMatrix inertiaTensor, 
    int materialID)
    {
        int clumpID = static_cast<int>(clump_.hostSize());
        size_t pebbleStart = ball_.hostSize();
        size_t pebbleEnd = pebbleStart + globalPoint.size();

        std::vector<double3> vel(globalPoint.size(), velocity);
        double volume = 0., invMass = 0.;
        if (mass > 1.e-20)
        {
            invMass = 1.0 / mass;
            for (size_t i = 0; i < globalPoint.size(); i++)
            {
                volume += 4.0 / 3.0 * pi() * pow(radius[i], 3.0);
                vel[i] += cross(angularVelocity, globalPoint[i] - centroidPosition);
            }
        }
        double density_ave = 0;
        if (volume > 0.) density_ave = mass / volume;

        for (size_t i = 0; i < globalPoint.size(); i++)
        {
            addBall(globalPoint[i], 
            vel[i], 
            angularVelocity, 
            radius[i], 
            density_ave, 
            materialID, 
            globalPoint[i] - centroidPosition,
            clumpID);
        }

        clump_.addHost(centroidPosition, 
        velocity, 
        angularVelocity, 
        make_double3(0.0, 0.0, 0.0), 
        make_double3(0.0, 0.0, 0.0), 
        make_quaternion(1.0,0.0,0.0,0.0), 
        inverse(inertiaTensor), 
        invMass, 
        materialID,
        static_cast<int>(pebbleStart), 
        static_cast<int>(pebbleEnd));
    }

    void addFixedClump(std::vector<double3> globalPoint, 
    std::vector<double> radius, 
    double3 centroidPosition, 
    int materialID)
    {
        int clumpID = static_cast<int>(clump_.hostSize());
        size_t pebbleStart = ball_.hostSize();
        size_t pebbleEnd = pebbleStart + globalPoint.size();

        for (size_t i = 0; i < globalPoint.size(); i++)
        {
            addFixedBall(globalPoint[i], 
            radius[i], 
            materialID, 
            globalPoint[i] - centroidPosition,
            clumpID);
        }

        clump_.addHost(centroidPosition, 
        make_double3(0., 0., 0.), 
        make_double3(0., 0., 0.), 
        make_double3(0., 0., 0.), 
        make_double3(0., 0., 0.), 
        make_quaternion(1., 0., 0., 0.), 
        make_symMatrix(0., 0., 0., 0., 0., 0.), 
        0., 
        materialID,
        static_cast<int>(pebbleStart), 
        static_cast<int>(pebbleEnd));
    }

    void copyClump(const clump& other)
    {
        clump_.copyFromHost(other);
    }

    void addMeshWall(const std::vector<double3>& vertexLocalPosition, 
    const std::vector<int>& triIndex0, 
    const std::vector<int>& triIndex1, 
    const std::vector<int>& triIndex2, 
    const double3 position, 
    const double3 velocity, 
    const double3 angularVelocity, 
    int matirialID)
    {
        if (triIndex0.size() != triIndex1.size() || triIndex0.size() != triIndex2.size()) return;
        int maxIndex0 = *std::max_element(triIndex0.begin(), triIndex0.end());
        int maxIndex1 = *std::max_element(triIndex1.begin(), triIndex1.end());
        int maxIndex2 = *std::max_element(triIndex2.begin(), triIndex2.end());
        int maxIndex = std::max(maxIndex0, maxIndex1);
        maxIndex = std::max(maxIndex, maxIndex2);
        if (maxIndex >= vertexLocalPosition.size()) return;

        int vertexStart = meshWall_.vertex_.hostSize();
        int wallIndex = meshWall_.body_.hostSize();
        meshWall_.body_.addHost(position, 
        make_double3(0.0, 0.0, 0.0), 
        make_double3(0.0, 0.0, 0.0), 
        make_double3(0.0, 0.0, 0.0), 
        make_double3(0.0, 0.0, 0.0), 
        make_quaternion(1.0, 0.0, 0.0, 0.0), 
        make_symMatrix(0., 0., 0., 0., 0., 0.), 
        0.0, 
        matirialID, 
        vertexStart, 
        vertexStart + static_cast<int>(vertexLocalPosition.size()));

        for (size_t i = 0; i < vertexLocalPosition.size(); i++)
        {
            meshWall_.vertex_.addHost(vertexLocalPosition[i], 
            vertexLocalPosition[i] + position, 
            wallIndex);
        }

        for (size_t i = 0; i < triIndex0.size(); i++)
        {
            double3 v0 = vertexLocalPosition[triIndex0[i]] + position;
            double3 v1 = vertexLocalPosition[triIndex1[i]] + position;
            double3 v2 = vertexLocalPosition[triIndex2[i]] + position;
            auto getTriangleCircumcenter = [&](const double3& a, const double3& b, const double3& c)->double3
            {
                double3 ab = b - a;
                double3 ac = c - a;
                double3 n  = cross(ab, ac);
                double n2  = dot(n, n);
                if (n2 < 1e-30)
                {
                    return (a + b + c) / 3.0;
                }
                double3 term1 = cross(n,  ab) * dot(ac, ac);
                double3 term2 = cross(ac, n ) * dot(ab, ab);
                double invDen = 1.0 / (2.0 * n2);
                return a + (term1 + term2) * invDen;
            };
            double3 c = getTriangleCircumcenter(v0, v1, v2);
            meshWall_.triangle_.addHost(triIndex0[i], 
            triIndex1[i], 
            triIndex2[i], 
            wallIndex, 
            c,
            -1, 
            -1);
        }
    }

    void setPeriodicBoundary(bool X, bool Y, bool Z)
    {
        if (X) periodicX_.activatedFlag_ = true;
        if (Y) 
        {
            periodicY_.activatedFlag_ = true;
            if (X && Y) periodicXY_.activatedFlag_ = true;
        }
        if (Z)
        {
            periodicZ_.activatedFlag_ = true;
            if (X && Z) periodicXZ_.activatedFlag_ = true;
            if (Y && Z) periodicYZ_.activatedFlag_ = true;
            if (X && Y && Z) periodicXYZ_.activatedFlag_ = true;
        }
    }

    void solve(const double3 minBoundary, 
    const double3 maxBoundary, 
    const double3 gravity, 
    const double timeStep, 
    const double maximumTime,
    const size_t numFrame,
    const std::string dir, 
    const size_t deviceID = 0, 
    const size_t maximumThread = 256)
    {
        removeVtuFiles(dir);
        removeDatFiles(dir);

        cudaError_t cudaStatus = cudaSetDevice(deviceID);
        if (cudaStatus != cudaSuccess) 
        {
            std::cout << "cudaSetDevice( " << deviceID
            << " ) failed! Do you have a CUDA-capable GPU installed?"
            << std::endl;
            exit(1);
        }

        if (maxBoundary.x < minBoundary.x || maxBoundary.y < minBoundary.y ||maxBoundary.z < minBoundary.z)
        {
            std::cout << "failed! Boundary setting is incorrect" << std::endl;
            return;           
        }
        
        if (timeStep <= 0.) 
        {
            std::cout << "failed! Time step is less than 0" << std::endl;
            return;
        }
        size_t numStep = size_t(maximumTime / timeStep) + 1;
        size_t frameInterval = numStep;
        if (numFrame > 0) frameInterval = numStep / numFrame;
        if (frameInterval < 1) frameInterval = 1;
        
        maxThread_ = maximumThread;
        DEMInitialize(minBoundary, maxBoundary);
        ballNeighborSearch();
        calculateBallContactForceTorque(0.);
        if (meshWall_.triangle_.deviceSize() > 0) addTriangleContactForceTorque(0.);
        if (addInitialCondition()) DEMInitialize(minBoundary, maxBoundary);

        if (ball_.deviceSize() == 0)
        {
            std::cout << "failed! The number of particles is equal to 0" << std::endl;
            return;
        }
        std::cout << "Initialization Completed." << std::endl;

        size_t iStep = 0, iFrame = 0;
        double time = 0.0;
        if (periodicX_.activatedFlag_ || periodicY_.activatedFlag_ || periodicZ_.activatedFlag_)
        {
            if (meshWall_.triangle_.deviceSize() > 0)
            {
                loop_ball_wall_periodicXYZ(time, iStep, iFrame, gravity, timeStep, numStep, frameInterval, dir);
            }
            else
            {
                loop_ball_periodicXYZ(time, iStep, iFrame, gravity, timeStep, numStep, frameInterval, dir);
            }
        }
        else
        {
            if (meshWall_.triangle_.deviceSize() > 0)
            {
                loop_ball_wall(time, iStep, iFrame, gravity, timeStep, numStep, frameInterval, dir);
            }
            else 
            {
                loop_ball(time, iStep, iFrame, gravity, timeStep, numStep, frameInterval, dir);
            }
        }

        ball_.copyDeviceToHost(stream_);
        clump_.copyDeviceToHost(stream_);
        meshWall_.body_.copyDeviceToHost(stream_);
        bondedInteraction_.bond_.copyDeviceToHost(stream_);
    }

private:
    cudaStream_t stream_;
    size_t maxThread_;

    std::vector<materialRow> materialTable_;
    std::vector<frictionRow> frictionTable_;
    std::vector<linearStiffnessRow> linearStiffnessTable_;
    std::vector<bondRow> bondTable_;
    contactParameters contactParameters_;

    ball ball_;
    clump clump_;
    meshWall meshWall_;

    spatialGrid ballSpatialGrid_;
    spatialGrid triangleSpatialGrid_;

    bondedInteraction bondedInteraction_;

    solidInteraction ballInteraction_;
    solidInteraction ballTriangleInteraction_;

    periodicBoundary periodicX_;
    periodicBoundary periodicY_;
    periodicBoundary periodicZ_;
    periodicBoundary periodicXY_;
    periodicBoundary periodicXZ_;
    periodicBoundary periodicYZ_;
    periodicBoundary periodicXYZ_;
};