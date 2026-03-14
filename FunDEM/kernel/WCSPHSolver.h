#include "particle.h"
#include "interaction.h"
#include "boundary.h"
#include "CUDAKernelFunction/buildHashStartEnd.cuh"
#include "CUDAKernelFunction/neighborSearchKernel.cuh"
#include "CUDAKernelFunction/SPHNeighborSearchKernel.cuh"
#include "CUDAKernelFunction/WCSPHIntegrationKernel.cuh"
#include "CUDAKernelFunction/myUtility/myFileEdit.h"
#include <algorithm>
#include <fstream>
#include <sstream>

struct SPHInteraction
{
    size_t numActivated_ {0};
    objectPointed objectPointed_;
    objectPointing objectPointing_;
    pair pair_;

    void resizePair(cudaStream_t stream)
    {
        numActivated_ = objectPointed_.numNeighborPairs(stream);
        if (numActivated_ > pair_.deviceSize())
        {
            pair_.allocateDevice(numActivated_, stream);
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

class WCSPHSolver
{
public:
    WCSPHSolver(cudaStream_t s)
    {
        maxThread_ = 256;
        stream_ = s;
    }

    ~WCSPHSolver() = default;

private:
    void upload()
    {
        WCSPH_.copyHostToDevice(stream_);
        dummy_.copyHostToDevice(stream_);

        size_t numSPH = WCSPH_.deviceSize();
        size_t numDummy = dummy_.deviceSize();
        SPHAndSPH_.objectPointed_.allocateDevice(numSPH, stream_);
        SPHAndSPH_.objectPointing_.allocateDevice(numSPH, stream_);
        //SPHAndSPH_.pair_.allocateDevice(80 * numSPH, stream_);

        SPHAndDummy_.objectPointed_.allocateDevice(numSPH, stream_);
        SPHAndDummy_.objectPointing_.allocateDevice(numDummy, stream_);
        //SPHAndDummy_.pair_.allocateDevice(80 * std::min(numSPH, numDummy), stream_);

        dummyAndDummy_.objectPointed_.allocateDevice(numDummy, stream_);
        dummyAndDummy_.objectPointing_.allocateDevice(numDummy, stream_);
        //dummyAndDummy_.pair_.allocateDevice(80 * numDummy, stream_);
    }

    void initializeSpatialGrid(const double3 minBoundary, const double3 maxBoundary)
    {
        double cellSizeOneDim = 0.0;
        const std::vector<double> h = WCSPH_.smoothLengthHostRef();
        if (h.size() > 0) cellSizeOneDim = *std::max_element(h.begin(), h.end()) * 2.0;
        const std::vector<double> h1 = dummy_.smoothLengthHostRef();
        if (h1.size() > 0) cellSizeOneDim = std::max(cellSizeOneDim, *std::max_element(h1.begin(), h1.end()) * 2.0);
        spatialGrid_.set(minBoundary, maxBoundary, cellSizeOneDim, stream_);
    }

protected:
    void initializeBoundaryCondition()
    {
        launchUpdateSpatialGridCellHashStartEnd(dummy_.position(), 
        dummy_.hashIndex(), 
        dummy_.hashValue(), 
        spatialGrid_.cellHashStart(), 
        spatialGrid_.cellHashEnd(), 
        spatialGrid_.minimumBoundary(), 
        spatialGrid_.maximumBoundary(), 
        spatialGrid_.cellSize(), 
        spatialGrid_.gridSize(), 
        spatialGrid_.numGrids(),
        dummy_.deviceSize(), 
        dummy_.gridDim(), 
        dummy_.blockDim(), 
        stream_);

#ifdef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif

        launchCountDummyInteractions(dummy_.position(), 
        dummy_.smoothLength(), 
        dummy_.clumpID(),
        dummy_.hashIndex(), 
        dummyAndDummy_.objectPointed_.neighborCount(), 
        dummyAndDummy_.objectPointed_.neighborPrefixSum(), 
        spatialGrid_.cellHashStart(), 
        spatialGrid_.cellHashEnd(), 
        spatialGrid_.minimumBoundary(),  
        spatialGrid_.cellSize(), 
        spatialGrid_.gridSize(),  
        dummy_.deviceSize(), 
        dummy_.gridDim(), 
        dummy_.blockDim(), 
        stream_);

#ifdef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif

        dummyAndDummy_.resizePair(stream_);

        launchWriteDummyInteractions(dummy_.position(), 
        dummy_.smoothLength(), 
        dummy_.clumpID(),
        dummy_.hashIndex(), 
        dummyAndDummy_.objectPointed_.neighborPrefixSum(),
        dummyAndDummy_.pair_.objectPointed(),
        dummyAndDummy_.pair_.objectPointing(),
        spatialGrid_.cellHashStart(), 
        spatialGrid_.cellHashEnd(), 
        spatialGrid_.minimumBoundary(),
        spatialGrid_.cellSize(), 
        spatialGrid_.gridSize(), 
        dummy_.deviceSize(), 
        dummy_.gridDim(), 
        dummy_.blockDim(), 
        stream_);

#ifdef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif

        launchCalDummyParticleNormal(dummy_.normal(), 
        dummy_.position(), 
        dummy_.density(), 
        dummy_.mass(), 
        dummy_.smoothLength(), 
        dummyAndDummy_.objectPointed_.neighborPrefixSum(), 
        dummyAndDummy_.pair_.objectPointing(), 
        dummy_.deviceSize(), 
        dummy_.gridDim(), 
        dummy_.blockDim(), 
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif
    }

    void SPHInitialize(const double3 minBoundary, const double3 maxBoundary)
    {
        WCSPH_.setBlockDim(WCSPH_.hostSize() < maxThread_ ? WCSPH_.hostSize() : maxThread_);
        dummy_.setBlockDim(dummy_.hostSize() < maxThread_ ? dummy_.hostSize() : maxThread_);

        upload();

        initializeSpatialGrid(minBoundary, maxBoundary);

        initializeBoundaryCondition();
    }

    void SPHNeighborSearch()
    {
        launchUpdateSpatialGridCellHashStartEnd(WCSPH_.position(), 
        WCSPH_.hashIndex(), 
        WCSPH_.hashValue(), 
        spatialGrid_.cellHashStart(), 
        spatialGrid_.cellHashEnd(), 
        spatialGrid_.minimumBoundary(), 
        spatialGrid_.maximumBoundary(), 
        spatialGrid_.cellSize(), 
        spatialGrid_.gridSize(), 
        spatialGrid_.numGrids(),
        WCSPH_.deviceSize(), 
        WCSPH_.gridDim(), 
        WCSPH_.blockDim(), 
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif

        launchCountSPHInteractions(WCSPH_.position(), 
        WCSPH_.smoothLength(), 
        WCSPH_.hashIndex(), 
        SPHAndSPH_.objectPointed_.neighborCount(), 
        SPHAndSPH_.objectPointed_.neighborPrefixSum(), 
        spatialGrid_.cellHashStart(), 
        spatialGrid_.cellHashEnd(), 
        spatialGrid_.minimumBoundary(), 
        spatialGrid_.cellSize(), 
        spatialGrid_.gridSize(), 
        WCSPH_.deviceSize(), 
        WCSPH_.gridDim(), 
        WCSPH_.blockDim(), 
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif

        SPHAndSPH_.resizePair(stream_);

        launchWriteSPHInteractions(WCSPH_.position(), 
        WCSPH_.smoothLength(), 
        WCSPH_.hashIndex(), 
        SPHAndSPH_.objectPointed_.neighborPrefixSum(),
        SPHAndSPH_.pair_.objectPointed(),
        SPHAndSPH_.pair_.objectPointing(),
        spatialGrid_.cellHashStart(), 
        spatialGrid_.cellHashEnd(), 
        spatialGrid_.minimumBoundary(),
        spatialGrid_.cellSize(), 
        spatialGrid_.gridSize(), 
        WCSPH_.deviceSize(), 
        WCSPH_.gridDim(), 
        WCSPH_.blockDim(), 
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif

        if (dummy_.deviceSize() == 0) return;

        launchUpdateSpatialGridCellHashStartEnd(dummy_.position(), 
        dummy_.hashIndex(), 
        dummy_.hashValue(), 
        spatialGrid_.cellHashStart(), 
        spatialGrid_.cellHashEnd(), 
        spatialGrid_.minimumBoundary(), 
        spatialGrid_.maximumBoundary(), 
        spatialGrid_.cellSize(), 
        spatialGrid_.gridSize(), 
        spatialGrid_.numGrids(),
        dummy_.deviceSize(), 
        dummy_.gridDim(), 
        dummy_.blockDim(), 
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif

        launchCountSPHDummyInteractions(WCSPH_.position(), 
        WCSPH_.smoothLength(), 
        SPHAndDummy_.objectPointed_.neighborCount(), 
        SPHAndDummy_.objectPointed_.neighborPrefixSum(), 
        dummy_.position(), 
        dummy_.smoothLength(), 
        dummy_.hashIndex(), 
        spatialGrid_.cellHashStart(), 
        spatialGrid_.cellHashEnd(), 
        spatialGrid_.minimumBoundary(), 
        spatialGrid_.cellSize(), 
        spatialGrid_.gridSize(), 
        WCSPH_.deviceSize(), 
        WCSPH_.gridDim(), 
        WCSPH_.blockDim(),
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif

        SPHAndDummy_.resizePair(stream_);

        launchWriteSPHDummyInteractions(WCSPH_.position(), 
        WCSPH_.smoothLength(), 
        SPHAndDummy_.objectPointed_.neighborPrefixSum(), 
        dummy_.position(), 
        dummy_.smoothLength(), 
        dummy_.hashIndex(), 
        SPHAndDummy_.pair_.objectPointed(),
        SPHAndDummy_.pair_.objectPointing(),
        spatialGrid_.cellHashStart(), 
        spatialGrid_.cellHashEnd(), 
        spatialGrid_.minimumBoundary(), 
        spatialGrid_.cellSize(), 
        spatialGrid_.gridSize(), 
        WCSPH_.deviceSize(), 
        WCSPH_.gridDim(), 
        WCSPH_.blockDim(),
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif

        SPHAndDummy_.buildObjectPointingInteractionStartEnd(maxThread_, stream_);
    }

    void SPHInteraction1stHalf(const double3 gravity, const double halfTimeStep)
    {
        launchWCSPH1stHalfIntegration(WCSPH_.position(), 
        WCSPH_.velocity(), 
        WCSPH_.acceleration(), 
        WCSPH_.density(), 
        WCSPH_.pressure(), 
        WCSPH_.soundSpeed(), 
        WCSPH_.mass(), 
        WCSPH_.initialDensity(), 
        WCSPH_.smoothLength(), 
        WCSPH_.viscosity(),
        SPHAndSPH_.objectPointed_.neighborPrefixSum(), 
        SPHAndDummy_.objectPointed_.neighborPrefixSum(), 
        dummy_.position(), 
        dummy_.velocity(), 
        dummy_.normal(), 
        dummy_.soundSpeed(), 
        dummy_.mass(), 
        dummy_.initialDensity(), 
        dummy_.smoothLength(), 
        dummy_.viscosity(),
        SPHAndSPH_.pair_.objectPointing(), 
        SPHAndDummy_.pair_.objectPointing(), 
        gravity, 
        halfTimeStep, 
        WCSPH_.deviceSize(), 
        WCSPH_.gridDim(), 
        WCSPH_.blockDim(),
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif
    }

    void SPHInteraction2ndHalf(const double3 gravity, const double halfTimeStep)
    {
        launchWCSPH2ndHalfIntegration(WCSPH_.position(), 
        WCSPH_.velocity(), 
        WCSPH_.acceleration(), 
        WCSPH_.densityChange(),
        WCSPH_.density(), 
        WCSPH_.pressure(), 
        WCSPH_.soundSpeed(), 
        WCSPH_.mass(), 
        WCSPH_.initialDensity(), 
        WCSPH_.smoothLength(), 
        WCSPH_.viscosity(),
        SPHAndSPH_.objectPointed_.neighborPrefixSum(), 
        SPHAndDummy_.objectPointed_.neighborPrefixSum(), 
        dummy_.position(), 
        dummy_.velocity(), 
        dummy_.normal(), 
        dummy_.soundSpeed(), 
        dummy_.mass(), 
        dummy_.initialDensity(), 
        dummy_.smoothLength(), 
        dummy_.viscosity(),
        SPHAndSPH_.pair_.objectPointing(), 
        SPHAndDummy_.pair_.objectPointing(), 
        gravity, 
        halfTimeStep, 
        WCSPH_.deviceSize(), 
        WCSPH_.gridDim(), 
        WCSPH_.blockDim(), 
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif
    }

    void outputWCSPHVTU(const std::string &dir, const size_t iFrame, const size_t iStep, const double time)
    {
        MKDIR(dir.c_str());
        std::ostringstream fname;
        fname << dir << "/SPH_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";
        std::ofstream out(fname.str().c_str());
        if (!out) throw std::runtime_error("Cannot open " + fname.str());
        out << std::fixed << std::setprecision(10);

        size_t N = WCSPH_.hostSize();
        std::vector<double3> p = WCSPH_.positionHostCopy();
        std::vector<double3> v = WCSPH_.velocityHostCopy();
        std::vector<double> d = WCSPH_.densityHostCopy();
        std::vector<double> pr = WCSPH_.pressureHostCopy();
        const std::vector<double> h = WCSPH_.smoothLengthHostRef();
        
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

        out << "      <PointData Scalars=\"smoothLength\">\n";

        out << "        <DataArray type=\"Float32\" Name=\"smoothLength\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << h[i];
        out << "\n        </DataArray>\n";

        out << "        <DataArray type=\"Float32\" Name=\"density\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << d[i];
        out << "\n        </DataArray>\n";

        out << "        <DataArray type=\"Float32\" Name=\"pressure\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << pr[i];
        out << "\n        </DataArray>\n";

        const struct {
            const char* name;
            const std::vector<double3>& vec;
        } vec3s[] = {
            { "velocity"       , v     }
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

    void outputDummyVTU(const std::string &dir, const size_t iFrame, const size_t iStep, const double time)
    {
        MKDIR(dir.c_str());
        std::ostringstream fname;
        fname << dir << "/dummy_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";
        std::ofstream out(fname.str().c_str());
        if (!out) throw std::runtime_error("Cannot open " + fname.str());
        out << std::fixed << std::setprecision(10);

        size_t N = dummy_.hostSize();
        std::vector<double3> p = dummy_.positionHostCopy();
        std::vector<double3> v = dummy_.velocityHostCopy();
        std::vector<double3> n = dummy_.normalHostCopy();
        const std::vector<double> h = dummy_.smoothLengthHostRef();
        
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

        out << "      <PointData Scalars=\"smoothLength\">\n";

        out << "        <DataArray type=\"Float32\" Name=\"smoothLength\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << h[i];
        out << "\n        </DataArray>\n";

        const struct {
            const char* name;
            const std::vector<double3>& vec;
        } vec3s[] = {
            { "velocity"       , v     },
            { "normal"         , n     }
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

    WCSPH& getSPH() { return WCSPH_; }

    WCSPH& getDummy() { return dummy_; }

    SPHInteraction& getSPHAndDummyInteraction() { return SPHAndDummy_; }

public:
    void addSPH(double3 position, double3 velocity, double soundSpeed, double spacing, double density, double viscosity)
    {
        if (density <= 0.) return;
        if (soundSpeed <= 0.) return;
        double mass = spacing * spacing * spacing * density;
        double3 zeroVec = make_double3(0., 0., 0.);
        WCSPH_.addHost(position, velocity, zeroVec, zeroVec, 0.0, density, 0.0, 
        density, 1.3 * spacing, mass, soundSpeed, viscosity, 0, -1, -1);
    }

    void addDummy(double3 position, double3 velocity, double soundSpeed, double spacing, double density, double viscosity, int clumpID = 0)
    {
        if (density <= 0.) return;
        if (soundSpeed <= 0.) return;
        double mass = spacing * spacing * spacing * density;
        double3 zeroVec = make_double3(0., 0., 0.);
        dummy_.addHost(position, velocity, zeroVec, zeroVec, 0.0, density, 0.0, 
        density, 1.3 * spacing, mass, soundSpeed, viscosity, clumpID, -1, -1);
    }

    void eraseSPH(const size_t index)
    {
        WCSPH_.eraseHost(index);
    }

    void eraseDummy(const size_t index)
    {
        dummy_.eraseHost(index);
    }

    void copySPH(const WCSPH& other)
    {
        WCSPH_.copyFromHost(other);
    }

    void copyDummy(const WCSPH& other)
    {
        dummy_.copyFromHost(other);
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
        SPHInitialize(minBoundary, maxBoundary);
        std::cout << "Initialization Completed." << std::endl;

        size_t iStep = 0, iFrame = 0;
        double time = 0.0;
        outputWCSPHVTU(dir, 0, 0, 0.0);
        outputDummyVTU(dir, 0, 0, 0.0);
        while (iStep <= numStep)
        {
            iStep++;
            time += timeStep;
            SPHNeighborSearch();
            SPHInteraction1stHalf(gravity, 0.5 * timeStep);
            SPHInteraction2ndHalf(gravity, 0.5 * timeStep);
            if (iStep % frameInterval == 0) 
            {
                iFrame++;
                std::cout << "Frame " << iFrame << " at Time " << time << std::endl;
                outputWCSPHVTU(dir, iFrame, iStep, time);
                outputDummyVTU(dir, iFrame, iStep, time);
            }
        }
        WCSPH_.copyDeviceToHost(stream_);
        dummy_.copyDeviceToHost(stream_);
    }

private:
    size_t maxThread_;
    cudaStream_t stream_;

    WCSPH WCSPH_;
    WCSPH dummy_;

    SPHInteraction SPHAndSPH_;
    SPHInteraction SPHAndDummy_;
    SPHInteraction dummyAndDummy_;
    spatialGrid spatialGrid_;
};