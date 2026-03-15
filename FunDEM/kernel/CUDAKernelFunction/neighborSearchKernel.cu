#include "neighborSearchKernel.cuh"
#include "myUtility/myVec.h"
#include "myUtility/buildHashStartEnd.cuh"

__global__ void calculateHash(int* hashValue, 
const double3* position, 
const double3 minBound, 
const double3 maxBound, 
const double3 cellSize, 
const int3 gridSize,
const size_t numObject)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numObject) return;
    
    double3 p = position[idx];
    if (p.x < minBound.x) p.x = minBound.x;
    else if (p.x >= maxBound.x) p.x = maxBound.x - 0.5 * cellSize.x;
    if (p.y < minBound.y) p.y = minBound.y;
    else if (p.y >= maxBound.y) p.y = maxBound.y - 0.5 * cellSize.y;
    if (p.z < minBound.z) p.z = minBound.z;
    else if (p.z >= maxBound.z) p.z = maxBound.z - 0.5 * cellSize.z;
    
    int3 gridPosition = calculateGridPosition(p, minBound, cellSize);
    hashValue[idx] = linearIndex3D(gridPosition, gridSize);
}

__global__ void calculateDummyHash(int* hashValue, 
const double3* position, 
const double3 minBound, 
const double3 maxBound, 
const double3 cellSize, 
const int3 gridSize,
const size_t numObject)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numObject) return;
    
    hashValue[idx] = -1;

    double3 p = position[idx];
    bool flag = false;
    if (p.x < minBound.x) { flag = true; p.x = minBound.x; }
    else if (p.x >= maxBound.x) { flag = true; p.x = maxBound.x - 0.5 * cellSize.x; }
    if (p.y < minBound.y) { flag = true; p.y = minBound.y; }
    else if (p.y >= maxBound.y) { flag = true; p.y = maxBound.y - 0.5 * cellSize.y; }
    if (p.z < minBound.z) { flag = true; p.z = minBound.z; }
    else if (p.z >= maxBound.z) { flag = true; p.z = maxBound.z - 0.5 * cellSize.z; }

    if (flag)
    {
        int3 gridPosition = calculateGridPosition(p, minBound, cellSize);
        hashValue[idx] = linearIndex3D(gridPosition, gridSize);
    }
}

__global__ void buildDummyPosition(double3* position_dummy, 
const double3* position, 
const double3 minBound, 
const double3 maxBound,
const double3 cellSize, 
const int3 directionFlag,
const size_t numObject)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numObject) return;

    double3 domainSize = maxBound - minBound;
    double3 p = position[idx];
    
    if (directionFlag.x == 1 && p.x - minBound.x < cellSize.x) { p.x += domainSize.x; }
    if (directionFlag.y == 1 && p.y - minBound.y < cellSize.y) { p.y += domainSize.y; }
    if (directionFlag.z == 1 && p.z - minBound.z < cellSize.z) { p.z += domainSize.z; }

    position_dummy[idx] = p;
}

extern "C" void launchUpdateSpatialGridCellHashStartEnd(double3* position, 
int* hashIndex, 
int* hashValue, 

int* cellHashStart,
int* cellHashEnd,

const double3 minBound,
const double3 maxBound,
const double3 cellSize,
const int3 gridSize,
const size_t numGrids,

const size_t numObject,
const size_t gridD_GPU, 
const size_t blockD_GPU, 
cudaStream_t stream_GPU)
{
    calculateHash <<< gridD_GPU, blockD_GPU, 0, stream_GPU >>> (hashValue, 
    position, 
    minBound, 
    maxBound, 
    cellSize, 
    gridSize,
    numObject);

    buildHashStartEnd(cellHashStart,
    cellHashEnd,
    hashIndex,
    hashValue,
    numGrids,
    numObject,
    gridD_GPU,
    blockD_GPU,
    stream_GPU);
}

extern "C" void launchUpdateDummySpatialGridCellHashStartEnd(double3* position, 
int* hashIndex, 
int* hashValue, 

int* cellHashStart,
int* cellHashEnd,

const double3 minBound,
const double3 maxBound,
const double3 cellSize,
const int3 gridSize,
const size_t numGrids,

const size_t numObject,
const size_t gridD_GPU, 
const size_t blockD_GPU, 
cudaStream_t stream_GPU)
{
    calculateDummyHash <<< gridD_GPU, blockD_GPU, 0, stream_GPU >>> (hashValue, 
    position, 
    minBound, 
    maxBound, 
    cellSize, 
    gridSize,
    numObject);

    buildHashStartEnd(cellHashStart,
    cellHashEnd,
    hashIndex,
    hashValue,
    numGrids,
    numObject,
    gridD_GPU,
    blockD_GPU,
    stream_GPU);
}

extern "C" void launchBuildDummyPosition(double3* position_dummy, 
const double3* position, 

const double3 minBound, 
const double3 maxBound,
const double3 cellSize, 
const int3 directionFlag,

const size_t numObject,
const size_t gridD_GPU, 
const size_t blockD_GPU, 
cudaStream_t stream_GPU)
{
    buildDummyPosition <<< gridD_GPU, blockD_GPU, 0, stream_GPU >>> (position_dummy, 
    position, 
    minBound, 
    maxBound,
    cellSize, 
    directionFlag,
    numObject);
}