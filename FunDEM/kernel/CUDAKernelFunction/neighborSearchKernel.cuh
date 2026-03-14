#pragma once
#include <vector_functions.h>

__device__ __forceinline__ int3 calculateGridPosition(double3 position, const double3 minBoundary, const double3 cellSize)
{
    return make_int3(int((position.x - minBoundary.x) / cellSize.x),
    int((position.y - minBoundary.y) / cellSize.y),
    int((position.z - minBoundary.z) / cellSize.z));
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
cudaStream_t stream_GPU);

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
cudaStream_t stream_GPU);

extern "C" void launchBuildDummyPosition(double3* position_dummy, 
const double3* position, 
const double3 minBound, 
const double3 maxBound,
const double3 cellSize, 
const int3 directionFlag,
const size_t numObject,
const size_t gridD_GPU, 
const size_t blockD_GPU, 
cudaStream_t stream_GPU);