#include "neighborSearchKernel.cuh"
#include "myUtility/myVec.h"
#include "myUtility/buildHashStartEnd.cuh"

/**
 * @brief Compute spatial-hash (cell linear index) for each object position.
 *
 * The position is clamped into the domain [minBound, maxBound) before hashing:
 * - if p < minBound  -> clamp to minBound
 * - if p >= maxBound -> clamp to (maxBound - 0.5 * cellSize) to avoid mapping to an out-of-range cell
 *
 * Hash definition:
 * - gridPosition = calculateGridPosition(p, minBound, cellSize)
 * - hashValue[idx] = linearIndex3D(gridPosition, gridSize)
 *
 * @param[out] hashValue     Per-object hash (cell linear index) array.
 * @param[in]  position      Per-object positions in global coordinates.
 * @param[in]  minBound      Global domain minimum corner.
 * @param[in]  maxBound      Global domain maximum corner (exclusive boundary conceptually).
 * @param[in]  cellSize      Cell size in each axis (dx, dy, dz).
 * @param[in]  gridSize      Number of cells in each axis (nx, ny, nz).
 * @param[in]  numObject     Number of objects.
 */
__global__ void calculateHashKernel(int* hashValue, 
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

/**
 * @brief Build "dummy" positions for periodic wrapping near selected boundaries.
 *
 * For each object:
 * - domainSize = maxBound - minBound
 * - If directionFlag.{x,y,z} == 1 and the object is within one cell of the min-side boundary,
 *   the coordinate is shifted by +domainSize in that axis:
 *     if (p.x - minBound.x < cellSize.x) p.x += domainSize.x   (same for y,z)
 *
 * This is typically used to create ghost copies across periodic boundaries.
 *
 * @param[out] position_dummy   Output dummy positions (global coordinates).
 * @param[in]  position         Original positions (global coordinates).
 * @param[in]  minBound         Global domain minimum corner.
 * @param[in]  maxBound         Global domain maximum corner.
 * @param[in]  cellSize         Cell size (used as the "near boundary" threshold).
 * @param[in]  directionFlag    Which axes are periodic/duplicated (1 => enable wrapping on that axis).
 * @param[in]  numObject        Number of objects.
 */
__global__ void buildDummyPositionKernel(double3* position_dummy, 
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

/**
 * @brief Compute spatial-hash only for objects that are outside the domain (dummy hashing).
 *
 * Behavior:
 * - hashValue[idx] is initialized to -1 (meaning "not a dummy / not outside")
 * - If the object position is outside [minBound, maxBound), it is clamped back into the domain,
 *   then hashed into a cell and hashValue[idx] is set to that cell index.
 *
 * This is typically used to build neighbor candidates for periodic/ghost/dummy particles near boundaries.
 *
 * @param[out] hashValue     Per-object dummy hash array. -1 means "not outside".
 * @param[in]  position      Per-object positions in global coordinates.
 * @param[in]  minBound      Global domain minimum corner.
 * @param[in]  maxBound      Global domain maximum corner.
 * @param[in]  cellSize      Cell size in each axis.
 * @param[in]  gridSize      Number of cells in each axis.
 * @param[in]  numObject     Number of objects.
 */
__global__ void calculateDummyHashKernel(int* hashValue, 
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
    calculateHashKernel <<< gridD_GPU, blockD_GPU, 0, stream_GPU >>> (hashValue, 
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
    buildDummyPositionKernel <<< gridD_GPU, blockD_GPU, 0, stream_GPU >>> (position_dummy, 
    position, 
    minBound, 
    maxBound,
    cellSize, 
    directionFlag,
    numObject);
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
    calculateDummyHashKernel <<< gridD_GPU, blockD_GPU, 0, stream_GPU >>> (hashValue, 
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