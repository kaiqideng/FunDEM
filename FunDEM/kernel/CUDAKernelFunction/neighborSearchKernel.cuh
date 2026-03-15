#pragma once
#include <vector_functions.h>

__device__ __forceinline__ int3 calculateGridPosition(double3 position, const double3 minBoundary, const double3 cellSize)
{
    return make_int3(int((position.x - minBoundary.x) / cellSize.x),
    int((position.y - minBoundary.y) / cellSize.y),
    int((position.z - minBoundary.z) / cellSize.z));
}

/**
 * @brief Update spatial grid cell ranges (cellHashStart/cellHashEnd) for a set of objects.
 *
 * Pipeline:
 *  1) calculateHashKernel(...) fills hashValue[idx] = cell linear index for each object
 *  2) buildHashStartEnd(...) sorts (hashValue, hashIndex) by hashValue and builds:
 *     - cellHashStart[cell] = first index in sorted list for that cell, or -1 if empty
 *     - cellHashEnd[cell]   = end index (exclusive) in sorted list for that cell, or -1 if empty
 *
 * After this, for any cell c:
 * - range is [cellHashStart[c], cellHashEnd[c]) in the sorted hashIndex/hashValue arrays.
 *
 * @param[in]  position        Object positions in global coordinates.
 * @param[out] hashIndex       Permutation index array (sorted by cell after buildHashStartEnd).
 * @param[in,out] hashValue    Hash (cell index) array; will be sorted in-place by buildHashStartEnd.
 *
 * @param[out] cellHashStart   Per-cell start offset into the sorted list (or -1).
 * @param[out] cellHashEnd     Per-cell end offset (exclusive) into the sorted list (or -1).
 *
 * @param[in]  minBound        Global domain minimum corner.
 * @param[in]  maxBound        Global domain maximum corner.
 * @param[in]  cellSize        Cell size.
 * @param[in]  gridSize        Number of cells per axis.
 * @param[in]  numGrids        Total number of cells (usually gridSize.x*gridSize.y*gridSize.z).
 *
 * @param[in]  numObject       Number of objects.
 * @param[in]  gridD_GPU       CUDA grid dimension for kernels.
 * @param[in]  blockD_GPU      CUDA block dimension for kernels.
 * @param[in]  stream_GPU      CUDA stream.
 */
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

/**
 * @brief Launch dummy position generation kernel for periodic boundary handling.
 *
 * Runs:
 *  1) buildDummyPositionKernel(...)
 *
 * @param[out] position_dummy   Output dummy positions.
 * @param[in]  position         Input positions.
 * @param[in]  minBound         Domain minimum corner.
 * @param[in]  maxBound         Domain maximum corner.
 * @param[in]  cellSize         Cell size (used as near-boundary threshold).
 * @param[in]  directionFlag    Axes to wrap (1 => wrap on that axis).
 * @param[in]  numObject        Number of objects.
 * @param[in]  gridD_GPU        CUDA grid dimension.
 * @param[in]  blockD_GPU       CUDA block dimension.
 * @param[in]  stream_GPU       CUDA stream.
 */
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

/**
 * @brief Update spatial grid cell ranges for "dummy" objects that are outside the domain.
 *
 * Pipeline:
 *  1) calculateDummyHashKernel(...) sets hashValue[idx] = cell index only if the object is outside;
 *     otherwise hashValue[idx] remains -1.
 *  2) buildHashStartEnd(...) sorts by hashValue and builds per-cell ranges.
 *
 * Note:
 * - Because many entries may have hashValue=-1, your downstream code must be consistent about whether
 *   it expects to include those -1 entries or ignore them.
 * - buildHashStartEnd currently skips invalid h in findStartAndEnd (h<0), so cells are built only
 *   for valid hash values.
 *
 * @param[in]  position        Object positions.
 * @param[out] hashIndex       Permutation index array after sorting.
 * @param[in,out] hashValue    Hash array (dummy hash); sorted in-place.
 * @param[out] cellHashStart   Per-cell start offsets.
 * @param[out] cellHashEnd     Per-cell end offsets (exclusive).
 * @param[in]  minBound        Global domain minimum corner.
 * @param[in]  maxBound        Global domain maximum corner.
 * @param[in]  cellSize        Cell size.
 * @param[in]  gridSize        Grid size in cells.
 * @param[in]  numGrids        Total number of cells.
 * @param[in]  numObject       Number of objects.
 * @param[in]  gridD_GPU       CUDA grid dimension.
 * @param[in]  blockD_GPU      CUDA block dimension.
 * @param[in]  stream_GPU      CUDA stream.
 */
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