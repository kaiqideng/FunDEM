#pragma once
#include <vector_functions.h>

/**
 * @brief Launch ball-ball neighbor counting and build inclusive prefix sum.
 *
 * Runs:
 *   1) countBallInteractionsKernel
 *   2) buildPrefixSum(neighborPrefixSum, neighborCount)
 *
 * @param[out] position              Ball center positions (device).
 * @param[out] radius                Ball radii (device).
 * @param[out] inverseMass           Ball inverse masses (device).
 * @param[out] clumpID               Ball clump ids (device).
 * @param[out] hashIndex             Sorted ball index list (device).
 * @param[out] neighborCount         neighborCount[idxA] output (device), length = numBall.
 * @param[out] neighborPrefixSum     inclusive_scan(neighborCount) (device), length = numBall.

 * @param[in]  cellHashStart         Spatial grid cell start array (device).
 * @param[in]  cellHashEnd           Spatial grid cell end array (device).
 * @param[in]  minBound              Global minimum boundary of spatial grid.
 * @param[in]  cellSize              Spatial grid cell size.
 * @param[in]  gridSize              Spatial grid resolution in cells.

 * @param[in]  numBall               Number of balls.
 * @param[in]  gridD_GPU             Grid dimension for launching kernels.
 * @param[in]  blockD_GPU            Block dimension for launching kernels.
 * @param[in]  stream_GPU            CUDA stream.
 */
extern "C" void launchCountBallInteractions(double3* position, 
double* radius,
double* inverseMass,
int* clumpID,
int* hashIndex, 
int* neighborCount,
int* neighborPrefixSum,

int* cellHashStart,
int* cellHashEnd,

const double3 minBound,
const double3 cellSize,
const int3 gridSize,

const size_t numBall,
const size_t gridD_GPU, 
const size_t blockD_GPU, 
cudaStream_t stream_GPU);

/**
 * @brief Launch ball-ball pair construction (write pairs + reuse springs).
 *
 * Runs:
 *   1) writeBallInteractionsKernel
 *
 * @param[in]  position                      Ball center positions (device).
 * @param[in]  radius                        Ball radii (device).
 * @param[in]  inverseMass                   Ball inverse masses (device).
 * @param[in]  clumpID                       Ball clump ids (device).
 * @param[in]  hashIndex                     Sorted ball index list (device).
 * @param[in]  neighborPrefixSum             inclusive_scan(neighborCount) (device), length = numBall.
 * @param[in]  interactionStart_old          Old "pointing list" start offsets (device).
 * @param[in]  interactionEnd_old            Old "pointing list" end offsets (device).
 *
 * @param[out] slidingSpring                 New sliding springs (device).
 * @param[out] rollingSpring                 New rolling springs (device).
 * @param[out] torsionSpring                 New torsion springs (device).
 * @param[out] objectPointed                 New pair pointed indices (device).
 * @param[out] objectPointing                New pair pointing indices (device).
 *
 * @param[in]  slidingSpring_old             Old sliding springs (device).
 * @param[in]  rollingSpring_old             Old rolling springs (device).
 * @param[in]  torsionSpring_old             Old torsion springs (device).
 * @param[in]  objectPointed_old             Old pair pointed indices (device).
 * @param[in]  neighborPairHashIndex_old     Old pair remap indices (device).
 *
 * @param[in]  cellHashStart                 Spatial grid cell start array (device).
 * @param[in]  cellHashEnd                   Spatial grid cell end array (device).
 * @param[in]  minBound                      Global minimum boundary of spatial grid.
 * @param[in]  cellSize                      Spatial grid cell size.
 * @param[in]  gridSize                      Spatial grid resolution.
 * @param[in]  numBall                       Number of balls.
 * @param[in]  gridD_GPU                     Grid dimension.
 * @param[in]  blockD_GPU                    Block dimension.
 * @param[in]  stream_GPU                    CUDA stream.
 */
extern "C" void launchWriteBallInteractions(double3* position, 
double* radius,
double* inverseMass,
int* clumpID,
int* hashIndex, 
int* neighborPrefixSum,
int* interactionStart_old,
int* interactionEnd_old,

double3* slidingSpring,
double3* rollingSpring,
double3* torsionSpring,
int* objectPointed,
int* objectPointing,

double3* slidingSpring_old,
double3* rollingSpring_old,
double3* torsionSpring_old,
int* objectPointed_old,
int* neighborPairHashIndex_old,

int* cellHashStart,
int* cellHashEnd,

const double3 minBound,
const double3 cellSize,
const int3 gridSize,

const size_t numBall,
const size_t gridD_GPU, 
const size_t blockD_GPU, 
cudaStream_t stream_GPU);

/**
 * @brief Launch ball-triangle neighbor counting and build inclusive prefix sum.
 *
 * Runs:
 *   1) countBallTriangleInteractionsKernel
 *   2) buildPrefixSum(neighborPrefixSum, neighborCount)
 *
 * @param[in]  position              Ball center positions (device).
 * @param[in]  radius                Ball radii (device).
 * @param[out] neighborCount         Candidate triangle count per ball (device), length = numBall.
 * @param[out] neighborPrefixSum     inclusive_scan(neighborCount) (device), length = numBall.
 *
 * @param[in]  index0_tri            Triangle vertex index0 (device).
 * @param[in]  index1_tri            Triangle vertex index1 (device).
 * @param[in]  index2_tri            Triangle vertex index2 (device).
 * @param[in]  hashIndex_tri         Sorted triangle index list (device).

 * @param[in]  globalPosition_ver    Vertex positions (device).
 *
 * @param[in]  cellHashStart         Triangle grid cell start array (device).
 * @param[in]  cellHashEnd           Triangle grid cell end array (device).
 * @param[in]  minBound              Global minimum boundary of triangle grid.
 * @param[in]  cellSize              Triangle grid cell size.
 * @param[in]  gridSize              Triangle grid resolution.
 *
 * @param[in]  numBall               Number of balls.
 * @param[in]  gridD_GPU             Grid dimension.
 * @param[in]  blockD_GPU            Block dimension.
 * @param[in]  stream_GPU            CUDA stream.
 */
extern "C" void launchCountBallTriangleInteractions(double3* position, 
double* radius,
int* neighborCount,
int* neighborPrefixSum,

int* index0_tri, 
int* index1_tri,
int* index2_tri,
int* hashIndex_tri,

double3* globalPosition_ver,

int* cellHashStart,
int* cellHashEnd,

const double3 minBound,
const double3 cellSize,
const int3 gridSize,

const size_t numBall,
const size_t gridD_GPU, 
const size_t blockD_GPU, 
cudaStream_t stream_GPU);

/**
 * @brief Launch ball-triangle pair construction (write pairs + reuse springs).
 *
 * Runs:
 *   1) writeBallTriangleInteractionsKernel
 *
 * @param[in]  position                      Ball center positions (device).
 * @param[in]  radius                        Ball radii (device).
 * @param[in]  neighborPrefixSum             inclusive_scan(neighborCount) (device), length = numBall.
 *
 * @param[in]  index0_tri                    Triangle vertex index0 (device).
 * @param[in]  index1_tri                    Triangle vertex index1 (device).
 * @param[in]  index2_tri                    Triangle vertex index2 (device).
 * @param[in]  hashIndex_tri                 Sorted triangle index list (device).
 * @param[in]  interactionStart_tri_old      Old "pointing list" start offsets per triangle (device).
 * @param[in]  interactionEnd_tri_old        Old "pointing list" end offsets per triangle (device).

 * @param[in]  globalPosition_ver            Vertex positions (device).
 *
 * @param[out] slidingSpring                 New sliding springs (device).
 * @param[out] rollingSpring                 New rolling springs (device).
 * @param[out] torsionSpring                 New torsion springs (device).
 * @param[out] objectPointed                 New pairs: ball idxA (device).
 * @param[out] objectPointing                New pairs: triangle idxB (device).
 *
 * @param[in]  slidingSpring_old             Old sliding springs (device).
 * @param[in]  rollingSpring_old             Old rolling springs (device).
 * @param[in]  torsionSpring_old             Old torsion springs (device).
 * @param[in]  objectPointed_old             Old pointed indices (ball idxA) per old pair (device).
 * @param[in]  neighborPairHashIndex_old     Old pair remap indices (device).
 *
 * @param[in]  cellHashStart                 Triangle grid cell start array (device).
 * @param[in]  cellHashEnd                   Triangle grid cell end array (device).
 * @param[in]  minBound                      Global minimum boundary of triangle grid.
 * @param[in]  cellSize                      Triangle grid cell size.
 * @param[in]  gridSize                      Triangle grid resolution.
 *
 * @param[in]  numBall                       Number of balls.
 * @param[in]  gridD_GPU                     Grid dimension.
 * @param[in]  blockD_GPU                    Block dimension.
 * @param[in]  stream_GPU                    CUDA stream.
 */
extern "C" void launchWriteBallTriangleInteractions(double3* position, 
double* radius,
int* neighborPrefixSum,

int* index0_tri, 
int* index1_tri,
int* index2_tri,
int* hashIndex_tri,
int* interactionStart_tri_old,
int* interactionEnd_tri_old,

double3* globalPosition_ver,

double3* slidingSpring,
double3* rollingSpring,
double3* torsionSpring,
int* objectPointed,
int* objectPointing,

double3* slidingSpring_old,
double3* rollingSpring_old,
double3* torsionSpring_old,
int* objectPointed_old,
int* neighborPairHashIndex_old,

int* cellHashStart,
int* cellHashEnd,

const double3 minBound,
const double3 cellSize,
const int3 gridSize,

const size_t numBall,
const size_t gridD_GPU, 
const size_t blockD_GPU, 
cudaStream_t stream_GPU);

