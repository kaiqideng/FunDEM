#pragma once
#include "myUtility/myQua.h"

__device__ __forceinline__ double interpolateLevelSetFunctionValue(const double x, 
const double y, 
const double z, 
const double phi000, 
const double phi100,
const double phi010,
const double phi110,
const double phi001,
const double phi101,
const double phi011,
const double phi111)
{
    // Standard trilinear weights
    const double wx0 = 1.0 - x;
    const double wx1 = x;
    const double wy0 = 1.0 - y;
    const double wy1 = y;
    const double wz0 = 1.0 - z;
    const double wz1 = z;

    return phi000 * wx0 * wy0 * wz0 +
    phi100 * wx1 * wy0 * wz0 +
    phi010 * wx0 * wy1 * wz0 +
    phi110 * wx1 * wy1 * wz0 +
    phi001 * wx0 * wy0 * wz1 +
    phi101 * wx1 * wy0 * wz1 +
    phi011 * wx0 * wy1 * wz1 +
    phi111 * wx1 * wy1 * wz1;
}

__device__ __forceinline__ double3 interpolateLevelSetFunctionGradient(const int x, 
const double y, 
const double z, 
const double phi000, 
const double phi100,
const double phi010,
const double phi110,
const double phi001,
const double phi101,
const double phi011,
const double phi111)
{
    const double wx0 = 1.0 - x;
    const double wx1 = x;
    const double wy0 = 1.0 - y;
    const double wy1 = y;
    const double wz0 = 1.0 - z;
    const double wz1 = z;

    // dphi/dx_normalized
    const double dphidx_n =
    (phi100 - phi000) * wy0 * wz0 +
    (phi110 - phi010) * wy1 * wz0 +
    (phi101 - phi001) * wy0 * wz1 +
    (phi111 - phi011) * wy1 * wz1;

    // dphi/dy_normalized
    const double dphidy_n =
    (phi010 - phi000) * wx0 * wz0 +
    (phi110 - phi100) * wx1 * wz0 +
    (phi011 - phi001) * wx0 * wz1 +
    (phi111 - phi101) * wx1 * wz1;

    // dphi/dz_normalized
    const double dphidz_n =
    (phi001 - phi000) * wx0 * wy0 +
    (phi101 - phi100) * wx1 * wy0 +
    (phi011 - phi010) * wx0 * wy1 +
    (phi111 - phi110) * wx1 * wy1;

    return make_double3(dphidx_n,
    dphidy_n,
    dphidz_n);
}

/**
 * @brief Launch: count LS-boundary-node interactions and build neighbor prefix sum.
 *
 * This launches:
 * 1) countLevelSetBoundaryNodeInteractionsKernel -> neighborCount_bNode
 * 2) buildPrefixSum (Thrust inclusive_scan)      -> neighborPrefixSum_bNode
 *
 * @param[out] neighborCount_bNode       Contact count per boundary node.
 * @param[out] neighborPrefixSum_bNode   Inclusive prefix sum over neighborCount_bNode.
 * @param[in]  localPosition_bNode       Boundary node local positions (in owner particle frame).
 * @param[in]  particleID_bNode          Owner particle id for each boundary node.

 * @param[in]  LSFV_gNode                Concatenated level-set grid node values for all particles.

 * @param[in]  position_p                Particle global positions.
 * @param[in]  orientation_p             Particle orientations.
 * @param[in]  radii_p                   Bounding radius per particle.
 * @param[in]  inverseMass_p             Inverse mass per particle.
 * @param[in]  gridSpacing_p             Grid spacing per particle.
 * @param[in]  gridNodeLocalOrigin_p     Local grid origin per particle.
 * @param[in]  gridNodeSize_p            Grid size per particle.
 * @param[in]  gridNodePrefixSum_p       Inclusive prefix sum of per-particle grid node counts.
 * @param[in]  hashIndex_p               Spatial hash sorted indices (cell range -> particle index).

 * @param[in]  cellHashStart             Spatial hash cell start indices.
 * @param[in]  cellHashEnd               Spatial hash cell end indices.

 * @param[in]  minBound                  Spatial grid min bound.
 * @param[in]  cellSize                  Spatial grid cell size.
 * @param[in]  gridSize                  Spatial grid resolution.

 * @param[in]  numBoundaryNode           Total number of boundary nodes.
 * @param[in]  gridD                     CUDA grid dimension (number of blocks).
 * @param[in]  blockD                    CUDA block dimension (threads per block).
 * @param[in]  stream                    CUDA stream used for kernels and prefix sum.
 */
extern "C" void launchCountLevelSetBoundaryNodeInteractions(int* neighborCount_bNode,
int* neighborPrefixSum_bNode,

const double3* localPosition_bNode,
const int* particleID_bNode,

const double* LSFV_gNode,

const double3* position_p,
const quaternion* orientation_p,
const double* radii_p,
const double* inverseMass_p,
const double* gridSpacing_p,
const double3* gridNodeLocalOrigin_p,
const int3* gridNodeSize_p,
const int* gridNodePrefixSum_p,
const int* hashIndex_p,

const int* cellHashStart,
const int* cellHashEnd,

const double3 minBound,
const double3 cellSize,
const int3 gridSize,

const size_t numBoundaryNode,
const size_t gridD,
const size_t blockD,
cudaStream_t stream);

/**
 * @brief Launch: write LS-boundary-node interactions and initialize contact state.
 *
 * This launches:
 * - writeLevelSetBoundaryNodeInteractionsKernel
 *
 * @param[out] slidingSpring             Sliding spring per interaction.
 * @param[out] rollingSpring             Rolling spring per interaction.
 * @param[out] torsionSpring             Torsion spring per interaction.
 * @param[out] contactPoint              Contact point (GLOBAL) per interaction.
 * @param[out] contactNormal             Contact normal (GLOBAL) per interaction.
 * @param[out] contactOverlap            Overlap per interaction (positive).
 * @param[out] objectPointed             Boundary node index per interaction.
 * @param[out] objectPointing            Particle index per interaction.
 *
 * @param[in]  slidingSpring_old          Old sliding spring (for state carry-over).
 * @param[in]  rollingSpring_old          Old rolling spring.
 * @param[in]  torsionSpring_old          Old torsion spring.
 * @param[in]  objectPointed_old          Old boundary node index per old interaction.
 * @param[in]  neighborPairHashIndex_old  Map old adjacency slot -> old packed interaction index.
 *
 * @param[in]  localPosition_bNode        Boundary node local positions.
 * @param[in]  particleID_bNode           Owner particle id per boundary node.
 * @param[in]  neighborPrefixSum_bNode    Prefix sum computed from neighbor counts (defines output segments).
 *
 * @param[in]  LSFV_gNode                 Concatenated level-set grid node values.
 *
 * @param[in]  position_p                 Particle global positions.
 * @param[in]  orientation_p              Particle orientations.
 * @param[in]  radii_p                    Bounding radius per particle.
 * @param[in]  inverseMass_p              Inverse mass per particle.
 * @param[in]  gridSpacing_p              Grid spacing per particle.
 * @param[in]  gridNodeLocalOrigin_p      Local grid origin per particle.
 * @param[in]  gridNodeSize_p             Grid size per particle.
 * @param[in]  gridNodePrefixSum_p        Inclusive prefix sum of grid node counts per particle.
 * @param[in]  hashIndex_p                Spatial hash sorted indices.
 *
 * @param[in]  interactionStart_p_old     Old interaction start per particle (for searching old pairs).
 * @param[in]  interactionEnd_p_old       Old interaction end per particle.
 *
 * @param[in]  cellHashStart              Spatial hash start.
 * @param[in]  cellHashEnd                Spatial hash end.

 * @param[in]  minBound                   Spatial grid min bound.
 * @param[in]  cellSize                   Spatial grid cell size.
 * @param[in]  gridSize                   Spatial grid resolution.
 *
 * @param[in]  numBoundaryNode            Total boundary nodes.
 * @param[in]  gridD                      CUDA grid dimension.
 * @param[in]  blockD                     CUDA block dimension.
 * @param[in]  stream                     CUDA stream.
 */
extern "C" void launchWriteLevelSetBoundaryNodeInteractions(double3* slidingSpring,
double3* rollingSpring,
double3* torsionSpring,
double3* contactPoint,
double3* contactNormal,
double* contactOverlap,
int* objectPointed,
int* objectPointing,

const double3* slidingSpring_old,
const double3* rollingSpring_old,
const double3* torsionSpring_old,
const int* objectPointed_old,
const int* neighborPairHashIndex_old,

const double3* localPosition_bNode,
const int* particleID_bNode,

const int* neighborPrefixSum_bNode,

const double* LSFV_gNode,

const double3* position_p,
const quaternion* orientation_p,
const double* radii_p,
const double* inverseMass_p,
const double* gridSpacing_p,
const double3* gridNodeLocalOrigin_p,
const int3* gridNodeSize_p,
const int* gridNodePrefixSum_p,
const int* hashIndex_p,

const int* interactionStart_p_old,
const int* interactionEnd_p_old,

const int* cellHashStart,
const int* cellHashEnd,

const double3 minBound,
const double3 cellSize,
const int3 gridSize,

const size_t numBoundaryNode,
const size_t gridD,
const size_t blockD,
cudaStream_t stream);