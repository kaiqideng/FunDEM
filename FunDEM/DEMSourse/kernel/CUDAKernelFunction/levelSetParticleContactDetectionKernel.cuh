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
const double g,
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

    const double invG = 1.0 / g;

    return make_double3(dphidx_n * invG,
    dphidy_n * invG,
    dphidz_n * invG);
}

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

const int* interactionStart_old_p,
const int* interactionEnd_old_p,

const int* cellHashStart,
const int* cellHashEnd,
const double3 minBound,
const double3 cellSize,
const int3 gridSize,

const size_t numBoundaryNode,
const size_t gridD,
const size_t blockD,
cudaStream_t stream);