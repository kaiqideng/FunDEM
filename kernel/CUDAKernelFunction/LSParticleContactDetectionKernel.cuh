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

extern "C" void launchBuildLevelSetBoundaryNodeInteractions1st(int* boundaryNodeNeighborCount,
const int* localPositionID_bNode_p,
const int* particleID_bNode_p,

const double* radius_geo_p,
const double3* gridNodeLocalOrigin_geo_p,
const double* gridNodeInverseSpacing_geo_p,
const int3* gridNodeSize_geo_p,
const int* SDFPrefixSum_geo_p,
const double* SDF_geo_p,
const double3* boundaryNodeLocalPosition_geo_p,

const double3* position_p,
const quaternion* orientation_p,
const int* geometryID_p,
const int* hashIndex_p,

const int* hashStart_spa_p,
const int* hashEnd_spa_p,

const double3 minBound_spa_p,
const double3 inverseCellSize_spa_p,
const int3 size3D_spa_p,

const size_t numBoundaryNode,
const size_t gridD,
const size_t blockD,
cudaStream_t stream);

extern "C" void launchBuildLevelSetBoundaryNodeInteractions2nd(double3* slidingSpring,
double3* contactPoint,
double3* contactNormal,
double* contactOverlap,
int* boundaryNodePointed,
int* objectPointing,
const double3* slidingSpring_old,
const int* objectPointing_old,

const int* boundaryNodeNeighborPrefixSum,
const int* boundaryNodeNeighborPrefixSum_old,

const int* localPositionID_bNode_p,
const int* particleID_bNode_p,

const double* radius_geo_p,
const double3* gridNodeLocalOrigin_geo_p,
const double* gridNodeInverseSpacing_geo_p,
const int3* gridNodeSize_geo_p,
const int* SDFPrefixSum_geo_p,
const double* SDF_geo_p,
const double3* boundaryNodeLocalPosition_geo_p,

const double3* position_p,
const quaternion* orientation_p,
const int* geometryID_p,
const int* hashIndex_p,

const int* hashStart_spa_p, 
const int* hashEnd_spa_p,

const double3 minBound_spa_p, 
const double3 inverseCellSize_spa_p, 
const int3 size3D_spa_p,

const size_t numBoundaryNode,
const size_t gridD,
const size_t blockD,
cudaStream_t stream);

extern "C" void launchBuildLevelSetBoundaryNodeFixedParticleInteractions1st(int* boundaryNodeNeighborCount,
const int* localPositionID_bNode_p,
const int* particleID_bNode_p,

const double3* boundaryNodeLocalPosition_geo_p,

const double3* position_p,
const quaternion* orientation_p,

const double3* gridNodeLocalOrigin_geo_fp,
const double* gridNodeInverseSpacing_geo_fp,
const int3* gridNodeSize_geo_fp,
const int* SDFPrefixSum_geo_fp,
const double* SDF_geo_fp,

const double3* position_fp,
const quaternion* orientation_fp,
const int* geometryID_fp,
const int* hashIndex_fp,

const int* hashStart_spa_fp, 
const int* hashEnd_spa_fp,

const double3 minBound_spa_fp, 
const double3 inverseCellSize_spa_fp, 
const int3 size3D_spa_fp, 

const size_t numBoundaryNode,
const size_t gridD,
const size_t blockD,
cudaStream_t stream);

extern "C" void launchBuildLevelSetBoundaryNodeFixedParticleInteractions2nd(double3* slidingSpring,
double3* contactPoint,
double3* contactNormal,
double* contactOverlap,
int* boundaryNodePointed,
int* objectPointing,
const double3* slidingSpring_old,
const int* objectPointing_old,

const int* boundaryNodeNeighborPrefixSum,
const int* boundaryNodeNeighborPrefixSum_old,

const int* localPositionID_bNode_p,
const int* particleID_bNode_p,

const double3* boundaryNodeLocalPosition_geo_p,

const double3* position_p,
const quaternion* orientation_p,

const double3* gridNodeLocalOrigin_geo_fp,
const double* gridNodeInverseSpacing_geo_fp,
const int3* gridNodeSize_geo_fp,
const int* SDFPrefixSum_geo_fp,
const double* SDF_geo_fp,

const double3* position_fp,
const quaternion* orientation_fp,
const int* geometryID_fp,
const int* hashIndex_fp,

const int* hashStart_spa_fp, 
const int* hashEnd_spa_fp,

const double3 minBound_spa_fp, 
const double3 inverseCellSize_spa_fp, 
const int3 size3D_spa_fp,

const size_t numBoundaryNode,
const size_t gridD,
const size_t blockD,
cudaStream_t stream);