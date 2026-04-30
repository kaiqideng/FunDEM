#include "LSParticleContactDetectionKernel.cuh"
#include "myUtility/mySpatialGrid.h"

__global__ void countLevelSetBoundaryNodeInteractionsKernel(int* boundaryNodeNeighborCount,
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

const size_t numBoundaryNode)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoundaryNode) return;

    int count = 0;

    const int idxA = particleID_bNode_p[idx];
    const int positionID_idx = localPositionID_bNode_p[idx];
    const double3 globalPosition_idx = rotateVectorByQuaternion(orientation_p[idxA], boundaryNodeLocalPosition_geo_p[positionID_idx]) + position_p[idxA];
    int3 gridPositionA = calculateGridPosition(globalPosition_idx, minBound_spa_p, inverseCellSize_spa_p);
    int3 gridStart = make_int3(-1, -1, -1);
    int3 gridEnd = make_int3(1, 1, 1);
    if (gridPositionA.x <= 0) { gridPositionA.x = 0; gridStart.x = 0; }
    if (gridPositionA.x >= size3D_spa_p.x - 1) { gridPositionA.x = size3D_spa_p.x - 1; gridEnd.x = 0; }
    if (gridPositionA.y <= 0) { gridPositionA.y = 0; gridStart.y = 0; }
    if (gridPositionA.y >= size3D_spa_p.y - 1) { gridPositionA.y = size3D_spa_p.y - 1; gridEnd.y = 0; }
    if (gridPositionA.z <= 0) { gridPositionA.z = 0; gridStart.z = 0; }
    if (gridPositionA.z >= size3D_spa_p.z - 1) { gridPositionA.z = size3D_spa_p.z - 1; gridEnd.z = 0; }
    for (int zz = gridStart.z; zz <= gridEnd.z; zz++)
    {
        for (int yy = gridStart.y; yy <= gridEnd.y; yy++)
        {
            for (int xx = gridStart.x; xx <= gridEnd.x; xx++)
            {
                const int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                const int hashB = linearIndex3D(gridPositionB, size3D_spa_p);
                const int startIndex = hashStart_spa_p[hashB];
                if (startIndex == -1) continue;
                const int endIndex = hashEnd_spa_p[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    const int idxB = hashIndex_p[i];
                    if (idxA >= idxB) continue;
                    const int geoIDB = geometryID_p[idxB];
                    const double3 relativePosition = globalPosition_idx - position_p[idxB];
                    const double radB = radius_geo_p[geoIDB];
                    if (lengthSquared(relativePosition) > radB * radB) continue;
                    const quaternion orientationB = orientation_p[idxB];
                    const double3 localPosition_idx = reverseRotateVectorByQuaternion(relativePosition, orientationB);
                    const double3 gridNodeLocalOriginB = gridNodeLocalOrigin_geo_p[geoIDB];
                    const double gridNodeInverseSpacingB = gridNodeInverseSpacing_geo_p[geoIDB];
                    const int3 gridNodeSizeB = gridNodeSize_geo_p[geoIDB];

                    const double gx = (localPosition_idx.x - gridNodeLocalOriginB.x) * gridNodeInverseSpacingB;
                    const double gy = (localPosition_idx.y - gridNodeLocalOriginB.y) * gridNodeInverseSpacingB;
                    const double gz = (localPosition_idx.z - gridNodeLocalOriginB.z) * gridNodeInverseSpacingB;

                    int i0 = (int)floor(gx);
                    int j0 = (int)floor(gy);
                    int k0 = (int)floor(gz);

                    if (i0 < 0) continue;
                    if (j0 < 0) continue;
                    if (k0 < 0) continue;

                    if (i0 >= gridNodeSizeB.x - 1) continue;
                    if (j0 >= gridNodeSizeB.y - 1) continue;
                    if (k0 >= gridNodeSizeB.z - 1) continue;

                    const int i1 = i0 + 1;
                    const int j1 = j0 + 1;
                    const int k1 = k0 + 1;

                    const double x = gx - static_cast<double>(i0);
                    const double y = gy - static_cast<double>(j0);
                    const double z = gz - static_cast<double>(k0);

                    int SDFStartB = 0;
                    if (geoIDB > 0) SDFStartB = SDFPrefixSum_geo_p[geoIDB - 1];
                    const double phi000 = SDF_geo_p[SDFStartB + linearIndex3D(make_int3(i0, j0, k0), gridNodeSizeB)];
                    const double phi100 = SDF_geo_p[SDFStartB + linearIndex3D(make_int3(i1, j0, k0), gridNodeSizeB)];
                    const double phi010 = SDF_geo_p[SDFStartB + linearIndex3D(make_int3(i0, j1, k0), gridNodeSizeB)];
                    const double phi110 = SDF_geo_p[SDFStartB + linearIndex3D(make_int3(i1, j1, k0), gridNodeSizeB)];
                    const double phi001 = SDF_geo_p[SDFStartB + linearIndex3D(make_int3(i0, j0, k1), gridNodeSizeB)];
                    const double phi101 = SDF_geo_p[SDFStartB + linearIndex3D(make_int3(i1, j0, k1), gridNodeSizeB)];
                    const double phi011 = SDF_geo_p[SDFStartB + linearIndex3D(make_int3(i0, j1, k1), gridNodeSizeB)];
                    const double phi111 = SDF_geo_p[SDFStartB + linearIndex3D(make_int3(i1, j1, k1), gridNodeSizeB)];

                    const double ovelap = -interpolateLevelSetFunctionValue(x, 
                    y, 
                    z, 
                    phi000,
                    phi100,
                    phi010,
                    phi110,
                    phi001,
                    phi101, 
                    phi011,
                    phi111);

                    if (ovelap >= 0.) count++;
                }
            }
        }
    }

    boundaryNodeNeighborCount[idx] = count;
}

__global__ void writeLevelSetBoundaryNodeInteractionsKernel(double3* slidingSpring,
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

const size_t numBoundaryNode)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoundaryNode) return;

    int base_w = 0, base_w_old = 0;
    if (idx > 0)
    {
        base_w = boundaryNodeNeighborPrefixSum[idx - 1];
        base_w_old = boundaryNodeNeighborPrefixSum_old[idx - 1];
    }
    const int end_w = boundaryNodeNeighborPrefixSum[idx];
    if (end_w - base_w == 0) return;
    int end_old = boundaryNodeNeighborPrefixSum_old[idx];

    int count = 0;

    const int idxA = particleID_bNode_p[idx];
    const int positionID_idx = localPositionID_bNode_p[idx];
    const double3 globalPosition_idx = rotateVectorByQuaternion(orientation_p[idxA], boundaryNodeLocalPosition_geo_p[positionID_idx]) + position_p[idxA];
    int3 gridPositionA = calculateGridPosition(globalPosition_idx, minBound_spa_p, inverseCellSize_spa_p);
    int3 gridStart = make_int3(-1, -1, -1);
    int3 gridEnd = make_int3(1, 1, 1);
    if (gridPositionA.x <= 0) { gridPositionA.x = 0; gridStart.x = 0; }
    if (gridPositionA.x >= size3D_spa_p.x - 1) { gridPositionA.x = size3D_spa_p.x - 1; gridEnd.x = 0; }
    if (gridPositionA.y <= 0) { gridPositionA.y = 0; gridStart.y = 0; }
    if (gridPositionA.y >= size3D_spa_p.y - 1) { gridPositionA.y = size3D_spa_p.y - 1; gridEnd.y = 0; }
    if (gridPositionA.z <= 0) { gridPositionA.z = 0; gridStart.z = 0; }
    if (gridPositionA.z >= size3D_spa_p.z - 1) { gridPositionA.z = size3D_spa_p.z - 1; gridEnd.z = 0; }
    for (int zz = gridStart.z; zz <= gridEnd.z; zz++)
    {
        for (int yy = gridStart.y; yy <= gridEnd.y; yy++)
        {
            for (int xx = gridStart.x; xx <= gridEnd.x; xx++)
            {
                const int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                const int hashB = linearIndex3D(gridPositionB, size3D_spa_p);
                const int startIndex = hashStart_spa_p[hashB];
                if (startIndex == -1) continue;
                const int endIndex = hashEnd_spa_p[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    const int idxB = hashIndex_p[i];
                    if (idxA >= idxB) continue; 
                    const int geoIDB = geometryID_p[idxB];
                    const double3 relativePosition = globalPosition_idx - position_p[idxB];
                    const double radB = radius_geo_p[geoIDB];
                    if (lengthSquared(relativePosition) > radB * radB) continue;
                    const quaternion orientationB = orientation_p[idxB];
                    const double3 localPosition_idx = reverseRotateVectorByQuaternion(relativePosition, orientationB);
                   
                    const double3 gridNodeLocalOriginB = gridNodeLocalOrigin_geo_p[geoIDB];
                    const double gridNodeInverseSpacingB = gridNodeInverseSpacing_geo_p[geoIDB];
                    const int3 gridNodeSizeB = gridNodeSize_geo_p[geoIDB];

                    const double gx = (localPosition_idx.x - gridNodeLocalOriginB.x) * gridNodeInverseSpacingB;
                    const double gy = (localPosition_idx.y - gridNodeLocalOriginB.y) * gridNodeInverseSpacingB;
                    const double gz = (localPosition_idx.z - gridNodeLocalOriginB.z) * gridNodeInverseSpacingB;

                    int i0 = (int)floor(gx);
                    int j0 = (int)floor(gy);
                    int k0 = (int)floor(gz);

                    if (i0 < 0) continue;
                    if (j0 < 0) continue;
                    if (k0 < 0) continue;

                    if (i0 >= gridNodeSizeB.x - 1) continue;
                    if (j0 >= gridNodeSizeB.y - 1) continue;
                    if (k0 >= gridNodeSizeB.z - 1) continue;

                    const int i1 = i0 + 1;
                    const int j1 = j0 + 1;
                    const int k1 = k0 + 1;

                    const double x = gx - static_cast<double>(i0);
                    const double y = gy - static_cast<double>(j0);
                    const double z = gz - static_cast<double>(k0);

                    int SDFStartB = 0;
                    if (geoIDB > 0) SDFStartB = SDFPrefixSum_geo_p[geoIDB - 1];
                    const double phi000 = SDF_geo_p[SDFStartB + linearIndex3D(make_int3(i0, j0, k0), gridNodeSizeB)];
                    const double phi100 = SDF_geo_p[SDFStartB + linearIndex3D(make_int3(i1, j0, k0), gridNodeSizeB)];
                    const double phi010 = SDF_geo_p[SDFStartB + linearIndex3D(make_int3(i0, j1, k0), gridNodeSizeB)];
                    const double phi110 = SDF_geo_p[SDFStartB + linearIndex3D(make_int3(i1, j1, k0), gridNodeSizeB)];
                    const double phi001 = SDF_geo_p[SDFStartB + linearIndex3D(make_int3(i0, j0, k1), gridNodeSizeB)];
                    const double phi101 = SDF_geo_p[SDFStartB + linearIndex3D(make_int3(i1, j0, k1), gridNodeSizeB)];
                    const double phi011 = SDF_geo_p[SDFStartB + linearIndex3D(make_int3(i0, j1, k1), gridNodeSizeB)];
                    const double phi111 = SDF_geo_p[SDFStartB + linearIndex3D(make_int3(i1, j1, k1), gridNodeSizeB)];

                    const double ovelap = -interpolateLevelSetFunctionValue(x,
                    y,
                    z,
                    phi000,
                    phi100,
                    phi010,
                    phi110,
                    phi001,
                    phi101,
                    phi011,
                    phi111);

                    if (ovelap >= 0.)
                    {
                        const int index_w = base_w + count;
                        double3 n_c = interpolateLevelSetFunctionGradient(x,
                        y,
                        z,
                        phi000,
                        phi100,
                        phi010,
                        phi110,
                        phi001,
                        phi101,
                        phi011,
                        phi111);

                        n_c = rotateVectorByQuaternion(orientationB, n_c);
                        n_c = normalize(n_c);
                        contactPoint[index_w] = globalPosition_idx + 0.5 * ovelap * n_c;
                        contactNormal[index_w] = n_c;
                        contactOverlap[index_w] = ovelap;
                        boundaryNodePointed[index_w] = idx;
                        objectPointing[index_w] = idxB;

                        double3 ss = make_double3(0., 0., 0.);
                        for (int j = base_w_old; j < end_old; j++)
                        {
                            if (idxB == objectPointing_old[j])
                            {
                                ss = slidingSpring_old[j];
                                break;
                            }
                        }
                        slidingSpring[index_w] = ss;

                        count++;
                    }
                }
            }
        }
    }
}

__global__ void countLevelSetBoundaryNodeFixedParticleInteractionsKernel(int* boundaryNodeNeighborCount,
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

const size_t numBoundaryNode)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoundaryNode) return;

    int count = 0;

    const int idxA = particleID_bNode_p[idx];
    const int positionID_idx = localPositionID_bNode_p[idx];
    const double3 globalPosition_idx = rotateVectorByQuaternion(orientation_p[idxA], boundaryNodeLocalPosition_geo_p[positionID_idx]) + position_p[idxA];
    int3 gridPositionA = calculateGridPosition(globalPosition_idx, minBound_spa_fp, inverseCellSize_spa_fp);
    int3 gridStart = make_int3(-1, -1, -1);
    int3 gridEnd = make_int3(1, 1, 1);
    if (gridPositionA.x <= 0) { gridPositionA.x = 0; gridStart.x = 0; }
    if (gridPositionA.x >= size3D_spa_fp.x - 1) { gridPositionA.x = size3D_spa_fp.x - 1; gridEnd.x = 0; }
    if (gridPositionA.y <= 0) { gridPositionA.y = 0; gridStart.y = 0; }
    if (gridPositionA.y >= size3D_spa_fp.y - 1) { gridPositionA.y = size3D_spa_fp.y - 1; gridEnd.y = 0; }
    if (gridPositionA.z <= 0) { gridPositionA.z = 0; gridStart.z = 0; }
    if (gridPositionA.z >= size3D_spa_fp.z - 1) { gridPositionA.z = size3D_spa_fp.z - 1; gridEnd.z = 0; }
    for (int zz = gridStart.z; zz <= gridEnd.z; zz++)
    {
        for (int yy = gridStart.y; yy <= gridEnd.y; yy++)
        {
            for (int xx = gridStart.x; xx <= gridEnd.x; xx++)
            {
                const int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                const int hashB = linearIndex3D(gridPositionB, size3D_spa_fp);
                const int startIndex = hashStart_spa_fp[hashB];
                if (startIndex == -1) continue;
                const int endIndex = hashEnd_spa_fp[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    const int idxB = hashIndex_fp[i];
                    const double3 relativePosition = globalPosition_idx - position_fp[idxB];
                    const quaternion orientationB = orientation_fp[idxB];
                    const double3 localPosition_idx = reverseRotateVectorByQuaternion(relativePosition, orientationB);
                    const int geoIDB = geometryID_fp[idxB];
                    const double3 gridNodeLocalOriginB = gridNodeLocalOrigin_geo_fp[geoIDB];
                    const double gridNodeInverseSpacingB = gridNodeInverseSpacing_geo_fp[geoIDB];
                    const int3 gridNodeSizeB = gridNodeSize_geo_fp[geoIDB];

                    const double gx = (localPosition_idx.x - gridNodeLocalOriginB.x) * gridNodeInverseSpacingB;
                    const double gy = (localPosition_idx.y - gridNodeLocalOriginB.y) * gridNodeInverseSpacingB;
                    const double gz = (localPosition_idx.z - gridNodeLocalOriginB.z) * gridNodeInverseSpacingB;

                    int i0 = (int)floor(gx);
                    int j0 = (int)floor(gy);
                    int k0 = (int)floor(gz);

                    if (i0 < 0) continue;
                    if (j0 < 0) continue;
                    if (k0 < 0) continue;

                    if (i0 >= gridNodeSizeB.x - 1) continue;
                    if (j0 >= gridNodeSizeB.y - 1) continue;
                    if (k0 >= gridNodeSizeB.z - 1) continue;

                    const int i1 = i0 + 1;
                    const int j1 = j0 + 1;
                    const int k1 = k0 + 1;

                    const double x = gx - static_cast<double>(i0);
                    const double y = gy - static_cast<double>(j0);
                    const double z = gz - static_cast<double>(k0);

                    int SDFStartB = 0;
                    if (geoIDB > 0) SDFStartB = SDFPrefixSum_geo_fp[geoIDB - 1];
                    const double phi000 = SDF_geo_fp[SDFStartB + linearIndex3D(make_int3(i0, j0, k0), gridNodeSizeB)];
                    const double phi100 = SDF_geo_fp[SDFStartB + linearIndex3D(make_int3(i1, j0, k0), gridNodeSizeB)];
                    const double phi010 = SDF_geo_fp[SDFStartB + linearIndex3D(make_int3(i0, j1, k0), gridNodeSizeB)];
                    const double phi110 = SDF_geo_fp[SDFStartB + linearIndex3D(make_int3(i1, j1, k0), gridNodeSizeB)];
                    const double phi001 = SDF_geo_fp[SDFStartB + linearIndex3D(make_int3(i0, j0, k1), gridNodeSizeB)];
                    const double phi101 = SDF_geo_fp[SDFStartB + linearIndex3D(make_int3(i1, j0, k1), gridNodeSizeB)];
                    const double phi011 = SDF_geo_fp[SDFStartB + linearIndex3D(make_int3(i0, j1, k1), gridNodeSizeB)];
                    const double phi111 = SDF_geo_fp[SDFStartB + linearIndex3D(make_int3(i1, j1, k1), gridNodeSizeB)];

                    const double ovelap = -interpolateLevelSetFunctionValue(x, 
                    y, 
                    z, 
                    phi000,
                    phi100,
                    phi010,
                    phi110,
                    phi001,
                    phi101, 
                    phi011,
                    phi111);

                    if (ovelap >= 0.) count++;
                }
            }
        }
    }

    boundaryNodeNeighborCount[idx] = count;
}

__global__ void writeLevelSetBoundaryNodeFixedParticleInteractionsKernel(double3* slidingSpring,
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

const size_t numBoundaryNode)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoundaryNode) return;

    int base_w = 0, base_w_old = 0;
    if (idx > 0)
    {
        base_w = boundaryNodeNeighborPrefixSum[idx - 1];
        base_w_old = boundaryNodeNeighborPrefixSum_old[idx - 1];
    }
    const int end_w = boundaryNodeNeighborPrefixSum[idx];
    if (end_w - base_w == 0) return;
    int end_old = boundaryNodeNeighborPrefixSum_old[idx];

    int count = 0;

    const int idxA = particleID_bNode_p[idx];
    const int positionID_idx = localPositionID_bNode_p[idx];
    const double3 globalPosition_idx = rotateVectorByQuaternion(orientation_p[idxA], boundaryNodeLocalPosition_geo_p[positionID_idx]) + position_p[idxA];
    int3 gridPositionA = calculateGridPosition(globalPosition_idx, minBound_spa_fp, inverseCellSize_spa_fp);
    int3 gridStart = make_int3(-1, -1, -1);
    int3 gridEnd = make_int3(1, 1, 1);
    if (gridPositionA.x <= 0) { gridPositionA.x = 0; gridStart.x = 0; }
    if (gridPositionA.x >= size3D_spa_fp.x - 1) { gridPositionA.x = size3D_spa_fp.x - 1; gridEnd.x = 0; }
    if (gridPositionA.y <= 0) { gridPositionA.y = 0; gridStart.y = 0; }
    if (gridPositionA.y >= size3D_spa_fp.y - 1) { gridPositionA.y = size3D_spa_fp.y - 1; gridEnd.y = 0; }
    if (gridPositionA.z <= 0) { gridPositionA.z = 0; gridStart.z = 0; }
    if (gridPositionA.z >= size3D_spa_fp.z - 1) { gridPositionA.z = size3D_spa_fp.z - 1; gridEnd.z = 0; }
    for (int zz = gridStart.z; zz <= gridEnd.z; zz++)
    {
        for (int yy = gridStart.y; yy <= gridEnd.y; yy++)
        {
            for (int xx = gridStart.x; xx <= gridEnd.x; xx++)
            {
                const int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                const int hashB = linearIndex3D(gridPositionB, size3D_spa_fp);
                const int startIndex = hashStart_spa_fp[hashB];
                if (startIndex == -1) continue;
                const int endIndex = hashEnd_spa_fp[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    const int idxB = hashIndex_fp[i];
                    const double3 relativePosition = globalPosition_idx - position_fp[idxB];
                    const quaternion orientationB = orientation_fp[idxB];
                    const double3 localPosition_idx = reverseRotateVectorByQuaternion(relativePosition, orientationB);
                    const int geoIDB = geometryID_fp[idxB];
                    const double3 gridNodeLocalOriginB = gridNodeLocalOrigin_geo_fp[geoIDB];
                    const double gridNodeInverseSpacingB = gridNodeInverseSpacing_geo_fp[geoIDB];
                    const int3 gridNodeSizeB = gridNodeSize_geo_fp[geoIDB];

                    const double gx = (localPosition_idx.x - gridNodeLocalOriginB.x) * gridNodeInverseSpacingB;
                    const double gy = (localPosition_idx.y - gridNodeLocalOriginB.y) * gridNodeInverseSpacingB;
                    const double gz = (localPosition_idx.z - gridNodeLocalOriginB.z) * gridNodeInverseSpacingB;

                    int i0 = (int)floor(gx);
                    int j0 = (int)floor(gy);
                    int k0 = (int)floor(gz);

                    if (i0 < 0) continue;
                    if (j0 < 0) continue;
                    if (k0 < 0) continue;

                    if (i0 >= gridNodeSizeB.x - 1) continue;
                    if (j0 >= gridNodeSizeB.y - 1) continue;
                    if (k0 >= gridNodeSizeB.z - 1) continue;

                    const int i1 = i0 + 1;
                    const int j1 = j0 + 1;
                    const int k1 = k0 + 1;

                    const double x = gx - static_cast<double>(i0);
                    const double y = gy - static_cast<double>(j0);
                    const double z = gz - static_cast<double>(k0);

                    int SDFStartB = 0;
                    if (geoIDB > 0) SDFStartB = SDFPrefixSum_geo_fp[geoIDB - 1];
                    const double phi000 = SDF_geo_fp[SDFStartB + linearIndex3D(make_int3(i0, j0, k0), gridNodeSizeB)];
                    const double phi100 = SDF_geo_fp[SDFStartB + linearIndex3D(make_int3(i1, j0, k0), gridNodeSizeB)];
                    const double phi010 = SDF_geo_fp[SDFStartB + linearIndex3D(make_int3(i0, j1, k0), gridNodeSizeB)];
                    const double phi110 = SDF_geo_fp[SDFStartB + linearIndex3D(make_int3(i1, j1, k0), gridNodeSizeB)];
                    const double phi001 = SDF_geo_fp[SDFStartB + linearIndex3D(make_int3(i0, j0, k1), gridNodeSizeB)];
                    const double phi101 = SDF_geo_fp[SDFStartB + linearIndex3D(make_int3(i1, j0, k1), gridNodeSizeB)];
                    const double phi011 = SDF_geo_fp[SDFStartB + linearIndex3D(make_int3(i0, j1, k1), gridNodeSizeB)];
                    const double phi111 = SDF_geo_fp[SDFStartB + linearIndex3D(make_int3(i1, j1, k1), gridNodeSizeB)];

                    const double ovelap = -interpolateLevelSetFunctionValue(x,
                    y,
                    z,
                    phi000,
                    phi100,
                    phi010,
                    phi110,
                    phi001,
                    phi101,
                    phi011,
                    phi111);

                    if (ovelap >= 0.)
                    {
                        const int index_w = base_w + count;
                        double3 n_c = interpolateLevelSetFunctionGradient(x,
                        y,
                        z,
                        phi000,
                        phi100,
                        phi010,
                        phi110,
                        phi001,
                        phi101,
                        phi011,
                        phi111);

                        n_c = rotateVectorByQuaternion(orientationB, n_c);
                        n_c = normalize(n_c);
                        contactPoint[index_w] = globalPosition_idx + 0.5 * ovelap * n_c;
                        contactNormal[index_w] = n_c;
                        contactOverlap[index_w] = ovelap;
                        boundaryNodePointed[index_w] = idx;
                        objectPointing[index_w] = idxB;

                        double3 ss = make_double3(0., 0., 0.);
                        for (int j = base_w_old; j < end_old; j++)
                        {
                            if (idxB == objectPointing_old[j])
                            {
                                ss = slidingSpring_old[j];
                                break;
                            }
                        }
                        slidingSpring[index_w] = ss;

                        count++;
                    }
                }
            }
        }
    }
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
cudaStream_t stream)
{
    countLevelSetBoundaryNodeInteractionsKernel<<<gridD, blockD, 0, stream>>>(boundaryNodeNeighborCount,
    localPositionID_bNode_p,
    particleID_bNode_p,
    radius_geo_p,
    gridNodeLocalOrigin_geo_p,
    gridNodeInverseSpacing_geo_p,
    gridNodeSize_geo_p,
    SDFPrefixSum_geo_p,
    SDF_geo_p,
    boundaryNodeLocalPosition_geo_p,
    position_p,
    orientation_p,
    geometryID_p,
    hashIndex_p,
    hashStart_spa_p,
    hashEnd_spa_p,
    minBound_spa_p,
    inverseCellSize_spa_p,
    size3D_spa_p,
    numBoundaryNode);
}

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
cudaStream_t stream)
{
    writeLevelSetBoundaryNodeInteractionsKernel<<<gridD, blockD, 0, stream>>>(slidingSpring,
    contactPoint,
    contactNormal,
    contactOverlap,
    boundaryNodePointed,
    objectPointing,
    slidingSpring_old,
    objectPointing_old,
    boundaryNodeNeighborPrefixSum,
    boundaryNodeNeighborPrefixSum_old,
    localPositionID_bNode_p,
    particleID_bNode_p,
    radius_geo_p,
    gridNodeLocalOrigin_geo_p,
    gridNodeInverseSpacing_geo_p,
    gridNodeSize_geo_p,
    SDFPrefixSum_geo_p,
    SDF_geo_p,
    boundaryNodeLocalPosition_geo_p,
    position_p,
    orientation_p,
    geometryID_p,
    hashIndex_p,
    hashStart_spa_p,
    hashEnd_spa_p,
    minBound_spa_p,
    inverseCellSize_spa_p,
    size3D_spa_p,
    numBoundaryNode);
}

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
cudaStream_t stream)
{
    countLevelSetBoundaryNodeFixedParticleInteractionsKernel<<<gridD, blockD, 0, stream>>>(boundaryNodeNeighborCount,
    localPositionID_bNode_p,
    particleID_bNode_p,
    boundaryNodeLocalPosition_geo_p,
    position_p,
    orientation_p,
    gridNodeLocalOrigin_geo_fp,
    gridNodeInverseSpacing_geo_fp,
    gridNodeSize_geo_fp,
    SDFPrefixSum_geo_fp,
    SDF_geo_fp,
    position_fp,
    orientation_fp,
    geometryID_fp,
    hashIndex_fp,
    hashStart_spa_fp,
    hashEnd_spa_fp,
    minBound_spa_fp,
    inverseCellSize_spa_fp,
    size3D_spa_fp,
    numBoundaryNode);
}

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
cudaStream_t stream)
{
    writeLevelSetBoundaryNodeFixedParticleInteractionsKernel<<<gridD, blockD, 0, stream>>>(slidingSpring,
    contactPoint,
    contactNormal,
    contactOverlap,
    boundaryNodePointed,
    objectPointing,
    slidingSpring_old,
    objectPointing_old,
    boundaryNodeNeighborPrefixSum,
    boundaryNodeNeighborPrefixSum_old,
    localPositionID_bNode_p,
    particleID_bNode_p,
    boundaryNodeLocalPosition_geo_p,
    position_p,
    orientation_p,
    gridNodeLocalOrigin_geo_fp,
    gridNodeInverseSpacing_geo_fp,
    gridNodeSize_geo_fp,
    SDFPrefixSum_geo_fp,
    SDF_geo_fp,
    position_fp,
    orientation_fp,
    geometryID_fp,
    hashIndex_fp,
    hashStart_spa_fp,
    hashEnd_spa_fp,
    minBound_spa_fp,
    inverseCellSize_spa_fp,
    size3D_spa_fp,
    numBoundaryNode);
}