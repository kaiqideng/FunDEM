#include "levelSetParticleContactDetectionKernel.cuh"
#include "neighborSearchKernel.cuh"
#include "buildHashStartEnd.cuh"

__global__ void countLevelSetBoundaryNodeInteractionsKernel(int* neighborCount_bNode,
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
const size_t numBoundaryNode)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoundaryNode) return;

    int count = 0;

    const int idxA = particleID_bNode[idx];
    const double3 globalPosition_idx = rotateVectorByQuaternion(orientation_p[idxA], localPosition_bNode[idx]) + position_p[idxA];
    int3 gridPositionA = calculateGridPosition(globalPosition_idx, minBound, cellSize);
    int3 gridStart = make_int3(-1, -1, -1);
    int3 gridEnd = make_int3(1, 1, 1);
    if (gridPositionA.x <= 0) { gridPositionA.x = 0; gridStart.x = 0; }
    if (gridPositionA.x >= gridSize.x - 1) { gridPositionA.x = gridSize.x - 1; gridEnd.x = 0; }
    if (gridPositionA.y <= 0) { gridPositionA.y = 0; gridStart.y = 0; }
    if (gridPositionA.y >= gridSize.y - 1) { gridPositionA.y = gridSize.y - 1; gridEnd.y = 0; }
    if (gridPositionA.z <= 0) { gridPositionA.z = 0; gridStart.z = 0; }
    if (gridPositionA.z >= gridSize.z - 1) { gridPositionA.z = gridSize.z - 1; gridEnd.z = 0; }
    for (int zz = gridStart.z; zz <= gridEnd.z; zz++)
    {
        for (int yy = gridStart.y; yy <= gridEnd.y; yy++)
        {
            for (int xx = gridStart.x; xx <= gridEnd.x; xx++)
            {
                const int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                const int hashB = linearIndex3D(gridPositionB, gridSize);
                const int startIndex = cellHashStart[hashB];
                if (startIndex == -1) continue;
                const int endIndex = cellHashEnd[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    const int idxB = hashIndex_p[i];
                    if (idxA >= idxB) continue;
                    if (inverseMass_p[idxA] == 0. && inverseMass_p[idxB] == 0.) continue;
                    const double3 r_idxidxB = globalPosition_idx - position_p[idxB];
                    if (lengthSquared(r_idxidxB) > radii_p[idxB] * radii_p[idxB]) continue;

                    const double3 localPositionFrameB_idx = reverseRotateVectorByQuaternion(r_idxidxB, orientation_p[idxB]);
                    const double3 gridNodeLocalOriginB = gridNodeLocalOrigin_p[idxB];
                    const double g = gridSpacing_p[idxB]; // g > 0.
                    const int3 gridNodeSizeB = gridNodeSize_p[idxB];

                    const double gx = (localPositionFrameB_idx.x - gridNodeLocalOriginB.x) / g;
                    const double gy = (localPositionFrameB_idx.y - gridNodeLocalOriginB.y) / g;
                    const double gz = (localPositionFrameB_idx.z - gridNodeLocalOriginB.z) / g;

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

                    int gridNodeStartB = 0;
                    if (idxB > 0) gridNodeStartB = gridNodePrefixSum_p[idxB - 1];
                    const double phi000 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i0, j0, k0), gridNodeSizeB)];
                    const double phi100 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i1, j0, k0), gridNodeSizeB)];
                    const double phi010 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i0, j1, k0), gridNodeSizeB)];
                    const double phi110 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i1, j1, k0), gridNodeSizeB)];
                    const double phi001 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i0, j0, k1), gridNodeSizeB)];
                    const double phi101 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i1, j0, k1), gridNodeSizeB)];
                    const double phi011 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i0, j1, k1), gridNodeSizeB)];
                    const double phi111 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i1, j1, k1), gridNodeSizeB)];

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

                    if (ovelap > 0.) count++;
                }
            }
        }
    }

    neighborCount_bNode[idx] = count;
}

__global__ void writeLevelSetBoundaryNodeInteractionsKernel(double3* slidingSpring,
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
const size_t numBoundaryNode)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoundaryNode) return;

    int count = 0;
    int base_w = 0;
    if (idx > 0) base_w = neighborPrefixSum_bNode[idx - 1];
    if (neighborPrefixSum_bNode[idx] - base_w == 0) return;

    const int idxA = particleID_bNode[idx];
    const double3 globalPosition_idx = rotateVectorByQuaternion(orientation_p[idxA], localPosition_bNode[idx]) + position_p[idxA];
    int3 gridPositionA = calculateGridPosition(globalPosition_idx, minBound, cellSize);
    int3 gridStart = make_int3(-1, -1, -1);
    int3 gridEnd = make_int3(1, 1, 1);
    if (gridPositionA.x <= 0) { gridPositionA.x = 0; gridStart.x = 0; }
    if (gridPositionA.x >= gridSize.x - 1) { gridPositionA.x = gridSize.x - 1; gridEnd.x = 0; }
    if (gridPositionA.y <= 0) { gridPositionA.y = 0; gridStart.y = 0; }
    if (gridPositionA.y >= gridSize.y - 1) { gridPositionA.y = gridSize.y - 1; gridEnd.y = 0; }
    if (gridPositionA.z <= 0) { gridPositionA.z = 0; gridStart.z = 0; }
    if (gridPositionA.z >= gridSize.z - 1) { gridPositionA.z = gridSize.z - 1; gridEnd.z = 0; }
    for (int zz = gridStart.z; zz <= gridEnd.z; zz++)
    {
        for (int yy = gridStart.y; yy <= gridEnd.y; yy++)
        {
            for (int xx = gridStart.x; xx <= gridEnd.x; xx++)
            {
                const int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                const int hashB = linearIndex3D(gridPositionB, gridSize);
                const int startIndex = cellHashStart[hashB];
                if (startIndex == -1) continue;
                const int endIndex = cellHashEnd[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    const int idxB = hashIndex_p[i];
                    if (idxA >= idxB) continue;
                    if (inverseMass_p[idxA] == 0. && inverseMass_p[idxB] == 0.) continue;
                    const double3 r_idxidxB = globalPosition_idx - position_p[idxB];
                    if (lengthSquared(r_idxidxB) > radii_p[idxB] * radii_p[idxB]) continue;

                    const double3 localPositionFrameB_idx = reverseRotateVectorByQuaternion(r_idxidxB, orientation_p[idxB]);
                    const double3 gridNodeLocalOriginB = gridNodeLocalOrigin_p[idxB];
                    const double g = gridSpacing_p[idxB]; // g > 0.
                    const int3 gridNodeSizeB = gridNodeSize_p[idxB];

                    const double gx = (localPositionFrameB_idx.x - gridNodeLocalOriginB.x) / g;
                    const double gy = (localPositionFrameB_idx.y - gridNodeLocalOriginB.y) / g;
                    const double gz = (localPositionFrameB_idx.z - gridNodeLocalOriginB.z) / g;

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

                    int gridNodeStartB = 0;
                    if (idxB > 0) gridNodeStartB = gridNodePrefixSum_p[idxB - 1];
                    const double phi000 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i0, j0, k0), gridNodeSizeB)];
                    const double phi100 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i1, j0, k0), gridNodeSizeB)];
                    const double phi010 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i0, j1, k0), gridNodeSizeB)];
                    const double phi110 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i1, j1, k0), gridNodeSizeB)];
                    const double phi001 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i0, j0, k1), gridNodeSizeB)];
                    const double phi101 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i1, j0, k1), gridNodeSizeB)];
                    const double phi011 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i0, j1, k1), gridNodeSizeB)];
                    const double phi111 = LSFV_gNode[gridNodeStartB + linearIndex3D(make_int3(i1, j1, k1), gridNodeSizeB)];
                    
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

                    if (ovelap > 0.)
                    {
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
                        n_c = rotateVectorByQuaternion(orientation_p[idxB], n_c); // boundary node is pointed
                        n_c = normalize(n_c);
                        double3 p_c = globalPosition_idx + 0.5 * ovelap * n_c;
                        const int index_w = base_w + count;
                        contactPoint[index_w] = p_c;
                        contactNormal[index_w] = n_c;
                        contactOverlap[index_w] = ovelap;
                        objectPointed[index_w] = idx; // node
                        objectPointing[index_w] = idxB; // particle
                        slidingSpring[index_w] = make_double3(0., 0., 0.);
                        rollingSpring[index_w] = make_double3(0., 0., 0.);
                        torsionSpring[index_w] = make_double3(0., 0., 0.);
                        if (interactionStart_old_p[idxB] != -1)
                        {
                            for (int j = interactionStart_old_p[idxB]; j < interactionEnd_old_p[idxB]; j++)
                            {
                                const int j1 = neighborPairHashIndex_old[j];
                                const int idx1 = objectPointed_old[j1];
                                if (idx == idx1)
                                {
                                    slidingSpring[index_w] = slidingSpring_old[j1];
                                    rollingSpring[index_w] = rollingSpring_old[j1];
                                    torsionSpring[index_w] = torsionSpring_old[j1];
                                    break;
                                }
                            }
                        }
                        count++;
                    }
                }
            }
        }
    }
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
cudaStream_t stream)
{
    countLevelSetBoundaryNodeInteractionsKernel<<<gridD, blockD, 0, stream>>>(neighborCount_bNode,
    localPosition_bNode,
    particleID_bNode,

    LSFV_gNode,

    position_p,
    orientation_p,
    radii_p,
    inverseMass_p,
    gridSpacing_p,
    gridNodeLocalOrigin_p,
    gridNodeSize_p,
    gridNodePrefixSum_p,
    hashIndex_p,

    cellHashStart,
    cellHashEnd,
    minBound,
    cellSize,
    gridSize,
    numBoundaryNode);

    buildPrefixSum(neighborPrefixSum_bNode,
    neighborCount_bNode,
    numBoundaryNode,
    stream);
}

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
cudaStream_t stream)
{
    writeLevelSetBoundaryNodeInteractionsKernel<<<gridD, blockD, 0, stream>>>(slidingSpring,
    rollingSpring,
    torsionSpring,
    contactPoint,
    contactNormal,
    contactOverlap,
    objectPointed,
    objectPointing,
    slidingSpring_old,
    rollingSpring_old,
    torsionSpring_old,
    objectPointed_old,
    neighborPairHashIndex_old,

    localPosition_bNode,
    particleID_bNode,

    neighborPrefixSum_bNode,

    LSFV_gNode,

    position_p,
    orientation_p,
    radii_p,
    inverseMass_p,
    gridSpacing_p,
    gridNodeLocalOrigin_p,
    gridNodeSize_p,
    gridNodePrefixSum_p,
    hashIndex_p,

    interactionStart_old_p,
    interactionEnd_old_p,

    cellHashStart,
    cellHashEnd,
    minBound,
    cellSize,
    gridSize,
    numBoundaryNode);
}