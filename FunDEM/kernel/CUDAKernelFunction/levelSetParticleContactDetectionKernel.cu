#include "levelSetParticleContactDetectionKernel.cuh"
#include "neighborSearchKernel.cuh"
#include "myUtility/buildHashStartEnd.cuh"

/**
 * @brief Count the number of LS-particle contacts for each boundary node (bNode).
 *
 * Each boundary node belongs to a source particle A (idxA = particleID_bNode[idx]).
 * We search candidate neighbor particles B (idxB) via a spatial hash grid.
 * For each candidate B, we transform the boundary node into particle-B's local frame,
 * trilinearly interpolate B's level-set value (phi) at that location, and count a contact
 * if overlap = -phi > 0.
 *
 * Output:
 * - neighborCount_bNode[idx] = number of detected contacts for boundary node idx
 *
 * @param[out] neighborCount_bNode  Contact count per boundary node.
 * @param[in]  localPosition_bNode  Boundary node local position in its owner particle A frame.
 * @param[in]  particleID_bNode     Owner particle index (A) for each boundary node.

 * @param[in]  LSFV_gNode           Concatenated level-set grid node values (phi) for all particles.

 * @param[in]  position_p           Particle global positions.
 * @param[in]  orientation_p        Particle orientations (used for frame transforms).
 * @param[in]  radii_p              Bounding radius for each particle (used for early rejection).
 * @param[in]  inverseMass_p        Inverse mass per particle (used to skip static-static pairs).
 * @param[in]  gridSpacing_p        Grid spacing of each particle's level-set grid.
 * @param[in]  gridNodeLocalOrigin_p Local origin (min corner) of each particle's level-set grid in that particle's local frame.
 * @param[in]  gridNodeSize_p       Grid resolution (nx,ny,nz) of each particle's level-set grid.
 * @param[in]  gridNodePrefixSum_p  Inclusive prefix sum of grid node counts per particle (used to locate each particle's grid in LSFV_gNode).
 * @param[in]  hashIndex_p          Spatial hash sorted index list mapping cell range -> particle indices.

 * @param[in]  cellHashStart        Spatial hash cell start index in hashIndex_p (size = numCells).
 * @param[in]  cellHashEnd          Spatial hash cell end index in hashIndex_p (size = numCells).

 * @param[in]  minBound             Spatial grid global minimum bound.
 * @param[in]  cellSize             Spatial grid cell size.
 * @param[in]  gridSize             Spatial grid resolution (nx,ny,nz) in cells.
 * @param[in]  numBoundaryNode      Total number of boundary nodes.
 */
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
    bool isInvMassAZero = isZero(inverseMass_p[idxA]);
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
                    if (isInvMassAZero && isZero(inverseMass_p[idxB])) continue;
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

/**
 * @brief Build LS-boundary-node interaction pairs and initialize per-interaction state.
 *
 * For each boundary node idx:
 * - Use neighborPrefixSum_bNode to locate its output segment in the interaction arrays.
 * - Traverse candidate particles idxB via spatial hash.
 * - Compute overlap = -phi via trilinear interpolation of particle-B's level-set grid.
 * - If overlap > 0, write an interaction record:
 *     objectPointed[index_w]  = idx   (boundary node index)
 *     objectPointing[index_w] = idxB  (target particle index)
 *     contactPoint/Normal/Overlap set accordingly
 *     sliding/rolling/torsion spring initialized (and optionally restored from old pairs)
 *
 * @param[out] slidingSpring         Sliding spring state per interaction.
 * @param[out] rollingSpring         Rolling spring state per interaction.
 * @param[out] torsionSpring         Torsion spring state per interaction.
 * @param[out] contactPoint          Contact point in GLOBAL coordinates per interaction.
 * @param[out] contactNormal         Contact normal in GLOBAL coordinates per interaction.
 * @param[out] contactOverlap        Positive overlap per interaction (overlap = -phi).
 * @param[out] objectPointed         Interaction "pointed" object index (boundary node idx).
 * @param[out] objectPointing        Interaction "pointing" object index (particle idxB).
 *
 * @param[in]  slidingSpring_old     Old sliding spring state per old interaction.
 * @param[in]  rollingSpring_old     Old rolling spring state per old interaction.
 * @param[in]  torsionSpring_old     Old torsion spring state per old interaction.
 * @param[in]  objectPointed_old     Old objectPointed (boundary node index) per old interaction.
 * @param[in]  neighborPairHashIndex_old  Mapping from old adjacency slot -> packed old interaction index.
 *
 * @param[in]  localPosition_bNode   Boundary node local position in its owner particle A frame.
 * @param[in]  particleID_bNode      Owner particle index (A) for each boundary node.
 * @param[in]  neighborPrefixSum_bNode Inclusive prefix sum of neighborCount_bNode (defines output segments per bNode).
 *
 * @param[in]  LSFV_gNode            Concatenated level-set grid node values for all particles.
 *
 * @param[in]  position_p            Particle global positions.
 * @param[in]  orientation_p         Particle orientations.
 * @param[in]  radii_p               Bounding radius per particle (early rejection).
 * @param[in]  inverseMass_p         Inverse mass per particle (skip static-static).
 * @param[in]  gridSpacing_p         Grid spacing per particle.
 * @param[in]  gridNodeLocalOrigin_p Local grid origin per particle (in particle local frame).
 * @param[in]  gridNodeSize_p        Grid resolution per particle.
 * @param[in]  gridNodePrefixSum_p   Inclusive prefix sum of per-particle grid node counts.
 * @param[in]  hashIndex_p           Spatial hash sorted index list mapping cell range -> particle indices.
 *
 * @param[in]  interactionStart_p_old Old interaction start index per particle (for restoring springs).
 * @param[in]  interactionEnd_p_old   Old interaction end index per particle (for restoring springs).
 *
 * @param[in]  cellHashStart         Spatial hash cell start in hashIndex_p.
 * @param[in]  cellHashEnd           Spatial hash cell end in hashIndex_p.

 * @param[in]  minBound              Spatial grid global minimum bound.
 * @param[in]  cellSize              Spatial grid cell size.
 * @param[in]  gridSize              Spatial grid resolution in cells.
 * @param[in]  numBoundaryNode       Total number of boundary nodes.
 */
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

const int* interactionStart_p_old,
const int* interactionEnd_p_old,

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
    bool isInvMassAZero = isZero(inverseMass_p[idxA]);
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
                    if (isInvMassAZero && isZero(inverseMass_p[idxB])) continue;
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
                        if (interactionStart_p_old[idxB] != -1)
                        {
                            for (int j = interactionStart_p_old[idxB]; j < interactionEnd_p_old[idxB]; j++)
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

    interactionStart_p_old,
    interactionEnd_p_old,

    cellHashStart,
    cellHashEnd,
    minBound,
    cellSize,
    gridSize,
    numBoundaryNode);
}