#include "ballNeighborSearchKernel.cuh"
#include "neighborSearchKernel.cuh"
#include "myUtility/myVec.h"
#include "myUtility/buildHashStartEnd.cuh"

__global__ void countBallInteractionsKernel(int* neighborCount,
const double3* position, 
const double* radius,
const double* inverseMass,
const int* clumpID, 
const int* hashIndex,
const int* cellHashStart, 
const int* cellHashEnd,
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numBall)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numBall) return;
    int count = 0;

    double3 posA = position[idxA];
    double radA = radius[idxA];
    int3 gridPositionA = calculateGridPosition(posA, minBound, cellSize);
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
                int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                int hashB = linearIndex3D(gridPositionB, gridSize);
                int startIndex = cellHashStart[hashB];
                if (startIndex == -1) continue;
                int endIndex = cellHashEnd[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = hashIndex[i];
                    if (idxA >= idxB) continue;
                    if (inverseMass[idxA] == 0. && inverseMass[idxB] == 0.) continue;
                    if (clumpID[idxB] >= 0 && clumpID[idxA] == clumpID[idxB]) continue;
                    double cut = radA + radius[idxB];
                    double3 rAB = posA - position[idxB];
                    if ((cut - length(rAB)) >= -0.1 * cut) count++;
                }
            }
        }
    }
    neighborCount[idxA] = count;
}

__global__ void writeBallInteractionsKernel(double3* slidingSpring,
double3* rollingSpring,
double3* torsionSpring,
int* objectPointed, 
int* objectPointing,
const double3* slidingSpring_old,
const double3* rollingSpring_old,
const double3* torsionSpring_old,
const int* objectPointed_old,
const int* neighborPairHashIndex_old,
const double3* position, 
const double* radius,
const double* inverseMass,
const int* clumpID, 
const int* hashIndex,
const int* neighborPrefixSum, 
const int* interactionStart_old,
const int* interactionEnd_old,
const int* cellHashStart, 
const int* cellHashEnd, 
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numBall)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numBall) return;

    int count = 0;
    int base_w = 0;
    if (idxA > 0) base_w = neighborPrefixSum[idxA - 1];
    double3 posA = position[idxA];
    double radA = radius[idxA];
    int3 gridPositionA = calculateGridPosition(posA, minBound, cellSize);
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
                int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                int hashB = linearIndex3D(gridPositionB, gridSize);
                int startIndex = cellHashStart[hashB];
                if (startIndex == -1) continue;
                int endIndex = cellHashEnd[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = hashIndex[i];
                    if (idxA >= idxB) continue;
                    if (inverseMass[idxA] == 0. && inverseMass[idxB] == 0.) continue;
                    if (clumpID[idxB] >= 0 && clumpID[idxA] == clumpID[idxB]) continue;
                    double cut = radA + radius[idxB];
                    double3 rAB = posA - position[idxB];
                    if ((cut - length(rAB)) >= -0.1 * cut)
                    {
                        int index_w = base_w + count;
                        objectPointed[index_w] = idxA;
                        objectPointing[index_w] = idxB;
                        slidingSpring[index_w] = make_double3(0., 0., 0.);
                        rollingSpring[index_w] = make_double3(0., 0., 0.);
                        torsionSpring[index_w] = make_double3(0., 0., 0.);
                        if (interactionStart_old[idxB] != -1)
                        {
                            for (int j = interactionStart_old[idxB]; j < interactionEnd_old[idxB]; j++)
                            {
                                int j1 = neighborPairHashIndex_old[j];
                                int idxA1 = objectPointed_old[j1];
                                if (idxA == idxA1)
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

__global__ void countBallTriangleInteractionsKernel(int* neighborCount,
const double3* position, 
const double* radius,
const int* index0_tri, 
const int* index1_tri,
const int* index2_tri,
const int* hashIndex_tri,
const double3* globalPosition_ver,
const int* cellHashStart, 
const int* cellHashEnd,
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numBall)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numBall) return;
    int count = 0;

    double3 posA = position[idxA];
    double radA = radius[idxA];
    int3 gridPositionA = calculateGridPosition(posA, minBound, cellSize);
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
                int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                int hashB = linearIndex3D(gridPositionB, gridSize);
                int startIndex = cellHashStart[hashB];
                if (startIndex == -1) continue;
                int endIndex = cellHashEnd[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = hashIndex_tri[i];
                    
                    const double3 v0 = globalPosition_ver[index0_tri[idxB]];
                    const double3 v1 = globalPosition_ver[index1_tri[idxB]];
                    const double3 v2 = globalPosition_ver[index2_tri[idxB]];

                    // triangle AABB
                    const double minx = fmin(v0.x, fmin(v1.x, v2.x));
                    const double miny = fmin(v0.y, fmin(v1.y, v2.y));
                    const double minz = fmin(v0.z, fmin(v1.z, v2.z));
                    const double maxx = fmax(v0.x, fmax(v1.x, v2.x));
                    const double maxy = fmax(v0.y, fmax(v1.y, v2.y));
                    const double maxz = fmax(v0.z, fmax(v1.z, v2.z));

                    // point-to-AABB distance^2
                    const double cx = fmin(fmax(posA.x, minx), maxx);
                    const double cy = fmin(fmax(posA.y, miny), maxy);
                    const double cz = fmin(fmax(posA.z, minz), maxz);

                    const double dx = posA.x - cx;
                    const double dy = posA.y - cy;
                    const double dz = posA.z - cz;

                    if (dx*dx + dy*dy + dz*dz <= radA * radA) 
                    {
                        count++;
                    }
                }
            }
        }
    }
    neighborCount[idxA] = count;
}

__global__ void writeBallTriangleInteractionsKernel(double3* slidingSpring,
double3* rollingSpring,
double3* torsionSpring,
int* objectPointed, 
int* objectPointing,
const double3* slidingSpring_old,
const double3* rollingSpring_old,
const double3* torsionSpring_old,
const int* objectPointed_old,
const int* neighborPairHashIndex_old,
const double3* position, 
const double* radius,
const int* neighborPrefixSum,
const int* index0_tri, 
const int* index1_tri,
const int* index2_tri,
const int* hashIndex_tri,
const int* interactionStart_old_tri,
const int* interactionEnd_old_tri,
const double3* globalPosition_ver,
const int* cellHashStart, 
const int* cellHashEnd,
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numBall)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numBall) return;

    int count = 0;
    int base_w = 0;
    if (idxA > 0) base_w = neighborPrefixSum[idxA - 1];
    double3 posA = position[idxA];
    double radA = radius[idxA];
    int3 gridPositionA = calculateGridPosition(posA, minBound, cellSize);
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
                int3 gridPositionB = make_int3(gridPositionA.x + xx, gridPositionA.y + yy, gridPositionA.z + zz);
                int hashB = linearIndex3D(gridPositionB, gridSize);
                int startIndex = cellHashStart[hashB];
                if (startIndex == -1) continue;
                int endIndex = cellHashEnd[hashB];
                for (int i = startIndex; i < endIndex; i++)
                {
                    int idxB = hashIndex_tri[i];

                    const double3 v0 = globalPosition_ver[index0_tri[idxB]];
                    const double3 v1 = globalPosition_ver[index1_tri[idxB]];
                    const double3 v2 = globalPosition_ver[index2_tri[idxB]];

                    // triangle AABB
                    const double minx = fmin(v0.x, fmin(v1.x, v2.x));
                    const double miny = fmin(v0.y, fmin(v1.y, v2.y));
                    const double minz = fmin(v0.z, fmin(v1.z, v2.z));
                    const double maxx = fmax(v0.x, fmax(v1.x, v2.x));
                    const double maxy = fmax(v0.y, fmax(v1.y, v2.y));
                    const double maxz = fmax(v0.z, fmax(v1.z, v2.z));

                    // point-to-AABB distance^2
                    const double cx = fmin(fmax(posA.x, minx), maxx);
                    const double cy = fmin(fmax(posA.y, miny), maxy);
                    const double cz = fmin(fmax(posA.z, minz), maxz);

                    const double dx = posA.x - cx;
                    const double dy = posA.y - cy;
                    const double dz = posA.z - cz;

                    if (dx*dx + dy*dy + dz*dz <= radA * radA)
                    {
                        int index_w = base_w + count;
                        objectPointed[index_w] = idxA;
                        objectPointing[index_w] = idxB;
                        slidingSpring[index_w] = make_double3(0., 0., 0.);
                        rollingSpring[index_w] = make_double3(0., 0., 0.);
                        torsionSpring[index_w] = make_double3(0., 0., 0.);
                        if (interactionStart_old_tri[idxB] != -1)
                        {
                            for (int j = interactionStart_old_tri[idxB]; j < interactionEnd_old_tri[idxB]; j++)
                            {
                                int j1 = neighborPairHashIndex_old[j];
                                int idxA1 = objectPointed_old[j1];
                                if (idxA == idxA1)
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
cudaStream_t stream_GPU)
{
    countBallInteractionsKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (neighborCount, 
    position,
    radius,
    inverseMass,
    clumpID,
    hashIndex,
    cellHashStart,
    cellHashEnd,
    minBound,
    cellSize,
    gridSize,
    numBall);

    //debug_dump_device_array(neighborCount, numBall, "neighborCount", stream_GPU);

    buildPrefixSum(neighborPrefixSum, 
    neighborCount, 
    numBall,
    stream_GPU);
}

extern "C" void launchWriteBallInteractions(double3* position, 
double* radius,
double* inverseMass,
int* clumpID,
int* hashIndex, 
int* neighborPrefixSum,
int* interactionStart,
int* interactionEnd,

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
cudaStream_t stream_GPU)
{
    writeBallInteractionsKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (slidingSpring,
    rollingSpring,
    torsionSpring,
    objectPointed,
    objectPointing,
    slidingSpring_old,
    rollingSpring_old,
    torsionSpring_old,
    objectPointed_old,
    neighborPairHashIndex_old,
    position,
    radius,
    inverseMass,
    clumpID,
    hashIndex,
    neighborPrefixSum,
    interactionStart,
    interactionEnd,
    cellHashStart,
    cellHashEnd,
    minBound,
    cellSize,
    gridSize,
    numBall);
}

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
cudaStream_t stream_GPU)
{
    countBallTriangleInteractionsKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (neighborCount, 
    position,
    radius,
    index0_tri, 
    index1_tri,
    index2_tri,
    hashIndex_tri,
    globalPosition_ver,
    cellHashStart,
    cellHashEnd,
    minBound,
    cellSize,
    gridSize,
    numBall);

    buildPrefixSum(neighborPrefixSum, 
    neighborCount, 
    numBall,
    stream_GPU);
}

extern "C" void launchWriteBallTriangleInteractions(double3* position, 
double* radius,
int* neighborPrefixSum,

int* index0_tri, 
int* index1_tri,
int* index2_tri,
int* hashIndex_tri,
int* interactionStart_tri,
int* interactionEnd_tri,

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
cudaStream_t stream_GPU)
{
    writeBallTriangleInteractionsKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (slidingSpring,
    rollingSpring,
    torsionSpring,
    objectPointed,
    objectPointing,
    slidingSpring_old,
    rollingSpring_old,
    torsionSpring_old,
    objectPointed_old,
    neighborPairHashIndex_old,
    position,
    radius,
    neighborPrefixSum,
    index0_tri, 
    index1_tri,
    index2_tri,
    hashIndex_tri,
    interactionStart_tri,
    interactionEnd_tri,
    globalPosition_ver,
    cellHashStart,
    cellHashEnd,
    minBound,
    cellSize,
    gridSize,
    numBall);
}