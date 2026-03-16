#include "SPHNeighborSearchKernel.cuh"
#include "neighborSearchKernel.cuh"
#include "myUtility/myVec.h"
#include "myUtility/buildHashStartEnd.cuh"

__global__ void countSPHInteractionsKernel(int* neighborCount,
const double3* position, 
const double* smoothLength, 
const int* hashIndex,
const int* cellHashStart, 
const int* cellHashEnd,
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numSPH)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numSPH) return;
    int count = 0;

    double3 posA = position[idxA];
    double radA = smoothLength[idxA];
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
                    if (idxA == idxB) continue;
                    double cut = 2.0 * fmax(radA, smoothLength[idxB]);
                    double3 rAB = posA - position[idxB];
                    if ((cut * cut - dot(rAB, rAB)) >= 0.) count++;
                }
            }
        }
    }
    neighborCount[idxA] = count;
}

__global__ void writeSPHInteractionsKernel(int* objectPointed, 
int* objectPointing,
const double3* position, 
const double* smoothLength, 
const int* hashIndex,
const int* neighborPrefixSum, 
const int* cellHashStart, 
const int* cellHashEnd, 
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numSPH)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numSPH) return;

    int count = 0;
    int base_w = 0;
    if (idxA > 0) base_w = neighborPrefixSum[idxA - 1];
    double3 posA = position[idxA];
    double radA = smoothLength[idxA];
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
                    if (idxA == idxB) continue;
                    double cut = 2.0 * fmax(radA, smoothLength[idxB]);
                    double3 rAB = posA - position[idxB];
                    if ((cut * cut - dot(rAB, rAB)) >= 0.)
                    {
                        int index_w = base_w + count;
                        objectPointed[index_w] = idxA;
                        objectPointing[index_w] = idxB;
                        count++;
                    }
                }
            }
        }
    }
}

__global__ void countDummyInteractionsKernel(int* neighborCount,
const double3* position, 
const double* smoothLength, 
const int* clumpID,
const int* hashIndex,
const int* cellHashStart, 
const int* cellHashEnd,
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numDummy)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numDummy) return;
    int count = 0;

    double3 posA = position[idxA];
    double radA = smoothLength[idxA];
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
                    if (clumpID[idxA] != clumpID[idxB]) continue;
                    if (idxA == idxB) continue;
                    double cut = 2.0 * fmax(radA, smoothLength[idxB]);
                    double3 rAB = posA - position[idxB];
                    if ((cut * cut - dot(rAB, rAB)) >= 0.) count++;
                }
            }
        }
    }
    neighborCount[idxA] = count;
}

__global__ void writeDummyInteractionsKernel(int* objectPointed, 
int* objectPointing,
const double3* position, 
const double* smoothLength, 
const int* clumpID,
const int* hashIndex,
const int* neighborPrefixSum, 
const int* cellHashStart, 
const int* cellHashEnd, 
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numDummy)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numDummy) return;

    int count = 0;
    int base_w = 0;
    if (idxA > 0) base_w = neighborPrefixSum[idxA - 1];
    double3 posA = position[idxA];
    double radA = smoothLength[idxA];
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
                    if (clumpID[idxA] != clumpID[idxB]) continue;
                    if (idxA == idxB) continue;
                    double cut = 2.0 * fmax(radA, smoothLength[idxB]);
                    double3 rAB = posA - position[idxB];
                    if ((cut * cut - dot(rAB, rAB)) >= 0.)
                    {
                        int index_w = base_w + count;
                        objectPointed[index_w] = idxA;
                        objectPointing[index_w] = idxB;
                        count++;
                    }
                }
            }
        }
    }
}

__global__ void countSPHDummyInteractionsKernel(int* neighborCount,
const double3* position, 
const double* smoothLength, 
const double3* position_dummy, 
const double* smoothLength_dummy, 
const int* hashIndex_dummy,
const int* cellHashStart, 
const int* cellHashEnd,
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numSPH)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numSPH) return;
    int count = 0;

    double3 posA = position[idxA];
    double radA = smoothLength[idxA];
    int3 gridPositionA = calculateGridPosition(posA, minBound, cellSize);
    int3 gridStart = make_int3(-1, -1, -1);
    int3 gridEnd = make_int3(1, 1, 1);
    if (gridPositionA.x <= 0) {gridPositionA.x = 0; gridStart.x = 0;}
    if (gridPositionA.x >= gridSize.x - 1) {gridPositionA.x = gridSize.x - 1; gridEnd.x = 0;}
    if (gridPositionA.y <= 0) {gridPositionA.y = 0; gridStart.y = 0;}
    if (gridPositionA.y >= gridSize.y - 1) {gridPositionA.y = gridSize.y - 1; gridEnd.y = 0;}
    if (gridPositionA.z <= 0) {gridPositionA.z = 0; gridStart.z = 0;}
    if (gridPositionA.z >= gridSize.z - 1) {gridPositionA.z = gridSize.z - 1; gridEnd.z = 0;}
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
                    int idxB = hashIndex_dummy[i];
                    double cut = 2.0 * fmax(radA, smoothLength_dummy[idxB]);
                    double3 rAB = posA - position_dummy[idxB];
                    if ((cut * cut - dot(rAB, rAB)) >= 0.) count++;
                }
            }
        }
    }
    neighborCount[idxA] = count;
}

__global__ void writeSPHDummyInteractionsKernel(int* objectPointed, 
int* objectPointing,
const double3* position, 
const double* smoothLength,
const int* neighborPrefixSum,
const double3* position_dummy, 
const double* smoothLength_dummy, 
const int* hashIndex_dummy,
const int* cellHashStart, 
const int* cellHashEnd, 
const double3 minBound, 
const double3 cellSize, 
const int3 gridSize, 
const size_t numSPH)
{
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= numSPH) return;

    int count = 0;
    int base_w = 0;
    if (idxA > 0) base_w = neighborPrefixSum[idxA - 1];
    double3 posA = position[idxA];
    double radA = smoothLength[idxA];
    int3 gridPositionA = calculateGridPosition(posA, minBound, cellSize);
    int3 gridStart = make_int3(-1, -1, -1);
    int3 gridEnd = make_int3(1, 1, 1);
    if (gridPositionA.x <= 0) {gridPositionA.x = 0; gridStart.x = 0;}
    if (gridPositionA.x >= gridSize.x - 1) {gridPositionA.x = gridSize.x - 1; gridEnd.x = 0;}
    if (gridPositionA.y <= 0) {gridPositionA.y = 0; gridStart.y = 0;}
    if (gridPositionA.y >= gridSize.y - 1) {gridPositionA.y = gridSize.y - 1; gridEnd.y = 0;}
    if (gridPositionA.z <= 0) {gridPositionA.z = 0; gridStart.z = 0;}
    if (gridPositionA.z >= gridSize.z - 1) {gridPositionA.z = gridSize.z - 1; gridEnd.z = 0;}
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
                    int idxB = hashIndex_dummy[i];
                    double cut = 2.0 * fmax(radA, smoothLength_dummy[idxB]);
                    double3 rAB = posA - position_dummy[idxB];
                    if ((cut * cut - dot(rAB, rAB)) >= 0.)
                    {
                        int index_w = base_w + count;
                        objectPointed[index_w] = idxA;
                        objectPointing[index_w] = idxB;
                        count++;
                    }
                }
            }
        }
    }
}

extern "C" void launchCountSPHInteractions(double3* position, 
double* smoothLength,
int* hashIndex, 
int* neighborCount,
int* neighborPrefixSum,

int* cellHashStart,
int* cellHashEnd,
const double3 minBound,
const double3 cellSize,
const int3 gridSize,

const size_t numSPH,
const size_t gridD_GPU, 
const size_t blockD_GPU, 
cudaStream_t stream_GPU)
{
    countSPHInteractionsKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (neighborCount, 
    position,
    smoothLength,
    hashIndex,
    cellHashStart,
    cellHashEnd,
    minBound,
    cellSize,
    gridSize,
    numSPH);

    buildPrefixSum(neighborPrefixSum, 
    neighborCount, 
    numSPH,
    stream_GPU);
}

extern "C" void launchWriteSPHInteractions(double3* position, 
double* smoothLength,
int* hashIndex, 
int* neighborPrefixSum,

int* objectPointed,
int* objectPointing,

int* cellHashStart,
int* cellHashEnd,
const double3 minBound,
const double3 cellSize,
const int3 gridSize,

const size_t numSPH,
const size_t gridD_GPU, 
const size_t blockD_GPU, 
cudaStream_t stream_GPU)
{
    writeSPHInteractionsKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (objectPointed,
    objectPointing,
    position,
    smoothLength,
    hashIndex,
    neighborPrefixSum,
    cellHashStart,
    cellHashEnd,
    minBound,
    cellSize,
    gridSize,
    numSPH);
}

extern "C" void launchCountDummyInteractions(double3* position, 
double* smoothLength,
int* clumpID,
int* hashIndex, 
int* neighborCount,
int* neighborPrefixSum,

int* cellHashStart,
int* cellHashEnd,

const double3 minBound,
const double3 cellSize,
const int3 gridSize,

const size_t numDummy,
const size_t gridD_GPU, 
const size_t blockD_GPU, 
cudaStream_t stream_GPU)
{
    countDummyInteractionsKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (neighborCount, 
    position,
    smoothLength,
    clumpID, 
    hashIndex,
    cellHashStart,
    cellHashEnd,
    minBound,
    cellSize,
    gridSize,
    numDummy);

    buildPrefixSum(neighborPrefixSum, 
    neighborCount, 
    numDummy,
    stream_GPU);
}

extern "C" void launchWriteDummyInteractions(double3* position, 
double* smoothLength,
int* clumpID,
int* hashIndex, 
int* neighborPrefixSum,

int* objectPointed,
int* objectPointing,

int* cellHashStart,
int* cellHashEnd,

const double3 minBound,
const double3 cellSize,
const int3 gridSize,

const size_t numDummy,
const size_t gridD_GPU, 
const size_t blockD_GPU, 
cudaStream_t stream_GPU)
{
    writeDummyInteractionsKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (objectPointed,
    objectPointing,
    position,
    smoothLength,
    clumpID,
    hashIndex,
    neighborPrefixSum,
    cellHashStart,
    cellHashEnd,
    minBound,
    cellSize,
    gridSize,
    numDummy);
}

extern "C" void launchCountSPHDummyInteractions(double3* position, 
double* smoothLength,
int* neighborCount,
int* neighborPrefixSum,

double3* position_dummy, 
double* smoothLength_dummy,
int* hashIndex_dummy, 

int* cellHashStart,
int* cellHashEnd,

const double3 minBound,
const double3 cellSize,
const int3 gridSize,

const size_t numSPH,
const size_t gridD_GPU, 
const size_t blockD_GPU,
cudaStream_t stream_GPU)
{
    countSPHDummyInteractionsKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (neighborCount, 
    position,
    smoothLength,
    position_dummy,
    smoothLength_dummy,
    hashIndex_dummy,
    cellHashStart,
    cellHashEnd,
    minBound,
    cellSize,
    gridSize,
    numSPH);

    buildPrefixSum(neighborPrefixSum, 
    neighborCount, 
    numSPH,
    stream_GPU);
}

extern "C" void launchWriteSPHDummyInteractions(double3* position, 
double* smoothLength,
int* neighborPrefixSum,

double3* position_dummy, 
double* smoothLength_dummy,
int* hashIndex_dummy, 

int* objectPointed,
int* objectPointing,

int* cellHashStart,
int* cellHashEnd,

const double3 minBound,
const double3 cellSize,
const int3 gridSize,

const size_t numSPH,
const size_t gridD_GPU, 
const size_t blockD_GPU,
cudaStream_t stream_GPU)
{
    writeSPHDummyInteractionsKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (objectPointed,
    objectPointing,
    position,
    smoothLength,
    neighborPrefixSum,
    position_dummy,
    smoothLength_dummy,
    hashIndex_dummy,
    cellHashStart,
    cellHashEnd,
    minBound,
    cellSize,
    gridSize,
    numSPH);
}