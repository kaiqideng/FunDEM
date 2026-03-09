#pragma once

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
cudaStream_t stream_GPU);

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
cudaStream_t stream_GPU);

