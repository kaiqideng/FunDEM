#pragma once
#include <vector_functions.h>

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
cudaStream_t stream_GPU);

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
cudaStream_t stream_GPU);

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
cudaStream_t stream_GPU);

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
cudaStream_t stream_GPU);

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
cudaStream_t stream_GPU);

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
cudaStream_t stream_GPU);