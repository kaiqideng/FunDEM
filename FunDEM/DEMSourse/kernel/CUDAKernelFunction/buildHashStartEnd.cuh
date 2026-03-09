#pragma once
#include <cstddef>
#include <driver_types.h>

extern "C" void buildHashStartEnd(int* start, 
int* end, 
int* hashIndex, 

int* hashValue, 

const size_t startEndSize,

const size_t hashListSize,
const size_t gridD_GPU, 
const size_t blockD_GPU,  
cudaStream_t stream_GPU);

extern "C" void buildPrefixSum(int* prefixSum,
int* count, 
const size_t size, 
cudaStream_t stream);