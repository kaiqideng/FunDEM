#pragma once
#include <cstddef>
#include <driver_types.h>
#include <vector_types.h>

extern "C" void launchAddConstantForce(double3* force,
const double3* force_external,
const size_t num,
const size_t gridD,
const size_t blockD,
cudaStream_t stream);

extern "C" void launchAddGlobalDampingForceTorque(double3* force,
double3* torque,
const double3* velocity,
const double3* angularVelocity,
const double dampCoeff,
const size_t num,
const size_t gridD,
const size_t blockD,
cudaStream_t stream);

extern "C" void launchAddBuoyancyDrag(double3* force,
const double3* position,
const double3* velocity,
const double* radius,
const double* inverseMass,
const double rhoFluid,
const double Cd,
const double3 gravity,
const double3 fluidVelocity,
const double waterLevel,
const size_t numBall,
const size_t gridD,
const size_t blockD,
cudaStream_t stream);