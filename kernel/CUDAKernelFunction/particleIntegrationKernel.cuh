#pragma once
#include "myUtility/myMat.h"

extern "C" void launchParticleVelocityAngularVelocityIntegration(double3* velocity,
double3* angularVelocity,
const double3* force,
const double3* torque,
const quaternion* orientation,
const int* geometryID,

const double* inverseMass_geo, 
const symMatrix* inverseInertiaTensor_geo, 

const double3 gravity,
const double timeStep,

const size_t num,
const size_t gridD,
const size_t blockD,
cudaStream_t stream);

extern "C" void launchParticlePositionOrientationIntegration(double3* position,
quaternion* orientation,
const double3* velocity,
const double3* angularVelocity,

const double timeStep,

const size_t num,
const size_t gridD,
const size_t blockD,
cudaStream_t stream);