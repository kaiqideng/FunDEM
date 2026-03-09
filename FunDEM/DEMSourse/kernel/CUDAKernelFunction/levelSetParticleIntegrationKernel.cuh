#pragma once
#include "myUtility/myMat.h"

extern "C" void launchLevelSetParticle1stHalfIntegration(double3* position, 
double3* velocity, 
double3* angularVelocity, 
double3* force, 
double3* torque, 
double* invMass, 
quaternion* orientation, 
symMatrix* inverseInertiaTensor, 

const double3 gravity, 
const double halfTimeStep,

const size_t numParticle,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream);

extern "C" void launchLevelSetParticle2ndHalfIntegration(double3* velocity, 
double3* angularVelocity, 
double3* force, 
double3* torque, 
double* invMass, 
quaternion* orientation, 
symMatrix* inverseInertiaTensor, 

const double3 gravity, 
const double halfTimeStep,

const size_t numParticle,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream);