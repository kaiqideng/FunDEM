#pragma once
#include "myUtility/myMat.h"

__global__ void clumpVelocityAngularVelocityIntegrationKernel(double3* velocity_c, 
double3* angularVelocity_c, 
const double3* force_c, 
const double3* torque_c, 
const double* invMass_c, 
const quaternion* orientation_c, 
const symMatrix* inverseInertiaTensor_c, 

const double3 g,
const double dt, 

const size_t numClump);

__global__ void clumpPositionOrientationIntegrationKernel(double3* position, 
quaternion* orientation, 
const double3* velocity, 
const double3* angularVelocity, 

const double dt,

const size_t num);

extern "C" void launchBall1stHalfIntegration(double3* position, 
double3* velocity, 
double3* angularVelocity, 
double3* force, 
double3* torque, 
double* radius, 
double* invMass, 
int* clumpID, 

const double3 gravity, 
const double halfTimeStep,

const size_t numBall,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream);

extern "C" void launchBall2ndHalfIntegration(double3* velocity, 
double3* angularVelocity, 
double3* force, 
double3* torque, 
double* radius, 
double* invMass, 
int* clumpID,

const double3 gravity, 
const double halfTimeStep, 

const size_t numBall,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream);

extern "C" void launchClump1stHalfIntegration(double3* position, 
double3* velocity, 
double3* angularVelocity, 
double3* force, 
double3* torque, 
double* invMass, 
quaternion* orientation, 
symMatrix* inverseInertiaTensor, 
const int* pebbleStart, 
const int* pebbleEnd,

double3* position_p, 
double3* force_p, 
double3* torque_p,

const double3 gravity, 
const double halfTimeStep,

const size_t numClump,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream);

extern "C" void launchClump2ndHalfIntegration(double3* position, 
double3* velocity, 
double3* angularVelocity, 
double3* force, 
double3* torque, 
double* invMass, 
quaternion* orientation, 
symMatrix* inverseInertiaTensor, 
const int* pebbleStart, 
const int* pebbleEnd,

double3* position_p, 
double3* force_p, 
double3* torque_p,

const double3 gravity, 
const double halfTimeStep,

const size_t numClump,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream);

extern "C" void launchSetPebblePositionVelocityAngularVelocityKernel(double3* position,
double3* velocity, 
double3* angularVelocity,
const double3* localPosition,
const int* clumpID,

const double3* position_c,
const double3* velocity_c, 
const double3* angularVelocity_c,
const quaternion* orientation_c,

const size_t numBall,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream);