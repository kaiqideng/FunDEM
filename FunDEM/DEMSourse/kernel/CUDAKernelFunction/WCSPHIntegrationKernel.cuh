#pragma once
#include "myUtility/myVec.h"

// Wendland 5th-order (C2) kernel in 3D
__device__ __forceinline__ double wendlandKernel3D(double r, double h)
{
	double q = r / h;
	if (q >= 2.0) return 0.0;

	double sigma = 21.0 / (16.0 * pi() * h * h * h);
	double term = 1.0 - 0.5 * q;
	return sigma * pow(term, 4) * (1.0 + 2.0 * q);
}

__device__ __forceinline__ double3 gradWendlandKernel3D(const double3& rij, double h)
{
	double r = length(rij);
	if (r < 1.e-20 || r >= 2.0 * h) return make_double3(0, 0, 0);

	double q = r / h;
	double sigma = 21.0 / (16.0 * pi() * h * h * h);

	double term = 1.0 - 0.5 * q;
	// d/dq of [ (1 - q/2)^4 (1 + 2q) ]
	double dW_dq = (-2.0 * pow(term, 3) * (1.0 + 2.0 * q) + 2.0 * pow(term, 4));

	// chain rule: dW/dr = (dW/dq) * (dq/dr) = (dW/dq) / h
	double dWdr = sigma * dW_dq / h;

	double factor = dWdr / r;  // multiply with rij/r
	return factor * rij;
}

extern "C" void launchCalDummyParticleNormal(double3* normal,
double3* position,
double* density,
double* mass,
double* smoothLength,
int* neighborPrifixSum,

int* objectPointing,

const size_t numDummy,
const size_t gridD_GPU,
const size_t blockD_GPU, 
cudaStream_t stream_GPU);

extern "C" void launchWCSPH1stHalfIntegration(double3* position,
double3* velocity,
double3* acceleration,
double* density,
double* pressure,
double* soundSpeed,
double* mass,
double* initialDensity,
double* smoothLength,
double* viscosity,
int* neighborPrifixSum,
int* neighborPrifixSum_dummy,

double3* position_dummy,
double3* velocity_dummy,
double3* normal_dummy,
double* soundSpeed_dummy,
double* mass_dummy,
double* initialDensity_dummy,
double* smoothLength_dummy,
double* viscosity_dummy,

int* objectPointing,

int* objectPointing_dummy,

const double3 gravity,
const double halfTimeStep,

const size_t numSPH,
const size_t gridD_GPU,
const size_t blockD_GPU, 
cudaStream_t stream_GPU);

extern "C" void launchWCSPH2ndHalfIntegration(double3* position,
double3* velocity,
double3* acceleration,
double* densityChange,
double* density,
double* pressure,
double* soundSpeed,
double* mass,
double* initialDensity,
double* smoothLength,
double* viscosity,
int* neighborPrifixSum,
int* neighborPrifixSum_dummy,

double3* position_dummy,
double3* velocity_dummy,
double3* normal_dummy,
double* soundSpeed_dummy,
double* mass_dummy,
double* initialDensity_dummy,
double* smoothLength_dummy,
double* viscosity_dummy,

int* objectPointing,

int* objectPointing_dummy,

const double3 gravity,
const double halfTimeStep,

const size_t numSPH,
const size_t gridD_GPU,
const size_t blockD_GPU, 
cudaStream_t stream_GPU);

extern "C" void launchUpdateDummyNormal(double3* normal_dummy,

const double3* angularVelocity_solid,

const double timeStep,

const size_t numDummy,
const size_t gridD_GPU,
const size_t blockD_GPU, 
cudaStream_t stream_GPU);

extern "C" void launchAddSPHForce(double3* force,
const double3* position,
const double3* velocity,
const double* inverseMass,

const double3* normal_dummy,
const double* soundSpeed_dummy,
const double* mass_dummy,
const double* initialDensity_dummy,
const double* smoothLength_dummy,
const double* viscosity_dummy,
const int* interactionStart_dummy,
const int* interactionEnd_dummy,

const double3* position_SPH,
const double3* velocity_SPH,
const double* pressure_SPH,
const double* soundSpeed_SPH,
const double* mass_SPH,
const double* density_SPH,
const double* smoothLength_SPH,
const double* viscosity_SPH,

const int* objectPointed,
const int* interactionPairHashIndex,

const double3 gravity,

const size_t numDummy,
const size_t gridD_GPU,
const size_t blockD_GPU, 
cudaStream_t stream_GPU);