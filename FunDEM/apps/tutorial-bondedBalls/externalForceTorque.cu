#include "externalForceTorque.cuh"
#include "kernel/CUDAKernelFunction/myUtility/myVec.h"

__global__ void addConstantForceToParticle(double3* force,
const double3* force_external,
const size_t num)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num) return;

    force[idx] += force_external[idx];
}

extern "C" void launchAddConstantForceToParticle(double3* force,
const double3* force_external,
const size_t num,
const size_t gridD,
const size_t blockD,
cudaStream_t stream)
{
    addConstantForceToParticle <<<gridD, blockD, 0, stream>>> (force, 
    force_external,
    num);
}

__global__ void addGlobalDampingForceTorqueToParticle(double3* force,
double3* torque,
const double3* velocity,
const double3* angularVelocity,
const double dampCoeff,
const size_t num)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num) return;

    double3 f = force[idx], t = torque[idx];
    force[idx] -= dampCoeff * length(f) * normalize(velocity[idx]);
    torque[idx] -= dampCoeff * length(t) * normalize(angularVelocity[idx]);
}

extern "C" void launchAddGlobalDampingForceTorqueToParticle(double3* force,
double3* torque,
const double3* velocity,
const double3* angularVelocity,
const double dampCoeff,
const size_t num,
const size_t gridD,
const size_t blockD,
cudaStream_t stream)
{
    addGlobalDampingForceTorqueToParticle <<<gridD, blockD, 0, stream>>> (force, 
    torque, 
    velocity, 
    angularVelocity, 
    dampCoeff, 
    num);
}

#include <math_constants.h> // CUDART_PI

// ------------------------------------------------------------
// Buoyancy + Drag for spheres against a horizontal free-surface plane
//   plane: z = waterLevel
// ------------------------------------------------------------

__device__ __forceinline__ double sphereSubmergedVolumePlaneZ(double zc,
double r,
double waterLevel)
{
    const double d = waterLevel - zc;

    if (d <= -r) return 0.0;
    if (d >=  r) return (4.0 / 3.0) * CUDART_PI * r * r * r;

    const double h = d + r; // (0, 2r)
    return (CUDART_PI * h * h * (3.0 * r - h)) / 3.0;
}

__global__ void addBuoyancyDragKernel(double3* force,
const double3* position,
const double3* velocity,
const double* radius,
const double* inverseMass,
const double rhoFluid,
const double Cd,
const double3 gravity,
const double3 fluidVelocity,
const double waterLevel,
const size_t numBall)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBall) return;

    if (inverseMass && inverseMass[idx] <= 0.0) return;

    const double3 r_i = position[idx];
    const double3 v_i = velocity[idx];
    const double  rad = radius[idx];

    const double Vsub = sphereSubmergedVolumePlaneZ(r_i.z, rad, waterLevel);
    if (Vsub <= 0.0) return;

    // Buoyancy: -rho * Vsub * g
    const double3 Fb = make_double3(-rhoFluid * Vsub * gravity.x,
                                   -rhoFluid * Vsub * gravity.y,
                                   -rhoFluid * Vsub * gravity.z);

    // Drag (quadratic), scaled by submerged ratio alpha
    const double Vs  = (4.0 / 3.0) * CUDART_PI * rad * rad * rad;
    const double alpha = (Vs > 1.e-30) ? (Vsub / Vs) : 0.0;

    const double3 vrel = make_double3(v_i.x - fluidVelocity.x,
                                     v_i.y - fluidVelocity.y,
                                     v_i.z - fluidVelocity.z);

    const double v2 = vrel.x * vrel.x + vrel.y * vrel.y + vrel.z * vrel.z;

    double3 Fd = make_double3(0.0, 0.0, 0.0);
    if (v2 > 1.e-30 && Cd > 0.0)
    {
        const double vmag = sqrt(v2);
        const double Aeff = alpha * (CUDART_PI * rad * rad);
        const double k = 0.5 * rhoFluid * Cd * Aeff * vmag;

        Fd.x = -k * vrel.x;
        Fd.y = -k * vrel.y;
        Fd.z = -k * vrel.z;
    }

    force[idx].x += Fb.x + Fd.x;
    force[idx].y += Fb.y + Fd.y;
    force[idx].z += Fb.z + Fd.z;
}

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
cudaStream_t stream)
{
    addBuoyancyDragKernel<<<gridD, blockD, 0, stream>>>(force,
                                                        position,
                                                        velocity,
                                                        radius,
                                                        inverseMass,
                                                        rhoFluid,
                                                        Cd,
                                                        gravity,
                                                        fluidVelocity,
                                                        waterLevel,
                                                        numBall);
}