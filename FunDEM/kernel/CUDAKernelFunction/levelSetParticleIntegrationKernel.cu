#include "levelSetParticleIntegrationKernel.cuh"
#include "ballIntegrationKernel.cuh"

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
cudaStream_t stream)
{
	clumpVelocityAngularVelocityIntegrationKernel <<<gridD, blockD, 0, stream>>> (velocity, 
	angularVelocity, 
	force, 
	torque, 
	invMass, 
    orientation, 
    inverseInertiaTensor, 
	gravity,
	halfTimeStep,
	numParticle);

	clumpPositionOrientationIntegrationKernel <<<gridD, blockD, 0, stream>>> (position, 
	orientation, 
	velocity, 
	angularVelocity, 
	2.0 * halfTimeStep,
	numParticle);
}

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
cudaStream_t stream)
{
	clumpVelocityAngularVelocityIntegrationKernel <<<gridD, blockD, 0, stream>>> (velocity, 
	angularVelocity, 
	force, 
	torque, 
	invMass, 
    orientation, 
    inverseInertiaTensor,
	gravity,
	halfTimeStep,
	numParticle);
}