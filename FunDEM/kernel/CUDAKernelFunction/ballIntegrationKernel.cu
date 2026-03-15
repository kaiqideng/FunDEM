#include "ballIntegrationKernel.cuh"

__global__ void sumClumpForceTorqueKernel(double3* force_c, 
double3* torque_c, 
const double3* position_c, 
const double* invMass_c, 
const int* pebbleStart_c, 
const int* pebbleEnd_c,
const double3* position_p,
const double3* force_p, 
const double3* torque_p,
const size_t numClump)
{
	size_t idx_c = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_c >= numClump) return;

	force_c[idx_c] = make_double3(0, 0, 0);
	torque_c[idx_c] = make_double3(0, 0, 0);

	if (invMass_c[idx_c] < 1.e-30) return;
	
	double3 r_c = position_c[idx_c];
	double3 F_c = make_double3(0, 0, 0);
	double3 T_c = make_double3(0, 0, 0);
	for (int i = pebbleStart_c[idx_c]; i < pebbleEnd_c[idx_c]; i++)
	{
		double3 r_i = position_p[i];
		double3 F_i = force_p[i];
		F_c += F_i;
		T_c += torque_p[i] + cross(r_i - r_c, F_i);
	}

	force_c[idx_c] = F_c;
	torque_c[idx_c] = T_c;
}

__global__ void ballVelocityAngularVelocityIntegrationKernel(double3* velocity, 
double3* angularVelocity, 
const double3* force, 
const double3* torque, 
const double* radius, 
const double* invMass, 
const int* clumpID, 
const double3 g,
const double dt,
const size_t numBall)
{
	size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numBall) return;
	if (clumpID[idx_i] >= 0) return;

	double invM_i = invMass[idx_i];
	if (invM_i < 1.e-30) return;

	velocity[idx_i] += (force[idx_i] * invM_i + g) * dt;

	double rad_i = radius[idx_i];
	if (rad_i < 1.e-30) return;

	double invI_i = invM_i / (0.4 * rad_i * rad_i);
	angularVelocity[idx_i] += torque[idx_i] * invI_i * dt;
}

__global__ void ballPositionIntegrationKernel(double3* position, 
const double3* velocity, 
const int* clumpID, 
const double dt,
const size_t num)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num) return;
	if (clumpID[idx] >= 0) return;

	position[idx] += dt * velocity[idx];
}

__global__ void clumpVelocityAngularVelocityIntegrationKernel(double3* velocity_c, 
double3* angularVelocity_c, 
const double3* force_c, 
const double3* torque_c, 
const double* invMass_c, 
const quaternion* orientation_c, 
const symMatrix* inverseInertiaTensor_c, 
const double3 g,
const double dt, 
const size_t numClump)
{
	size_t idx_c = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_c >= numClump) return;

    double invM_c = invMass_c[idx_c];
	if (invM_c < 1.e-30) return;

	velocity_c[idx_c] += (force_c[idx_c] * invM_c + g) * dt;
	angularVelocity_c[idx_c] += (rotateInverseInertiaTensor(orientation_c[idx_c], inverseInertiaTensor_c[idx_c]) * torque_c[idx_c]) * dt;
}

__global__ void clumpPositionOrientationIntegrationKernel(double3* position, 
quaternion* orientation, 
const double3* velocity, 
const double3* angularVelocity, 
const double dt,
const size_t num)
{
	size_t idx_c = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_c >= num) return;

	position[idx_c] += dt * velocity[idx_c];
	orientation[idx_c] = quaternionRotate(orientation[idx_c], angularVelocity[idx_c], dt);
}

__global__ void setPebblePositionVelocityAngularVelocityKernel(double3* position, 
double3* velocity, 
double3* angularVelocity, 
const double3* localPosition,
const int* clumpID,
const double3* position_c,
const double3* velocity_c, 
const double3* angularVelocity_c,
const quaternion* orientation_c,
const size_t numBall)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numBall) return;
	
	int idx_c = clumpID[idx];
	if (idx_c < 0) return;

	const double3 r_c = position_c[idx_c];
    const double3 r_i = r_c + rotateVectorByQuaternion(orientation_c[idx_c], localPosition[idx]);
	position[idx] = r_i;
    const double3 w_c = angularVelocity_c[idx_c];
    angularVelocity[idx] = w_c;
	velocity[idx] = velocity_c[idx_c] + cross(w_c, r_i - r_c);
}

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
cudaStream_t stream)
{
	ballVelocityAngularVelocityIntegrationKernel <<<gridD, blockD, 0, stream>>> (velocity, 
	angularVelocity, 
	force, 
	torque, 
	radius, 
	invMass, 
	clumpID, 
	gravity,
	halfTimeStep,
	numBall);

	ballPositionIntegrationKernel <<<gridD, blockD, 0, stream>>> (position, 
	velocity, 
	clumpID,
	2.0 * halfTimeStep,
	numBall);
}

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
cudaStream_t stream)
{
	ballVelocityAngularVelocityIntegrationKernel <<<gridD, blockD, 0, stream>>> (velocity, 
	angularVelocity, 
	force, 
	torque, 
	radius, 
	invMass, 
	clumpID, 
	gravity,
	halfTimeStep,
	numBall);
}

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
cudaStream_t stream)
{
	sumClumpForceTorqueKernel <<<gridD, blockD, 0, stream>>> (force, 
	torque,
	position,
	invMass,
	pebbleStart,
	pebbleEnd,
	position_p,
	force_p,
	torque_p,
	numClump);

	clumpVelocityAngularVelocityIntegrationKernel <<<gridD, blockD, 0, stream>>> (velocity, 
	angularVelocity, 
	force, 
	torque, 
	invMass, 
    orientation, 
    inverseInertiaTensor, 
	gravity,
	halfTimeStep,
	numClump);

	clumpPositionOrientationIntegrationKernel <<<gridD, blockD, 0, stream>>> (position, 
	orientation, 
	velocity, 
	angularVelocity, 
	2.0 * halfTimeStep,
	numClump);
}

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
cudaStream_t stream)
{
	sumClumpForceTorqueKernel <<<gridD, blockD, 0, stream>>> (force, 
	torque,
	position,
	invMass,
	pebbleStart,
	pebbleEnd,
	position_p,
	force_p,
	torque_p,
	numClump);

	clumpVelocityAngularVelocityIntegrationKernel <<<gridD, blockD, 0, stream>>> (velocity, 
	angularVelocity, 
	force, 
	torque, 
	invMass, 
    orientation, 
    inverseInertiaTensor,
	gravity,
	halfTimeStep,
	numClump);
}

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
cudaStream_t stream)
{
	setPebblePositionVelocityAngularVelocityKernel <<<gridD, blockD, 0, stream>>> (position,
	velocity, 
	angularVelocity, 
	localPosition, 
	clumpID, 
	position_c, 
	velocity_c, 
	angularVelocity_c, 
	orientation_c, 
	numBall);
}