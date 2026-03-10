#include "WCSPHIntegrationKernel.cuh"

__global__ void calDummyParticleNormalKernel(double3* normal,
const double3* position,
const double* density,
const double* mass,
const double* smoothLength,
const int* neighborPrifixSum,
const int* objectPointing,
const size_t numDummy)
{
    size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numDummy) return;
    normal[idx_i] = make_double3(0., 0., 0.);

    double3 r_i = position[idx_i];
    double h_i = smoothLength[idx_i];

    double3 Theta = make_double3(0., 0., 0.);
    int start = 0;
    if (idx_i > 0) start = neighborPrifixSum[idx_i - 1];
    int end = neighborPrifixSum[idx_i];
    for (int k = start; k < end; k++)
    {
        int idx_j = objectPointing[k];

        double3 r_j = position[idx_j];
        double h_j = smoothLength[idx_j];
        double m_j = mass[idx_j];
        double rho_j = density[idx_j];

        double3 r_ij = r_i - r_j;
        double3 dW_ij = gradWendlandKernel3D(r_ij, 0.5 * (h_i + h_j));

        Theta -= m_j / rho_j * dW_ij;
    }

    if (lengthSquared(Theta) > 1.e-10) normal[idx_i] = Theta / length(Theta);
}

__global__ void updateWCSPHAccelerationKernel(double3* acceleration,
const double3* position,
const double3* velocity,
const double* density,
const double* pressure,
const double* soundSpeed,
const double* mass,
const double* initialDensity,
const double* smoothLength,
const double* viscosity,
const int* neighborPrifixSum,
const int* neighborPrifixSum_dummy,
const double3* position_dummy,
const double3* velocity_dummy,
const double3* normal_dummy,
const double* soundSpeed_dummy,
const double* mass_dummy,
const double* initialDensity_dummy,
const double* smoothLength_dummy,
const double* viscosity_dummy,
const int* objectPointing,
const int* objectPointing_dummy,
const double3 gravity,
const size_t numSPH)
{
    int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numSPH) return;

    double3 r_i = position[idx_i];
    double3 v_i = velocity[idx_i];
    double c_i = soundSpeed[idx_i];
    double h_i = smoothLength[idx_i];
    double nu_i = viscosity[idx_i];
    double rho_i = density[idx_i];
    double P_L = pressure[idx_i];

    double3 acc = make_double3(0.0, 0.0, 0.0);
    int start = 0;
    if (idx_i > 0) start = neighborPrifixSum[idx_i - 1];
    int end = neighborPrifixSum[idx_i];
    for (int k = start; k < end; k++)
    {
        int idx_j = objectPointing[k];

        double3 r_j = position[idx_j];
        double3 v_j = velocity[idx_j];
        double c_j = soundSpeed[idx_j];
        double h_j = smoothLength[idx_j];
        double nu_j = viscosity[idx_j];
        double rho_j = density[idx_j];
        double P_R = pressure[idx_j];
        double m_j = mass[idx_j];

        double3 r_ij = r_i - r_j;
        double3 e_ij = -normalize(r_ij);
        double U_L = dot(v_i, e_ij);
        double U_R = dot(v_j, e_ij);
        double3 dW_ij = gradWendlandKernel3D(r_ij, 0.5 * (h_i + h_j));

        double c_bar = 0.5 * (c_i + c_j);
        double rho_bar = 0.5 * (rho_i + rho_j);
        double beta = fmin(3.0 * fmax(U_L - U_R, 0.0), c_bar);
        double P_star = 0.5 * (P_L + P_R) + 0.5 * beta * rho_bar * (U_L - U_R);
        acc -= 2.0 * m_j * (P_star / (rho_i * rho_j)) * dW_ij;

        double nu_ij = 0.5 * (nu_i + nu_j);
        acc += 2.0 * nu_ij * dot(normalize(r_ij), dW_ij) * (v_i - v_j) * m_j / (rho_i * rho_j);
    }

    int start_d = 0;
    if (idx_i > 0) start_d = neighborPrifixSum_dummy[idx_i - 1];
    int end_d = neighborPrifixSum_dummy[idx_i];
    for (int k = start_d; k < end_d; k++)
    {
        int idx_j = objectPointing_dummy[k];

        double3 r_j = position_dummy[idx_j];
        double3 v_j = velocity_dummy[idx_j];
        double c_j = soundSpeed_dummy[idx_j];
        double h_j = smoothLength_dummy[idx_j];
        double nu_j = viscosity_dummy[idx_j];
        double rho_j = initialDensity_dummy[idx_j];
        double m_j = mass_dummy[idx_j];

        double3 n_w = normal_dummy[idx_j];
        double U_L = dot(-n_w, v_i);
        double U_R = -U_L + 2.0 * dot(-n_w, v_j);
        double g_r_ji = dot(gravity, r_j - r_i);
        if (g_r_ji < 0.) g_r_ji = 0.;
        double P_R = P_L + rho_i * g_r_ji;
        rho_j = P_R / (c_j * c_j) + initialDensity_dummy[idx_j];

        double3 r_ij = r_i - r_j;
        double3 dW_ij = gradWendlandKernel3D(r_ij, 0.5 * (h_i + h_j));

        double c_bar = 0.5 * (c_i + c_j);
        double rho_bar = 0.5 * (rho_i + rho_j);
        double beta = fmin(3.0 * fmax(U_L - U_R, 0.0), c_bar);
        double P_star = 0.5 * (P_L + P_R) + 0.5 * beta * rho_bar * (U_L - U_R);
        acc -= 2.0 * m_j * (P_star / (rho_i * rho_j)) * dW_ij;

        double nu_ij = 0.5 * (nu_i + nu_j);
        acc += 2.0 * nu_ij * dot(normalize(r_ij), dW_ij) * (v_i - v_j) * m_j / (rho_i * rho_j);
    }

    acceleration[idx_i] = acc + gravity;
}

__global__ void SPHVelocityIntegrationKernel(double3* velocity,
const double3* acceleration,
const double timeStep,
const size_t numSPH)
{
	size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numSPH) return;

	velocity[idx_i] += timeStep * acceleration[idx_i];
}

__global__ void SPHPositionIntegrationKernel(double3* position,
const double3* velocity,
const double timeStep,
const size_t numSPH)
{
	size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numSPH) return;

	position[idx_i] += timeStep * velocity[idx_i];
}

__global__ void updateWCSPHDensityChangeKernel(double* densityChange,
const double* pressure,
const double3* position,
const double3* velocity,
const double* soundSpeed,
const double* mass,
const double* density,
const double* smoothLength,
const int* neighborPrifixSum,
const int* neighborPrifixSum_dummy,
const double3* position_dummy,
const double3* velocity_dummy,
const double3* normal_dummy,
const double* soundSpeed_dummy,
const double* mass_dummy,
const double* initialDensity_dummy,
const double* smoothLength_dummy,
const int* objectPointing,
const int* objectPointing_dummy,
const double3 gravity,
const double timeStep,
const size_t numSPH)
{
    int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numSPH) return;

    double3 r_i = position[idx_i];
    double3 v_i = velocity[idx_i];
    double c_i = soundSpeed[idx_i];
    double h_i = smoothLength[idx_i];
    double rho_i = density[idx_i];
    double P_L = pressure[idx_i];

    double dRho = 0.0;
    int start = 0;
    if (idx_i > 0) start = neighborPrifixSum[idx_i - 1];
    int end = neighborPrifixSum[idx_i];
    for (int k = start; k < end; k++)
    {
        int idx_j = objectPointing[k];

        double3 r_j = position[idx_j];
        double3 v_j = velocity[idx_j];
        double c_j = soundSpeed[idx_j];
        double h_j = smoothLength[idx_j];
        double rho_j = density[idx_j];
        double P_R = pressure[idx_j];
        double m_j = mass[idx_j];

        double3 r_ij = r_i - r_j;
        double3 dW_ij = gradWendlandKernel3D(r_ij, 0.5 * (h_i + h_j));
        double3 e_ij = -normalize(r_ij);

        double c_bar = 0.5 * (c_i + c_j);
        double rho_bar = 0.5 * (rho_i + rho_j);
        double3 v_star = 0.5 * (v_i + v_j);
        v_star += 0.5 * ((P_L - P_R) / (rho_bar * c_bar)) * e_ij;
        dRho += m_j / rho_j * dot(v_i - v_star, dW_ij);
    }

    int start_d = 0;
    if (idx_i > 0) start_d = neighborPrifixSum_dummy[idx_i - 1];
    int end_d = neighborPrifixSum_dummy[idx_i];
    for (int k = start_d; k < end_d; k++)
    {
        int idx_j = objectPointing_dummy[k];

        double3 r_j = position_dummy[idx_j];
        double3 v_j = velocity_dummy[idx_j];
        double c_j = soundSpeed_dummy[idx_j];
        double h_j = smoothLength_dummy[idx_j];
        double rho_j = initialDensity_dummy[idx_j];
        double m_j = mass_dummy[idx_j];

        double g_r_ji = dot(gravity, r_j - r_i);
        if (g_r_ji < 0.) g_r_ji = 0.;
        double P_R = P_L + rho_i * g_r_ji;
        rho_j = P_R / (c_j * c_j) + initialDensity_dummy[idx_j];

        double3 r_ij = r_i - r_j;
        double3 dW_ij = gradWendlandKernel3D(r_ij, 0.5 * (h_i + h_j));
        double3 e_ij = -normalize(r_ij);

        double c_bar = 0.5 * (c_i + c_j);
        double rho_bar = 0.5 * (rho_i + rho_j);
        double3 v_star = 0.5 * (v_i + v_j);
        v_star += 0.5 * ((P_L - P_R) / (rho_bar * c_bar)) * e_ij;
        dRho += m_j / rho_j * dot(v_i - v_star, dW_ij);
    }

    dRho *= 2.0 * rho_i;
    densityChange[idx_i] += dRho * timeStep;
}

__global__ void SPHDensityPressureIntegrationKernel(double* density,
double* pressure,
double* densityChange,
const double* initialDensity,
const double* soundSpeed,
const size_t numSPH)
{
    size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numSPH) return;

    double change = densityChange[idx_i];
    double rho0 = initialDensity[idx_i];
    if (change < -0.01 * rho0)
    {
        change = -0.01 * rho0;
        densityChange[idx_i] = change;
    }
    density[idx_i] = rho0 + change;
    double c = soundSpeed[idx_i];
    pressure[idx_i] = c * c * change;
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
cudaStream_t stream_GPU)
{
    calDummyParticleNormalKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (normal, 
    position, 
    density, 
    mass, 
    smoothLength, 
    neighborPrifixSum, 
    objectPointing,
    numDummy);
}

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
cudaStream_t stream_GPU)
{
    updateWCSPHAccelerationKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (acceleration,
    position,
    velocity,
    density,
    pressure,
    soundSpeed,
    mass,
    initialDensity,
    smoothLength,
    viscosity,
    neighborPrifixSum,
    neighborPrifixSum_dummy,
    position_dummy,
    velocity_dummy,
    normal_dummy,
    soundSpeed_dummy,
    mass_dummy,
    initialDensity_dummy,
    smoothLength_dummy,
    viscosity_dummy,
    objectPointing,
    objectPointing_dummy,
    gravity,
    numSPH);

    SPHVelocityIntegrationKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (velocity, 
    acceleration, 
    halfTimeStep, 
    numSPH);

    SPHPositionIntegrationKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (position, 
    velocity, 
    2. * halfTimeStep, 
    numSPH);
}

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
cudaStream_t stream_GPU)
{
    updateWCSPHDensityChangeKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (densityChange,
    pressure,
    position,
    velocity,
    soundSpeed,
    mass,
    density,
    smoothLength,
    neighborPrifixSum,
    neighborPrifixSum_dummy,
    position_dummy,
    velocity_dummy,
    normal_dummy,
    soundSpeed_dummy,
    mass_dummy,
    initialDensity_dummy,
    smoothLength_dummy,
    objectPointing,
    objectPointing_dummy,
    gravity,
    2. * halfTimeStep, 
    numSPH);

    SPHDensityPressureIntegrationKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (density, 
    pressure,
    densityChange,
    initialDensity,
    soundSpeed,
    numSPH);

    updateWCSPHAccelerationKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (acceleration,
    position,
    velocity,
    density,
    pressure,
    soundSpeed,
    mass,
    initialDensity,
    smoothLength,
    viscosity,
    neighborPrifixSum,
    neighborPrifixSum_dummy,
    position_dummy,
    velocity_dummy,
    normal_dummy,
    soundSpeed_dummy,
    mass_dummy,
    initialDensity_dummy,
    smoothLength_dummy,
    viscosity_dummy,
    objectPointing,
    objectPointing_dummy,
    gravity,
    numSPH);

    SPHVelocityIntegrationKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (velocity, 
    acceleration, 
    halfTimeStep, 
    numSPH);
}

__global__ void updateDummyNormalKernel(double3* normal_dummy,
const double3* angularVelocity_solid,
const double timeStep,
const size_t numDummy)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numDummy) return;

    double3 dAngle = angularVelocity_solid[idx] * timeStep;
    normal_dummy[idx] = rotateVector(normal_dummy[idx], dAngle);
}

__global__ void addSPHForceKernel(double3* force,
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
const size_t numDummy)
{
    int idx_j = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_j >= numDummy) return;
    if (interactionStart_dummy[idx_j] < 0) return;
    if (inverseMass[idx_j] < 1.e-20) return;

    double3 r_j = position[idx_j];
    double3 v_j = velocity[idx_j];
    double3 n_w = normal_dummy[idx_j];
    double c_j = soundSpeed_dummy[idx_j];
    double h_j = smoothLength_dummy[idx_j];
    double nu_j = viscosity_dummy[idx_j];
    double rho0_j = initialDensity_dummy[idx_j];
    double m_j = mass_dummy[idx_j];

    double3 f = make_double3(0.0, 0.0, 0.0);
    for (int k = interactionStart_dummy[idx_j]; k < interactionEnd_dummy[idx_j]; k++)
    {
        int k1 = interactionPairHashIndex[k];
        int idx_i = objectPointed[k1];

        double3 r_i = position_SPH[idx_i];
        double3 v_i = velocity_SPH[idx_i];
        double c_i = soundSpeed_SPH[idx_i];
        double h_i = smoothLength_SPH[idx_i];
        double nu_i = viscosity_SPH[idx_i];
        double rho_i = density_SPH[idx_i];
        double m_i = mass_SPH[idx_i];

        double P_L = pressure_SPH[idx_i];

        double U_L = dot(-n_w, v_i);
        double U_R = -U_L + 2.0 * dot(-n_w, v_j);
        double g_r_ji = dot(gravity, r_j - r_i);
        if (g_r_ji < 0.) g_r_ji = 0.;
        double P_R = P_L + rho_i * g_r_ji;
        double rho_j = P_R / (c_j * c_j) + rho0_j;

        double3 r_ij = r_i - r_j;
        double3 dW_ij = gradWendlandKernel3D(r_ij, 0.5 * (h_i + h_j));

        double c_bar = 0.5 * (c_i + c_j);
        double rho_bar = 0.5 * (rho_i + rho_j);
        double beta = fmin(3.0 * fmax(U_L - U_R, 0.0), c_bar);
        double P_star = 0.5 * (P_L + P_R) + 0.5 * beta * rho_bar * (U_L - U_R);
        f += m_i * (-2.0 * m_j * (P_star / (rho_i * rho_j)) * dW_ij);

        double nu_ij = 0.5 * (nu_i + nu_j);
        f += m_i * (2.0 * nu_ij * dot(normalize(r_ij), dW_ij) * (v_i - v_j) * m_j / (rho_i * rho_j));
    }

    force[idx_j] += -f;
}

extern "C" void launchUpdateDummyNormal(double3* normal_dummy,

const double3* angularVelocity_solid,

const double timeStep,

const size_t numDummy,
const size_t gridD_GPU,
const size_t blockD_GPU, 
cudaStream_t stream_GPU)
{
    updateDummyNormalKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (normal_dummy,
    angularVelocity_solid,
    timeStep,
    numDummy);
}

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
cudaStream_t stream_GPU)
{
    addSPHForceKernel <<<gridD_GPU, blockD_GPU, 0, stream_GPU>>> (force,
    position,
    velocity,
    inverseMass,
    normal_dummy,
    soundSpeed_dummy,
    mass_dummy,
    initialDensity_dummy,
    smoothLength_dummy,
    viscosity_dummy,
    interactionStart_dummy,
    interactionEnd_dummy,
    position_SPH,
    velocity_SPH,
    pressure_SPH,
    soundSpeed_SPH,
    mass_SPH,
    density_SPH,
    smoothLength_SPH,
    viscosity_SPH,
    objectPointed,
    interactionPairHashIndex,
    gravity,
    numDummy);
}