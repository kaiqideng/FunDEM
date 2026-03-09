#include "contactKernel.cuh"
#include "myUtility/contactParameters.h"

__constant__ paramsDevice para;

void contactParameters::buildFromTables(const std::vector<materialRow>& materialTable,
const std::vector<frictionRow>& frictionTable,
const std::vector<linearStiffnessRow>& linearStiffnessTable,
const std::vector<bondRow>& bondTable,
cudaStream_t stream)
{
	const int nMat = inferNumberOfMaterials_(materialTable, frictionTable, linearStiffnessTable, bondTable);
	numberOfMaterials_ = static_cast<std::size_t>(nMat);
	if (nMat > 0)
	{
		for (std::size_t i = 0; i < materialTable.size(); i++)
		{
			YoungsModulus_.pushHost(materialTable[i].YoungsModulus);
			poissonRatio_.pushHost(materialTable[i].poissonRatio);
			restitutionCoefficient_.pushHost(materialTable[i].restitutionCoefficient);
		}

		if (materialTable.size() < nMat)
		{
			for (std::size_t i = 0; i < nMat + 1 - materialTable.size(); i++)
			{
				YoungsModulus_.pushHost(0.);
				poissonRatio_.pushHost(0.);
				restitutionCoefficient_.pushHost(1.0);
			}
		}
		YoungsModulus_.copyHostToDevice(stream);
		poissonRatio_.copyHostToDevice(stream);
		restitutionCoefficient_.copyHostToDevice(stream);	
	}

    pairTableSize_ = computePairTableSize_(nMat);
    if (pairTableSize_ > 0)
	{
		std::vector<double> f(static_cast<std::size_t>(f_COUNT) * pairTableSize_, 0.0);
		std::vector<double> l(static_cast<std::size_t>(l_COUNT) * pairTableSize_, 0.0);
		std::vector<double> b(static_cast<std::size_t>(b_COUNT) * pairTableSize_, 0.0);

		auto pairIdx = [&](int a, int b) -> std::size_t
		{
			return static_cast<std::size_t>(contactParameterIndex(a, b, nMat, static_cast<int>(pairTableSize_)));
		};

		for (const auto& row : frictionTable)
		{
			const std::size_t idx = pairIdx(row.materialIndexA, row.materialIndexB);

			setPacked_(f, pairTableSize_, f_MUS, idx, row.slidingFrictionCoefficient);
			setPacked_(f, pairTableSize_, f_MUR, idx, row.rollingFrictionCoefficient);
			setPacked_(f, pairTableSize_, f_MUT, idx, row.torsionFrictionCoefficient);
		}

		for (const auto& row : linearStiffnessTable)
		{
			const std::size_t idx = pairIdx(row.materialIndexA, row.materialIndexB);

			setPacked_(l, pairTableSize_, l_KN, idx, row.normalStiffness);
			setPacked_(l, pairTableSize_, l_KS, idx, row.slidingStiffness);
			setPacked_(l, pairTableSize_, l_KR, idx, row.rollingStiffness);
			setPacked_(l, pairTableSize_, l_KT, idx, row.torsionStiffness);
		}

		for (const auto& row : bondTable)
		{
			const std::size_t idx = pairIdx(row.materialIndexA, row.materialIndexB);

			setPacked_(b, pairTableSize_, b_GAMMA, idx, row.bondRadiusMultiplier);
			setPacked_(b, pairTableSize_, b_E, idx, row.bondYoungsModulus);
			setPacked_(b, pairTableSize_, b_KNKS, idx, row.normalToShearStiffnessRatio);
			setPacked_(b, pairTableSize_, b_SIGMAS, idx, row.tensileStrength);
			setPacked_(b, pairTableSize_, b_C, idx, row.cohesion);
			setPacked_(b, pairTableSize_, b_MU, idx, row.frictionCoefficient);
		}

		friction_.setHost(f);
		linearStiffness_.setHost(l);
		bond_.setHost(b);
		friction_.copyHostToDevice(stream);
		linearStiffness_.copyHostToDevice(stream);
		bond_.copyHostToDevice(stream);
	}

	paramsDevice dev;
    dev.nMaterials = static_cast<int>(numberOfMaterials_);
    dev.cap = static_cast<int>(pairTableSize_);
	dev.YoungsModulus = YoungsModulus_.d_ptr;
	dev.poissonRatio = poissonRatio_.d_ptr;
	dev.restitutionCoefficient = restitutionCoefficient_.d_ptr;
    dev.friction = friction_.d_ptr;
    dev.linearStiffness = linearStiffness_.d_ptr;
    dev.bond = bond_.d_ptr;
    cudaMemcpyToSymbolAsync(para,
    &dev,
    sizeof(paramsDevice),
    0,
    cudaMemcpyHostToDevice,
    stream);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)       // sm 6.0+
__device__ __forceinline__ double atomicAddDouble(double* addr, double val)
{
	return atomicAdd(addr, val);
}
#else                                                   
__device__ __forceinline__ double atomicAddDouble(double* addr, double val)
{
	auto  addr_ull = reinterpret_cast<unsigned long long*>(addr);
	unsigned long long old = *addr_ull, assumed;

	do {
		assumed = old;
		double  old_d = __longlong_as_double(assumed);
		double  new_d = old_d + val;
		old = atomicCAS(addr_ull, assumed, __double_as_longlong(new_d));
	} while (assumed != old);

	return __longlong_as_double(old);
}
#endif

__device__ void atomicAddDouble3(double3* arr, size_t idx, const double3& v)
{
    atomicAddDouble(&(arr[idx].x), v.x);
	atomicAddDouble(&(arr[idx].y), v.y);
	atomicAddDouble(&(arr[idx].z), v.z);
}

__global__ void updateBallContactKernel(double3* contactPoint,
double3* contactNormal,
double* overlap,
const int* objectPointed, 
const int* objectPointing, 
const double3* position, 
const double* radius,
const size_t numInteraction)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numInteraction) return;

	const int idx_i = objectPointed[idx];
	const int idx_j = objectPointing[idx];

    const double3 r_i = position[idx_i];
	const double3 r_j = position[idx_j];
    const double rad_i = radius[idx_i];
	const double rad_j = radius[idx_j];

    const double3 r_ij = r_i - r_j;
    const double3 n_ij = normalize(r_ij);
    const double delta = rad_i + rad_j - length(r_ij);
	const double3 r_c = r_j + (rad_j - 0.5 * delta) * n_ij;

    contactPoint[idx] = r_c;
    contactNormal[idx] = n_ij;
    overlap[idx] = delta;
}

__global__ void calBallContactForceTorqueKernel(double3* contactForce, 
double3* contactTorque,
double3* slidingSpring, 
double3* rollingSpring, 
double3* torsionSpring, 
const double3* contactPoint,
const double3* contactNormal,
const double* overlap,
const int* objectPointed, 
const int* objectPointing, 
const double3* position, 
const double3* velocity, 
const double3* angularVelocity, 
const double* radius, 
const double* inverseMass,
const int* materialID,
const double dt,
const size_t numInteraction)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numInteraction) return;

	contactForce[idx] = make_double3(0., 0., 0.);
	contactTorque[idx] = make_double3(0., 0., 0.);

	const int idx_i = objectPointed[idx];
	const int idx_j = objectPointing[idx];

    const double3 r_c = contactPoint[idx];
    const double3 n_ij = contactNormal[idx];
	const double delta = overlap[idx];
	
    const double3 r_i = position[idx_i];
	const double3 r_j = position[idx_j];
	const double rad_i = radius[idx_i];
	const double rad_j = radius[idx_j];
	
	const double rad_ij = rad_i * rad_j / (rad_i + rad_j);
	const double m_ij = 1. / (inverseMass[idx_i] + inverseMass[idx_j]); // exclude (inverseMass[idx_i] == 0 && inverseMass[idx_j] == 0) 

	const double3 v_i = velocity[idx_i];
	const double3 v_j = velocity[idx_j];
	const double3 w_i = angularVelocity[idx_i];
	const double3 w_j = angularVelocity[idx_j];
	const double3 v_c_ij = v_i + cross(w_i, r_c - r_i) - (v_j + cross(w_j, r_c - r_j));
	const double3 w_ij = w_i - w_j;

	double3 F_c = make_double3(0, 0, 0);
	double3 T_c = make_double3(0, 0, 0);
	double3 epsilon_s = slidingSpring[idx];
	double3 epsilon_r = rollingSpring[idx];
	double3 epsilon_t = torsionSpring[idx];

    const int mat_i = materialID[idx_i];
	const int mat_j = materialID[idx_j];
    const int ip = contactParameterIndex(mat_i, mat_j, para.nMaterials, para.cap);

	const double e_i = getRestitutionCoefficientParam(mat_i);
	const double e_j = getRestitutionCoefficientParam(mat_j);
	const double e_ij = 2. * e_i * e_j / (e_i + e_j);
	const double logE = log(e_ij);
	const double d_n = -logE / sqrt(logE * logE + pi() * pi());
	const double mu_s = getFrictionParam(ip, f_MUS);
    const double mu_r = getFrictionParam(ip, f_MUR);
    const double mu_t = getFrictionParam(ip, f_MUT);
    const double k_n = getLinearStiffnessParam(ip, l_KN);
    if (k_n > 1.e-20)
    {
        const double k_s = getLinearStiffnessParam(ip, l_KS);
        const double k_r = getLinearStiffnessParam(ip, l_KR);
        const double k_t = getLinearStiffnessParam(ip, l_KT);

        LinearContact(F_c, T_c, epsilon_s, epsilon_r, epsilon_t,
		v_c_ij, w_ij, n_ij, delta, m_ij, rad_ij, dt, 
		k_n, k_s, k_r, k_t, d_n, mu_s, mu_r, mu_t);
    }
    else
    {
		const double E_i = getYoungsModulusParam(mat_i);
		const double E_j = getYoungsModulusParam(mat_j);
		if (E_i < 1.e-20 || E_j < 1.e-20) return;
		const double nu_i = getPoissonRatioParam(mat_i);
		const double nu_j = getPoissonRatioParam(mat_j);
		const double G_i = E_i / (2. * (1. + nu_i));
		const double G_j = E_j / (2. * (1. + nu_j));
		const double E_ij = E_i * E_j / (E_j * (1. +  nu_i * nu_i) + E_i * (1. +  nu_j * nu_j));
		const double G_ij = G_i * G_j / (G_j * (2. - nu_i) + G_i * (2. - nu_j));

        HertzianMindlinContact(F_c, T_c, epsilon_s, epsilon_r, epsilon_t,
		v_c_ij, w_ij, n_ij, delta, m_ij, rad_ij, dt,
		E_ij, G_ij, d_n, mu_s, mu_r, mu_t);
    }

    contactForce[idx] = F_c;
	contactTorque[idx] = T_c;
	slidingSpring[idx] = epsilon_s;
	rollingSpring[idx] = epsilon_r;
	torsionSpring[idx] = epsilon_t;
}

__global__ void updateBallTriangleContact(double3* slidingSpring, 
double3* rollingSpring, 
double3* torsionSpring,
double3* contactPoint,
double3* contactNormal,
double* overlap,
int* cancelFlag, 
 
const int* objectPointing,

const double3* position, 
const double* radius, 
const int* neighborPrefixSum,

const int* index0_t, 
const int* index1_t, 
const int* index2_t, 

const double3* globalPosition_v,

const size_t numBall)
{
	size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numBall) return;

    int start = 0;
	if (idx_i > 0) start = neighborPrefixSum[idx_i - 1];
	int end = neighborPrefixSum[idx_i];
	for (int idx_c = start; idx_c < end; idx_c++)
	{
		cancelFlag[idx_c] = 0;
		
		const int idx_j = objectPointing[idx_c];

		const double rad_i = radius[idx_i];
		const double3 r_i = position[idx_i];

		const double3 p0 = globalPosition_v[index0_t[idx_j]];
		const double3 p1 = globalPosition_v[index1_t[idx_j]];
		const double3 p2 = globalPosition_v[index2_t[idx_j]];
		
		double3 r_c;
		SphereTriangleContactType type = classifySphereTriangleContact(r_i, 
		rad_i,
		p0, 
		p1, 
		p2,
		r_c);

		contactPoint[idx_c] = r_c;
        contactNormal[idx_c] = normalize(r_i - r_c);
        overlap[idx_c] = rad_i - length(r_i - r_c);

        if (type == SphereTriangleContactType::None) 
        {
			slidingSpring[idx_c] = make_double3(0., 0., 0.);
			rollingSpring[idx_c] = make_double3(0., 0., 0.);
			torsionSpring[idx_c] = make_double3(0., 0., 0.);
			cancelFlag[idx_c] = 1;
            continue;
        }

		if (type != SphereTriangleContactType::Face)
		{
			for (int idx_c1 = start; idx_c1 < end; idx_c1++)
			{
				if (idx_c1 == idx_c) continue;

				const int idx_j1 = objectPointing[idx_c1];
				const double3 p01 = globalPosition_v[index0_t[idx_j1]];
				const double3 p11 = globalPosition_v[index1_t[idx_j1]];
				const double3 p21 = globalPosition_v[index2_t[idx_j1]];

				double3 r_c1;
				SphereTriangleContactType type1 = classifySphereTriangleContact(r_i, 
				rad_i,
				p01, 
				p11, 
				p21,
				r_c1);
				
				if (type1 == SphereTriangleContactType::None) continue;
				else if (type1 == SphereTriangleContactType::Face)
				{
					if (lengthSquared(cross(r_c - p01, p11 - p01)) < 1.e-20) 
					{
						slidingSpring[idx_c] = slidingSpring[idx_c1];
						rollingSpring[idx_c] = rollingSpring[idx_c1];
						torsionSpring[idx_c] = torsionSpring[idx_c1];
						cancelFlag[idx_c] = 1;
						break;
					}
					if (lengthSquared(cross(r_c - p11, p21 - p11)) < 1.e-20)
					{
						slidingSpring[idx_c] = slidingSpring[idx_c1];
						rollingSpring[idx_c] = rollingSpring[idx_c1];
						torsionSpring[idx_c] = torsionSpring[idx_c1];
						cancelFlag[idx_c] = 1;
						break;
					}
					if (lengthSquared(cross(r_c - p01, p21 - p01)) < 1.e-20) 
					{
						slidingSpring[idx_c] = slidingSpring[idx_c1];
						rollingSpring[idx_c] = rollingSpring[idx_c1];
						torsionSpring[idx_c] = torsionSpring[idx_c1];
						cancelFlag[idx_c] = 1;
						break;
					}
				}
				else if (type1 == SphereTriangleContactType::Edge)
				{
					if (type == type1)
					{
						if (idx_c1 < idx_c)
						{
							if (lengthSquared(r_c - r_c1) < 1.e-20)
							{
								slidingSpring[idx_c] = slidingSpring[idx_c1];
								rollingSpring[idx_c] = rollingSpring[idx_c1];
								torsionSpring[idx_c] = torsionSpring[idx_c1];
								cancelFlag[idx_c] = 1;
								break;
							}
						}
					}
					else 
					{
						if (lengthSquared(r_c - r_c1) < 1.e-20)
						{
							slidingSpring[idx_c] = slidingSpring[idx_c1];
							rollingSpring[idx_c] = rollingSpring[idx_c1];
							torsionSpring[idx_c] = torsionSpring[idx_c1];
							cancelFlag[idx_c] = 1;
							break;
						}
					}
				}
				else
				{
					if (type == type1)
					{
						if (idx_c1 < idx_c)
						{
							if (lengthSquared(r_c - r_c1) < 1.e-20)
							{
								slidingSpring[idx_c] = slidingSpring[idx_c1];
								rollingSpring[idx_c] = rollingSpring[idx_c1];
								torsionSpring[idx_c] = torsionSpring[idx_c1];
								cancelFlag[idx_c] = 1;
								break;
							}
						}
					}
				}
			}
		}
	}
}

__global__ void addBallWallContactForceTorqueKernel(double3* force,
double3* torque,
double3* contactForce, 
double3* contactTorque,
double3* slidingSpring, 
double3* rollingSpring, 
double3* torsionSpring, 
const double3* contactPoint,
const double3* contactNormal,
const double* overlap,
const int* objectPointed, 
const int* objectPointing, 
const int* cancelFlag, 
const double3* position, 
const double3* velocity, 
const double3* angularVelocity, 
const double* radius, 
const double* inverseMass,
const int* materialID,
const int* neighborPrefixSum,
const double3* position_w, 
const double3* velocity_w, 
const double3* angularVelocity_w, 
const int* materialID_w,
const int* wallIndex_tri,
const double dt,
const size_t numBall)
{
    size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= numBall) return;
	if (inverseMass[idx_i] < 1.e-20) return;

	const int mat_i = materialID[idx_i];
	const double E_i = getYoungsModulusParam(mat_i);
	const double nu_i = getPoissonRatioParam(mat_i);
	const double G_i = E_i / (2. * (1. + nu_i));
	const double e_i = getRestitutionCoefficientParam(mat_i);

	const double3 r_i = position[idx_i];
	const double3 v_i = velocity[idx_i];
	const double3 w_i = angularVelocity[idx_i];

	const double rad_ij = radius[idx_i];
	const double m_ij = 1. / inverseMass[idx_i];

	int start = 0;
	if (idx_i > 0) start = neighborPrefixSum[idx_i - 1];
	int end = neighborPrefixSum[idx_i];
	for (int idx_c = start; idx_c < end; idx_c++)
	{
		contactForce[idx_c] = make_double3(0, 0, 0);
		contactTorque[idx_c] = make_double3(0, 0, 0);
		if (cancelFlag[idx_c] == 1) continue;

		const int idx_j = objectPointing[idx_c];
		const int idx_w = wallIndex_tri[idx_j];

		const double3 r_c = contactPoint[idx_c];
		const double3 n_ij = contactNormal[idx_c];
		const double delta = overlap[idx_c];

		const double3 r_j = position_w[idx_w];
		const double3 v_j = velocity_w[idx_w];
		const double3 w_j = angularVelocity_w[idx_w];
		const double3 v_c_ij = v_i + cross(w_i, r_c - r_i) - (v_j + cross(w_j, r_c - r_j));
		const double3 w_ij = w_i - w_j;

		double3 F_c = make_double3(0, 0, 0);
		double3 T_c = make_double3(0, 0, 0);
		double3 epsilon_s = slidingSpring[idx_c];
		double3 epsilon_r = rollingSpring[idx_c];
		double3 epsilon_t = torsionSpring[idx_c];

        const int mat_j = materialID_w[idx_w];
		const int ip = contactParameterIndex(mat_i, mat_j, para.nMaterials, para.cap);

		const double e_j = getRestitutionCoefficientParam(mat_j);
		const double e_ij = 2. * e_i * e_j / (e_i + e_j);
		const double logE = log(e_ij);
		const double d_n = -logE / sqrt(logE * logE + pi() * pi());
		const double mu_s = getFrictionParam(ip, f_MUS);
		const double mu_r = getFrictionParam(ip, f_MUR);
		const double mu_t = getFrictionParam(ip, f_MUT);
		const double k_n = getLinearStiffnessParam(ip, l_KN);
		if (k_n > 1.e-20)
		{
			const double k_s = getLinearStiffnessParam(ip, l_KS);
			const double k_r = getLinearStiffnessParam(ip, l_KR);
			const double k_t = getLinearStiffnessParam(ip, l_KT);

			LinearContact(F_c, T_c, epsilon_s, epsilon_r, epsilon_t,
			v_c_ij, w_ij, n_ij, delta, m_ij, rad_ij, dt, 
			k_n, k_s, k_r, k_t, d_n, mu_s, mu_r, mu_t);
		}
		else
		{
			const double E_j = getYoungsModulusParam(mat_j);
			if (E_i < 1.e-20 && E_j < 1.e-20) return;
			const double nu_j = getPoissonRatioParam(mat_j);
			const double G_j = E_j / (2. * (1. + nu_j));
			const double E_ij = E_i * E_j / (E_j * (1. +  nu_i * nu_i) + E_i * (1. +  nu_j * nu_j));
			const double G_ij = G_i * G_j / (G_j * (2. - nu_i) + G_i * (2. - nu_j));

			HertzianMindlinContact(F_c, T_c, epsilon_s, epsilon_r, epsilon_t,
			v_c_ij, w_ij, n_ij, delta, m_ij, rad_ij, dt,
			E_ij, G_ij, d_n, mu_s, mu_r, mu_t);
		}

		contactForce[idx_c] = F_c;
		contactTorque[idx_c] = T_c;
		slidingSpring[idx_c] = epsilon_s;
		rollingSpring[idx_c] = epsilon_r;
		torsionSpring[idx_c] = epsilon_t;

		force[idx_i] += F_c;
		torque[idx_i] += T_c + cross(r_c - r_i, F_c);
	}
}

__global__ void addBondedForceTorqueKernel(double3* bondPoint,
double3* bondNormal,
double3* shearForce, 
double3* bendingTorque,
double* normalForce, 
double* torsionTorque, 
double* maxNormalStress,
double* maxShearStress,
int* isBonded, 

double3* contactForce, 
double3* contactTorque,

double3* force, 
double3* torque, 

const int* objectPointed_b, 
const int* objectPointing_b,

const double3* contactPoint,
const double3* contactNormal,
const int* objectPointing,

const double3* position, 
const double3* velocity, 
const double3* angularVelocity, 
const double* radius, 
const int* materialID, 
const int* neighborPrefixSum,

const double dt,
const size_t numBondedInteraction)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numBondedInteraction) return;

	if (isBonded[idx] == 0)
	{
		normalForce[idx] = 0;
		torsionTorque[idx] = 0;
		shearForce[idx] = make_double3(0, 0, 0);
		bendingTorque[idx] = make_double3(0, 0, 0);
		return;
	}

	const int idx_i = objectPointed_b[idx];
	const int idx_j = objectPointing_b[idx];

    const double3 r_i = position[idx_i];
	const double3 r_j = position[idx_j];
	const double rad_i = radius[idx_i];
	const double rad_j = radius[idx_j];

    const double3 n_ij0 = bondNormal[idx];
	double3 n_ij = normalize(r_i - r_j);
    const double delta = rad_i + rad_j - length(r_i - r_j);
	double3 r_c = r_j + (rad_j - 0.5 * delta) * n_ij;
    bondPoint[idx] = r_c;
    bondNormal[idx] = n_ij;

	bool flag = false;
	int idx_c = 0;
	const int neighborStart_i = idx_i > 0 ? neighborPrefixSum[idx_i - 1] : 0;
	const int neighborEnd_i = neighborPrefixSum[idx_i];
	for (int k = neighborStart_i; k < neighborEnd_i; k++)
	{
		if (objectPointing[k] == idx_j)
		{
			flag = true;
			idx_c = k;
            r_c = contactPoint[idx_c];
            n_ij = contactNormal[idx_c];
            bondPoint[idx] = r_c;
            bondNormal[idx] = n_ij;
			break;
		}
	}
	
	const double3 v_i = velocity[idx_i];
	const double3 v_j = velocity[idx_j];
	const double3 w_i = angularVelocity[idx_i];
	const double3 w_j = angularVelocity[idx_j];
	const double3 v_c_ij = v_i + cross(w_i, r_c - r_i) - (v_j + cross(w_j, r_c - r_j));

    const int ip = contactParameterIndex(materialID[idx_i], materialID[idx_j], para.nMaterials, para.cap);
	const double gamma = getBondParam(ip, b_GAMMA);
    const double E = getBondParam(ip, b_E);
    const double k_n_k_s = getBondParam(ip, b_KNKS);
    const double sigma_s = getBondParam(ip, b_SIGMAS);
    const double C = getBondParam(ip, b_C);
    const double mu = getBondParam(ip, b_MU);

	double F_n = normalForce[idx];
	double3 F_s = shearForce[idx];
	double T_t = torsionTorque[idx];
	double3 T_b = bendingTorque[idx];
	double sigma_max = maxNormalStress[idx];
	double tau_max = maxShearStress[idx];
	isBonded[idx] = ParallelBondedContact(F_n, T_t, F_s, T_b, sigma_max, tau_max,
	n_ij0, n_ij, v_c_ij, w_i, w_j, rad_i, rad_j, dt,
	gamma, E, k_n_k_s, sigma_s, C, mu);

	normalForce[idx] = F_n;
	shearForce[idx] = F_s;
	torsionTorque[idx] = T_t;
	bendingTorque[idx] = T_b;
	maxNormalStress[idx] = sigma_max;
	maxShearStress[idx] = tau_max;

	if (!flag)
	{
		atomicAddDouble3(force, idx_i, F_n * n_ij + F_s);
		atomicAddDouble3(torque, idx_i, T_t * n_ij + T_b + cross(r_c - r_i, F_s));
		atomicAddDouble3(force, idx_j, -F_n * n_ij - F_s);
		atomicAddDouble3(torque, idx_j, -T_t * n_ij - T_b + cross(r_c - r_j, -F_s));
		return;
	}

	contactForce[idx_c] += F_n * n_ij + F_s;
	contactTorque[idx_c] += T_t * n_ij + T_b;
}

__global__ void calLevelSetParticleContactForceTorqueKernel(double3* contactForce, 
double3* slidingSpring, 
double3* force_p,
double3* torque_p,

const double3* contactPoint,
const double3* contactNormal,
const double* overlap,
const int* objectPointed, 
const int* objectPointing, 

const int* particleID_bNode,

const double3* position_p, 
const double3* velocity_p, 
const double3* angularVelocity_p,
const int* materialID_p,
const double dt,
const size_t numInteraction)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numInteraction) return;

	contactForce[idx] = make_double3(0., 0., 0.);

	const int idx_i = particleID_bNode[objectPointed[idx]];
	const int idx_j = objectPointing[idx];

    const double3 r_c = contactPoint[idx];
    const double3 n_ij = contactNormal[idx];
	const double delta = overlap[idx];
	
    const double3 r_i = position_p[idx_i];
	const double3 r_j = position_p[idx_j];

	const double3 v_i = velocity_p[idx_i];
	const double3 v_j = velocity_p[idx_j];
	const double3 w_i = angularVelocity_p[idx_i];
	const double3 w_j = angularVelocity_p[idx_j];
	const double3 v_c_ij = v_i + cross(w_i, r_c - r_i) - (v_j + cross(w_j, r_c - r_j));
	const double3 w_ij = w_i - w_j;

	double3 F_c = make_double3(0, 0, 0);
	double3 epsilon_s = slidingSpring[idx];

    const int mat_i = materialID_p[idx_i];
	const int mat_j = materialID_p[idx_j];
    const int ip = contactParameterIndex(mat_i, mat_j, para.nMaterials, para.cap);

	const double mu_s = getFrictionParam(ip, f_MUS);

    const double k_n = getLinearStiffnessParam(ip, l_KN);
    const double k_s = getLinearStiffnessParam(ip, l_KS);

	LinearContactForLevelSetParticle(F_c, epsilon_s, 
	v_c_ij, w_ij, n_ij, delta, dt, 
	k_n, k_s, mu_s);

    contactForce[idx] = F_c;
	slidingSpring[idx] = epsilon_s;

	atomicAddDouble3(force_p, idx_i, F_c);
	atomicAddDouble3(torque_p, idx_i, cross(r_c - r_i, F_c));
	atomicAddDouble3(force_p, idx_j, -F_c);
	atomicAddDouble3(torque_p, idx_j, cross(r_c - r_j, -F_c));
}

__global__ void sumObjectPointedForceTorqueFromInteractionKernel(double3* force, 
double3* torque, 
const double3* position, 
const int* neighborPrefixSum,
const double3* contactForce, 
const double3* contactTorque, 
const double3* contactPoint,
const size_t num)
{
	size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= num) return;

	double3 r_i = position[idx_i];
	double3 F_i = make_double3(0., 0., 0.);
	double3 T_i = make_double3(0., 0., 0.);
	for (int k = idx_i > 0 ? neighborPrefixSum[idx_i - 1] : 0; k < neighborPrefixSum[idx_i]; k++)
	{
		double3 r_c = contactPoint[k];
		F_i += contactForce[k];
		T_i += contactTorque[k] + cross(r_c - r_i, contactForce[k]);
	}

	force[idx_i] += F_i;
	torque[idx_i] += T_i;
}

__global__ void sumObjectPointingForceTorqueFromInteractionKernel(double3* force, 
double3* torque, 
const double3* position, 
const int* interactionStart, 
const int* interactionEnd,
const double3* contactForce, 
const double3* contactTorque, 
const double3* contactPoint,
const int* neighborPairHashIndex,
const size_t num)
{
	size_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_i >= num) return;

	double3 r_i = position[idx_i];
	double3 F_i = make_double3(0., 0., 0.);
	double3 T_i = make_double3(0., 0., 0.);
	if (interactionStart[idx_i] >= 0)
	{
		for (int k = interactionStart[idx_i]; k < interactionEnd[idx_i]; k++)
		{
			int k1 = neighborPairHashIndex[k];
			double3 r_c = contactPoint[k1];
			F_i -= contactForce[k1];
			T_i -= contactTorque[k1];
			T_i -= cross(r_c - r_i, contactForce[k1]);
		}
	}

	force[idx_i] += F_i;
	torque[idx_i] += T_i;
}

extern "C" void launchCalculateBallContactForceTorque(double3* position, 
double3* velocity, 
double3* angularVelocity, 
double* radius,
double* inverseMass,
int* materialID,

double3* slidingSpring, 
double3* rollingSpring, 
double3* torsionSpring, 
double3* contactForce,
double3* contactTorque,
double3* contactPoint,
double3* contactNormal,
double* overlap,
int* objectPointed, 
int* objectPointing,

const double timeStep,

const size_t numInteraction,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream)
{
	updateBallContactKernel <<<gridD, blockD, 0, stream>>> (contactPoint, 
	contactNormal, 
	overlap, 
	objectPointed, 
	objectPointing, 
	position, 
	radius, 
	numInteraction);

	calBallContactForceTorqueKernel <<<gridD, blockD, 0, stream>>> (contactForce,
	contactTorque,
	slidingSpring,
	rollingSpring,
	torsionSpring,
	contactPoint,
	contactNormal,
	overlap,
	objectPointed,
	objectPointing,
	position,
	velocity,
	angularVelocity,
	radius,
	inverseMass,
	materialID,
	timeStep,
	numInteraction);
}

extern "C" void launchAddBondedForceTorque(double3* position, 
double3* velocity, 
double3* angularVelocity, 
double3* force, 
double3* torque,
double* radius,
int* materialID,
int* neighborPrefixSum,

double3* contactForce,
double3* contactTorque,
double3* contactPoint,
double3* contactNormal,
int* objectPointing, 

double3* bondPoint,
double3* bondNormal,
double3* shearForce, 
double3* bendingTorque,
double* normalForce, 
double* torsionTorque, 
double* maxNormalStress,
double* maxShearStress,
int* isBonded, 
int* objectPointed_b, 
int* objectPointing_b,

const double timeStep,

const size_t numBondedInteraction,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream)
{
	addBondedForceTorqueKernel <<<gridD, blockD, 0, stream>>> (bondPoint, 
	bondNormal, 
	shearForce, 
	bendingTorque, 
	normalForce, 
	torsionTorque, 
	maxNormalStress, 
	maxShearStress, 
	isBonded, 
	contactForce, 
	contactTorque, 
	force, 
	torque, 
	objectPointed_b, 
	objectPointing_b, 
	contactPoint, 
	contactNormal, 
	objectPointing, 
	position, 
	velocity, 
	angularVelocity, 
	radius, 
	materialID, 
	neighborPrefixSum, 
	timeStep, 
	numBondedInteraction);
}

extern "C" void launchSumContactForceTorque(double3* position, 
double3* force, 
double3* torque,
int* neighborPrefixSum,
int* interactionStart, 
int* interactionEnd,

double3* contactForce,
double3* contactTorque,
double3* contactPoint,
int* neighborPairHashIndex,

const size_t num,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream)
{
	sumObjectPointedForceTorqueFromInteractionKernel <<<gridD, blockD, 0, stream>>> (force, 
	torque, 
	position, 
	neighborPrefixSum, 
	contactForce, 
	contactTorque, 
	contactPoint, 
	num);

	sumObjectPointingForceTorqueFromInteractionKernel <<<gridD, blockD, 0, stream>>> (force, 
	torque, 
	position, 
	interactionStart, 
	interactionEnd, 
	contactForce, 
	contactTorque, 
	contactPoint, 
	neighborPairHashIndex, 
	num);
}

void launchCalculateBallWallContactForceTorque(double3* position, 
double3* velocity, 
double3* angularVelocity, 
double3* force, 
double3* torque,
double* radius,
double* inverseMass,
int* materialID,
int* neighborPrefixSum,

double3* position_w, 
double3* velocity_w, 
double3* angularVelocity_w, 
int* materialID_w,

int* index0_t, 
int* index1_t, 
int* index2_t, 
int* wallIndex_tri,

double3* globalPosition_v, 

double3* slidingSpring, 
double3* rollingSpring, 
double3* torsionSpring,
double3* contactForce,
double3* contactTorque,
double3* contactPoint,
double3* contactNormal,
double* overlap,
int* objectPointed, 
int* objectPointing,
int* cancelFlag,

const double timeStep,

const size_t numBall,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream)
{
	updateBallTriangleContact <<<gridD, blockD, 0, stream>>> (slidingSpring, 
	rollingSpring, 
	torsionSpring, 
	contactPoint, 
	contactNormal, 
	overlap, 
	cancelFlag, 
	objectPointing, 
	position, 
	radius, 
	neighborPrefixSum, 
	index0_t, 
	index1_t, 
	index2_t, 
	globalPosition_v, 
	numBall);

	addBallWallContactForceTorqueKernel <<<gridD, blockD, 0, stream>>> (force, 
	torque, 
	contactForce, 
	contactTorque, 
	slidingSpring, 
	rollingSpring, 
	torsionSpring, 
	contactPoint, 
	contactNormal, 
	overlap, 
	objectPointed,
	objectPointing, 
	cancelFlag, 
	position, 
	velocity, 
	angularVelocity, 
	radius, 
	inverseMass, 
	materialID, 
    neighborPrefixSum,
	position_w, 
	velocity_w, 
	angularVelocity_w, 
	materialID_w, 
	wallIndex_tri,
	timeStep, 
	numBall);
}

extern "C" void launchCalLevelSetParticleContactForceTorque(double3* contactForce,
double3* slidingSpring,
const double3* contactPoint,
const double3* contactNormal,
const double* overlap,
const int* objectPointed,
const int* objectPointing,

const int* particleID_bNode,

double3* force_p,
double3* torque_p,
const double3* position_p,
const double3* velocity_p,
const double3* angularVelocity_p,
const int* materialID_p,
const double timeStep,

const size_t numInteraction,
const size_t gridD,
const size_t blockD,
cudaStream_t stream)
{
    calLevelSetParticleContactForceTorqueKernel<<<gridD, blockD, 0, stream>>>(contactForce,
	slidingSpring,
	force_p,
	torque_p,

	contactPoint,
	contactNormal,
	overlap,
	objectPointed,
	objectPointing,

	particleID_bNode,

	position_p,
	velocity_p,
	angularVelocity_p,
	materialID_p,
	timeStep,
	numInteraction);
}