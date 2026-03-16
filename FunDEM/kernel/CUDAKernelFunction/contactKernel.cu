#include "contactKernel.cuh"
#include "levelSetParticleContactDetectionKernel.cuh"
#include "myUtility/myQua.h"
#include "myUtility/myContactParameters.h"

__constant__ paramsDevice para;

/**
 * @brief Build and upload contact parameter tables, then publish device pointers via constant memory (`para`).
 *
 * This function prepares all material- and pair-dependent parameters needed by contact kernels:
 *
 * 1) Material table normalization
 *    - Ensures the material table has at least @p nMat entries by padding missing rows with defaults:
 *      Young's modulus = 0, Poisson ratio = 0, restitution coefficient = 1.
 *
 * 2) Pair table allocation (upper-triangular indexing)
 *    - Computes @p pairTableSize_ = nMat*(nMat+1)/2 + 1 (with a fallback slot, depending on implementation).
 *    - Builds per-pair scalar arrays:
 *      - effectiveYoungsModulus_ (E_ij): effective Young's modulus for the pair (i,j)
 *      - effectiveShearModulus_  (G_ij): effective shear modulus for the pair (i,j)
 *      - dissipation_           (d_ij): normal dissipation factor derived from restitution coefficients
 *    - Builds per-pair packed arrays (param-major layout using setPacked_ and the pair index):
 *      - friction_        : sliding/rolling/torsion friction coefficients (mu_s, mu_r, mu_t)
 *      - linearStiffness_ : linear stiffness values (k_n, k_s, k_r, k_t)
 *      - bond_            : bond parameters (gamma, E_bond, kn/ks, sigma_s, cohesion, mu)
 *
 * 3) Upload to device
 *    - Copies the above arrays to device memory using the provided CUDA stream.
 *
 * 4) Commit constant-memory descriptor
 *    - Writes a small POD descriptor @c paramsDevice (sizes + device pointers) into
 *      CUDA constant memory symbol @c para via cudaMemcpyToSymbolAsync.
 *    - Contact kernels read parameters through @c para on the device side.
 *
 * Notes / conventions:
 * - Pair indexing is symmetric: (i,j) == (j,i), using @c upperTriangularIndex(i,j,nMat,cap).
 * - Effective moduli (E_ij, G_ij) are only computed when both inputs are non-zero; otherwise set to 0.
 * - Dissipation d_ij is computed from restitution using: d = -log(e)/sqrt(log(e)^2 + pi^2),
 *   where e_ij is combined from e_i and e_j (harmonic-like blend). If restitution is invalid, d_ij = 0.
 *
 * @param[in] materialTable         Per-material properties (Young's modulus, Poisson ratio, restitution coefficient).
 * @param[in] frictionTable         Per-pair friction coefficients (mu_s, mu_r, mu_t).
 * @param[in] linearStiffnessTable  Per-pair linear stiffness values (k_n, k_s, k_r, k_t).
 * @param[in] bondTable             Per-pair bond parameters (gamma, E_bond, kn/ks, sigma_s, cohesion, mu).
 * @param[in] stream                CUDA stream used for all async H2D copies and cudaMemcpyToSymbolAsync.
 */
void contactParameters::buildFromTables(const vector<materialRow>& materialTable,
const std::vector<frictionRow>& frictionTable,
const std::vector<linearStiffnessRow>& linearStiffnessTable,
const std::vector<bondRow>& bondTable,
cudaStream_t stream)
{
	const int nMat = inferNumberOfMaterials_(materialTable, frictionTable, linearStiffnessTable, bondTable);
	numberOfMaterials_ = static_cast<size_t>(nMat);

    vector<materialRow> materialTableNew = materialTable;
	for (size_t i = 0; i < nMat - materialTableNew.size() + 1; i++)
	{
		materialRow row;
		row.YoungsModulus = 0.;
		row.poissonRatio = 0.;
		row.restitutionCoefficient = 1.;
		materialTableNew.push_back(row);
	}

    pairTableSize_ = computePairTableSize_(numberOfMaterials_);
    if (pairTableSize_ > 0)
	{
		std::vector<double> E(pairTableSize_, 0.0);
		std::vector<double> G(pairTableSize_, 0.0);
		std::vector<double> d(pairTableSize_, 0.0);
		std::vector<double> f(static_cast<size_t>(f_COUNT) * pairTableSize_, 0.0);
		std::vector<double> l(static_cast<size_t>(l_COUNT) * pairTableSize_, 0.0);
		std::vector<double> b(static_cast<size_t>(b_COUNT) * pairTableSize_, 0.0);

		auto pairIdx = [&](int a, int b) -> size_t
		{
			return static_cast<size_t>(upperTriangularIndex(a, b, nMat, static_cast<int>(pairTableSize_)));
		};

		for (int i = 0; i < nMat; i++)
		{
			for (int j = i; j < nMat; j++)
			{
				const double E_i = materialTableNew[i].YoungsModulus;
				const double E_j = materialTableNew[j].YoungsModulus;
				const double nu_i = materialTableNew[i].poissonRatio;
				const double nu_j = materialTableNew[j].poissonRatio;
				const double e_i = materialTableNew[i].restitutionCoefficient;
				const double e_j = materialTableNew[j].restitutionCoefficient;
				double E_ij = 0., G_ij = 0., G_i = 0., G_j = 0.;
				if (nu_i > -1. && nu_i < 2.) G_i = E_i / (2. * (1. + nu_i));
				if (nu_j > -1. && nu_j < 2.) G_j = E_j / (2. * (1. + nu_j));
				if ((!isZero(E_i)) && (!isZero(E_j))) E_ij = E_i * E_j / (E_j * (1. + nu_i * nu_i) + E_i * (1. + nu_j * nu_j));
				if ((!isZero(G_i)) && (!isZero(G_j))) G_ij = G_i * G_j / (G_j * (2. - nu_i) + G_i * (2. - nu_j));
				double d_ij = 0.;
				if ((e_i <= 1.&& (!isZero(e_i))) || (e_j <= 1.&& (!isZero(e_j))))
				{
					const double e_ij = 2. * e_i * e_j / (e_i + e_j);
					const double logE = log(e_ij);
					d_ij = -logE / sqrt(logE * logE + pi() * pi());
				}

				const size_t idx = pairIdx(i, j);

				setPacked_(E, pairTableSize_, 0, idx, E_ij);
				setPacked_(G, pairTableSize_, 0, idx, G_ij);
				setPacked_(d, pairTableSize_, 0, idx, d_ij);
			}
		}

		for (const auto& row : frictionTable)
		{
			const size_t idx = pairIdx(row.materialIndexA, row.materialIndexB);

			setPacked_(f, pairTableSize_, f_MUS, idx, row.slidingFrictionCoefficient);
			setPacked_(f, pairTableSize_, f_MUR, idx, row.rollingFrictionCoefficient);
			setPacked_(f, pairTableSize_, f_MUT, idx, row.torsionFrictionCoefficient);
		}

		for (const auto& row : linearStiffnessTable)
		{
			const size_t idx = pairIdx(row.materialIndexA, row.materialIndexB);

			setPacked_(l, pairTableSize_, l_KN, idx, row.normalStiffness);
			setPacked_(l, pairTableSize_, l_KS, idx, row.slidingStiffness);
			setPacked_(l, pairTableSize_, l_KR, idx, row.rollingStiffness);
			setPacked_(l, pairTableSize_, l_KT, idx, row.torsionStiffness);
		}

		for (const auto& row : bondTable)
		{
			const size_t idx = pairIdx(row.materialIndexA, row.materialIndexB);

			setPacked_(b, pairTableSize_, b_GAMMA, idx, row.bondRadiusMultiplier);
			setPacked_(b, pairTableSize_, b_E, idx, row.bondYoungsModulus);
			setPacked_(b, pairTableSize_, b_KNKS, idx, row.normalToShearStiffnessRatio);
			setPacked_(b, pairTableSize_, b_SIGMAS, idx, row.tensileStrength);
			setPacked_(b, pairTableSize_, b_C, idx, row.cohesion);
			setPacked_(b, pairTableSize_, b_MU, idx, row.frictionCoefficient);
		}

        effectiveYoungsModulus_.setHost(E);
		effectiveShearModulus_.setHost(G);
		dissipation_.setHost(d);
		friction_.setHost(f);
		linearStiffness_.setHost(l);
		bond_.setHost(b);

		effectiveYoungsModulus_.copyHostToDevice(stream);
		effectiveShearModulus_.copyHostToDevice(stream);
		dissipation_.copyHostToDevice(stream);
		friction_.copyHostToDevice(stream);
		linearStiffness_.copyHostToDevice(stream);
		bond_.copyHostToDevice(stream);
	}

	paramsDevice dev;
    dev.nMaterials = static_cast<int>(numberOfMaterials_);
    dev.cap = static_cast<int>(pairTableSize_);
	dev.effectiveYoungsModulus = effectiveYoungsModulus_.d_ptr;
	dev.effectiveShearModulus = effectiveShearModulus_.d_ptr;
	dev.dissipation = dissipation_.d_ptr;
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

/**
 * @brief Atomic add for double on device. Uses native atomicAdd on sm_60+; CAS loop otherwise.
 *
 * @param[in,out] addr   Address to add into.
 * @param[in]     val    Value to add.
 * @return The old value stored at *addr before the add (CUDA atomicAdd semantics).
 */
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

/**
 * @brief Atomic add a double3 vector into arr[idx] component-wise.
 *
 * @param[in,out] arr   Target array of double3.
 * @param[in]     idx   Index into arr.
 * @param[in]     v     Value to add to arr[idx].
 */
__device__ void atomicAddDouble3(double3* arr, size_t idx, const double3& v)
{
    atomicAddDouble(&(arr[idx].x), v.x);
	atomicAddDouble(&(arr[idx].y), v.y);
	atomicAddDouble(&(arr[idx].z), v.z);
}

/**
 * @brief Update ball-ball contact kinematics (contact point, normal, overlap) from current positions/radii.
 *
 * Pair convention:
 * - objectPointed[idx]  = i (ball i)
 * - objectPointing[idx] = j (ball j)
 *
 * Contact geometry (sphere-sphere):
 * - n_ij = normalized (r_i - r_j) (points from j to i)
 * - overlap delta = (r_i + r_j) - |r_i - r_j|
 * - contact point r_c located on the line of centers (your current formula).
 *
 * @param[out] contactPoint     Contact point in global coordinates per interaction.
 * @param[out] contactNormal    Unit normal n_ij per interaction (points from j -> i).
 * @param[out] overlap          Normal overlap delta per interaction (positive if penetrating).
 * @param[in]  objectPointed    Ball index i per interaction.
 * @param[in]  objectPointing   Ball index j per interaction.

 * @param[in]  position         Ball centers r_i / r_j in global coordinates.
 * @param[in]  radius           Ball radii.

 * @param[in]  numInteraction   Number of interactions in arrays.
 */
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

/**
 * @brief Compute ball-ball contact force/torque for each interaction and update spring histories.
 *
 * This kernel:
 * - reads contact kinematics: contactPoint, contactNormal, overlap
 * - computes relative contact velocity v_c_ij and relative angular velocity w_ij
 * - loads/updates history variables (sliding/rolling/torsion spring states)
 * - chooses contact model:
 *     - if linear stiffness k_n != 0: LinearContact
 *     - else: HertzianMindlinContact (needs E_i,E_j,nu_i,nu_j)
 * - writes per-interaction contactForce/contactTorque and updated spring histories.
 *
 * Pair convention:
 * - objectPointed[idx]  = i
 * - objectPointing[idx] = j
 *
 * @param[out] contactForce       Contact force on i (interaction-local result).
 * @param[out] contactTorque      Contact torque on i (interaction-local result).
 * @param[in,out] slidingSpring   Sliding spring history per interaction.
 * @param[in,out] rollingSpring   Rolling spring history per interaction.
 * @param[in,out] torsionSpring   Torsion spring history per interaction.
 *
 * @param[in]  contactPoint       Contact point (global).
 * @param[in]  contactNormal      Contact normal n_ij (global, unit).
 * @param[in]  overlap            Overlap delta.
 * @param[in]  objectPointed      Ball i index per interaction.
 * @param[in]  objectPointing     Ball j index per interaction.
 *
 * @param[in]  position           Ball positions r (global).
 * @param[in]  velocity           Ball translational velocities.
 * @param[in]  angularVelocity    Ball angular velocities.
 * @param[in]  radius             Ball radii.
 * @param[in]  inverseMass        Ball inverse masses.
 * @param[in]  materialID         Ball material ids (used to lookup parameters in `para`).

 * @param[in]  dt                 Time step.
 * @param[in]  numInteraction     Number of interactions.
 */
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
    const int ip = upperTriangularIndex(mat_i, mat_j, para.nMaterials, para.cap);

	const double d_n = getDissipationParam(ip);
	const double mu_s = getFrictionParam(ip, f_MUS);
    const double mu_r = getFrictionParam(ip, f_MUR);
    const double mu_t = getFrictionParam(ip, f_MUT);
    const double k_n = getLinearStiffnessParam(ip, l_KN);
    if (!isZero(k_n))
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
		const double E_ij = getEffectiveYoungsModulusParam(ip);
		const double G_ij = getEffectiveShearModulusParam(ip);

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

/**
 * @brief Update ball-triangle contact kinematics for each ball based on candidate triangle list.
 *
 * For each ball idx_i:
 * - loops over its candidate triangle interactions [start,end) using neighborPrefixSum
 * - calls classifySphereTriangleContact(...) to get closest point r_c and contact type
 * - sets contactPoint/contactNormal/overlap
 * - sets cancelFlag=1 and zero springs when no contact
 *
 * Additional "dedup" logic:
 * - For non-face contacts (edge/vertex), it scans other candidates for the same ball to avoid duplicates
 *   (e.g., shared edges/vertices), and may copy spring state from another candidate and cancel this one.
 *
 * @param[in,out] slidingSpring      Sliding spring per candidate (may be copied/zeroed).
 * @param[in,out] rollingSpring      Rolling spring per candidate (may be copied/zeroed).
 * @param[in,out] torsionSpring      Torsion spring per candidate (may be copied/zeroed).
 * @param[out]    contactPoint       Closest/contact point on triangle (global) per candidate.
 * @param[out]    contactNormal      Normal from contact point to sphere center (unit) per candidate.
 * @param[out]    overlap            Overlap = rad_i - |r_i - r_c| per candidate.
 * @param[out]    cancelFlag         1 => ignore this candidate in force stage, 0 => keep.
 *
 * @param[in]  objectPointing        Triangle index per candidate interaction (pointing).
 * @param[in]  position              Ball positions.
 * @param[in]  radius                Ball radii.
 * @param[in]  neighborPrefixSum     Inclusive prefix sum of candidate triangle counts per ball.
 *
 * @param[in]  index0_t              Triangle vertex index0.
 * @param[in]  index1_t              Triangle vertex index1.
 * @param[in]  index2_t              Triangle vertex index2.

 * @param[in]  globalPosition_v      Global vertex positions.
 *
 * @param[in]  numBall               Number of balls (threads map to balls).
 */
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
					if (isZero(lengthSquared(cross(r_c - p01, p11 - p01)))) 
					{
						slidingSpring[idx_c] = slidingSpring[idx_c1];
						rollingSpring[idx_c] = rollingSpring[idx_c1];
						torsionSpring[idx_c] = torsionSpring[idx_c1];
						cancelFlag[idx_c] = 1;
						break;
					}
					if (isZero(lengthSquared(cross(r_c - p11, p21 - p11))))
					{
						slidingSpring[idx_c] = slidingSpring[idx_c1];
						rollingSpring[idx_c] = rollingSpring[idx_c1];
						torsionSpring[idx_c] = torsionSpring[idx_c1];
						cancelFlag[idx_c] = 1;
						break;
					}
					if (isZero(lengthSquared(cross(r_c - p01, p21 - p01)))) 
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
							if (isZero(lengthSquared(r_c - r_c1)))
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
						if (isZero(lengthSquared(r_c - r_c1)))
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
							if (isZero(lengthSquared(r_c - r_c1)))
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

/**
 * @brief Compute ball-wall (triangle mesh) contact force/torque per candidate and accumulate to ball force/torque.
 *
 * Thread per ball idx_i:
 * - skips fixed balls (inverseMass==0)
 * - iterates its candidate contacts [start,end) using neighborPrefixSum
 * - ignores candidates with cancelFlag==1
 * - uses wallIndex_tri[triangle] to map triangle -> wall rigid body index
 * - computes relative contact velocity using ball and wall rigid body kinematics
 * - applies either LinearContact or HertzianMindlinContact (material pair i vs wall material)
 * - writes per-candidate contactForce/contactTorque, updates spring histories
 * - accumulates force[idx_i] and torque[idx_i] (non-atomic, because one thread owns idx_i)
 *
 * Pair meaning:
 * - objectPointed is ball index (not used explicitly here since thread is idx_i)
 * - objectPointing[idx_c] is triangle index
 *
 * @param[in,out] force               Ball forces (accumulated).
 * @param[in,out] torque              Ball torques (accumulated).

 * @param[out]    contactForce        Per-candidate contact force.
 * @param[out]    contactTorque       Per-candidate contact torque.
 * @param[in,out] slidingSpring       Sliding spring history per candidate.
 * @param[in,out] rollingSpring       Rolling spring history per candidate.
 * @param[in,out] torsionSpring       Torsion spring history per candidate.
 * @param[in]     contactPoint        Contact point per candidate.
 * @param[in]     contactNormal       Contact normal per candidate.
 * @param[in]     overlap             Overlap per candidate.
 * @param[in]     objectPointed       Ball index per candidate (redundant with idx_i).
 * @param[in]     objectPointing      Triangle index per candidate.
 * @param[in]     cancelFlag          Candidate validity flag.
 *
 * @param[in]  position               Ball positions.
 * @param[in]  velocity               Ball velocities.
 * @param[in]  angularVelocity        Ball angular velocities.
 * @param[in]  radius                 Ball radii.
 * @param[in]  inverseMass            Ball inverse masses.
 * @param[in]  materialID             Ball material ids.
 * @param[in]  neighborPrefixSum      Candidate counts prefix sum per ball.
 *
 * @param[in]  position_w             Wall rigid body positions.
 * @param[in]  velocity_w             Wall rigid body velocities.
 * @param[in]  angularVelocity_w      Wall rigid body angular velocities.
 * @param[in]  materialID_w           Wall rigid body material ids.

 * @param[in]  wallIndex_tri          Map triangle index -> wall rigid body index.
 *
 * @param[in]  dt                     Time step.
 * @param[in]  numBall                Number of balls.
 */
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
	if (isZero(inverseMass[idx_i])) return;

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
		const int ip = upperTriangularIndex(mat_i, mat_j, para.nMaterials, para.cap);

		const double d_n = getDissipationParam(ip);
		const double mu_s = getFrictionParam(ip, f_MUS);
		const double mu_r = getFrictionParam(ip, f_MUR);
		const double mu_t = getFrictionParam(ip, f_MUT);
		const double k_n = getLinearStiffnessParam(ip, l_KN);
		if (!isZero(k_n))
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
			const double E_ij = getEffectiveYoungsModulusParam(ip);
			const double G_ij = getEffectiveShearModulusParam(ip);

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

/**
 * @brief Compute bonded forces/torques between ball-ball pairs and add into either:
 * - global force/torque arrays (atomic) if not already in contact list, or
 * - contactForce/contactTorque of the matching contact entry if a contact exists.
 *
 * This kernel tries to locate the corresponding contact entry (idx_c) for (idx_i, idx_j) by scanning
 * ball i's candidate contact list (neighborPrefixSum + objectPointing). If found, the bonded
 * contribution is accumulated into contactForce/contactTorque to be later summed.
 *
 * @param[in,out] bondPoint          Bond contact point (global) per bonded interaction.
 * @param[in,out] bondNormal         Bond normal (global) per bonded interaction.
 * @param[in,out] shearForce         Bond shear force history.
 * @param[in,out] bendingTorque      Bond bending torque history.
 * @param[in,out] normalForce        Bond normal force history (scalar).
 * @param[in,out] torsionTorque      Bond torsion torque history (scalar).
 * @param[in,out] maxNormalStress    Peak normal stress for damage check.
 * @param[in,out] maxShearStress     Peak shear stress for damage check.
 * @param[in,out] isBonded           Bond active flag (0 breaks and stays inactive).
 *
 * @param[in,out] contactForce       Per-contact force array (ball-ball contact list).
 * @param[in,out] contactTorque      Per-contact torque array (ball-ball contact list).

 * @param[in,out] force              Global ball force accumulation (atomic when used).
 * @param[in,out] torque             Global ball torque accumulation (atomic when used).
 *
 * @param[in]  objectPointed_b       Bonded pair i indices.
 * @param[in]  objectPointing_b      Bonded pair j indices.
 *
 * @param[in]  contactPoint          Contact list contact points (for matching).
 * @param[in]  contactNormal         Contact list normals (for matching).
 * @param[in]  objectPointing        Contact list pointing indices (ball j) per contact entry.
 *
 * @param[in]  position              Ball positions.
 * @param[in]  velocity              Ball velocities.
 * @param[in]  angularVelocity       Ball angular velocities.
 * @param[in]  radius                Ball radii.
 * @param[in]  materialID            Ball material ids.
 * @param[in]  neighborPrefixSum     Contact list prefix sum (defines per-ball contact range).
 *
 * @param[in]  dt                    Time step.
 * @param[in]  numBondedInteraction  Number of bonded interactions.
 */
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

    const int ip = upperTriangularIndex(materialID[idx_i], materialID[idx_j], para.nMaterials, para.cap);
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
		double3 F_c = F_n * n_ij + F_s;
		double3 T_c = T_t * n_ij + T_b;
		atomicAddDouble3(force, idx_i, F_c);
		atomicAddDouble3(torque, idx_i, T_c + cross(r_c - r_i, F_c));
		atomicAddDouble3(force, idx_j, -F_c);
		atomicAddDouble3(torque, idx_j, -T_c + cross(r_c - r_j, -F_c));
		return;
	}

	contactForce[idx_c] += F_n * n_ij + F_s;
	contactTorque[idx_c] += T_t * n_ij + T_b;
}

/**
 * @brief Compute level-set boundary-node vs level-set particle contact forces and accumulate to particle forces/torques.
 *
 * Pair meaning for this interaction list:
 * - objectPointed[idx]  = boundary node index (bNode)
 * - objectPointing[idx] = particle index j (the "other" LS particle)
 * - particleID_bNode[objectPointed[idx]] gives particle index i that owns that boundary node
 *
 * This kernel computes a (linear) tangential/normal contact for LS particles and updates sliding spring history.
 * Forces are accumulated to force_p/torque_p using atomic adds.
 *
 * @param[out]    contactForce      Per-interaction contact force.
 * @param[in,out] slidingSpring     Sliding spring history per interaction.

 * @param[in,out] force_p           Particle forces (accumulated, atomic).
 * @param[in,out] torque_p          Particle torques (accumulated, atomic).
 *
 * @param[in]  contactPoint         Contact point (global) per interaction.
 * @param[in]  contactNormal        Contact normal (global, unit) per interaction.
 * @param[in]  overlap              Overlap per interaction.
 * @param[in]  objectPointed        Boundary node index per interaction.
 * @param[in]  objectPointing       Particle index j per interaction.

 * @param[in]  particleID_bNode     Owner particle id for each boundary node index.
 *
 * @param[in]  position_p           Particle positions (global).
 * @param[in]  velocity_p           Particle velocities.
 * @param[in]  angularVelocity_p    Particle angular velocities.
 * @param[in]  materialID_p         Particle material ids.

 * @param[in]  dt                   Time step.
 * @param[in]  numInteraction       Number of interactions.
 */
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

	double3 F_c = make_double3(0., 0., 0.);
	double3 epsilon_s = slidingSpring[idx];

    const int mat_i = materialID_p[idx_i];
	const int mat_j = materialID_p[idx_j];
    const int ip = upperTriangularIndex(mat_i, mat_j, para.nMaterials, para.cap);

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

/**
 * @brief Compute bonded force/torque for level-set particle pairs (bond endpoints are stored in each particle local frame).
 *
 * Pair meaning:
 * - objectPointed_b[idx]  = particle i
 * - objectPointing_b[idx] = particle j
 *
 * Local points:
 * - localPointA_b[idx] is bond endpoint in particle i local frame
 * - localPointB_b[idx] is bond endpoint in particle j local frame
 *
 * This kernel:
 * - reconstructs global bond endpoints rb_i, rb_j using quaternion orientation + position
 * - computes bond center r_c and current bond direction n_ij
 * - evaluates bond constitutive law ParallelBondedContact2(...)
 * - accumulates forces and torques to both particles using atomic adds
 *
 * @param[out]    bondPoint          Bond center point (global).
 * @param[in,out] bondNormal         Bond direction (unit).
 * @param[in,out] shearForce         Bond shear force history.
 * @param[in,out] bendingTorque      Bond bending torque history.
 * @param[in,out] normalForce        Bond normal force history.
 * @param[in,out] torsionTorque      Bond torsion torque history.
 * @param[in,out] maxNormalStress    Peak normal stress.
 * @param[in,out] maxShearStress     Peak shear stress.
 * @param[in,out] isBonded           Bond active flag.
 *
 * @param[in,out] force              Particle forces (accumulated, atomic).
 * @param[in,out] torque             Particle torques (accumulated, atomic).
 *
 * @param[in]  localPointA_b         Local endpoint on particle i.
 * @param[in]  localPointB_b         Local endpoint on particle j.
 * @param[in]  length_b              Reference bond length (scalar per bond).
 * @param[in]  radius_b              Bond radius (scalar per bond).
 * @param[in]  objectPointed_b       Particle i indices.
 * @param[in]  objectPointing_b      Particle j indices.
 *
 * @param[in]  position              Particle positions.
 * @param[in]  velocity              Particle velocities.
 * @param[in]  angularVelocity       Particle angular velocities.
 * @param[in]  orientation           Particle orientations (quaternions).
 * @param[in]  materialID            Particle material ids.

 * @param[in]  dt                    Time step.
 * @param[in]  numBondedInteraction  Number of bonds.
 */
__global__ void addLevelSetParticleBondedForceTorqueKernel(double3* bondPoint,
double3* bondNormal,
double3* shearForce, 
double3* bendingTorque,
double* normalForce, 
double* torsionTorque, 
double* maxNormalStress,
double* maxShearStress,
int* isBonded, 

double3* force, 
double3* torque, 

const double3* localPointA_b,
const double3* localPointB_b,
const double* length_b,
const double* radius_b,
const int* objectPointed_b,
const int* objectPointing_b,

const double3* position, 
const double3* velocity, 
const double3* angularVelocity, 
const quaternion* orientation,
const int* materialID, 

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
	const double3 n_ij0 = bondNormal[idx];

    const double3 r_i = position[idx_i];
	const double3 r_j = position[idx_j];
	const double3 rb_i = rotateVectorByQuaternion(orientation[idx_i], localPointA_b[idx]) + r_i;
	const double3 rb_j = rotateVectorByQuaternion(orientation[idx_j], localPointB_b[idx]) + r_j;
	const double3 r_c = 0.5 * (rb_i + rb_j);
	const double3 n_ij = normalize(rb_i - rb_j);
	const double3 v_i = velocity[idx_i];
	const double3 v_j = velocity[idx_j];
	const double3 w_i = angularVelocity[idx_i];
	const double3 w_j = angularVelocity[idx_j];
	const double3 v_c_ij = v_i + cross(w_i, r_c - r_i) - (v_j + cross(w_j, r_c - r_j));

	bondPoint[idx] = r_c;
    bondNormal[idx] = n_ij;

	const int ip = upperTriangularIndex(materialID[idx_i], materialID[idx_j], para.nMaterials, para.cap);
    const double E = getBondParam(ip, b_E);
    const double k_n_k_s = getBondParam(ip, b_KNKS);
	const double sigma_s = getBondParam(ip, b_SIGMAS);
    const double C = getBondParam(ip, b_C);
    const double mu = getBondParam(ip, b_MU);

	const double G = E / k_n_k_s;
	const double l_b = length_b[idx];
	const double rad_b = radius_b[idx];
	const double A_b = pi() * rad_b * rad_b;
	const double I_b = pi() * rad_b * rad_b * rad_b * rad_b / 4.;
	const double J_b = 2. * I_b;
	const double k_n = E * A_b / l_b;
    const double k_s = 12. * E * I_b / (l_b * l_b * l_b);
    const double k_b = E * I_b / l_b;
    const double k_t = G * J_b / l_b;

	double F_n = normalForce[idx];
	double3 F_s = shearForce[idx];
	double T_t = torsionTorque[idx];
	double3 T_b = bendingTorque[idx];
	double sigma_max = maxNormalStress[idx];
	double tau_max = maxShearStress[idx];

	isBonded[idx] = ParallelBondedContactForLevelSetParticle(F_n, T_t, F_s, T_b, sigma_max, tau_max,
	n_ij0, n_ij, v_c_ij, w_i, w_j, dt, rad_b, k_n, k_s, k_b, k_t, 
	sigma_s, C, mu);

	normalForce[idx] = F_n;
	shearForce[idx] = F_s;
	torsionTorque[idx] = T_t;
	bendingTorque[idx] = T_b;
	maxNormalStress[idx] = sigma_max;
	maxShearStress[idx] = tau_max;

    double3 F_c = F_n * n_ij + F_s;
	double3 T_c = T_t * n_ij + T_b;
	atomicAddDouble3(force, idx_i, F_c);
	atomicAddDouble3(torque, idx_i, T_c + cross(r_c - r_i, F_c));
	atomicAddDouble3(force, idx_j, -F_c);
	atomicAddDouble3(torque, idx_j, -T_c + cross(r_c - r_j, -F_c));
}

/**
 * @brief Apply level-set wall reaction to each level-set boundary node (8-node grid) and accumulate to owning particle.
 *
 * Each thread handles one boundary node:
 * - idx_i = particleID_bNode[idx] is owner particle
 * - compute boundary node global position using orientation + particle position
 * - trilinearly interpolate wall phi from LSFV_w (grid defined by origin/spacing/size)
 * - if overlap = -phi > 0, compute gradient-based normal and apply a linear penalty force F = k_n * overlap * n
 * - accumulate to force_p / torque_p (atomic)
 *
 * @param[in,out] force_p               Particle forces (accumulated, atomic).
 * @param[in,out] torque_p              Particle torques (accumulated, atomic).
 *
 * @param[in]  position_p               Particle positions.
 * @param[in]  orientation_p            Particle orientations.
 * @param[in]  inverseMass_p            Particle inverse masses (<=0 means fixed).
 * @param[in]  materialID_p             Particle material ids.
 *
 * @param[in]  localPosition_bNode      Boundary node local positions (in owner particle local frame).
 * @param[in]  particleID_bNode         Owner particle index for each boundary node.
 *
 * @param[in]  LSFV_w                   Wall grid level-set values (phi) stored in linearIndex3D order.
 * @param[in]  gridSpacing_w            Wall grid spacing.
 * @param[in]  gridNodeGlobalOrigin_w   Wall grid origin in global coordinates.
 * @param[in]  gridNodeSize_w           Wall grid node counts (x,y,z).
 *
 * @param[in]  numBoundaryNode          Number of boundary nodes.
 */
__global__ void addLevelSetParticleWallForce(double3* force_p,
double3* torque_p,

const double3* position_p,
const quaternion* orientation_p,
const double* inverseMass_p,
const int* materialID_p,

const double3* localPosition_bNode, 
const int* particleID_bNode, 

const double* LSFV_w, 
const double gridSpacing_w,
const double3 gridNodeGlobalOrigin_w,
const int3 gridNodeSize_w,

const size_t numBoundaryNode)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoundaryNode) return;

    const int idx_i = particleID_bNode[idx];
	if (inverseMass_p[idx_i] <= 0.) return;

    const double3 r_i = position_p[idx_i];
    const double3 globalPosition_idx = rotateVectorByQuaternion(orientation_p[idx_i], localPosition_bNode[idx]) + r_i;
    const double gx = (globalPosition_idx.x - gridNodeGlobalOrigin_w.x) / gridSpacing_w;
    const double gy = (globalPosition_idx.y - gridNodeGlobalOrigin_w.y) / gridSpacing_w;
    const double gz = (globalPosition_idx.z - gridNodeGlobalOrigin_w.z) / gridSpacing_w;

    int i0 = (int)floor(gx);
    int j0 = (int)floor(gy);
    int k0 = (int)floor(gz);

    if (i0 < 0) return;
    if (j0 < 0) return;
    if (k0 < 0) return;

    if (i0 >= gridNodeSize_w.x - 1) return;
    if (j0 >= gridNodeSize_w.y - 1) return;
    if (k0 >= gridNodeSize_w.z - 1) return;

    const int i1 = i0 + 1;
    const int j1 = j0 + 1;
    const int k1 = k0 + 1;

    const double x = gx - static_cast<double>(i0);
    const double y = gy - static_cast<double>(j0);
    const double z = gz - static_cast<double>(k0);

	const double phi000 = LSFV_w[linearIndex3D(make_int3(i0, j0, k0), gridNodeSize_w)];
	const double phi100 = LSFV_w[linearIndex3D(make_int3(i1, j0, k0), gridNodeSize_w)];
	const double phi010 = LSFV_w[linearIndex3D(make_int3(i0, j1, k0), gridNodeSize_w)];
	const double phi110 = LSFV_w[linearIndex3D(make_int3(i1, j1, k0), gridNodeSize_w)];
	const double phi001 = LSFV_w[linearIndex3D(make_int3(i0, j0, k1), gridNodeSize_w)];
	const double phi101 = LSFV_w[linearIndex3D(make_int3(i1, j0, k1), gridNodeSize_w)];
	const double phi011 = LSFV_w[linearIndex3D(make_int3(i0, j1, k1), gridNodeSize_w)];
	const double phi111 = LSFV_w[linearIndex3D(make_int3(i1, j1, k1), gridNodeSize_w)];

    const double ovelap = -interpolateLevelSetFunctionValue(x, 
    y, 
    z, 
    phi000,
    phi100,
    phi010,
    phi110,
    phi001,
    phi101, 
    phi011,
    phi111);

    if (ovelap > 0.) 
    {
        double3 n_c = interpolateLevelSetFunctionGradient(x, 
        y, 
        z, 
        phi000,
        phi100,
        phi010,
        phi110,
        phi001,
        phi101, 
        phi011,
        phi111);

		n_c = normalize(n_c);
		const double3 r_c = globalPosition_idx + 0.5 * n_c * ovelap;

		const int ip = upperTriangularIndex(materialID_p[idx_i], materialID_p[idx_i], para.nMaterials, para.cap);
		const double k_n = getLinearStiffnessParam(ip, l_KN);
		double3 F_c = k_n * ovelap * n_c;

		atomicAddDouble3(force_p, idx_i, F_c);
		atomicAddDouble3(torque_p, idx_i, cross(r_c - r_i, F_c));
    }
}

/**
 * @brief Sum per-interaction forces/torques for "pointed" objects using neighborPrefixSum ranges and add to global force/torque.
 *
 * Assumes one thread per object idx_i. The interactions belonging to idx_i are:
 *   k in [ (idx_i>0 ? neighborPrefixSum[idx_i-1] : 0), neighborPrefixSum[idx_i] )
 *
 * @param[in,out] force             Global force per object (accumulated).
 * @param[in,out] torque            Global torque per object (accumulated).
 * @param[in]     position          Object positions (used for torque arm).
 * @param[in]     neighborPrefixSum Inclusive prefix sum of per-object interaction counts.

 * @param[in]     contactForce      Per-interaction force (applied on the pointed object).
 * @param[in]     contactTorque     Per-interaction torque (about contact point).
 * @param[in]     contactPoint      Per-interaction contact point (global).

 * @param[in]     num               Number of pointed objects.
 */
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

/**
 * @brief Sum per-interaction forces/torques for "pointing" objects using interactionStart/End list and add to global force/torque.
 *
 * For each object idx_i:
 * - iterate k in [interactionStart[idx_i], interactionEnd[idx_i]) if start>=0
 * - map k -> interaction index k1 = neighborPairHashIndex[k]
 * - subtract the corresponding contactForce/contactTorque (Newton's third law)
 *
 * @param[in,out] force               Global force per object (accumulated).
 * @param[in,out] torque              Global torque per object (accumulated).

 * @param[in]     position            Object positions (used for torque arm).
 * @param[in]     interactionStart    Start offset per pointing object, -1 if none.
 * @param[in]     interactionEnd      End offset per pointing object.

 * @param[in]     contactForce        Per-interaction force (stored on pointed object side).
 * @param[in]     contactTorque       Per-interaction torque.
 * @param[in]     contactPoint        Per-interaction contact point.
 * @param[in]     neighborPairHashIndex Mapping from compact adjacency list entry to interaction index.

 * @param[in]     num                 Number of pointing objects.
 */
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

void launchAddBallWallContactForceTorque(double3* position, 
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

extern "C" void launchAddLevelSetParticleBondedForceTorque(double3* bondPoint,
double3* bondNormal,
double3* shearForce,
double3* bendingTorque,
double* normalForce,
double* torsionTorque,
double* maxNormalStress,
double* maxShearStress,
int* isBonded,
const double3* localPointA_b,
const double3* localPointB_b,
const double* length_b,
const double* radius_b,
const int* objectPointed_b,
const int* objectPointing_b,

double3* force,
double3* torque,
const double3* position,
const double3* velocity,
const double3* angularVelocity,
const quaternion* orientation,
const int* materialID,

const double dt,

const size_t numBondedInteraction,
const size_t gridD,
const size_t blockD,
cudaStream_t stream)
{
    addLevelSetParticleBondedForceTorqueKernel<<<gridD, blockD, 0, stream>>>(bondPoint,
    bondNormal,
    shearForce,
    bendingTorque,
    normalForce,
    torsionTorque,
    maxNormalStress,
    maxShearStress,
    isBonded,
    force,
    torque,
    localPointA_b,
    localPointB_b,
    length_b,
    radius_b,
    objectPointed_b,
    objectPointing_b,
    position,
    velocity,
    angularVelocity,
    orientation,
    materialID,
    dt,
    numBondedInteraction);
}

extern "C" void launchAddLevelSetParticleWallForce(double3* force_p,
double3* torque_p,
const double3* position_p,
const quaternion* orientation_p,
const double* inverseMass_p,
const int* materialID_p,

const double3* localPosition_bNode,
const int* particleID_bNode,

const double* LSFV_w,
const double gridSpacing_w,
const double3 gridNodeGlobalOrigin_w,
const int3 gridNodeSize_w,

const size_t numBoundaryNode,
const size_t gridD,
const size_t blockD,
cudaStream_t stream)
{
    addLevelSetParticleWallForce<<<gridD, blockD, 0, stream>>>(force_p,
	torque_p,

    position_p,
    orientation_p,
    inverseMass_p,
    materialID_p,

    localPosition_bNode,
    particleID_bNode,

    LSFV_w,
    gridSpacing_w,
    gridNodeGlobalOrigin_w,
    gridNodeSize_w,

    numBoundaryNode);
}