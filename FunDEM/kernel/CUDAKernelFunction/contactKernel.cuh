#pragma once
#include "myUtility/myQua.h"

__device__ __forceinline__ int ParallelBondedContact(double& bondNormalForce, 
double& bondTorsionalTorque, 
double3& bondShearForce, 
double3& bondBendingTorque,
double& maxNormalStress,
double& maxShearStress,
const double3 contactNormalPrev,
const double3 contactNormal,
const double3 relativeVelocityAtContact,
const double3 angularVelocityA,
const double3 angularVelocityB,
const double radiusA,
const double radiusB,
const double timeStep,
const double bondMultiplier,
const double bondElasticModulus,
const double bondStiffnessRatioNormalToShear,
const double bondTensileStrength,
const double bondCohesion,
const double bondFrictionCoefficient)
{
	const double3 n0 = normalize(contactNormalPrev);
	const double3 n1 = normalize(contactNormal);
	const double3 axis1 = cross(n0, n1);
	const double sinTheta1 = length(axis1);
	bondShearForce = rotateVectorAxisSin(bondShearForce, axis1, sinTheta1);
	bondBendingTorque = rotateVectorAxisSin(bondBendingTorque, axis1, sinTheta1);
	const double3 theta2 = dot(0.5 * (angularVelocityA + angularVelocityB) * timeStep, contactNormal) * contactNormal;
	bondShearForce = rotateVector(bondShearForce, theta2);
	bondBendingTorque = rotateVector(bondBendingTorque, theta2);
	const double minRadius = radiusA < radiusB ? radiusA : radiusB;
	const double bondRadius = bondMultiplier * minRadius;
	const double bondArea = bondRadius * bondRadius * pi();// cross-section area of beam of the bond
	const double bondInertiaMoment = bondRadius * bondRadius * bondRadius * bondRadius / 4. * pi();// inertia moment
	const double bondPolarInertiaMoment = 2 * bondInertiaMoment;// polar inertia moment
	const double normalStiffnessUnitArea = bondElasticModulus / (radiusA + radiusB);
	const double shearStiffnessUnitArea = normalStiffnessUnitArea / bondStiffnessRatioNormalToShear;

	const double3 normalTranslationIncrement = dot(relativeVelocityAtContact, contactNormal) * contactNormal * timeStep;
	const double3 tangentialTranslationIncrement = relativeVelocityAtContact * timeStep - normalTranslationIncrement;
	bondNormalForce -= dot(normalTranslationIncrement * normalStiffnessUnitArea * bondArea, contactNormal);
	bondShearForce -= tangentialTranslationIncrement * shearStiffnessUnitArea * bondArea;
	const double3 relativeAngularVelocity = angularVelocityA - angularVelocityB;
	const double3 normalRotationIncrement = dot(relativeAngularVelocity, contactNormal) * contactNormal * timeStep;
	const double3 tangentialRotationIncrement = relativeAngularVelocity * timeStep - normalRotationIncrement;
	bondTorsionalTorque -= dot(normalRotationIncrement * shearStiffnessUnitArea * bondPolarInertiaMoment, contactNormal);
	bondBendingTorque -= tangentialRotationIncrement * normalStiffnessUnitArea * bondInertiaMoment;

	maxNormalStress = -bondNormalForce / bondArea + length(bondBendingTorque) / bondInertiaMoment * bondRadius;// maximum tension stress
	maxShearStress = length(bondShearForce) / bondArea + fabs(bondTorsionalTorque) / bondPolarInertiaMoment * bondRadius;// maximum shear stress

	int isBonded = 1;
	if (bondTensileStrength > 0 && maxNormalStress > bondTensileStrength)
	{
		isBonded = 0;
	}
	else if (bondCohesion > 0 && maxShearStress > bondCohesion - bondFrictionCoefficient * maxNormalStress)
	{
		isBonded = 0;
	}
	return isBonded;
}

__device__ __forceinline__ int ParallelBondedContact2(double& bondNormalForce, 
double& bondTorsionalTorque, 
double3& bondShearForce, 
double3& bondBendingTorque,
double& maxNormalStress,
double& maxShearStress,
const double3 contactNormalPrev,
const double3 contactNormal,
const double3 relativeVelocityAtContact,
const double3 angularVelocityA,
const double3 angularVelocityB,
const double timeStep,
const double bondRadius,
const double normalStiffness,
const double shearStiffness,
const double bendingStiffness,
const double torsionStiffness,
const double bondTensileStrength,
const double bondCohesion,
const double bondFrictionCoefficient)
{
	const double3 n0 = normalize(contactNormalPrev);
	const double3 n1 = normalize(contactNormal);
	const double3 axis1 = cross(n0, n1);
	const double sinTheta1 = length(axis1);
	bondShearForce = rotateVectorAxisSin(bondShearForce, axis1, sinTheta1);
	bondBendingTorque = rotateVectorAxisSin(bondBendingTorque, axis1, sinTheta1);
	const double3 theta2 = dot(0.5 * (angularVelocityA + angularVelocityB) * timeStep, contactNormal) * contactNormal;
	bondShearForce = rotateVector(bondShearForce, theta2);
	bondBendingTorque = rotateVector(bondBendingTorque, theta2);

	const double bondArea = bondRadius * bondRadius * pi();// cross-section area of beam of the bond
	const double bondInertiaMoment = bondRadius * bondRadius * bondRadius * bondRadius / 4. * pi();// inertia moment
	const double bondPolarInertiaMoment = 2 * bondInertiaMoment;// polar inertia moment

	const double3 normalTranslationIncrement = dot(relativeVelocityAtContact, contactNormal) * contactNormal * timeStep;
	const double3 tangentialTranslationIncrement = relativeVelocityAtContact * timeStep - normalTranslationIncrement;
	bondNormalForce -= dot(normalTranslationIncrement * normalStiffness, contactNormal);
	bondShearForce -= tangentialTranslationIncrement * shearStiffness;
	const double3 relativeAngularVelocity = angularVelocityA - angularVelocityB;
	const double3 normalRotationIncrement = dot(relativeAngularVelocity, contactNormal) * contactNormal * timeStep;
	const double3 tangentialRotationIncrement = relativeAngularVelocity * timeStep - normalRotationIncrement;
	bondTorsionalTorque -= dot(normalRotationIncrement * torsionStiffness, contactNormal);
	bondBendingTorque -= tangentialRotationIncrement * bendingStiffness;

	maxNormalStress = -bondNormalForce / bondArea + length(bondBendingTorque) / bondInertiaMoment * bondRadius;// maximum tension stress
	maxShearStress = length(bondShearForce) / bondArea + fabs(bondTorsionalTorque) / bondPolarInertiaMoment * bondRadius;// maximum shear stress

	int isBonded = 1;
	if (bondTensileStrength > 0 && maxNormalStress > bondTensileStrength)
	{
		isBonded = 0;
	}
	else if (bondCohesion > 0 && maxShearStress > bondCohesion - bondFrictionCoefficient * maxNormalStress)
	{
		isBonded = 0;
	}
	return isBonded;
}

static __device__ __forceinline__ double3 integrateSlidingOrRollingSpring(const double3 springPrev, 
const double3 springVelocity, 
const double3 contactNormal, 
const double3 normalContactForce, 
const double frictionCoefficient, 
const double stiffness, 
const double dampingCoefficient, 
const double timeStep)
{
	double3 spring = make_double3(0., 0., 0.);
	if (frictionCoefficient > 0. && stiffness > 0.)
	{
		double3 springPrev1 = springPrev - dot(springPrev, contactNormal) * contactNormal;
		double absoluteSpringPrev1 = length(springPrev1);
		if (absoluteSpringPrev1 > 1e-10)
		{
			springPrev1 *= length(springPrev) / absoluteSpringPrev1;
		}
		spring = springPrev1 + springVelocity * timeStep;
		double3 springForce = -stiffness * spring - dampingCoefficient * springVelocity;
		double absoluteSpringForce = length(springForce);
		double absoluteNormalContactForce = length(normalContactForce);
		if (absoluteSpringForce > frictionCoefficient * absoluteNormalContactForce)
		{
			double ratio = frictionCoefficient * absoluteNormalContactForce / absoluteSpringForce;
			springForce *= ratio;
			spring = -(springForce + dampingCoefficient * springVelocity) / stiffness;
		}
	}
	return spring;
}

static __device__ __forceinline__ double3 integrateTorsionSpring(const double3 springPrev, 
const double3 torsionRelativeVelocity, 
const double3 contactNormal, 
const double3 normalContactForce, 
const double frictionCoefficient, 
const double stiffness, 
const double dampingCoefficient, 
const double timeStep)
{
	double3 spring = make_double3(0., 0., 0.);
	if (frictionCoefficient > 0. && stiffness > 0.)
	{
		spring = dot(springPrev + torsionRelativeVelocity * timeStep, contactNormal) * contactNormal;
		double3 springForce = -stiffness * spring - dampingCoefficient * torsionRelativeVelocity;
		double absoluteSpringForce = length(springForce);
		double absoluteNormalContactForce = length(normalContactForce);
		if (absoluteSpringForce > frictionCoefficient * absoluteNormalContactForce)
		{
			double ratio = frictionCoefficient * absoluteNormalContactForce / absoluteSpringForce;
			springForce *= ratio;
			spring = -(springForce + dampingCoefficient * torsionRelativeVelocity) / stiffness;
		}
	}
	return spring;
}

__device__ __forceinline__ void LinearContact(double3& contactForce, 
double3& contactTorque, 
double3& slidingSpring, 
double3& rollingSpring, 
double3& torsionSpring,
const double3 relativeVelocityAtContact,
const double3 relativeAngularVelocityAtContact,
const double3 contactNormal,
const double normalOverlap,
const double effectiveMass,
const double effectiveRadius,
const double timeStep,
const double normalStiffness,
const double slidingStiffness,
const double rollingStiffness,
const double torsionStiffness,
const double dissipation,
const double slidingFrictionCoefficient,
const double rollingFrictionCoefficient,
const double torsionFrictionCoefficient)
{
	if (normalOverlap > 0)
	{
		const double normalDampingCoefficient = 2. * dissipation * sqrt(effectiveMass * normalStiffness);
		const double slidingDampingCoefficient = 2. * dissipation * sqrt(effectiveMass * slidingStiffness);
		const double rollingDampingCoefficient = 2. * dissipation * sqrt(effectiveMass * rollingStiffness);
		const double torsionDampingCoefficient = 2. * dissipation * sqrt(effectiveMass * torsionStiffness);

		const double3 normalRelativeVelocityAtContact = dot(relativeVelocityAtContact, contactNormal) * contactNormal;
		const double3 normalContactForce = normalStiffness * normalOverlap * contactNormal - normalRelativeVelocityAtContact * normalDampingCoefficient;

		const double3 slidingRelativeVelocity = relativeVelocityAtContact - normalRelativeVelocityAtContact;
		slidingSpring = integrateSlidingOrRollingSpring(slidingSpring, 
		slidingRelativeVelocity, 
		contactNormal, 
		normalContactForce, 
		slidingFrictionCoefficient, 
		slidingStiffness, 
		slidingDampingCoefficient, 
		timeStep);
		const double3 slidingForce = -slidingStiffness * slidingSpring - slidingDampingCoefficient * slidingRelativeVelocity;

		const double3 rollingRelativeVelocity = -effectiveRadius * cross(contactNormal, relativeAngularVelocityAtContact);
		rollingSpring = integrateSlidingOrRollingSpring(rollingSpring, 
		rollingRelativeVelocity, 
		contactNormal, 
		normalContactForce, 
		rollingFrictionCoefficient, 
		rollingStiffness, 
		rollingDampingCoefficient, 
		timeStep);
		const double3 rollingForce = -rollingStiffness * rollingSpring - rollingDampingCoefficient * rollingRelativeVelocity;
		const double3 rollingTorque = effectiveRadius * cross(contactNormal, rollingForce);

		const double effectiveDiameter = 2 * effectiveRadius;
		const double3 torsionRelativeVelocity = effectiveDiameter * dot(relativeAngularVelocityAtContact, contactNormal) * contactNormal;
		torsionSpring = integrateTorsionSpring(torsionSpring, 
		torsionRelativeVelocity, 
		contactNormal, 
		normalContactForce, 
		torsionFrictionCoefficient, 
		torsionStiffness, 
		torsionDampingCoefficient, 
		timeStep);
		const double3 torsionForce = -torsionStiffness * torsionSpring - torsionDampingCoefficient * torsionRelativeVelocity;
		const double3 torsionTorque = effectiveDiameter * torsionForce;

		contactForce = normalContactForce + slidingForce;
		contactTorque = rollingTorque + torsionTorque;
	}
	else
	{
		slidingSpring = make_double3(0., 0., 0.);
		rollingSpring = make_double3(0., 0., 0.);
		torsionSpring = make_double3(0., 0., 0.);
	}
}

__device__ __forceinline__ void LinearContactForLevelSetParticle(double3& contactForce, 
double3& slidingSpring, 
const double3 relativeVelocityAtContact,
const double3 relativeAngularVelocityAtContact,
const double3 contactNormal,
const double normalOverlap,
const double timeStep,
const double normalStiffness,
const double slidingStiffness,
const double slidingFrictionCoefficient)
{
	if (normalOverlap > 0)
	{
		const double3 normalRelativeVelocityAtContact = dot(relativeVelocityAtContact, contactNormal) * contactNormal;
		const double3 normalContactForce = normalStiffness * normalOverlap * contactNormal;

		const double3 slidingRelativeVelocity = relativeVelocityAtContact - normalRelativeVelocityAtContact;
		slidingSpring = integrateSlidingOrRollingSpring(slidingSpring, 
		slidingRelativeVelocity, 
		contactNormal, 
		normalContactForce, 
		slidingFrictionCoefficient, 
		slidingStiffness, 
		0.0, 
		timeStep);
		const double3 slidingForce = -slidingStiffness * slidingSpring;

		contactForce = normalContactForce + slidingForce;
	}
	else
	{
		slidingSpring = make_double3(0., 0., 0.);
	}
}

__device__ __forceinline__ void HertzianMindlinContact(double3& contactForce, 
double3& contactTorque, 
double3& slidingSpring, 
double3& rollingSpring, 
double3& torsionSpring,
const double3 relativeVelocityAtContact,
const double3 relativeAngularVelocityAtContact,
const double3 contactNormal,
const double normalOverlap,
const double effectiveMass,
const double effectiveRadius,
const double timeStep,
const double effectiveElasticModulus,
const double effectiveShearModulus,
const double dissipation,
const double slidingFrictionCoefficient,
const double rollingFrictionCoefficient,
const double torsionFrictionCoefficient)
{
	if (normalOverlap > 0)
	{
		const double normalStiffness = 4. / 3. * effectiveElasticModulus * sqrt(effectiveRadius * normalOverlap);
		const double slidingStiffness = 8. * effectiveShearModulus * sqrt(effectiveRadius * normalOverlap);
		const double normalDampingCoefficient = 2. * sqrt(5. / 6.) * dissipation * sqrt(effectiveMass * normalStiffness);
		const double slidingDampingCoefficient = 2. * sqrt(5. / 6.) * dissipation * sqrt(effectiveMass * slidingStiffness);

		const double rollingStiffness = slidingStiffness;
		const double torsionStiffness = slidingStiffness;
		const double rollingDampingCoefficient = slidingDampingCoefficient;
		const double torsionDampingCoefficient = slidingDampingCoefficient;

		const double3 normalRelativeVelocityAtContact = dot(relativeVelocityAtContact, contactNormal) * contactNormal;
		const double3 normalContactForce = normalStiffness * normalOverlap * contactNormal - normalRelativeVelocityAtContact * normalDampingCoefficient;

		const double3 slidingRelativeVelocity = relativeVelocityAtContact - normalRelativeVelocityAtContact;
		slidingSpring = integrateSlidingOrRollingSpring(slidingSpring, 
		slidingRelativeVelocity, 
		contactNormal, 
		normalContactForce, 
		slidingFrictionCoefficient, 
		slidingStiffness, 
		slidingDampingCoefficient, 
		timeStep);
		const double3 slidingForce = -slidingStiffness * slidingSpring - slidingDampingCoefficient * slidingRelativeVelocity;

		const double3 rollingRelativeVelocity = -effectiveRadius * cross(contactNormal, relativeAngularVelocityAtContact);
		rollingSpring = integrateSlidingOrRollingSpring(rollingSpring, 
		rollingRelativeVelocity, 
		contactNormal, 
		normalContactForce, 
		rollingFrictionCoefficient, 
		rollingStiffness, 
		rollingDampingCoefficient, 
		timeStep);
		const double3 rollingForce = -rollingStiffness * rollingSpring - rollingDampingCoefficient * rollingRelativeVelocity;
		const double3 rollingTorque = effectiveRadius * cross(contactNormal, rollingForce);

		const double effectiveDiameter = 2 * effectiveRadius;
		const double3 torsionRelativeVelocity = effectiveDiameter * dot(relativeAngularVelocityAtContact, contactNormal) * contactNormal;
		torsionSpring = integrateTorsionSpring(torsionSpring, 
		torsionRelativeVelocity, 
		contactNormal, 
		normalContactForce, 
		torsionFrictionCoefficient, 
		torsionStiffness, 
		torsionDampingCoefficient, 
		timeStep);
		const double3 torsionForce = -torsionStiffness * torsionSpring - torsionDampingCoefficient * torsionRelativeVelocity;
		const double3 torsionTorque = effectiveDiameter * torsionForce;

		contactForce = normalContactForce + slidingForce;
		contactTorque = rollingTorque + torsionTorque;
	}
	else
	{
		slidingSpring = make_double3(0., 0., 0.);
		rollingSpring = make_double3(0., 0., 0.);
		torsionSpring = make_double3(0., 0., 0.);
	}
}

enum class SphereTriangleContactType
{
    None,
    Face,
    Edge,
    Vertex
};

// ------------------------------------------------------------
// Small helpers
// ------------------------------------------------------------
__device__ __forceinline__ double clamp01(const double x)
{
    return (x < 0.0) ? 0.0 : ((x > 1.0) ? 1.0 : x);
}

__device__ __forceinline__ double dist2(const double3& a, const double3& b)
{
    const double3 d = a - b;
    return dot(d, d);
}

// Closest point on segment [a,b] to p. Returns q and outputs t in [0,1].
__device__ __forceinline__ double3 closestPointOnSegment(const double3& a,
const double3& b,
const double3& p,
double& tOut)
{
    const double3 ab  = b - a;
    const double ab2 = dot(ab, ab);
    if (ab2 <= 1e-30)
    {
        tOut = 0.0;
        return a;
    }
    tOut = clamp01(dot(p - a, ab) / ab2);
    return a + ab * tOut;
}

// ------------------------------------------------------------
// Edge-contact test (strictly edge interior, not vertices)
// Optimized: fewer temporaries + early exits + scale-aware eps
// ------------------------------------------------------------
__device__ __forceinline__ bool isSphereEdgeContact(const double3& p0,
const double3& p1,
const double3& c,
const double  r)
{
    const double3 e = p1 - p0;
    const double  e2 = dot(e, e);
    if (e2 <= 1e-30) return false;

    // Project center onto edge line: t in R
    const double3 v = c - p0;
    const double  t = dot(v, e) / e2;

    // Interior only (exclude endpoints) with a scale-aware margin.
    // A good choice: epsilon in parameter space roughly corresponds to
    // a small physical length along the edge.
    const double edgeLen = sqrt(e2);
    const double physEps = 1e-12 * fmax(1.0, edgeLen); // physical tiny length
    const double tEps = physEps / edgeLen; // convert to [0,1] param
    // (edgeLen>0 ensured above)
    if (t <= tEps || t >= 1.0 - tEps) return false;

    // Closest point q = p0 + t*e
    const double3 q = p0 + e * t;

    // Distance^2 check (inline, no helper call)
    const double3 d  = c - q;
    const double  d2 = dot(d, d);

    const double r2 = r * r;

    // Scale-aware tolerance on d2
    const double eps2 = 1e-12 * fmax(1.0, r2);

    return d2 <= r2 + eps2;
}

// ------------------------------------------------------------
// Main classifier: returns contact type and closestPoint.
// - Uses Ericson region tests for non-degenerate triangle.
// - Degenerate triangle handled as 3 segments.
// ------------------------------------------------------------
__device__ __forceinline__ SphereTriangleContactType classifySphereTriangleContact(const double3& sphereCenter,
const double  sphereRadius,
const double3& v0,
const double3& v1,
const double3& v2,
double3& closestPoint)
{
    const double r2 = sphereRadius * sphereRadius;
    const double eps2 = 1e-12 * fmax(1.0, r2);

    const double3 ab = v1 - v0;
    const double3 ac = v2 - v0;

    const double3 n = cross(ab, ac);
    const double area2 = dot(n, n);

    // --------------------------------------------------------
    // Degenerate triangle -> treat as 3 segments
    // --------------------------------------------------------
    if (area2 <= 1e-20)
    {
        double t01, t02, t12;
        const double3 q01 = closestPointOnSegment(v0, v1, sphereCenter, t01);
        const double3 q02 = closestPointOnSegment(v0, v2, sphereCenter, t02);
        const double3 q12 = closestPointOnSegment(v1, v2, sphereCenter, t12);

        double dmin = dist2(sphereCenter, q01);
        closestPoint = q01;
        SphereTriangleContactType type = (t01 > 0.0 && t01 < 1.0) ? SphereTriangleContactType::Edge
        : SphereTriangleContactType::Vertex;

        const double d02 = dist2(sphereCenter, q02);
        if (d02 < dmin)
        {
            dmin = d02;
            closestPoint = q02;
            type = (t02 > 0.0 && t02 < 1.0) ? SphereTriangleContactType::Edge
            : SphereTriangleContactType::Vertex;
        }

        const double d12 = dist2(sphereCenter, q12);
        if (d12 < dmin)
        {
            dmin = d12;
            closestPoint = q12;
            type = (t12 > 0.0 && t12 < 1.0) ? SphereTriangleContactType::Edge
            : SphereTriangleContactType::Vertex;
        }

        return (dmin <= r2 + eps2) ? type : SphereTriangleContactType::None;
    }

    // --------------------------------------------------------
    // Ericson region tests (non-degenerate triangle)
    // Naming follows "Real-Time Collision Detection" style:
    // d1 = ab·ap, d2 = ac·ap, etc.
    // --------------------------------------------------------
    const double3 ap = sphereCenter - v0;
    const double d1 = dot(ab, ap);
    const double d2 = dot(ac, ap);

    // Vertex region v0
    if (d1 <= 0.0 && d2 <= 0.0)
    {
        closestPoint = v0;
        return (dist2(sphereCenter, v0) <= r2 + eps2) ? SphereTriangleContactType::Vertex
        : SphereTriangleContactType::None;
    }

    // Vertex region v1
    const double3 bp = sphereCenter - v1;
    const double d3 = dot(ab, bp);
    const double d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3)
    {
        closestPoint = v1;
        return (dist2(sphereCenter, v1) <= r2 + eps2) ? SphereTriangleContactType::Vertex
        : SphereTriangleContactType::None;
    }

    // Edge region v0-v1
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
    {
        const double t = d1 / (d1 - d3); // in [0,1]
        closestPoint = v0 + ab * t;
        return (dist2(sphereCenter, closestPoint) <= r2 + eps2) ? SphereTriangleContactType::Edge
        : SphereTriangleContactType::None;
    }

    // Vertex region v2
    const double3 cp = sphereCenter - v2;
    const double d5 = dot(ab, cp);
    const double d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6)
    {
        closestPoint = v2;
        return (dist2(sphereCenter, v2) <= r2 + eps2) ? SphereTriangleContactType::Vertex
        : SphereTriangleContactType::None;
    }

    // Edge region v0-v2
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
    {
        const double t = d2 / (d2 - d6); // in [0,1]
        closestPoint = v0 + ac * t;
        return (dist2(sphereCenter, closestPoint) <= r2 + eps2) ? SphereTriangleContactType::Edge
        : SphereTriangleContactType::None;
    }

    // Edge region v1-v2
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
    {
        const double t = (d4 - d3) / ((d4 - d3) + (d5 - d6)); // in [0,1]
        closestPoint = v1 + (v2 - v1) * t;
        return (dist2(sphereCenter, closestPoint) <= r2 + eps2) ? SphereTriangleContactType::Edge
        : SphereTriangleContactType::None;
    }

    // Face region
    const double sum = va + vb + vc;
    if (fabs(sum) <= 1e-30) return SphereTriangleContactType::None;

    const double invSum = 1.0 / sum;
    const double v = vb * invSum;
    const double w = vc * invSum;
    closestPoint = v0 + ab * v + ac * w;

    return (dist2(sphereCenter, closestPoint) <= r2 + eps2) ? SphereTriangleContactType::Face
    : SphereTriangleContactType::None;
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

const size_t numInteractions,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream);

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
cudaStream_t stream);

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

const size_t numBall,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream);

extern "C" void launchCalculateBallWallContactForceTorque(double3* position, 
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
cudaStream_t stream);

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
const double dt,

const size_t numInteraction,
const size_t gridD,
const size_t blockD,
cudaStream_t stream);

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
cudaStream_t stream);

extern "C" void launchAddLevelSetParticleWallForce(double3* force_p,
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
cudaStream_t stream);