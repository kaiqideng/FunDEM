#pragma once
#include "myUtility/myQua.h"

__device__ __forceinline__ double3 triangleCircumcenter(const double3& a,
const double3& b,
const double3& c)
{
    // Edges from vertex a
    double3 ab = b - a;
    double3 ac = c - a;

    // Triangle normal
    double3 n  = cross(ab, ac);
    double n2  = dot(n, n);   // |n|^2

    // Degenerate triangle: fall back to centroid
    if (n2 < 1e-30)
    {
        return (a + b + c) / 3.0;
    }

    // Formula:
    // O = a + ( |ac|^2 * (n × ab) + |ab|^2 * (ac × n) ) / (2 |n|^2)
    double3 term1 = cross(n,  ab) * dot(ac, ac);
    double3 term2 = cross(ac, n ) * dot(ab, ab);
    double invDen = 1.0 / (2.0 * n2);

    return a + (term1 + term2) * invDen;
}

extern "C" void launchWallIntegration(double3* position, 
double3* velocity, 
double3* angularVelocity,
quaternion* orientation, 
const double timeStep,
const size_t numWall,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream);

extern "C" void launchUpdateWallVertexGlobalPosition(double3* globalPosition_v,
double3* localPosition_v,
int* wallIndex_v,

double3* position_w,
quaternion* orientation_w,

const size_t numVertex,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream);

extern "C" void launchUpdateTriangleCircumcenter(double3* circumcenter,
int* index0_tri,
int* index1_tri,
int* index2_tri,

double3* globalPosition_v,

const size_t numTri,
const size_t gridD,
const size_t blockD, 
cudaStream_t stream);