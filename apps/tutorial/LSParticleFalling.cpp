#include "kernel/LSSolver.h"
#include "kernel/helper/LevelSetObjectGenerator.h"
#include "kernel/helper/SpherePackingGenerator.h"
#include "kernel/helper/OBJLoader.h"
#include "globalDamping.cuh"

inline quaternion randomQuaternionUniform_deterministic()
{
    static std::mt19937 rng(123456);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    const double u1 = U(rng), u2 = U(rng), u3 = U(rng);
    const double s1 = std::sqrt(1.0 - u1);
    const double s2 = std::sqrt(u1);
    const double a = 2.0 * pi() * u2;
    const double b = 2.0 * pi() * u3;

    return quaternion{
        s2 * std::cos(b),
        s1 * std::sin(a),
        s1 * std::cos(a),
        s2 * std::sin(b)
    };
}

class solver:
    public LSSolver
{
public:
    solver(): LSSolver("tutorial1") {}

    double dampingCoefficient = 0.2;

    void addExternalForceTorque(const double time) override
    {
        launchAddGlobalDampingForceTorque(getLSParticle().force(),
        getLSParticle().torque(),
        getLSParticle().velocity(),
        getLSParticle().angularVelocity(),
        dampingCoefficient,
        getLSParticle().num_device(),
        getLSParticle().gridDim(),
        getLSParticle().blockDim(),
        0);
    }
};

int main(const int argc, char** argv)
{
    const double l = 1.;
    const double3 boxMin = make_double3(0., 0., 0.);
    const double3 boxMax = make_double3(l, l, 3. * l);

    const double density = 1000.;
    std::vector<double3> vertexPosition; 
    std::vector<int3> triangleVertexIndex;
    OBJLoader::loadOBJMesh("bunny.obj", vertexPosition, triangleVertexIndex, argc, argv);
    for (auto& p:vertexPosition) p.y -= 0.1;
    LevelSetObject::TriangleMeshParticle TMP;
    TMP.setMesh(vertexPosition, triangleVertexIndex);
    TMP.buildGridByResolution();
    TMP.outputGridVTU("build/bunny");

    SpherePacking::Pack Pack_ = SpherePacking::buildRegularInBox(boxMin, 
    boxMax,
    0.5 * length(TMP.boundingBoxMax() - TMP.boundingBoxMin()));

    solver solver_;
    solver_.addParticleMaterial(6.e5, 1.8e5, 0.577, 1.);

    std::vector<quaternion> q;
    std::vector<double3> v, w;
    q.reserve(Pack_.centers.size());
    v.reserve(Pack_.centers.size());
    w.reserve(Pack_.centers.size());
    for (size_t i = 0; i < Pack_.centers.size(); i++)
    {
        q.push_back(randomQuaternionUniform_deterministic());
        v.push_back(make_double3(0., 0., 0.));
        w.push_back(make_double3(0., 0., 0.));
    }

    solver_.addParticleCluster(Pack_.centers, 
    v, 
    w, 
    q, 
    0, 
    density,
    TMP.gridInfo().gridOrigin,
    TMP.gridInfo().gridNodeSpacing,
    TMP.gridInfo().gridNodeSize,
    TMP.gridInfo().gridNodeSignedDistance,
    TMP.vertexPosition(),
    TMP.triangleVertexIndex());

    LevelSetObject::BoxWall BW;
    BW.setParams(boxMax.x - boxMin.x, boxMax.y - boxMin.y, boxMax.z - boxMin.z);
    BW.buildGridByResolution();
    solver_.addWallMaterial(0.577, 1.);
    solver_.addWallGeometry(BW.gridInfo().gridOrigin, 
    BW.gridInfo().gridNodeSpacing, 
    BW.gridInfo().gridNodeSize, 
    BW.gridInfo().gridNodeSignedDistance, 
    BW.vertexPosition(), 
    BW.triangleVertexIndex());
    solver_.addWall(0.5 * (boxMin + boxMax),
    make_double3(0., 0., 0.),
    make_double3(0., 0., 0.),
    quaternion{1., 0., 0., 0.},
    0,
    0);

    solver_.solve(boxMin, 
    boxMax, 
    make_double3(0., 0., -9.81), 
    5.e-5, 
    5., 
    50,
    argc,
    argv);
}