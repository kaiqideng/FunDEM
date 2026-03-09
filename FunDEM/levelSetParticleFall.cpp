#include "kernel/LSDEMSolver.h"
#include "buildLevelSetParticle.h"

int main()
{
    LSDEMSolver test(0);
    superellipsoidParams s;
    LevelSetParticleGridInfo ls = buildLevelSetSuperellipsoidGridGlobal(s, make_double3(0., 0., 0.));
    std::vector<double3> sp = generateSuperellipsoidSurfacePointsGlobal_Uniform10k(s, make_double3(0., 0., 0.), 2000);
    test.setLinearStiffness(0, 0, 6.e5, 1.8e5);
    test.setFriction(0, 0, 0.577);
    test.addLSParticle(sp, ls.gridNodeLSF, ls.gridMin, ls.gridNodeSize, ls.gridSpacing, 
    0, 1000., make_double3(0., 0., 0.), make_double3(0., 0, 0.));

    std::vector<double3> sp1;
    sp1.push_back(make_double3(-1., -1., -1.));
    sp1.push_back(make_double3(1., -1., -1.));
    sp1.push_back(make_double3(1., 1., -1.));
    sp1.push_back(make_double3(-1., 1., -1.));
    double3 grdMin1 = make_double3(-1., -1., -2.);
    int3 gridNodeSize1 = make_int3(2, 2, 2);
    double gridSpacing1 = 2.;
    std::vector<double> gridNodeLSF1(gridNodeSize1.x * gridNodeSize1.y * gridNodeSize1.z, 0.);
    for (int x = 0; x < gridNodeSize1.x; x++)
    {
        for (int y = 0; y < gridNodeSize1.y; y++)
        {
            for (int z = 0; z < gridNodeSize1.z; z++)
            {
                const int index = x + gridNodeSize1.x * (y + gridNodeSize1.y * z);
                double3 pxyz = grdMin1 + make_double3(gridSpacing1 * double(x), gridSpacing1 * double(y), gridSpacing1 * double(z));
                gridNodeLSF1[index] = pxyz.z + 1;
            }
        }
    }
    test.addFixedLSParticle(sp1, gridNodeLSF1, grdMin1, gridNodeSize1, gridSpacing1, 0);

    test.solve(make_double3(-1., -1., -2.), make_double3(1., 1., 1.), make_double3(0., 0., -9.81), 1.e-5, 1., 20, "levelSetParticleFall");
}