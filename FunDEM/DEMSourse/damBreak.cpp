#include "kernel/WCSPHSolver.h"
#include "pointCloudGeneration.h"

class problem:
    public WCSPHSolver
{
public:
    problem(): WCSPHSolver(0) {}

    double H = 0.1;
    double spacing = 0.005;
};

int main()
{
    problem test;
    int n = int(test.H / test.spacing);
    if (n > 0) test.spacing = double(test.H / n);
    std::vector<double3> p_SPH = generateUniformPointCloud(make_double3(0., 0., 0.), 
    make_double3(test.H,2 * test.H, test.H), 
    test.spacing);

    double3 thick_w = make_double3(3 * test.spacing, 3 * test.spacing, 3 * test.spacing);

    std::vector<double3> p0_dummy = generateUniformPointCloud(make_double3(0,0,0) - thick_w, 
    make_double3(5 * test.H,2 * test.H,3 * test.H) + thick_w, 
    test.spacing);

    std::vector<double3> p_dummy;
    for(size_t i = 0; i < p0_dummy.size(); i++)
    {
        if (p0_dummy[i].x < 0 || p0_dummy[i].y < 0 || p0_dummy[i].z < 0 || 
        p0_dummy[i].x > 5 * test.H || p0_dummy[i].y > 2 * test.H) p_dummy.push_back(p0_dummy[i]);
    }

    double c = 20. * std::sqrt(9.81 * test.H);

    for(size_t i = 0; i < p_SPH.size(); i++)
    {
        test.addSPH(p_SPH[i], make_double3(0, 0, 0), c, test.spacing, 1000, 1.e-3);
    }

    for(size_t i = 0; i < p_dummy.size(); i++)
    {
        test.addDummy(p_dummy[i], make_double3(0, 0, 0), c, test.spacing, 1000, 1.e-3);
    }
    
    test.solve(make_double3(0, 0, 0) - thick_w, 
    make_double3(5 * test.H,2 * test.H,3 * test.H) + thick_w, 
    make_double3(0, 0, -9.81), 
    0.25 * 1.3 * test.spacing / (c + 2.0 * std::sqrt(9.81 * test.H)), 
    5., 
    100,
    "damBreak");
}
