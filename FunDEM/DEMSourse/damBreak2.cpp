#include "kernel/DEMSPHSolver.h"
#include "pointCloudGeneration.h"

class problem:
    public DEMSPHSolver
{
public:
    problem(cudaStream_t s_other): DEMSPHSolver(0, s_other) {}

    double spacing = 0.025;
    
    double L = 3.5;
    double H = 0.4;
    double W = 0.7;
    double L_box = 8.;

    double fluidDensity = 1000.;
    double soundSpeed = 20. * std::sqrt(H * 9.81);
    double viscosity = 0.001;

    double E_cube = 3.e9;
    double cubeDensity = 800;
    double3 cubeSize = make_double3(0.15, 0.15, 0.15);
    double3 position1_cub = make_double3(5.275, 0.35, 0.075);
    double3 position2_cub = make_double3(5.275, 0.35, 0.225);
    double3 position3_cub = make_double3(5.275, 0.35, 0.375);
};

int main()
{
    cudaStream_t s_other;
    cudaStreamCreate(&s_other);
    problem test(s_other);

    double3 minBound_dam = make_double3(0., 0., 0.);
    double3 maxBound_dam = make_double3(test.L, test.W, test.H);
    double3 thick_w = make_double3(3. * test.spacing, 3. * test.spacing, 3. * test.spacing);
    double3 minBound_box = make_double3(0., 0., 0.) - thick_w;
    double3 maxBound_box = make_double3(test.L_box, test.W, 2. * test.H) + thick_w;
    std::vector<double3> p_SPH = generateUniformPointCloud(minBound_dam, maxBound_dam, test.spacing);
    std::vector<double3> p0_wall = generateUniformPointCloud(minBound_box, maxBound_box, test.spacing);
    std::vector<double3> p1_wall;
    for (const auto& p : p0_wall)
    {
        if (p.x < 0 || p.y < 0 || p.z < 0 || p.x > test.L_box || p.y > test.W) p1_wall.push_back(p);
    }

    for (const auto& p : p_SPH)
    {
        test.addFluidSPH(p, 
        make_double3(0., 0., 0.), 
        test.soundSpeed, 
        test.spacing, 
        test.fluidDensity, 
        test.viscosity);
    }

    double r_solid = 0.5 * test.spacing;
    std::vector<double> radius(p1_wall.size(), r_solid);

    test.addFixedSolid(p1_wall, 
    radius, 
    0.5 * (minBound_box + maxBound_box), 
    0, 
    test.soundSpeed, 
    test.fluidDensity, 
    test.viscosity);

    std::vector<double3> p1_cube = generateUniformPointCloud(test.position1_cub - 0.5 * test.cubeSize,
    test.position1_cub + 0.5 * test.cubeSize, 
    test.spacing);

    std::vector<double3> p2_cube = generateUniformPointCloud(test.position2_cub - 0.5 * test.cubeSize,
    test.position2_cub + 0.5 * test.cubeSize, 
    test.spacing);

    std::vector<double3> p3_cube = generateUniformPointCloud(test.position3_cub - 0.5 * test.cubeSize,
    test.position3_cub + 0.5 * test.cubeSize, 
    test.spacing);

    double mass = test.cubeDensity * test.cubeSize.x * test.cubeSize.y * test.cubeSize.z;
    symMatrix inertia = make_symMatrix(mass / 12. * (test.cubeSize.y * test.cubeSize.y + test.cubeSize.z * test.cubeSize.z), 
    mass / 12. * (test.cubeSize.x * test.cubeSize.x + test.cubeSize.z * test.cubeSize.z), 
    mass / 12. * (test.cubeSize.x * test.cubeSize.x + test.cubeSize.y * test.cubeSize.y), 
    0., 
    0., 
    0.);

    std::vector<double> radius1(p1_cube.size(), r_solid);
    std::vector<double> radius2(p2_cube.size(), r_solid);
    std::vector<double> radius3(p3_cube.size(), r_solid);
    test.addSolid(p1_cube, 
    radius1, 
    test.position1_cub, 
    make_double3(0., 0., 0.),
    make_double3(0., 0., 0.),
    mass,
    inertia,
    1, 
    test.soundSpeed, 
    test.fluidDensity, 
    test.viscosity);

    test.addSolid(p2_cube, 
    radius2, 
    test.position2_cub, 
    make_double3(0., 0., 0.),
    make_double3(0., 0., 0.),
    mass,
    inertia,
    1, 
    test.soundSpeed, 
    test.fluidDensity, 
    test.viscosity);

    test.addSolid(p3_cube, 
    radius3, 
    test.position3_cub, 
    make_double3(0., 0., 0.),
    make_double3(0., 0., 0.),
    mass,
    inertia,
    1, 
    test.soundSpeed, 
    test.fluidDensity, 
    test.viscosity);

    double m = std::pow(test.spacing, 3.) * test.cubeDensity;
    double k = 0.5 * test.E_cube * pi() * r_solid;
    double dt = 0.1 * std::sqrt(m / k);
    double dt_SPH = 0.25 * 1.3 * test.spacing / (1.1 * test.soundSpeed);
    size_t interval = size_t(dt_SPH / dt);

    test.setSolidMaterial(0, test.E_cube, 0.3, 0.9);
    test.setSolidMaterial(1, test.E_cube, 0.3, 0.9);
    test.setSolidFriction(0, 1, 0.35, 0., 0.);
    test.setSolidFriction(1, 1, 0.45, 0., 0.);

    test.solve(minBound_box,
    maxBound_box,
    make_double3(0., 0., -9.81),
    dt,
    interval,
    3.,
    60,
    "damBreak2");
}