#include "kernel/DEMSolver.h"
#include "externalForceTorque.cuh"
#include "pointCloudGeneration.h"

inline double waterSurfaceHeightForStaticSphere(double R,
double rhoBall,
double rhoWater)
{
    if (!(R > 0.0) || !(rhoWater > 0.0) || !(rhoBall >= 0.0))
        return R; // simple fallback

    const double f = rhoBall / rhoWater; // required submerged volume fraction

    if (f <= 0.0) return -R;  // no buoyancy needed -> essentially not submerged
    if (f >= 1.0) return  R;  // sinks or neutral -> return R (your rule)

    // Submerged volume fraction for plane z = h:
    // f(h) = (R + h)^2 * (2R - h) / (4 R^3), monotone on [-R, R]
    auto submergedFraction = [&](double h) -> double
    {
        const double b = R + h;
        return (b * b) * (2.0 * R - h) / (4.0 * R * R * R);
    };

    // Bisection on h in [-R, R]
    double lo = -R, hi = R;
    for (int it = 0; it < 80; ++it)
    {
        const double mid = 0.5 * (lo + hi);
        const double fm  = submergedFraction(mid);
        if (fm < f) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
}

class problem:
    public DEMSolver
{
public:
    double fluidVel_x = -0.2;
    double waterLevel = 0.;

    problem(): DEMSolver(0) {}

    void addExternalForceTorque(const double time) override
    {
        const size_t num = getBallNumber();
        size_t blockD = 256;
        if (num < 256) blockD = num;
        size_t gridD = (num + blockD - 1) / blockD;

        launchAddBuoyancyDrag(getBallForceDevicePtr(),
        getBallPositionDevicePtr(),
        getBallVelocityDevicePtr(),
        getBallRadiusDevicePtr(),
        getBallInverseMassDevicePtr(),
        1000.,
        0.1,
        make_double3(0., 0., -9.81),
        make_double3(fluidVel_x, 0., 0.),
        waterLevel,
        num,
        gridD,
        blockD,
        0);
    }

    bool addInitialCondition() override
    {
        std::vector<int> obj0 = getBallPairPointed();
        std::vector<int> obj1 = getBallPairPointing();
        addBondedInteraction(obj0, obj1);
        return true;
    }
};

int main()
{
    double tc = 0.0002; // collide time
    double h = 0.1; // ice thickness
    int nx = 100; // L = 100 * h
    int ny = 200; // W = 100 * h
    double den = 900.; // ice density

    double r = 0.5 * h; // ball radius
    double m = r * r * r * pi() * 4. / 3. * den; // ball mass
    double kn = 0.5 * m * pi() * pi() / tc / tc; // ball stiffness
    double E = kn * 2. * r / (r * r * pi()); // Young's modulus

    problem test;
    test.waterLevel = waterSurfaceHeightForStaticSphere(r, den, 1000);

    test.setBond(0, 0, 1., E, 2.6, 0.5e6, 0.5e6, 0.1);
    test.setMaterial(0, E, 0.3, 0.8);
    test.setMaterial(1, 210e9, 0.3, 1.0);
    test.setLinearStiffness(0, 0, kn, kn / 2.6, 0., 0.);
    test.setLinearStiffness(0, 1, kn, kn / 2.6, 0., 0.);
    test.setFriction(0, 0, 0.1, 0., 0.);
    test.setFriction(0, 1, 0.1, 0., 0.);

    std::vector<double3> points;
    HexPBC2D pbc;
    generateHexPointCloud(make_double3(0., 0., 0.), 
    make_double3(nx * 2. * r, ny * 2. * r, 0.), 
    2 * r, 
    false, 
    true, 
    points, 
    pbc);

    for (size_t i = 0; i < points.size(); i++)
    {
        if (points[i].y > pbc.min.y + 3. * r || points[i].y < pbc.max.y - 3. * r)
        {
            test.addBall(points[i], 
            make_double3(test.fluidVel_x, 0., 0.), 
            make_double3(0., 0., 0.), 
            r, 
            den, 
            0);
        }
        else
        {
            test.addBall(points[i], 
            make_double3(test.fluidVel_x, 0., 0.), 
            make_double3(0., 0., 0.), 
            r, 
            0., 
            0);
        }
    }

    double d_top = 3.;
    double d_bottom = 2.25;
    double H_struct = 0.5 * (d_top - d_bottom) * std::sqrt(3.);

    TriangleMesh mesh = makeConeFrustumMeshZ(make_double3(0., 0., test.waterLevel - 0.3 * H_struct), 
    d_top, 
    d_bottom, 
    H_struct, 
    36);

    test.addMeshWall(mesh.vertices, 
    mesh.tri0, 
    mesh.tri1, 
    mesh.tri2, 
    make_double3(-0.5 * d_top - r, 0.5 * ny * 2. * r, 0.), 
    make_double3(0., 0., 0.), 
    make_double3(0., 0., 0.), 
    1);

    test.solve(make_double3(-pbc.max.x, pbc.min.y, -2. * H_struct), 
    make_double3(pbc.max.x, pbc.max.y, H_struct), 
    make_double3(0., 0., -9.81), 
    tc / 50., 
    50., 
    500, 
    "ice-struct");
}
