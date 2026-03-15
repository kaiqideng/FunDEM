#include "kernel/DEMSolver.h"
#include "externalForceTorque.cuh"

class problem:
    public DEMSolver
{
public:
    const double3 f_c = make_double3(0, 0, 100.e3);
    const double L_beam = 4.0;
    const size_t n_bond = 10;
    const double c_d = 0.1;
    const double dt = 1.e-5;
    const double T_max = 5.0;
    HostDeviceArray1D<double3> f_e;

    problem(): DEMSolver(0) {}

    void addExternalForceTorque(const double time) override
    {
        double c = time;
        if (c > 1.0) c = 1.0;
        std::vector<double3> f(n_bond + 1, make_double3(0., 0., 0.));
        f[n_bond] = f_c * c;
        f_e.setHost(f);
        f_e.copyHostToDevice(0);
        
        const size_t num = getBallNumber();
        size_t blockD = 256;
        if (num < 256) blockD = num;
        size_t gridD = (num + blockD - 1) / blockD;

        launchAddConstantForce(getBallForceDevicePtr(), 
        f_e.d_ptr, 
        num, 
        gridD, 
        blockD, 
        0);

        launchAddGlobalDampingForceTorque(getBallForceDevicePtr(), 
        getBallTorqueDevicePtr(), 
        getBallVelocityDevicePtr(), 
        getBallAngularVelocityDevicePtr(), 
        c_d, 
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
    problem test;
    test.setBond(0, 0, 1.0, 200e9, 2.6, 0.0, 0.0, 0.0);

    const double r = test.L_beam / double(test.n_bond) / 2.0;
    test.addFixedBall(make_double3(0., 0., 0.), r, 0);
    for (size_t i = 0; i < test.n_bond; i++)
    {
        test.addBall(make_double3((i + 1) * 2. * r, 0., 0.), 
        make_double3(0., 0., 0.), 
        make_double3(0., 0., 0.), 
        r, 
        7800, 
        0);
    }

    test.solve(make_double3(0., -r, -r), 
    make_double3(test.L_beam, r, r), 
    make_double3(0., 0., 0.), 
    test.dt, 
    test.T_max, 
    10, 
    "bondedParticleBeam");
}