#include "kernel/LSDEMSolver.h"
#include "globalDamping.cuh"
#include <random>
#include <unordered_map>
#include <array>

inline int rand_deterministic(int min, int max)
{
    if (max <= min) return min;
    static std::mt19937 rng(123456); // fixed seed
    std::uniform_int_distribution<int> dist(min, max);
    return dist(rng);
}

inline quaternion randomQuaternionUniform_deterministic()
{
    static std::mt19937 rng(123456);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    const double u1 = U(rng), u2 = U(rng), u3 = U(rng);
    const double s1 = std::sqrt(1.0 - u1);
    const double s2 = std::sqrt(u1);
    const double a = 2.0 * M_PI * u2;
    const double b = 2.0 * M_PI * u3;

    return quaternion{
        s2 * std::cos(b),
        s1 * std::sin(a),
        s1 * std::cos(a),
        s2 * std::sin(b)
    };
}

inline void generateNonOverlappingSpheresInBox(std::vector<double3>& centers,
std::vector<double>& radii,
const double3 boxMin,
const double3 boxMax,
const int nSpheres,
const double rMin,
const double rMax,
const uint32_t seed = 12345u,
const int maxAttemptsPerSphere = 2000)
{
    centers.clear();
    radii.clear();

    if (nSpheres <= 0) return;
    if (rMin <= 0.0) return;
    if (rMax < rMin) return;

    const double3 bmin = make_double3(std::min(boxMin.x, boxMax.x),
    std::min(boxMin.y, boxMax.y),
    std::min(boxMin.z, boxMax.z));

    const double3 bmax = make_double3(std::max(boxMin.x, boxMax.x),
    std::max(boxMin.y, boxMax.y),
    std::max(boxMin.z, boxMax.z));

    // Quick feasibility check: at least fit the smallest sphere
    if ((bmax.x - bmin.x) < 2.0 * rMin) return;
    if ((bmax.y - bmin.y) < 2.0 * rMin) return;
    if ((bmax.z - bmin.z) < 2.0 * rMin) return;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ur01(0.0, 1.0);
    std::uniform_real_distribution<double> urR(rMin, rMax);

    // Spatial hash grid: cell size based on max radius
    const double cellSize = 2.0 * rMax;

    auto cellCoord = [&](const double3& p) -> std::array<int,3>
    {
        const double fx = (p.x - bmin.x) / cellSize;
        const double fy = (p.y - bmin.y) / cellSize;
        const double fz = (p.z - bmin.z) / cellSize;
        return { (int)std::floor(fx), (int)std::floor(fy), (int)std::floor(fz) };
    };

    auto packKey = [&](int cx, int cy, int cz) -> int64_t
    {
        const int64_t B = (1LL << 20);
        const int64_t x = (int64_t)cx + B;
        const int64_t y = (int64_t)cy + B;
        const int64_t z = (int64_t)cz + B;
        return (x << 42) ^ (y << 21) ^ z;
    };

    std::unordered_map<int64_t, std::vector<int>> buckets;
    buckets.reserve((size_t)nSpheres * 2);

    auto overlapsAny = [&](const double3& c, double r) -> bool
    {
        const auto cc = cellCoord(c);

        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
        {
            const int nx = cc[0] + dx;
            const int ny = cc[1] + dy;
            const int nz = cc[2] + dz;

            const int64_t key = packKey(nx, ny, nz);
            auto it = buckets.find(key);
            if (it == buckets.end()) continue;

            for (int id : it->second)
            {
                const double3 cj = centers[(size_t)id];
                const double rj = radii[(size_t)id];

                const double3 d = c - cj;
                const double rr = r + rj;
                if (dot(d, d) < rr * rr) return true;
            }
        }
        return false;
    };

    centers.reserve((size_t)nSpheres);
    radii.reserve((size_t)nSpheres);

    for (int s = 0; s < nSpheres; ++s)
    {
        bool placed = false;

        for (int attempt = 0; attempt < maxAttemptsPerSphere; ++attempt)
        {
            const double r = urR(rng);

            // Sample inside the shrunk box [bmin+r, bmax-r]
            const double3 c = make_double3(
                (bmin.x + r) + ur01(rng) * ((bmax.x - r) - (bmin.x + r)),
                (bmin.y + r) + ur01(rng) * ((bmax.y - r) - (bmin.y + r)),
                (bmin.z + r) + ur01(rng) * ((bmax.z - r) - (bmin.z + r))
            );

            if (overlapsAny(c, r)) continue;

            const int id = (int)centers.size();
            centers.push_back(c);
            radii.push_back(r);

            const auto cc = cellCoord(c);
            buckets[packKey(cc[0], cc[1], cc[2])].push_back(id);

            placed = true;
            break;
        }

        if (!placed) break;
    }
}

class problemSolver:
    public LSDEMSolver
{
public:
    problemSolver(): LSDEMSolver(0) {}

    void addExternalForceTorque(const double time) override
    {
        int numberOfParticles = getLevelSetParticleNumber();
        size_t blockD = 256, gridD = 0;
        if (numberOfParticles < blockD && numberOfParticles > 0) blockD = numberOfParticles;
        gridD = (numberOfParticles + blockD - 1) / blockD;
        launchAddGlobalDampingForceTorque(getLSParticleForceDevicePtr(),
        getLSParticleTorqueDevicePtr(),
        getLSParticleVelocityDevicePtr(),
        getLSParticleAngularVelocityDevicePtr(),
        0.1,
        numberOfParticles,
        gridD,
        blockD,
        0);
    }

    virtual bool addInitialCondition() override
    {
        const size_t n = getLevelSetParticlePointed().size();
        if (n == 0) return false;
        std::vector<double> radius_b(n, 1e-4), length_b(n, 2e-4);
        addLSBondedInteraction(getLevelSetParticlePointed(), 
        getLevelSetParticlePointing(), 
        getContactPoint(), 
        getContactNormal(), 
        radius_b,
        length_b);
        return true;
    }
};

int main()
{
    problemSolver problemSolver_;

    const double3 boxMin = make_double3(0., 0., 0.);
    const double3 boxMax = make_double3(1., 1., 4.);
    const double rMin = 0.1;
    const double rMax = 0.3;
    std::vector<double3> p;
    std::vector<double> r;
    generateNonOverlappingSpheresInBox(p, r, boxMin, boxMax, 200, rMin, rMax, 12345u, 10000);

    const double density_p = 1000.;

    for(size_t i = 0; i < p.size(); i++)
    {
        superellipsoidParams s;
        int shapeIndex = rand_deterministic(0, 3);
        if (shapeIndex == 0)
        {
            s.rx = 0.4, s.ry = 1., s.rz = 0.8, s.ee = 0.4, s.en = 1.6;
        }
        else if (shapeIndex == 1)
        {
            s.rx = 0.42, s.ry = 1., s.rz = 0.83, s.ee = 0.1, s.en = 1.;
        }
        else if (shapeIndex == 2)
        {
            s.rx = 0.2, s.ry = 1., s.rz = 1., s.ee = 1., s.en = 0.5;
        }
        else if (shapeIndex == 3)
        {
            s.rx = 0.5, s.ry = 0.7, s.rz = 1., s.ee = 1.4, s.en = 1.2;
        }
        s.rx *= 0.95 * r[i];
        s.ry *= 0.95 * r[i];
        s.rz *= 0.95 * r[i];
        quaternion q = randomQuaternionUniform_deterministic();
        problemSolver_.addLSSuperellipsoid(s.rx, 
        s.ry, 
        s.rz, 
        s.ee, 
        s.en, 
        0,
        density_p,
        p[i],
        make_double3(0., 0., 0.),
        make_double3(0., 0., 0.),
        q,
        int(r[i] * 10000));
    }

    problemSolver_.setLinearStiffness(0, 0, 6.e5, 1.8e5);
    problemSolver_.setFriction(0, 0, 0.577);
    problemSolver_.setFixedBoxWall(boxMin, boxMax);

    problemSolver_.solve(boxMin, boxMax, make_double3(0., 0., -9.81), 1.e-5, 7., 140, "levelSetParticleBeforeBonding");

    problemSolver_.setBond(0, 0, 5.e9, 0.3, 0., 0., 0.);
    problemSolver_.setFixedPlaneWall(make_double3(0., -3., -5.), 7., make_double3(1., 0.5, -1.), make_double3(1., 0., 1.));
    
    problemSolver_.solve(make_double3(-2., -3., -5.), make_double3(5., 4, 2.), make_double3(0., 0., -9.81), 1.e-5, 3., 60, "levelSetParticleAfterBonding");
};