#include "DEMSolver.h"
#include "CUDAKernelFunction/levelSetParticleContactDetectionKernel.cuh"
#include "CUDAKernelFunction/levelSetParticleIntegrationKernel.cuh"

struct LSBondedInteraction
{
    pair pair_;
    bond bond_;
    HostDeviceArray1D<double> bondLength_;
    HostDeviceArray1D<double> bondRadius_;
    HostDeviceArray1D<double3> localbondPointA_;
    HostDeviceArray1D<double3> localbondPointB_;
};

class LSDEMSolver
{
public:
    LSDEMSolver(cudaStream_t s)
    {
        maxThread_ = 256;
        stream_ = s;
    }

    ~LSDEMSolver() = default;

private:
    struct LevelSetParticleGridInfo
    {
        // Grid geometry in WORLD/GLOBAL frame
        double3 gridMin {0.0, 0.0, 0.0}; // background grid min corner (global)
        int3 gridNodeSize {0, 0, 0}; // (nx, ny, nz)
        double gridSpacing {0.0}; // h = diameter / 20

        // Flattened phi array, row-major:
        // idx = (k * ny + j) * nx + i
        // Convention here: phi > 0 INSIDE, phi < 0 OUTSIDE
        std::vector<double> gridNodeLSF;
    };

    inline double signedPow(double x, double e)
    {
        // sign(x) * |x|^e (works for negative x)
        if (x >= 0.0) return std::pow(x, e);
        return -std::pow(-x, e);
    }

    inline double safeAbsPow(double x, double e)
    {
        // |x|^e with protection
        return std::pow(std::fabs(x), e);
    }

    // ---------------------------------------------
    // Superellipsoid parameters
    // ---------------------------------------------
    struct superellipsoidParams
    {
        double rx {1.0};
        double ry {1.0};
        double rz {1.0};
        double ee {1.0}; // epsilon_e
        double en {1.0}; // epsilon_n
    };

    // ---------------------------------------------
    // Implicit function F(x,y,z) for superellipsoid (LOCAL coordinates)
    // From paper:  ( |x/rx|^(2/ee) + |y/ry|^(2/ee) )^(ee/en) + |z/rz|^(2/en) - 1
    // Convention: inside => F < 0, outside => F > 0
    // ---------------------------------------------
    inline double superellipsoidF_local(const double3& p, const superellipsoidParams& s)
    {
        const double ax = std::fabs(p.x / s.rx);
        const double ay = std::fabs(p.y / s.ry);
        const double az = std::fabs(p.z / s.rz);

        const double px = safeAbsPow(ax, 2.0 / s.ee);
        const double py = safeAbsPow(ay, 2.0 / s.ee);
        const double pxy = safeAbsPow(px + py, s.ee / s.en);

        const double pz = safeAbsPow(az, 2.0 / s.en);

        return (pxy + pz - 1.0);
    }

    // ---------------------------------------------
    // Numerical gradient of F (LOCAL), central difference
    // ---------------------------------------------
    inline double3 gradF_local(const double3& p, const superellipsoidParams& s, double h)
    {
        // Use a small step tied to grid spacing
        const double dh = std::max(1e-12, 0.25 * h);

        const double3 ex = make_double3(dh, 0.0, 0.0);
        const double3 ey = make_double3(0.0, dh, 0.0);
        const double3 ez = make_double3(0.0, 0.0, dh);

        const double fx1 = superellipsoidF_local(p + ex, s);
        const double fx0 = superellipsoidF_local(p - ex, s);
        const double fy1 = superellipsoidF_local(p + ey, s);
        const double fy0 = superellipsoidF_local(p - ey, s);
        const double fz1 = superellipsoidF_local(p + ez, s);
        const double fz0 = superellipsoidF_local(p - ez, s);

        return make_double3((fx1 - fx0) / (2.0 * dh),
        (fy1 - fy0) / (2.0 * dh),
        (fz1 - fz0) / (2.0 * dh));
    }

    // ---------------------------------------------
    // Build level-set grid for shape index 1 in GLOBAL coordinates.
    // - centerPoint: shape center in GLOBAL coordinates
    // - paddingLayers: how many grid layers beyond bbox
    // Rule: gridSpacing = diameter/20, diameter = 2*max(rx,ry,rz)
    // Output: phi < 0 inside, phi > 0 outside
    //
    // Notes:
    // - Uses phi ≈ F / |∇F| (signed-distance-like). Sign comes from F.
    // - Uses a dedicated finite-difference step (fd) for ∇F (NOT the grid spacing).
    // - Adds robust protection for small |∇F|.
    // ---------------------------------------------
    inline LevelSetParticleGridInfo buildLevelSetSuperellipsoidGridGlobal(const superellipsoidParams s,
    const double3 centerPoint,
    const int paddingLayers = 2)
    {
        LevelSetParticleGridInfo info;

        const double Rmax = std::max(s.rx, std::max(s.ry, s.rz));
        const double diameter = 2.0 * Rmax;
        if (diameter <= 0.0) return info;

        const double h = diameter / 20.0;
        info.gridSpacing = h;

        // BBox of the shape in GLOBAL (axis-aligned because shape is axis-aligned here)
        const double3 bmin = centerPoint + make_double3(-s.rx, -s.ry, -s.rz);
        const double3 bmax = centerPoint + make_double3( s.rx,  s.ry,  s.rz);

        const double pad = double(std::max(1, paddingLayers)) * h;

        double3 gmin = make_double3(bmin.x - pad, bmin.y - pad, bmin.z - pad);
        double3 gmax = make_double3(bmax.x + pad, bmax.y + pad, bmax.z + pad);

        auto snapDown = [&](double a) { return h * std::floor(a / h); };
        auto snapUp = [&](double a) { return h * std::ceil (a / h); };

        gmin.x = snapDown(gmin.x); gmin.y = snapDown(gmin.y); gmin.z = snapDown(gmin.z);
        gmax.x = snapUp (gmax.x); gmax.y = snapUp (gmax.y); gmax.z = snapUp (gmax.z);

        const int nx = int(std::llround((gmax.x - gmin.x) / h)) + 1;
        const int ny = int(std::llround((gmax.y - gmin.y) / h)) + 1;
        const int nz = int(std::llround((gmax.z - gmin.z) / h)) + 1;

        if (nx <= 0 || ny <= 0 || nz <= 0) return info;

        info.gridMin = gmin;
        info.gridNodeSize = make_int3(nx, ny, nz);
        info.gridNodeLSF.assign(size_t(nx) * size_t(ny) * size_t(nz), 0.0);

        auto idx3 = [&](int i, int j, int k) -> size_t
        {
            return size_t((k * ny + j) * nx + i);
        };

        // Dedicated finite-difference step for gradient (more stable than using grid spacing directly).
        // You can tune this: smaller -> more accurate but can be noisier with finite precision.
        const double fd = std::max(1e-12, 0.25 * h);

        // Robust epsilon for |∇F|
        const double gradEps = 1e-12;

        for (int k = 0; k < nz; ++k)
        {
            const double z = gmin.z + double(k) * h;

            for (int j = 0; j < ny; ++j)
            {
                const double y = gmin.y + double(j) * h;

                for (int i = 0; i < nx; ++i)
                {
                    const double x = gmin.x + double(i) * h;

                    // Convert to LOCAL coordinates (shape centered at centerPoint)
                    const double3 p_global = make_double3(x, y, z);
                    const double3 p_local  = p_global - centerPoint;

                    // Implicit function value
                    const double F = superellipsoidF_local(p_local, s);

                    // Gradient of implicit function (should be in local coordinates)
                    const double3 gF = gradF_local(p_local, s, fd);
                    const double gNorm = length(gF);

                    // Signed-distance-like level set (sign from F)
                    // Protect against tiny gradient norm to avoid blow-ups
                    double phi;
                    if (gNorm > gradEps)
                    {
                        phi = F / gNorm;
                    }
                    else
                    {
                        // Fallback: preserve sign but clamp magnitude to something reasonable (order of grid spacing)
                        // This avoids NaNs/inf and prevents huge spikes inside degenerate gradient regions.
                        const double sgn = (F >= 0.0) ? 1.0 : -1.0;
                        phi = sgn * 0.5 * h;
                    }

                    info.gridNodeLSF[idx3(i, j, k)] = phi; // phi<0 inside, phi>0 outside
                }
            }
        }

        return info;
    }

    inline std::vector<double3> generateSuperellipsoidSurfacePointsGlobal_Uniform(const superellipsoidParams s, 
    const double3 centerPoint,
    const int nPoints = 10000)
    {
        std::vector<double3> pts;
        if (nPoints <= 0) return pts;

        // ---------------------------------------------------------
        // Helper: evaluate F along ray p = t * dir (LOCAL coords)
        // ---------------------------------------------------------
        auto F_ray = [&](double t, const double3& dir) -> double
        {
            const double3 p = make_double3(t * dir.x, t * dir.y, t * dir.z);
            return superellipsoidF_local(p, s);
        };

        // ---------------------------------------------------------
        // Helper: bracket + bisection solve F(t)=0 for t>0
        // Superellipsoid is star-shaped, so F(t) is monotone after a small t.
        // ---------------------------------------------------------
        auto solveRadius = [&](const double3& dir) -> double
        {
            // Start bracket
            double t0 = 0.0;
            double f0 = F_ray(t0, dir); // should be -1 at origin

            // Upper bound: use a conservative radius
            double t1 = 2.0 * std::max(s.rx, std::max(s.ry, s.rz));
            double f1 = F_ray(t1, dir);

            // Expand if needed
            int expandIters = 0;
            while (f1 < 0.0 && expandIters < 30)
            {
                t1 *= 2.0;
                f1 = F_ray(t1, dir);
                expandIters++;
            }

            // If still not bracketed, give up (shouldn't happen for valid shapes)
            if (f1 < 0.0) return t1;

            // Bisection
            double a = t0, b = t1;
            for (int it = 0; it < 60; ++it)
            {
                double m = 0.5 * (a + b);
                double fm = F_ray(m, dir);
                if (fm > 0.0) b = m;
                else a = m;
            }
            return 0.5 * (a + b);
        };

        // ---------------------------------------------------------
        // Fibonacci sphere directions (approximately uniform on S^2)
        // ---------------------------------------------------------
        pts.reserve((size_t)nPoints);

        const double golden = (1.0 + std::sqrt(5.0)) * 0.5;
        const double goldenAngle = 2.0 * M_PI * (1.0 - 1.0 / golden);

        for (int i = 0; i < nPoints; ++i)
        {
            // y in (-1,1), avoid exact poles
            const double t = (i + 0.5) / double(nPoints);
            const double y = 1.0 - 2.0 * t;
            const double r = std::sqrt(std::max(0.0, 1.0 - y * y));

            const double phi = goldenAngle * double(i);
            const double x = r * std::cos(phi);
            const double z = r * std::sin(phi);

            // Direction on unit sphere
            const double3 dir = make_double3(x, y, z);

            // Solve intersection with superellipsoid surface (LOCAL), then shift to GLOBAL
            const double radius = solveRadius(dir);
            const double3 p_local = make_double3(radius * dir.x, radius * dir.y, radius * dir.z);
            pts.push_back(centerPoint + p_local);
        }

        return pts;
    }


    void upload()
    {
        int matID_max = *std::max_element(levelSetParticle_.materialIDHostRef().begin(), levelSetParticle_.materialIDHostRef().end());
        if (matID_max >= materialTable_.size()) 
        {
            materialRow row;
            row.poissonRatio = 0.;
            row.YoungsModulus = 0.;
            row.restitutionCoefficient = 1.;
            for (size_t i = 0; i < matID_max + 1 - materialTable_.size(); i++)
            {
                materialTable_.push_back(row);
            }
        }
        contactParameters_.buildFromTables(materialTable_, 
        frictionTable_, 
        linearStiffnessTable_, 
        bondTable_,
        stream_);

        levelSetBoundaryNode_.copyHostToDevice(stream_);
        levelSetGridNode_.copyHostToDevice(stream_);
        levelSetParticle_.copyHostToDevice(stream_);

        LSBondedInteraction_.pair_.copyHostToDevice(stream_);
        LSBondedInteraction_.bond_.copyHostToDevice(stream_);
        LSBondedInteraction_.bondRadius_.copyHostToDevice(stream_);
        LSBondedInteraction_.bondLength_.copyHostToDevice(stream_);
        LSBondedInteraction_.localbondPointA_.copyHostToDevice(stream_);
        LSBondedInteraction_.localbondPointB_.copyHostToDevice(stream_);

        if (levelSetParticle_.deviceSize() > 0 && LSParticleInteraction_.numActivated_ == 0)
        {
            LSParticleInteraction_.objectPointed_.allocateDevice(levelSetBoundaryNode_.deviceSize(), stream_);
            LSParticleInteraction_.objectPointing_.allocateDevice(levelSetParticle_.deviceSize(), stream_);
        }

        wallLSFV_.copyHostToDevice(stream_);
    }

    void initializeSpatialGrid(const double3 minBoundary, const double3 maxBoundary)
    {
        double cellSizeOneDim = *std::max_element(levelSetParticle_.radiiHostRef().begin(), levelSetParticle_.radiiHostRef().end()) * 2.0;
        spatialGrid_.set(minBoundary, maxBoundary, cellSizeOneDim, stream_);
    }

protected:
    void LSDEMInitialize(const double3 minBoundary, const double3 maxBoundary)
    {
        levelSetBoundaryNode_.setBlockDim(maxThread_ < levelSetBoundaryNode_.hostSize() ? maxThread_ : levelSetBoundaryNode_.hostSize());
        levelSetGridNode_.setBlockDim(maxThread_ < levelSetGridNode_.hostSize() ? maxThread_ : levelSetGridNode_.hostSize());
        levelSetParticle_.setBlockDim(maxThread_ < levelSetParticle_.hostSize() ? maxThread_ : levelSetParticle_.hostSize());

        upload();

        initializeSpatialGrid(minBoundary, maxBoundary);
    }

    void updateLSSpatialGridCellHashStartEnd()
    {
        launchUpdateSpatialGridCellHashStartEnd(levelSetParticle_.position(), 
        levelSetParticle_.hashIndex(), 
        levelSetParticle_.hashValue(), 
        spatialGrid_.cellHashStart(), 
        spatialGrid_.cellHashEnd(), 
        spatialGrid_.minimumBoundary(), 
        spatialGrid_.maximumBoundary(), 
        spatialGrid_.cellSize(), 
        spatialGrid_.gridSize(), 
        spatialGrid_.numGrids(),
        levelSetParticle_.deviceSize(), 
        levelSetParticle_.gridDim(), 
        levelSetParticle_.blockDim(), 
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif
    }

    void LSParticleContactDetection()
    {
        LSParticleInteraction_.updateOldPairOldSpring(stream_);

        // 1) count + prefix sum
        launchCountLevelSetBoundaryNodeInteractions(LSParticleInteraction_.objectPointed_.neighborCount(),
        LSParticleInteraction_.objectPointed_.neighborPrefixSum(),
        levelSetBoundaryNode_.localPosition(),
        levelSetBoundaryNode_.particleID(),
        levelSetGridNode_.LSFV(),
        levelSetParticle_.position(),
        levelSetParticle_.orientation(),
        levelSetParticle_.radii(),
        levelSetParticle_.inverseMass(),
        levelSetParticle_.gridSpacing(),
        levelSetParticle_.gridNodeLocalOrigin(),
        levelSetParticle_.gridNodeSize(),
        levelSetParticle_.gridNodePrefixSum(),
        levelSetParticle_.hashIndex(),
        spatialGrid_.cellHashStart(),
        spatialGrid_.cellHashEnd(),
        spatialGrid_.minimumBoundary(),
        spatialGrid_.cellSize(),
        spatialGrid_.gridSize(),
        levelSetBoundaryNode_.deviceSize(),
        levelSetBoundaryNode_.gridDim(),
        levelSetBoundaryNode_.blockDim(),
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif

        LSParticleInteraction_.resizePair(stream_);

        // 2) write pairs + springs + contact geom
        launchWriteLevelSetBoundaryNodeInteractions(LSParticleInteraction_.spring_.sliding(),
        LSParticleInteraction_.spring_.rolling(),
        LSParticleInteraction_.spring_.torsion(),
        LSParticleInteraction_.contact_.point(),
        LSParticleInteraction_.contact_.normal(),
        LSParticleInteraction_.contact_.overlap(),
        LSParticleInteraction_.pair_.objectPointed(),
        LSParticleInteraction_.pair_.objectPointing(),
        LSParticleInteraction_.oldSpring_.sliding(),
        LSParticleInteraction_.oldSpring_.rolling(),
        LSParticleInteraction_.oldSpring_.torsion(),
        LSParticleInteraction_.oldPair_.objectPointed(),
        LSParticleInteraction_.oldPair_.hashIndex(),
        levelSetBoundaryNode_.localPosition(),
        levelSetBoundaryNode_.particleID(),
        LSParticleInteraction_.objectPointed_.neighborPrefixSum(),
        levelSetGridNode_.LSFV(),
        levelSetParticle_.position(),
        levelSetParticle_.orientation(),
        levelSetParticle_.radii(),
        levelSetParticle_.inverseMass(),
        levelSetParticle_.gridSpacing(),
        levelSetParticle_.gridNodeLocalOrigin(),
        levelSetParticle_.gridNodeSize(),
        levelSetParticle_.gridNodePrefixSum(),
        levelSetParticle_.hashIndex(),
        LSParticleInteraction_.objectPointing_.interactionStart(),
        LSParticleInteraction_.objectPointing_.interactionEnd(),
        spatialGrid_.cellHashStart(),
        spatialGrid_.cellHashEnd(),
        spatialGrid_.minimumBoundary(),
        spatialGrid_.cellSize(),
        spatialGrid_.gridSize(),
        levelSetBoundaryNode_.deviceSize(),
        levelSetBoundaryNode_.gridDim(),
        levelSetBoundaryNode_.blockDim(),
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif

        LSParticleInteraction_.buildObjectPointingInteractionStartEnd(maxThread_, stream_);
    }

    void LSParticleContactForceTorque(const double timeStep)
    {
#ifndef NDEBUG
CUDA_CHECK(cudaMemsetAsync(levelSetParticle_.force(), 0, levelSetParticle_.deviceSize() * sizeof(double3), stream_));
CUDA_CHECK(cudaMemsetAsync(levelSetParticle_.torque(), 0, levelSetParticle_.deviceSize() * sizeof(double3), stream_));
#else
        cudaMemsetAsync(levelSetParticle_.force(), 0, levelSetParticle_.deviceSize() * sizeof(double3), stream_);
        cudaMemsetAsync(levelSetParticle_.torque(), 0, levelSetParticle_.deviceSize() * sizeof(double3), stream_);
#endif

        if (LSParticleInteraction_.numActivated_ > 0) 
        {
            size_t blockD = maxThread_;
            if (LSParticleInteraction_.numActivated_ < maxThread_) blockD = LSParticleInteraction_.numActivated_ ;
            size_t gridD = (LSParticleInteraction_.numActivated_ + blockD - 1) / blockD;

            launchCalLevelSetParticleContactForceTorque(LSParticleInteraction_.contact_.force(), 
            LSParticleInteraction_.spring_.sliding(), 
            LSParticleInteraction_.contact_.point(), 
            LSParticleInteraction_.contact_.normal(), 
            LSParticleInteraction_.contact_.overlap(), 
            LSParticleInteraction_.pair_.objectPointed(), 
            LSParticleInteraction_.pair_.objectPointing(), 
            levelSetBoundaryNode_.particleID(), 
            levelSetParticle_.force(), 
            levelSetParticle_.torque(), 
            levelSetParticle_.position(), 
            levelSetParticle_.velocity(), 
            levelSetParticle_.angularVelocity(), 
            levelSetParticle_.materialID(), 
            timeStep, 
            LSParticleInteraction_.numActivated_, 
            gridD, 
            blockD, 
            stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif
        }

        size_t numBondedInteraction = LSBondedInteraction_.bond_.deviceSize();
        if (numBondedInteraction > 0)
        {
            size_t blockD = maxThread_;
            if (numBondedInteraction < maxThread_) blockD = numBondedInteraction;
            size_t gridD = (numBondedInteraction + blockD - 1) / blockD;

            launchAddLevelSetParticleBondedForceTorque(LSBondedInteraction_.bond_.point(),
            LSBondedInteraction_.bond_.normal(),
            LSBondedInteraction_.bond_.shearForce(),
            LSBondedInteraction_.bond_.bendingTorque(),
            LSBondedInteraction_.bond_.normalForce(),
            LSBondedInteraction_.bond_.torsionTorque(),
            LSBondedInteraction_.bond_.maxNormalStress(),
            LSBondedInteraction_.bond_.maxShearStress(),
            LSBondedInteraction_.bond_.isBonded(),
            LSBondedInteraction_.localbondPointA_.d_ptr, 
            LSBondedInteraction_.localbondPointB_.d_ptr, 
            LSBondedInteraction_.bondLength_.d_ptr, 
            LSBondedInteraction_.bondRadius_.d_ptr, 
            LSBondedInteraction_.pair_.objectPointed(), 
            LSBondedInteraction_.pair_.objectPointing(), 
            levelSetParticle_.force(), 
            levelSetParticle_.torque(), 
            levelSetParticle_.position(), 
            levelSetParticle_.velocity(), 
            levelSetParticle_.angularVelocity(), 
            levelSetParticle_.orientation(), 
            levelSetParticle_.materialID(), 
            timeStep, 
            numBondedInteraction, 
            gridD, 
            blockD, 
            stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif
        }

        if (wallLSFV_.deviceSize() >= 8)
        {
            launchAddLevelSetParticleWallForce(levelSetParticle_.force(),
            levelSetParticle_.position(),
            levelSetParticle_.orientation(), 
            levelSetParticle_.inverseMass(),
            levelSetParticle_.materialID(),
            levelSetBoundaryNode_.localPosition(),
            levelSetBoundaryNode_.particleID(),
            wallLSFV_.d_ptr,
            wallGridSpacing_,
            wallGridNodeOrigin_,
            wallGridNodeSize_,
            levelSetBoundaryNode_.deviceSize(),
            levelSetBoundaryNode_.gridDim(),
            levelSetBoundaryNode_.blockDim(),
            stream_);
        }
    }

    void LSParticleIntegration1stHalf(const double3 gravity, const double halfTimeStep)
    {
        launchLevelSetParticle1stHalfIntegration(levelSetParticle_.position(), 
        levelSetParticle_.velocity(), 
        levelSetParticle_.angularVelocity(), 
        levelSetParticle_.force(), 
        levelSetParticle_.torque(), 
        levelSetParticle_.inverseMass(),
        levelSetParticle_.orientation(), 
        levelSetParticle_.inverseInertiaTensor(), 
        gravity, 
        halfTimeStep, 
        levelSetParticle_.deviceSize(), 
        levelSetParticle_.gridDim(), 
        levelSetParticle_.blockDim(), 
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif
    }

    void LSParticleIntegration2ndHalf(const double3 gravity, const double halfTimeStep)
    {
        launchLevelSetParticle2ndHalfIntegration(levelSetParticle_.velocity(), 
        levelSetParticle_.angularVelocity(), 
        levelSetParticle_.force(), 
        levelSetParticle_.torque(), 
        levelSetParticle_.inverseMass(),
        levelSetParticle_.orientation(), 
        levelSetParticle_.inverseInertiaTensor(), 
        gravity, 
        halfTimeStep, 
        levelSetParticle_.deviceSize(), 
        levelSetParticle_.gridDim(), 
        levelSetParticle_.blockDim(), 
        stream_);

#ifndef NDEBUG
CUDA_CHECK(cudaGetLastError());
#endif
    }

    void outputLevelSetParticleVTU(const std::string &dir, const size_t iFrame, const size_t iStep, const double time)
    {
        MKDIR(dir.c_str());
        std::ostringstream fname;
        fname << dir << "/LS-Particle_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";
        std::ofstream out(fname.str().c_str());
        if (!out) throw std::runtime_error("Cannot open " + fname.str());
        out << std::fixed << std::setprecision(10);

        const size_t N = levelSetBoundaryNode_.deviceSize();
        std::vector<double3> p = levelSetBoundaryNode_.localPositionHostRef();
        std::vector<double3> v(N, make_double3(0.0, 0.0, 0.0)) , a(N, make_double3(0.0, 0.0, 0.0));
        std::vector<int> m(N, 0);

        std::vector<double3> p_p = levelSetParticle_.positionHostCopy();
        std::vector<double3> v_p = levelSetParticle_.velocityHostCopy();
        std::vector<double3> a_p = levelSetParticle_.angularVelocityHostCopy();
        std::vector<quaternion> q_p = levelSetParticle_.orientationHostCopy();
        const std::vector<int>& m_p = levelSetParticle_.materialIDHostRef();

        for (size_t i = 0; i < N; i++)
        {
            const int pID = levelSetBoundaryNode_.particleIDHostRef()[i];
            p[i] = p_p[pID] + rotateVectorByQuaternion(q_p[pID], p[i]);
            v[i] = v_p[pID] + cross(p[i] - p_p[pID], a_p[pID]);
            a[i] = a_p[pID];
            m[i] = m_p[pID];
        }

        out << "<?xml version=\"1.0\"?>\n"
            "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
            "  <UnstructuredGrid>\n";

        out << "    <FieldData>\n"
            "      <DataArray type=\"Float32\" Name=\"TIME\"  NumberOfTuples=\"1\" format=\"ascii\"> "
            << time << " </DataArray>\n"
            "      <DataArray type=\"Int32\"   Name=\"STEP\"  NumberOfTuples=\"1\" format=\"ascii\"> "
            << iStep << " </DataArray>\n"
            "    </FieldData>\n";

        out << "    <Piece NumberOfPoints=\"" << N
            << "\" NumberOfCells=\"" << N << "\">\n";

        out << "      <Points>\n"
            "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) {
            out << ' ' << p[i].x << ' ' << p[i].y << ' ' << p[i].z;
        }
        out << "\n        </DataArray>\n"
            "      </Points>\n";

        out << "      <Cells>\n"
            "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << i;
        out << "\n        </DataArray>\n"
            "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
        for (int i = 1; i <= N; ++i) out << ' ' << i;
        out << "\n        </DataArray>\n"
            "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << " 1";          // 1 = VTK_VERTEX
        out << "\n        </DataArray>\n"
            "      </Cells>\n";

        out << "      <PointData Scalars=\"material ID\">\n";

        out << "        <DataArray type=\"Int32\" Name=\"material ID\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << m[i];
        out << "\n        </DataArray>\n";

        const struct {
            const char* name;
            const std::vector<double3>& vec;
        } vec3s[] = {
            { "velocity"       , v     },
            { "angularVelocity", a     }
        };
        for (size_t k = 0; k < sizeof(vec3s) / sizeof(vec3s[0]); ++k) {
            out << "        <DataArray type=\"Float32\" Name=\"" << vec3s[k].name
                << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
            const std::vector<double3>& v = vec3s[k].vec;
            for (size_t i = 0; i < v.size(); ++i)
                out << ' ' << v[i].x << ' ' << v[i].y << ' ' << v[i].z;
            out << "\n        </DataArray>\n";
        }

        out << "      </PointData>\n"
            "    </Piece>\n"
            "  </UnstructuredGrid>\n"
            "</VTKFile>\n";
    }

    void outputLevelSetParticleInteractionVTU(const std::string &dir, const size_t iFrame, const size_t iStep, const double time)
    {
        MKDIR(dir.c_str());
        std::ostringstream fname;
        fname << dir << "/LS-ParticleInteraction_" << std::setw(4) << std::setfill('0') << iFrame << ".vtu";
        std::ofstream out(fname.str().c_str());
        if (!out) throw std::runtime_error("Cannot open " + fname.str());
        out << std::fixed << std::setprecision(10);

        const size_t N = LSParticleInteraction_.numActivated_;
        std::vector<double3> p = LSParticleInteraction_.contact_.pointHostCopy();
        std::vector<double> o = LSParticleInteraction_.contact_.overlapHostCopy();
        std::vector<double3> n = LSParticleInteraction_.contact_.normalHostCopy();
        std::vector<double3> f_c = LSParticleInteraction_.contact_.forceHostCopy();
        std::vector<double3> s_s = LSParticleInteraction_.spring_.slidingHostCopy();
        
        out << "<?xml version=\"1.0\"?>\n"
            "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
            "  <UnstructuredGrid>\n";

        out << "    <FieldData>\n"
            "      <DataArray type=\"Float32\" Name=\"TIME\"  NumberOfTuples=\"1\" format=\"ascii\"> "
            << time << " </DataArray>\n"
            "      <DataArray type=\"Int32\"   Name=\"STEP\"  NumberOfTuples=\"1\" format=\"ascii\"> "
            << iStep << " </DataArray>\n"
            "    </FieldData>\n";

        out << "    <Piece NumberOfPoints=\"" << N
            << "\" NumberOfCells=\"" << N << "\">\n";

        out << "      <Points>\n"
            "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) {
            out << ' ' << p[i].x << ' ' << p[i].y << ' ' << p[i].z;
        }
        out << "\n        </DataArray>\n"
            "      </Points>\n";

        out << "      <Cells>\n"
            "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << i;
        out << "\n        </DataArray>\n"
            "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
        for (int i = 1; i <= N; ++i) out << ' ' << i;
        out << "\n        </DataArray>\n"
            "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << " 1";          // 1 = VTK_VERTEX
        out << "\n        </DataArray>\n"
            "      </Cells>\n";

        out << "      <PointData Scalars=\"overlap\">\n";

        out << "        <DataArray type=\"Float32\" Name=\"overlap\" format=\"ascii\">\n";
        for (int i = 0; i < N; ++i) out << ' ' << o[i];
        out << "\n        </DataArray>\n";

        const struct {
            const char* name;
            const std::vector<double3>& vec;
        } vec3s[] = {
            { "contact normal" , n     },
            { "contact force"  , f_c   },
            { "sliding spring" , s_s   }
        };

        for (size_t k = 0; k < sizeof(vec3s) / sizeof(vec3s[0]); ++k) {
            out << "        <DataArray type=\"Float32\" Name=\"" << vec3s[k].name
                << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
            const std::vector<double3>& v = vec3s[k].vec;
            for (size_t i = 0; i < N; ++i)
                out << ' ' << v[i].x << ' ' << v[i].y << ' ' << v[i].z;
            out << "\n        </DataArray>\n";
        }

        out << "      </PointData>\n"
            "    </Piece>\n"
            "  </UnstructuredGrid>\n"
            "</VTKFile>\n";
    }

    const double3* getLSParticlePositionDevicePtr()
    {
        return levelSetParticle_.position();
    }

    const double3* getLSParticleVelocityDevicePtr()
    {
        return levelSetParticle_.velocity();
    }

    const double3* getLSParticleAngularVelocityDevicePtr()
    {
        return levelSetParticle_.angularVelocity();
    }

    double3* getLSParticleForceDevicePtr()
    {
        return levelSetParticle_.force();
    }

    double3* getLSParticleTorqueDevicePtr()
    {
        return levelSetParticle_.torque();
    }

    std::vector<double3> getLSParticlePosition()
    {
        return levelSetParticle_.positionHostCopy();
    }

    std::vector<double3> getLSParticleVelocity()
    {
        return levelSetParticle_.velocityHostCopy();
    }

    std::vector<double3> getLSParticleAngularVelocity()
    {
        return levelSetParticle_.angularVelocityHostCopy();
    }

    std::vector<int> getLevelSetParticlePairA()
    { 
        std::vector<int> a = LSParticleInteraction_.pair_.objectPointingHostCopy();
        std::vector<int> a1(a.begin(), a.begin() + LSParticleInteraction_.numActivated_);
        for (auto& aa: a1) aa = levelSetBoundaryNode_.particleIDHostRef()[aa];
        return a1;
    }

    std::vector<int> getLevelSetParticlePairB()
    { 
        std::vector<int> b = LSParticleInteraction_.pair_.objectPointingHostCopy();
        std::vector<int> b1(b.begin(), b.begin() + LSParticleInteraction_.numActivated_);
        return b1;
    }

    std::vector<double3> getContactPoint() 
    { 
        std::vector<double3> p = LSParticleInteraction_.contact_.pointHostCopy();
        std::vector<double3> p1(p.begin(), p.begin() + LSParticleInteraction_.numActivated_);
        return p1;
    }

    std::vector<double3> getContactNormal() 
    { 
        std::vector<double3> n = LSParticleInteraction_.contact_.normalHostCopy();
        std::vector<double3> n1(n.begin(), n.begin() + LSParticleInteraction_.numActivated_);
        return n1;
    }

    void addLSBondedInteraction(std::vector<int> particleA, 
    std::vector<int> particleB, 
    std::vector<double3> point_b, 
    std::vector<double3> normal_b, 
    std::vector<double> radius_b,
    std::vector<double> length_b)
    {
        std::vector<quaternion> orientation_p = levelSetParticle_.orientationHostCopy();
        std::vector<double3> position_p = levelSetParticle_.positionHostCopy();
        for (size_t i = 0; i < particleA.size(); i++)
        {
            int idxA = particleA[i], idxB = particleB[i];
            if (idxA >= orientation_p.size()) continue;
            if (idxB >= orientation_p.size()) continue;
            LSBondedInteraction_.pair_.addHost(idxA, idxB);
            LSBondedInteraction_.bond_.addHost(point_b[i], normal_b[i]);
            LSBondedInteraction_.bondRadius_.pushHost(radius_b[i]);
            LSBondedInteraction_.bondLength_.pushHost(length_b[i]);
            double3 pointA_global = point_b[i] + 0.5 * length_b[i] * normal_b[i];
            double3 pointB_global = point_b[i] - 0.5 * length_b[i] * normal_b[i];
            double3 pointA_local = reverseRotateVectorByQuaternion(pointA_global - position_p[idxA], orientation_p[idxA]);
            double3 pointB_local = reverseRotateVectorByQuaternion(pointB_global - position_p[idxB], orientation_p[idxB]);
            LSBondedInteraction_.localbondPointA_.pushHost(pointA_local);
            LSBondedInteraction_.localbondPointB_.pushHost(pointB_local);
        }
    }

    virtual void addExternalForceTorque(const double time) 
    {
    }

    virtual bool addInitialCondition() 
    {
        return false;
    }

public:
    void setFriction(const int materialIndexA,
    const int materialIndexB,
    const double slidingFrictionCoefficient)
    {
        if (materialIndexA < 0) return;
        if (materialIndexB < 0) return;
        frictionRow row;
        row.materialIndexA = materialIndexA;
        row.materialIndexB = materialIndexB;
        row.slidingFrictionCoefficient = 0.;
        row.rollingFrictionCoefficient = 0.;
        row.torsionFrictionCoefficient = 0.;
        if (slidingFrictionCoefficient > 0.) row.slidingFrictionCoefficient = slidingFrictionCoefficient;
        frictionTable_.push_back(row);
    }

    void setLinearStiffness(const int materialIndexA,
    const int materialIndexB,
    const double normalStiffness,
    const double slidingStiffness)
    {
        if (materialIndexA < 0) return;
        if (materialIndexB < 0) return;
        linearStiffnessRow row;
        row.materialIndexA = materialIndexA;
        row.materialIndexB = materialIndexB;
        row.normalStiffness = 0.;
        row.slidingStiffness = 0.;
        row.rollingStiffness = 0.;
        row.torsionStiffness = 0.;
        if (normalStiffness > 0.) row.normalStiffness = normalStiffness;
        if (slidingStiffness > 0.) row.slidingStiffness = slidingStiffness;
        linearStiffnessTable_.push_back(row);
    }

    void setBond(const int materialIndexA,
    const int materialIndexB,
    const double bondYoungsModulus,
    const double bondPoissonRatio,
    const double tensileStrength,
    const double cohesion,
    const double frictionCoefficient)
    {
        if (materialIndexA < 0) return;
        if (materialIndexB < 0) return;
        bondRow row;
        row.materialIndexA = materialIndexA;
        row.materialIndexB = materialIndexB;
        row.bondYoungsModulus = 0.;
        row.normalToShearStiffnessRatio = 1.;
        row.tensileStrength = 0.;
        row.cohesion = 0.;
        row.frictionCoefficient = 0.;
        if (bondYoungsModulus > 0.) row.bondYoungsModulus = bondYoungsModulus;
        if (bondPoissonRatio > -0.5) row.normalToShearStiffnessRatio = 1 + 2. * bondPoissonRatio;
        if (tensileStrength > 0.) row.tensileStrength = tensileStrength;
        if (cohesion > 0.) row.cohesion = cohesion;
        if (frictionCoefficient > 0.) row.frictionCoefficient = frictionCoefficient;
        bondTable_.push_back(row);
    }

    void addLSParticle(const std::vector<double3>& globalPosition_boundaryNode,
    const std::vector<double>& gridNodeLSFV, //The grid node index is calculated by gN_x + gridNodeSize.x * (gN_y + gridNodeSize.y * gN_z)
    const double3 gridNodeGlobalOrigin,
    const int3 gridNodeSize,
    const double gridSpacing,
    const int materialID,
    const double density,
    const double3 velocity,
    const double3 angularvelocity)
    {
        if (gridNodeSize.x < 2 || gridNodeSize.y < 2 || gridNodeSize.z < 2 || gridSpacing <= 0.) return;
        if (gridNodeSize.x * gridNodeSize.y * gridNodeSize.z != gridNodeLSFV.size()) return;

        double mass = 0.;
        double3 center = make_double3(0., 0., 0.);
        symMatrix I = make_symMatrix(0., 0., 0., 0., 0., 0.);

        auto smoothHeaviside = [&](const double phi_dimensionless, const double smoothParameter) -> double
        {
            if (smoothParameter <= 0.0) return (phi_dimensionless > 0.0) ? 0.0 : 1.0;

            if (phi_dimensionless < -smoothParameter) return 1.0;
            if (phi_dimensionless > smoothParameter) return 0.0;

            const double x = -phi_dimensionless / smoothParameter;
            return 0.5 * (1.0 + x + std::sin(M_PI * x) / M_PI);
        };

        const double m_gridNode = density * gridSpacing * gridSpacing * gridSpacing;
        for (int x = 0; x < gridNodeSize.x; x++)
        {
            for (int y = 0; y < gridNodeSize.y; y++)
            {
                for (int z = 0; z < gridNodeSize.z; z++)
                {
                    const int index = x + gridNodeSize.x * (y + gridNodeSize.y * z);
                    const double H = smoothHeaviside(gridNodeLSFV[index] / gridSpacing, 1.5);
                    mass += H;
                }
            }
        }
        mass *= m_gridNode;

        double invM = 0.;
        symMatrix invI = make_symMatrix(0., 0., 0., 0., 0., 0.);
        if (mass > 0.)
        {
            invM = 1. / mass;
            for (int x = 0; x < gridNodeSize.x; x++)
            {
                for (int y = 0; y < gridNodeSize.y; y++)
                {
                    for (int z = 0; z < gridNodeSize.z; z++)
                    {
                        const int index = x + gridNodeSize.x * (y + gridNodeSize.y * z);
                        const double H = smoothHeaviside(gridNodeLSFV[index] / gridSpacing, 1.5);
                        center.x += H * (gridNodeGlobalOrigin.x + double(x) * gridSpacing);
                        center.y += H * (gridNodeGlobalOrigin.y + double(y) * gridSpacing);
                        center.z += H * (gridNodeGlobalOrigin.z + double(z) * gridSpacing);
                    }
                }
            }
            center *= m_gridNode / mass;

            for (int x = 0; x < gridNodeSize.x; x++)
            {
                for (int y = 0; y < gridNodeSize.y; y++)
                {
                    for (int z = 0; z < gridNodeSize.z; z++)
                    {
                        const int index = x + gridNodeSize.x * (y + gridNodeSize.y * z);
                        const double H = smoothHeaviside(gridNodeLSFV[index] / gridSpacing, 1.5);
                        double3 r = gridNodeGlobalOrigin + gridSpacing * make_double3(double(x), double(y), double(z)) - center;
                        I.xx += H * (r.y * r.y + r.z * r.z) * m_gridNode;
                        I.yy += H * (r.x * r.x + r.z * r.z) * m_gridNode;
                        I.zz += H * (r.y * r.y + r.x * r.x) * m_gridNode;
                        I.xy -= H * r.x * r.y * m_gridNode;
                        I.xz -= H * r.x * r.z * m_gridNode;
                        I.yz -= H * r.y * r.z * m_gridNode;
                    }
                }
            }
            invI = inverse(I);
        }
        else 
        {
            center = gridNodeGlobalOrigin + 0.5 * gridSpacing * make_double3(double(gridNodeSize.x), double(gridNodeSize.y), double(gridNodeSize.z));
        }


        double radii = 0.;
        int particleID = static_cast<int>(levelSetParticle_.hostSize());
        for (const auto& p : globalPosition_boundaryNode)
        {
            levelSetBoundaryNode_.addHost(p - center, particleID);
            radii = std::max(length(p - center), radii);
        }

        for (const auto& p : gridNodeLSFV)
        {
            levelSetGridNode_.addHost(p, particleID);
        }

        int prefixSum_b = static_cast<int>(levelSetBoundaryNode_.hostSize());
        int prefixSum_g = static_cast<int>(levelSetGridNode_.hostSize());
        levelSetParticle_.addHost(center, 
        velocity, 
        angularvelocity, 
        make_double3(0., 0., 0.), 
        make_double3(0., 0., 0.), 
        radii, 
        invM, 
        make_quaternion(1., 0., 0., 0.), 
        invI, 
        materialID, 
        gridNodeGlobalOrigin - center, 
        gridSpacing, 
        gridNodeSize, 
        prefixSum_g, 
        prefixSum_b, 
        -1,
        -1);
    }

    void addFixedLSParticle(const std::vector<double3>& globalPosition_boundaryNode,
    const std::vector<double>& gridNodeLSFV,
    const double3 globalOrigin_grid,
    const int3 gridNodeSize,
    const double gridSpacing,
    const int materialID)
    {
        addLSParticle(globalPosition_boundaryNode,
        gridNodeLSFV,
        globalOrigin_grid,
        gridNodeSize,
        gridSpacing,
        materialID,
        0.,
        make_double3(0., 0., 0.),
        make_double3(0., 0., 0.));
    }

    void addLSSuperellipsoid(const double rx, 
    const double ry, 
    const double rz, 
    const double ee, 
    const double en, 
    const int materialID,
    const double density,
    const double3 position,
    const double3 velocity,
    const double3 angularvelocity,
    const int nPoints = 10000)
    {
        superellipsoidParams s;
        s.rx = rx;
        s.ry = ry;
        s.rz = rz;
        s.ee = ee;
        s.en = en;
        LevelSetParticleGridInfo ls = buildLevelSetSuperellipsoidGridGlobal(s, position);
        std::vector<double3> sp = generateSuperellipsoidSurfacePointsGlobal_Uniform(s, position, nPoints);
        addLSParticle(sp, 
        ls.gridNodeLSF, 
        ls.gridMin, 
        ls.gridNodeSize, 
        ls.gridSpacing, 
        materialID, 
        density, 
        velocity, 
        angularvelocity);
    }

    void setLevelSetBox(const double3 boxMin_global,
    const double3 boxMax_global,
    const int paddingLayers = 2)
    {
        const double3 bmin = make_double3(std::min(boxMin_global.x, boxMax_global.x),
        std::min(boxMin_global.y, boxMax_global.y),
        std::min(boxMin_global.z, boxMax_global.z));

        const double3 bmax = make_double3(std::max(boxMin_global.x, boxMax_global.x),
        std::max(boxMin_global.y, boxMax_global.y),
        std::max(boxMin_global.z, boxMax_global.z));

        const double dx = bmax.x - bmin.x;
        const double dy = bmax.y - bmin.y;
        const double dz = bmax.z - bmin.z;

        const double diameter = std::max(dx, std::max(dy, dz));
        if (diameter <= 0.0) return;

        LevelSetParticleGridInfo info;
        const double h = diameter / 20.0;
        info.gridSpacing = h;

        const double pad = double(std::max(1, paddingLayers)) * h;

        double3 gmin = make_double3(bmin.x - pad, bmin.y - pad, bmin.z - pad);
        double3 gmax = make_double3(bmax.x + pad, bmax.y + pad, bmax.z + pad);

        auto snapDown = [&](double a) { return h * std::floor(a / h); };
        auto snapUp = [&](double a) { return h * std::ceil (a / h); };

        gmin.x = snapDown(gmin.x); gmin.y = snapDown(gmin.y); gmin.z = snapDown(gmin.z);
        gmax.x = snapUp  (gmax.x); gmax.y = snapUp (gmax.y); gmax.z = snapUp  (gmax.z);

        const int nx = int(std::llround((gmax.x - gmin.x) / h)) + 1;
        const int ny = int(std::llround((gmax.y - gmin.y) / h)) + 1;
        const int nz = int(std::llround((gmax.z - gmin.z) / h)) + 1;

        if (nx <= 0 || ny <= 0 || nz <= 0) return info;

        info.gridMin = gmin;
        info.gridNodeSize = make_int3(nx, ny, nz);
        info.gridNodeLSF.assign(size_t(nx) * size_t(ny) * size_t(nz), 0.0);

        auto idx3 = [&](int i, int j, int k) -> size_t
        {
            return size_t((k * ny + j) * nx + i);
        };

        // ---------------------------------------------------------
        // Signed distance to an axis-aligned box (SDF)
        // Reference: iq's box SDF (adapted to "inside positive")
        // We first compute outside distance (>=0) and inside distance (<=0),
        // then flip sign so that inside is positive.
        // ---------------------------------------------------------
        for (int k = 0; k < nz; ++k)
        {
            const double z = gmin.z + double(k) * h;

            for (int j = 0; j < ny; ++j)
            {
                const double y = gmin.y + double(j) * h;

                for (int i = 0; i < nx; ++i)
                {
                    const double x = gmin.x + double(i) * h;
                    const double3 p = make_double3(x, y, z);

                    // Box center and half extents
                    const double3 c = 0.5 * (bmin + bmax);
                    const double3 e = 0.5 * (bmax - bmin);

                    // q = abs(p - c) - e
                    const double3 d = make_double3(std::fabs(p.x - c.x) - e.x,
                                                std::fabs(p.y - c.y) - e.y,
                                                std::fabs(p.z - c.z) - e.z);

                    // outside distance
                    const double3 d_out = make_double3(std::max(d.x, 0.0),
                                                    std::max(d.y, 0.0),
                                                    std::max(d.z, 0.0));
                    const double outside = length(d_out);

                    // inside distance (negative or zero)
                    const double inside = std::min(std::max(d.x, std::max(d.y, d.z)), 0.0);

                    // Standard box SDF: sdf = outside + inside (inside <= 0)
                    const double sdf = outside + inside;

                    // Convention required:
                    //   inside  -> positive
                    //   outside -> negative
                    info.gridNodeLSF[idx3(i, j, k)] = -sdf;
                }
            }
        }

        for (const auto& ptr: info.gridNodeLSF)
        {
            wallLSFV_.pushHost(ptr);
        }
        wallGridNodeOrigin_ = info.gridMin;
        wallGridNodeSize_ = info.gridNodeSize;
        wallGridSpacing_ = info.gridSpacing;
    }

    void solve(const double3 minBoundary, 
    const double3 maxBoundary, 
    const double3 gravity, 
    const double timeStep, 
    const double maximumTime,
    const size_t numFrame,
    const std::string dir, 
    const size_t deviceID = 0, 
    const size_t maximumThread = 256)
    {
        removeVtuFiles(dir);
        removeDatFiles(dir);

        cudaError_t cudaStatus = cudaSetDevice(deviceID);
        if (cudaStatus != cudaSuccess) 
        {
            std::cout << "cudaSetDevice( " << deviceID
            << " ) failed! Do you have a CUDA-capable GPU installed?"
            << std::endl;
            exit(1);
        }

        if (maxBoundary.x < minBoundary.x || maxBoundary.y < minBoundary.y ||maxBoundary.z < minBoundary.z)
        {
            std::cout << "failed! Boundary setting is incorrect" << std::endl;
            return;           
        }
        
        if (timeStep <= 0.) 
        {
            std::cout << "failed! Time step is less than 0" << std::endl;
            return;
        }
        size_t numStep = size_t(maximumTime / timeStep) + 1;
        size_t frameInterval = numStep;
        if (numFrame > 0) frameInterval = numStep / numFrame;
        if (frameInterval < 1) frameInterval = 1;
        
        maxThread_ = maximumThread;
        LSDEMInitialize(minBoundary, maxBoundary);
        updateLSSpatialGridCellHashStartEnd();
        LSParticleContactDetection();
        if (addInitialCondition()) LSDEMInitialize(minBoundary, maxBoundary);

        if (levelSetParticle_.deviceSize() == 0)
        {
            std::cout << "failed! The number of particles is equal to 0" << std::endl;
            return;
        }

        std::cout << "Initialization Completed." << std::endl;

        size_t iStep = 0, iFrame = 0;
        double time = 0.0;
        outputLevelSetParticleVTU(dir, 0, 0, 0.0);
        outputLevelSetParticleInteractionVTU(dir, 0, 0, 0.0);
        const double halfTimeStep = 0.5 * timeStep;
        while (iStep <= numStep)
        {
            iStep++;
            time += timeStep;
            if (iStep % 10 == 0) updateLSSpatialGridCellHashStartEnd();
            LSParticleContactDetection();
            LSParticleContactForceTorque(halfTimeStep);
            LSParticleIntegration1stHalf(gravity, halfTimeStep);
            LSParticleContactDetection();
            LSParticleContactForceTorque(halfTimeStep);
            LSParticleIntegration2ndHalf(gravity, halfTimeStep);
            if (iStep % frameInterval == 0)
            {
                iFrame++;
                std::cout << "Frame " << iFrame << " at Time " << time << std::endl;
                outputLevelSetParticleVTU(dir, iFrame, iStep, time);
                outputLevelSetParticleInteractionVTU(dir, iFrame, iStep, time);
            }
        }
        levelSetParticle_.copyDeviceToHost(stream_);
        LSBondedInteraction_.bond_.copyDeviceToHost(stream_);
    }

private:
    cudaStream_t stream_;
    size_t maxThread_;

    std::vector<materialRow> materialTable_;
    std::vector<frictionRow> frictionTable_;
    std::vector<linearStiffnessRow> linearStiffnessTable_;
    std::vector<bondRow> bondTable_;
    contactParameters contactParameters_;

    levelSetBoundaryNode levelSetBoundaryNode_;
    levelSetGridNode levelSetGridNode_;
    levelSetParticle levelSetParticle_;
    spatialGrid spatialGrid_;

    solidInteraction LSParticleInteraction_;
    LSBondedInteraction LSBondedInteraction_;

    HostDeviceArray1D<double> wallLSFV_;
    double3 wallGridNodeOrigin_;
    int3 wallGridNodeSize_;
    double wallGridSpacing_;
};