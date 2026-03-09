#include "DEMSolver.h"
#include "CUDAKernelFunction/levelSetParticleContactDetectionKernel.cuh"
#include "CUDAKernelFunction/levelSetParticleIntegrationKernel.cuh"

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

        if (levelSetParticle_.deviceSize() > 0 && LSParticleInteraction_.numActivated_ == 0)
        {
            LSParticleInteraction_.objectPointed_.allocateDevice(levelSetBoundaryNode_.deviceSize(), stream_);
            LSParticleInteraction_.objectPointing_.allocateDevice(levelSetParticle_.deviceSize(), stream_);
        }
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
        std::vector<double3> t_c = LSParticleInteraction_.contact_.torqueHostCopy();
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
            { "contact torque" , t_c   },
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

    void addLSParticle(const std::vector<double3>& globalPosition_boundaryNode,
    const std::vector<double>& gridNodeLSFV, //The grid number is calculated by gN_x + gridNodeSize.x * (gN_y + gridNodeSize.y * gN_z)
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
};