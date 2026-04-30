#pragma once
#include "LSParticle.h"
#include "SolidInteraction.h"
#include "VBondedInteraction.h"
#include "PeriodicBoundary.h"
#include "CUDAKernelFunction/LSParticleContactDetectionKernel.cuh"
#include "CUDAKernelFunction/contactKernel.cuh"
#include "CUDAKernelFunction/particleIntegrationKernel.cuh"
#include <filesystem>

class LSSolver
{
public:
    LSSolver(const std::string dir = "Problem", cudaStream_t stream = 0, const size_t maxGPUThread = 256, const int device = 0)
    {
        dir_ = dir;
        phase_ = 0;
        stream_ = stream;
        maxGPUThread_ = maxGPUThread;
        device_ = device;
    }

    ~LSSolver() = default;

    /**
    * @brief Add a particle material to the LSParticle system.
    *
    * @param normalStiffness        Normal contact stiffness.
    * @param shearStiffness         Tangential (shear) stiffness.
    * @param frictionCoefficient    Coulomb friction coefficient.
    * @param restitutionCoefficient Restitution coefficient (0~1).
    */
    void addParticleMaterial(const double normalStiffness,
    const double shearStiffness,
    const double frictionCoefficient,
    const double restitutionCoefficient)
    {
        LSParticle_.addMaterial(normalStiffness, 
        shearStiffness, 
        frictionCoefficient, 
        restitutionCoefficient);
    }

    /**
    * @brief Add a particle geometry (Level Set + boundary mesh).
    *
    * This defines:
    * - Level-set grid (SDF)
    * - Boundary nodes (surface mesh)
    * - Mass & inertia
    *
    * @param mass                         Particle mass.
    * @param inertiaTensor                Inertia tensor.
    * @param gridNodeOrigin               Origin of level-set grid.
    * @param gridNodeSpacing              Grid spacing.
    * @param gridNodeSize                 Grid resolution (Nx, Ny, Nz).
    * @param gridNodeSignedDistance       Flattened SDF array.
    * @param boundaryNodePosition         Surface vertex positions.
    * @param boundaryNodeConnectivity     Triangle connectivity.
    */
    void addParticleGeometry(const double mass,
    const symMatrix inertiaTensor,
    const double3 gridNodeOrigin,
    const double gridNodeSpacing,
    const int3 gridNodeSize,
    const std::vector<double>& gridNodeSignedDistance,
    const std::vector<double3>& boundaryNodePosition,
    const std::vector<int3>& boundaryNodeConnectivity)
    {
        LSParticle_.addGeometry(mass,
        inertiaTensor,
        gridNodeOrigin,
        gridNodeSpacing,
        gridNodeSize,
        gridNodeSignedDistance,
        boundaryNodePosition, 
        boundaryNodeConnectivity);
    }

    /**
    * @brief Add a single particle instance.
    *
    * @param position       World position.
    * @param velocity       Linear velocity.
    * @param angularVelocity Angular velocity.
    * @param orientation    Quaternion orientation.
    * @param materialID     Material index.
    * @param geometryID     Geometry index.
    */
    void addParticle(const double3 position,
    const double3 velocity,
    const double3 angularVelocity,
    const quaternion orientation,
    const int materialID,
    const int geometryID)
    {
        LSParticle_.addParticle(position, 
        velocity, 
        angularVelocity, 
        orientation, 
        materialID, 
        geometryID);
    }

    /**
    * @brief Add a cluster of particles sharing the same geometry.
    *
    * Automatically:
    * - Builds geometry (SDF + mesh) from input parameters
    * - Computes mass properties from the SDF
    *
    * @param position               Positions of particles.
    * @param velocity               Velocities of particles.
    * @param angularVelocity        Angular velocities.
    * @param orientation            Orientations.
    * @param materialID             Material index.
    * @param density                Density (used to compute mass).
    * @param gridNodeOrigin         SDF grid origin.
    * @param gridNodeSpacing        SDF spacing.
    * @param gridNodeSize           Grid resolution.
    * @param gridNodeSignedDistance SDF values.
    * @param boundaryNodePosition   Surface vertices.
    * @param boundaryNodeConnectivity Triangle connectivity.
    */
    void addParticleCluster(const std::vector<double3>& position,
    const std::vector<double3>& velocity,
    const std::vector<double3>& angularVelocity,
    const std::vector<quaternion>& orientation,
    const int materialID,
    const double density,
    const double3 gridNodeOrigin,
    const double gridNodeSpacing,
    const int3 gridNodeSize,
    const std::vector<double>& gridNodeSignedDistance,
    const std::vector<double3>& boundaryNodePosition,
    const std::vector<int3>& boundaryNodeConnectivity)
    {
        if (position.size() != velocity.size() || position.size() != angularVelocity.size() || position.size() != orientation.size())
        {
            std::cerr << "[LSSolver] Inconsistent particle cluster data size. "
                    << "Position, velocity, angular velocity, and orientation vectors must have the same size."
                    << std::endl;
            return;
        }
        
        double3 localOffset;
        addParticleGeometry(localOffset,
        density,
        gridNodeOrigin,
        gridNodeSpacing,
        gridNodeSize,
        gridNodeSignedDistance,
        boundaryNodePosition,
        boundaryNodeConnectivity);

        for (size_t i = 0; i < position.size(); ++i)
        {
            addParticle(position[i] + rotateVectorByQuaternion(orientation[i], localOffset),
            velocity[i], 
            angularVelocity[i], 
            orientation[i], 
            materialID, 
            LSParticle_.geometryInfo_.num() - 1);
        }
    }

    /**
    * @brief Add a wall material.
    *
    * Walls are treated as infinite mass (no deformation).
    *
    * @param frictionCoefficient    Friction coefficient.
    * @param restitutionCoefficient Restitution coefficient.
    */
    void addWallMaterial(const double frictionCoefficient,
    const double restitutionCoefficient)
    {
        Wall_.addMaterial(0.0, 
        0.0, 
        frictionCoefficient, 
        restitutionCoefficient);
    }

    /**
    * @brief Add wall geometry (Level Set + boundary mesh).
    *
    * @param gridNodeOrigin           Grid origin.
    * @param gridNodeSpacing          Grid spacing.
    * @param gridNodeSize             Grid resolution.
    * @param gridNodeSignedDistance   SDF values.
    * @param boundaryNodePosition     Surface vertices.
    * @param boundaryNodeConnectivity Triangle connectivity.
    */
    void addWallGeometry(const double3 gridNodeOrigin,
    const double gridNodeSpacing,
    const int3 gridNodeSize,
    const std::vector<double>& gridNodeSignedDistance,
    const std::vector<double3>& boundaryNodePosition,
    const std::vector<int3>& boundaryNodeConnectivity)
    {
        Wall_.addGeometry(0.0,
        symMatrix(),
        gridNodeOrigin,
        gridNodeSpacing,
        gridNodeSize,
        gridNodeSignedDistance,
        boundaryNodePosition,
        boundaryNodeConnectivity);
    }

    /**
    * @brief Add a wall instance.
    *
    * @param position        Position.
    * @param velocity        Velocity.
    * @param angularVelocity Angular velocity.
    * @param orientation     Orientation.
    * @param materialID      Material index.
    * @param geometryID      Geometry index.
    */
    void addWall(const double3 position,
    const double3 velocity,
    const double3 angularVelocity,
    const quaternion orientation,
    const int materialID,
    const int geometryID)
    {
        Wall_.addParticle(position, 
        velocity, 
        angularVelocity, 
        orientation, 
        materialID, 
        geometryID);
    }

    /**
    * @brief Translate a level-set particle by a given offset.
    *
    * @param index  Particle index.
    * @param offset Translation vector in WORLD frame.
    */
    void moveLSParticle(const size_t index, const double3 offSet) 
    { 
        LSParticle_.move(index, offSet); 
    }

    /**
    * @brief Translate a wall object by a given offset.
    *
    * @param index  Wall index.
    * @param offset Translation vector in WORLD frame.
    */
    void moveWall(const size_t index, const double3 offset)
    {
        Wall_.move(index, offset);
    }

    /**
    * @brief Assign a fixed translational velocity to a wall object.
    *
    * The wall moves at this constant velocity each time step.
    *
    * @param index    Wall index.
    * @param velocity Translational velocity in WORLD frame.
    */
    void setFixedVelocityToWall(const size_t index, const double3 velocity)
    {
        Wall_.setVelocity(index, velocity);
    }

    /**
    * @brief Assign a fixed angular velocity to a wall object.
    *
    * The wall rotates at this constant angular velocity each time step.
    *
    * @param index           Wall index.
    * @param angularVelocity Angular velocity in WORLD frame.
    */
    void setFixedAngularVelocityToWall(const size_t index, const double3 angularVelocity)
    {
        Wall_.setAngularVelocity(index, angularVelocity);
    }

    /**
    * @brief Add a single bonded interaction between two level-set particles.
    *
    * Creates a virtual parallel bond between the specified master and slave
    * particles at the given bond point and orientation. The bond stiffness
    * coefficients are derived from beam theory using the provided geometry
    * and material parameters.
    *
    * @param masterObjectID      Index of the master particle.
    * @param slaveObjectID       Index of the slave particle.
    * @param bondPoint           Bond center point in WORLD frame.
    * @param bondNormal          Bond axis direction in WORLD frame (need not be normalized).
    * @param radius              Bond cross-section radius (must be > 0).
    * @param initialLength       Bond initial length (must be > 0).
    * @param YoungsModulus       Bond Young's modulus (must be > 0).
    * @param poissonRatio        Bond Poisson ratio (must be in (-1, 0.5)).
    * @param tensileStrength     Bond tensile strength. Default = 0 (no tensile failure).
    * @param cohesion            Bond cohesion. Default = 0 (no shear failure).
    * @param frictionCoefficient Bond friction coefficient. Default = 0.
    *
    * @note Does nothing if either particle index is out of range.
    */
    void addBondedInteraction(const int masterObjectID, 
    const int slaveObjectID, 
    const double3 bondPoint, 
    const double3 bondNormal, 
    const double radius, 
    const double initialLength, 
    const double YoungsModulus, 
    const double poissonRatio, 
    const double tensileStrength = 0., 
    const double cohesion= 0., 
    const double frictionCoefficient= 0.)
    {
        if (masterObjectID >= LSParticle_.num() || slaveObjectID >= LSParticle_.num()) return;
        VBondedInteraction_.add(masterObjectID, 
        slaveObjectID, 
        LSParticle_.positionHostRef()[masterObjectID], 
        LSParticle_.positionHostRef()[slaveObjectID], 
        LSParticle_.orientationHostRef()[masterObjectID], 
        LSParticle_.orientationHostRef()[slaveObjectID], 
        bondPoint, 
        bondNormal, 
        radius, 
        initialLength, 
        YoungsModulus, 
        poissonRatio, 
        tensileStrength, 
        cohesion, 
        frictionCoefficient);
    }

     /**
     * @brief Create bonded interactions between free level-set particles.
     *
     * @param YoungsModulus Bond Young's modulus.
     * @param poissonRatio Bond Poisson ratio.
     * @param radius Bond creation radius.
     * @param initialLength Bond creation length.
     * @param tensileStrength Bond tensile strength.
     * @param cohesion Bond cohesion.
     * @param frictionCoefficient Bond friction coefficient.
     */
    void createBondsFromLSInteractions(const double radius, 
    const double initialLength, 
    const double YoungsModulus, 
    const double poissonRatio, 
    const double tensileStrength = 0., 
    const double cohesion= 0., 
    const double frictionCoefficient= 0.)
    {
        const std::vector<int> masterBoundaryNodeID = LSParticleInteraction_.masterIDHostRef();
        const std::vector<int> slaveParticleID = LSParticleInteraction_.slaveIDHostRef();
        const std::vector<double3> point = LSParticleInteraction_.contactPointHostRef();
        const std::vector<double3> normal = LSParticleInteraction_.contactNormalHostRef();

        for(size_t k = 0; k < point.size(); k++)
        {
            const int i = LSParticle_.boundaryNodeInfo_.particleIDHostRef()[masterBoundaryNodeID[k]];
            const int j = slaveParticleID[k];
            VBondedInteraction_.add(i, 
            j, 
            LSParticle_.positionHostRef()[i], 
            LSParticle_.positionHostRef()[j], 
            LSParticle_.orientationHostRef()[i], 
            LSParticle_.orientationHostRef()[j], 
            point[k], 
            normal[k], 
            radius, 
            initialLength, 
            YoungsModulus, 
            poissonRatio, 
            tensileStrength, 
            cohesion, 
            frictionCoefficient);
        }
    }

    /**
    * @brief Enable periodic boundary condition in the X direction.
    *
    * Particles leaving one side of the domain in X are mirrored as ghost
    * particles on the opposite side to maintain periodicity.
    */
    void addPeriodicBoundaryXD()
    {
        PeriodicBoundaryXY2D_.turnXOn();
    }

    /**
    * @brief Enable periodic boundary condition in the Y direction.
    *
    * Particles leaving one side of the domain in Y are mirrored as ghost
    * particles on the opposite side to maintain periodicity.
    */
    void addPeriodicBoundaryYD()
    {
        PeriodicBoundaryXY2D_.turnYOn();
    }

    /**
    * @brief Enable periodic boundary condition for a sector geometry.
    *
    * Particles are replicated at 90°, 180°, and 270° rotations to enforce
    * rotational periodicity within a quarter-sector domain.
    */
    void addPeriodicBoundarySector()
    {
        PeriodicBoundarySector_.turnOn();
    }

    /**
    * @brief Disable all active periodic boundary conditions.
    *
    * Turns off both XY planar periodicity (X and Y directions) and
    * sector rotational periodicity.
    */
    void removePeriodicBoundary()
    {
        PeriodicBoundarySector_.turnOff();
        PeriodicBoundaryXY2D_.turnXOff();
        PeriodicBoundaryXY2D_.turnYOff();
    }

    /**
     * @brief Run the solver time integration loop.
     *
     * @param minDomain Minimum simulation domain corner.
     * @param maxDomain Maximum simulation domain corner.
     * @param gravity Gravity acceleration.
     * @param timeStep Time step size.
     * @param maximumTime Maximum physical simulation time.
     * @param numFrame Requested number of output frames.
     * @param argc Argument count from main().
     * @param argv Argument array from main().
     */
    void solve(const double3 minDomain, 
    const double3 maxDomain, 
    const double3 gravity, 
    const double timeStep, 
    const double maximumTime, 
    const size_t numFrame, 
    const int argc,
    char** argv)
    {
        if (!checkSolverInput(minDomain, maxDomain, timeStep, maximumTime)) return;
        if (!activateGPUDevice()) return;
        phase_ += 1;
        updateDir(argc, argv);
        removeFiles();
        upload(minDomain, maxDomain);
        
        const size_t numStep = size_t(maximumTime / timeStep) + 1;
        size_t frameInterval = numStep;
        if (numFrame > 0) frameInterval = numStep / numFrame;
        if (frameInterval < 1) frameInterval = 1;
        
        bool addBondFlag = VBondedInteraction_.numPair_device() > 0;
        bool addPeriodicBoundaryFlag = (PeriodicBoundaryXY2D_.isActived() || PeriodicBoundarySector_.isActived());
        if ((!addBondFlag) && (!addPeriodicBoundaryFlag)) compute(gravity, timeStep, numStep, frameInterval);
        else if (addBondFlag && (!addPeriodicBoundaryFlag)) compute_addBond(gravity, timeStep, numStep, frameInterval);
        else if ((!addBondFlag) && addPeriodicBoundaryFlag) compute_addPeriodicBoundary(gravity, timeStep, numStep, frameInterval);
        else compute_addBondAndPeriodicBoundary(gravity, timeStep, numStep, frameInterval);
    }

protected:
    void addParticleGeometry(double3& localOffset,
    const double density,
    const double3 gridNodeOrigin,
    const double gridNodeSpacing,
    const int3 gridNodeSize,
    const std::vector<double>& gridNodeSignedDistance,
    const std::vector<double3>& boundaryNodePosition,
    const std::vector<int3>& boundaryNodeConnectivity)
    {
        localOffset = make_double3(0., 0., 0.);

        if (gridNodeSize.x < 2 || gridNodeSize.y < 2 || gridNodeSize.z < 2)
        {
            std::cerr << "[LSSolver] Invalid grid node size: " << gridNodeSize.x << " x " << gridNodeSize.y << " x " << gridNodeSize.z 
            << ". Each dimension must be at least 2." << std::endl;
            return;
        }

        if (gridNodeSpacing <= 0.0)
        {
            std::cerr << "[LSSolver] Invalid grid node spacing: " << gridNodeSpacing << ". Must be greater than 0." << std::endl;
            return;
        }

        if (static_cast<size_t>(gridNodeSize.x) * static_cast<size_t>(gridNodeSize.y) * static_cast<size_t>(gridNodeSize.z) != gridNodeSignedDistance.size())
        {
            std::cerr << "[LSSolver] Inconsistent grid node data size. Expected " 
            << static_cast<size_t>(gridNodeSize.x) * static_cast<size_t>(gridNodeSize.y) * static_cast<size_t>(gridNodeSize.z) 
            << " signed distance values, but got " << gridNodeSignedDistance.size() << "."
            << std::endl;
            return;
        }

        double mass = 0.;
        auto smoothHeaviside = [](const double phi_dimensionless, const double smoothParameter) -> double
        {
            if (smoothParameter <= 0.0) return (phi_dimensionless > 0.0) ? 0.0 : 1.0;

            if (phi_dimensionless < -smoothParameter) return 1.0;
            if (phi_dimensionless > smoothParameter) return 0.0;

            const double x = -phi_dimensionless / smoothParameter;
            return 0.5 * (1.0 + x + std::sin(pi() * x) / pi());
        };
        const double m_gridNode = density * gridNodeSpacing * gridNodeSpacing * gridNodeSpacing;
        for (int x = 0; x < gridNodeSize.x; x++)
        {
            for (int y = 0; y < gridNodeSize.y; y++)
            {
                for (int z = 0; z < gridNodeSize.z; z++)
                {
                    const int index = linearIndex3D(make_int3(x, y, z), gridNodeSize);
                    const double H = smoothHeaviside(gridNodeSignedDistance[index] / gridNodeSpacing, 1.5);
                    mass += H;
                }
            }
        }
        mass *= m_gridNode;

        double3 centroid = make_double3(0., 0., 0.);
        symMatrix inertiaTensor = symMatrix();
        if (mass > 0.)
        {
            for (int x = 0; x < gridNodeSize.x; x++)
            {
                for (int y = 0; y < gridNodeSize.y; y++)
                {
                    for (int z = 0; z < gridNodeSize.z; z++)
                    {
                        const int index = linearIndex3D(make_int3(x, y, z), gridNodeSize);
                        const double H = smoothHeaviside(gridNodeSignedDistance[index] / gridNodeSpacing, 1.5);
                        centroid.x += H * (gridNodeOrigin.x + double(x) * gridNodeSpacing);
                        centroid.y += H * (gridNodeOrigin.y + double(y) * gridNodeSpacing);
                        centroid.z += H * (gridNodeOrigin.z + double(z) * gridNodeSpacing);
                    }
                }
            }
            centroid *= m_gridNode / mass;

            for (int x = 0; x < gridNodeSize.x; x++)
            {
                for (int y = 0; y < gridNodeSize.y; y++)
                {
                    for (int z = 0; z < gridNodeSize.z; z++)
                    {
                        const int index = linearIndex3D(make_int3(x, y, z), gridNodeSize);
                        const double H = smoothHeaviside(gridNodeSignedDistance[index] / gridNodeSpacing, 1.5);
                        double3 r = gridNodeOrigin + gridNodeSpacing * make_double3(double(x), double(y), double(z)) - centroid;
                        inertiaTensor.xx += H * (r.y * r.y + r.z * r.z) * m_gridNode;
                        inertiaTensor.yy += H * (r.x * r.x + r.z * r.z) * m_gridNode;
                        inertiaTensor.zz += H * (r.y * r.y + r.x * r.x) * m_gridNode;
                        inertiaTensor.xy -= H * r.x * r.y * m_gridNode;
                        inertiaTensor.xz -= H * r.x * r.z * m_gridNode;
                        inertiaTensor.yz -= H * r.y * r.z * m_gridNode;
                    }
                }
            }
        }

        std::vector<double3> boundaryNodePosition_new;
        boundaryNodePosition_new.reserve(boundaryNodePosition.size());
        for (const auto& p : boundaryNodePosition)
        {
            boundaryNodePosition_new.push_back(p - centroid);
        }
        const double3 gridNodeOrigin_new = gridNodeOrigin - centroid;
        LSParticle_.addGeometry(mass,
        inertiaTensor,
        gridNodeOrigin_new,
        gridNodeSpacing, 
        gridNodeSize,
        gridNodeSignedDistance,
        boundaryNodePosition_new,
        boundaryNodeConnectivity);

        localOffset = centroid;
    }

    virtual void addExternalForceTorque(const double time) {}

    LSParticle& getLSParticle() { return LSParticle_; }

    LSParticle& getWall() { return Wall_; }

    SolidInteraction& getLSParticleInteraction() { return LSParticleInteraction_; }

    SolidInteraction& getLSParticleWallInteraction() { return LSParticleWallInteraction_; }

    VBondedInteraction& getLSParticleBondedInteraction() { return VBondedInteraction_; }

private:
    bool activateGPUDevice()
    {
        cudaError_t cudaStatus = cudaSetDevice(device_);
        if (cudaStatus != cudaSuccess) 
        {
            std::cout << "cudaSetDevice( " << device_ 
            << " ) failed! Do you have a CUDA-capable GPU installed?" 
            << std::endl; 
            exit(1);
            return false;
        }
        return true;
    }

    inline std::filesystem::path getBuildDirectoryFromExecutable(const char* argv0)
    {
        if (argv0 == nullptr) return std::filesystem::current_path();
        std::filesystem::path exePath(argv0);
        if (exePath.is_relative())
        {
            exePath = std::filesystem::absolute(exePath);
        }
        exePath = std::filesystem::weakly_canonical(exePath);
        std::filesystem::path outputDir = exePath.parent_path();
        while (!outputDir.empty() && outputDir.filename() != "build")
        {
            const std::filesystem::path parent = outputDir.parent_path();
            if (parent == outputDir) break;
            outputDir = parent;
        }
        if (!outputDir.empty() && outputDir.filename() == "build")
        {
            return outputDir;
        }
        return exePath.parent_path();
    }

    void updateDir(const int argc, char** argv)
    {
        const char* argv0 = (argc > 0) ? argv[0] : nullptr;
        const std::filesystem::path p(dir_);
        const std::filesystem::path resolvedPath = p.is_absolute() ? p : getBuildDirectoryFromExecutable(argv0) / p;
        dir_ = resolvedPath.string();
    }

    bool checkSolverInput(const double3 minDomain, const double3 maxDomain, const double timeStep, const double maximumTime)
    {
        const double3 domainSize = maxDomain - minDomain;
        if (domainSize.x <= 0.0 || domainSize.y <= 0.0 || domainSize.z <= 0.0)
        {
            std::cerr << "[Solver] Invalid simulation domain size: ("
                    << domainSize.x << ", "
                    << domainSize.y << ", "
                    << domainSize.z << ")."
                    << std::endl;
            return false;
        }

        if (timeStep <= 0.0)
        {
            std::cerr << "[Solver] Invalid timeStep: "
                    << timeStep << "."
                    << std::endl;
            return false;
        }

        if (maximumTime < 0.0)
        {
            std::cerr << "[Solver] Invalid maximumTime: "
                    << maximumTime << "."
                    << std::endl;
            return false;
        }

        return true;
    }

    void removeFiles()
    {
        const std::string dir = dir_ + "_phase" + std::to_string(phase_);
        const std::string dir1 = dir + "/LSParticle";
        const std::string dir2 = dir + "/LSParticleInteraction";
        const std::string dir3 = dir + "/Wall";
        const std::string dir4 = dir + "/LSParticle-WallInteraction";
        MKDIR(dir.c_str());
        removeVtuFiles(dir1);
        removeVtuFiles(dir2);
        removeVtuFiles(dir3);
        removeVtuFiles(dir4);
    }

    void upload(const double3 minDomain, const double3 maxDomain)
    {
        std::cout << "[Solver] Uploading..." << std::endl;
        LSParticle_.initialize(minDomain, maxDomain, maxGPUThread_, stream_);
        LSParticleInteraction_.initialize(LSParticle_.boundaryNodeInfo_.num(), maxGPUThread_, stream_);
        Wall_.initialize(minDomain, maxDomain, maxGPUThread_, stream_);
        LSParticleWallInteraction_.initialize(LSParticle_.boundaryNodeInfo_.num(), maxGPUThread_, stream_);
        VBondedInteraction_.initialize(maxGPUThread_, stream_);
        PeriodicBoundaryXY2D_.initialize(LSParticle_, maxGPUThread_, stream_);
        PeriodicBoundarySector_.initialize(LSParticle_, maxGPUThread_, stream_);
        std::cout << "[Solver] Upload Completed" << std::endl;
    }

    void download()
    {
        std::cout << "[Solver] Downloading..." << std::endl;
        LSParticle_.finalize(stream_);
        LSParticleInteraction_.finalize(stream_);
        Wall_.finalize(stream_);
        LSParticleWallInteraction_.finalize(stream_);
        VBondedInteraction_.finalize(stream_);
        cudaStreamSynchronize(stream_);
        std::cout << "[Solver] Download Completed" << std::endl;
    }

    void output(const size_t iFrame, const size_t iStep, const double time)
    {
        std::cout << "[Solver] Phase " << phase_ << ": ------ Frame " << iFrame << " at Time " << time << " ------ " << std::endl;
        const double memTotal = LSParticle_.deviceMemoryGB()
        + LSParticleInteraction_.deviceMemoryGB()
        + Wall_.deviceMemoryGB()
        + LSParticleWallInteraction_.deviceMemoryGB()
        + VBondedInteraction_.deviceMemoryGB()
        + PeriodicBoundaryXY2D_.deviceMemoryGB()
        + PeriodicBoundarySector_.deviceMemoryGB();
        std::cout << "[Solver] GPU Memory Usage: " << memTotal << " GB" << std::endl;

        download();
        
        std::cout << "[Solver] Outputting... " << std::endl;
        const std::string dir = dir_ + "_phase" + std::to_string(phase_);
        const std::string dir1 = dir + "/LSParticle";
        const std::string dir2 = dir + "/LSParticleInteraction";
        const std::string dir3 = dir + "/Wall";
        const std::string dir4 = dir + "/LSParticle-WallInteraction";
        MKDIR(dir.c_str());
        LSParticle_.outputVTU(dir1, iFrame, iStep, time);
        LSParticle_.outputParticleVTU(dir1, iFrame, iStep, time);
        LSParticleInteraction_.outputVTU(dir2, iFrame, iStep, time);
        VBondedInteraction_.outputVTU(dir2, iFrame, iStep, time);
        Wall_.outputVTU(dir3, iFrame, iStep, time);
        LSParticleWallInteraction_.outputVTU(dir4, iFrame, iStep, time);
        std::cout << "[Solver] Output Completed" << std::endl;
    }

    void updateLSParticleSpatialGrid()
    {
        launchUpdateSpatialGridHashStartEnd(LSParticle_.hashIndex(), 
        LSParticle_.hashValue(), 
        LSParticle_.position(), 
        LSParticle_.spatialGrid_.hashStart(), 
        LSParticle_.spatialGrid_.hashEnd(), 
        LSParticle_.spatialGrid_.minimumBoundary(), 
        LSParticle_.spatialGrid_.maximumBoundary(), 
        LSParticle_.spatialGrid_.inverseCellSize(), 
        LSParticle_.spatialGrid_.size3D(), 
        LSParticle_.spatialGrid_.num_device(), 
        LSParticle_.num_device(), 
        LSParticle_.gridDim(), 
        LSParticle_.blockDim(), 
        stream_);

        launchUpdateSpatialGridHashStartEnd(Wall_.hashIndex(), 
        Wall_.hashValue(), 
        Wall_.position(), 
        Wall_.spatialGrid_.hashStart(), 
        Wall_.spatialGrid_.hashEnd(), 
        Wall_.spatialGrid_.minimumBoundary(), 
        Wall_.spatialGrid_.maximumBoundary(), 
        Wall_.spatialGrid_.inverseCellSize(), 
        Wall_.spatialGrid_.size3D(), 
        Wall_.spatialGrid_.num_device(), 
        Wall_.num_device(), 
        Wall_.gridDim(), 
        Wall_.blockDim(), 
        stream_);
    }

    void updateGhostSpatialGrid()
    {
        PeriodicBoundaryXY2D_.updateGhostSpatialGrid(LSParticle_, stream_);
        PeriodicBoundarySector_.updateGhostSpatialGrid(LSParticle_, stream_);
    }

    void buildLSParticleInteraction()
    {
        launchBuildLevelSetBoundaryNodeInteractions1st(LSParticleInteraction_.masterNeighborCount(), 
        LSParticle_.boundaryNodeInfo_.localPositionID(), 
        LSParticle_.boundaryNodeInfo_.particleID(), 
        LSParticle_.geometryInfo_.radius(),
        LSParticle_.geometryInfo_.gridNodeOrigin(), 
        LSParticle_.geometryInfo_.gridNodeInverseSpacing(), 
        LSParticle_.geometryInfo_.gridNodeSize(), 
        LSParticle_.geometryInfo_.SDFPrefixSum(), 
        LSParticle_.geometryInfo_.signedDistanceField(),
        LSParticle_.geometryInfo_.boundaryNodePosition(),
        LSParticle_.position(),
        LSParticle_.orientation(), 
        LSParticle_.geometryID(),
        LSParticle_.hashIndex(),
        LSParticle_.spatialGrid_.hashStart(), 
        LSParticle_.spatialGrid_.hashEnd(), 
        LSParticle_.spatialGrid_.minimumBoundary(), 
        LSParticle_.spatialGrid_.inverseCellSize(), 
        LSParticle_.spatialGrid_.size3D(), 
        LSParticleInteraction_.numMaster_device(), 
        LSParticleInteraction_.masterGridDim(), 
        LSParticleInteraction_.masterBlockDim(), 
        stream_);

        LSParticleInteraction_.updateNeighborPrefixSum(stream_);
        LSParticleInteraction_.updateNumPair(maxGPUThread_, stream_);
        LSParticleInteraction_.savePreviousStep(stream_);

        launchBuildLevelSetBoundaryNodeInteractions2nd(LSParticleInteraction_.slidingSpring(), 
        LSParticleInteraction_.contactPoint(), 
        LSParticleInteraction_.contactNormal(), 
        LSParticleInteraction_.contactOverlap(), 
        LSParticleInteraction_.masterID(), 
        LSParticleInteraction_.slaveID(), 
        LSParticleInteraction_.previousSlidingSpring(), 
        LSParticleInteraction_.previousSlaveID(), 
        LSParticleInteraction_.masterNeighborPrefixSum(), 
        LSParticleInteraction_.previousMasterNeighborPrefixSum(), 
        LSParticle_.boundaryNodeInfo_.localPositionID(), 
        LSParticle_.boundaryNodeInfo_.particleID(),
        LSParticle_.geometryInfo_.radius(),
        LSParticle_.geometryInfo_.gridNodeOrigin(), 
        LSParticle_.geometryInfo_.gridNodeInverseSpacing(), 
        LSParticle_.geometryInfo_.gridNodeSize(), 
        LSParticle_.geometryInfo_.SDFPrefixSum(), 
        LSParticle_.geometryInfo_.signedDistanceField(),
        LSParticle_.geometryInfo_.boundaryNodePosition(),
        LSParticle_.position(),
        LSParticle_.orientation(), 
        LSParticle_.geometryID(),
        LSParticle_.hashIndex(),
        LSParticle_.spatialGrid_.hashStart(), 
        LSParticle_.spatialGrid_.hashEnd(), 
        LSParticle_.spatialGrid_.minimumBoundary(), 
        LSParticle_.spatialGrid_.inverseCellSize(), 
        LSParticle_.spatialGrid_.size3D(), 
        LSParticleInteraction_.numMaster_device(), 
        LSParticleInteraction_.masterGridDim(), 
        LSParticleInteraction_.masterBlockDim(), 
        stream_);

        launchBuildLevelSetBoundaryNodeFixedParticleInteractions1st(LSParticleWallInteraction_.masterNeighborCount(), 
        LSParticle_.boundaryNodeInfo_.localPositionID(), 
        LSParticle_.boundaryNodeInfo_.particleID(), 
        LSParticle_.geometryInfo_.boundaryNodePosition(), 
        LSParticle_.position(), 
        LSParticle_.orientation(), 
        Wall_.geometryInfo_.gridNodeOrigin(), 
        Wall_.geometryInfo_.gridNodeInverseSpacing(), 
        Wall_.geometryInfo_.gridNodeSize(), 
        Wall_.geometryInfo_.SDFPrefixSum(), 
        Wall_.geometryInfo_.signedDistanceField(), 
        Wall_.position(), 
        Wall_.orientation(), 
        Wall_.geometryID(),
        Wall_.hashIndex(),
        Wall_.spatialGrid_.hashStart(), 
        Wall_.spatialGrid_.hashEnd(), 
        Wall_.spatialGrid_.minimumBoundary(), 
        Wall_.spatialGrid_.inverseCellSize(), 
        Wall_.spatialGrid_.size3D(), 
        LSParticleWallInteraction_.numMaster_device(), 
        LSParticleWallInteraction_.masterGridDim(), 
        LSParticleWallInteraction_.masterBlockDim(), 
        stream_);

        LSParticleWallInteraction_.updateNeighborPrefixSum(stream_);
        LSParticleWallInteraction_.updateNumPair(maxGPUThread_, stream_);
        LSParticleWallInteraction_.savePreviousStep(stream_);

        launchBuildLevelSetBoundaryNodeFixedParticleInteractions2nd(LSParticleWallInteraction_.slidingSpring(), 
        LSParticleWallInteraction_.contactPoint(), 
        LSParticleWallInteraction_.contactNormal(), 
        LSParticleWallInteraction_.contactOverlap(), 
        LSParticleWallInteraction_.masterID(), 
        LSParticleWallInteraction_.slaveID(), 
        LSParticleWallInteraction_.previousSlidingSpring(), 
        LSParticleWallInteraction_.previousSlaveID(), 
        LSParticleWallInteraction_.masterNeighborPrefixSum(), 
        LSParticleWallInteraction_.previousMasterNeighborPrefixSum(), 
        LSParticle_.boundaryNodeInfo_.localPositionID(), 
        LSParticle_.boundaryNodeInfo_.particleID(), 
        LSParticle_.geometryInfo_.boundaryNodePosition(), 
        LSParticle_.position(), 
        LSParticle_.orientation(), 
        Wall_.geometryInfo_.gridNodeOrigin(), 
        Wall_.geometryInfo_.gridNodeInverseSpacing(), 
        Wall_.geometryInfo_.gridNodeSize(), 
        Wall_.geometryInfo_.SDFPrefixSum(), 
        Wall_.geometryInfo_.signedDistanceField(), 
        Wall_.position(), 
        Wall_.orientation(), 
        Wall_.geometryID(), 
        Wall_.hashIndex(), 
        Wall_.spatialGrid_.hashStart(), 
        Wall_.spatialGrid_.hashEnd(), 
        Wall_.spatialGrid_.minimumBoundary(), 
        Wall_.spatialGrid_.inverseCellSize(), 
        Wall_.spatialGrid_.size3D(), 
        LSParticleWallInteraction_.numMaster_device(), 
        LSParticleWallInteraction_.masterGridDim(), 
        LSParticleWallInteraction_.masterBlockDim(), 
        stream_);
    }

    void buildGhostInteraction()
    {
        PeriodicBoundaryXY2D_.buildGhostInteraction(LSParticle_, maxGPUThread_, stream_);
        PeriodicBoundarySector_.buildGhostInteraction(LSParticle_, maxGPUThread_, stream_);
    }

    void clearLSParticleForceTorque()
    {
        cudaMemsetAsync(LSParticle_.force(), 0, LSParticle_.num_device() * sizeof(double3), stream_);
        cudaMemsetAsync(LSParticle_.torque(), 0, LSParticle_.num_device() * sizeof(double3), stream_);
    }

    void addLSParticleContactForceTorque(const double timeStep)
    {
        launchAddLevelSetParticleContactForceTorque(LSParticleInteraction_.slidingSpring(), 
        LSParticleInteraction_.normalElasticEnergy(),
        LSParticleInteraction_.slidingElasticEnergy(),
        LSParticleInteraction_.contactPoint(),
        LSParticleInteraction_.contactNormal(), 
        LSParticleInteraction_.contactOverlap(), 
        LSParticleInteraction_.masterID(), 
        LSParticleInteraction_.slaveID(), 
        LSParticle_.boundaryNodeInfo_.particleID(), 
        LSParticle_.force(), 
        LSParticle_.torque(), 
        LSParticle_.position(),
        LSParticle_.velocity(),
        LSParticle_.angularVelocity(),
        LSParticle_.geometryID(),
        LSParticle_.materialID(),
        LSParticle_.geometryInfo_.inverseMass(),
        LSParticle_.materialInfo_.normalStiffness(),
        LSParticle_.materialInfo_.shearStiffness(),
        LSParticle_.materialInfo_.frictionCoefficient(),
        LSParticle_.materialInfo_.restitutionCoefficient(),
        timeStep, 
        LSParticleInteraction_.numPair_device(), 
        LSParticleInteraction_.pairGridDim(), 
        LSParticleInteraction_.pairBlockDim(), 
        stream_);

        launchAddFixedLevelSetParticleContactForceTorque(LSParticleWallInteraction_.slidingSpring(), 
        LSParticleWallInteraction_.normalElasticEnergy(),
        LSParticleWallInteraction_.slidingElasticEnergy(),
        LSParticleWallInteraction_.contactPoint(),
        LSParticleWallInteraction_.contactNormal(), 
        LSParticleWallInteraction_.contactOverlap(),
        LSParticleWallInteraction_.masterID(),
        LSParticleWallInteraction_.slaveID(), 
        LSParticle_.boundaryNodeInfo_.particleID(), 
        LSParticle_.force(), 
        LSParticle_.torque(), 
        LSParticle_.position(),
        LSParticle_.velocity(),
        LSParticle_.angularVelocity(),
        LSParticle_.geometryID(),
        LSParticle_.materialID(),
        LSParticle_.geometryInfo_.inverseMass(),
        LSParticle_.materialInfo_.normalStiffness(),
        LSParticle_.materialInfo_.shearStiffness(),
        LSParticle_.materialInfo_.frictionCoefficient(),
        LSParticle_.materialInfo_.restitutionCoefficient(),
        Wall_.position(),
        Wall_.velocity(),
        Wall_.angularVelocity(),
        Wall_.materialID(),
        Wall_.materialInfo_.frictionCoefficient(),
        Wall_.materialInfo_.restitutionCoefficient(),
        timeStep, 
        LSParticleWallInteraction_.numPair_device(), 
        LSParticleWallInteraction_.pairGridDim(), 
        LSParticleWallInteraction_.pairBlockDim(), 
        stream_);
    }

    void addBondedForceTorque()
    {
        launchAddBondedForceTorque(VBondedInteraction_.centerPoint(), 
        VBondedInteraction_.maxNormalStress(), 
        VBondedInteraction_.maxShearStress(), 
        VBondedInteraction_.Un(), 
        VBondedInteraction_.Us(), 
        VBondedInteraction_.Ub(), 
        VBondedInteraction_.Ut(), 
        VBondedInteraction_.activated(), 
        VBondedInteraction_.B1(), 
        VBondedInteraction_.B2(), 
        VBondedInteraction_.B3(), 
        VBondedInteraction_.B4(), 
        VBondedInteraction_.radius(), 
        VBondedInteraction_.initialLength(), 
        VBondedInteraction_.tensileStrength(), 
        VBondedInteraction_.cohesion(), 
        VBondedInteraction_.frictionCoefficient(), 
        VBondedInteraction_.masterVBondPointLocalVectorN1(), 
        VBondedInteraction_.masterVBondPointLocalVectorN2(), 
        VBondedInteraction_.masterVBondPointLocalVectorN3(), 
        VBondedInteraction_.masterVBondPointLocalPosition(), 
        VBondedInteraction_.slaveVBondPointLocalVectorN1(), 
        VBondedInteraction_.slaveVBondPointLocalVectorN2(), 
        VBondedInteraction_.slaveVBondPointLocalVectorN3(), 
        VBondedInteraction_.slaveVBondPointLocalPosition(), 
        VBondedInteraction_.masterObjectID(), 
        VBondedInteraction_.slaveObjectID(), 
        LSParticle_.force(), 
        LSParticle_.torque(), 
        LSParticle_.position(), 
        LSParticle_.orientation(), 
        VBondedInteraction_.numPair_device(), 
        VBondedInteraction_.gridDim(), 
        VBondedInteraction_.blockDim(), 
        stream_);
    }

    void addGhostForceTorque(const double timeStep)
    {
        PeriodicBoundaryXY2D_.addGhostForceTorque(LSParticle_, timeStep, stream_); 
        PeriodicBoundarySector_.addGhostForceTorque(LSParticle_, timeStep, stream_); 
    }

    void integration1stHalf(const double3 gravity, const double timeStep)
    {
        launchParticleVelocityAngularVelocityIntegration(LSParticle_.velocity(),
        LSParticle_.angularVelocity(),
        LSParticle_.force(), 
        LSParticle_.torque(), 
        LSParticle_.orientation(), 
        LSParticle_.geometryID(),
        LSParticle_.geometryInfo_.inverseMass(),
        LSParticle_.geometryInfo_.inverseInertiaTensor(), 
        gravity, 
        0.5 * timeStep, 
        LSParticle_.num_device(), 
        LSParticle_.gridDim(), 
        LSParticle_.blockDim(),
        stream_);

        launchParticlePositionOrientationIntegration(LSParticle_.position(), 
        LSParticle_.orientation(), 
        LSParticle_.velocity(),
        LSParticle_.angularVelocity(), 
        timeStep, 
        LSParticle_.num_device(), 
        LSParticle_.gridDim(), 
        LSParticle_.blockDim(),
        stream_);

        launchParticlePositionOrientationIntegration(Wall_.position(), 
        Wall_.orientation(), 
        Wall_.velocity(),
        Wall_.angularVelocity(), 
        timeStep, 
        Wall_.num_device(), 
        Wall_.gridDim(), 
        Wall_.blockDim(),
        stream_);
    }

    void integration2ndHalf(const double3 gravity, const double timeStep)
    {
        launchParticleVelocityAngularVelocityIntegration(LSParticle_.velocity(),
        LSParticle_.angularVelocity(),
        LSParticle_.force(), 
        LSParticle_.torque(), 
        LSParticle_.orientation(), 
        LSParticle_.geometryID(),
        LSParticle_.geometryInfo_.inverseMass(),
        LSParticle_.geometryInfo_.inverseInertiaTensor(), 
        gravity, 
        0.5 * timeStep, 
        LSParticle_.num_device(), 
        LSParticle_.gridDim(), 
        LSParticle_.blockDim(),
        stream_);
    }

    void compute(const double3 gravity, const double timeStep, const size_t numStep, const size_t frameInterval, 
    size_t iStep = 0, size_t iFrame = 0, double time = 0.)
    {
        updateLSParticleSpatialGrid();
        buildLSParticleInteraction();
        output(iFrame, iStep, time);
        while (iStep <= numStep)
        {
            integration1stHalf(gravity, timeStep);

            updateLSParticleSpatialGrid();
            buildLSParticleInteraction();

            clearLSParticleForceTorque();
            addLSParticleContactForceTorque(timeStep);
            addExternalForceTorque(time);

            integration2ndHalf(gravity, timeStep);

            iStep += 1;
            time += timeStep;
            if (iStep % frameInterval == 0)
            {
                iFrame++;
                output(iFrame, iStep, time);
            }
        }
        iFrame++;
        output(iFrame, iStep, time);
        std::cout << "[Solver] Computation Completed" << std::endl;
    }

    void compute_addBond(const double3 gravity, const double timeStep, const size_t numStep, const size_t frameInterval, 
    size_t iStep = 0, size_t iFrame = 0, double time = 0.)
    {
        updateLSParticleSpatialGrid();
        buildLSParticleInteraction();
        output(iFrame, iStep, time);
        while (iStep <= numStep)
        {
            integration1stHalf(gravity, timeStep);

            updateLSParticleSpatialGrid();
            buildLSParticleInteraction();

            clearLSParticleForceTorque();
            addLSParticleContactForceTorque(timeStep);
            addBondedForceTorque();
            addExternalForceTorque(time);

            integration2ndHalf(gravity, timeStep);

            iStep += 1;
            time += timeStep;
            if (iStep % frameInterval == 0)
            {
                iFrame++;
                output(iFrame, iStep, time);
            }
        }
        iFrame++;
        output(iFrame, iStep, time);
        std::cout << "[Solver] Computation Completed" << std::endl;
    }

    void compute_addPeriodicBoundary(const double3 gravity, const double timeStep, const size_t numStep, const size_t frameInterval, 
    size_t iStep = 0, size_t iFrame = 0, double time = 0.)
    {
        updateLSParticleSpatialGrid();
        buildLSParticleInteraction();
        updateGhostSpatialGrid();
        buildGhostInteraction();
        output(iFrame, iStep, time);
        while (iStep <= numStep)
        {
            integration1stHalf(gravity, timeStep);

            updateLSParticleSpatialGrid();
            buildLSParticleInteraction();
            updateGhostSpatialGrid();
            buildGhostInteraction();

            clearLSParticleForceTorque();
            addLSParticleContactForceTorque(timeStep);
            addGhostForceTorque(timeStep);
            addExternalForceTorque(time);

            integration2ndHalf(gravity, timeStep);

            iStep += 1;
            time += timeStep;
            if (iStep % frameInterval == 0)
            {
                iFrame++;
                output(iFrame, iStep, time);
            }
        }
        iFrame++;
        output(iFrame, iStep, time);
        std::cout << "[Solver] Computation Completed" << std::endl;
    }

    void compute_addBondAndPeriodicBoundary(const double3 gravity, const double timeStep, const size_t numStep, const size_t frameInterval, 
    size_t iStep = 0, size_t iFrame = 0, double time = 0.)
    {
        updateLSParticleSpatialGrid();
        buildLSParticleInteraction();
        updateGhostSpatialGrid();
        buildGhostInteraction();
        output(iFrame, iStep, time);
        while (iStep <= numStep)
        {
            integration1stHalf(gravity, timeStep);

            updateLSParticleSpatialGrid();
            buildLSParticleInteraction();
            updateGhostSpatialGrid();
            buildGhostInteraction();
            
            clearLSParticleForceTorque();
            addLSParticleContactForceTorque(timeStep);
            addGhostForceTorque(timeStep);
            addBondedForceTorque();
            addExternalForceTorque(time);

            integration2ndHalf(gravity, timeStep);

            iStep += 1;
            time += timeStep;
            if (iStep % frameInterval == 0)
            {
                iFrame++;
                output(iFrame, iStep, time);
            }
        }
        iFrame++;
        output(iFrame, iStep, time);
        std::cout << "[Solver] Computation Completed" << std::endl;
    }

    LSParticle LSParticle_;
    LSParticle Wall_;
    SolidInteraction LSParticleInteraction_;
    SolidInteraction LSParticleWallInteraction_;
    VBondedInteraction VBondedInteraction_;
    PeriodicBoundaryXY2D PeriodicBoundaryXY2D_;
    PeriodicBoundarySector PeriodicBoundarySector_;

    std::string dir_;
    size_t phase_;
    cudaStream_t stream_;
    size_t maxGPUThread_;
    int device_;
};