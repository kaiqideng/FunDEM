#include "DEMSolver.h"
#include "WCSPHSolver.h"

class DEMSPHSolver:
protected DEMSolver, WCSPHSolver
{
public:
    DEMSPHSolver(cudaStream_t s1, cudaStream_t s2): DEMSolver(s1), WCSPHSolver(s2)
    {
    }

    ~DEMSPHSolver() = default;

protected:
    void addSPHForceToSolid(const double3 gravity)
    {
        if (getBallNumber() != getDummy().deviceSize()) return;

        launchAddSPHForce(getBallForceDevicePtr(), 
        getBallPositionDevicePtr(), 
        getBallVelocityDevicePtr(), 
        getBallInverseMassDevicePtr(),
        getDummy().normal(), 
        getDummy().soundSpeed(), 
        getDummy().mass(), 
        getDummy().initialDensity(), 
        getDummy().smoothLength(), 
        getDummy().viscosity(), 
        getSPHAndDummyInteraction().objectPointing_.interactionStart(), 
        getSPHAndDummyInteraction().objectPointing_.interactionEnd(), 
        getSPH().position(), 
        getSPH().velocity(), 
        getSPH().pressure(), 
        getSPH().soundSpeed(), 
        getSPH().mass(), 
        getSPH().density(), 
        getSPH().smoothLength(), 
        getSPH().viscosity(), 
        getSPHAndDummyInteraction().pair_.objectPointed(), 
        getSPHAndDummyInteraction().pair_.hashIndex(), 
        gravity, 
        getDummy().deviceSize(), 
        getDummy().gridDim(), 
        getDummy().blockDim(), 
        getDEMCUDAStream());
    }

    void updateDummyNormal(const double timeStep)
    {
        if (getBallNumber() != getDummy().deviceSize()) return;
        
        launchUpdateDummyNormal(getDummy().normal(), 
        getBallAngularVelocityDevicePtr(), 
        timeStep, 
        getDummy().deviceSize(), 
        getDummy().gridDim(), 
        getDummy().blockDim(), 
        getDEMCUDAStream());
    }

public:
    void setSolidMaterial(const int index,
    const double YoungsModulus,
    const double poissonRatio,
    const double restitutionCoefficient)
    {
        setMaterial(index,
        YoungsModulus,
        poissonRatio,
        restitutionCoefficient);
    }

    void setSolidFriction(const int materialIndexA,
    const int materialIndexB,
    const double slidingFrictionCoefficient,
    const double rollingFrictionCoefficient,
    const double torsionFrictionCoefficient)
    {
        setFriction(materialIndexA,
        materialIndexB,
        slidingFrictionCoefficient,
        rollingFrictionCoefficient,
        torsionFrictionCoefficient);
    }

    void setSolidLinearStiffness(const int materialIndexA,
    const int materialIndexB,
    const double normalStiffness,
    const double slidingStiffness,
    const double rollingStiffness,
    const double torsionStiffness)
    {
        setLinearStiffness(materialIndexA,
        materialIndexB,
        normalStiffness,
        slidingStiffness,
        rollingStiffness,
        torsionStiffness);
    }

    void setSolidBond(const int materialIndexA,
    const int materialIndexB,
    const double bondRadiusMultiplier,
    const double bondYoungsModulus,
    const double normalToShearStiffnessRatio,
    const double tensileStrength,
    const double cohesion,
    const double frictionCoefficient)
    {
        setBond(materialIndexA,
        materialIndexB,
        bondRadiusMultiplier,
        bondYoungsModulus,
        normalToShearStiffnessRatio,
        tensileStrength,
        cohesion,
        frictionCoefficient);
    }

    void addFluidSPH(double3 position, double3 velocity, double soundSpeed, double spacing, double density, double viscosity)
    {
        addSPH(position, velocity, soundSpeed, spacing, density, viscosity);
    }

    void addSolid(std::vector<double3> globalPoint, 
    std::vector<double> radius, 
    double3 centroidPosition, 
    double3 velocity, 
    double3 angularVelocity, 
    double mass, 
    symMatrix inertiaTensor, 
    int materialID,
    double soundSpeed_SPHDummy, 
    double density_SPHDummy, 
    double viscosity_SPHDummy)
    {
        int clumpID = static_cast<int>(getClump().hostSize());

        addClump(globalPoint, radius, centroidPosition, velocity, angularVelocity, mass, inertiaTensor, materialID);

        for (size_t i = 0; i < globalPoint.size(); i++)
        {
            addDummy(globalPoint[i], 
            velocity + cross(angularVelocity, globalPoint[i] - centroidPosition), 
            soundSpeed_SPHDummy, 
            2. * radius[i], 
            density_SPHDummy, 
            viscosity_SPHDummy,
            clumpID);
        }
    }

    void addFixedSolid(std::vector<double3> globalPoint, 
    std::vector<double> radius, 
    double3 centroidPosition, 
    int materialID,
    double soundSpeed_SPHDummy, 
    double density_SPHDummy, 
    double viscosity_SPHDummy)
    {
        int clumpID = static_cast<int>(getClump().hostSize());

        addFixedClump(globalPoint, radius, centroidPosition, materialID);

        for (size_t i = 0; i < globalPoint.size(); i++)
        {
            addDummy(globalPoint[i], 
            make_double3(0., 0., 0.), 
            soundSpeed_SPHDummy, 
            2. * radius[i], 
            density_SPHDummy, 
            viscosity_SPHDummy,
            clumpID);
        }
    }

    void eraseSolid(const size_t index)
    {
        if (index >= getClump().hostSize()) return;
        const size_t pebbleStart = getClump().pebbleStartHostRef()[index];
        const size_t pebbleEnd = getClump().pebbleEndHostRef()[index];
        for(size_t i = 0; i < pebbleEnd - pebbleStart; i++)
        {
            eraseBall(pebbleStart); //each erasing will move all elements(after the index number) forward 
            eraseDummy(pebbleStart); //each erasing will move all elements(after the index number) forward 
        }
    }

    void solve(const double3 minBoundary, 
    const double3 maxBoundary, 
    const double3 gravity, 
    const double timeStep, 
    const size_t SPHIntegrationInterval,
    const double maximumTime,
    const size_t numFrame,
    const std::string dir, 
    const size_t deviceID = 0)
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

        DEMInitialize(minBoundary, maxBoundary); //DEM stream
        if (getBallNumber() == 0) return;
        ballNeighborSearch(); //DEM stream
        calculateBallContactForceTorque(0.); //DEM stream
        if (addInitialCondition()) DEMInitialize(minBoundary, maxBoundary); //DEM stream
        SPHInitialize(minBoundary, maxBoundary); //SPH stream
        if (getSPH().deviceSize() == 0 || getBallNumber() != getDummy().deviceSize()) return;
        SPHNeighborSearch(); //SPH stream
        cudaDeviceSynchronize();
        std::cout << "Initialization Completed." << std::endl;

        size_t iStep = 0, iFrame = 0;
        double time = 0.0;
        size_t interval = 1;
        if (SPHIntegrationInterval > 0) interval = SPHIntegrationInterval;
        const double halfTimeStep = 0.5 * timeStep;
        const double halfTimeStep_SPH = double(interval) * halfTimeStep;
        outputBallVTU(dir, 0, 0, 0.0);
        outputBallInteractionVTU(dir, 0, 0, 0.0);
        outputBondedInteractionVTU(dir, 0, 0, 0.0);
        outputWCSPHVTU(dir, 0, 0, 0.0);
        while (iStep <= numStep)
        {
            iStep++;
            time += timeStep;
            if (iStep % 5 == 0) ballNeighborSearch(); //DEM stream
            calculateBallContactForceTorque(halfTimeStep); //DEM stream
            addSPHForceToSolid(gravity); //DEM stream
            ballClumpIntegration1stHalf(gravity, halfTimeStep); //DEM stream
            updateDummyNormal(timeStep); //DEM stream, DEM and SPH parameters
            if (iStep % interval == 0)
            {
                cudaDeviceSynchronize();
                cudaMemcpy(getDummy().position(), getBallPositionDevicePtr(), getBallNumber() * sizeof(double3), cudaMemcpyDeviceToDevice);
                cudaMemcpy(getDummy().velocity(), getBallVelocityDevicePtr(), getBallNumber() * sizeof(double3), cudaMemcpyDeviceToDevice);
                //initializeBoundaryCondition(); //SPH stream
                SPHNeighborSearch(); //SPH stream
                SPHInteraction1stHalf(gravity, halfTimeStep_SPH); //SPH stream
                SPHInteraction2ndHalf(gravity, halfTimeStep_SPH); //SPH stream
            }
            calculateBallContactForceTorque(halfTimeStep); //DEM stream
            cudaDeviceSynchronize();
            addSPHForceToSolid(gravity); //DEM stream, DEM and SPH parameters
            ballClumpIntegration2ndHalf(gravity, halfTimeStep); //DEM stream

            if (iStep % frameInterval == 0) 
            {
                iFrame++;
                std::cout << "Frame " << iFrame << " at Time " << time << std::endl;
                outputBallVTU(dir, iFrame, iStep, time);
                outputBallInteractionVTU(dir, iFrame, iStep, time);
                outputBondedInteractionVTU(dir, iFrame, iStep, time);
                outputWCSPHVTU(dir, iFrame, iStep, time);
            }
        }
    }
};