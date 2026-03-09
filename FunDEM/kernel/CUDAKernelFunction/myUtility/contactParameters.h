#pragma once
#include "myHostDeviceArray.h"
#include <algorithm>

// ============================================================================
// Pair parameter index (host/device)
//   - symmetric (i,j)==(j,i)
//   - last slot (cap-1) is fallback for invalid indices
// ============================================================================
__host__ __device__ inline int contactParameterIndex(int a, int b, int nMaterials, int cap)
{
    if (nMaterials <= 0 || cap <= 0) return 0;

    if (a < 0 || b < 0 || a >= nMaterials || b >= nMaterials)
    {
        return cap - 1;
    }

    int i = a, j = b;
    if (i > j) { int t = i; i = j; j = t; }

    long long idx = (static_cast<long long>(i) * (2LL * nMaterials - i + 1LL)) / 2LL
    + static_cast<long long>(j - i);

    if (idx < 0) idx = 0;
    if (idx >= cap) idx = cap - 1;
    return static_cast<int>(idx);
}

struct materialRow
{
    double YoungsModulus; 
    double poissonRatio;
    double restitutionCoefficient; // e
};

struct frictionRow
{
    int materialIndexA;
    int materialIndexB;

    double slidingFrictionCoefficient; // mu_s
    double rollingFrictionCoefficient; // mu_r
    double torsionFrictionCoefficient; // mu_t
};

struct linearStiffnessRow
{
    int materialIndexA;
    int materialIndexB;

    double normalStiffness; // k_n
    double slidingStiffness; // k_s
    double rollingStiffness; // k_r
    double torsionStiffness; // k_t
};

struct bondRow
{
    int materialIndexA;
    int materialIndexB;

    double bondRadiusMultiplier; // gamma
    double bondYoungsModulus; // E_bond
    double normalToShearStiffnessRatio; // k_n / k_s
    double tensileStrength; // sigma_s
    double cohesion; // C
    double frictionCoefficient; // mu
};

enum frictionParam : int
{
    f_MUS = 0,
    f_MUR,
    f_MUT,
    f_COUNT
};

enum linearStiffnessParam : int
{
    l_KN = 0,
    l_KS,
    l_KR,
    l_KT,
    l_COUNT
};

enum bondParam : int
{
    b_GAMMA = 0,
    b_E,
    b_KNKS,
    b_SIGMAS,
    b_C,
    b_MU,
    b_COUNT
};

struct paramsDevice
{
    int nMaterials {0};
    int cap {0}; // pairTableSize

    const double* YoungsModulus {nullptr};
    const double* poissonRatio {nullptr};
    const double* restitutionCoefficient {nullptr};
    
    const double* friction {nullptr};
    const double* linearStiffness {nullptr};
    const double* bond {nullptr};
};

// ============================================================================
// Device access helpers
// ============================================================================
#if defined(__CUDACC__)
extern __constant__ paramsDevice para;

__device__ inline double getYoungsModulusParam(const int idx)
{
    return para.YoungsModulus[idx];
}

__device__ inline double getPoissonRatioParam(const int idx)
{
    return para.poissonRatio[idx];
}

__device__ inline double getRestitutionCoefficientParam(const int idx)
{
    return para.restitutionCoefficient[idx];
}

__device__ inline double getFrictionParam(const int pairIdx, const int p)
{
    return para.friction[p * para.cap + pairIdx];
}

__device__ inline double getLinearStiffnessParam(const int pairIdx, const int p)
{
    return para.linearStiffness[p * para.cap + pairIdx];
}

__device__ inline double getBondParam(const int pairIdx, const int p)
{
    return para.bond[p * para.cap + pairIdx];
}
#endif

class contactParameters
{
public:
    std::size_t numberOfMaterials_ {0};
    std::size_t pairTableSize_ {0};

private:
    // ---------------------------------------------------------------------
    // Packed arrays (param-major)
    // ---------------------------------------------------------------------
    HostDeviceArray1D<double> YoungsModulus_;
    HostDeviceArray1D<double> poissonRatio_;
    HostDeviceArray1D<double> restitutionCoefficient_;

    HostDeviceArray1D<double> friction_;
    HostDeviceArray1D<double> linearStiffness_;
    HostDeviceArray1D<double> bond_;

private:
    // ---------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------
    static int inferNumberOfMaterials_(const std::vector<materialRow>& materialTable,
    const std::vector<frictionRow>& frictionTable,
    const std::vector<linearStiffnessRow>& linearStiffnessTable,
    const std::vector<bondRow>& bondTable)
    {
        int maxIndex = materialTable.size() - 1;

        auto updateMax = [&](int a, int b)
        {
            maxIndex = std::max(maxIndex, a);
            maxIndex = std::max(maxIndex, b);
        };

        for (const auto& row : frictionTable) updateMax(row.materialIndexA, row.materialIndexB);
        for (const auto& row : linearStiffnessTable) updateMax(row.materialIndexA, row.materialIndexB);
        for (const auto& row : bondTable) updateMax(row.materialIndexA, row.materialIndexB);

        return (maxIndex < 0) ? 0 : (maxIndex + 1);
    }

    static std::size_t computePairTableSize_(int nMaterials)
    {
        if (nMaterials <= 0) return 0;
        return (static_cast<std::size_t>(nMaterials) * (static_cast<std::size_t>(nMaterials) + 1)) / 2 + 1;
    }

    static void setPacked_(std::vector<double>& buf,
    const std::size_t cap,
    const int param,
    const std::size_t idx,
    const double value)
    {
        buf[static_cast<std::size_t>(param) * cap + idx] = value;
    }

public:
    // ---------------------------------------------------------------------
    // Rule of Five
    // ---------------------------------------------------------------------
    contactParameters() = default;
    ~contactParameters() = default;

    contactParameters(const contactParameters&) = delete;
    contactParameters& operator=(const contactParameters&) = delete;

    contactParameters(contactParameters&&) noexcept = default;
    contactParameters& operator=(contactParameters&&) noexcept = default;

public:
    // ---------------------------------------------------------------------
    // Build + upload + commit constant memory
    // ---------------------------------------------------------------------
    void buildFromTables(const std::vector<materialRow>& materialTable,
    const std::vector<frictionRow>& frictionTable,
    const std::vector<linearStiffnessRow>& linearStiffnessTable,
    const std::vector<bondRow>& bondTable,
    cudaStream_t stream);
};