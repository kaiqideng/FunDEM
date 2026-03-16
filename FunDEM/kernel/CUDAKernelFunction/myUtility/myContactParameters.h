/**
 * @file myContactParameters.h
 * @brief Host-side builder and device-side accessors for contact model parameters.
 *
 * This header defines:
 * - Row structs describing user-friendly input tables (material/pairwise parameters).
 * - A packed device layout (`paramsDevice`) published via CUDA constant memory (`para`).
 * - Lightweight device accessors to fetch parameters inside CUDA kernels.
 * - A host-side owner class (`contactParameters`) that builds, packs, uploads, and commits tables.
 *
 * Design notes:
 * - Pairwise parameters are stored in an upper-triangular table (symmetric pairs).
 * - Packed arrays use param-major layout: `buf[param * cap + pairIdx]` for coalesced loads.
 * - `cap` is `pairTableSize_`, including an optional fallback slot for invalid indices.
 */
#pragma once
#include "myHostDeviceArray.h"
#include <algorithm>

/// @brief Per-material scalar properties (indexed by material ID).
struct materialRow
{
    double YoungsModulus; ///< Young's modulus E (Pa).
    double poissonRatio; ///< Poisson ratio ν (dimensionless).
    double restitutionCoefficient;  ///< Restitution coefficient e in (0,1] (dimensionless).
};

/// @brief Per-material-pair friction properties (symmetric in A/B).
struct frictionRow
{
    int materialIndexA; ///< Material index A.
    int materialIndexB; ///< Material index B.

    double slidingFrictionCoefficient; ///< Sliding friction coefficient μ_s.
    double rollingFrictionCoefficient; ///< Rolling friction coefficient μ_r.
    double torsionFrictionCoefficient; ///< Torsion friction coefficient μ_t.
};

/// @brief Per-material-pair linear stiffness properties (symmetric in A/B).
struct linearStiffnessRow
{
    int materialIndexA; ///< Material index A.
    int materialIndexB; ///< Material index B.

    double normalStiffness; ///< Normal stiffness k_n.
    double slidingStiffness; ///< Sliding (tangential) stiffness k_s.
    double rollingStiffness; ///< Rolling stiffness k_r.
    double torsionStiffness; ///< Torsion stiffness k_t.
};

/// @brief Per-material-pair bonded-contact parameters (symmetric in A/B).
struct bondRow
{
    int materialIndexA; ///< Material index A.
    int materialIndexB; ///< Material index B.

    double bondRadiusMultiplier; ///< Bond radius multiplier γ.
    double bondYoungsModulus; ///< Bond Young's modulus E_bond.
    double normalToShearStiffnessRatio; ///< Stiffness ratio k_n / k_s.
    double tensileStrength; ///< Tensile strength σ_s.
    double cohesion; ///< Cohesion C.
    double frictionCoefficient; ///< Bond friction coefficient μ.
};

/// @brief Indices for packed friction parameter array.
enum frictionParam : int
{
    f_MUS = 0, ///< Sliding friction coefficient μ_s
    f_MUR,     ///< Rolling friction coefficient μ_r
    f_MUT,     ///< Torsion friction coefficient μ_t
    f_COUNT    ///< Number of friction parameters
};

/// @brief Indices for packed linear stiffness parameter array.
enum linearStiffnessParam : int
{
    l_KN = 0, ///< Normal stiffness k_n
    l_KS,     ///< Sliding stiffness k_s
    l_KR,     ///< Rolling stiffness k_r
    l_KT,     ///< Torsion stiffness k_t
    l_COUNT   ///< Number of stiffness parameters
};

/// @brief Indices for packed bond parameter array.
enum bondParam : int
{
    b_GAMMA = 0, ///< Bond radius multiplier γ
    b_E,         ///< Bond Young's modulus E_bond
    b_KNKS,      ///< Ratio k_n / k_s
    b_SIGMAS,    ///< Tensile strength σ_s
    b_C,         ///< Cohesion C
    b_MU,        ///< Bond friction coefficient μ
    b_COUNT      ///< Number of bond parameters
};

/// @brief Device-side parameter descriptor stored in CUDA constant memory.
/// Stores only meta + device pointers; underlying arrays live in global memory.
struct paramsDevice
{
    int nMaterials {0}; ///< Number of materials.
    int cap {0}; ///< Pair-table capacity (pairTableSize_).

    const double* effectiveYoungsModulus {nullptr}; ///< Per-pair E_ij array (size = cap).
    const double* effectiveShearModulus {nullptr}; ///< Per-pair G_ij array (size = cap).
    const double* dissipation {nullptr}; ///< Per-pair dissipation d_ij array (size = cap).

    const double* friction {nullptr}; ///< Packed friction array (size = f_COUNT * cap).
    const double* linearStiffness {nullptr}; ///< Packed stiffness array (size = l_COUNT * cap).
    const double* bond {nullptr}; ///< Packed bond array (size = b_COUNT * cap).
};

// ============================================================================
// Device access helpers
// ============================================================================
#if defined(__CUDACC__)
/**
 * @brief Constant-memory view of contact parameters used by CUDA kernels.
 *
 * This symbol is written from host code (see contactParameters::buildFromTables).
 * All device accessors below read through this constant-memory descriptor.
 */
extern __constant__ paramsDevice para;

/// @brief Fetch effective Young's modulus E_ij for a material pair index.
__device__ inline double getEffectiveYoungsModulusParam(const int pairIdx)
{
    return para.effectiveYoungsModulus[pairIdx];
}

/// @brief Fetch effective shear modulus G_ij for a material pair index.
__device__ inline double getEffectiveShearModulusParam(const int pairIdx)
{
    return para.effectiveShearModulus[pairIdx];
}

/// @brief Fetch dissipation factor d_ij for a material pair index.
__device__ inline double getDissipationParam(const int pairIdx)
{
    return para.dissipation[pairIdx];
}

/// @brief Fetch friction parameter `p` (see frictionParam) for a material pair index.
__device__ inline double getFrictionParam(const int pairIdx, const int p)
{
    return para.friction[p * para.cap + pairIdx];
}

/// @brief Fetch linear stiffness parameter `p` (see linearStiffnessParam) for a material pair index.
__device__ inline double getLinearStiffnessParam(const int pairIdx, const int p)
{
    return para.linearStiffness[p * para.cap + pairIdx];
}

/// @brief Fetch bond parameter `p` (see bondParam) for a material pair index.
__device__ inline double getBondParam(const int pairIdx, const int p)
{
    return para.bond[p * para.cap + pairIdx];
}
#endif

/**
 * @brief Host-side owner/builder of contact parameter tables.
 *
 * This class:
 * - infers the number of materials from provided tables,
 * - computes the pair-table capacity,
 * - packs all pairwise parameters into flat arrays,
 * - uploads them to device memory,
 * - and commits a paramsDevice descriptor into constant memory (`para`).
 */
class contactParameters
{
public:
    size_t numberOfMaterials_ {0}; ///< Number of materials inferred/built.
    size_t pairTableSize_ {0}; ///< Pair-table capacity (upper-triangular + fallback slot).

private:
    // ---------------------------------------------------------------------
    // Packed arrays (param-major)
    // ---------------------------------------------------------------------
    HostDeviceArray1D<double> effectiveYoungsModulus_; ///< E_ij per pairIdx (size = cap)
    HostDeviceArray1D<double> effectiveShearModulus_;  ///< G_ij per pairIdx (size = cap)
    HostDeviceArray1D<double> dissipation_;            ///< d_ij per pairIdx (size = cap)
    HostDeviceArray1D<double> friction_;               ///< friction params (size = f_COUNT * cap)
    HostDeviceArray1D<double> linearStiffness_;        ///< linear stiffness params (size = l_COUNT * cap)
    HostDeviceArray1D<double> bond_;                   ///< bond params (size = b_COUNT * cap)

private:
    // ---------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------

    /**
     * @brief Infer the number of materials from input tables.
     *
     * Finds the maximum material index referenced by any pairwise row (and also considers
     * the size of materialTable). Returns maxIndex+1.
     *
     * @param[in] materialTable         Per-material property table.
     * @param[in] frictionTable         Per-pair friction table.
     * @param[in] linearStiffnessTable  Per-pair stiffness table.
     * @param[in] bondTable             Per-pair bond table.
     * @return Number of materials inferred (>=0).
     */
    static int inferNumberOfMaterials_(const std::vector<materialRow>& materialTable,
    const std::vector<frictionRow>& frictionTable,
    const std::vector<linearStiffnessRow>& linearStiffnessTable,
    const std::vector<bondRow>& bondTable)
    {
        int maxIndex = static_cast<int>(materialTable.size()) - 1;

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

    /**
     * @brief Compute upper-triangular pair table size with a fallback slot.
     *
     * @param[in] nMaterials Number of materials.
     * @return Pair table capacity (cap).
     */
    static std::size_t computePairTableSize_(int nMaterials)
    {
        if (nMaterials <= 0) return 0;
        return (static_cast<std::size_t>(nMaterials) * (static_cast<std::size_t>(nMaterials) + 1)) / 2 + 1;
    }

    /**
     * @brief Write one value into a param-major packed buffer.
     *
     * Layout: buf[param * cap + idx] = value.
     *
     * @param[in,out] buf   Destination buffer.
     * @param[in]     cap   Pair table capacity.
     * @param[in]     param Parameter slot (enum value).
     * @param[in]     idx   Pair index.
     * @param[in]     value Value to store.
     */
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

    /**
     * @brief Build and upload all contact parameter tables, then commit `paramsDevice` into constant memory.
     *
     * @param[in] materialTable         Per-material properties.
     * @param[in] frictionTable         Per-pair friction coefficients.
     * @param[in] linearStiffnessTable  Per-pair linear stiffness.
     * @param[in] bondTable             Per-pair bond parameters.
     * @param[in] stream                CUDA stream for async allocations/copies and cudaMemcpyToSymbolAsync.
     */
    void buildFromTables(const vector<materialRow>& materialTable,
    const std::vector<frictionRow>& frictionTable,
    const std::vector<linearStiffnessRow>& linearStiffnessTable,
    const std::vector<bondRow>& bondTable,
    cudaStream_t stream);
};