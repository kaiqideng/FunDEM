#include "kernel/CUDAKernelFunction/myUtility/myVec.h"
#include <cassert>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

inline std::vector<double3> generateUniformPointCloud(const double3 minBound,
                                                      const double3 maxBound,
                                                      const double dx)
{
    std::vector<double3> pts;
    if (dx <= 0.0) return pts;

    const double x0 = std::min(minBound.x, maxBound.x);
    const double y0 = std::min(minBound.y, maxBound.y);
    const double z0 = std::min(minBound.z, maxBound.z);

    const double x1 = std::max(minBound.x, maxBound.x);
    const double y1 = std::max(minBound.y, maxBound.y);
    const double z1 = std::max(minBound.z, maxBound.z);

    // epsilon scaled with dx (more robust than a fixed 1e-12)
    const double eps = 1e-12 * (std::fabs(dx) + 1.0);

    auto countAxis = [&](double a0, double a1) -> int
    {
        const double start = a0 + 0.5 * dx;
        const double end   = a1 - 0.5 * dx;
        if (end < start - eps) return 0;

        const double len = end - start;
        return static_cast<int>(std::floor(len / dx + 1.0 + eps));
    };

    const int nx = countAxis(x0, x1);
    const int ny = countAxis(y0, y1);
    const int nz = countAxis(z0, z1);

    pts.reserve(static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz));

    const double xs = x0 + 0.5 * dx;
    const double ys = y0 + 0.5 * dx;
    const double zs = z0 + 0.5 * dx;

    for (int k = 0; k < nz; ++k)
    {
        const double z = zs + k * dx;
        for (int j = 0; j < ny; ++j)
        {
            const double y = ys + j * dx;
            for (int i = 0; i < nx; ++i)
            {
                const double x = xs + i * dx;
                pts.push_back(make_double3(x, y, z));
            }
        }
    }

    return pts;
}

struct HexPBC2D
{
    double3 min {0.0, 0.0, 0.0};  // periodic cell min
    double3 max {0.0, 0.0, 0.0};  // periodic cell max (seam+L), half-open [min,max)
    double Lx {0.0};
    double Ly {0.0};

    // For THIS hex layout (odd rows shifted by +0.5*dx):
    // when wrapping y by ±Ly, also shift x by shearX to match lattice.
    double shearX {0.0};

    int nx {0};
    int ny {0};

    bool periodicX {false};
    bool periodicY {false};
};

inline double ceilDivLen(const double len, const double step)
{
    if (step <= 0.0) return 0.0;
    // small epsilon avoids ceil(1.0000000002) type surprises
    return std::ceil(len / step - 1e-12);
}

// -----------------------------------------------------------------------------
// Build periodic cell for hex packing inside [minBound, maxBound) (x-y plane).
// - spacing = a
// - dx = a, dy = a*sqrt(3)/2
// - if periodic axis enabled, we SNAP that axis length to integer multiples of step
//   and return snapped pbc.max.
// -----------------------------------------------------------------------------
inline HexPBC2D makeHexPBC2D(const double3 minBound,
                            const double3 maxBound,
                            const double spacing,
                            const bool periodicX,
                            const bool periodicY)
{
    HexPBC2D pbc;
    pbc.periodicX = periodicX;
    pbc.periodicY = periodicY;

    if (spacing <= 0.0) return pbc;

    const double dx = spacing;
    const double dy = spacing * std::sqrt(3.0) * 0.5;

    const double Lx_req = std::max(0.0, maxBound.x - minBound.x);
    const double Ly_req = std::max(0.0, maxBound.y - minBound.y);

    // choose nx/ny so that [min, min+nx*dx) and [min, min+ny*dy) cover the box
    const int nx = (periodicX)
        ? std::max(1, static_cast<int>(ceilDivLen(Lx_req, dx)))
        : std::max(1, static_cast<int>(std::floor(Lx_req / dx + 1e-12)));

    const int ny = (periodicY)
        ? std::max(1, static_cast<int>(ceilDivLen(Ly_req, dy)))
        : std::max(1, static_cast<int>(std::floor(Ly_req / dy + 1e-12)));

    pbc.nx = nx;
    pbc.ny = ny;

    pbc.min = minBound;

    // snapped periodic cell lengths (only meaningful if periodic axis enabled)
    pbc.Lx = nx * dx;
    pbc.Ly = ny * dy;

    // output max boundary:
    // - if periodic -> snapped
    // - else -> keep original maxBound (you can still generate within it)
    pbc.max = make_double3(
        periodicX ? (minBound.x + pbc.Lx) : maxBound.x,
        periodicY ? (minBound.y + pbc.Ly) : maxBound.y,
        maxBound.z
    );

    // y-wrap shear for THIS “odd row shifted in x” hex layout
    pbc.shearX = (periodicY && (ny & 1)) ? (0.5 * dx) : 0.0;

    return pbc;
}

// -----------------------------------------------------------------------------
// Generate hex-packed points in x-y, z fixed to minBound.z.
// - points are inside [pbc.min, pbc.max) for periodic axes.
// - if non-periodic axis, points are clamped to original [minBound,maxBound).
//
// IMPORTANT for tiling:
// - Copy in x: add (kx * pbc.Lx, 0, 0)
// - Copy in y: add (ky * pbc.shearX, ky * pbc.Ly, 0)
// -----------------------------------------------------------------------------
inline void generateHexPointCloud(const double3 minBound,
                                  const double3 maxBound,
                                  const double spacing,
                                  const bool periodicX,
                                  const bool periodicY,
                                  std::vector<double3>& points,
                                  HexPBC2D& pbc)
{
    points.clear();
    pbc = makeHexPBC2D(minBound, maxBound, spacing, periodicX, periodicY);
    if (pbc.nx <= 0 || pbc.ny <= 0 || spacing <= 0.0) return;

    const double dx = spacing;
    const double dy = spacing * std::sqrt(3.0) * 0.5;

    const double x0 = pbc.min.x;
    const double y0 = pbc.min.y;
    const double z0 = pbc.min.z;

    // limits for non-periodic axes use the original maxBound
    const double xMaxKeep = maxBound.x;
    const double yMaxKeep = maxBound.y;

    points.reserve(static_cast<size_t>(pbc.nx) * static_cast<size_t>(pbc.ny));

    for (int j = 0; j < pbc.ny; ++j)
    {
        const double y = y0 + j * dy;
        if (!periodicY && y >= yMaxKeep) break;

        const double xShift = (j & 1) ? (0.5 * dx) : 0.0;

        for (int i = 0; i < pbc.nx; ++i)
        {
            double x = x0 + i * dx + xShift;
            if (!periodicX && x >= xMaxKeep) break;

            // If periodicX and you ever choose to overshoot, you can wrap here:
            // x = x0 + std::fmod(std::fmod(x - x0, pbc.Lx) + pbc.Lx, pbc.Lx);

            points.push_back(make_double3(x, y, z0));
        }
    }
}

struct TriangleMesh
{
    std::vector<double3> vertices;

    std::vector<int> tri0;
    std::vector<int> tri1;
    std::vector<int> tri2;

    void reserveTriangles(size_t n)
    {
        tri0.reserve(n); tri1.reserve(n); tri2.reserve(n);
    }

    void addTriangle(int i0, int i1, int i2)
    {
        tri0.push_back(i0);
        tri1.push_back(i1);
        tri2.push_back(i2);
    }

    size_t numTriangles() const { return tri0.size(); }
};

inline void buildOrthonormalBasis(const double3& axisUnit, double3& u, double3& v)
{
    // pick a vector not parallel to axis
    const double3 a = axisUnit;
    const double3 tmp = (std::fabs(a.z) < 0.9) ? make_double3(0,0,1) : make_double3(0,1,0);

    u = normalize(cross(tmp, a));   // u ⟂ a
    v = normalize(cross(a, u));     // v ⟂ a, and {u,v,a} right-handed
}

inline TriangleMesh makeCylinderMesh(const double3& center,
                                     const double3& axis,
                                     double radius,
                                     double height,
                                     int nAround,
                                     int nHeight,
                                     bool caps = true)
{
    TriangleMesh mesh;

    assert(nAround >= 3);
    assert(nHeight >= 1);
    assert(radius > 0.0);
    assert(height > 0.0);
    assert(dot(axis, axis) > 1e-30);   // avoid NaN normalize

    const double3 w = normalize(axis);
    double3 u, v;
    buildOrthonormalBasis(w, u, v);

    const int ringCount = nHeight + 1;
    const int sideVertCount = ringCount * nAround;

    mesh.vertices.reserve(sideVertCount + (caps ? 2 : 0));
    mesh.reserveTriangles(
        size_t(nHeight) * size_t(nAround) * 2 + (caps ? size_t(nAround) * 2 : 0)
    );

    // vertices (rings)
    for (int iz = 0; iz < ringCount; ++iz)
    {
        const double t = double(iz) / double(nHeight); // 0..1
        const double z = (t - 0.5) * height;           // -h/2 .. +h/2
        const double3 c = center + z * w;

        for (int ia = 0; ia < nAround; ++ia)
        {
            const double ang = (2.0 * M_PI) * (double(ia) / double(nAround));
            const double ca = std::cos(ang);
            const double sa = std::sin(ang);
            const double3 p = c + radius * (ca * u + sa * v);
            mesh.vertices.push_back(p);
        }
    }

    auto vid = [&](int iz, int ia) -> int
    {
        int a = ia % nAround;
        if (a < 0) a += nAround;
        return iz * nAround + a;
    };

    // side triangles
    for (int iz = 0; iz < nHeight; ++iz)
    {
        for (int ia = 0; ia < nAround; ++ia)
        {
            const int i00 = vid(iz,   ia);
            const int i01 = vid(iz,   ia + 1);
            const int i10 = vid(iz+1, ia);
            const int i11 = vid(iz+1, ia + 1);

            mesh.addTriangle(i00, i10, i11);
            mesh.addTriangle(i00, i11, i01);
        }
    }

    if (!caps) return mesh;

    // caps
    const int bottomCenter = (int)mesh.vertices.size();
    mesh.vertices.push_back(center + (-0.5 * height) * w);

    const int topCenter = (int)mesh.vertices.size();
    mesh.vertices.push_back(center + (0.5 * height) * w);

    const int bottomRingZ = 0;
    const int topRingZ    = nHeight;

    // Bottom cap: want normal ~ -w
    for (int ia = 0; ia < nAround; ++ia)
    {
        const int i0 = vid(bottomRingZ, ia);
        const int i1 = vid(bottomRingZ, ia + 1);
        mesh.addTriangle(bottomCenter, i1, i0);
    }

    // Top cap: want normal ~ +w
    for (int ia = 0; ia < nAround; ++ia)
    {
        const int i0 = vid(topRingZ, ia);
        const int i1 = vid(topRingZ, ia + 1);
        mesh.addTriangle(topCenter, i0, i1);
    }

    return mesh;
}

// -----------------------------------------------------------------------------
// Make a cone frustum mesh aligned with +Z axis, centered at `center`.
// - bottom circle at z = center.z - height/2
// - top    circle at z = center.z + height/2
// - bottomRadius = bottomDiameter * 0.5
// - topRadius    = topDiameter    * 0.5
// - nAround >= 3
// - caps: add bottom/top caps (top cap is skipped if topRadius == 0)
// Winding: consistent; if you need opposite normals, swap i1/i2 in addTriangle.
// -----------------------------------------------------------------------------
inline TriangleMesh makeConeFrustumMeshZ(const double3& center,
                                        double topDiameter,
                                        double bottomDiameter,
                                        double height,
                                        int nAround,
                                        int nHeight = 1,
                                        bool caps = true)
{
    TriangleMesh mesh;

    assert(nAround >= 3);
    assert(nHeight >= 1);
    assert(height > 0.0);
    assert(topDiameter >= 0.0);
    assert(bottomDiameter >= 0.0);

    const double rTop = 0.5 * topDiameter;
    const double rBot = 0.5 * bottomDiameter;

    // If both radii are zero, nothing to build
    if (rTop <= 0.0 && rBot <= 0.0) return mesh;

    const int ringCount = nHeight + 1;
    const int ringVerts = nAround;

    // side vertices: ringCount * nAround
    const int sideVertCount = ringCount * ringVerts;

    // Reserve vertices (+ cap centers)
    mesh.vertices.reserve(sideVertCount + (caps ? 2 : 0));
    mesh.reserveTriangles(
        size_t(nHeight) * size_t(nAround) * 2
        + (caps ? size_t(nAround) : 0)              // bottom cap
        + (caps && rTop > 0.0 ? size_t(nAround) : 0) // top cap if non-degenerate
    );

    const double z0 = center.z - 0.5 * height;
    const double z1 = center.z + 0.5 * height;

    // -------------------------
    // vertices (rings)
    // -------------------------
    for (int iz = 0; iz < ringCount; ++iz)
    {
        const double t = double(iz) / double(nHeight); // 0..1
        const double z = (1.0 - t) * z0 + t * z1;

        // linear radius along height
        const double r = (1.0 - t) * rBot + t * rTop;

        for (int ia = 0; ia < nAround; ++ia)
        {
            const double ang = (2.0 * M_PI) * (double(ia) / double(nAround));
            const double x = center.x + r * std::cos(ang);
            const double y = center.y + r * std::sin(ang);
            mesh.vertices.push_back(make_double3(x, y, z));
        }
    }

    auto vid = [&](int iz, int ia) -> int
    {
        int a = ia % nAround;
        if (a < 0) a += nAround;
        return iz * nAround + a;
    };

    // -------------------------
    // side triangles
    // -------------------------
    for (int iz = 0; iz < nHeight; ++iz)
    {
        for (int ia = 0; ia < nAround; ++ia)
        {
            const int i00 = vid(iz,   ia);
            const int i01 = vid(iz,   ia + 1);
            const int i10 = vid(iz+1, ia);
            const int i11 = vid(iz+1, ia + 1);

            // quad -> 2 triangles
            mesh.addTriangle(i00, i10, i11);
            mesh.addTriangle(i00, i11, i01);
        }
    }

    if (!caps) return mesh;

    // -------------------------
    // bottom cap (always if rBot>0)
    // -------------------------
    if (rBot > 0.0)
    {
        const int bottomCenter = (int)mesh.vertices.size();
        mesh.vertices.push_back(make_double3(center.x, center.y, z0));

        const int bottomRingZ = 0;
        for (int ia = 0; ia < nAround; ++ia)
        {
            const int i0 = vid(bottomRingZ, ia);
            const int i1 = vid(bottomRingZ, ia + 1);
            // bottom cap (winding chosen; flip if needed)
            mesh.addTriangle(bottomCenter, i1, i0);
        }
    }

    // -------------------------
    // top cap (only if rTop>0; if rTop==0 it's a cone tip => no cap)
    // -------------------------
    if (rTop > 0.0)
    {
        const int topCenter = (int)mesh.vertices.size();
        mesh.vertices.push_back(make_double3(center.x, center.y, z1));

        const int topRingZ = nHeight;
        for (int ia = 0; ia < nAround; ++ia)
        {
            const int i0 = vid(topRingZ, ia);
            const int i1 = vid(topRingZ, ia + 1);
            mesh.addTriangle(topCenter, i0, i1);
        }
    }

    return mesh;
}