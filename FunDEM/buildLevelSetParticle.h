#pragma once
#include "kernel/CUDAKernelFunction/myUtility/myVec.h"
#include <array>
#include <cfloat>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <random>

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

inline double3 safeNormalize(const double3& v)
{
    const double n2 = dot(v, v);
    if (n2 < 1e-30) return make_double3(1.0, 0.0, 0.0);
    return v / std::sqrt(n2);
}

inline void buildOrthonormalBasis(const double3& nUnit, double3& u, double3& v)
{
    // pick a vector not parallel to n
    double3 tmp = (std::fabs(nUnit.z) < 0.9) ? make_double3(0,0,1) : make_double3(0,1,0);
    u = safeNormalize(cross(tmp, nUnit));
    v = cross(nUnit, u);
}

struct Plane
{
    double3 n;   // unit normal
    double  d;   // plane offset: n·x = d
};

// Solve A x = b for 3x3, where A rows are a0,a1,a2 and b = (b0,b1,b2).
inline bool solve3x3Rows(const double3& a0, const double3& a1, const double3& a2,
                         const double b0, const double b1, const double b2,
                         double3& x)
{
    // determinant det(A) = a0 · (a1 x a2)
    const double3 c = cross(a1, a2);
    const double det = dot(a0, c);
    if (std::fabs(det) < 1e-20) return false;

    // Cramer's rule using triple products:
    // x = (b0*(a1 x a2) + b1*(a2 x a0) + b2*(a0 x a1)) / det
    const double3 term0 = c * b0;
    const double3 term1 = cross(a2, a0) * b1;
    const double3 term2 = cross(a0, a1) * b2;
    x = (term0 + term1 + term2) / det;
    return true;
}

inline bool insideAllHalfspaces(const double3& p, const std::vector<Plane>& planes, const double tol)
{
    for (const auto& pl : planes)
    {
        if (dot(pl.n, p) - pl.d > tol) return false; // violates n·x <= d
    }
    return true;
}

inline void dedupPoints(std::vector<double3>& pts, const double eps)
{
    auto key = [&](const double3& p)
    {
        // quantize to grid eps for stable dedup
        long long ix = (long long)std::llround(p.x / eps);
        long long iy = (long long)std::llround(p.y / eps);
        long long iz = (long long)std::llround(p.z / eps);
        return std::array<long long,3>{ix,iy,iz};
    };

    std::vector<size_t> order(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) order[i] = i;

    std::sort(order.begin(), order.end(), [&](size_t a, size_t b)
    {
        return key(pts[a]) < key(pts[b]);
    });

    std::vector<double3> out;
    out.reserve(pts.size());

    std::array<long long,3> lastKey = {LLONG_MIN, LLONG_MIN, LLONG_MIN};
    for (size_t idx : order)
    {
        auto k = key(pts[idx]);
        if (k != lastKey)
        {
            out.push_back(pts[idx]);
            lastKey = k;
        }
    }
    pts.swap(out);
}

// -----------------------------------------------------------------------------
// Random convex polyhedron = intersection of halfspaces
//
// bound     : initial cube half extent (poly lives roughly in [-bound, bound]^3)
// numCuts   : number of random cutting planes
// seed      : RNG seed
//
// Returns a TRIANGULATED convex polyhedron mesh.
// -----------------------------------------------------------------------------
inline TriangleMesh makeRandomConvexPolyhedron(const int numCuts,
                                              const unsigned seed = 1u,
                                              const double bound = 1.0)
{
    TriangleMesh mesh;
    if (bound <= 0.0) return mesh;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    std::normal_distribution<double> normal01(0.0, 1.0);

    // ---------------------------------------------------------
    // 1) Build halfspaces: cube + random cuts
    // Cube:  x<=b, -x<=b, y<=b, -y<=b, z<=b, -z<=b
    // ---------------------------------------------------------
    std::vector<Plane> planes;
    planes.reserve(6 + std::max(0, numCuts));

    auto addPlane = [&](double3 n, double d)
    {
        Plane p;
        p.n = safeNormalize(n);
        p.d = d;
        planes.push_back(p);
    };

    addPlane(make_double3( 1, 0, 0),  bound);
    addPlane(make_double3(-1, 0, 0),  bound);
    addPlane(make_double3( 0, 1, 0),  bound);
    addPlane(make_double3( 0,-1, 0),  bound);
    addPlane(make_double3( 0, 0, 1),  bound);
    addPlane(make_double3( 0, 0,-1),  bound);

    // Random planes n·x <= d, choose d so it actually cuts inside cube
    for (int k = 0; k < numCuts; ++k)
    {
        double3 n = make_double3(normal01(rng), normal01(rng), normal01(rng));
        n = safeNormalize(n);

        // pick d in (0.3..0.95)*bound to keep some volume
        double d = (0.30 + 0.65 * uni01(rng)) * bound;
        addPlane(n, d);
    }

    // ---------------------------------------------------------
    // 2) Enumerate triple-plane intersections -> candidate vertices
    // ---------------------------------------------------------
    const double insideTol = 1e-9;
    std::vector<double3> verts;
    verts.reserve(planes.size() * planes.size());

    for (size_t i = 0; i < planes.size(); ++i)
    {
        for (size_t j = i + 1; j < planes.size(); ++j)
        {
            for (size_t k = j + 1; k < planes.size(); ++k)
            {
                double3 p;
                if (!solve3x3Rows(planes[i].n, planes[j].n, planes[k].n,
                                 planes[i].d, planes[j].d, planes[k].d, p))
                {
                    continue;
                }
                if (insideAllHalfspaces(p, planes, insideTol))
                {
                    verts.push_back(p);
                }
            }
        }
    }

    if (verts.size() < 4) return mesh;

    // Deduplicate vertices
    dedupPoints(verts, 1e-8);

    if (verts.size() < 4) return mesh;

    // ---------------------------------------------------------
    // 3) Build faces: for each plane, collect vertices on that plane,
    //    sort them in CCW order, triangulate fan.
    // ---------------------------------------------------------
    mesh.vertices = verts;

    const double onPlaneTol = 5e-7;
    const int nx = (int)mesh.vertices.size();

    for (size_t pi = 0; pi < planes.size(); ++pi)
    {
        const Plane& pl = planes[pi];

        // collect indices of vertices close to this plane
        std::vector<int> face;
        face.reserve(64);

        for (int vi = 0; vi < nx; ++vi)
        {
            const double s = dot(pl.n, mesh.vertices[(size_t)vi]) - pl.d;
            if (std::fabs(s) <= onPlaneTol)
            {
                face.push_back(vi);
            }
        }

        // Need at least a triangle
        if (face.size() < 3) continue;

        // Compute centroid
        double3 c = make_double3(0,0,0);
        for (int id : face) c += mesh.vertices[(size_t)id];
        c /= (double)face.size();

        // Build plane basis (u,v)
        double3 u, v;
        buildOrthonormalBasis(pl.n, u, v);

        // Sort by angle around centroid
        struct Item { int id; double ang; };
        std::vector<Item> items;
        items.reserve(face.size());

        for (int id : face)
        {
            double3 r = mesh.vertices[(size_t)id] - c;
            const double x = dot(r, u);
            const double y = dot(r, v);
            items.push_back({id, std::atan2(y, x)});
        }

        std::sort(items.begin(), items.end(),
                  [](const Item& a, const Item& b){ return a.ang < b.ang; });

        // Remove near-duplicate consecutive vertices in angle order
        std::vector<int> poly;
        poly.reserve(items.size());
        for (size_t t = 0; t < items.size(); ++t) poly.push_back(items[t].id);

        // If poly degenerates, skip
        if (poly.size() < 3) continue;

        // Triangulate fan: (poly[0], poly[i], poly[i+1])
        // Winding: depends on outward direction.
        // Our halfspace is n·x <= d, so outward normal is +n.
        // The sorted order based on (u,v) gives CCW w.r.t +n.
        for (size_t t = 1; t + 1 < poly.size(); ++t)
        {
            mesh.addTriangle(poly[0], poly[t], poly[t + 1]);
        }
    }

    return mesh;
}

struct LevelSetParticleGridInfo
{
    // Grid geometry in WORLD/GLOBAL frame
    double3 gridMin {0.0, 0.0, 0.0};     // background grid min corner (global)
    int3    gridNodeSize {0, 0, 0};      // (nx, ny, nz)
    double  gridSpacing {0.0};           // h = diameter / 20

    // Flattened phi array, row-major:
    // idx = (k * ny + j) * nx + i
    // Convention here: phi > 0 INSIDE, phi < 0 OUTSIDE
    std::vector<double> gridNodeLSF;
};

// -----------------------------------------------------------------------------
// Build a level-set particle background grid from a triangle mesh (GLOBAL frame).
//
// Inputs:
//   vertices : mesh vertices in GLOBAL frame
//   tri0/tri1/tri2: triangle index arrays (same length = numTriangles)
//   paddingLayers : padding in number of grid cells around mesh bbox
//
// Grid spacing rule:
//   h = diameter / 20, where diameter is the max bbox extent.
//
// Output:
//   info.gridMin, info.gridNodeSize, info.gridSpacing, info.gridNodeLSF
//
// Notes:
// - phi > 0 inside the closed mesh, phi < 0 outside.
// - Uses:
//    (1) unsigned distance = min distance to triangles
//    (2) inside test by ray casting (+x direction), assuming mesh is CLOSED
// - Complexity: O(Ngrid * Ntri), intended for initialization.
// -----------------------------------------------------------------------------
inline LevelSetParticleGridInfo buildLevelSetParticleFromTriangleMeshGlobal(const std::vector<double3>& vertices,
                                                                           const std::vector<int>& tri0,
                                                                           const std::vector<int>& tri1,
                                                                           const std::vector<int>& tri2,
                                                                           const int paddingLayers = 2)
{
    LevelSetParticleGridInfo info;

    if (vertices.empty()) return info;
    if (tri0.size() != tri1.size() || tri0.size() != tri2.size()) return info;
    if (tri0.empty()) return info;

    // ---------------------------------------------------------
    // Helper: point-triangle squared distance (Ericson style)
    // ---------------------------------------------------------
    auto pointTriangleDist2 = [&](const double3& p,
                                  const double3& a,
                                  const double3& b,
                                  const double3& c) -> double
    {
        const double3 ab = b - a;
        const double3 ac = c - a;
        const double3 ap = p - a;

        const double d1 = dot(ab, ap);
        const double d2 = dot(ac, ap);
        if (d1 <= 0.0 && d2 <= 0.0) return dot(ap, ap);

        const double3 bp = p - b;
        const double d3 = dot(ab, bp);
        const double d4 = dot(ac, bp);
        if (d3 >= 0.0 && d4 <= d3) return dot(bp, bp);

        const double vc = d1 * d4 - d3 * d2;
        if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
        {
            const double v = d1 / (d1 - d3);
            const double3 proj = a + v * ab;
            const double3 diff = p - proj;
            return dot(diff, diff);
        }

        const double3 cp = p - c;
        const double d5 = dot(ab, cp);
        const double d6 = dot(ac, cp);
        if (d6 >= 0.0 && d5 <= d6) return dot(cp, cp);

        const double vb = d5 * d2 - d1 * d6;
        if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
        {
            const double w = d2 / (d2 - d6);
            const double3 proj = a + w * ac;
            const double3 diff = p - proj;
            return dot(diff, diff);
        }

        const double va = d3 * d6 - d5 * d4;
        if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
        {
            const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            const double3 proj = b + w * (c - b);
            const double3 diff = p - proj;
            return dot(diff, diff);
        }

        const double3 n = cross(ab, ac);
        const double n2 = dot(n, n);
        if (n2 < 1e-30) return dot(ap, ap);

        const double t = dot(p - a, n) / n2;
        const double3 proj = p - t * n;
        const double3 diff = p - proj;
        return dot(diff, diff);
    };

    // ---------------------------------------------------------
    // Helper: ray-triangle intersection (Möller–Trumbore)
    // Return true and write tOut when hit.
    // ---------------------------------------------------------
    auto rayIntersectsTriangleT = [&](const double3& p,
                                      const double3& dir,
                                      const double3& a,
                                      const double3& b,
                                      const double3& c,
                                      double& tOut) -> bool
    {
        const double3 e1 = b - a;
        const double3 e2 = c - a;

        const double3 h = cross(dir, e2);
        const double det = dot(e1, h);

        const double detEps = 1e-14;
        if (std::fabs(det) <= detEps) return false;

        const double invDet = 1.0 / det;

        const double3 s = p - a;
        const double u = dot(s, h) * invDet;

        const double uvEps = 1e-14;
        if (u < -uvEps || u > 1.0 - uvEps) return false;

        const double3 q = cross(s, e1);
        const double v = dot(dir, q) * invDet;

        if (v < -uvEps || (u + v) > 1.0 - uvEps) return false;

        const double t = dot(e2, q) * invDet;

        const double tEps = 1e-12;
        if (t <= tEps) return false;

        tOut = t;
        return true;
    };

    // ---------------------------------------------------------
    // 1) Mesh bbox (GLOBAL)
    // ---------------------------------------------------------
    double3 bmin = vertices[0];
    double3 bmax = vertices[0];
    for (size_t i = 1; i < vertices.size(); ++i)
    {
        const double3 p = vertices[i];
        bmin.x = std::min(bmin.x, p.x); bmin.y = std::min(bmin.y, p.y); bmin.z = std::min(bmin.z, p.z);
        bmax.x = std::max(bmax.x, p.x); bmax.y = std::max(bmax.y, p.y); bmax.z = std::max(bmax.z, p.z);
    }

    // ---------------------------------------------------------
    // 2) Grid spacing: diameter / 20
    // ---------------------------------------------------------
    const double dx = bmax.x - bmin.x;
    const double dy = bmax.y - bmin.y;
    const double dz = bmax.z - bmin.z;
    const double diameter = std::max(dx, std::max(dy, dz));
    if (diameter <= 0.0) return info;

    const double h = diameter / 20.0;
    info.gridSpacing = h;

    // ---------------------------------------------------------
    // 3) Grid bbox with padding (GLOBAL)
    // ---------------------------------------------------------
    const double pad = double(std::max(1, paddingLayers)) * h;
    double3 gmin = make_double3(bmin.x - pad, bmin.y - pad, bmin.z - pad);
    double3 gmax = make_double3(bmax.x + pad, bmax.y + pad, bmax.z + pad);

    auto snapDown = [&](double a) { return h * std::floor(a / h); };
    auto snapUp   = [&](double a) { return h * std::ceil (a / h); };

    gmin.x = snapDown(gmin.x); gmin.y = snapDown(gmin.y); gmin.z = snapDown(gmin.z);
    gmax.x = snapUp  (gmax.x); gmax.y = snapUp  (gmax.y); gmax.z = snapUp  (gmax.z);

    const int nx = (int)std::round((gmax.x - gmin.x) / h) + 1;
    const int ny = (int)std::round((gmax.y - gmin.y) / h) + 1;
    const int nz = (int)std::round((gmax.z - gmin.z) / h) + 1;
    if (nx <= 0 || ny <= 0 || nz <= 0) return info;

    info.gridMin = gmin;
    info.gridNodeSize = make_int3(nx, ny, nz);
    info.gridNodeLSF.resize((size_t)nx * (size_t)ny * (size_t)nz, 0.0);

    // ---------------------------------------------------------
    // 4) Signed distance: + inside, - outside
    // ---------------------------------------------------------
    const double3 rayDir = normalize(make_double3(1.0, 0.37, 0.23));

    auto idx3 = [&](int i, int j, int k) -> size_t
    {
        return (size_t)((k * ny + j) * nx + i);
    };

    // temp buffer for hit distances (t) per grid point
    std::vector<double> hitTs;
    hitTs.reserve(128);

    // tolerance for "same intersection" bucketing (scale by grid)
    const double tBucket = 1e-6 * h;

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

                // jitter point for ray casting (scale aware)
                const double jj = 1e-6 * h;
                const double3 p2 = make_double3(p.x + 0.17 * jj,
                                                p.y + 0.53 * jj,
                                                p.z + 0.29 * jj);

                // (a) unsigned distance
                double bestD2 = DBL_MAX;
                for (size_t t = 0; t < tri0.size(); ++t)
                {
                    const int i0 = tri0[t];
                    const int i1 = tri1[t];
                    const int i2 = tri2[t];
                    if ((unsigned)i0 >= vertices.size()) continue;
                    if ((unsigned)i1 >= vertices.size()) continue;
                    if ((unsigned)i2 >= vertices.size()) continue;

                    const double3 a = vertices[(size_t)i0];
                    const double3 b = vertices[(size_t)i1];
                    const double3 c = vertices[(size_t)i2];

                    const double d2 = pointTriangleDist2(p, a, b, c);
                    if (d2 < bestD2) bestD2 = d2;
                }
                const double dist = std::sqrt(std::max(0.0, bestD2));

                // (b) inside/outside by ray parity, BUT de-duplicate intersections
                hitTs.clear();

                for (size_t t = 0; t < tri0.size(); ++t)
                {
                    const int i0 = tri0[t];
                    const int i1 = tri1[t];
                    const int i2 = tri2[t];
                    if ((unsigned)i0 >= vertices.size()) continue;
                    if ((unsigned)i1 >= vertices.size()) continue;
                    if ((unsigned)i2 >= vertices.size()) continue;

                    const double3 a = vertices[(size_t)i0];
                    const double3 b = vertices[(size_t)i1];
                    const double3 c = vertices[(size_t)i2];

                    double tt = 0.0;
                    if (rayIntersectsTriangleT(p2, rayDir, a, b, c, tt))
                    {
                        hitTs.push_back(tt);
                    }
                }

                // sort + unique by bucket to avoid double counting at shared edges/vertices
                std::sort(hitTs.begin(), hitTs.end());
                int hits = 0;
                double last = -DBL_MAX;

                for (double tt : hitTs)
                {
                    if (tt - last > tBucket)
                    {
                        hits++;
                        last = tt;
                    }
                }

                const bool inside = (hits & 1);
                info.gridNodeLSF[idx3(i, j, k)] = inside ? -dist : dist;
            }
        }
    }

    return info;
}

// -----------------------------------------------------------------------------
// Write level-set grid nodes to VTU (UnstructuredGrid with VTK_VERTEX cells)
// - Points: all grid nodes
// - PointData: phi
//
// Usage:
//   writeLevelSetGridVTU("ls_grid.vtu", info, /*writeAsFloat=*/true);
//
// Note:
// - This writes a point cloud (not a structured VTK image).
// - For ParaView, this is usually enough to visualize phi as a scalar field.
// -----------------------------------------------------------------------------
inline void writeLevelSetGridVTU(const std::string& filename,
                                 const LevelSetParticleGridInfo& grid,
                                 const bool writeAsFloat = true)
{
    const int nx = grid.gridNodeSize.x;
    const int ny = grid.gridNodeSize.y;
    const int nz = grid.gridNodeSize.z;

    if (nx <= 0 || ny <= 0 || nz <= 0) return;

    const size_t N = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
    if (grid.gridNodeLSF.size() != N) return;

    std::ofstream out(filename.c_str());
    if (!out) throw std::runtime_error("Cannot open " + filename);

    out << std::fixed << std::setprecision(10);

    out << "<?xml version=\"1.0\"?>\n"
        << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
        << "  <UnstructuredGrid>\n"
        << "    <Piece NumberOfPoints=\"" << N << "\" NumberOfCells=\"" << N << "\">\n";

    // ---------------------------------------------------------
    // Points
    // ---------------------------------------------------------
    out << "      <Points>\n";
    out << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";

    const double h = grid.gridSpacing;
    const double3 g0 = grid.gridMin;

    for (int k = 0; k < nz; ++k)
    {
        const double z = g0.z + double(k) * h;
        for (int j = 0; j < ny; ++j)
        {
            const double y = g0.y + double(j) * h;
            for (int i = 0; i < nx; ++i)
            {
                const double x = g0.x + double(i) * h;
                out << ' ' << x << ' ' << y << ' ' << z;
            }
        }
    }

    out << "\n        </DataArray>\n";
    out << "      </Points>\n";

    // ---------------------------------------------------------
    // Cells: one VTK_VERTEX per point
    // ---------------------------------------------------------
    out << "      <Cells>\n";

    // connectivity
    out << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (size_t i = 0; i < N; ++i) out << ' ' << static_cast<int>(i);
    out << "\n        </DataArray>\n";

    // offsets
    out << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (size_t i = 1; i <= N; ++i) out << ' ' << static_cast<int>(i);
    out << "\n        </DataArray>\n";

    // types (1 = VTK_VERTEX)
    out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (size_t i = 0; i < N; ++i) out << " 1";
    out << "\n        </DataArray>\n";

    out << "      </Cells>\n";

    // ---------------------------------------------------------
    // PointData: phi
    // ---------------------------------------------------------
    out << "      <PointData Scalars=\"phi\">\n";

    out << "        <DataArray type=\"" << (writeAsFloat ? "Float32" : "Float64")
        << "\" Name=\"phi\" format=\"ascii\">\n";

    if (writeAsFloat)
    {
        for (size_t i = 0; i < N; ++i) out << ' ' << static_cast<float>(grid.gridNodeLSF[i]);
    }
    else
    {
        for (size_t i = 0; i < N; ++i) out << ' ' << grid.gridNodeLSF[i];
    }

    out << "\n        </DataArray>\n";
    out << "      </PointData>\n";

    out << "    </Piece>\n"
        << "  </UnstructuredGrid>\n"
        << "</VTKFile>\n";
}

// ------------------------------------------------------------
// Optional: write the surface mesh as OBJ for quick viewing
// ------------------------------------------------------------
inline void writeTriangleMeshOBJ(const std::string& filename, const TriangleMesh& m)
{
    std::ofstream out(filename.c_str());
    if (!out) throw std::runtime_error("Cannot open " + filename);

    out << std::fixed << std::setprecision(10);

    for (const auto& v : m.vertices)
        out << "v " << v.x << " " << v.y << " " << v.z << "\n";

    // OBJ uses 1-based indices
    for (size_t t = 0; t < m.tri0.size(); ++t)
        out << "f " << (m.tri0[t] + 1) << " " << (m.tri1[t] + 1) << " " << (m.tri2[t] + 1) << "\n";
}

// Write surface points as an OBJ point cloud (vertices only).
// You can open it in ParaView / MeshLab. No faces are written.
inline void writePointCloudObj(const std::string& filename,
                               const std::vector<double3>& points)
{
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Cannot open OBJ file: " + filename);

    out << "# Point cloud OBJ (vertices only)\n";
    out << "# vertex count: " << points.size() << "\n";
    out << std::fixed << std::setprecision(10);

    for (const auto& p : points)
    {
        out << "v " << p.x << " " << p.y << " " << p.z << "\n";
    }
}