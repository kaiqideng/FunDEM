#pragma once
#include "kernel/CUDAKernelFunction/myUtility/myVec.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

class SpherePack
{
public:
    //=====================================================================
    // Data
    //=====================================================================
    std::vector<double3> centers;
    std::vector<double>  radii;

public:
    //=====================================================================
    // Build: Non-overlapping (large first)
    //=====================================================================
    void buildNonOverlappingInBox_LargeFirst(
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

        if (nSpheres <= 0 || rMin <= 0.0 || rMax < rMin) return;

        const double3 bmin = sortedMin(boxMin, boxMax);
        const double3 bmax = sortedMax(boxMin, boxMax);

        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> ur01(0.0, 1.0);
        std::uniform_real_distribution<double> urR(rMin, rMax);

        // ------------------------------------------------------------
        // radii: large -> small
        // ------------------------------------------------------------
        std::vector<double> targetRadii(nSpheres);
        for (auto& r : targetRadii) r = urR(rng);
        std::sort(targetRadii.begin(), targetRadii.end(), std::greater<double>());

        // ------------------------------------------------------------
        // spatial hash
        // ------------------------------------------------------------
        const double cellSize = 2.0 * rMax;

        auto cellCoord = [&](const double3& p)
        {
            return std::array<int,3>{
                (int)std::floor((p.x - bmin.x)/cellSize),
                (int)std::floor((p.y - bmin.y)/cellSize),
                (int)std::floor((p.z - bmin.z)/cellSize)
            };
        };

        std::unordered_map<int64_t, std::vector<int>> buckets;
        buckets.reserve(nSpheres * 2);

        auto overlapsAny = [&](const double3& c, double r)
        {
            const auto cc = cellCoord(c);

            for (int dz=-1; dz<=1; ++dz)
            for (int dy=-1; dy<=1; ++dy)
            for (int dx=-1; dx<=1; ++dx)
            {
                int64_t key = packKey(cc[0]+dx, cc[1]+dy, cc[2]+dz);
                auto it = buckets.find(key);
                if (it == buckets.end()) continue;

                for (int id : it->second)
                {
                    const double3 d = c - centers[id];
                    const double rr = r + radii[id];
                    if (dot(d,d) < rr*rr) return true;
                }
            }
        return false;
        };

        auto sampleCenter = [&](double r)
        {
            return make_double3(
                (bmin.x+r) + ur01(rng)*(bmax.x-bmin.x-2*r),
                (bmin.y+r) + ur01(rng)*(bmax.y-bmin.y-2*r),
                (bmin.z+r) + ur01(rng)*(bmax.z-bmin.z-2*r)
            );
        };

        // ------------------------------------------------------------
        // place spheres
        // ------------------------------------------------------------
        for (double r : targetRadii)
        {
            for (int attempt=0; attempt<maxAttemptsPerSphere; ++attempt)
            {
                const double3 c = sampleCenter(r);
                if (overlapsAny(c,r)) continue;

                const int id = (int)centers.size();
                centers.push_back(c);
                radii.push_back(r);

                const auto cc = cellCoord(c);
                buckets[packKey(cc[0],cc[1],cc[2])].push_back(id);
                break;
            }
        }
    }

    //=====================================================================
    // Build: Regular grid
    //=====================================================================
    void buildRegularInBox(
        const double3 boxMin,
        const double3 boxMax,
        const double r)
    {
        centers.clear();
        radii.clear();

        if (r <= 0.0) return;

        const double dx = 2.0 * r;

        for (double z = boxMin.z + r; z <= boxMax.z - r; z += dx)
        for (double y = boxMin.y + r; y <= boxMax.y - r; y += dx)
        for (double x = boxMin.x + r; x <= boxMax.x - r; x += dx)
        {
            centers.push_back(make_double3(x,y,z));
            radii.push_back(r);
        }
    }

    //=====================================================================
    // Build: Hex packing
    //=====================================================================
    void buildHexInBox(
        const double3 boxMin,
        const double3 boxMax,
        const double r)
    {
        centers.clear();
        radii.clear();

        if (r <= 0.0) return;

        const double dx = 2*r;
        const double dy = std::sqrt(3.0)*r;
        const double dz = std::sqrt(8.0/3.0)*r;

        int k = 0;
        for (double z = boxMin.z + r; z <= boxMax.z - r; z += dz, ++k)
        {
            const bool oddLayer = (k % 2 == 1);

            int j = 0;
            for (double y = boxMin.y + r; y <= boxMax.y - r; y += dy, ++j)
            {
                const bool oddRow = (j % 2 == 1);

                double xStart = boxMin.x + r;
                if (oddRow)   xStart += r;
                if (oddLayer) xStart += r;

                for (double x = xStart; x <= boxMax.x - r; x += dx)
                {
                    centers.push_back(make_double3(x,y,z));
                    radii.push_back(r);
                }
            }
        }
    }

private:
    //=====================================================================
    // Utilities
    //=====================================================================
    static double3 sortedMin(const double3& a, const double3& b)
    {
        return make_double3(std::min(a.x,b.x),
                            std::min(a.y,b.y),
                            std::min(a.z,b.z));
    }

    static double3 sortedMax(const double3& a, const double3& b)
    {
        return make_double3(std::max(a.x,b.x),
                            std::max(a.y,b.y),
                            std::max(a.z,b.z));
    }

    static int64_t packKey(int x,int y,int z)
    {
        const int64_t B = (1LL<<20);
        return ((int64_t)(x+B)<<42)^((int64_t)(y+B)<<21)^(z+B);
    }
};