#include "buildLevelSetParticle.h"
#include <iostream>

int main(int argc, char** argv)
{
    // You can change these from command line if you want.
    const int    numCuts = (argc > 1) ? std::max(0, std::atoi(argv[1])) : 12;
    const unsigned seed  = (argc > 2) ? (unsigned)std::atoi(argv[2]) : 42u;
    const double bound   = (argc > 3) ? std::atof(argv[3]) : 1.0;

    std::cout << "Generating random convex polyhedron...\n";
    TriangleMesh poly = makeRandomConvexPolyhedron(numCuts, seed, bound);
    std::cout << "  vertices = " << poly.vertices.size() << "\n";
    std::cout << "  triangles = " << poly.numTriangles() << "\n";

    if (poly.vertices.size() < 4 || poly.numTriangles() < 4)
    {
        std::cout << "Degenerate polyhedron, try different seed/cuts.\n";
        return 0;
    }

    // Optional: export surface for reference
    writeTriangleMeshOBJ("random_poly.obj", poly);
    std::cout << "Wrote surface: random_poly.obj\n";

    std::cout << "Building level-set grid (phi<0 inside)...\n";
    LevelSetParticleGridInfo ls = buildLevelSetParticleFromTriangleMeshGlobal(poly.vertices, 
    poly.tri0, 
    poly.tri1, 
    poly.tri2,
    /*paddingLayers=*/2);

    std::cout << "  gridMin = (" << ls.gridMin.x << ", " << ls.gridMin.y << ", " << ls.gridMin.z << ")\n";
    std::cout << "  gridSize = (" << ls.gridNodeSize.x << ", " << ls.gridNodeSize.y << ", " << ls.gridNodeSize.z << ")\n";
    std::cout << "  gridSpacing = " << ls.gridSpacing << "\n";

    writeTriangleMeshOBJ("random_poly.obj", poly);
    writeLevelSetGridVTU("levelset_grid.vtu", ls, /*writeAsFloat=*/true);
    std::cout << "Wrote grid VTU: levelset_grid.vtu\n";

    superellipsoidParams s;
    LevelSetParticleGridInfo ls1 = buildLevelSetSuperellipsoidGridGlobal(s, make_double3(0., 0., 0.));
    writeLevelSetGridVTU("levelset_grid1.vtu", ls1, /*writeAsFloat=*/true);
    std::vector<double3> sp = generateSuperellipsoidSurfacePointsGlobal_Uniform10k(s, make_double3(0., 0., 0.));
    writePointCloudObj("superellipsoid.obj", sp);

    return 0;
}
