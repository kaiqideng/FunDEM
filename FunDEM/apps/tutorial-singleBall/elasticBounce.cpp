#include "kernel/DEMSolver.h"

int main()
{
    double r_ball = 0.01;
    double t_c = 0.005;

    DEMSolver test(0);

    test.addBall(make_double3(0.0, 0.0, 0.0), 
    make_double3(0.0, 0.0, 0.0), 
    make_double3(0.0, 0.0, 0.0), 
    r_ball, 
    2000, 
    0);

    std::vector<double3> vertices(4);
    vertices[0] = make_double3(-1.0, -1.0, -1.0);
    vertices[1] = make_double3(1.0, -1.0, -1.0);
    vertices[2] = make_double3(1.0, 1.0, -1.0);
    vertices[3] = make_double3(-1.0, 1.0, -1.0);
    std::vector<int> tri0(2);
    std::vector<int> tri1(2);
    std::vector<int> tri2(2);
    tri0[0] = 0, tri0[1] = 1;
    tri1[0] = 1, tri1[1] = 2;
    tri2[0] = 3, tri2[1] = 3;

    test.addMeshWall(vertices, 
    tri0, 
    tri1, 
    tri2, 
    make_double3(0.0, 0.0, 0.0), 
    make_double3(0.0, 0.0, 0.0), 
    make_double3(0.0, 0.0, 0.0), 
    0);

    double mass = 4.0 / 3.0 * pow(r_ball, 3.0) * pi() * 2000;
    test.setLinearStiffness(0, 0, 0.5 * mass * pi() * pi() / t_c / t_c, 0., 0., 0.);

    test.solve(make_double3(-1.0, -1.0, -1.0),
    make_double3(1.0, 1.0, 1.0),
    make_double3(0.0, 0.0, -9.81),
    t_c / 50.,
    2.0,
    200,
    "elasticBounce");
}