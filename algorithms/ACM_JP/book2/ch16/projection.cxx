#include <iomanip>
#include <ios>
#include <iostream>
#include "geometry_object.h"

int main() {
    Segment s;
    std::cin >> s.p1.x >> s.p1.y >> s.p2.x >> s.p2.y;

    int q;
    std::cin >> q;
    
    for (int i = 0; i < q; ++i) {
        Point p;
        std::cin >> p.x >> p.y;

        Point x = project(s, p);
        std::cout << std::setprecision(8) << std::fixed << x.x << ' ' << x.y << '\n';
    }

    return 0;
}