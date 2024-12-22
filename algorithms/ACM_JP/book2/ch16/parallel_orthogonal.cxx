#include <iostream>
#include "geometry_object.h"

int main() {
    int q;
    std::cin >> q;

    for (int i = 0; i < q; ++i) {
        Segment s1, s2;
        std::cin >> s1.p1.x >> s1.p1.y >> s1.p2.x >> s1.p2.y;
        std::cin >> s2.p1.x >> s2.p1.y >> s2.p2.x >> s2.p2.y;

        if (isParallel(s1, s2)) {
            std::cout << 2 << '\n';
        }
        else if (isOrthogonal(s1, s2)) {
            std::cout << 1 << '\n';
        }
        else {
            std::cout << 0 << '\n';
        }
    }

    return 0;
}