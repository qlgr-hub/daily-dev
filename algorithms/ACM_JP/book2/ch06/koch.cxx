#include <iomanip>
#include <iostream>
#include <cmath>

struct Point {
    double x;
    double y;

    Point() : x{ 0.0 }, y{ 0.0 } {
    }

    Point(double x_, double y_) : x{ x_ }, y{ y_ } {
    }
};

std::ostream& operator<<(std::ostream& out, const Point& p) {
    out << std::setprecision(8) << std::fixed << std::left << std::setw(15) 
        << p.x << ' ' << p.y << '\n';
    return out;
}

void koch(int n, const Point& p1, const Point& p2) {
    if (n == 0)
        return;

    Point s, t, u;
    s.x = (2.0 * p1.x + 1.0 * p2.x) / 3.0;
    s.y = (2.0 * p1.y + 1.0 * p2.y) / 3.0;
    t.x = (1.0 * p1.x + 2.0 * p2.x) / 3.0;
    t.y = (1.0 * p1.y + 2.0 * p2.y) / 3.0;
    
    double th = M_PI * 60.0 / 180.0;
    u.x = (t.x - s.x) * std::cos(th) - (t.y - s.y) * std::sin(th) + s.x;
    u.y = (t.x - s.x) * std::sin(th) + (t.y - s.y) * std::cos(th) + s.y;

    koch(n - 1, p1, s);
    std::cout << s;
    koch(n - 1, s, u);
    std::cout << u;
    koch(n - 1, u, t);
    std::cout << t;
    koch(n - 1, t, p2);
}

int main() {
    int n = 0;
    std::cin >> n;

    Point p1{ 0.0, 0.0 };
    Point p2{ 100.0, 0.0 };

    std::cout << p1;
    koch(n, p1, p2);
    std::cout << p2;

    return 0;
}
