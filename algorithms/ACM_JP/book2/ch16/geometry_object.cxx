#include "geometry_object.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <cassert>

std::ostream& operator<<(std::ostream& out, const CCW& c) {
    switch (c) {
    case CCW::ONLINE_FRONT:      out << "ONLINE_FRONT" << '\n';      break;
    case CCW::CLOCKWISE:         out << "CLOCKWISE" << '\n';         break;
    case CCW::COUNTER_CLOCKWISE: out << "COUNTER_CLOCKWISE" << '\n'; break;
    case CCW::ONLINE_BACK:       out << "ONLINE_BACK" << '\n';       break;
    default:                     out << "ON_SEGMENT" << '\n';        break;
    }
    return out;
}


CCW ccw(Point p0, Point p1, Point p2) {
    Vector a = p1 - p0;
    Vector b = p2 - p0;

    if (cross(a, b) > EPS)
        return CCW::COUNTER_CLOCKWISE;
    if (cross(a, b) < -EPS)
        return CCW::CLOCKWISE;
    if (dot(a, b) < -EPS)
        return CCW::ONLINE_BACK;
    if (norm(a) < norm(b))
        return CCW::ONLINE_FRONT;

    return  CCW::ON_SEGMENT;
}


double norm(const Vector& a) {
    return a.x * a.x + a.y * a.y;
}

double abs_v(const Vector& a) {
    return std::sqrt(norm(a));
}

double dot(const Vector& a, const Vector& b) {
    return a.x * b.x + a.y * b.y;
}

double cross(const Vector& a, const Vector& b) {
    return a.x * b.y - a.y * b.x;
}

bool isOrthogonal(const Segment& s1, const Segment& s2) {
    return EQ_D(dot(s1.p2 - s1.p1, s2.p2 - s2.p1), 0.0);
}

bool isParallel(const Segment& s1, const Segment& s2) {
    return EQ_D(cross(s1.p2 - s1.p1, s2.p2 - s2.p1), 0.0);
}

bool intersect(const Segment& s1, const Segment& s2) {
    int i1 = static_cast<int>(ccw(s1.p1, s1.p2, s2.p1));
    int i2 = static_cast<int>(ccw(s1.p1, s1.p2, s2.p2));
    int i3 = static_cast<int>(ccw(s2.p1, s2.p2, s1.p1));
    int i4 = static_cast<int>(ccw(s2.p1, s2.p2, s1.p2));
    return ( (i1 * i2) <= 0 && (i2 * i3) <= 0);
}

Point project(const Segment& s, const Point& p) {
    Vector base = s.p2 - s.p1;
    double r = dot(p - s.p1, base) / norm(base);
    return s.p1 + base * r;
}

Point reflect(const Segment& s, const Point& p) {
    return p + (project(s, p) - p) * 2.0;
}

Point getCrossPoint(const Segment& s1, const Segment& s2) {
    Vector base = s2.p2 - s2.p1;
    double d1 = abs_v(cross(base, s1.p1 - s2.p1));
    double d2 = abs_v(cross(base, s1.p2 - s2.p1));
    double t = d1 / (d1 + d2);
    return s1.p1 + (s1.p2 - s1.p1) * t;
}

double getDistance(const Point& p1, const Point& p2) {
    return abs_v(p1 - p2);
}

double getDistanceL(const Line& l, const Point& p) {
    return abs_v(cross(l.p2 - l.p1, p - l.p1)) / abs_v(l.p2 - l.p1);
}

double getDistanceS(const Segment& s, const Point& p) {
    if (dot(s.p2 - s.p1, p - s.p1) < 0.0)
        return abs_v(p - s.p1);
    if (dot(s.p2 - s.p1, p - s.p2) < 0.0)
        return abs_v(p - s.p2);
    return getDistanceL(s, p);
}

double getDistance(const Segment& s1, const Segment& s2) {
    if (intersect(s1, s2))
        return 0.0;
    return std::min(std::min(getDistanceS(s1, s2.p1), getDistanceS(s1, s2.p2)),
                    std::min(getDistanceS(s2, s1.p1), getDistanceS(s2, s1.p2)));
}

bool intersect(const Circle& c, const Line& l) {
    double d = getDistanceL(l, c.c);
    return d <= c.r;
}

bool intersect(const Circle& c1, const Circle& c2) {
    double d = getDistance(c1.c, c2.c);
    return d <= (c1.r + c2.r);
}

bool getCrossPoints(const Circle &c, const Line& l, std::pair<Point, Point>& cross) {
    bool ret = intersect(c, l);
    if (ret) {
        Vector pr = project(l, c.c);
        Vector e = (l.p2 - l.p1) / abs_v(l.p2 - l.p1);
        double base = std::sqrt(c.r * c.r - norm(pr - c.c));
        cross.first = pr + e * base;
        cross.second = pr - e * base;
    }
    return ret;
}

static constexpr double arg(const Vector& p) {
    return std::atan2(p.y, p.x);
}

static Vector polar(double a, double r) {
    return Point{ std::cos(r) * a, std::sin(r) * a };
}

bool getCrossPoints(const Circle &c1, const Circle& c2, std::pair<Point, Point>& cross) {
    bool ret = intersect(c1, c2);
    if (ret) {
        double d = getDistance(c1.c, c2.c);
        double a = std::acos((c1.r * c1.r + d * d - c2.r * c2.r) / (2 * c1.r * d));
        double t = arg(c2.c - c1.c);
        cross.first = c1.c + polar(c1.r, t + a);
        cross.second = c1.c + polar(c1.r, t - a);
    }
    return ret;
}

PC contains(const Polygon& g, const Point& p) {
    int n = g.size();
    bool x = false;
    for (int i = 0; i < n; ++i) {
        Point a = g[i] - p;
        Point b = g[(i + 1) % n] - p;

        if (abs_v(cross(a, b)) < EPS && dot(a, b) < EPS)
            return PC::ON;
        if (a.y > b.y)
            std::swap(a, b);
        if (a.y < EPS && EPS < b.y && cross(a, b) > EPS)
            x = !x;
    }
    return (x ? PC::IN : PC::OUT);
}

Polygon andrewScan(const Polygon& g) {
     if (g.size() < 3)
        return g;

    std::sort(g.begin(), g.end());
    
    Polygon u, l;
    u.push_back(g[0]);
    u.push_back(g[1]);
    l.push_back(g[g.size() - 1]);
    l.push_back(g[g.size() - 2]);

    for (int i = 2; i < g.size(); ++i) {
        for (int n = u.size(); n >= 2 && ccw(u[n - 2], u[n - 1], g[i]) != CCW::CLOCKWISE; --n) {
            u.pop_back();
        }
        u.push_back(g[i]);
    }

    for (int i = g.size() - 3; i >= 0; --i) {
        for (int n = l.size(); n >= 2 && ccw(l[n - 2], l[n - 1], g[i]) != CCW::CLOCKWISE; --n) {
            l.pop_back();
        }
        l.push_back(g[i]);
    }

    std::reverse(l.begin(), l.end());
    for (int i = u.size() - 2; i >= 1; --i)
        l.push_back(u[i]);

    return l;
}
