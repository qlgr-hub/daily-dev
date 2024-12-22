#ifndef _GEOMETRY_OBJECT_H_
#define _GEOMETRY_OBJECT_H_

#include <iostream>
#include <cmath>
#include <utility>
#include <vector>

constexpr double EPS = (1.e-10);

constexpr bool EQ_D(double a, double b) {
    return std::fabs(a - b) < EPS;
}


struct Point {
    double x;
    double y;

    Point(double _x = 0.0, double _y = 0.0) : x{ _x }, y{ _y } {
    }


// member operator overloading
    Point& operator+=(const Point& othr) {
        x += othr.x;
        y += othr.y;
        return *this;
    }

    Point& operator-=(const Point& othr) {
        x -= othr.x;
        y -= othr.y;
        return *this;
    }

    Point& operator*=(double k) {
        x *= k;
        y *= k;
        return *this;
    }

    Point& operator/=(double k) {
        x /= k;
        y /= k;
        return *this;
    }


// friend operator overloading
    friend bool operator<(const Point& p1, const Point& p2) {
        return (p1.x != p2.x) ? (p1.x < p2.x) : (p1.y < p2.y);
    }

    friend bool operator>=(const Point& p1, const Point& p2) {
        return !(p1 < p2);
    }

    friend bool operator>(const Point& p1, const Point& p2) {
        return (p1.x != p2.x) ? (p1.x > p2.x) : (p1.y > p2.y);
    }

    friend bool operator<=(const Point& p1, const Point& p2) {
        return !(p1 > p2);
    }

    friend bool operator==(const Point& p1, const Point& p2) {
        return EQ_D(p1.x, p2.x) && EQ_D(p1.y, p2.y);
    }

    friend bool operator!=(const Point& p1, const Point& p2) {
        return !(p1 == p2);
    }

    friend Point operator+(const Point& p1, const Point& p2) {
        Point ret(p1);
        ret += p2;
        return ret;
    }

    friend Point operator-(const Point& p1, const Point& p2) {
        Point ret(p1);
        ret -= p2;
        return ret;
    }

    friend Point operator*(double k, const Point& p) {
        Point ret(p);
        ret *= k;
        return ret;
    }

    friend Point operator/(double k, const Point& p) {
        Point ret(p);
        ret /= k;
        return ret;
    }

    friend Point operator*(const Point& p, double k) {
        Point ret(p);
        ret *= k;
        return ret;
    }

    friend Point operator/(const Point& p, double k) {
        Point ret(p);
        ret /= k;
        return ret;
    }
};


enum class CCW {
    ONLINE_FRONT = -2,
    CLOCKWISE = -1,
    ON_SEGMENT = 0,
    COUNTER_CLOCKWISE = 1,
    ONLINE_BACK = 2
};
std::ostream& operator<<(std::ostream& out, const CCW& c);

CCW ccw(Point p1, Point p2, Point p3);

using Vector = Point;
double norm(const Vector& a);
double abs_v(const Vector& a);
double dot(const Vector& a, const Vector& b);
double cross(const Vector& a, const Vector& b);


struct Segment {
    Point p1;
    Point p2;
};

bool isOrthogonal(const Segment& s1, const Segment& s2);
bool isParallel(const Segment& s1, const Segment& s2);
bool intersect(const Segment& s1, const Segment& s2);

Point project(const Segment& s, const Point& p);
Point reflect(const Segment& s, const Point& p);
Point getCrossPoint(const Segment& s1, const Segment& s2);

using Line = Segment;
double getDistance(const Point& p1, const Point& p2);
double getDistanceL(const Line& l, const Point& p);
double getDistanceS(const Segment& s, const Point& p);
double getDistance(const Segment& s1, const Segment& s2);

struct Circle {
    Point c;
    double r;

    Circle(const Point& _c = Point{}, double _r = 0.0) : c{ _c }, r{ _r } {
    }
};

bool intersect(const Circle& c, const Line& l);
bool intersect(const Circle& c1, const Circle& c2);

std::pair<Point, Point> getCrossPoints(const Circle &c, const Line& l);
std::pair<Point, Point> getCrossPoints(const Circle &c1, const Circle& c2);

using Polygon = std::vector<Point>;

enum class PC { OUT = 0, ON = 1, IN = 2 };

PC contains(const Polygon& g, const Point& p);

Polygon andrewScan(const Polygon& g);

#endif // !_GEOMETRY_OBJECT_H_
