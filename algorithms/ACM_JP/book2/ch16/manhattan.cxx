#include <algorithm>
#include <array>
#include <iostream>
#include <iterator>
#include <set>
#include <vector>
#include "geometry_object.h"

enum class ST { BOTTOM = 0, LEFT = 1, RIGHT = 2, TOP = 3 };

struct EndPoint {
    Point p;
    int seg;
    ST st;

    EndPoint(const Point& _p = Point{}, int _seg = 0, ST _st = ST::BOTTOM)
        : p{ _p }, seg{ _seg }, st{ _st } {
    }

// friend operator overloading
    friend bool operator<(const EndPoint& p1, const EndPoint& p2) {
        return (EQ_D(p1.p.y, p2.p.y) ? (p1.st < p2.st) : (p1.p.y < p2.p.y));
    }

    friend bool operator>=(const EndPoint& p1, const EndPoint& p2) {
        return !(p1 < p2);
    }

    friend bool operator>(const EndPoint& p1, const EndPoint& p2) {
        return (EQ_D(p1.p.y, p2.p.y) ? (p1.st > p2.st) : (p1.p.y > p2.p.y));
    }

    friend bool operator<=(const EndPoint& p1, const EndPoint& p2) {
        return !(p1 > p2);
    }

    friend bool operator==(const EndPoint& p1, const EndPoint& p2) {
        return (EQ_D(p1.p.y, p2.p.y) && (p1.st == p2.st));
    }

    friend bool operator!=(const EndPoint& p1, const EndPoint& p2) {
        return !(p1 == p2);
    }
};

static constexpr int MAX = 2 * 100000;
static std::array<EndPoint, MAX> EP;

int manhattanIntersection(std::vector<Segment>& S) {
    int n = S.size();

    for (int i = 0, k = 0; i < n; ++i) {
        if (EQ_D(S[i].p1.y, S[i].p2.y)) {
            if (S[i].p1.x > S[i].p2.x)
                std::swap(S[i].p1, S[i].p2);
        }
        else if (S[i].p1.y > S[i].p2.y) {
            std::swap(S[i].p1, S[i].p2);
        }

        if (EQ_D(S[i].p1.y, S[i].p2.y)) {
            EP[k++] = EndPoint(S[i].p1, i, ST::LEFT);
            EP[k++] = EndPoint(S[i].p2, i, ST::RIGHT);
        }
        else {
            EP[k++] = EndPoint(S[i].p1, i, ST::BOTTOM);
            EP[k++] = EndPoint(S[i].p2, i, ST::TOP);
        }
    }
    
    std::sort(EP.begin(), EP.begin() + (2 * n));

    std::set<int> BT;
    BT.insert(1000000001);

    int cnt = 0;
    for (int i = 0; i < 2 * n; ++i) {
        if (EP[i].st == ST::TOP) {
            BT.erase(EP[i].p.x);
        }
        else if (EP[i].st == ST::BOTTOM) {
            BT.insert(EP[i].p.x);
        }
        else if (EP[i].st == ST::LEFT) {
            auto b = BT.lower_bound(S[EP[i].seg].p1.x);
            auto e = BT.upper_bound(S[EP[i].seg].p2.x);
            cnt += std::distance(b, e);
        }
    }

    return cnt;
}
