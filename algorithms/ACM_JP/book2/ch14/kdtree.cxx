#include <iostream>
#include <algorithm>
#include <vector>
#include <array>

struct Point {
    int id;
    int x;
    int y;

    Point() : id{ 0 }, x{ 0 }, y{ 0 } {
    }

    Point(int id_, int x_, int y_) : id{ id_ }, x{ x_ }, y{ y_ } {
    }
};

static bool operator < (const Point& p1, const Point& p2) {
    return p1.id < p2.id;
}

static bool operator >= (const Point& p1, const Point& p2) {
    return !(p1 < p2);
}

static bool operator > (const Point& p1, const Point& p2) {
    return p1.id > p2.id;
}

static bool operator <= (const Point& p1, const Point& p2) {
    return !(p1 > p2);
}

static bool operator == (const Point& p1, const Point& p2) {
    return p1.id == p2.id;
}

static bool operator != (const Point& p1, const Point& p2) {
    return !(p1 == p2);
}

static std::ostream& operator<<(std::ostream& out, const Point& p) {
    out << p.id << '\n';
    return out;
}

static bool lessX(const Point& p1, const Point& p2) {
    return p1.x < p2.x;
}

static bool lessY(const Point& p1, const Point& p2) {
    return p1.y < p2.y;
}

class KDTree {
    static constexpr int NIL = -1;
    static constexpr int MAX = 1000000;

    struct Node {
        int location;
        int p;
        int l;
        int r;

        Node() : location{ 0 }, p{ NIL }, l{ NIL }, r{ NIL } {
        }
    };

    int n;
    std::array<Point, MAX> P;
    std::array<Node, MAX> T;
    int np;
    int root;

    int _makeKDTree(int l, int r, int depth) {
        if (l >= r)
            return NIL;
        int mid = (l + r) / 2; // l + (r - l) / 2;
        int t = np++;
        auto compFunc = (depth % 2 == 0) ? lessX : lessY;
        std::sort(P.begin() + l, P.begin() + r, compFunc);

        T[t].location = mid;
        T[t].l = _makeKDTree(l, mid, depth + 1);
        T[t].r = _makeKDTree(mid + 1, r, depth + 1);
        return t;
    }

    void printT(int r) {
        if (r == NIL)
            return;
        std::cout << T[r].location << '\n';
        printT(T[r].l);
        printT(T[r].r);
    }

public:
    KDTree() : n{ 0 }, np{ 0 }, root{ 0 } {
    }

    void make() {
        std::cin >> n;

        for (int i = 0; i < n; ++i) {
            int x, y;
            std::cin >> x >> y;

            P[i] = Point(i, x, y);
        }

        np = 0;
        root = _makeKDTree(0, n, 0);

        // printT(root);
    }

    void Dump() {
        int q;
        std::cin >> q;

        for (int i = 0; i < q; ++i) {
            int sx, tx, sy, ty;
            std::cin >> sx >> tx >> sy >> ty;

            std::vector<Point> ans;
            find(root, sx, tx, sy, ty, 0, ans);
            std::sort(ans.begin(), ans.end());
            for (const auto& p : ans)
                std::cout << p;
            std::cout << '\n';
        }
    }

    void find(int v, int sx, int tx, int sy, int ty, int depth, std::vector<Point>& ans) {
        int x = P[T[v].location].x;
        int y = P[T[v].location].y;
        if (sx <= x && x <= tx && sy <= y && y <= ty)
            ans.push_back(P[T[v].location]);

        if (depth % 2 == 0) {
            if (T[v].l != NIL && sx <= x)
                find(T[v].l, sx, tx, sy, ty, depth + 1, ans);

            if (T[v].r != NIL && x <= tx)
                find(T[v].r, sx, tx, sy, ty, depth + 1, ans);
        }
        else {
            if (T[v].l != NIL && sy <= y)
                find(T[v].l, sx, tx, sy, ty, depth + 1, ans);

            if (T[v].r != NIL && y <= ty)
                find(T[v].r, sx, tx, sy, ty, depth + 1, ans);
        }
    }
};

KDTree t;

int main() {
    t.make();
    t.Dump();
    return 0;
}