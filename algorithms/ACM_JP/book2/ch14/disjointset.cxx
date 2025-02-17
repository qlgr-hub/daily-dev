#include <iostream>
#include <vector>

class DisjointSet {
    std::vector<int> rank;
    std::vector<int> p;

public:
    DisjointSet() {
    }

    DisjointSet(int size) : rank(size, 0), p(size, 0) {
        for (int i = 0; i < size; ++i)
            makeSet(i);
    }

    void makeSet(int x) {
        p[x] = x;
        rank[x] = 0;
    }

    int findSet(int x) {
        if (x != p[x])
            p[x] = findSet(p[x]);
        return p[x];
    }

    void link(int x, int y) {
        if (rank[x] > rank[y])
            p[y] = x;
        else {
            p[x] = y;
            if (rank[x] == rank[y])
                ++rank[y];
        }
    } 

    bool same(int x, int y) {
        return findSet(x) == findSet(y);
    }

    void unite(int x, int y) {
        link(findSet(x), findSet(y));
    }
};

int main() {
    int n, q;
    std::cin >> n >> q;
    DisjointSet ds(n);

    for (int i = 0; i < q; ++i) {
        int t, a, b;
        std::cin >> t >> a >> b;
        if (t == 0)
            ds.unite(a, b);
        else if (t == 1) {
            std::cout << (ds.same(a, b) ? 1 : 0) << '\n';
        }
    }

    return 0;
}