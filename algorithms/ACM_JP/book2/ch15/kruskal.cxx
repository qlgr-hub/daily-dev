#include <algorithm>
#include <iostream>
#include <vector>

static constexpr int MAX = 10000;
static constexpr int INFTY = (1 << 29);

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

struct Edge {
    int source;
    int target;
    int cost;

    Edge(int s = 0, int t = 0, int c = 0) : source{ s }, target{ t }, cost{ c } {
    }
};

static bool operator < (const Edge& e1, const Edge& e2) {
    return e1.cost < e2.cost;
}

static bool operator >= (const Edge& e1, const Edge& e2) {
    return !(e1 < e2);
}

static bool operator > (const Edge& e1, const Edge& e2) {
    return e1.cost > e2.cost;
}

static bool operator <= (const Edge& e1, const Edge& e2) {
    return !(e1 > e2);
}

static bool operator == (const Edge& e1, const Edge& e2) {
    return e1.cost == e2.cost;
}

static bool operator != (const Edge& e1, const Edge& e2) {
    return !(e1 == e2);
}

int kruskal(int N, std::vector<Edge>& edges) {
    int totalCost = 0;
    std::sort(edges.begin(), edges.end());

    DisjointSet dset(N + 1);

    for (int i = 0; i < N; ++i)
        dset.makeSet(i);

    for (const auto& e : edges) {
        if (!dset.same(e.source, e.target)) {
            totalCost += e.cost;
            dset.unite(e.source, e.target);
        }
    }

    return totalCost;
}

int main() {
    int N, M;
    std::cin >> N >> M;

    std::vector<Edge> edges(M);
    for (int i = 0; i < M; ++i) {
        int s, t, c;
        std::cin >> s >> t >> c;

        edges[i] = Edge(s, t, c);
    }

    std::cout << kruskal(N, edges) << '\n';

    return 0;
}
