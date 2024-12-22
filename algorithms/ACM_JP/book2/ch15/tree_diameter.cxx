#include <array>
#include <iostream>
#include <queue>
#include <vector>

static constexpr int MAX = 100000;
static constexpr int INFTY = (1 << 30);

struct Edge {
    int t;
    int w;

    Edge() : t{ 0 }, w{ 0 } {
    }

    Edge(int _t, int _w) : t{ _t }, w{ _w } {
    }
};

static std::array<std::vector<Edge>, MAX> G;
static std::array<int, MAX> vis;
static std::array<int, MAX> d;

void bfs(int s) {
    d.fill(INFTY);

    std::queue<int> Q;
    Q.push(s);
    d[s] = 0;

    while (!Q.empty()) {
        int u = Q.front(); Q.pop();

        for (int i = 0; i < G[u].size(); ++i) {
            Edge e = G[u][i];
            if (d[e.t] == INFTY) {
                d[e.t] = d[u] + e.w;
                Q.push(e.t);
            }
        }
    }
}

void solve(int N) {
    bfs(0);

    int maxv = 0;
    int tgt = 0;
    for (int i = 0; i < N; ++i) {
        if (d[i] == INFTY)
            continue;

        if (maxv < d[i]) {
            maxv = d[i];
            tgt = i;
        }
    }

    bfs(tgt);

    for (int i = 0; i < N; ++i) {
        if (d[i] == INFTY)
            continue;

        maxv = std::max(maxv, d[i]);
    }

    std::cout << maxv << '\n';
}

int main() {
    int N = 0;
    std::cin >> N;

    for (int i = 0; i < N - 1; ++i) {
        int s, t, w;
        std::cin >> s >> t >> w;
        
        G[s].push_back(Edge{t, w});
        G[t].push_back(Edge{s, w});
    }

    solve(N);

    return 0;
}
