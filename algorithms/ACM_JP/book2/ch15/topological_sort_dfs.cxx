#include <array>
#include <iostream>
#include <list>
#include <queue>
#include <vector>

namespace GTS {
static constexpr int MAX = 100000;
static constexpr int INFTY = (1<<29);

static std::array<std::vector<int>, MAX> G;
static std::array<bool, MAX> V;

void dfs(int u, std::list<int>& out) {
    V[u] = true;

    for (int i = 0; i < G[u].size(); ++i) {
        int v = G[u][i];
        if (!V[v])
            dfs(v, out);
    }

    out.push_front(u);
}

void topoSort_dfs(int N, std::list<int>& out) {
    V.fill(false);
    for (int u = 0; u < N; ++u)
        if (!V[u])
            dfs(u, out);
}
}

int main() {
    int N, M;
    std::cin >> N >> M;

    for (int i = 0; i < M; ++i) {
        int s, t;
        std::cin >> s >> t;
        GTS::G[s].push_back(t);
    }

    std::list<int> out;
    GTS::topoSort_dfs(N, out);

    for (auto it = out.begin(); it != out.end(); ++it)
        std::cout << *it << '\n';

    return 0;
}