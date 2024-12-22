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
static std::array<int, MAX> indeg;


void bfs(int s, std::list<int>& out) {
    std::queue<int> q;
    q.push(s);
    V[s] = true;

    while (!q.empty()) {
        int u = q.front(); q.pop();
        out.push_back(u);

        for (int i = 0; i < G[u].size(); ++i) {
            int v = G[u][i];
            --indeg[v];

            if (indeg[v] == 0 && !V[v]) {
                V[v] = true;
                q.push(v);
            }
        }
    }
}

void topoSort_bfs(int N, std::list<int>& out) {
    V.fill(false);
    indeg.fill(0);
    
    for (int u = 0; u < N; ++u) {
        for (int i = 0; i < G[u].size(); ++i) {
            int v = G[u][i];
            ++indeg[v];
        }
    }

    for (int u = 0; u < N; ++u)
        if (indeg[u] == 0 && !V[u])
            bfs(u, out);
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
    GTS::topoSort_bfs(N, out);

    for (auto it = out.begin(); it != out.end(); ++it)
        std::cout << *it << '\n';

    return 0;
}