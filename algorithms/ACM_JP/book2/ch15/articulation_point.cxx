#include <array>
#include <iostream>
#include <set>
#include <vector>

static constexpr int MAX = 100000;

static std::array<std::vector<int>, MAX> G;
static std::array<bool, MAX> visited;

static std::array<int, MAX> prenum;
static std::array<int, MAX> parent;
static std::array<int, MAX> lowest;
static int timer = 0;

void dfs(int cur, int pre) {
    prenum[cur] = lowest[cur] = timer;
    ++timer;

    visited[cur] = true;

    for (int i = 0; i < G[cur].size(); ++i) {
        int next = G[cur][i];
        if (!visited[next]) {
            parent[next] = cur;

            dfs(next, cur);

            lowest[cur] = std::min(lowest[cur], lowest[next]);
        }
        else if (next != pre) {
            lowest[cur] = std::min(lowest[cur], prenum[next]);
        }
    }
}

void art_points(int N, std::set<int>& ap) {
    visited.fill(false);
    timer = 1;

    dfs(0, -1);

    int np = 0;
    for (int i = 1; i < N; ++i) {
        int p = parent[i];
        if (p == 0)
            ++np;
        else if (prenum[p] <= lowest[i])
            ap.insert(p);
    }

    if (np > 1)
        ap.insert(0);
}

int main() {
    int N, M;
    std::cin >> N >> M;

    for (int i = 0; i < M; ++i) {
        int s, t;
        std::cin >> s >> t;

        G[s].push_back(t);
        G[t].push_back(s);
    }

    std::set<int> ap;
    art_points(N, ap);

    for (auto it = ap.begin(); it != ap.end(); ++it)
        std::cout << *it << '\n';

    return 0;
}
