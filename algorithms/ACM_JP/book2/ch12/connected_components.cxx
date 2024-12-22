#include <array>
#include <exception>
#include <iostream>
#include <stack>
#include <vector>

static const int NIL = -1;
static const int MAX = 100000;

static std::array<std::vector<int>, MAX> G;
static std::array<int, MAX> color;

void dfs(int r, int c) {
    std::stack<int> S;

    S.push(r);
    color[r] = c;

    while (!S.empty()) {
        int u = S.top();
        S.pop();
        for (int i = 0; i < G[u].size(); ++i) {
            int v = G[u][i];
            if (color[v] == NIL) {
                color[v] = c;
                S.push(v);
            }
        }
    }
}

void assignColor(int n) {
    color.fill(NIL);
    int id = 1;
    for (int i = 0; i < n; ++i)
        if (color[i] == NIL)
            dfs(i, id++);
}

int main() {
    int n, m;
    std::cin >> n >> m;

    for (int i = 0; i < m; ++i) {
        int s, t;
        std::cin >> s >> t;

        G[s].push_back(t);
        G[t].push_back(s);
    }

    assignColor(n);

    int q;
    std::cin >> q;
    for (int i = 0; i < q; ++i) {
        int s, t;
        std::cin >> s >> t;

        if (color[s] != NIL && color[s] == color[t]) {
            std::cout << "yes" << '\n';
        }
        else {
            std::cout << "no" << '\n';
        }
    }

    return 0;
}
