#ifndef _GRAPH_A_M_H_
#define _GRAPH_A_M_H_

#include <functional>
#include <iostream>
#include <array>
#include <limits>
#include <queue>
#include <stack>

template <int N = 100>
class Graph_M {
    static constexpr int _INFTY = (1 << 21);
    static const int INVALID_IDX = -1;
    enum class State { WHITE = 0, GRAY = 1, BLACK = 2 };

    std::array<std::array<bool, N>, N> M;
    std::array<State, N> color;

    std::array<int, N> nt;
    std::array<int, N> d;
    std::array<int, N> f;
    int tt;
    int n;

    std::array<int, N> dd;

private:
    void dfs_visit_r(int u) {
        color[u] = State::GRAY;
        d[u] = ++tt;
        for (int v = 0; v < n; ++v) {
            if (M[u][v] == 0)
                continue;
            if (color[v] == State::WHITE)
                dfs_visit_r(v);
        }
        color[u] = State::BLACK;
        f[u] = ++tt;
    }

    int next(int u) {
        for (int v = nt[u]; v < n; ++v) {
            nt[u] = v + 1;
            if (M[u][v])
                return v;
        }
        return INVALID_IDX;
    }

    void dfs_visit_s(int r) {
        nt.fill(0);

        std::stack<int> S;
        S.push(r);
        color[r] = State::GRAY;
        d[r] = ++tt;

        while (!S.empty()) {
            int u = S.top();
            int v = next(u);

            if (v != INVALID_IDX) {
                if (color[v] == State::WHITE) {
                    S.push(v);
                    color[v] = State::GRAY;
                    d[v] = ++tt;
                }
            }
            else {
                S.pop();
                color[u] = State::BLACK;
                f[u] = ++tt;
            }
        }
    }

    void _bfs(int s) {
        dd.fill(_INFTY);
        std::queue<int> q;

        dd[s] = 0;
        q.push(s);
        
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            for (int v = 0; v < n; ++v) {
                if (!M[u][v])
                    continue;
                if (dd[v] != _INFTY)
                    continue;
                dd[v] = dd[u] + 1;
                q.push(v);
            }
        }

        for (int i = 0; i < n; ++i)
            std::cout << i + 1 << ' ' << ((dd[i] == _INFTY) ? (-1) : dd[i]) << '\n';
    }
 
public:
    Graph_M() : M{ 0 }, color{ State::WHITE }, nt{ 0 }, d{ 0 }, f{ 0 }, tt{ 0 }, n{ 0 } {
        dd.fill(_INFTY);
    }

    void reconstruct() {
        n = 0;
        std::cin >> n;

        M.fill({false});
        for (int i = 0; i < n; ++i) {
            int u, k;
            std::cin >> u >> k;
            --u;
            for (int j = 0; j < k; ++j) {
                int v;
                std::cin >> v;
                --v;
                M[u][v] = true;
            }
        }
    } 

    void dfs(bool recursive = true) {
        color.fill(State::WHITE);
        d.fill(0);
        f.fill(0);
        tt = 0;

        std::function<void(Graph_M<N>*, int)> dfs_visit = recursive ? &Graph_M<N>::dfs_visit_r : &Graph_M<N>::dfs_visit_s;
        for (int u = 0; u < n; ++u)
            if (color[u] == State::WHITE)
                dfs_visit(this, u);

        for (int u = 0; u < n; ++u)
            std::cout << u + 1 << ' ' << d[u] << ' ' << f[u] << '\n';
    }

    void bfs() {
        _bfs(0);
    }
};

#endif /* !_GRAPH_A_M_H_ */
