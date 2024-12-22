#include <iostream>
#include <array>
#include <queue>
#include <utility>
#include <vector>


template <int N = 100>
class Graph {
    static constexpr int INFTY = (1 << 21);

    enum class S{ WHITE = 0, GRAY = 1, BLACK = 2};

    std::array<std::vector<std::pair<int, int>>, N> adj;
    int n;

    std::array<int, N> d;
    std::array<S, N> c;


public:
    Graph() : n{ 0 }, d{ 0 }, c{ S::WHITE } {
    }

    void reconstruct() {
        std::cin >> n;

        for (int i = 0; i < n; ++i) {
            int u, k;
            std::cin >> u >> k;
            for (int j = 0; j < k; ++j) {
                int v, c;
                std::cin >> v >> c;
                adj[u].push_back(std::make_pair(v, c));
            }
        }
    }

    void dijkstra() {
        d.fill(INFTY);
        c.fill(S::WHITE);

        std::priority_queue<std::pair<int, int>> PQ;
        d[0] = 0;
        c[0] = S::GRAY;
        PQ.push(std::make_pair(0, 0));

        while (!PQ.empty()) {
            std::pair<int, int> f = PQ.top();
            PQ.pop();

            int u = f.second;
            c[u] = S::BLACK;
            
            if ( d[u] < f.first + (-1) )
                continue;

            for (int j = 0; j < adj[u].size(); ++j) {
                int v = adj[u][j].first;
                if (c[v] == S::BLACK)
                    continue;
                if (d[v] > d[u] + adj[u][j].second) {
                    d[v] = d[u] + adj[u][j].second;
                    PQ.push(std::make_pair(d[v] * (-1), v));
                    c[v] = S::GRAY;
                }
            }
        }

        for (int i = 0; i < n; ++i)
            std::cout << i << ' ' << ((d[i] == INFTY) ? -1 : d[i]) << '\n';
    }
};

int main() {
    Graph<100> G;
    G.reconstruct();
    G.dijkstra();
    return 0;
}
