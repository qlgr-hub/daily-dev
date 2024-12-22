#include <iostream>
#include <array>


template <int N = 100>
class Graph {
    static constexpr int INFTY = (1 << 21);

    enum class S{ WHITE = 0, GRAY = 1, BLACK = 2};

    std::array<std::array<int, N>, N> M;
    int n;

    std::array<int, N> d;
    std::array<S, N> c;


public:
    Graph() : M{ 0 }, n{ 0 }, d{ 0 }, c{ S::WHITE } {
    }

    void reconstruct() {
        std::array<int, N> a;
        a.fill(INFTY);
        M.fill(a);
        std::cin >> n;

        for (int i = 0; i < n; ++i) {
            int u, k;
            std::cin >> u >> k;
            for (int j = 0; j < k; ++j) {
                int v, c;
                std::cin >> v >> c;
                M[u][v] = c;
            }
        }
    }

    void dijkstra() {
        d.fill(INFTY);
        c.fill(S::WHITE);

        d[0] = 0;
        c[0] = S::GRAY;

        while (true) {
            int minv = INFTY;
            int u = -1;

            for (int i = 0; i < n; ++i) {
                if (minv > d[i] && c[i] != S::BLACK) {
                    u = i;
                    minv = d[i];
                }
            }

            if (u == -1)
                break;

            c[u] = S::BLACK;
            for (int v = 0; v < n; ++v) {
                if (c[v] != S::BLACK && M[u][v] != INFTY) {
                    if ( d[v] > (d[u] + M[u][v]) ) {
                        d[v] = d[u] + M[u][v];
                        c[v] = S::GRAY;
                    }
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
