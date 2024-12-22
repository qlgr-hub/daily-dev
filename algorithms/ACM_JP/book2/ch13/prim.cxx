#include <iostream>
#include <array>


template <int N = 100>
class Graph {
    static constexpr int INFTY = (1 << 21);

    enum class S{ WHITE = 0, GRAY = 1, BLACK = 2};

    std::array<std::array<int, N>, N> M;
    int n;

    std::array<int, N> d;
    std::array<int, N> p;
    std::array<S, N> c;


public:
    Graph() : M{ 0 }, n{ 0 }, d{ 0 }, p{ 0 }, c{ S::WHITE } {
    }

    void reconstruct() {
        std::cin >> n;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                int e;
                std::cin >> e;
                M[i][j] = (e == -1) ? INFTY : e;
            }
        }
    }

    int prim() {
        d.fill(INFTY);
        p.fill(-1);
        c.fill(S::WHITE);

        d[0] = 0;

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
                    if (d[v] > M[u][v]) {
                        d[v] = M[u][v];
                        p[v] = u;
                        c[v] = S::GRAY;
                    }
                }
            }
        }

        int sum = 0;
        for (int i = 0; i < n; ++i)
            if (p[i] != -1)
                sum += M[i][p[i]];

        return sum;
    }
};

int main() {
    Graph<100> G;
    G.reconstruct();
    std::cout << G.prim() << '\n';
    return 0;
}
