#include <array>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <vector>

static const constexpr int MAX = 100;
static const constexpr int64_t INFTY = (1LL << 32);

static std::array<std::array<int64_t, MAX>, MAX> d;

void floyd(int n) {
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            if (d[i][k] == INFTY)
                continue;
            for (int j = 0; j < n; ++j) {
                if (d[k][j] == INFTY)
                    continue;
                d[i][j] = std::min(d[i][j], d[i][k] + d[k][j]);
            }
        }
    }
}

int main() {
    int n, e;
    std::cin >> n >> e;

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            d[i][j] = ((i == j) ? 0 : INFTY);

    for (int i = 0; i < e; ++i) {
        int u, v, c;
        std::cin >> u >> v >> c;
        d[u][v] = c;
    }

    floyd(n);

    bool negative = false;
    for (int i = 0; i < n; ++i)
        if (d[i][i] < 0)
            negative = true;

    if (negative) {
        std::cout << "NEGATIVE CYCLE" << '\n';
    }
    else {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                std::cout << ((d[i][j] == INFTY) ? "INF" : std::to_string(d[i][j])) << ' ';
            }
            std::cout << '\n';
        }
    }

    return 0;
}
