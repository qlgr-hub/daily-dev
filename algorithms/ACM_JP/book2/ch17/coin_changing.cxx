#include <array>
#include <iostream>

static constexpr int MMAX = 20;
static constexpr int NMAX = 50000;
static constexpr int INFTY = (1 << 29);

static std::array<int, MMAX + 1> C;
static std::array<int, NMAX + 1> T;

int main() {
    int n, m;
    std::cin >> n >> m;

    for (int i = 1; i <= m; ++i) {
        std::cin >> C[i];
    }

    T.fill(INFTY);
    T[0] = 0;

    for (int i = 1; i <= m; ++i) {
        for (int j = 0; j + C[i] <= n; ++j) {
            T[j + C[i]] = std::min(T[j + C[i]], T[j] + 1);
        }
    }

    std::cout << T[n] << '\n';
    return 0;
}