#include <array>
#include <iostream>
#include <algorithm>

static const int N = 101;

int main() {
    std::array<int, N> p;
    std::array<std::array<int, N>, N> m{ 0 };

    int n;
    std::cin >> n;
    for (int i = 1; i <= n; ++i)
        std::cin >> p[i - 1] >> p[i];

    for (int l = 2; l <= n; ++l) {
        for (int i = 1; i <= n - l + 1; ++i) {
            int j = i + l - 1;
            m[i][j] = (1 << 21);
            for (int k = i; k <= j - 1; ++k) {
                m[i][j] = std::min(m[i][j], m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j]);
            }
        }
    }

    std::cout << m[1][n] << '\n';
    return 0;
}