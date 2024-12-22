#include <iostream>
#include <array>

static const int N = 100;

int main() {
    std::array<std::array<bool, N>, N> M{ 0 };

    int n;
    std::cin >> n;

    for (int i = 0; i < n; ++i) {
        int u, k;
        std::cin >> u >> k;
        --u; // 转换为 0 起点

        for (int j = 0; j < k; ++j) {
            int v;
            std::cin >> v;
            --v; // 转换为 0 起点
            M[u][v] = true;
        }
    }

    std::cout << '\n';
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << M[i][j] << ' ';
        }
        std::cout << '\n';
    }

    return 0;
}