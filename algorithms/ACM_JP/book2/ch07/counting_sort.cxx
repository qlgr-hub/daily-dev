#include <cstdint>
#include <iostream>
#include <array>
#include <vector>

static const int VMAX = 10000;

int main() {
    int n = 0;
    std::cin >> n;

    std::array<int, VMAX + 1> C{ 0 };

    std::vector<uint16_t> A(n + 1);
    for (int i = 1; i <= n; ++i) {
        std::cin >> A[i];
        ++C[A[i]];
    }

    for (int i = 1; i <= VMAX; ++i) {
        C[i] = C[i] + C[i - 1];
    }

    std::vector<uint16_t> B(n + 1);
    for (int i = 1; i <= n; ++i) {
        B[C[A[i]]] = A[i];
        --C[A[i]];
    }

    for (int i = 1; i <= n; ++i) {
        std::cout << B[i] << ' ';
    }
    std::cout << '\n';

    return 0;
}

