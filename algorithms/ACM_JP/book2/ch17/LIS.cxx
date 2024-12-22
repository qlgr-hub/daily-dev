#include <algorithm>
#include <array>
#include <iostream>

static constexpr int MAX = 100000;

static std::array<int, MAX + 1> A;
static std::array<int, MAX + 1> L;

int lis(int n) {
    L[0] = A[0];
    int length = 1;

    for (int i = 1; i < n; ++i) {
        if (L[length - 1] < A[i])
            L[length++] = A[i];
        else
            *std::lower_bound(L.begin(), L.begin() + length, A[i]) = A[i];
    }

    return length;
}

int main() {
    int n = 0;
    std::cin >> n;

    for (int i = 0; i < n; ++i)
        std::cin >> A[i];

    std::cout << lis(n) << '\n';
    
    return 0;
}
