#include <iostream>
#include <array>

static const int MAX = 1000;
static std::array<int, MAX> A;

int partition(int p, int r) {
    int x = A[r];
    int i = p - 1;
    for (int j = p; j < r; ++j) {
        if (A[j] <= x) {
            std::swap(A[j], A[++i]);
        }
    }
    std::swap(A[r], A[++i]);
    return i;
}

int main() {
    int n = 0;
    std::cin >> n;

    for (int i = 0; i < n; ++i)
        std::cin >> A[i];

    int q = partition(0, n - 1);

    for (int i = 0; i < n; ++i) {
        if (i == q)
            std::cout << '[';
        std::cout << A[i];
        if (i == q)
            std::cout << ']';
        std::cout << ' ';
    }
    std::cout << '\n';

    return 0;
}
