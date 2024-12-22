#include <iostream>
#include <array>

int n = 0;
std::array<int, 50> A;

bool solve(int i, int m) {
    if (m == 0)
        return true;

    if (i >= n)
        return false;

    bool res = solve(i + 1, m) || solve(i + 1, m - A[i]);
    return res;
}

int main() {
    std::cin >> n;
    for (int i = 0; i < n; ++i)
        std::cin >> A[i];

    int q = 0;
    std::cin >> q;
    for (int i = 0; i < q; ++i) {
        int m = 0;
        std::cin >> m;
        if (solve(0, m))
            std::cout << "yes\n";
        else
            std::cout << "no\n";
    }

    return 0;
}
