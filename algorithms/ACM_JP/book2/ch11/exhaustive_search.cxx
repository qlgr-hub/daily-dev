#include <iostream>
#include <array>

static const int N = 21;
static const int MAX_M = 2001;

static int n = 0;
static std::array<int, N> A;
static std::array<std::array<bool, MAX_M>, N> DP;

bool solve(int i, int m) {
    if (DP[i][m])
        return DP[i][m];

    if (m == 0)
        DP[i][m] = true;
    else if (i >= n)
        DP[i][m] = false;
    else if (solve(i + 1, m))
        DP[i][m] = true;
    else if (solve(i + 1, m - A[i]))
        DP[i][m] = true;
    else
        DP[i][m] = false;

    return DP[i][m];
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
