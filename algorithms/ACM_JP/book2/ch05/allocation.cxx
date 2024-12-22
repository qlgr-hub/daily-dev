#include <array>
#include <cstdint>
#include <iostream>

const int MAX = 100000;
std::array<int64_t, MAX> T;

int check(int64_t P, int n, int k) {
    int i = 0;
    for (int j = 0; j < k; ++j) {
        int64_t s = 0;

        while (s + T[i] <= P) {
            s += T[i];
            ++i;
            if (i == n)
                return n;
        }
    }
    return i;
}

int64_t solve(int n, int k) {
    int64_t left = 0;
    int64_t right = MAX * 10000;
    while ((right - left) > 1) {
        int64_t mid = (left + right) / 2;
        int v = check(mid, n, k);
        if (v >= n)
            right = mid;
        else
            left = mid;
    }
    return right;
}

int main() {
    int n, k;
    std::cin >> n >> k;
    for (int i = 0; i < n; ++i)
        std::cin >> T[i];

    int64_t ans = solve(n, k);
    std::cout << ans << '\n';
    return 0;
}