#include <iostream>
#include <array>
#include <string_view>

static const int N = 1001;

int lcs(std::string X, std::string Y) {
    static std::array<std::array<int, N>, N> DP{ 0 };

    int maxl = 0;
    int m = X.size();
    int n = Y.size();

    X = ' ' + X;
    Y = ' ' + Y;

    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (X[i] == Y[j])
                DP[i][j] = DP[i - 1][j - 1] + 1;
            else
                DP[i][j] = std::max(DP[i - 1][j], DP[i][j - 1]);
            maxl = std::max(maxl, DP[i][j]);
        }
    }

    return maxl;
}

int main() {
    std::string s1;
    std::string s2;

    int n = 0;
    std::cin >> n;
    for (int i = 0; i < n; ++i) {
        std::cin >> s1 >> s2;
        std::cout << lcs(s1, s2) << '\n';
    }

    return 0;
}