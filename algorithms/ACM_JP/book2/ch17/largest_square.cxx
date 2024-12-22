#include <array>
#include <iostream>

static constexpr int MAX = 1400;

static std::array<std::array<int , MAX>, MAX> dp;
static std::array<std::array<int , MAX>, MAX> G;

int getLargestSquare(int H, int W) {
    int maxWidth = 0;

    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            dp[i][j] = (G[i][j] + 1) % 2;
            maxWidth |= dp[i][j];
        }
    }

    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            if (G[i][j]) {
                dp[i][j] = 0;
            }
            else {
                dp[i][j] = std::min(dp[i - 1][j - 1], std::min(dp[i - 1][j], dp[i][j - 1])) + 1;
                maxWidth = std::max(maxWidth, dp[i][j]);
            }
        }
    }

    return maxWidth * maxWidth;
}

int main() {
    int H, W;
    std::cin >> H >> W;

    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            std::cin >> G[i][j];
        }
    }

    std::cout << getLargestSquare(H, W) << '\n';
    return 0;
}
