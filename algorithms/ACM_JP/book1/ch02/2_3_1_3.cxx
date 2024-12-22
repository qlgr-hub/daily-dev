#include <iostream>
#include <utility>
#include <vector>

int solve(const std::vector<std::pair<int, int>>& field, int w) {
    static std::vector<std::vector<int>> dp(field.size() + 1, std::vector<int>(w + 1, 0));

    for (int i = field.size() - 1; i >= 0; --i) {
        for (int j = 0; j <= w; ++j) {
            if (j < field[i].first) {
                dp[i][j] = dp[i + 1][j];
            }
            else {
                dp[i][j] = std::max(dp[i + 1][j], dp[i + 1][j - field[i].first] + field[i].second);
            }
        }
    }

    return dp[0][w];
}

int main() {
    std::vector<std::pair<int, int>> field {
        {2, 3},
        {1, 2},
        {3, 4},
        {2, 2}
    };

    int W = 5;
    std::cout << solve(field, W) << std::endl;
    return 0;
}