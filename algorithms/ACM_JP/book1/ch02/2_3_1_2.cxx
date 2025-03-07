#include <iostream>
#include <utility>
#include <vector>

int solve(const std::vector<std::pair<int, int>>& field, int i, int j) {
    static std::vector<std::vector<int>> dp(field.size() + 1, std::vector<int>(j + 1, -1));

    if (dp[i][j] >= 0) {
        return dp[i][j];
    }

    int res = 0;
    if (i == field.size()) {
        res = 0;
    }
    else if (j < field[i].first) {
        res = solve(field, i + 1, j);
    }
    else {
        res = std::max(solve(field, i + 1, j), solve(field, i + 1, j - field[i].first) + field[i].second);
    }

    return (dp[i][j] = res);
}

int main() {
    std::vector<std::pair<int, int>> field {
        {2, 3},
        {1, 2},
        {3, 4},
        {2, 2}
    };

    int W = 5;
    std::cout << solve(field, 0, W) << std::endl;
    return 0;
}