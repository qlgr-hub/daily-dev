#include <fenv.h>
#include <iostream>
#include <vector>

void dfs(std::vector<std::vector<char>>& field, int x, int y) {
    field[x][y] = '.';

    // for (int dx = -1; dx <= 1; ++dx) {
    //     for (int dy = -1; dy <= 1; ++dy) {
    //         int nx = x + dx;
    //         int ny = y + dy;
    //         if (nx >= 0 && nx < field.size() && ny >= 0 && ny < field[nx].size() && field[nx][ny] == 'W') {
    //             dfs(field, nx, ny);
    //         }
    //     }
    // }

    auto do_ = [&](int nx, int ny) {
        if (nx >= 0 && nx < field.size() && ny >= 0 && ny < field[nx].size() && field[nx][ny] == 'W') {
            dfs(field, nx, ny);
        }
    };

    do_(x - 1, y - 1);
    do_(x - 1, y);
    do_(x - 1, y + 1);
    do_(x, y - 1);
    do_(x, y + 1);
    do_(x + 1, y - 1);
    do_(x + 1, y);
    do_(x + 1, y + 1);
}

int main() {
    std::vector<std::vector<char>> field{
        {'W', '.', '.', '.', '.', '.', '.', '.', '.', 'W', 'W', '.'},
        {'.', 'W', 'W', 'W', '.', '.', '.', '.', '.', 'W', 'W', 'W'},
        {'.', '.', '.', '.', 'W', 'W', '.', '.', '.', 'W', 'W', '.'},
        {'.', '.', '.', '.', '.', '.', '.', '.', '.', 'W', 'W', '.'},
        {'.', '.', '.', '.', '.', '.', '.', '.', '.', 'W', '.', '.'},
        {'.', '.', 'W', '.', '.', '.', '.', '.', '.', 'W', '.', '.'},
        {'.', 'W', '.', 'W', '.', '.', '.', '.', '.', 'W', 'W', '.'},
        {'W', '.', 'W', '.', 'W', '.', '.', '.', '.', '.', 'W', '.'},
        {'.', 'W', '.', 'W', '.', '.', '.', '.', '.', '.', 'W', '.'},
        {'.', '.', 'W', '.', '.', '.', '.', '.', '.', '.', 'W', '.'}
    };

    int res = 0;
    for (int i = 0; i < field.size(); ++i) {
        for (int j = 0; j < field[i].size(); ++j) {
            if (field[i][j] == 'W') {
                dfs(field, i, j);
                res++;
            }
        }
    }

    std::cout << res << std::endl;
    return 0;
}