#include <array>
#include <iostream>
#include <utility>
#include <vector>
#include <queue>


const int INF = 100000000;

int bfs(std::vector<std::vector<char>>& maze, int sx, int sy, int gx, int gy) {
    static std::vector<std::vector<int>> d{
        maze.size(),
        std::vector<int>(maze[0].size(), INF)
    };

    static const std::array<int, 4> dx{ 1, 0, -1, 0 };
    static const std::array<int, 4> dy{ 0, 1, 0, -1 };

    std::queue<std::pair<int, int>> que;
    que.push(std::pair<int, int>(sx, sy));
    d[sx][sy] = 0;

    while (!que.empty()) {
        std::pair<int, int> p = que.front();
        que.pop();

        if (p.first == gx && p.second == gy)
            break;

        for (int i = 0; i < 4; ++i) {
            int nx = p.first + dx[i];
            int ny = p.second + dy[i];

            if (nx >= 0 && nx < maze.size() && ny >= 0 && ny < maze[0].size()
                && maze[nx][ny] != '#' && d[nx][ny] == INF) {
                que.push(std::pair<int, int>(nx, ny));
                d[nx][ny] = d[p.first][p.second] + 1;
            }
        }
    }

    return d[gx][gy];
}

int main() {
    std::vector<std::vector<char>> maze{
        {'#', 'S', '#', '#', '#', '#', '#', '#', '.', '#'},
        {'.', '.', '.', '.', '.', '.', '#', '.', '.', '#'},
        {'.', '#', '.', '#', '#', '.', '#', '#', '.', '#'},
        {'.', '#', '.', '.', '.', '.', '.', '.', '.', '.'},
        {'#', '#', '.', '#', '#', '.', '#', '#', '#', '#'},
        {'.', '.', '.', '.', '#', '.', '.', '.', '.', '#'},
        {'.', '#', '#', '#', '#', '#', '#', '#', '.', '#'},
        {'.', '.', '.', '.', '#', '.', '.', '.', '.', '.'},
        {'.', '#', '#', '#', '#', '.', '#', '#', '#', '.'},
        {'.', '.', '.', '.', '#', '.', '.', '.', 'G', '#'}
    };

    std::cout << bfs(maze, 0, 1, 9, 8) << std::endl;
    return 0;
}