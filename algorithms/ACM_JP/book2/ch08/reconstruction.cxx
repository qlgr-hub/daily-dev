#include <array>
#include <iostream>
#include <algorithm>
#include <vector>

static const int MAX = 10000;

std::array<int, MAX> pre;
std::array<int, MAX> in;
std::vector<int> post;

void rec(int& pos, int l, int r) {
    if (l >= r)
        return;

    int root = pre[pos++];
    int m = std::distance(in.cbegin(), std::find(in.cbegin(), in.cend(), root));

    rec(pos, l, m);
    rec(pos, m + 1, r);
    post.push_back(root);
}

void solve(int n) {
    int pos = 0;
    rec(pos, 0, n);

    for (auto v : post) {
        std::cout << v << ' ';
    }
    std::cout << '\n';
}

int main() {
    int n = 0;
    std::cin >> n;

    for (int i = 0; i < n; ++i)
        std::cin >> pre[i];

    for (int i = 0; i < n; ++i)
        std::cin >> in[i];

    solve(n);

    return 0;
}
