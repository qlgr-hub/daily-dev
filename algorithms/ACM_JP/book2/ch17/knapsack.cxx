#include <algorithm>
#include <array>
#include <iostream>
#include <vector>

static constexpr int NMAX = 100;
static constexpr int WMAX = 10000;

struct Item {
    int value;
    int weight;
};

static std::array<Item, NMAX + 1> items;
static std::array<std::array<int,  WMAX + 1>, NMAX + 1> C;
static std::array<std::array<bool, WMAX + 1>, NMAX + 1> G;

int compute(int N, int W, std::vector<int>& selection) {
    for (int w = 0; w < W; ++w) {
        C[0][w] = 0;
        G[0][w] = true;
    }

    for (int i = 1; i <= N; ++i)
        C[i][0] = 0;

    for (int i = 1; i <= N; ++i) {
        for (int w = 1; w <= W; ++w) {
            C[i][w] = C[i - 1][w];
            G[i][w] = false;

            if (items[i].weight > w)
                continue;
            if (items[i].value + C[i - 1][w - items[i].weight] > C[i - 1][w]) {
                C[i][w] = items[i].value + C[i - 1][w - items[i].weight];
                G[i][w] = true;
            }
        }
    }

    selection.clear();
    for (int i = N, w = W; i >= 1; i--) {
        if (G[i][w]) {
            selection.push_back(i);
            w -= items[i].weight;
        }
    }
    std::reverse(selection.begin(), selection.end());
    
    return C[N][W];
}

int main() {
    int N, W;
    std::cin >> N >> W;

    for (int i = 1; i <= N; i++)
        std::cin >> items[i].value >> items[i].weight;

    std::vector<int> selection;
    int maxValue = compute(N, W, selection);

    std::cout << maxValue << '\n';

    for (auto i : selection) {
        std::cout << items[i].value << ' ';
    }
    std::cout << '\n';

    return 0;
}
