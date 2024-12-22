#include <iostream>
#include <algorithm>
#include <vector>

int main() {
    int n = 0;
    std::cin >> n;

    std::vector<int> V(n);
    for (auto& v : V) {
        std::cin >> v;
    }

    // std::sort(V.begin(), V.end());
    std::stable_sort(V.begin(), V.end());

    for (auto v : V) {
        std::cout << v << ' ';
    }
    std::cout << '\n';

    return 0;
}