#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

std::pair<int, int> solve(const std::vector<int>& x, int L) {

    int minT = 0;
    int maxT = 0;
    for (const auto& x_i : x) {
        minT =  std::max(minT, std::min(x_i, L - x_i));
        maxT =  std::max(maxT, std::max(x_i, L - x_i));
    }

    return std::pair<int, int>{ minT, maxT };
}

int main() {
    int L, n;
    std::cin >> L >> n;

    std::vector<int> arr(n);
    for (size_t i = 0; i < arr.size(); ++i) {
        std::cin >> arr[i];
    }

    auto ret = solve(arr, L);

    std::cout << "min: " << ret.first << ", max: " << ret.second << std::endl;
    return 0;
}