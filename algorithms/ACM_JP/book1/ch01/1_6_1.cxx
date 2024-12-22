#include <cstddef>
#include <iostream>
#include <vector>

int solve(const std::vector<int>& a) {
    int ans = 0;

    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = i + 1; j < a.size(); ++j) {
            for (size_t k = j + 1; k < a.size(); ++k) {
                int len = a[i] + a[j] + a[k];
                int ma = std::max(a[i], std::max(a[j], a[k]));
                int rest = len - ma;

                if (ma < rest) {
                    ans = std::max(ans, len);
                }
            }
        }
    }

    return ans;
}

int main() {
    int n;
    std::cin >> n;

    std::vector<int> arr(n);
    for (size_t i = 0; i < arr.size(); ++i) {
        std::cin >> arr[i];
    }

    int ret = solve(arr);

    std::cout << ret << std::endl;
    return 0;
}