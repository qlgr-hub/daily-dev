#include <iostream>
#include <algorithm>
#include <vector>

int main() {
    int n;
    std::cin >> n;

    std::vector<int> R(n);
    for (int i = 0; i < n; ++i)
        std::cin >> R[i];

    int maxv = -2000000000;
    int minv = (R.size() > 0) ? R[0] : 0;

    for (int i = 1; i < n; ++i) {
        maxv = std::max(maxv, R[i] - minv);
        minv = std::min(minv, R[i]);
    }

    std::cout << maxv << '\n';
    return 0;
}