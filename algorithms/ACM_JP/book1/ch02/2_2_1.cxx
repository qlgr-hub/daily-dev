#include <iostream>
#include <vector>


int solve(const std::vector<int>& v, const std::vector<int>& c, int a) {
    int ans = 0;

    for (int i = v.size() - 1; i >= 0; --i) {
        int t = std::min(a / v[i], c[i]);
        a -= t * v[i];
        ans += t;
    }

    return ans;
}


int main() {
    const std::vector<int> V{1, 5, 10, 50, 100, 500};
    std::vector<int> C(V.size());
    for (auto& val : C) {
        std::cin >> val;
    }

    int A = 0;
    std::cin >> A;

    std::cout << solve(V, C, A) << std::endl;
    return 0;
}