#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>

int solve(const std::vector<int>& s, const std::vector<int>& t) {
    std::vector<std::pair<int, int>> itv(s.size());
    for (int i = 0; i < s.size(); ++i) {
        itv[i].first  = t[i];
        itv[i].second = s[i];
    }
    std::sort(itv.begin(), itv.end());

    int ans = 0;
    int t_temp = 0;
    for (int i = 0; i < itv.size(); ++i) {
        if (t_temp < itv[i].second) {
            ++ans;
            t_temp = itv[i].first;
        }
    }

    return ans;
}

int main() {
    std::vector<int> S{1, 2, 4, 6, 8};
    std::vector<int> T{3, 5, 7, 9, 10};

    std::cout << solve(S, T) << std::endl;
    return 0;
}