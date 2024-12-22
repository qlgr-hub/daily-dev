#include <algorithm>
#include <iostream>
#include <stack>
#include <utility>
#include <vector>

int main() {
    std::stack<int> S1;
    std::stack<std::pair<int, int>> S2;

    char ch;
    int sum = 0;
    for (int i = 0; std::cin >> ch; ++i) {
        if (ch == '\\') {
            S1.push(i);
        }
        else if (ch == '/' && !S1.empty()) {
            int j = S1.top(); S1.pop();
            int a = i - j;
            sum += a;

            while (!S2.empty() && S2.top().first > j) {
                a += S2.top().second; S2.pop();
            }
            S2.push(std::make_pair(j, a));
        }
    }

    std::vector<int> ans;
    while (!S2.empty()) {
        ans.push_back(S2.top().second);
        S2.pop();
    }
    std::reverse(ans.begin(), ans.end());

    std::cout << sum << '\n';
    std::cout << ans.size() << ' ';
    for (auto v : ans) {
        std::cout << v << ' ';
    }
    std::cout << '\n';
    return 0;
}