#include <algorithm>
#include <iostream>
#include <vector>

int solve(std::vector<int>& x, int r) {
    std::sort(x.begin(), x.end());

    int i = 0;
    int ans = 0;
    while (i < x.size()) {
        ++ans;

        int s = x[i++];
        for (; i < x.size() && x[i] <= s + r; ++i);
        
        int p = x[i - 1];
        for (; i < x.size() && x[i] <= p + r; ++i);
    }

    return ans;
}

int main() {
    std::vector<int> X{1, 7, 15, 20, 30, 50};
    int R = 10;
    
    std::cout << solve(X, R) << std::endl;
    return 0;
}