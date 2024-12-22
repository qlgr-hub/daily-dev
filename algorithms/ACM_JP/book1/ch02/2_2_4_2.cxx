#include <iostream>
#include <vector>
#include <algorithm>

int solve(std::vector<int>& l) {
    int ans = 0;

    for (int i = l.size(); i > 1; --i) {
        int mii1 = 0;
        int mii2 = 1;
        if (l[mii1] > l[mii2])
            std::swap(mii1, mii2);

        for (int j = 2; j < i; ++j) {
            if (l[j] < l[mii1]) {
                mii2 = mii1;
                mii1 = j;
            }
            else if (l[j] < l[mii2]) {
                mii1 = mii2;
                mii2 = j;
            }
        }

        int t = l[mii1] + l[mii2];
        ans += t;

        if (mii1 == i - 1)
            std::swap(mii1, mii2);

        l[mii1] = t;
        l[mii2] = l[i - 1];
    }

    return ans;
}

int main() {
    std::vector<int> L{8, 5, 8};
    
    std::cout << solve(L) << std::endl;
    return 0;
}