#include <algorithm>
#include <iostream>
#include <string>
#include <string_view>

std::string solve(std::string_view s) {
    int s_s = 0;
    int r_s = s.size() - 1;

    std::string ret;
    while (s_s <= r_s) {
        bool left = (s[s_s] < s[r_s]) ? true : false;

        if (left)
            ret.append(1, s[s_s++]);
        else
            ret.append(1, s[r_s--]);
    }

    return std::move(ret);
}

int main() {
    std::string S;
    std::cin >> S;

    std::cout << solve(S) << std::endl;
    return 0;
}