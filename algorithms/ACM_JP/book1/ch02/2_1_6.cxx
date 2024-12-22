#include <iostream>
#include <vector>
#include <algorithm>

void permutation1(std::vector<int>& perm, int pos) {
    static std::vector<bool> used(perm.size(), false);

    if (pos == perm.size()) {
        for (const auto& val : perm) {
            std::cout << val << " ";
        }
        std::cout << "\n";
        return;
    }

    for (int i = 0; i < perm.size(); ++i) {
        if (!used[i]) {
            perm[pos] = i;
            used[i] = true;

            permutation1(perm, pos + 1);

            used[i] = false;
        }
    }
}

void permutation2(std::vector<int>& perm) {
    do {
        for (const auto& val : perm) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    } while (std::next_permutation(perm.begin(), perm.end()));
}

int main() {
    std::vector<int> perm{ 0, 1, 2, 3, 4, 5, 6 };
    permutation2(perm);
    return 0;
}