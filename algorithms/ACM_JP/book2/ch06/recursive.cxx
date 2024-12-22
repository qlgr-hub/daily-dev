#include <cstdint>
#include <iostream>
#include <vector>


int64_t factotial(int n) {
    if (n == 1)
        return 1;
    return n * factotial(n - 1);
}

int findMax(const std::vector<int>& A, int l, int r) {
    if (l == r - 1)
        return A[l];
    
    int m = (l + r) / 2;
    int u = findMax(A, l, m);
    int v = findMax(A, m, r);
    int x = std::max(u, v);
    return x;
}

int main() {

    std::cout << factotial(12) << '\n';

    std::vector<int> A{1, 7, 88, 9, 87, 73, 3, 4, 9, 65};
    std::cout << findMax(A, 0, A.size()) << '\n';

    return 0;
}
