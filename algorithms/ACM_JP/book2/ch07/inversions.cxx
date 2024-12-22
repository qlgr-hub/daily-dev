#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

int merge(std::vector<int>& V, int l, int m, int r) {
    std::vector<int> L(V.cbegin() + l, V.cbegin() + m);
    std::vector<int> R(V.cbegin() + m, V.cbegin() + r);
    L.push_back(std::numeric_limits<int>::max());
    R.push_back(std::numeric_limits<int>::max());

    int cnt = 0;
    for (int i = 0, j = 0, k = l; k < r; ++k) {
        if (L[i] <= R[j])
            V[k] = L[i++];
        else {
            V[k] = R[j++];
            cnt += (m - l) - i;
        }
    }

    return cnt;
}

int mergeSort(std::vector<int>& V, int l, int r) {
    if (l + 1 < r) {
        int v1 = 0, v2 = 0, v3 = 0;
        // int m = (l + r) / 2;
        int m = l + (r - l) / 2;
        v1 = mergeSort(V, l, m);
        v2 = mergeSort(V, m, r);
        v3 = merge(V, l, m, r);
        return (v1 + v2 + v3);
    }
    return 0;
}

int main() {
    int n = 0;
    std::cin >> n;

    std::vector<int> V(n);
    for (int i = 0; i < n; ++i)
        std::cin >> V[i];

    int cnt = mergeSort(V, 0, V.size());
    std::cout << cnt << '\n';
    return 0;
}
