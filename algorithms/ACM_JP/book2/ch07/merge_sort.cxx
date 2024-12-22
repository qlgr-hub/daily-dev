#include <iostream>
#include <limits>
#include <vector>

static int cnt = 0;

void print(const std::vector<int>& A) {
    for (auto val : A) {
        std::cout << val << ' ';
    }
    std::cout << '\n';
};

void merge(std::vector<int>& V, int l, int m, int r) {
    std::vector<int> L(V.cbegin() + l, V.cbegin() + m);
    std::vector<int> R(V.cbegin() + m, V.cbegin() + r);
    L.push_back(std::numeric_limits<int>::max());
    R.push_back(std::numeric_limits<int>::max());

    for (int i = 0, j = 0, k = l; k < r; ++k) {
        ++cnt;
        if (L[i] <= R[j])
            V[k] = L[i++];
        else
            V[k] = R[j++];
    }
}

void mergeSort(std::vector<int>& V, int l, int r) {
    if (l + 1 < r) {
        // int m = (l + r) / 2;
        int m = l + (r - l) / 2;
        mergeSort(V, l, m);
        mergeSort(V, m, r);
        merge(V, l, m, r);
    }
}

int main() {
    int n = 0;
    std::cin >> n;

    std::vector<int> V(n);
    for (int i = 0; i < n; ++i)
        std::cin >> V[i];

    // print(V);
    mergeSort(V, 0, V.size());
    print(V);

    std::cout << cnt << '\n';
    return 0;
}
