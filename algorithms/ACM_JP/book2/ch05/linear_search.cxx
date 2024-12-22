#include <iostream>
#include <array>

template <typename T, int MAX_SIZE = 100>
bool search(std::array<T, MAX_SIZE>& A, int n, int key) {
    int i = 0;
    A[n] = key;
    while (A[i] != key)
        ++i;
    return i != n;
}

int main() {
    int n = 0;
    std::cin >> n;
    if (n > 1000) {
        std::cout << "n is to big\n";
        return -1;
    }

    std::array<int, 1001> A;
    for (int i = 0; i < n; ++i) {
        std::cin >> A[i];
    }

    int sum = 0;
    int q = 0;
    std::cin >> q;
    for (int i = 0; i < q; ++i) {
        int key = 0;
        std::cin >> key;
        if ( search<int, 1001>(A, n, key) )
            ++sum;
    }

    std::cout << sum << '\n'; 
    return 0;
}