#include <algorithm>
#include <iostream>
#include <array>

template <typename T, int MAX_SIZE = 100>
bool bsearch(std::array<T, MAX_SIZE>& A, int n, int key) {
    int left = 0;
    int right = n;

    bool ret = false;
    while (left < right) {
        int mid = (left + right) / 2;

        if (A[mid] == key) {
            ret = true;
            break;
        }
        else if (A[mid] > key) {
            right = mid;
        }
        else /* if (A[mid] < key) */ {
            left = mid + 1;
        }
    }

    return ret;
}

int main1() {
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
        if ( bsearch<int, 1001>(A, n, key) )
            ++sum;
    }

    std::cout << sum << '\n'; 
    return 0;
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
        if ( key == *std::lower_bound(A.begin(), A.begin() + n, key) )
            ++sum;
    }

    std::cout << sum << '\n'; 
    return 0;
}
