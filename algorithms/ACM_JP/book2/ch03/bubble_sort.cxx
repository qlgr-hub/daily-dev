#include <iostream>
#include <type_traits>
#include <vector>

void trace(const std::vector<int>& A) {
    for (const auto& v : A) {
        std::cout << v << " ";
    }
    std::cout << '\n';
}

int bubbleSort(std::vector<int>& A) {
    int sc = 0;
    for (int i = 0; i < A.size(); ++i) {
        for (int j = i + 1; j < A.size(); ++j) {
            if (A[j] < A[i]) {
                std::swap(A[j], A[i]);
                ++sc;
            }
        }
    }

    return sc;
}

int bubbleSortWithFlag(std::vector<int>& A) {
    int sc = 0;
    const int N = A.size();

    bool flag = true;
    for (int i = 0; flag; ++i) {
        flag = 0;
        for (int j = N - 1; j >= i + 1; --j) {
            if (A[j] < A[j - 1]) {
                std::swap(A[j], A[j - 1]);
                ++sc;
                flag = true;
            }
        }
    }

    return sc;
}

int main() {
    int N = 0;
    std::cin >> N;

    std::vector<int> A(N);
    for (auto& v : A) {
        std::cin >> v;
    }

    int sc = bubbleSortWithFlag(A);
    trace(A);
    std::cout << sc << '\n';

    return 0;
}