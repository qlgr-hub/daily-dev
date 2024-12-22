#include <iostream>
#include <type_traits>
#include <vector>

int selectionSort(std::vector<int>& A) {
    int sc = 0;

    for (int i = 0; i < A.size() - 1; ++i) {
        int minj = i;
        for (int j = i; j < A.size(); ++j) {
            if (A[j] < A[minj])
                minj = j;
        }

        if (i != minj) {
            std::swap(A[i], A[minj]);
            ++sc;
        }
    }

    return sc;
}

void trace(const std::vector<int>& A) {
    for (const auto& v : A) {
        std::cout << v << " ";
    }
    std::cout << '\n';
}

int main() {
    int N = 0;
    std::cin >> N;

    std::vector<int> A(N);
    for (auto& v : A) {
        std::cin >> v;
    }

    int sc = selectionSort(A);
    trace(A);
    std::cout << sc << '\n';

    return 0;
}
