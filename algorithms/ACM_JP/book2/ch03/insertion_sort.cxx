#include <iostream>
#include <vector>

void trace(const std::vector<int>& A) {
    for (const auto& v : A) {
        std::cout << v << " ";
    }
    std::cout << '\n';
}

void insertionSort(std::vector<int>& A) {
    for (int i = 1; i < A.size(); ++i) {
        int v = A[i];
        int j = i - 1;

        while (j >= 0 && A[j] > v) {
            A[j + 1] = A[j];
            --j;
        }
        A[j + 1] = v;

        trace(A);
    }
}


int main() {
    int N = 0;
    std::cin >> N;

    std::vector<int> A(N);
    for (auto& v : A) {
        std::cin >> v;
    }

    trace(A);
    insertionSort(A);

    return 0;
}
