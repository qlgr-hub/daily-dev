#include <iostream>
#include <array>

static const int MAX = 100001;

static std::array<int, MAX> A;

int parent(int i) {
    return i / 2;
}

int left(int i) {
    return 2 * i;
}

int right(int i) {
    return 2 * i + 1;
}

int main() {
    int H;
    std::cin >> H;
    for (int i = 1; i <= H; ++i)
        std::cin >> A[i];

    for (int i = 1; i <= H; ++i) {
        std::cout << "node " << i << ": key = " << A[i] << ", ";

        if (parent(i) >= 1)
            std::cout << "parent key = " << A[parent(i)] << ", ";

        if (left(i) <= H)
            std::cout << "left key = " << A[left(i)] << ", ";

        if (right(i) <= H)
            std::cout << "right key = " << A[right(i)] << ", ";

        std::cout << '\n';
    }

    return 0;
}

