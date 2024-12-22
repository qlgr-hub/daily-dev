#include <iostream>
#include <array>

static const int MAX = 2000001;
static std::array<int, MAX> A;

void maxHeapify(int i, int H) {
    int l = 2 * i;
    int r = 2 * i + 1;

    int largest = i;
    if (l <= H && A[l] > A[i])
        largest = l;

    if (r <= H && A[r] > A[largest])
        largest = r;

    if (i != largest) {
        std::swap(A[i], A[largest]);
        maxHeapify(largest, H);
    }
}

int main() {
    int H = 0;
    std::cin >> H;

    for (int i = 1; i <= H; ++i)
        std::cin >> A[i];

    for (int i = H / 2; i >= 1; --i)
        maxHeapify(i, H);

    for (int i = 1; i <= H; ++i)
        std::cout << ' ' << A[i];
    std::cout << '\n';

    return 0;
}