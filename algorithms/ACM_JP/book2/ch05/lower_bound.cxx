#include <iostream>
#include <algorithm>
#include <iterator>

int main() {
    int A[] = {
        1, 1, 2, 2, 2, 4, 5, 5, 6, 8, 8, 8, 10, 15
    };

    int* pos;
    int idx;
    pos = std::lower_bound(A, A + 14, 3);
    idx = std::distance(A, pos);
    std::cout << "A[" << idx << "] = " << *pos << '\n';

    pos = std::lower_bound(A, A + 14, 2);
    idx = std::distance(A, pos);
    std::cout << "A[" << idx << "] = " << *pos << '\n';

    return 0;
}