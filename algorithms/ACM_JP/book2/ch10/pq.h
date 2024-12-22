#ifndef _PRIORITY_QUEUE_H_
#define _PRIORITY_QUEUE_H_

#include <array>

template <typename  T>
class PQ {
    static const int INFTY = (1 << 30);
    static const int MAX = 2000001;
    std::array<T, MAX> A;
    int H;

    void maxHeapify(int i) {
        int l = 2 * i;
        int r = 2 * i + 1;

        int largest = i;
        if (l <= H && A[l] > A[i])
            largest = l;

        if (r <= H && A[r] > A[largest])
            largest = r;

        if (i != largest) {
            std::swap(A[i], A[largest]);
            maxHeapify(largest);
        }
    }

    void increaseKey(int i, const T& key) {
        if (key < A[i])
            return;
        A[i] = key;
        while (i > 1 && A[i / 2] < A[i]) {
            std::swap(A[i], A[i / 2]);
            i = i / 2;
        }
    }

public:
    PQ() : H{ 0 } {
    }

    void insert(const T& key) {
        A[++H] = -INFTY;
        increaseKey(H, key);
    }

    int extract() {
        if (H < 1)
            return -INFTY;

        int maxv = A[1];
        A[1] = A[H--];
        maxHeapify(1);
        return maxv;
    }
};

#endif /* ! _PRIORITY_QUEUE_H_ */
