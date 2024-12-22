#pragma once

#include <limits>
#include <vector>
#include <algorithm>
#include <iostream>


// 归并排序实现类型
enum class MergeSortImpl {
    MST_RFEM = 0,  // 递归、固定额外内存（Recursive fixed extra memory）
    MST_RDEM= 1,   // 递归、动态额外内存（Recursive dynamic extra memory）
    MST_IFEM = 2,  // 迭代、固定额外内存（Iterative fixed extra memory）
    MST_IDEM = 3   // 迭代、固定额外内存（Iterative dynamic extra memory）
};

template <typename T, MergeSortImpl MSI = MergeSortImpl::MST_IFEM>
class MergeSort {
    // 参考自：挑战程序设计竞赛 2（算法和数据结构）
    // 不过这个频繁创建销毁 vector 会影响效率
    static void _mergeDEM(std::vector<int>& arr, int l, int m, int r) {
        std::vector<T> L{ arr.cbegin() + l, arr.cbegin() + m };
        std::vector<T> R{ arr.cbegin() + m, arr.cbegin() + r };
        L.push_back(std::numeric_limits<T>::max());
        R.push_back(std::numeric_limits<T>::max());

        int i{ 0 };
        int j{ 0 };
        for (int k{ l }; k < r; ++k) {
            if (L[i] <= R[j]) {
                arr[k] = L[i++];
            }
            else {
                arr[k] = R[j++];
            }
        }
    }

    // 参考自：挑战程序设计竞赛 2（算法和数据结构）
    static void _sortRDEM(std::vector<int>& arr, int l, int r) {
        if (l + 1 < r) {
            int m = l + ((r - l) >> 1);
            _sortRDEM(arr, l, m);
            _sortRDEM(arr, m, r);
            _mergeDEM(arr, l, m, r);
        }
    }

    static void _mergeFEM(std::vector<T>& arr, std::vector<T>& arrCpy, int lo, int mid, int hi) {
        for (int i{lo}; i <= hi; ++i) {
            arrCpy[i] = arr[i];
        }

        auto i{ lo };
        auto j{ mid + 1 };
        for (int k{lo}; k <= hi; ++k) {
            if (i > mid) {
                arr[k] = arrCpy[j++];
            }
            else if (j > hi) {
                arr[k] = arrCpy[i++];
            }
            else if (arrCpy[j] < arrCpy[i]) {
                arr[k] = arrCpy[j++];
            }
            else {
                arr[k] = arrCpy[i++];
            }
        }
    }

    static void _sortRFEM(std::vector<T>& arr, std::vector<T>& arrCpy, int lo, int hi) {
        if (hi <= lo)
            return;

        auto mid{ lo + ((hi - lo) >> 1) };
        _sortRFEM(arr, arrCpy, lo, mid);
        _sortRFEM(arr, arrCpy, mid + 1, hi);
        _mergeFEM(arr, arrCpy, lo, mid, hi);
    }

public:
    static void sort(std::vector<T>& arr) {
        int N{ static_cast<int>(arr.size()) };

        if constexpr (MSI == MergeSortImpl::MST_RFEM) {
            std::vector<T> arrCpy(N, T{});
            _sortRFEM(arr, arrCpy, 0, N - 1);
        }
        else if (MSI == MergeSortImpl::MST_RDEM) {
            _sortRDEM(arr, 0, N);
        }
        else if (MSI == MergeSortImpl::MST_IFEM){
            std::vector<T> arrCpy(N, T{});
            
            for (int sz{ 1 }; sz < N; sz += sz) {
                for (int lo{ 0 }; lo < N - sz; lo += sz + sz) {
                    _mergeFEM(arr, arrCpy, lo, lo + sz - 1, std::min(lo + sz + sz - 1, N - 1));
                }
            }
        }
        else {
            for (int sz{ 1 }; sz < N; sz += sz) {
                for (int lo{ 0 }; lo < N - sz; lo += sz + sz) {
                    _mergeDEM(arr, lo, lo + sz, std::min(lo + sz + sz, N));
                }
            }
        }
    }
};


template <typename T, bool Compact = false>
class QuickSort {
    // 参考自：挑战程序设计竞赛 2（算法和数据结构）
    // 感觉这个可读性更好，代码还简洁
    static int _partitionCompact(std::vector<T>& arr, int lo, int hi) {
        auto v = arr[hi];

        int i{ lo - 1 };
        for (int j{ lo }; j < hi; ++j) {
            if (arr[j] < v) {
                std::swap(arr[++i], arr[j]);
            }
        }
        std::swap(arr[++i], arr[hi]);
        return i;
    }

    static int _partition(std::vector<T>& arr, int lo, int hi) {
        auto v{ arr[lo] };

        auto i{ lo };
        auto j{ hi + 1 };
        while (true) {
            while (arr[++i] < v) {
                if (i == hi)
                    break;
            }

            while (v < arr[--j]) {
                if (j == lo)
                    break;
            }

            if (i >= j)
                break;
            
            std::swap(arr[i], arr[j]);
        }

        std::swap(arr[lo], arr[j]);
        return j;
    }

    static void _sort(std::vector<T>& arr, int lo, int hi) {
        if (hi <= lo)
            return;
        int j{};
        if constexpr (Compact) {
            j = _partitionCompact(arr, lo, hi);
        }
        else {
             j = _partition(arr, lo, hi);
        }
        _sort(arr, lo, j - 1);
        _sort(arr, j + 1, hi);
    }

    // 它的 partition 来自著名的荷兰国旗问题
    static void _sort3way(std::vector<T>& arr, int lo, int hi) {
        if (hi <= lo)
            return;

        auto lt{ lo };
        auto i{ lo + 1 };
        auto gt{ hi };

        auto v{ arr[lo] };
        while (i <= gt) {
            if (arr[i] < v) {
                std::swap(arr[lt++], arr[i++]);
            }
            else if (arr[i] > v) {
                std::swap(arr[i], arr[gt--]);
            }
            else {
                ++i;
            }
        }

        _sort3way(arr, lo, lt - 1);
        _sort3way(arr, gt + 1, hi);
    }

public:
    static void sort(std::vector<T>& arr) {
        _sort(arr, 0, arr.size() - 1);
    }

    static void sort3way(std::vector<T>& arr) {
        _sort3way(arr, 0, arr.size() - 1);
    }
};


template <typename T, bool JustSink = true>
class HeapSort {
    // 参考自：挑战程序设计竞赛 2（算法和数据结构）
    static void _heapify(std::vector<T>& arr, int i, int len) {
        auto left{ 2 * i + 1  };
        auto right{ 2 * i + 2 };
        auto topIdx{ i };
        
        if (left < len && arr[left] > arr[topIdx])
            topIdx = left;

        if (right < len && arr[right] > arr[topIdx])
            topIdx = right;

        if (i != topIdx) {
            std::swap(arr[i], arr[topIdx]);
            _heapify(arr, topIdx, len);
        }
    }

    static bool _less(const std::vector<T>& arr, int i, int j) {
        return arr[i - 1] < arr[j - 1];
    }

    static void _exch(std::vector<T>& arr, int i, int j) {
        // std::swap(arr[i - 1], arr[j - 1]);
        auto temp{ arr[i - 1] };
        arr[i - 1] = arr[j - 1];
        arr[j - 1] = temp;
    }

    static void _sink(std::vector<T>& arr, int k, int N) {
        while (2 * k <= N) {
            int j{ 2 * k };
            if (j < N && _less(arr, j, j + 1))
                ++j;
            if (!_less(arr, k, j))
                break;
            _exch(arr, k, j);
            k = j;
        }
    }

public:
    static void sort(std::vector<T>& arr) {
        if constexpr (JustSink) {
            int N{ static_cast<int>(arr.size()) };
            for (int k{ N / 2 }; k >= 1; --k) {
                _sink(arr, k, N);
            }

            while (N > 1) {
                _exch(arr, 1, N--);
                _sink(arr, 1, N);
            }
        }
        else {
            // 参考自：挑战程序设计竞赛 2（算法和数据结构）
            for (int i = arr.size() / 2 - 1; i >= 0; --i) {
                _heapify(arr, i, arr.size());
            }

            for (int i = arr.size() - 1; i >= 1; --i) {
                std::swap(arr[0], arr[i]);
                _heapify(arr, 0, i);
            }
        }
    }
};
