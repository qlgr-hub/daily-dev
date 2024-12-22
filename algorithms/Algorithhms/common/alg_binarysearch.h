#pragma once

#include <vector>

template <typename T, bool Iterative = true>
class BinarySearch {
    // 这个实现其实不太好，递归函数这么多参数感觉不太好（虽然参数都不大）
    static int _search_r(const std::vector<T>& seq, const T& key, int lo, int hi) {
        if (lo > hi)
            return -1;

        auto mid{ lo + ((hi - lo) >> 2) };
        if (key < seq[mid])
            return _search_r(seq, key, lo, mid - 1);
        else if (key > seq[mid])
            return _search_r(seq, key, mid + 1, hi);
        else
            return mid;
    }

    static int _search_i(const std::vector<T>& seq, const T& key) {
        auto lo{ 0 };
        auto hi{ static_cast<int>(seq.size() - 1) };

        while (lo <= hi) {
            auto mid{ lo + ((hi - lo) >> 2) };

            if (key < seq[mid])
                hi = mid - 1;
            else if (key > seq[mid])
                lo = mid + 1;
            else
                return mid;
        }

        return -1;
    }

public:
    static int search(const std::vector<T>& seq, const T& key) {
        if constexpr (Iterative) {
            return _search_i(seq, key);
        }
        else {
            return _search_r(seq, key, 0, seq.size() - 1);   
        }
    }
};
