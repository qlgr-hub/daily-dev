#pragma once

#include <algorithm>
#include <vector>


// 书中没有提到，但是作为有知名度的算法，咱先实现
template <typename T, bool UseFlag = true>
class BubbleSort {
    static void _sortBasic(std::vector<T>& arr) {
        for (int i{ 0 }; i < arr.size() - 1; ++i) {
            for (int j{ 0 }; j < arr.size() - 1 - i; ++j) {
                if (arr[j] > arr[j + 1]) {
                    std::swap(arr[j], arr[j + 1]);
                }
            }
        }
    }

    // 参考自：挑战程序设计竞赛 2（算法和数据结构）
    static void _sortFlag(std::vector<T>& arr) {
        const int N{ static_cast<int>(arr.size()) };
        bool flag{ true };
        for (int i{ 0 }; flag; ++i) {
            flag = false;
            for (int j{ N - 1 }; j >= i; --j) {
                if (arr[j] < arr[j - 1]) {
                    std::swap(arr[j], arr[j - 1]);
                    flag = true;
                }
            }
        }
    }

public:
    static void sort(std::vector<T>& arr) {
        if constexpr (UseFlag) {
            _sortFlag(arr);
        }
        else {
            _sortBasic(arr);
        }
    }
};


template <typename T>
class SelectionSort {
public:
    static void sort(std::vector<T>& arr) {
        for (int i{ 0 }; i < arr.size(); ++i) {
            auto minIdx{ i };
            for (int j{ i + 1 }; j < arr.size(); ++j) {
                if (arr[j] < arr[minIdx])
                    minIdx = j;
            }

            std::swap(arr[i], arr[minIdx]);
        }
    }
};


template <typename T, bool UseMove = true>
class InsertionSort {
    static void _sortSwap(std::vector<T>& arr) {
        for (int i{ 0 }; i < arr.size(); ++i) {
            for (int j{ i }; j > 0 && arr[j] < arr[j - 1]; --j) {
                std::swap(arr[j], arr[j - 1]);
            }
        }
    }

    // 参考自：挑战程序设计竞赛 2（算法和数据结构）
    static void _sortMove(std::vector<T>& arr) {
        for (int i{ 0 }; i < arr.size(); ++i) {
            auto v{ arr[i] };
            auto j{ i - 1 };

            while (j >= 0 && v < arr[j]) {
                arr[j + 1] = arr[j];
                --j;
            }

            arr[j + 1] = v;
        }
    }

public:
    static void sort(std::vector<T>& arr) {
        if constexpr (UseMove) {
            _sortMove(arr);
        }
        else {
            _sortSwap(arr);
        }
    }

    // 参考自：挑战程序设计竞赛 2（算法和数据结构）
    static void sortGap(std::vector<T>& arr, int gap) {
        for (int i{ gap }; i < arr.size(); ++i) {
            auto v{ arr[i] };
            auto j{ i - gap };

            while (j >= 0 && v < arr[j]) {
                arr[j + gap] = arr[j];
                j -= gap;
            }

            arr[j + gap] = v;
        }
    }
};


template <typename T, bool UseMove = true>
class ShellSort {
    // 参考自：挑战程序设计竞赛 2（算法和数据结构）
    static void _sortMove(std::vector<T>& arr) {
        static std::vector<int> G{};

        G.clear();
        for (int h{ 1 }; h <= arr.size(); h = 3 * h + 1) {
            G.push_back(h);
        }

        for (int i{ static_cast<int>(G.size()) - 1 }; i >= 0; --i)
            InsertionSort<T>::sortGap(arr, G[i]);
    }

    static void _sortSwap(std::vector<T>& arr) {
        const int N{ static_cast<int>(arr.size()) };

        int h{ 1 };
        while (h < N / 3)
            h = 3 * h + 1;

        while (h >= 1) {
            for (int i{ h }; i < N; ++i) {
                for (int j{ i }; j >= h && arr[j] < arr[j - h]; j -= h)
                    std::swap(arr[j], arr[j - h]);
            }
            h /= 3;
        }
    }

public:
    static void sort(std::vector<T>& arr) {
        if constexpr (UseMove) {
            _sortMove(arr);
        }
        else {
            _sortSwap(arr);
        }
    }
};