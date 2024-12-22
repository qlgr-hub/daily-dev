#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>
#include "../common/alg_utils.h"
#include "../common/alg_binarysearch.h"


class TwoSum {
public:
    static int count(const std::vector<int>& arr) {
        const auto N{ arr.size() };

        int cnt{ 0 };
        for (int i{ 0 }; i < N; ++i) {
            for (int j{ i + 1 }; j < N; ++j) {
                if (arr[i] + arr[j] == 0) {
                    ++cnt;
                }
            }
        }

        return cnt;
    }
};


class TwoSumFast {
public:
    static int count(std::vector<int>& arr) {
        std::sort(arr.begin(), arr.end());

        const auto N{ arr.size() };
        int cnt{ 0 };
        for (int i{ 0 }; i < N; ++i) {
            if (BinarySearch<int>::search(arr, -arr[i]) > i) {
                ++cnt;
            }
        }

        return cnt;
    }
};


class TreeSum {
public:
    static int count(const std::vector<int>& arr) {
        const auto N{ arr.size() };

        int cnt{ 0 };
        for (int i{ 0 }; i < N; ++i) {
            for (int j{ i + 1 }; j < N; ++j) {
                for (int k{ j + 1 }; k < N; ++k) {
                    if (arr[i] + arr[j] + arr[k] == 0) {
                        ++cnt;
                    }
                }
            }
        }

        return cnt;
    }
};



class TreeSumFast {
public:
    static int count(std::vector<int>& arr) {
        std::sort(arr.begin(), arr.end());

        const auto N{ arr.size() };
        int cnt{ 0 };
        for (int i{ 0 }; i < N; ++i) {
            for (int j{ i + 1 }; j < N; ++j)
            if (BinarySearch<int>::search(arr, -(arr[i] + arr[j])) > j) {
                ++cnt;
            }
        }

        return cnt;
    }
};


class AlgorithmEvaluationDemo {
public:
    static void test(int argc, char* argv[]) {
        if (argc < 2)
            return;

        std::vector<int> nums1;
        std::ifstream in(argv[1]);
        int num{ 0 };
        while (in >> num) {
            nums1.push_back(num);
        }
        std::vector<int> nums2{ nums1.cbegin(), nums1.cend() };

        // fiexd floating output precision is 10
        std::cout.flags(std::ios::fixed);
        std::cout.precision(10);

        // TwoSum naive ver.
        TimeStats ts1{};
        auto res1{ TwoSum::count(nums1) };
        std::cout << "result: " << res1 << ", count elaps: " << ts1.elapsSec() << " sec. <- [TwoSum]\n";
        
        // TwoSum fast ver.
        TimeStats ts2{};
        auto res2{ TwoSumFast::count(nums1) };
        std::cout << "result: " << res2 << ", count elaps: " << ts2.elapsSec() << " sec. <- [TwoSumFast]\n";

        // TreeSum naive ver.
        TimeStats ts3{};
        auto res3{ TreeSum::count(nums2) };
        std::cout << "result: " << res3 << ", count elaps: " << ts3.elapsSec() << " sec. <- [TreeSum]\n";
        
        // TreeSumFast fast ver.
        TimeStats ts4{};
        auto res4{ TreeSumFast::count(nums2) };
        std::cout << "result: " << res4 << ", count elaps: " << ts4.elapsSec() << " sec. <- [TreeSumFast]\n";
    }
};


int main(int argc, char* argv[]) {
    AlgorithmEvaluationDemo::test(argc, argv);

    return 0;
}
