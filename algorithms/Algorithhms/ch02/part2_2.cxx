#include <iostream>
#include "../common/alg_highersorting.h"
#include "../common/alg_utils.h"

class Comparator {
    static void _show(std::vector<int>& arr) {
        for (const auto& v : arr) {
            std::cout << v << ' ';
        }
        std::cout << '\n';
    }

    static bool _isSorted(const std::vector<int>& arr) {
        bool res{ true };
        for (int i{ 1 }; i < arr.size(); ++i) {
            if (arr[i - 1] > arr[i])
                res = false;
        }
        return res;
    }

public:
    // AMD 7950x （数据量越大使用动态额外内存的比固定额外内存的要越慢, 迭代版的比递归的快些但是差距不明显）
    // ./build/part2_2 10
    // Elaps: 0.0000146000 sec for array of size(10) <- merge sort (recursive fiexd extra memory)
    // Elaps: 0.0000054400 sec for array of size(10) <- merge sort (recursive dynamic extra memory)
    // Elaps: 0.0000009100 sec for array of size(10) <- merge sort (iterative fiexd extra memory)
    // Elaps: 0.0000051200 sec for array of size(10) <- merge sort (iterative dynamic extra memory)
    //
    // ./build/part2_2 100
    // Elaps: 0.0000190800 sec for array of size(100) <- merge sort (recursive fiexd extra memory)
    // Elaps: 0.0000558400 sec for array of size(100) <- merge sort (recursive dynamic extra memory)
    // Elaps: 0.0000090900 sec for array of size(100) <- merge sort (iterative fiexd extra memory)
    // Elaps: 0.0000534300 sec for array of size(100) <- merge sort (iterative dynamic extra memory)
    //
    // ./build/part2_2 1000
    // Elaps: 0.0001378100 sec for array of size(1000) <- merge sort (recursive fiexd extra memory)
    // Elaps: 0.0005597710 sec for array of size(1000) <- merge sort (recursive dynamic extra memory)
    // Elaps: 0.0001183100 sec for array of size(1000) <- merge sort (iterative fiexd extra memory)
    // Elaps: 0.0005512210 sec for array of size(1000) <- merge sort (iterative dynamic extra memory)
    //
    // ./build/part2_2 10000
    // Elaps: 0.0016380020 sec for array of size(10000) <- merge sort (recursive fiexd extra memory)
    // Elaps: 0.0060336690 sec for array of size(10000) <- merge sort (recursive dynamic extra memory)
    // Elaps: 0.0015665320 sec for array of size(10000) <- merge sort (iterative fiexd extra memory)
    // Elaps: 0.0060133790 sec for array of size(10000) <- merge sort (iterative dynamic extra memory)
    //
    // ./build/part2_2 200000
    // Elaps: 0.0427276030 sec for array of size(200000) <- merge sort (recursive fiexd extra memory)
    // Elaps: 0.1247779800 sec for array of size(200000) <- merge sort (recursive dynamic extra memory)
    // Elaps: 0.0414084500 sec for array of size(200000) <- merge sort (iterative fiexd extra memory)
    // Elaps: 0.1238012180 sec for array of size(200000) <- merge sort (iterative dynamic extra memory)
    static void test(int argc, char* argv[]) {
        if (argc < 2)
            return;

        int N = std::stoi(argv[1]);
        std::vector<int> arr{};
        Uniform<int> uni{ { 0, N } };
        uni(arr, N);

        std::vector<int> arrCpy1{ arr.cbegin(), arr.cend() };
        std::vector<int> arrCpy2{ arr.cbegin(), arr.cend() };
        std::vector<int> arrCpy3{ arr.cbegin(), arr.cend() };

        std::cout.flags(std::ios::fixed);
        std::cout.precision(10);

        TimeStats ts1{};
        MergeSort<int, MergeSortImpl::MST_RFEM>::sort(arr);
        std::cout << "Elaps: " << ts1.elapsSec() << " sec for array of size(" << N << ") <- merge sort (recursive fiexd extra memory)\n";

        TimeStats ts2{};
        MergeSort<int, MergeSortImpl::MST_RDEM>::sort(arrCpy1);
        std::cout << "Elaps: " << ts2.elapsSec() << " sec for array of size(" << N << ") <- merge sort (recursive dynamic extra memory)\n";

        TimeStats ts3{};
        MergeSort<int, MergeSortImpl::MST_IFEM>::sort(arrCpy2);
        std::cout << "Elaps: " << ts3.elapsSec() << " sec for array of size(" << N << ") <- merge sort (iterative fiexd extra memory)\n";

        TimeStats ts4{};
        MergeSort<int, MergeSortImpl::MST_IDEM>::sort(arrCpy3);
        std::cout << "Elaps: " << ts4.elapsSec() << " sec for array of size(" << N << ") <- merge sort (iterative dynamic extra memory)\n";

        // check results
        std::cout << _isSorted(arr) << '\n';
        std::cout << _isSorted(arrCpy1) << '\n';
        std::cout << _isSorted(arrCpy2) << '\n';
        std::cout << _isSorted(arrCpy3) << '\n';

        // show results in small size
        if (N <= 20) {
            _show(arr);
            _show(arrCpy1);
            _show(arrCpy2);
            _show(arrCpy3);
        }
    }
};


int main(int argc, char* argv[]) {
    Comparator::test(argc, argv);

    return 0;
}