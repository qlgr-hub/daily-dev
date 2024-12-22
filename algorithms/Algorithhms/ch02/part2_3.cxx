#include <iostream>
#include "../common/alg_utils.h"
#include "../common/alg_highersorting.h"
#include "../common/alg_elementarysorting.h"


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
    // AMD 7950x （3 way partition 好不好分场景，快排 partition 的元素应该是影响比较大的）
    // ./build/part2_3 10    
    // Elaps: 0.0000154000 sec for array of size(10) <- merge sort
    // Elaps: 0.0000005300 sec for array of size(10) <- quick sort
    // Elaps: 0.0000005600 sec for array of size(10) <- quick sort (compact)
    // Elaps: 0.0000005200 sec for array of size(10) <- quick sort (3 way partition)
    //
    // ./build/part2_3 100
    // Elaps: 0.0000238700 sec for array of size(100) <- merge sort
    // Elaps: 0.0000056100 sec for array of size(100) <- quick sort
    // Elaps: 0.0000058300 sec for array of size(100) <- quick sort (compact)
    // Elaps: 0.0000074500 sec for array of size(100) <- quick sort (3 way partition)
    //
    // ./build/part2_3 1000
    // Elaps: 0.0001437410 sec for array of size(1000) <- merge sort
    // Elaps: 0.0000761400 sec for array of size(1000) <- quick sort
    // Elaps: 0.0000877000 sec for array of size(1000) <- quick sort (compact)
    // Elaps: 0.0001171300 sec for array of size(1000) <- quick sort (3 way partition)
    //
    // ./build/part2_3 10000
    // Elaps: 0.0016916220 sec for array of size(10000) <- merge sort
    // Elaps: 0.0009833110 sec for array of size(10000) <- quick sort
    // Elaps: 0.0012980110 sec for array of size(10000) <- quick sort (compact)
    // Elaps: 0.0016128820 sec for array of size(10000) <- quick sort (3 way partition)
    //
    // ./build/part2_3 200000
    // Elaps: 0.0438666750 sec for array of size(200000) <- merge sort
    // Elaps: 0.0250570650 sec for array of size(200000) <- quick sort
    // Elaps: 0.0329478430 sec for array of size(200000) <- quick sort (compact)
    // Elaps: 0.0443952820 sec for array of size(200000) <- quick sort (3 way partition)
    static void test(int argc, char* argv[]) {
        if (argc < 2)
            return;

        int N{ std::stoi(argv[1]) };
        std::vector<int> arr{};
        Uniform<int> uni{ { 0, N } };
        uni(arr, N);

        std::vector<int> arrCpy1{ arr.cbegin(), arr.cend() };
        std::vector<int> arrCpy2{ arr.cbegin(), arr.cend() };
        std::vector<int> arrCpy3{ arr.cbegin(), arr.cend() };

        std::cout.flags(std::ios::fixed);
        std::cout.precision(10);

        TimeStats ts1{};
        MergeSort<int>::sort(arr);
        std::cout << "Elaps: " << ts1.elapsSec() << " sec for array of size(" << N << ") <- merge sort\n";

        TimeStats ts2{};
        QuickSort<int, false>::sort(arrCpy1);
        std::cout << "Elaps: " << ts2.elapsSec() << " sec for array of size(" << N << ") <- quick sort\n";

        TimeStats ts3{};
        QuickSort<int, true>::sort(arrCpy2);
        std::cout << "Elaps: " << ts3.elapsSec() << " sec for array of size(" << N << ") <- quick sort (compact)\n";

        TimeStats ts4{};
        QuickSort<int>::sort3way(arrCpy3);
        std::cout << "Elaps: " << ts4.elapsSec() << " sec for array of size(" << N << ") <- quick sort (3 way partition)\n";

        std::cout << _isSorted(arr) << '\n';
        std::cout << _isSorted(arrCpy1) << '\n';
        std::cout << _isSorted(arrCpy2) << '\n';
        std::cout << _isSorted(arrCpy3) << '\n';

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