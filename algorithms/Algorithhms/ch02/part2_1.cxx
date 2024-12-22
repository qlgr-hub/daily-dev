#include <iostream>
#include <vector>
#include "../common/alg_utils.h"
#include "../common/alg_elementarysorting.h"

class Comparator {
    static void _show(const std::vector<int>& arr) {
        for (const auto& e : arr) {
            std::cout << e << ' ';
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
    // AMD 7950x 
    // ./build/part2_1 10    
    // Elaps: 0.0000132500 sec for array of size(10) <- bubble sort
    // Elaps: 0.0000005500 sec for array of size(10) <- selection sort
    // Elaps: 0.0000003500 sec for array of size(10) <- insertion sort
    // Elaps: 0.0000015800 sec for array of size(10) <- shell sort
    //
    // ./build/part2_1 100
    // Elaps: 0.0000440100 sec for array of size(100) <- bubble sort
    // Elaps: 0.0000217700 sec for array of size(100) <- selection sort
    // Elaps: 0.0000102300 sec for array of size(100) <- insertion sort
    // Elaps: 0.0000085400 sec for array of size(100) <- shell sort
    //
    // ./build/part2_1 1000
    // Elaps: 0.0028549330 sec for array of size(1000) <- bubble sort
    // Elaps: 0.0017004120 sec for array of size(1000) <- selection sort
    // Elaps: 0.0008545710 sec for array of size(1000) <- insertion sort
    // Elaps: 0.0001122700 sec for array of size(1000) <- shell sort
    //
    // ./build/part2_1 10000
    // Elaps: 0.2743657340 sec for array of size(10000) <- bubble sort
    // Elaps: 0.1704587260 sec for array of size(10000) <- selection sort
    // Elaps: 0.0806659920 sec for array of size(10000) <- insertion sort
    // Elaps: 0.0016299420 sec for array of size(10000) <- shell sort
    //
    // ./build/part2_1 200000
    // Elaps: 138.7085848750 sec for array of size(200000) <- bubble sort
    // Elaps: 62.4383004490 sec for array of size(200000) <- selection sort
    // Elaps: 31.5358536920 sec for array of size(200000) <- insertion sort
    // Elaps: 0.0516538160 sec for array of size(200000) <- shell sort
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

        // 看了实际跑出来的结果终于明白【算法】这本书中为啥不提到它了
        TimeStats ts1{};
        BubbleSort<int>::sort(arr);
        std::cout << "Elaps: " << ts1.elapsSec() << " sec for array of size(" << N << ") <- bubble sort\n";
        
        TimeStats ts2{};
        SelectionSort<int>::sort(arrCpy1);
        std::cout << "Elaps: " << ts2.elapsSec() << " sec for array of size(" << N << ") <- selection sort\n";

        TimeStats ts3{};
        InsertionSort<int>::sort(arrCpy2);
        std::cout << "Elaps: " << ts3.elapsSec() << " sec for array of size(" << N << ") <- insertion sort\n";

        TimeStats ts4{};
        ShellSort<int>::sort(arrCpy3);
        std::cout << "Elaps: " << ts4.elapsSec() << " sec for array of size(" << N << ") <- shell sort\n";
        
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

    // flag 优化版本，数据量变大时有些效果 AMD 7950x
    // ./build/part2_1 10
    // Elaps: 0.0000156900 sec for array of size(10) <- bubble sort (use flag)
    // Elaps: 0.0000007300 sec for array of size(10) <- bubble sort (normal)
    //
    // ./build/part2_1 100
    // Elaps: 0.0000413300 sec for array of size(100) <- bubble sort (use flag)
    // Elaps: 0.0000372010 sec for array of size(100) <- bubble sort (normal)
    //
    // ./build/part2_1 1000
    // Elaps: 0.0028796350 sec for array of size(1000) <- bubble sort (use flag)
    // Elaps: 0.0034831180 sec for array of size(1000) <- bubble sort (normal)
    //
    // ./build/part2_1 10000
    // Elaps: 0.2815344430 sec for array of size(10000) <- bubble sort (use flag)
    // Elaps: 0.3384021260 sec for array of size(10000) <- bubble sort (normal)
    static void testBubble(int argc, char* argv[]) {
        if (argc < 2)
            return;

        int N = std::stoi(argv[1]);
        std::vector<int> arr{};
        Uniform<int> uni{ { 0, N } };
        uni(arr, N);

        std::vector<int> arrCpy1{ arr.cbegin(), arr.cend() };

        std::cout.flags(std::ios::fixed);
        std::cout.precision(10);

        TimeStats ts1{};
        BubbleSort<int, true>::sort(arr);
        std::cout << "Elaps: " << ts1.elapsSec() << " sec for array of size(" << N << ") <- bubble sort (use flag)\n";
        
        TimeStats ts2{};
        BubbleSort<int, false>::sort(arrCpy1);
        std::cout << "Elaps: " << ts2.elapsSec() << " sec for array of size(" << N << ") <- bubble sort (normal)\n";

        // check results
        std::cout << _isSorted(arr) << '\n';
        std::cout << _isSorted(arrCpy1) << '\n';

        // show results in small size
        if (N <= 20) {
            _show(arr);
            _show(arrCpy1);
        }
    }

    // 数据量大时，移动元素的版本，明显优于交换元素的版本 AMD 7950x
    // ./build/part2_1 10
    // Elaps: 0.0000125600 sec for array of size(10) <- insertion sort (use move)
    // Elaps: 0.0000005400 sec for array of size(10) <- insertion sort (use swap)
    //
    // ./build/part2_1 100
    // Elaps: 0.0000227500 sec for array of size(100) <- insertion sort (use move)
    // Elaps: 0.0000207900 sec for array of size(100) <- insertion sort (use swap)
    //
    // ./build/part2_1 1000
    // Elaps: 0.0008702530 sec for array of size(1000) <- insertion sort (use move)
    // Elaps: 0.0020485870 sec for array of size(1000) <- insertion sort (use swap)
    //
    // ./build/part2_1 10000
    // Elaps: 0.0815798790 sec for array of size(10000) <- insertion sort (use move)
    // Elaps: 0.2070244780 sec for array of size(10000) <- insertion sort (use swap)
    static void testInsertion(int argc, char* argv[]) {
        if (argc < 2)
            return;

        int N = std::stoi(argv[1]);
        std::vector<int> arr{};
        Uniform<int> uni{ { 0, N } };
        uni(arr, N);

        std::vector<int> arrCpy1{ arr.cbegin(), arr.cend() };

        std::cout.flags(std::ios::fixed);
        std::cout.precision(10);

        TimeStats ts1{};
        InsertionSort<int, true>::sort(arr);
        std::cout << "Elaps: " << ts1.elapsSec() << " sec for array of size(" << N << ") <- insertion sort (use move)\n";
        
        TimeStats ts2{};
        InsertionSort<int, false>::sort(arrCpy1);
        std::cout << "Elaps: " << ts2.elapsSec() << " sec for array of size(" << N << ") <- insertion sort (use swap)\n";

        // check results
        std::cout << _isSorted(arr) << '\n';
        std::cout << _isSorted(arrCpy1) << '\n';

        // show results in small size
        if (N <= 20) {
            _show(arr);
            _show(arrCpy1);
        }
    }

    // 移动元素的版本，明显优于交换元素的版本 AMD 7950x
    // ./build/part2_1 10   
    // Elaps: 0.0000142100 sec for array of size(10) <- shell sort (use swap)
    // Elaps: 0.0000016900 sec for array of size(10) <- shell sort (use move)
    //
    // ./build/part2_1 100
    // Elaps: 0.0000222000 sec for array of size(100) <- shell sort (use swap)
    // Elaps: 0.0000078310 sec for array of size(100) <- shell sort (use move)
    //
    // ./build/part2_1 1000
    // Elaps: 0.0001469900 sec for array of size(1000) <- shell sort (use swap)
    // Elaps: 0.0001111900 sec for array of size(1000) <- shell sort (use move)
    //
    // ./build/part2_1 10000
    // Elaps: 0.0022127850 sec for array of size(10000) <- shell sort (use swap)
    // Elaps: 0.0016861840 sec for array of size(10000) <- shell sort (use move)
    static void testShell(int argc, char* argv[]) {
        if (argc < 2)
            return;

        int N = std::stoi(argv[1]);
        std::vector<int> arr{};
        Uniform<int> uni{ { 0, N } };
        uni(arr, N);

        std::vector<int> arrCpy1{ arr.cbegin(), arr.cend() };

        std::cout.flags(std::ios::fixed);
        std::cout.precision(10);

        TimeStats ts1{};
        ShellSort<int, false>::sort(arr);
        std::cout << "Elaps: " << ts1.elapsSec() << " sec for array of size(" << N << ") <- shell sort (use swap)\n";
        
        TimeStats ts2{};
        ShellSort<int, true>::sort(arrCpy1);
        std::cout << "Elaps: " << ts2.elapsSec() << " sec for array of size(" << N << ") <- shell sort (use move)\n";

        // check results
        std::cout << _isSorted(arr) << '\n';
        std::cout << _isSorted(arrCpy1) << '\n';

        // show results in small size
        if (N <= 20) {
            _show(arr);
            _show(arrCpy1);
        }
    }
};


int main(int argc, char* argv[]) {
    // Comparator::testBubble(argc, argv);

    // Comparator::testInsertion(argc, argv);

    // Comparator::testShell(argc, argv);

    Comparator::test(argc, argv);

    return 0;
}