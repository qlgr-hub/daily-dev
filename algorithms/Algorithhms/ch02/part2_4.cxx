#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <stack>
#include <vector>
#include "../common/alg_utils.h"
#include "../common/alg_priorityqueue.h"
#include "../common/alg_highersorting.h"
#include "../common/alg_elementarysorting.h"


struct Transaction {
    std::string _name;
    std::string _date;
    double _magic{ 0.0 };

    Transaction() = default;

    Transaction(const std::string& str) {
        std::istringstream istr{ str };
        istr >> _name >> _date >> _magic;
    }
};

struct TransLt {
    bool operator()(const Transaction* lhs, const Transaction* rhs) {
        return lhs->_magic < rhs->_magic;
    }
};

struct TransGt {
    bool operator()(const Transaction* lhs, const Transaction* rhs) {
        return lhs->_magic > rhs->_magic;
    }
};

static std::ostream& operator<<(std::ostream& out, const Transaction& obj) {
    out.flags(std::ios::fixed);
    out.precision(2);
    out << std::setw(10) << std::left << obj._name << ' ' << obj._date << ' ' << obj._magic;
    return out;
}


class TopM {
    template <typename MinPQ>
    static void _collectTopM(const std::vector<Transaction*>& trans, int M, std::vector<Transaction*>& topM) {
        MinPQ pq{ static_cast<int>(trans.size()) };
        for (auto t : trans) {
            pq.insert(t);
            if (pq.size() > M) {
                pq.delTop();
            }
        }

        std::stack<Transaction*> stack;
        while (!pq.empty()) {
            stack.push(pq.delTop());
        }

        while (!stack.empty()) {
            topM.push_back(stack.top());
            stack.pop();
        }
    }

    static void _showTopM(const std::vector<Transaction*>& topM) {
        for (auto t : topM) {
            std::cout << *t << '\n';
        }
    }

public:
    // tinyBatch.txt 测试数据中找出 top 5 AMD 7950x 上运行结果（数据量少的时候插入排序性能好不是吹的）
    // ./build/part2_4 5 < tinyBatch.txt 
    // Elaps: 0.0000067800sec <- resizing array priority queue ver1   // 用 selection 选取 top 版本
    // Elaps: 0.0000017000sec <- resizing array priority queue ver2   // 用 insertion 插入元素版本
    // Elaps: 0.0000025600sec <- heap priority queue                  // 二叉堆的版本
    static void test(int argc, char* argv[]) {
        if (argc < 2)
            return;

        int M{ std::stoi(argv[1]) };
        std::vector<Transaction*> trans;
        std::string str;
        while (std::getline(std::cin, str)) {
            trans.push_back(new Transaction{ str });
        }

        using MinPQ1 = ResizingArrayPQ<Transaction*, TransGt, true>;
        using MinPQ2 = ResizingArrayPQ<Transaction*, TransGt, false>;
        using MinPQ3 = HeapPQ<Transaction*, TransGt>;
        
        std::vector<Transaction*> topM1{};
        std::vector<Transaction*> topM2{};
        std::vector<Transaction*> topM3{};

        std::cout.flags(std::ios::fixed);
        std::cout.precision(10);

        TimeStats ts1;
        _collectTopM<MinPQ1>(trans, M, topM1);
        std::cout << "Elaps: " << ts1.elapsSec() << "sec <- resizing array priority queue ver1\n";

        TimeStats ts2;
        _collectTopM<MinPQ2>(trans, M, topM1);
        std::cout << "Elaps: " << ts2.elapsSec() << "sec <- resizing array priority queue ver2\n";

        TimeStats ts3;
        _collectTopM<MinPQ3>(trans, M, topM1);
        std::cout << "Elaps: " << ts3.elapsSec() << "sec <- heap priority queue\n";

        _showTopM(topM1);
        std::cout << '\n';
        _showTopM(topM2);
        std::cout << '\n';
        _showTopM(topM3);

        for (auto& t : trans) {
            delete t;
            t = nullptr;
        }
    }
};


struct StrGt {
    bool operator()(const std::string& lhs, const std::string& rhs) {
        return lhs > rhs;
    }
};

using IndexMinPQ = IndexPQ<std::string, StrGt>;


class Multiway {
public:
    static void merge(int argc, char* argv[]) {
        int streamCnt{ argc - 1 };
        IndexMinPQ pq{ streamCnt };
        std::vector<std::ifstream> streams( streamCnt );
        for (int i{ 0 }; i < streams.size(); ++i) {
            streams[i].open(argv[i + 1]);

            std::string str;
            if (std::getline(streams[i], str, ' '))
                pq.insert(i, str);
        }
    
        while (!pq.empty()) {
            std::cout << pq.top() << ' ';
            
            auto i{ pq.delTop() };

            std::string str;
            if (std::getline(streams[i], str, ' '))
                pq.insert(i, str);
        }
        std::cout << '\n';
    }
};


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
    // AMD 7950x （快排确实快，希尔排序表现还是不错的）
    // ./build/part2_4 10    
    // Elaps: 0.0000115600 sec for array of size(10) <- insertion sort
    // Elaps: 0.0000017100 sec for array of size(10) <- shell sort
    // Elaps: 0.0000010700 sec for array of size(10) <- merge sort
    // Elaps: 0.0000004800 sec for array of size(10) <- quick sort
    // Elaps: 0.0000007300 sec for array of size(10) <- heap sort
    // Elaps: 0.0000009200 sec for array of size(10) <- std::sort
    //
    // ./build/part2_4 100
    // Elaps: 0.0000233600 sec for array of size(100) <- insertion sort
    // Elaps: 0.0000081200 sec for array of size(100) <- shell sort
    // Elaps: 0.0000092400 sec for array of size(100) <- merge sort
    // Elaps: 0.0000056700 sec for array of size(100) <- quick sort
    // Elaps: 0.0000108800 sec for array of size(100) <- heap sort
    // Elaps: 0.0000088900 sec for array of size(100) <- std::sort
    //
    // ./build/part2_4 1000
    // Elaps: 0.0009974710 sec for array of size(1000) <- insertion sort
    // Elaps: 0.0001101010 sec for array of size(1000) <- shell sort
    // Elaps: 0.0001280800 sec for array of size(1000) <- merge sort
    // Elaps: 0.0000758400 sec for array of size(1000) <- quick sort
    // Elaps: 0.0001615600 sec for array of size(1000) <- heap sort
    // Elaps: 0.0001268900 sec for array of size(1000) <- std::sort
    //
    // ./build/part2_4 10000
    // Elaps: 0.0926717950 sec for array of size(10000) <- insertion sort
    // Elaps: 0.0015616910 sec for array of size(10000) <- shell sort
    // Elaps: 0.0016179920 sec for array of size(10000) <- merge sort
    // Elaps: 0.0009830510 sec for array of size(10000) <- quick sort
    // Elaps: 0.0021897230 sec for array of size(10000) <- heap sort
    // Elaps: 0.0014799120 sec for array of size(10000) <- std::sort
    //
    // ./build/part2_4 200000
    // Elaps: 35.2185330370 sec for array of size(200000) <- insertion sort
    // Elaps: 0.0516033380 sec for array of size(200000) <- shell sort
    // Elaps: 0.0402616350 sec for array of size(200000) <- merge sort
    // Elaps: 0.0238259270 sec for array of size(200000) <- quick sort
    // Elaps: 0.0557909430 sec for array of size(200000) <- heap sort
    // Elaps: 0.0360330210 sec for array of size(200000) <- std::sort
    static void test(int argc, char* argv[]) {
        if (argc < 2)
            return;

        int N{ std::stoi(argv[1]) };
        std::vector<int> arr{};
        Uniform<int> uni{ {0, N} };
        uni(arr, N);
        
        std::vector<int> arrCpy1{ arr.cbegin(), arr.cend() };
        std::vector<int> arrCpy2{ arr.cbegin(), arr.cend() };
        std::vector<int> arrCpy3{ arr.cbegin(), arr.cend() };
        std::vector<int> arrCpy4{ arr.cbegin(), arr.cend() };
        std::vector<int> arrCpy5{ arr.cbegin(), arr.cend() };

        std::cout.flags(std::ios::fixed);
        std::cout.precision(10);

        TimeStats ts1{};
        InsertionSort<int>::sort(arr);
        std::cout << "Elaps: " << ts1.elapsSec() << " sec for array of size(" << N << ") <- insertion sort\n";

        TimeStats ts2{};
        ShellSort<int>::sort(arrCpy1);
        std::cout << "Elaps: " << ts2.elapsSec() << " sec for array of size(" << N << ") <- shell sort\n";

        TimeStats ts3{};
        MergeSort<int>::sort(arrCpy2);
        std::cout << "Elaps: " << ts3.elapsSec() << " sec for array of size(" << N << ") <- merge sort\n";

        TimeStats ts4{};
        QuickSort<int>::sort(arrCpy3);
        std::cout << "Elaps: " << ts4.elapsSec() << " sec for array of size(" << N << ") <- quick sort\n";

        TimeStats ts5{};
        HeapSort<int>::sort(arrCpy4);
        std::cout << "Elaps: " << ts5.elapsSec() << " sec for array of size(" << N << ") <- heap sort\n";

        TimeStats ts6{};
        std::sort(arrCpy5.begin(), arrCpy5.end());
        std::cout << "Elaps: " << ts6.elapsSec() << " sec for array of size(" << N << ") <- std::sort\n";

        std::cout << _isSorted(arr) << '\n';
        std::cout << _isSorted(arrCpy1) << '\n';
        std::cout << _isSorted(arrCpy2) << '\n';
        std::cout << _isSorted(arrCpy3) << '\n';
        std::cout << _isSorted(arrCpy4) << '\n';
        std::cout << _isSorted(arrCpy5) << '\n';

        if (N <= 20) {
            _show(arr);
            _show(arrCpy1);
            _show(arrCpy2);
            _show(arrCpy3);
            _show(arrCpy4);
            _show(arrCpy5);
        }
    }

    // AMD 7950x heapify 的版本数据量小的时候表现很好，随着数据量的增加被只用下沉（sink）版本的反超了
    // heapify 应该时上浮（swim）下沉（sink）包含到一起了
    // ./build/part2_4 10    
    // Elaps: 0.0000142400 sec for array of size(10) <- heap sort (just sink)
    // Elaps: 0.0000008800 sec for array of size(10) <- heap sort (heapify)
    //
    // ./build/part2_4 100
    // Elaps: 0.0000235300 sec for array of size(100) <- heap sort (just sink)
    // Elaps: 0.0000120000 sec for array of size(100) <- heap sort (heapify)
    //
    // ./build/part2_4 1000
    // Elaps: 0.0001764500 sec for array of size(1000) <- heap sort (just sink)
    // Elaps: 0.0001755300 sec for array of size(1000) <- heap sort (heapify)
    //
    // ./build/part2_4 10000
    // Elaps: 0.0022515430 sec for array of size(10000) <- heap sort (just sink)
    // Elaps: 0.0023775530 sec for array of size(10000) <- heap sort (heapify)
    //
    // ./build/part2_4 200000
    // Elaps: 0.0603126830 sec for array of size(200000) <- heap sort (just sink)
    // Elaps: 0.0640790400 sec for array of size(200000) <- heap sort (heapify)
    static void testHeapSort(int argc, char* argv[]) {
        if (argc < 2)
            return;

        int N{ std::stoi(argv[1]) };
        std::vector<int> arr{};
        Uniform<int> uni{ {0, N} };
        uni(arr, N);
        
        std::vector<int> arrCpy1{ arr.cbegin(), arr.cend() };

        std::cout.flags(std::ios::fixed);
        std::cout.precision(10);

        TimeStats ts1{};
        HeapSort<int, true>::sort(arr);
        std::cout << "Elaps: " << ts1.elapsSec() << " sec for array of size(" << N << ") <- heap sort (just sink)\n";

        TimeStats ts2{};
        HeapSort<int, false>::sort(arrCpy1);
        std::cout << "Elaps: " << ts2.elapsSec() << " sec for array of size(" << N << ") <- heap sort (heapify)\n";

        std::cout << _isSorted(arr) << '\n';
        std::cout << _isSorted(arrCpy1) << '\n';

        if (N <= 20) {
            _show(arr);
            _show(arrCpy1);
        }
    } 
};


int main(int argc, char* argv[]) {
    // TopM::test(argc, argv);
    
    // Multiway::merge(argc, argv);

    // Comparator::testHeapSort(argc, argv);

    Comparator::test(argc, argv);

    return 0;
}