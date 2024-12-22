#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include "../common/alg_utils.h"
#include "../common/alg_symboltable.h"


class FrequencyCounter {
    template <typename ST>
    static std::pair<std::string, int> _count(const std::vector<std::string>& contents) {
        ST st{ static_cast<int>(contents.size()) + 1 };
        for (const auto& word : contents) {
            auto res{ st.get(word) };
            if (!res.has_value()) {
                st.put(word, 1);
            }
            else {
                st.put(word, res.value() + 1);
            }
        }

        std::string max{ " " };
        int maxCount{ 0 };
        st.put(max, maxCount);

        for (const auto& k : st.keys()) {
            auto curCount{ st.get(k) };
            if (curCount.value() > maxCount) {
                max = k;
                maxCount = curCount.value();
            }
        }

        return {max, maxCount};
    }
 
 public: 
    // AMD 7950x
    // ./build/part3_2 1 < tinyTale.txt 
    // Elaps: 0.0000234000 sec, result: [it, 10] <- binary search symbol table.
    // Elaps: 0.0000130690 sec, result: [it, 10] <- binary search tree symbol table.
    //
    // ./build/part3_2 8 < tale.txt 
    // Elaps: 0.0786829230 sec, result: [business, 122] <- binary search symbol table.
    // Elaps: 0.0070218810 sec, result: [business, 122] <- binary search tree symbol table.
    //
    // ./build/part3_2 10 < leipzig100K.txt 
    // Elaps: 3.9343961600 sec, result: [government, 2549] <- binary search symbol table.
    // Elaps: 0.0841870800 sec, result: [government, 2549] <- binary search tree symbol table.
    //
    // ./build/part3_2 10 < leipzig1m.txt 
    // Elaps: 73.4467966270 sec, result: [government, 24763] <- binary search symbol table.
    // Elaps: 0.8116974170 sec, result: [government, 24763] <- binary search tree symbol table.
    //
    // 数据升序排序后在测试（相当于普通 BST 最坏场景，最大的数据没测太慢了）
    // 对有序的数据 BinarySearchST 的 put 明显效率变高了（这个是类插入排序的插入，再次印证了接近有序数的插入排序效率很高）
    // 普通 BST 则在有序数据上退化为了类线性查找
    // ./build/part3_2 1 < tinyTale.txt 
    // Elaps: 0.0000205800 sec, result: [it, 10] <- binary search symbol table.
    // Elaps: 0.0000273000 sec, result: [it, 10] <- binary search tree symbol table.
    //
    // ./build/part3_2 8 < tale.txt 
    // Elaps: 0.0062040920 sec, result: [business, 122] <- binary search symbol table.
    // Elaps: 1.1676486060 sec, result: [business, 122] <- binary search tree symbol table.
    //
    // ./build/part3_2 10 < leipzig100K.txt 
    // Elaps: 0.0817760800 sec, result: [government, 2549] <- binary search symbol table.
    // Elaps: 97.5808049220 sec, result: [government, 2549] <- binary search tree symbol table.
    static void test(int argc, char* argv[]) {
        if (argc < 2)
            return;

        int minLen{ std::stoi(argv[1]) };
        std::vector<std::string> contents;

        std::string str;
        while (std::cin >> str) {
            if (str.length() < minLen)
                continue;
            contents.emplace_back(str);
        }

        // 排序 为了查看普通 BST 的最坏场景
        // std::sort(contents.begin(), contents.end());

        std::cout.flags(std::ios::fixed);
        std::cout.precision(10);

        TimeStats ts1{};
        auto [max1, maxCnt1] = _count<BinarySearchST<std::string, int>>(contents);
        std::cout << "Elaps: " << ts1.elapsSec() << " sec, result: [" << max1 << ", " << maxCnt1 << "] <- binary search symbol table.\n";

        TimeStats ts2{};
        auto [max2, maxCnt2] = _count<BST<std::string, int>>(contents);
        std::cout << "Elaps: " << ts2.elapsSec() << " sec, result: [" << max2 << ", " << maxCnt2 << "] <- binary search tree symbol table.\n";
    }
 };


int main(int argc, char* argv[]) {
    FrequencyCounter::test(argc, argv);

    return 0;
}
