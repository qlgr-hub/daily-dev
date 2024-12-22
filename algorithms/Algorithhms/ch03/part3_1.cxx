#include <iostream>
#include <string>
#include <utility>
#include <vector>
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
    // AMD 7950x 最大的那个测试数据上，第一个版本基本不行等了两个多小时没完成
    // ./build/part3_1 1 < tinyTale.txt 
    // Elaps: 0.0000387100 sec, result: [of, 10] <- sequential search symbol table.
    // Elaps: 0.0000162900 sec, result: [it, 10] <- binary search symbol table.
    //
    // ./build/part3_1 8 < tale.txt 
    // Elaps: 1.4090733950 sec, result: [business, 122] <- sequential search symbol table.
    // Elaps: 0.0752077670 sec, result: [business, 122] <- binary search symbol table.
    //
    // ./build/part3_1 10 < leipzig100K.txt 
    // Elaps: 156.2965331600 sec, result: [government, 2549] <- sequential search symbol table.
    // Elaps: 3.7913938100 sec, result: [government, 2549] <- binary search symbol table.
    //
    // ./build/part3_1 10 < leipzig1m.txt   
    // Elaps: NaN sec, result: [xxx, xxx] <- sequential search symbol table.
    // Elaps: 73.6888784160 sec, result: [government, 24763] <- binary search symbol table.
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

        std::cout.flags(std::ios::fixed);
        std::cout.precision(10);

        TimeStats ts1{};
        auto [max1, maxCnt1] = _count<SequentialSearchST<std::string, int>>(contents);
        std::cout << "Elaps: " << ts1.elapsSec() << " sec, result: [" << max1 << ", " << maxCnt1 << "] <- sequential search symbol table.\n";

        TimeStats ts2{};
        auto [max2, maxCnt2] = _count<BinarySearchST<std::string, int>>(contents);
        std::cout << "Elaps: " << ts2.elapsSec() << " sec, result: [" << max2 << ", " << maxCnt2 << "] <- binary search symbol table.\n";
    }
 };


int main(int argc, char* argv[]) {
    FrequencyCounter::test(argc, argv);

    return 0;
}
