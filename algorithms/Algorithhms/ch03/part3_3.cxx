#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <map>
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

    static std::pair<std::string, int> _count_stdmap(const std::vector<std::string>& contents) {
        std::map<std::string, int> st;
        for (const auto& word : contents) {
            if (st.count(word) == 0) {
                st[word] = 1;
            }
            else {
                ++st[word];
            }
        }

        std::string max{ " " };
        int maxCount{ 0 };
        st[max] = maxCount;

        for (const auto& [k, v] : st) {
            if (v > maxCount) {
                max = k;
                maxCount = v;
            }
        }

        return {max, maxCount};
    }
 
 public: 
    // AMD 7950x
    // 本例中衡量的性能是 插入 + 查找 的效率（至于其余的部分大家都一样所以可以作为比较）
    // 普通 BST 效率最好 可以猜测到测试数据直接构造出的普通 BST 平衡性就够好
    // 由于红黑树是牺牲一些插入、删除的效率来保证树的平衡 所以这里插入的性能反而拖累了它
    //
    // ./build/part3_3 1 < tinyTale.txt 
    // Elaps: 0.0000188300 sec, result: [it, 10] <- binary search tree symbol table.
    // Elaps: 0.0000164700 sec, result: [it, 10] <- red black binary search tree symbol table.
    // Elaps: 0.0000188400 sec, result: [it, 10] <- std::map.
    //
    // ./build/part3_3 8 < tale.txt 
    // Elaps: 0.0071898500 sec, result: [business, 122] <- binary search tree symbol table.
    // Elaps: 0.0075958690 sec, result: [business, 122] <- red black binary search tree symbol table.
    // Elaps: 0.0077843310 sec, result: [business, 122] <- std::map.
    //
    // ./build/part3_3 10 < leipzig100K.txt 
    // Elaps: 0.0908026460 sec, result: [government, 2549] <- binary search tree symbol table.
    // Elaps: 0.0948916710 sec, result: [government, 2549] <- red black binary search tree symbol table.
    // Elaps: 0.0994531780 sec, result: [government, 2549] <- std::map.
    //
    // ./build/part3_3 10 < leipzig1m.txt 
    // Elaps: 0.8258817630 sec, result: [government, 24763] <- binary search tree symbol table.
    // Elaps: 0.9267550020 sec, result: [government, 24763] <- red black binary search tree symbol table.
    // Elaps: 0.9987369980 sec, result: [government, 24763] <- std::map.
    //
    // 数据升序排序后在测试（相当于普通 BST 最坏场景，最大的数据没测太慢了）
    // ./build/part3_3 1 < tinyTale.txt 
    // Elaps: 0.0000336400 sec, result: [it, 10] <- binary search tree symbol table.
    // Elaps: 0.0000190600 sec, result: [it, 10] <- red black binary search tree symbol table.
    // Elaps: 0.0000200400 sec, result: [it, 10] <- std::map.
    //
    // ./build/part3_3 8 < tale.txt 
    // Elaps: 1.2328336900 sec, result: [business, 122] <- binary search tree symbol table.
    // Elaps: 0.0065757960 sec, result: [business, 122] <- red black binary search tree symbol table.
    // Elaps: 0.0080512980 sec, result: [business, 122] <- std::map.
    //
    // ./build/part3_3 10 < leipzig100K.txt 
    // Elaps: 105.6735794510 sec, result: [government, 2549] <- binary search tree symbol table.
    // Elaps: 0.0865232950 sec, result: [government, 2549] <- red black binary search tree symbol table.
    // Elaps: 0.1127634410 sec, result: [government, 2549] <- std::map.
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
        auto [max1, maxCnt1] = _count<BST<std::string, int>>(contents);
        std::cout << "Elaps: " << ts1.elapsSec() << " sec, result: [" << max1 << ", " << maxCnt1 << "] <- binary search tree symbol table.\n";

        TimeStats ts2{};
        auto [max2, maxCnt2] = _count<RedBlackBST<std::string, int>>(contents);
        std::cout << "Elaps: " << ts2.elapsSec() << " sec, result: [" << max2 << ", " << maxCnt2 << "] <- red black binary search tree symbol table.\n";

        TimeStats ts3{};
        auto [max3, maxCnt3] = _count_stdmap(contents);
        std::cout << "Elaps: " << ts3.elapsSec() << " sec, result: [" << max3 << ", " << maxCnt3 << "] <- std::map.\n";
    }
 };


int main(int argc, char* argv[]) {
    FrequencyCounter::test(argc, argv);

    return 0;
}
