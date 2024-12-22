#include <iostream>
#include <utility>
#include <vector>
#include "../common/alg_unionfind.h"
#include "../common/alg_utils.h"

class UnionFindDemo {
    template <typename UF>
    static int _statsUF(const std::vector<std::pair<int, int>>& pairs) {
        const int N{ static_cast<int>(pairs.size()) };
        UF uf{ N };
        for (const auto [p, q] : pairs) {
            if (uf.connected(p, q)) {
                continue;
            }
            uf.unite(p, q);
        }

        return uf.count();
    }

public:
    // AMD 7950x 跑的结果
    // ./build/part1_5 < largeUF.txt 
    // Elaps: 3201.7261315230 sec, 79956 componenets <- union find (quick find)
    // Elaps: 357.7355207330 sec, 79956 componenets <- union find (quick union)
    // Elaps: 0.0889066760 sec, 79956 componenets <- union find (weighted quick union)
    //
    // ./build/part1_5 < mediumUF.txt 
    // Elaps: 0.0011507230 sec, 66 componenets <- union find (quick find)
    // Elaps: 0.0000463800 sec, 66 componenets <- union find (quick union)
    // Elaps: 0.0000417700 sec, 66 componenets <- union find (weighted quick union)
    //
    // ./build/part1_5 < tinyUF.txt 
    // Elaps: 0.0000072100 sec, 2 componenets <- union find (quick find)
    // Elaps: 0.0000011300 sec, 2 componenets <- union find (quick union)
    // Elaps: 0.0000011200 sec, 2 componenets <- union find (weighted quick union)
    static void test(int argc, char* argv[]) {
        int N{ 0 };
        std::cin >> N;
        std::vector<std::pair<int, int>> pairs(N);
        for (int i{ 0 }; i < N; ++i) {
            int p{ 0 };
            int q{ 0 };
            std::cin >> p >> q;
            pairs[i] = { p, q };
        }

        std::cout.flags(std::ios::fixed);
        std::cout.precision(10);

        TimeStats ts1{};
        auto count1{ _statsUF<QuickFindUF>(pairs) };
        std::cout << "Elaps: " << ts1.elapsSec() << " sec, " << count1 << " componenets <- union find (quick find)\n";

        TimeStats ts2{};
        auto count2{ _statsUF<QuickUnionUF>(pairs) };
        std::cout << "Elaps: " << ts2.elapsSec() << " sec, " << count2 << " componenets <- union find (quick union)\n";

        TimeStats ts3{};
        auto count3{ _statsUF<WeightedQuickUnionUF>(pairs) };
        std::cout << "Elaps: " << ts3.elapsSec() << " sec, " << count3 << " componenets <- union find (weighted quick union)\n";
    }
};


int main(int argc, char* argv[]) {
    UnionFindDemo::test(argc, argv);

    return 0;
}