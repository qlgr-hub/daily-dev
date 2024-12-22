#include <array>
#include <memory>
#include <iostream>
#include "../common/alg_counter.h"
#include "../common/alg_utils.h"


class Flips {
public:
    static void test(int argc, char* argv[]) {
        if (argc < 2)
            return;

        Counter heads{ "heads" };
        Counter tails{ "tails" };
        Bernoulli br{ 0.5 };

        int T{ std::stoi(argv[1]) };
        for (int t{ 0 }; t < T; ++t) {
            if (br()) {
                heads.increment();
            }
            else {
                tails.increment();
            }
        }

        std::cout << heads << '\n';
        std::cout << tails << '\n';

        int d = heads.tally() - tails.tally();
        std::cout << "delta: " << d << '\n';
    }
};


class FlipsMax {
public:
    static void test(int argc, char* argv[]) {
        if (argc < 2)
            return;

        Counter heads{ "heads" };
        Counter tails{ "tails" };
        Bernoulli br{ 0.5 };

        int T{ std::stoi(argv[1]) };
        for (int t{ 0 }; t < T; ++t) {
            if (br()) {
                heads.increment();
            }
            else {
                tails.increment();
            }
        }

        if (heads.tally() == tails.tally()) {
            std::cout << "Tie\n";
        }
        else {
            auto maxOne{ std::max(heads, tails) };
            std::cout << maxOne << " wins\n"; 
        }
    }
};

class Rolls {
    static constexpr int SIDES{ 6 };
public:
    static void test(int argc, char* argv[]) {
        if (argc < 2)
            return;

        std::array<std::unique_ptr<Counter>, SIDES + 1> rolls{ nullptr };
        for (int i{ 1 }; i <= SIDES; ++i) {
            rolls[i] = std::unique_ptr<Counter>(new Counter(std::to_string(i) + "'s"));
        }

        Uniform<int> ud{ {1, SIDES} };
        int T{ std::stoi(argv[1]) };
        for (int t{ 0 }; t < T; ++t) {
            auto result = ud();
            rolls[result]->increment();
        }

        for (int i{ 1 }; i <= SIDES; ++i) {
            std::cout << *rolls[i].get() << '\n';
        }
    }
};



int main(int argc, char* argv[]) {
    // Flips::test(argc, argv);

    // FlipsMax::test(argc, argv);

    Rolls::test(argc, argv);

    return 0;
}
