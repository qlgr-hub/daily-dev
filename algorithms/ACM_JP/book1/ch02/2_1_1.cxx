#include <iostream>
#include <array>
#include <chrono>
#include <stdint.h>

int fact(int n) {
    if (n == 0) 
        return 1;

    return n * fact(n - 1);
}

int64_t fib(int n) {
    if (n <= 1)
        return n;

    return fib(n - 1) + fib(n - 2); 
}


const int MAX_N = 50;
int64_t fib_m(int n) {
    static std::array<int64_t, MAX_N> memo{ 0 };

    if (n <= 1)
        return n;

    if (memo[n] != 0)
        return memo[n];

    return memo[n] = fib_m(n - 1) + fib_m(n - 2);
}

int main() {

    for (int i = 0; i < 11; ++i) {
        std::cout << fact(i) << " ";
    }
    std::cout << "\n";

    for (int i = 0; i < 11; ++i) {
        std::cout << fib(i) << " ";
    }
    std::cout << "\n";

    // for (int i = 0; i < 11; ++i) {
    //     std::cout << fib_m(i) << " ";
    // }
    // std::cout << "\n";

    auto t1 = std::chrono::system_clock::now();
    std::cout << "1.1: " <<  fib(40) << " => ";
    auto t2 = std::chrono::system_clock::now();
    std::cout << (t2 - t1).count() << "\n";

    t1 = std::chrono::system_clock::now();
    std::cout << "1.2: " <<  fib_m(40) << " => ";
    t2 = std::chrono::system_clock::now();
    std::cout << (t2 - t1).count() << "\n";

    t1 = std::chrono::system_clock::now();
    std::cout << "2.1: " <<  fib(40) << " => ";
    t2 = std::chrono::system_clock::now();
    std::cout << (t2 - t1).count() << "\n";

    t1 = std::chrono::system_clock::now();
    std::cout << "2.2: " <<  fib_m(40) << " => ";
    t2 = std::chrono::system_clock::now();
    std::cout << (t2 - t1).count() << "\n";

    return 0;
}