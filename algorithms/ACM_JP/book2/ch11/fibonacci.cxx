#include <array>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <cstdint>
#include <string_view>

int64_t fibonacci(int n) {
    if (n == 0 || n == 1)
        return 1;
    return fibonacci(n - 2) + fibonacci(n - 1);
}

static std::array<int64_t, 50> DP{ 0 };
int64_t fibonacci_dp(int n) {
    if (DP[n] != 0)
        return DP[n];
    if (n == 0 || n == 1)
        return DP[n] = 1;
    return DP[n] = fibonacci_dp(n - 1) + fibonacci_dp(n - 2);
}

constexpr int64_t fibonacci_c(int n) {
    if (n == 0 || n == 1)
        return 1;
    return fibonacci_c(n - 2) + fibonacci_c(n - 1);
}

template <int N>
constexpr std::array<int64_t, N> fibonacci_c1(int n) {
    std::array<int64_t, N> ret;
    ret[0] = 1;
    ret[1] = 1;
    for (int i = 2; i < N; ++i)
        ret[i] = ret[i - 1] + ret[i - 2];
    return ret;
}

// 定义斐波那契函数模板
template<int N> struct Fibonacci { static const int value = Fibonacci<N-1>::value + Fibonacci<N-2>::value; };
template<> struct Fibonacci<0> { static const int value = 1; }; // 基本情况
template<> struct Fibonacci<1> { static const int value = 1; }; // 基本情况

void test(int n, std::function<int(int)> func, std::string_view func_name) {
    auto bgn = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) {
        std::cout << func(i) << ' ';
    }
    std::cout << '\n';
    auto end = std::chrono::steady_clock::now();
    auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end - bgn);
    std::cout << func_name << ": " << std::fixed << std::setw(15) << std::setprecision(12) << d.count() << " seconds\n";
}
 
 int main() {
    int n = 0;
    std::cin >> n;

    test(n, fibonacci, "fibonacci");
    test(n, fibonacci_dp, "fibonacci_dp");
    test(n, fibonacci_c, "fibonacci_c");

    auto bgn = std::chrono::steady_clock::now();
    for (auto c : fibonacci_c1<45>(n)) {
        std::cout << c << ' ';
    }
    std::cout << '\n';
    auto end = std::chrono::steady_clock::now();
    auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end - bgn);
    std::cout << "fibonacci_c1<45>" << ": " << std::fixed << std::setw(15) << std::setprecision(12) << d.count() << " seconds\n";

    std::cout << Fibonacci<44>::value << '\n';
    return 0;
 }
