#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iomanip>
#include <ios>
#include <iostream>
#include <numbers>
#include <string>
#include <utility>
#include <vector>
#include <cassert>

// 求最大公约数算法
static int gcd(int p, int q) {
    if (q == 0)
        return p;

    auto r = p % q;
    return gcd(q, r);
}

static std::pair<int, int> binarySearch(std::vector<int>& arr, int key) {
    int lo{ 0 };
    int hi{ static_cast<int>(arr.size()) - 1 };
    
    int res{ -1 };
    int loop_cnt{ 0 };
    while (lo <= hi) {
        ++loop_cnt;
        auto mid = lo + ((hi - lo) >> 1);
        if (key < arr[mid]) {
            hi = mid - 1;
        }
        else if (key > arr[mid]) {
            lo = mid + 1;
        }
        else {
            res = mid;
            break;
        }
    }

    return { res, loop_cnt };
}

std::string exR1(int n) {
    if (n <= 0) 
        return std::string("");
    return exR1(n - 3) + std::to_string(n) + exR1(n - 2) + std::to_string(n);
};

std::string exR2(int n) {
    auto s = exR2(n - 3) + std::to_string(n) + exR2(n - 2) + std::to_string(n);
    if (n <= 0) 
        return std::string("");
    return s;
};


// 来自 文言一心
// 泰勒级数展开逼近ln(1+x)，其中x接近0  
double my_ln_approx(double x, int terms) {  
    double result = 0.0;  
    double term = x;  
    bool negative = false;  
      
    for (int i = 0; i < terms; ++i) {  
        if (negative) {  
            result -= term;  
        } else {  
            result += term;  
        }  
        term *= -x / (i + 1);  
        negative = !negative;  
    }  
      
    return result;  
}  
  
// 使用上述函数来计算ln(x)，其中x可以是任意正数  
double my_ln(double x, int terms) {  
    if (x <= 0) {  
        throw std::invalid_argument("x must be greater than 0");  
    }  
      
    if (x == 1) {  
        return 0.0;  
    }  
      
    // 使用换底公式 ln(x) = ln(1 + (x - 1)) / ln(e)  
    // 但由于ln(e) = 1，所以直接计算ln(1 + (x - 1))即可  
    double y = x - 1;  
    return my_ln_approx(y, terms);  
}


int main() {
    // faster io setting
    // std::ios::sync_with_stdio(false);
    // std::cin.tie(nullptr);

    // std::vector<int> arr{ 10, 11, 12, 16, 18, 23, 29, 33, 48, 54, 57, 68, 77, 84, 98 };
    //
    // auto res = binarySearch(arr, 33);
    // std::cout << res.first << "-->" << res.second << '\n';
    // res = binarySearch(arr, 99);
    // std::cout << res.first << "-->" << res.second << '\n';


    // 练习
    // 1.1.1
    std::cout << (0 + 15) / 2 << '\n';                      // a
    std::cout << 2.0e-6 * 100000000.1 << '\n';              // b
    std::cout << (true && false || true && true) << '\n';   // c

    // 1.1.2
    auto a{ (1 + 2.236) / 2 };
    std::cout << typeid(a).name() << " => " << a << '\n';    // a
    auto b{ 1 + 2 + 3 + 4.0 };
    std::cout << typeid(b).name() << " => " << b << '\n';    // b
    auto c{ 4.1 >= 4 };
    std::cout << typeid(c).name() << " => " << c << '\n';    // c
    auto d{ std::to_string(1 + 2) + std::string("3")};
    std::cout << typeid(d).name() << " => " << d << '\n';    // d


    // 1.1.3
    // int cin1{ 0 };
    // int cin2{ 0 };
    // int cin3{ 0 };
    // std::cout << "input 3 number: ";
    // std::cin >> cin1 >> cin2 >> cin3;
    // if (cin1 == cin2 && cin1 == cin3) {
    //     std::cout << "equal\n";
    // }
    // else {
    //     std::cout << "not equal\n";
    // }


    // 1.1.4
    // a. if (a > b) then c = 0;         // 没有 then 这个关键字
    // b. if a > b { c = 0 };            // 判断条件语句需要用圆括号括起来
    // c. if (a > b) c = 0;              // ok
    // d. if (a > b) c = 0 else b = 0;   // c = 0 是单独的语句需要分号结尾


    // 1.1.5
    double x{ 0.00000000000001 };
    double y{ 0.99999999999999 };
    if (x > 0 && x < 1 && y > 0 && y < 1) {
        std::cout << "true\n";
    }
    else {
        std::cout << "false\n";
    }


    // 1.1.6
    int f{ 0 };
    int g{ 1 };
    for (int i{ 0 }; i <= 15; ++i) {
        std::cout << f << '\n';
        f = f + g;
        g = f - g;
    }
    std::cout << '\n';


    // 1.1.7
    // a
    double t{ 9.0 };
    while (std::fabs(t - 9.0/t) > .001) {
        t = (9.0/t + t) / 2.0;
    }
    std::cout << std::setprecision(5) << t << '\n';

    // b
    int sum{ 0 };
    for (int i{ 1 }; i < 1000; ++i) {
        for (int j{ 0 }; j < i; ++j) {
            ++sum;
        }
    }
    std::cout << sum << '\n';

    // c
    sum = 0;
    for (int i{ 1 }; i < 1000; i *= 2) {
        for (int j{ 0 }; j < 1000; ++j) {
            ++sum;
        }
    }
    std::cout << sum << '\n';


    // 1.1.8
    std::cout << 'b' << '\n';
    std::cout << 'b' + 'c' << '\n';
    std::cout << static_cast<char>('a' + 4) << '\n';


    // 1.1.9
    std::string s;
    constexpr int N{ 55 };
    for (int n{ N }; n > 0; n /= 2) {
        s = std::to_string(n % 2) + s;
    }
    std::cout << s << '\n';


    // 1.1.10
    // int aa[];
    // for (int i{ 0 }; i < 10; ++i)
    //     aa[i] = i * i;


    // 1.1.11
    std::array<std::array<bool, 4>, 4> state{
        {
            { true, false, false, true },
            { false, true, true, false },
            { true, false, true, false },
            { false, true, false, true },
        } 
    };

    std::cout << "  " << "1 2 3 4\n";
    int col{ 1 };
    for (const auto& row : state) {
        std::cout << col++ << ' ';
        for (auto s : row) {
            std::cout << (s ? '*' : ' ') << ' ';
        }
        std::cout << '\n';
    }


    // 1.1.12
    int aa1[10] = { 0 };
    for (int i{ 0 }; i < 10; ++i)
        aa1[i] = 9 - i;
    for (int i{ 0 }; i < 10; ++i)
        aa1[i] = aa1[aa1[i]];
    for (int i{ 0 }; i < 10; ++i)
        std::cout << aa1[i] << '\n';


    // 1.1.13
    std::array<std::array<int, 7>, 4> m1{
        {
            { 2, 3, 5, 2, 1, 6, 9 },
            { 3, 2, 6, 9, 4, 9, 7 },
            { 4, 5, 8, 0, 7, 2, 5 },
            { 1, 7, 4, 1, 3, 0, 4 },
        }
    };
    std::cout << '\n';
    for (const auto& row : m1) {
        for (auto e : row) {
            std::cout << e << ' ';
        }
        std::cout << '\n';
    }

    std::array<std::array<int, 4>, 7> m2{};
    for (int i{ 0 }; i < 7; ++i) {
        for (int j{ 0 }; j < 4; ++j) {
            m2[i][j] = m1[j][i];
        }
    }
    std::cout << '\n';
    for (const auto& row : m2) {
        for (auto e : row) {
            std::cout << e << ' ';
        }
        std::cout << '\n';
    }


    // 1.1.14
    auto lg = [](int N) {
        int res{ 0 };
        while (N > 1) {
            N /= 2;
            ++res;
        }
        return res;
    };
    std::cout << lg(8) << '\n';
    std::cout << lg(10) << '\n';


    // 1.1.15
    std::cout << '\n';
    auto histogram = [](std::array<int, 10>& arr, int M) {
        std::vector<int> res(M, 0);
        for (auto v : arr) {
            ++res[v];
        }
        return res;
    };
    srand((unsigned)time(nullptr));
    const int M = 8;
    std::array<int, 10> values{ 0 };
    for (auto& v : values) {
        v = rand() % (M - 1);
    }
    auto his = histogram(values, M);
    int acc{ 0 };
    for (auto v : his) {
        std::cout << v << ' ';
        acc += v;
    }
    std::cout << '\n';
    assert(acc == values.size());


    // 1.1.16
    std::cout << exR1(6) << '\n';
    

    // 1.1.17
    // std::cout << exR2(6) << '\n';


    // 1.1.18
    // a * b
    // std::function<int(int, int)> mystery = [&](int a, int b) {
    //     if (b == 0)
    //         return 0;
    //     if (b % 2 == 0)
    //         return mystery(a + a, b/2);
    //     return mystery(a + a, b/2) + a;
    // };
    // a ^ b
    std::function<int(int, int)> mystery = [&](int a, int b) {
        if (b == 0)
            return 1;
        if (b % 2 == 0)
            return mystery(a * a, b / 2);
        return mystery(a * a, b / 2) * a;
    };
    std::cout << mystery(2, 25) << '\n';
    std::cout << mystery(3, 11) << '\n';


    // 1.1.19
    // std::function<int64_t(int)> fib = [&](int N) -> int64_t {
    //     if (N == 0 || N == 1)
    //         return  N;
    //     return fib(N - 1) + fib(N - 2);
    // };
    // constexpr int MAX_N{ 100 };
    // std::function<int64_t(int)> fib = [&](int N) -> int64_t {
    //     static std::vector<int64_t> DP(MAX_N, 0);
    //     if (N == 0 || N == 1)
    //         return DP[N] = N;
    //     return DP[N] = DP[N - 1] + DP[N - 2];
    // };
    // for (int i{ 0 }; i < MAX_N; ++i) {
    //     std::cout << fib(i) << '\n';
    // }


    // 1.1.20
    auto lnNN = [](int n) {
        std::function<int64_t(int)> fac = [&](int N) -> int64_t {
            if (N == 0)
                return 1;
            return N * fac(N - 1);
        };

        int res{ 0 };
        auto nn = fac(n);
        return my_ln(nn, 10);
    };
    std::cout << "ln(5!) = " << lnNN(5) << '\n';


    std::cout << gcd(105, 24) << '\n';
    std::cout << gcd(gcd(1, 111), 111) << '\n';
    std::cout << gcd(gcd(1, 234), 567) << '\n';


    // 1.1.26
    int n1{ 7 };
    int n2{ 2 };
    int n3{ 9 };
    int tmp{ 0 };
    if (n1 > n2) {
        t = n1;
        n1 = n2;
        n2 = t;
    }
    if (n1 > n3) {
        t = n1;
        n1 = n3;
        n3 = t;
    }
    if (n2 > n3) {
        t = n2;
        n2 = n3;
        n3 = t;
    }
    std::cout << n1 << ' ' << n2 << ' ' << n3 << '\n';


    // 1.1.27
    static std::array<std::array<double, 101>, 101> DP;
    static std::array<double, 101> temp;
    temp.fill(-1.0);
    DP.fill(temp);
    std::function<double(int, int , double)> binomial = [&](int N, int k, double p) {
        if (N == 0 && k == 0)
            return 1.0;
        if (N < 1 || k < 0) 
            return 0.0;
        if (std::fabs(DP[N][k] - (-1.0)) < 0.001) {
            DP[N][k] = (1.0 - p) * binomial(N - 1, k, p) + p * binomial(N - 1, k - 1, p);
        }
        return DP[N][k];
    };
    std::cout << binomial(100, 50, 0.25) << '\n';

    return 0;
}
