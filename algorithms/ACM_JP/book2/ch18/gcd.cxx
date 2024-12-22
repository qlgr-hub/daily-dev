#include <iostream>
#include <optional>

int gcd_r(int x, int y) {
    return y ? gcd_r(y, x % y) : x;
}

int gcd(int x, int y) {
    if (x < y)
        std::swap(x, y);

    while (y > 0) {
        int r = x % y;
        x = y;
        y = r;
    }

    return x;
}

int main() {
    int a, b;
    std::cin >> a >> b;
    std::cout << gcd_r(a, b) << '\n';
    std::cout << gcd(a, b) << '\n';
    return 0;
}
