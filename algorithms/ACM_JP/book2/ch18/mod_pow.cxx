#include <cstdint>
#include <iostream>


uint64_t mod_pow(uint64_t m, uint64_t n, uint64_t M) {
    uint64_t res = 1;
    while (n > 0) {
        if (n & 1)
            res = (res * m) % M;
        m = (m * m) % M;
        n >>= 1;
    }
    return res;
}

static constexpr uint64_t M_ = 1000000007;

int main() {
    uint64_t m, n;
    std::cin >> m >> n;
    std::cout << mod_pow(m, n, M_) << '\n';
    return 0;
}
