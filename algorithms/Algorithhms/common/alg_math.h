#pragma once

#include <algorithm>


// 参考自：挑战程序设计竞赛 2（算法和数据结构）
template <typename UIT>
bool isprime(UIT x) {
    if (x == 2)
        return true;
    else if ( x < 2 || ((x % 2) == 0) )
        return false;

    for (UIT i{ 3 }; i * i <= x; i += 2) {
        if ( (x % i) == 0 )
            return false;
    }

    return true;
}


// 参考自：挑战程序设计竞赛 2（算法和数据结构）
template <typename UIT>
int gcd(UIT p, UIT q) {
    if (q == UIT{})
        return p;

    if (p < q)
        std::swap(p, q);

    while (q > 0) {
        auto r{ p % q };
        p = q;
        q = r;
    }

    return p;
}


template <typename UIT>
int gcd_r(UIT p, UIT q) {
    if (q == UIT{})
        return p;

    auto r{ p % q };
    return gcd_r(q, r);
}


