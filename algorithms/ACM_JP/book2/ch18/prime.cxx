#include <iostream>


bool isPrime(int x) {
    if (x < 2)
        return false;

    if (x == 2)
        return true;

    if ( (x % 2) == 0 ) 
        return false;

    bool p = true;
    for (int i = 3; i * i <= x; ++i ) {
        if ( (x % i) == 0 ) {
            p = false;
            break;
        }
    }
    return p;
}

int main() {
    int n;
    std::cin >> n;

    int cnt = 0;
    while (n--) {
        int x;
        std::cin >> x;
        if (isPrime(x))
            ++cnt;
    }

    std::cout << cnt << '\n';
    return 0;
}
