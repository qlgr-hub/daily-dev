#include <iostream>
#include <vector>


void print(const std::vector<int>& v) {
    for (auto it = v.begin(); it != v.end(); ++it) {
        std::cout << *it;
    }
    std::cout << '\n';
}

int main() {
    const int N = 4;
    std::vector<int> V;

    for (int i = 0; i < N; ++i) {
        int x;
        std::cin >> x;
        V.push_back(x);
    }

    print(V);

    auto it = V.begin();
    *it = 3;
    it++;
    (*it)++;

    print(V);

    return 0;
}