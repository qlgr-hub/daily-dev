#include <iostream>
#include <set>

void print(const std::set<int>& S) {
    std::cout << S.size() << ": ";
    for (auto v : S) {
        std::cout << v << ' ';
    }
    std::cout << '\n';
}

int main() {
    std::set<int> S;

    S.insert(8);
    S.insert(1);
    S.insert(7);
    S.insert(4);
    S.insert(8);
    S.insert(4);

    print(S);

    S.erase(7);
    print(S);

    S.insert(2);
    print(S);

    if (S.find(10) == S.end())
        std::cout << "not found.\n";

    return 0;
}