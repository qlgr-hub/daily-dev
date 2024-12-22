#include <iostream>
#include <vector>

std::ostream& operator<<(std::ostream& out, const std::vector<double>& V) {
    for (const auto& v : V) {
        out << v << ' ';
    }
    out << '\n';
    return out;
}

int main() {
    std::vector<double> V;

    V.push_back(0.1);
    V.push_back(0.2);
    V.push_back(0.3);
    V[2] = 0.4;
    std::cout << V;

    V.insert(V.begin() + 2, 0.8);
    std::cout << V;

    V.erase(V.begin() + 1);
    std::cout << V;

    V.push_back(0.9);
    std::cout << V;

    return 0;
}
