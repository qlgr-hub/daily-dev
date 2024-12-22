#include <iostream>
#include <map>
#include <string>
#include <utility>

void print(const std::map<std::string, int>& T) {
    std::cout << T.size() << ": ";
    for (const auto& [k, v] : T) {
        std::cout << k << " --> " << v << ", ";
    }
    std::cout << '\n';
}

int main() {
    std::map<std::string, int> T;

    T["red"] = 32;
    T["blue"] = 688;
    T["yellow"] = 122;

    T["blue"] += 312;

    print(T);

    T.insert(std::make_pair("zebra", 101010));
    T.insert(std::make_pair("white", 0));
    T.erase("yellow");

    print(T);

    const auto& [k, v] = *T.find("red");
    std::cout << k << " --> " << v << "\n";

    return 0;
}