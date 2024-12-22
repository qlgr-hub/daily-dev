#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <array>
#include <map>


class HashTable {
    static const int M = 1046527;

    std::array<std::string, M> H;

    int getChar(char ch) {
        int ret = 0;
        switch (ch) {
        case 'A': ret = 1; break;
        case 'C': ret = 2; break;
        case 'G': ret = 3; break;
        case 'T': ret = 4; break;
        default: break;
        }
        return ret;
    }

    int64_t getKey(std::string_view str) {
        int sum = 0;
        int p = 1;
        for (auto c : str) {
            sum += p * getChar(c);
            p *= 5;
        }
        return sum;
    }

    int h1(int key) {
        return key % M;
    }

    int h2(int key) {
        return 1 + (key % (M - 1));
    }

public:
    HashTable() {

    }

    bool find(std::string_view str) {
        int64_t key = getKey(str);

        bool ret = false;
        for (int i = 0; ; ++i) {
            int h = (h1(key) + i * h2(key)) % M;
            if (H[h].empty())
                break;
            else if (H[h] == str) {
                ret = true;
                break;
            }
        }

        return ret;
    }

    bool insert(std::string_view str) {
        int64_t key = getKey(str);

        bool ret = false;
        for (int i = 0; ; ++i) {
            int h = (h1(key) + i * h2(key)) % M;
            if (H[h] == str) {
                ret = true;
                break;
            }
            else if (H[h].empty()) {
                H[h] = str;
                ret = true;
                break;
            }
        }

        return ret;
    }
};

HashTable HT;

int main() {
    int n = 0;
    std::cin >> n;

    for (int i = 0; i < n; ++i) {
        std::string command;
        std::string str;
        std::cin >> command >> str;

        if (command[0] == 'i') {
            HT.insert(str);
        }
        else {
            if (HT.find(str)) {
                std::cout << "yes\n";
            }
            else {
                std::cout << "no\n";
            }
        }
    }

    return 0;
}
