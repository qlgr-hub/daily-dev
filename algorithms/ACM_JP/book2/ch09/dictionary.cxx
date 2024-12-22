#include <iostream>
#include <map>
#include <string>

int main() {
    int n;
    std::cin >> n;

    std::map<std::string, bool> D;
    for (int i = 0; i < n; ++i) {
        std::string com;
        std::string str;
        std::cin >> com >> str;

        if (com[0] == 'i')
            D[str] = true;
        else {
            if (D[str])
                std::cout << "yes\n";
            else
                std::cout << "no\n";
        } 
    }

    return 0;
}