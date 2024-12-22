#include "bstree.h"

int main() {
    int n;
    std::cin >> n;

    BSTree<int> T;
    for (int i = 0; i < n; ++i) {
        std::string com;
        std::cin >> com;
        if (com == "insert") {
            int key;
            std::cin >> key;
            T.insert(key);
        }
        else if (com == "print") {
            T.inorder();
            T.preorder();
        }
        else if (com == "find") {
            int key;
            std::cin >> key;
            if (T.find(key))
                std::cout << "yes\n";
            else
                std::cout << "no\n";
        }
        else if (com == "delete") {
            int key;
            std::cin >> key;
            T.del(key);
        }
    }

    return 0;
}