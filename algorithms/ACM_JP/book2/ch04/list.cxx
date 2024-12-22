#include <iostream>
#include <list>
#include <ostream>
#include <utility>


template <typename T>
class DoubleLinkList {
    template <typename N>
    friend std::ostream& operator<<(std::ostream& out, const DoubleLinkList<N>& lst);

private:
    struct Node;

    struct Node {
        T key;
        Node *prev;
        Node *next;

        Node() : key{ T() }, prev{ nullptr }, next{ nullptr } {
        }

        Node(T&& k) : key{ k }, prev{ nullptr }, next{ nullptr } {
        }

        Node(T&& k, Node* p, Node* n) : key{ k }, prev{ p }, next{ n } {
        }
    };

    Node* head;

private:
    Node* search(T&& key) {
        Node* p = head->next;
        while (p != head && p->key != key)
            p = p->next;
        return p;
    }

    void deleteNode(Node*& t) {
        if (t == head)
            return;
        t->prev->next = t->next;
        t->next->prev = t->prev;
        delete t;
        t = nullptr;
    }

public:
    DoubleLinkList() : head{ nullptr } {
        head = new Node();
        head->next = head;
        head->prev = head;
    }

    virtual ~DoubleLinkList() {
        delete head;
        head = nullptr;
    }

    void deleteFirst() {
        deleteNode(head->next);
    }

    void deleteLast() {
        deleteNode(head->prev);
    }
    
    void deleteKey(T&& key) {
        Node* p = search(std::move(key));
        deleteNode(p);
    }

    void insert(T&& key) {
        Node* t = new Node(std::move(key), head, head->next);
        head->next->prev = t;
        head->next = t;
    }
};

template <typename N>
std::ostream& operator<<(std::ostream& out, const DoubleLinkList<N>& lst) {
    typename DoubleLinkList<N>::Node* p = lst.head->next;
    while (p != lst.head) {
        out << p->key << ' ';
        p = p->next;
    }
    out << '\n';
    return out;
}

int main1() {
    int N = 0;
    std::cin >> N;

    DoubleLinkList<int> lst;
    std::string command;
    int key;
    while (N--) {
        std::cin >> command >> key;

        if (command[0] == 'i') {
            lst.insert(std::move(key));
        }
        else if (command[0] == 'd') {
            if (command.size() > 6) {
                if (command[6] == 'F')
                    lst.deleteFirst();
                else
                    lst.deleteLast();
            }
            else {
                lst.deleteKey(std::move(key));
            }
        }
    }

    std::cout << lst;
    return 0;
}

int main2() {
    std::list<char> L;

    L.push_front('b');
    L.push_back('c');
    L.push_front('a');

    std::cout << L.front();
    std::cout << L.back();

    L.pop_front();
    L.push_back('d');

    std::cout << L.front();
    std::cout << L.back() << '\n';

    return 0;
}

std::ostream& operator<<(std::ostream& out, const std::list<int>& lst) {
    for (const auto& v : lst) {
        out << v << ' ';
    }
    out << '\n';
    return out;
}

int main() {
    int N = 0;
    std::cin >> N;

    std::list<int> lst;
    std::string command;
    int key;
    while (N--) {
        std::cin >> command >> key;

        if (command[0] == 'i') {
            lst.push_front(key);
        }
        else if (command[0] == 'd') {
            if (command.size() > 6) {
                if (command[6] == 'F')
                    lst.pop_front();
                else
                    lst.pop_back();
            }
            else {
                for (auto it = lst.begin(); it != lst.end(); ++it) {
                    if (*it == key) {
                        lst.erase(it);
                        break;
                    }
                }
            }
        }
    }

    std::cout << lst;
    return 0;
}
