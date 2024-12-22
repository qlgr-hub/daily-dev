#include <iostream>
#include <array>

static const int MAX = 100005;
static const int NIL = -1;

struct TNode {
    int p;
    int l;
    int r;

    TNode() : p{ NIL }, l{ NIL }, r{ NIL } {
    }
};

static std::array<TNode, MAX> T;
static std::array<int, MAX> D;

void print(int u) {
    std::cout << "node " << u << ": ";
    std::cout << "parent = " << T[u].p << ", ";
    std::cout << "depth = " << D[u] << ", ";

    if (T[u].p == NIL)
        std::cout << "root, ";
    else if (T[u].l == NIL) 
        std::cout << "leaf, ";
    else
        std::cout << "internal node, ";

    std::cout << "[";
    for (int i = 0, c = T[u].l; c != NIL; ++i, c = T[c].r) {
        std::cout << c << ' ';
    }
    std::cout << "]\n";
}

void rec(int u, int p) {
    D[u] = p;
    
    if (T[u].r != NIL)
        rec(T[u].r, p);

    if (T[u].l != NIL)
        rec(T[u].l, p + 1);
}

int main() {
    int n = 0;
    std::cin >> n;

    int l = 0;
    for (int i = 0; i < n; ++i) {
        int v, d;
        std::cin >> v >> d;

        for (int j = 0; j < d; ++j) {
            int c;
            std::cin >> c;
            if (j == 0)
                T[v].l = c;
            else
                T[l].r = c;
            l = c;
            T[c].p = v;
        }
    }

    int r = 0;
    for (int i = 0; i < n; ++i)
        if (T[i].p == NIL)
            r = i;

    rec(r, 0);

    for (int i = 0; i < n; ++i)
        print(i);

    return 0;
}
