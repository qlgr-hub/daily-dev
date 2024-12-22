#include <iostream>
#include <array>

static const int MAX = 10000;
static const int NIL = -1;

struct BNode {
    int p;
    int l;
    int r;

    BNode() : p{ NIL }, l{ NIL }, r{ NIL } {
    }
};

static std::array<BNode, MAX> T;
static std::array<int, MAX> D;
static std::array<int, MAX> H;

void setDepth(int u, int d) {
    if (u == NIL)
        return;
    D[u] = d;
    setDepth(T[u].l, d + 1);
    setDepth(T[u].r, d + 1);
}

int setHeight(int u) {
    int h1 = 0;
    int h2 = 0;

    if (T[u].l != NIL)
        h1 = setHeight(T[u].l) + 1;

    if (T[u].r != NIL)
        h2 = setHeight(T[u].r) + 1;

    H[u] = (h1 > h2) ? h1 : h2;
    return H[u];
}

int getSibling(int u) {
    if (T[u].p == NIL)
        return NIL;

    if (T[T[u].p].l != u && T[T[u].p].l != NIL)
        return T[T[u].p].l;

    if (T[T[u].p].r != u && T[T[u].p].r != NIL)
        return T[T[u].p].r;

    return NIL;
}

void print(int u) {
    std::cout << "node " << u << ": ";
    std::cout << "parent = " << T[u].p << ", ";
    std::cout << "sibline = " << getSibling(u) << ", ";

    int deg = 0;
    if (T[u].l != NIL)
        ++deg;
    if (T[u].r != NIL)
        ++deg;

    std::cout << "degree = " << deg << ", ";
    std::cout << "depth = " << D[u] << ", ";
    std::cout << "height = " << H[u] << ", ";

    if (T[u].p == NIL)
        std::cout << "root\n";
    else if (T[u].l == NIL && T[u].r == NIL)
        std::cout << "leaf\n";
    else
        std::cout << "internal node\n";
}

int main() {
    int n = 0;
    std::cin >> n;

    for (int i = 0; i < n; ++i) {
        int v, l, r;
        std::cin >> v >> l >> r;
        T[v].l = l;
        T[v].r = r;

        if (l != NIL)
            T[l].p = v;

        if (r != NIL)
            T[r].p = v;
    }

    int root = 0;
    for (int i = 0; i < n; ++i)
        if (T[i].p == NIL)
            root = i;

    setDepth(root, 0);
    setHeight(root);

    for (int i = 0; i < n; ++i)
        print(i);

    return 0;
}
