#include <iostream>
#include <array>

static const int MAX = 10000;
static const int NIL = -1;

struct Node {
    int p;
    int l;
    int r;

    Node() : p{ NIL }, l{ NIL }, r{ NIL } {
    }
};

static std::array<Node, MAX> T;

void preParse(int u) {
    if (u == NIL)
        return;

    std::cout << ' ' << u;
    preParse(T[u].l);
    preParse(T[u].r);
}

void inParse(int u) {
    if (u == NIL)
        return;

    inParse(T[u].l);
    std::cout << ' ' << u;
    inParse(T[u].r);
}

void postParse(int u) {
    if (u == NIL)
        return;

    postParse(T[u].l);
    postParse(T[u].r);
    std::cout << ' ' << u;
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

    std::cout << "Preorder\n";
    preParse(root);
    std::cout << '\n';

    std::cout << "Inorder\n";
    inParse(root);
    std::cout << '\n';

    std::cout << "Postoeder\n";
    postParse(root);
    std::cout << '\n';

    return 0;
}