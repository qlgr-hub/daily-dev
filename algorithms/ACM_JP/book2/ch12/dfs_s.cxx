#include "graph_m.h"


Graph_M<100> G;

int main() {
    G.reconstruct();
    G.dfs(false);
    return 0;
}