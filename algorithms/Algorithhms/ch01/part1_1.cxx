#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include "../common/alg_math.h"
#include "../common/alg_binarysearch.h"
#include "../common/alg_utils.h"

static void test(int argc, char* argv[]) {
    if (argc < 2)
        return;

    std::ifstream wlFile(argv[1]);
    std::vector<int> whiteList;

    int val{ 0 };
    while (wlFile >> val) {
        whiteList.push_back(val);
    }
    std::sort(whiteList.begin(), whiteList.end());

    // print black list
    while (std::cin >> val) {
        if (BinarySearch<int>::search(whiteList, val) < 0) {
            std::cout << val << '\n';
        }
    }
}

int main(int argc, char* argv[]) {
    test(argc, argv);
    return 0;
}