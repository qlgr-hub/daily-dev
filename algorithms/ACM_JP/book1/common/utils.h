#ifndef _ACM_JP_BOOK1_COMMON_H_
#define _ACM_JP_BOOK1_COMMON_H_

#include <vector>
#include <iostream>

template<typename T>
bool cmn_bsearch(const std::vector<T>& arr, int x) {
    int l = 0;
    int r = arr.size();

    bool find = false;
    while ( (r - l) >= 1 ) {
        int i = (l + r) / 2;
        if (arr[i] == x) {
            find = true;
            break;
        }
        else if (arr[i] < x) {
            l = i + 1;
        }
        else {
            r = i;
        }
    }

    return find;
}


#endif // !_ACM_JP_BOOK1_COMMON_H_
