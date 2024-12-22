#include <array>
#include <iostream>
#include <stack>

static constexpr int MAX = 1400;

struct Rectangle {
    int height;
    int pos;

    Rectangle(int h = 0, int p = 0) : height{ h }, pos{ p } {
    }
};

int getLargestRectangle(std::array<int, MAX>& buff, int size) {
    std::stack<Rectangle> S;
    buff[size] = 0;

    int maxv = 0;
    for (int i = 0; i <= size; ++i) {
        Rectangle rect{ buff[i], i };
        
        if (S.empty()) {
            S.push(rect);
        }
        else {
            if (S.top().height < rect.height) {
                S.push(rect);
            }
            else if (S.top().height > rect.height) {
                int target = i;
                while (!S.empty() && S.top().height >= rect.height) {
                    Rectangle pre = S.top(); S.pop();
                    int area = pre.height * (i - pre.pos);
                    maxv = std::max(maxv, area);
                    target = pre.pos;
                }

                rect.pos = target;
                S.push(rect);
            }
            else {
                // do nothing
            }
        }
    }

    return maxv;
}

static std::array<std::array<int , MAX>, MAX> buffer;
static std::array<std::array<int , MAX>, MAX> T;

int getLargestRectangle(int H, int W) {
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            if (buffer[i][j]) {
                T[i][j] = 0;
            }
            else {
                T[i][j] = (i > 0) ? T[i - 1][j] + 1 : 1;
            }
        }
    }

    int maxv = 0;
    for (int i = 0; i < H; ++i) {
        maxv = std::max(maxv, getLargestRectangle(T[i], W));
    }

    return maxv;
}

int main() {
    int H, W;
    std::cin >> H >> W;

    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            std::cin >> buffer[i][j];
        }
    }

    std::cout << getLargestRectangle(H, W) << '\n';
    return 0;
}
