#include <iostream>
#include <format>
#include <print>
#include <stdfloat>

int main() {
    std::float16_t fval = 0.6578f16;
    std::cout << std::format("std::float16_t and std::format are already supported, fval={}\n", fval);
    std::print("std::print is already supported\n");
    return 0;
}