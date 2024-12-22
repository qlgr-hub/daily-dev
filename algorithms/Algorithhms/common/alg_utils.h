#pragma once

#include <cstdint>
#include <utility>
#include <vector>
#include <chrono>
#include <random>

class RandomGenerator {
protected:
    std::default_random_engine _randomEngine{};

    RandomGenerator() {
        using namespace std::chrono;
        auto rdSeed{ duration_cast<seconds>(system_clock::now().time_since_epoch()).count() };
        _randomEngine.seed(rdSeed);
    }
};


template <typename NT>
class Uniform : public RandomGenerator {
    std::uniform_int_distribution<NT> _und;

public:
    Uniform(const std::pair<NT, NT>& range)
        : RandomGenerator{}, _und{ range.first, range.second } {
    }

public:
    void operator()(std::vector<NT>& arr, int N) {
        arr.resize(N, NT{});

        for (auto& e : arr) {
            e = _und(_randomEngine);
        }
    }

    NT operator()() {
        return _und(_randomEngine);
    }
};


class Bernoulli : public RandomGenerator {
    std::bernoulli_distribution _br;
public:
    Bernoulli(double p) : RandomGenerator{}, _br{ p } {
    }

public:
    bool operator()() {
        return _br(_randomEngine);
    }
};


class TimeStats {
    std::chrono::time_point<std::chrono::high_resolution_clock> _start;

public:
    TimeStats() {
        _start = std::chrono::high_resolution_clock::now();
    }

public:
    double elapsNano() const {
        using namespace std::chrono;
        auto elaps{ duration_cast<nanoseconds>(high_resolution_clock::now() - _start).count() };
        return static_cast<double>(elaps);
    }

    double elapsMicro() const {
        auto nano{ elapsNano() };
        return (nano / 1000.);
    }

    double elapsMilli() const {
        auto micro{ elapsMicro() };
        return (micro / 1000.);
    }

    double elapsSec() const {
        auto milli{ elapsMilli() };
        return (milli / 1000.);
    }
};
