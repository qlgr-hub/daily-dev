#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <string_view>

template <typename IT>
class _CounterBase {
    std::string _id;
    IT _counter;

public:
    _CounterBase() = default;

    _CounterBase(std::string_view id) : _id{ id }, _counter{ IT() } {
    }

public:
    void increment() {
        ++_counter;
    }

    IT tally() const {
        return _counter;
    }

    
public:
    friend std::ostream& operator<<(std::ostream& out, const _CounterBase& c) {
        out << c._counter << ' ' << c._id;
        return out;
    }

    friend bool operator<(const _CounterBase& rhs, const _CounterBase& lhs) {
        return (rhs.tally() < lhs.tally());
    }

    friend bool operator>(const _CounterBase& rhs, const _CounterBase& lhs) {
        return !(rhs < lhs);
    }

    friend bool operator==(const _CounterBase& rhs, const _CounterBase& lhs) {
        return (rhs.tally() == lhs.tally());
    }

    friend bool operator!=(const _CounterBase& rhs, const _CounterBase& lhs) {
        return !(rhs == lhs);
    }
};

using Counter = _CounterBase<int>;
