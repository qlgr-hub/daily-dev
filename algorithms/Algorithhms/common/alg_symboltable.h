#pragma once

#include <optional>
#include <vector>


template <typename KT, typename VT>
class SequentialSearchST {
    struct _Node;

    struct _Node {
        KT _key;
        VT _val;
        _Node* _next;

        _Node(const KT& key, const VT& val, _Node* next = nullptr)
            : _key{ key }, _val{ val }, _next{ next } {
        }
    };

    _Node* _first;
    int _N;

public:
    SequentialSearchST() : _first{ nullptr }, _N{ 0 } {
    }

    // 为了多个不同的 ST 结构作为模板参数，并用带 int 参数的构造函数统一构造加了个 int 类型作为占位符
    SequentialSearchST(int) : SequentialSearchST{} {
    }

    ~SequentialSearchST() {
        auto p{ _first };
        while (p) {
            auto tmp{ p };
            p = p->_next;

            delete tmp;
        }

        p = nullptr;
        _first = nullptr;
    }

public:
    std::optional<VT> get(const KT& key) const {
        for (_Node* x{ _first }; x != nullptr; x = x->_next) {
            if (x->_key == key)
                return x->_val;
        }
        return {};
    }

    void put(const KT& key, const VT& val) {
        for (_Node* x{ _first }; x != nullptr; x = x->_next) {
            if (x->_key == key)
                x->_val = val;
        }
        _first = new _Node{ key, val, _first };
        ++_N;
    }

    void del(const KT& key) {
        _Node* pre{ nullptr };
        _Node* tmp{ _first };
        while (tmp) {
            if (tmp->_key == key) {
                if (pre) {
                    pre->_next = tmp->_next;
                }
                else {
                    _first = tmp->_next;
                }

                break;
            }

            pre = tmp;
            tmp = tmp->_next;
        }

        if (tmp) {
            delete tmp;
            tmp = nullptr;
            --_N;
        }
    }

    int size() const {
        return _N;
    }

    std::vector<KT> keys() const {
        std::vector<KT> res(_N, KT{});

        int i{ 0 };
        for (_Node* x{ _first }; x != nullptr; x = x->_next) {
            res[i++] = x->_key;
        }

        return res;
    }

    bool contains(const KT& key) const {
        return get(key).has_value();
    }
};


template <typename KT, typename VT>
class BinarySearchST {
    std::vector<KT> _keys;
    std::vector<VT> _vals;
    int _N;

    int _rank(const KT& key) const {
        int lo{ 0 };
        int hi{ _N - 1 };
        while (lo <= hi) {
            auto mid{ lo + ((hi - lo) >> 1) };
            if (key < _keys[mid]) {
                hi = mid - 1;
            }
            else if (key > _keys[mid]) {
                lo = mid + 1;
            }
            else {
                return mid;
            }
        }
        return lo;
    }

public:
    BinarySearchST(int cap) : _keys(cap, KT{}), _vals(cap, VT{}), _N{ 0 } {
    }

    std::optional<VT> get(const KT& key) const {
        auto i{ _rank(key) };
        if (key == _keys[i])
            return _vals[i];
        return {};
    }

    void put(const KT& key, const VT& val) {
        auto i{ _rank(key) };
        if (key == _keys[i]) {
            _vals[i] = val;
            return;
        }
        
        for (int j{ _N }; j > i; --j) {
            _keys[j] = _keys[j - 1];
            _vals[j] = _vals[j - 1];
        }
        _keys[i] = key;
        _vals[i] = val;
        ++_N;
    }

    void del(const KT& key) {
        auto i{ _rank(key) };
        if (key == _keys[i]) {
            for (; i < _N - 1; ++i) {
                _keys[i] = _keys[i + 1];
                _keys[i] = _keys[i + 1];
            }
            --_N;
        }
    }

    std::vector<KT> keys() const {
        std::vector<KT> res{ _keys.cbegin(), _keys.cbegin() + _N };
        return res;
    }

    bool contains(const KT& key) const {
        return get(key).has_value();
    }
};


template <typename KT, typename VT>
class BST {
    struct _Node;

    struct _Node {
        KT _key;
        VT _val;

        _Node* _left;
        _Node* _right;

        int _N;

        _Node(const KT& key, const VT& val, int N, _Node* l = nullptr, _Node* r = nullptr)
            : _key{ key }, _val{ val }, _N{ N }, _left{ l }, _right{ r } {
        }
    };

    _Node* _root;

private:
    int _size(_Node* x) const {
        if (x)
            return x->_N;
        return 0;
    }

    _Node* _put(_Node* x, const KT& key, const VT& val) {
        if (x == nullptr)
            return new _Node{ key, val, 1};

        if (key < x->_key) {
            x->_left = _put(x->_left, key, val);
        }
        else if (key > x->_key) {
            x->_right = _put(x->_right, key, val);
        }
        else {
            x->_val = val;
        }
        x->_N = _size(x->_left) + _size(x->_right) + 1;
        return x;
    }

    _Node* _get(_Node* x, const KT& key) const {
        if (x == nullptr)
            return nullptr;

        if (key < x->_key) {
            return _get(x->_left, key);
        }
        else if (key > x->_key) {
            return _get(x->_right, key);
        }

        return x;
    }

    _Node* _min(_Node* x) const {
        if (x == nullptr || x->_left == nullptr)
            return x;
        return _min(x->_left);
    }

    _Node* _max(_Node* x) const {
        if (x == nullptr || x->_right == nullptr)
            return x;
        return _max(x->_right);
    }

    _Node* _floor(_Node* x, const KT& key) const {
        if (x == nullptr || key == x->_key)
            return x;

        if (key < x->_key) {
            return _floor(x->_left, key);
        }

        auto t{ _floor(x->_right, key) };
        return (t ? t : x);
    }

    _Node* _ceiling(_Node* x, const KT& key) const {
        if (x == nullptr || key == x->_key)
            return x;

        if (key > x->_key) {
            return _ceiling(x->_right, key);
        }

        auto t{ _ceiling(x->_left, key) };
        return (t ? t : x);
    }

    _Node* _select(_Node* x, int k) const {
        if (x == nullptr)
            return x;

        auto t{ _size(x->_left) };
        if (t > k) {
            return _select(x->_left, k);
        }
        else if (t < k) {
            return _select(x->_right, k - t - 1);
        }

        return x;
    }

    int _rank(_Node* x, const KT& key) const {
        if (x == nullptr)
            return 0;

        if (key < x->_key) {
            return _rank(x->_left, key);
        }
        else if (key > x->_key) {
            return 1 + _size(x->_left) + _rank(x->_right, key);
        }

        return _size(x->_left);
    }

    _Node* _deleteMin(_Node* x) {
        if (x == nullptr)
            return x;

        if (x->_left == nullptr) {
            delete x;
            return x->_right;
        }

        x->_left = _deleteMin(x->_left);
        x->_N = _size(x->_left) + _size(x->_right) + 1;
        return x;
    }

    _Node* _deleteMax(_Node* x) {
        if (x == nullptr)
            return x;

        if (x->_right == nullptr) {
            delete x;
            return x->_left;
        }

        x->_right = _deleteMax(x->_right);
        x->_N = _size(x->_left) + _size(x->_right) + 1;
        return x;
    }

    _Node* _delete(_Node* x, const KT& key) {
        if (x == nullptr)
            return x;

        if (key < x->_key) {
            x->_left = _delete(x->_right, key);
        }
        else if (key > x->_key) {
            x->_right = _delete(x->_right, key);
        }
        else {
            if (x->_right == nullptr) {
                return x->_left;
            }
            if (x->_left == nullptr) {
                return x->_right;
            }

            auto t{ x };
            x = _min(t->_right);
            x->_right = _deleteMin(t->_right);
            x->_left = t->_left;
            delete t;
            t = nullptr;
        }
        x->_N = _size(x->_left) + _size(x->_right) + 1;
        return x;
    }

    void _keys(_Node* x, const KT& lo, const KT& hi, std::vector<KT>& ks) const {
        if (x == nullptr)
            return;

        if (lo < x->_key) {
            _keys(x->_left, lo, hi, ks);
        }
        
        if (lo <= x->_key && x->_key <= hi) {
            ks.emplace_back(x->_key);
        }

        if (hi > x->_key) {
            _keys(x->_right, lo, hi, ks);
        }
    }

public:
    BST() :  _root{ nullptr } {
    }

    // 为了多个不同的 ST 结构作为模板参数，并用带 int 参数的构造函数统一构造加了个 int 类型作为占位符
    BST(int) : BST{} {
    }

public:
    std::optional<VT> get(const KT& key) const {
        auto x{ _get(_root, key) };
        if (x)
            return x->_val;
        return {};
    }

    void put(const KT& key, const VT& val) {
        _root = _put(_root, key, val);
    }

    int size() const {
        return ((_root == nullptr) ? 0 : _root->_N);
    }

    std::vector<KT> keys() const {
        std::vector<KT> ks;
        if (_root) {
            auto lo{ _min(_root)->_key };
            auto hi{ _max(_root)->_key };
            _keys(_root, lo, hi, ks);
        }
        return ks;
    }
};


template <typename KT, typename VT>
class RedBlackBST {
    static constexpr bool RED{ false };
    static constexpr bool BLACK{ true };

    struct _Node {
        KT _key;
        VT _val;

        bool _color;
        int _N;

        _Node* _left;
        _Node* _right;

        _Node(const KT& key, const VT& val, bool color, int N, _Node* l = nullptr, _Node* r = nullptr)
            : _key{ key }, _val{ val }, _color{ color }, _N{ N }, _left{ l }, _right{ r } {
        }
    };

    _Node* _root;

private:
    int _size(_Node* x) const {
        if (x)
            return x->_N;
        return 0;
    }

    _Node* _get(_Node* x, const KT& key) const {
        if (x == nullptr)
            return nullptr;

        if (key < x->_key) {
            return _get(x->_left, key);
        }
        else if (key > x->_key) {
            return _get(x->_right, key);
        }

        return x;
    }

    _Node* _min(_Node* x) const {
        if (x == nullptr || x->_left == nullptr)
            return x;
        return _min(x->_left);
    }

    _Node* _max(_Node* x) const {
        if (x == nullptr || x->_right == nullptr)
            return x;
        return _max(x->_right);
    }

    _Node* _floor(_Node* x, const KT& key) const {
        if (x == nullptr || key == x->_key)
            return x;

        if (key < x->_key) {
            return _floor(x->_left, key);
        }

        auto t{ _floor(x->_right, key) };
        return (t ? t : x);
    }

    _Node* _ceiling(_Node* x, const KT& key) const {
        if (x == nullptr || key == x->_key)
            return x;

        if (key > x->_key) {
            return _ceiling(x->_right, key);
        }

        auto t{ _ceiling(x->_left, key) };
        return (t ? t : x);
    }

    _Node* _select(_Node* x, int k) const {
        if (x == nullptr)
            return x;

        auto t{ _size(x->_left) };
        if (t > k) {
            return _select(x->_left, k);
        }
        else if (t < k) {
            return _select(x->_right, k - t - 1);
        }

        return x;
    }

    int _rank(_Node* x, const KT& key) const {
        if (x == nullptr)
            return 0;

        if (key < x->_key) {
            return _rank(x->_left, key);
        }
        else if (key > x->_key) {
            return 1 + _size(x->_left) + _rank(x->_right, key);
        }

        return _size(x->_left);
    }

    void _keys(_Node* x, const KT& lo, const KT& hi, std::vector<KT>& ks) const {
        if (x == nullptr)
            return;

        if (lo < x->_key) {
            _keys(x->_left, lo, hi, ks);
        }
        
        if (lo <= x->_key && x->_key <= hi) {
            ks.emplace_back(x->_key);
        }

        if (hi > x->_key) {
            _keys(x->_right, lo, hi, ks);
        }
    }

    _Node* _rotateLeft(_Node* h) {
        auto x{ h->_right };
        h->_right = x->_left;
        x->_left = h;
        x->_color = h->_color;
        h->_color = RED;
        x->_N = h->_N;
        h->_N = 1 + _size(h->_left) + _size(h->_right);
        return x;
    }

    _Node* _rotateRight(_Node* h) {
        auto x{ h->_left };
        h->_left = x->_right;
        x->_right = h;
        x->_color = h->_color;
        h->_color = RED;
        x->_N = h->_N;
        h->_N = 1 + _size(h->_left) + _size(h->_right);
        return x;
    }

    void _flipColors(_Node* h) {
        h->_color = RED;
        h->_left->_color = BLACK;
        h->_right->_color = BLACK;
    }

    bool isRed(_Node* h) const {
        if (h && h->_color == RED)
            return true;
        return false;
    }

    _Node* _put(_Node* h, const KT& key, const VT& val) {
        if (h == nullptr)
            return new _Node{ key, val, RED, 1};

        if (key < h->_key) {
            h->_left = _put(h->_left, key, val);
        }
        else if (key > h->_key) {
            h->_right = _put(h->_right, key, val);
        }
        else {
            h->_val = val;
        }

        return _balance(h);
    }

    _Node* _balance(_Node* h) {
        // 左黑 右红 左旋转
        if (isRed(h->_right) && !isRed(h->_left)) {
            h = _rotateLeft(h);
        }
        // 连续的左红 右旋转
        else if (isRed(h->_left) && isRed(h->_left->_left)) {
            h = _rotateRight(h);
        }

        // 左右都红 反转颜色
        if (isRed(h->_left) && isRed(h->_right)) {
            _flipColors(h);
        }

        h->_N = _size(h->_left) + _size(h->_right) + 1;
        return h;
    }

    _Node* _moveRedLeft(_Node* h) {
        _flipColors(h);
        if (isRed(h->_right->_left)) {
            h->_right = _rotateRight(h->_right);
            h = _rotateLeft(h);
        }
        return h;
    }

    _Node* _moveRedRight(_Node* h) {
        _flipColors(h);
        if (isRed(h->_left->_left)) {
            h = _rotateRight(h);
        }
        return h;
    }

    _Node* _deleteMin(_Node* h) {
        if (h->_left == nullptr)
            return nullptr;

        if (!isRed(h->_left) && !isRed(h->_left->_left)) {
            h = _moveRedLeft(h);
        }
        h->_left = _deleteMin(h->_left);
        return _balance(h);
    }

    _Node* _deleteMax(_Node* h) {
        if (isRed(h->_left)) {
            h = _rotateRight(h);
        }

        if (h->_right == nullptr) 
            return nullptr;

        if (!isRed(h->_right) && !isRed(h->_right->_right)) {
            h = _moveRedRight(h);
        }
        h->_right = _deleteMax(h->_right);
        return _balance(h);
    }

    _Node* _delete(_Node* h, const KT& key) {
        if (key < h->_key) {
            if (!isRed(h->_left) && !isRed(h->_left->_left)) {
                h = _moveRedLeft(h);
            }
            h->_left = _delete(h->_left, key);
        }
        else {
            if (isRed(h->_left)) {
                h = _rotateRight(h);
            }

            if (key == h->_key && h->_left == nullptr)
                return nullptr;

            if (!isRed(h->_right) && !isRed(h->_right->_left)) {
                h = _moveRedRight(h);
            }

            if (key == h->_key) {
                h->_val = _get(h->_right, _min(h->_right)->_key);
                h->_key = _min(h->_right)->_key;
                h->_right = _deleteMin(h->_right);
            }
            else {
                h->_right = _delete(h->_right, key);
            }

            return _balance(h);
        }
    }

public:
    RedBlackBST() : _root{ nullptr } {
    }

    // 为了多个不同的 ST 结构作为模板参数，并用带 int 参数的构造函数统一构造加了个 int 类型作为占位符
    RedBlackBST(int) : RedBlackBST{} {
    }

public:
    std::optional<VT> get(const KT& key) const {
        auto x{ _get(_root, key) };
        if (x)
            return x->_val;
        return {};
    }

    void put(const KT& key, const VT& val) {
        _root = _put(_root, key, val);
        _root->_color = BLACK;
    }

    int size() const {
        return ((_root == nullptr) ? 0 : _root->_N);
    }

    std::vector<KT> keys() const {
        std::vector<KT> ks;
        if (_root) {
            auto lo{ _min(_root)->_key };
            auto hi{ _max(_root)->_key };
            _keys(_root, lo, hi, ks);
        }
        return ks;
    }
};