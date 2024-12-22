#pragma once

#include <vector>


enum class AlgoT {
    AT_quickFind = 0,
    AT_quickUnion = 1,
    AT_weightedQuickUnion = 2,
};

template <AlgoT AT = AlgoT::AT_weightedQuickUnion>
class UnionFind {
    std::vector<int> _id;
    std::vector<int> _sz;
    int _count;

    // quick find 实现中：
    //   _id 数组的索引代表触点 
    //   _id 数组的元素是每个触电对应的连通分量的 id （ccid）
    int _quickFind_find(int p) const {
        return _id[p];
    }

    void _quickFind_union(int p, int q) {
        // ccid 可以理解为 连通分量的 id
        int ccid_p{ _quickFind_find(p) };
        int ccid_q{ _quickFind_find(q) };
        
        // 已经是同一个连通分量中了 就直接返回
        if (ccid_p == ccid_q)
            return;

        // 将原先在 ccid_p 连通分量中的触点合并到 ccid_q 连通分量
        // 其实反过来也没问题
        for (int i{ 0 }; i < _id.size(); ++i) {
            if (_quickFind_find(i) == ccid_p)
                _id[i] = ccid_q;
        }
        --_count; // 合并了两个连通分量，那么连通分量个数就得减一
    }


    // quick union 实现中：
    //   _id 数组的索引代表触点 
    //   _id 数组的元素是与每个触电连通的一个触点（其实是父链接方式表示的森林结构）
    int _quickUnion_find(int p) const {
        // 这里可以理解为 找一个指向自己的触点
        // 这种触点是一个根触点
        while (p != _id[p]) 
            p = _id[p];
        return p;
    }

    void _quickUnion_union(int p, int q) {
        int pRoot{ _quickUnion_find(p) };
        int qRoot{ _quickUnion_find(q) };
        
        // 已经是同一个连通分量中了 就直接返回
        if (pRoot == qRoot)
            return;

        // 两个触点的根触点之间建立了连接就等同于合并了两个连通分量
        // 父链接方式表示的森林中合并两棵树
        _id[pRoot] = qRoot;
        --_count; // 合并了两个连通分量，那么连通分量个数就得减一
    }

    int _weightedQuickUnion_find(int p) const {
        // 这里 find 的实现根 quick find 的一样 就是父链接森林中的树找根
        return _quickUnion_find(p);
    }

    void _weightedQuickUnion_union(int p, int q) {
        int pRoot{ _quickUnion_find(p) };
        int qRoot{ _quickUnion_find(q) };
        
        // 已经是同一个连通分量中了 就直接返回
        if (pRoot == qRoot)
            return;

        // 这里总是优先将高度小的树合并到高度大的树（保证树的相对平衡）
        if (_sz[pRoot] < _sz[qRoot]) {
            _id[pRoot] = qRoot;
            _sz[qRoot] += _sz[pRoot];
        }
        else {
            _id[qRoot] = pRoot;
            _sz[pRoot] += _sz[qRoot];
        }
        --_count; // 合并了两个连通分量，那么连通分量个数就得减一
    }

public:
    UnionFind(int N) : _id(N), _sz(N), _count{ N } {
        for (int i{ 0 }; i < N; ++i) {
            _id[i] = i;
            _sz[i] = 1;
        }
    }

    // 关键字冲突所以用 unite
    void unite(int p, int q) {
        if constexpr (AT == AlgoT::AT_quickFind) {
            _quickFind_union(p, q);
        }
        else if (AT == AlgoT::AT_quickUnion) {
            _quickUnion_union(p, q);
        }
        else {
            _weightedQuickUnion_union(p, q);
        }
    }

    int find(int p) const {
        if constexpr (AT == AlgoT::AT_quickFind) {
            return _quickFind_find(p);
        }
        else if (AT == AlgoT::AT_quickUnion) {
            return _quickUnion_find(p);
        }
        else {
            return _weightedQuickUnion_find(p);
        }
    }

    bool connected(int p, int q) const {
        return (find(p) == find(q));
    }

    int count() const {
        return _count;
    }
};

using QuickFindUF = UnionFind<AlgoT::AT_quickFind>;
using QuickUnionUF = UnionFind<AlgoT::AT_quickUnion>;
using WeightedQuickUnionUF = UnionFind<AlgoT::AT_weightedQuickUnion>;
