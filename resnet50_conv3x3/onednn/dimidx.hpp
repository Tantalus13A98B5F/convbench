#ifndef _DIMIDX_HPP_
#define _DIMIDX_HPP_
#include <initializer_list>
#include <type_traits>
#include <cassert>

namespace DI {

using std::initializer_list;
using std::size_t;

struct NoneType {} None, Any;


// struct tryAssign

template <typename target_t, bool valid = std::is_integral<target_t>::value>
struct tryAssign {
    tryAssign(target_t &t, size_t val) {
        t = val;
    }
};

template <typename target_t>
struct tryAssign<target_t, false> {
    tryAssign(target_t &t, size_t val) { }
};


// struct tryCompare

template <typename target_t, bool valid = std::is_integral<target_t>::value>
struct tryCompare {
    const bool value;
    tryCompare(target_t t, size_t val): value(t == val) { }
};

template <typename target_t>
struct tryCompare<target_t, false> {
    const bool value;
    tryCompare(target_t t, size_t val): value(true) { }
};


// struct refElem

template <typename contTy, bool isPointer = std::is_pointer<contTy>::value>
struct refElem {
    typedef typename std::conditional<std::is_const<contTy>::value,
            typename contTy::const_reference,
            typename contTy::reference>::type type;
};

template <typename contTy>
struct refElem<contTy, true> {
    typedef typename std::remove_pointer<contTy>::type &type;
};


// struct resizeCont

template <typename contTy, bool resize>
struct resizeCont {
    resizeCont(contTy &cont, size_t size) {
        cont.resize(size);
    }
};

template <typename contTy>
struct resizeCont<contTy, false> {
    resizeCont(contTy &cont, size_t size) { }
};


// strcut DimIdx

template <int Dim>
struct DimIdx {
    const size_t range;
    DimIdx<Dim-1> sub;
    const size_t stride, totalsize;

    DimIdx(initializer_list<size_t> dims)
        : DimIdx(dims.begin()) {
        assert (dims.size() == Dim);
    }

    DimIdx(initializer_list<size_t>::iterator it)
        : range(*it), sub(it+1), stride(sub.totalsize), totalsize(stride * range) { }

    DimIdx(const DimIdx<Dim> &rhs)
        : range(rhs.range), sub(rhs.sub), stride(rhs.stride), totalsize(rhs.totalsize) {}

    template <typename First, typename... Args>
    size_t idx(First cur, Args... args) const {
        return cur * stride + sub.idx(args...);
    }

    template <typename First, typename... Args>
    void unpack(First &cur, Args&... args) const {
        tryAssign<First>(cur, range);
        sub.unpack(args...);
    }

    template <typename First, typename... Args>
    bool validate(First cur, Args... args) const {
        return tryCompare<First>(cur, range).value && sub.validate(args...);
    }

    template <typename contTy, typename... Args>
    auto operator() (contTy &arr, Args... args) const
            -> typename refElem<contTy>::type {
        return arr[idx(args...)];
    }

    template <typename contTy>
    struct Accessor {
        const DimIdx<Dim> dim;
        contTy &cont;

        Accessor(const DimIdx<Dim> &s, contTy &c)
            : dim(s), cont(c) { }

        template <typename... Args>
        auto operator() (Args... args) const -> typename refElem<contTy>::type {
            return cont[dim.idx(args...)];
        }
    };

    template <bool resize=false, typename contTy>
    Accessor<contTy> bind(contTy &cont) const {
        resizeCont<contTy, resize>(cont, totalsize);
        return Accessor<contTy>(*this, cont);
    }
};

template <>
struct DimIdx<0> {
    const size_t totalsize;

    DimIdx(initializer_list<size_t>::iterator it)
        : totalsize(1) { }

    DimIdx(const DimIdx<0> &rhs)
        : totalsize(1) { }

    size_t idx() const { return 0; }

    void unpack() const { }

    bool validate() const { return true; }
};


// class Range

template <typename dtype = std::size_t, int stepLen = 1>
struct Range {
    const dtype start, stop;

    Range(dtype start, dtype stop)
        : start(start), stop(stop) { }

    class iterator {
        dtype val;

    public:
        iterator(dtype v): val(v) {}

        bool operator!=(const iterator& rhs) const {
            return val != rhs.val;
        }

        void operator++() { val += stepLen; }

        dtype operator*() const { return val; }
    };

    iterator begin() const {
        return iterator(start);
    }

    iterator end() const {
        dtype tmp = stop + stepLen - 1;
        return iterator(tmp - tmp % stepLen);
    }

    template <int newStepLen>
    auto stepBy() -> Range<dtype, newStepLen> {
        return Range<dtype, newStepLen>(start, stop);
    }
};


}  // end namespace
#endif  // _DIMIDX_HPP_