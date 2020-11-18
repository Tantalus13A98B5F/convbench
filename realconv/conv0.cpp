#include <cassert>
#include <iostream>
#include <fstream>
#include <initializer_list>
#include <vector>
#include <chrono>


template <typename contTy>
auto product(const contTy &cobj) -> typename contTy::value_type {
    typename contTy::value_type ret = 1;
    for (auto i: cobj) ret *= i;
    return ret;
}


template <int Dim, bool Check = true>
struct DimIdx {
    DimIdx<Dim-1, false> sub;
    const std::size_t step, range;

    DimIdx(const std::vector<std::size_t> &dims)
        : sub(dims), step(sub.step * sub.range), range(dims[dims.size() - Dim]) {
        if (Check) assert (dims.size() == Dim);
    }

    template <typename First, typename... Args>
    std::size_t idx(First cur, Args... args) const {
        return cur * step + sub.idx(args...);
    }

    template <typename First, typename... Args>
    void unpack(First &cur, Args&... args) const {
        cur = range;
        sub.unpack(args...);
    }
};

template <bool Check>
struct DimIdx<1, Check> {
    const std::size_t step, range;

    DimIdx(const std::vector<std::size_t> &dims)
        : step(1), range(dims[dims.size() - 1]) {
        if (Check) assert (dims.size() == 1);
    }

    template <typename First>
    std::size_t idx(First cur) const {
        return cur;
    }

    template <typename First>
    void unpack(First &cur) const {
        cur = range;
    }
};


template <typename dtype = float>
class tensor {
public:
    std::vector<std::size_t> dims;
    std::vector<dtype> data;

    tensor(std::initializer_list<std::size_t> args)
        : dims(args), data(product(args), 0) { }

    template <typename DI, typename... Args>
    dtype& operator()(const DI& dimidx, Args... args) {
        return data[dimidx.idx(args...)];
    }

    template <typename DI, typename... Args>
    const dtype& operator()(const DI& dimidx, Args... args) const {
        return data[dimidx.idx(args...)];
    }

    std::ifstream& readBinary(std::ifstream& is) {
        auto ptr = &*(data.begin());
        std::size_t size = data.size() * sizeof(dtype);
        is.read((char*)ptr, size);
        return is;
    }

    std::ofstream& dumpBinary(std::ofstream& os) {
        auto ptr = &*(data.begin());
        std::size_t size = data.size() * sizeof(dtype);
        os.write((char*)ptr, size);
        return os;
    }
};


template <typename dtype, int unrollLen, int stepLen>
struct RangeUnroll {
    const dtype start;

    RangeUnroll(dtype start)
        : start(start) { }
    
    template <typename Func>
    void foreach(Func func) const {
        func(start, start+stepLen);
        RangeUnroll<dtype, unrollLen-1, stepLen>(start+stepLen).foreach(func);
    }
};

template <typename dtype, int stepLen>
struct RangeUnroll<dtype, 1, stepLen> {
    const dtype start;

    RangeUnroll(dtype start)
        : start(start) { }

    template <typename Func>
    void foreach(Func func) const {
        func(start, start+stepLen);
    }
};

template <typename dtype = std::size_t, int unrollLen = 1, int stepLen = 1>
struct Range {
    const dtype start, stop;

    Range(dtype start, dtype stop)
        : start(start), stop(stop) { }

    template <int newUnrollLen>
    auto fullUnroll() -> RangeUnroll<dtype, newUnrollLen, stepLen> {
        return RangeUnroll<dtype, newUnrollLen, stepLen>(start);
    }

    template <int newUnrollLen>
    auto unrollBy() -> Range<dtype, newUnrollLen, stepLen> {
        return Range<dtype, newUnrollLen, stepLen>(start, stop);
    }

    template <int newStepLen>
    auto stepBy() -> Range<dtype, unrollLen, newStepLen> {
        return Range<dtype, unrollLen, newStepLen>(start, stop);
    }

    template <typename Func>
    void foreach(Func func) const {
        constexpr int unrolledStep = unrollLen * stepLen;
        dtype idx = start;
        if (unrollLen > 1)
            for (; idx + unrolledStep <= stop; idx += unrolledStep)
                RangeUnroll<dtype, unrollLen, stepLen>(idx).foreach(func);
        for (; idx < stop; idx += stepLen) {
            dtype idx2 = (stepLen != 1) ? std::min(idx+stepLen, stop) : idx+1;
            func(idx, idx2);
        }
    }
};


template <typename dtype>
tensor<dtype> conv2d(const tensor<dtype> &img, const tensor<dtype> &weight) {
    DimIdx<4> dimImg(img.dims), dimWeight(weight.dims);
    std::size_t N, C, H, W, Cout, Cin, Ker, Ker2;
    dimImg.unpack(N, C, H, W);
    dimWeight.unpack(Cout, Cin, Ker, Ker2);
    assert (Cin == C);
    assert (Ker == Ker2);
    assert (Ker == 3);
    tensor<dtype> ret { N, Cout, H-2, W-2 };
    DimIdx<4> dimRet(ret.dims);
    Range<>(0, N).foreach([&] (int in, int in2) {
    //Range<>(0, H-2).stepBy<2>().foreach([&] (int th, int th2) {
    Range<>(0, Cout).foreach([&] (int co, int co2) {
    Range<>(0, Cin).foreach([&] (int ci, int ci2) {
    Range<>(0, H-2).foreach([&] (int ih, int ih2) {
    Range<>(0, W-2).foreach([&] (int iw, int iw2) {
    //Range<>(th, th2).unrollBy<2>().foreach([&] (int ih, int ih2) {
        dtype val = 0;
        Range<>(0, 3).fullUnroll<3>().foreach([&] (int kh, int kh2) {
        Range<>(0, 3).fullUnroll<3>().foreach([&] (int kw, int kw2) {
             val += weight(dimWeight, co, ci, kh, kw)
                  * img(dimImg, in, ci, ih+kh, iw+kw);
        });
        });
        ret(dimRet, in, co, ih, iw) += val;
    });
    });
    });
    });
    });
    //});
    return ret;
}


int main() {
    tensor<float> weight {64, 3, 3, 3}, img {20, 3, 1000, 2000};
    std::ifstream fin;
    fin.open("weight.dat", std::ifstream::binary);
    weight.readBinary(fin).close();
    fin.open("input.dat", std::ifstream::binary);
    img.readBinary(fin).close();

    using namespace std::chrono;
    std::cout << "start" << std::endl;
    auto t0 = steady_clock::now();
    auto result = conv2d(img, weight);
    auto t1 = steady_clock::now();
    auto span = duration_cast<duration<double> >(t1 - t0);
    std::cout << "time: " << span.count() << std::endl;

    //std::ofstream ofs("output2.dat", std::ofstream::binary);
    //result.dumpBinary(ofs);
    return 0;
}
