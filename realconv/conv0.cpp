#include <cassert>
#include <iostream>
#include <fstream>
#include <initializer_list>
#include <vector>


template <typename contTy>
auto product(const contTy &cobj) -> typename contTy::value_type {
    typename contTy::value_type ret = 1;
    for (auto i: cobj) ret *= i;
    return ret;
}


template <typename dtype = float>
class tensor {
public:
    std::vector<std::size_t> dims;
    std::vector<dtype> data;

    tensor(std::initializer_list<std::size_t> args)
        : dims(args), data(product(args)) { }
};


template <typename dtype>
tensor<dtype> conv2d(const tensor<dtype> &img, const tensor<dtype> &weight) {
    const std::size_t N = img.dims[0], C = img.dims[1],
                 H = img.dims[2], W = img.dims[3];
    const std::size_t Cout = weight.dims[0], Cin = weight.dims[1],
                 Ker = weight.dims[2];
    const std::size_t CHW = C * H * W, HW = H * W;
    const std::size_t CKK = Cin * Ker * Ker, KK = Ker * Ker;
    const std::size_t CHW2 = Cout * (H-2) * (W-2), HW2 = (H-2) * (W-2), W2 = W-2;
    assert (Cin == C);
    assert (Ker == weight.dims[3]);
    assert (Ker == 3);
    tensor<dtype> ret { N, Cout, H-2, W-2 };
    for (int in = 0; in < N; in++) {
        for (int co = 0; co < Cout; co++) {
            for (int ih = 0; ih < H-2; ih++) {
            for (int iw = 0; iw < W-2; iw++) {
                dtype val = 0;
                for (int ci = 0; ci < Cin; ci++) {
                    for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        std::size_t widx = co * CKK + ci * KK + kh * Ker + kw,
                               iidx = in * CHW + ci * HW + (ih+kh) * W + (iw+kw);
                        val += weight.data[widx] * img.data[iidx];
                    }
                    }
                }
                std::size_t ridx = in * CHW2 + co * HW2 + ih * W2 + iw;
                ret.data[ridx] = val;
            }
            }
        }
    }
    return ret;
}


template <typename dtype>
std::ifstream& readBinary(tensor<dtype> &arg, std::ifstream& is) {
    auto ptr = &*(arg.data.begin());
    std::size_t size = arg.data.size() * sizeof(dtype);
    is.read((char*)ptr, size);
    return is;
}

template <typename dtype>
void dumpBinary(const tensor<dtype> &arg, std::ofstream& os) {
    auto ptr = &*(arg.data.begin());
    std::size_t size = arg.data.size() * sizeof(dtype);
    os.write((char*)ptr, size);
}


int main() {
    tensor<> weight {64, 3, 3, 3}, img {20, 3, 1000, 2000};
    std::ifstream fin;
    fin.open("weight.dat", std::ifstream::binary);
    readBinary(weight, fin).close();
    fin.open("input.dat", std::ifstream::binary);
    readBinary(img, fin).close();
    auto result = conv2d(img, weight);
    std::ofstream ofs("output2.dat", std::ofstream::binary);
    dumpBinary(result, ofs);
    return 0;
}
