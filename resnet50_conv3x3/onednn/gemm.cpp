#include <vector>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <cassert>
#include "mkl.h"
#include "dimidx.hpp"
using DI::DimIdx;
typedef std::vector<float> tensor_t;

template <typename contTy>
void read_binary(std::istream &is, contTy &vec) {
    is.read((char*) vec.data(),
            vec.size() * sizeof(contTy::value_type));
}

template <typename contTy>
void init_rand(contTy &vec) {
    std::minstd_rand0 randgen(0);
    std::generate(vec.begin(), vec.end(), randgen);
}

struct SolverNCHW {
    tensor_t weight, data, scratch, result;
    int N, C, H, W, F, K;

    SolverNCHW(const tensor_t &weight, const tensor_t &data,
               const DimIdx<4> &dWeight, const DimIdx<4> &dData)
        : weight(weight), data(data) {
        dData.unpack(N, C, H, W);
        dWeight.unpack(F, DI::None, K, DI::None);
        assert (dWeight.validate(DI::Any, C, 3, K));
        scratch.resize(dData.totalsize * K * K);
        result.resize(N * F * H * W);
    }

    void compute() {
        DimIdx<4> dData {N, C, H, W};
        DimIdx<6> dScratch {N, C, K, K, H, W};
        for (int in = 0; in < N; in++)
        for (int ic = 0; ic < C; ic++)
        {
            for (int ih = 0; ih < H; ih++)
            for (int iw = 0; iw < W; iw++)
            {
                for (int kh = 0; kh < K-1; kh++)
                for (int kw = 0; kw < K-1; kw++)
                {
                    int th = ih + kh - 1, tw = iw + kw - 1;
                    dScratch(scratch, in, ic, kh, kw, ih, iw) =
                        (th < 0 || th >= H || tw < 0 || tw >= W)
                        ? 0 : dData(data, in, ic, th, tw);
                }
            }
        }
        int CKK = C * K * K, HW = H * W;
        for (int in = 0; in < N; in++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    F, HW, CKK, 1, weight.data(), CKK,
                    scratch.data() + in * CKK * HW, HW,
                    0, result.data() + in * F * HW, HW);
        }
    }

    const tensor_t& rebuild() {
        return result;
    }
};

int main() {
    std::ifstream infmt("../fmt.txt");
    std::ifstream weightfile("../dat.bin", std::ios::binary);
    int cnt_data_sets;
    infmt >> cnt_data_sets;
    int nbatch = 10;
    tensor_t indata(nbatch * 64 * 256 * 256);
    init_rand(indata);
    for (int i = 0; i < cnt_data_sets; i++) {
        int Co, Ci, Kh, Kw, total, HW;
        infmt >> Co >> Ci >> Kh >> Kw;
        HW = 64 * 256 / Co;
        total = Co * Ci * Kh * Kw;
        tensor_t weight(total);
        read_binary(weightfile, weight);
        SolverNCHW solver(weight, indata,
                {Co, Ci, Kh, Kw}, {nbatch, Ci, HW, HW});
        std::cout << "start" << std::endl;
        for (int r = 0; r < 10; r++)
            solver.compute();
        std::cout << "finished" << std::endl;
    }
    return 0;
}
