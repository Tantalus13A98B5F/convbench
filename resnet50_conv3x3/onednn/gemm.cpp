#include <vector>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>
#include <cassert>
#include "mkl.h"
#include "dimidx.hpp"
using namespace std::chrono;
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
    double time_convert, time_gemm;

    SolverNCHW(const tensor_t &weight, const tensor_t &data,
               const DimIdx<4> &dWeight, const DimIdx<4> &dData)
        : weight(weight), data(data), time_convert(0), time_gemm(0) {
        dData.unpack(N, C, H, W);
        dWeight.unpack(F, DI::None, K, DI::None);
        assert (dWeight.validate(DI::Any, C, 3, K));
        scratch.resize(dData.totalsize * K * K);
        result.resize(N * F * H * W);
    }

    void compute() {
        DimIdx<4> dData {N, C, H, W};
        DimIdx<6> dScratch {N, C, K, K, H, W};
        auto t1 = steady_clock::now();
        #pragma omp parallel for
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
                        (th < 0 || th >= H || tw < 0 || tw >= W) ? 0 :
                        dData(data, in, ic, th, tw);
                }
            }
        }
        auto t2 = steady_clock::now();
        int CKK = C * K * K, HW = H * W;
        for (int in = 0; in < N; in++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    F, HW, CKK, 1, weight.data(), CKK,
                    scratch.data() + in * CKK * HW, HW,
                    0, result.data() + in * F * HW, HW);
        }
        auto t3 = steady_clock::now();
        time_convert = duration_cast<duration<double>>(t2 - t1).count();
        time_gemm = duration_cast<duration<double>>(t3 - t2).count();
    }

    const tensor_t& rebuild() {
        return result;
    }
};

struct SolverNHWC {
    tensor_t weight, data, scratch, result;
    int N, C, H, W, F, K;
    double time_convert, time_gemm;

    SolverNHWC(const tensor_t &weight, const tensor_t &data,
               const DimIdx<4> &dWeight, const DimIdx<4> &dData)
        : time_convert(0), time_gemm(0) {
        // unpack & check
        dData.unpack(N, C, H, W);
        dWeight.unpack(F, DI::None, K, DI::None);
        assert (dWeight.validate(DI::Any, C, 3, K));
        // convert payload
        DimIdx<4> dData2 {N, H, W, C}, dWeight2 {F, K, K, C};
        this->data.resize(dData.totalsize);
        for (int in = 0; in < N; in++)
        for (int ic = 0; ic < C; ic++)
        for (int ih = 0; ih < H; ih++)
        for (int iw = 0; iw < W; iw++)
            dData2(this->data, in, ih, iw, ic) = dData(data, in, ic, ih, iw);
        this->weight.resize(dWeight.totalsize);
        for (int jf = 0; jf < F; jf++)
        for (int kh = 0; kh < K; kh++)
        for (int kw = 0; kw < K; kw++)
        for (int jc = 0; jc < C; jc++)
            dWeight2(this->weight, jf, kh, kw, jc) = dWeight(weight, jf, jc, kh, kw);
        scratch.resize(dData.totalsize * K * K);
        result.resize(N * F * H * W);
    }

    void compute() {
        DimIdx<4> dData {N, H, W, C};
        DimIdx<6> dScratch {N, H, W, K, K, C};
        auto t1 = steady_clock::now();
        #pragma omp parallel for
        for (int in = 0; in < N; in++)
        for (int ih = 0; ih < H; ih++)
        for (int iw = 0; iw < W; iw++)
        {
            for (int kh = 0; kh < K-1; kh++)
            for (int kw = 0; kw < K-1; kw++)
            {
                int th = ih + kh - 1, tw = iw + kw - 1;
                for (int ic = 0; ic < C; ic++)
                {
                    dScratch(scratch, in, ih, iw, kh, kw, ic) =
                        (th < 0 || th >= H || tw < 0 || tw >= W) ? 0 :
                        dData(data, in, th, tw, ic);
                }
            }
        }
        auto t2 = steady_clock::now();
        int CKK = C * K * K, NHW = N * H * W;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                NHW, F, CKK, 1, scratch.data(), CKK,
                weight.data(), F,
                0, result.data(), F);
        auto t3 = steady_clock::now();
        time_convert = duration_cast<duration<double>>(t2 - t1).count();
        time_gemm = duration_cast<duration<double>>(t3 - t2).count();
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
        SolverNHWC solver(weight, indata,
                {Co, Ci, Kh, Kw}, {nbatch, Ci, HW, HW});
        for (int r = 0; r < 10; r++)
        {
            solver.compute();
            std::cout << "conv,nhwc," << Co << ',' << HW
                      << ',' << solver.time_convert * 1000
                      << ',' << solver.time_gemm * 1000
                      << std::endl;
        }
    }
    return 0;
}
