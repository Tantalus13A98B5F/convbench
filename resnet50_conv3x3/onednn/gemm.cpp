#include <vector>
#include <fstream>
#include <iostream>
#include <cassert>
#include <mkl.h>
#include <mkl_spblas.h>
#include "dimidx.hpp"
#include "tensorutils.hpp"
using DI::DimIdx;

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
        DimIdx<6> dScratch {N, H, W, C, K, K};
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
                    dScratch(scratch, in, ih, iw, ic, kh, kw) =
                        (th < 0 || th >= H || tw < 0 || tw >= W) ? 0 :
                        dData(data, in, ic, th, tw);
                }
            }
        }
        auto t2 = steady_clock::now();
        int CKK = C * K * K, HW = H * W;
        for (int in = 0; in < N; in++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    F, HW, CKK, 1, weight.data(), CKK,
                    scratch.data() + in * CKK * HW, CKK,
                    0, result.data() + in * F * HW, HW);
        }
        auto t3 = steady_clock::now();
        time_convert = time_diff(t2, t1);
        time_gemm = time_diff(t3, t2);
        std::cout << "conv,nchw," << F << ',' << H
                  << ',' << time_convert * 1000
                  << ',' << time_gemm * 1000
                  << std::endl;
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
        auto aData = dData.bind(data);
        auto aWeight = dWeight.bind(weight);
        auto aData2 = dData2.bind(this->data);
        auto aWeight2 = dWeight2.bind(this->weight);

        this->data.resize(dData.totalsize);
        this->weight.resize(dWeight.totalsize);
        scratch.resize(dData.totalsize * K * K);
        result.resize(N * F * H * W);

        for (int in = 0; in < N; in++)
        for (int ic = 0; ic < C; ic++)
        for (int ih = 0; ih < H; ih++)
        for (int iw = 0; iw < W; iw++)
            aData2(in, ih, iw, ic) = aData(in, ic, ih, iw);

        for (int jf = 0; jf < F; jf++)
        for (int kh = 0; kh < K; kh++)
        for (int kw = 0; kw < K; kw++)
        for (int jc = 0; jc < C; jc++)
            aWeight2(jf, kh, kw, jc) = aWeight(jf, jc, kh, kw);
    }

    void compute() {
        DimIdx<4> dData {N, H, W, C};
        DimIdx<6> dScratch {N, H, W, K, K, C};
        auto aScratch = dScratch.bind(scratch);
        auto aData = dData.bind(data);
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
                    aScratch(in, ih, iw, kh, kw, ic) =
                        (th < 0 || th >= H || tw < 0 || tw >= W) ? 0 :
                        aData(in, th, tw, ic);
                }
            }
        }
        auto t2 = steady_clock::now();
        int CKK = C * K * K, NHW = N * H * W;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                NHW, F, CKK, 1, scratch.data(), CKK,
                weight.data(), CKK,
                0, result.data(), F);
        auto t3 = steady_clock::now();
        time_convert = time_diff(t2, t1);
        time_gemm = time_diff(t3, t2);
        std::cout << "conv,nhwc," << F << ',' << H
                  << ',' << time_convert * 1000
                  << ',' << time_gemm * 1000
                  << std::endl;
    }

    const tensor_t& rebuild() {
        // TODO
        return result;
    }
};

struct SolverSparse {
    // NCHW, CSR
    tensor_t data, scratch, result;
    sparse_matrix_t weight;
    int N, C, H, W, F, K;

    SolverSparse(const tensor_t &weight, const DimIdx<4> &dWeight,
                 const tensor_t &data, const DimIdx<4> &dData)
        : data(data) {
        dData.unpack(N, C, H, W);
        dWeight.unpack(F, DI::None, K, DI::None);
        assert (dWeight.validate(DI::Any, C, 3, K));

        // sparse weight by every filter
        int CKK = C * K * K;
        std::vector<int> pB, pE, colidx;
        tensor_t vals;
        for (int jf = 0; jf < F; jf++) {
            pB.push_back(colidx.size());
            tensor_t currow(weight.begin() + CKK*jf,
                            weight.begin() + CKK*jf + CKK);
            std::sort(currow.begin(), currow.end());
            auto flag = currow[0.4 * CKK];
            for (int jckk = 0; jckk < CKK; jckk++) {
                auto curval = weight[jf * CKK + jckk];
                if (curval > flag) {
                    vals.push_back(curval);
                    colidx.push_back(jckk);
                }
            }
            pE.push_back(colidx.size());
        }
        auto status = mkl_sparse_s_create_csr(
            &this->weight, SPARSE_INDEX_BASE_ZERO, F, CKK,
            pB.data(), pE.data(), colidx.data(), vals.data());
        assert(status == SPARSE_STATUS_SUCCESS);
    }
    
    void compute() {}

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
        SolverNCHW solver_nchw(weight, indata,
                {Co, Ci, Kh, Kw}, {nbatch, Ci, HW, HW});
        SolverNHWC solver_nhwc(weight, indata,
                {Co, Ci, Kh, Kw}, {nbatch, Ci, HW, HW});
        for (int r = 0; r < 10; r++)
        {
            solver_nchw.compute();
            solver_nhwc.compute();
        }
    }
    return 0;
}
