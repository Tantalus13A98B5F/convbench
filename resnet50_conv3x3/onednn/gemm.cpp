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

    SolverNCHW(const tensor_t &weight, const DimIdx<4> &dWeight, 
               const tensor_t &data, const DimIdx<4> &dData)
        : weight(weight), data(data), time_convert(0), time_gemm(0) {
        int H2, W2;
        dData.unpack(N, C, H2, W2);
        H = H2 - 2; W = W2 - 2;
        dWeight.unpack(F, DI::None, K, DI::None);
        assert (dWeight.validate(DI::Any, C, 3, K));
        scratch.resize(N*H*W * C*K*K);
        result.resize(N * F * H * W);
    }

    void compute() {
        auto t1 = steady_clock::now();
        auto aData = DimIdx<4>{N, C, H+2, W+2}.bind(data);
        auto aScratch = DimIdx<6>{N, H, W, C, K, K}.bind(scratch);
        #pragma omp parallel for
        for (int in = 0; in < N; in++)
        for (int ic = 0; ic < C; ic++)
        for (int ih = 0; ih < H; ih++)
        for (int iw = 0; iw < W; iw++)
        for (int kh = 0; kh < K; kh++)
        for (int kw = 0; kw < K; kw++)
            aScratch(in, ih, iw, ic, kh, kw) = aData(in, ic, ih + kh, iw + kw);

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
    tensor_t weight, data, scratch, result, nchwresult;
    int N, C, H, W, F, K;
    double time_convert, time_gemm;

    SolverNHWC(const tensor_t &weight, const DimIdx<4> &dWeight, 
               const tensor_t &data, const DimIdx<4> &dData)
        : time_convert(0), time_gemm(0) {
        // unpack & check
        int H2, W2;
        dData.unpack(N, C, H2, W2);
        H = H2 - 2; W = W2 - 2;
        dWeight.unpack(F, DI::None, K, DI::None);
        assert (dWeight.validate(DI::Any, C, 3, K));
        scratch.resize(N*H*W * C*K*K);
        result.resize(N * F * H * W);
    
        // convert payload
        auto aData = dData.bind(data);
        auto aWeight = dWeight.bind(weight);
        auto aData2 = DimIdx<4>{N, H2, W2, C}.bind<true>(this->data);
        auto aWeight2 = DimIdx<4>{F, K, K, C}.bind<true>(this->weight);

        for (int in = 0; in < N; in++)
        for (int ic = 0; ic < C; ic++)
        for (int ih = 0; ih < H2; ih++)
        for (int iw = 0; iw < W2; iw++)
            aData2(in, ih, iw, ic) = aData(in, ic, ih, iw);

        for (int jf = 0; jf < F; jf++)
        for (int kh = 0; kh < K; kh++)
        for (int kw = 0; kw < K; kw++)
        for (int jc = 0; jc < C; jc++)
            aWeight2(jf, kh, kw, jc) = aWeight(jf, jc, kh, kw);
    }

    void compute() {
        auto t1 = steady_clock::now();
        auto aScratch = DimIdx<6>{N, H, W, K, K, C}.bind(scratch);
        auto aData = DimIdx<4>{N, H+2, W+2, C}.bind(data);
        #pragma omp parallel for
        for (int in = 0; in < N; in++)
        for (int ih = 0; ih < H; ih++)
        for (int iw = 0; iw < W; iw++)
        for (int kh = 0; kh < K; kh++)
        for (int kw = 0; kw < K; kw++)
        for (int ic = 0; ic < C; ic++)
            aScratch(in, ih, iw, kh, kw, ic) = aData(in, ih + kh, iw + kw, ic);

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
        auto aNHWC = DimIdx<4>{N, H, W, F}.bind(result);
        auto aNCHW = DimIdx<4>{N, F, H, W}.bind<true>(nchwresult);
        for (int in = 0; in < N; in++)
        for (int ih = 0; ih < H; ih++)
        for (int iw = 0; iw < W; iw++)
        for (int ic = 0; ic < F; ic++)
            aNCHW(in, ic, ih, iw) = aNHWC(in, ih, iw, ic);
        return nchwresult;
    }
};

struct SolverSparse {
    // NCHW, CSR
    tensor_t data, scratch, result, wvals;
    std::vector<int> ptrB, ptrE, wcols;
    sparse_matrix_t weight;
    int N, C, H, W, F, K;
    double time_convert, time_gemm;

    SolverSparse(const tensor_t &weight, const DimIdx<4> &dWeight,
                 const tensor_t &data, const DimIdx<4> &dData)
        : data(data) {
        int H2, W2;
        dData.unpack(N, C, H2, W2);
        H = H2 - 2; W = W2 - 2;
        dWeight.unpack(F, DI::None, K, DI::None);
        assert (dWeight.validate(DI::Any, C, 3, K));
        scratch.resize(N*H*W * C*K*K);
        result.resize(N * F * H * W);

        // sparse weight by every filter
        int CKK = C * K * K;
        for (int jf = 0; jf < F; jf++) {
            ptrB.push_back(wcols.size());
            tensor_t currow(weight.begin() + CKK*jf,
                            weight.begin() + CKK*jf + CKK);
            std::sort(currow.begin(), currow.end());
            auto flag = currow[0.8 * CKK];
            for (int jckk = 0; jckk < CKK; jckk++) {
                auto curval = weight[jf * CKK + jckk];
                if (curval > flag) {
                    wvals.push_back(curval);
                    wcols.push_back(jckk);
                }
            }
            ptrE.push_back(wcols.size());
        }
        auto status = mkl_sparse_s_create_csr(
            &this->weight, SPARSE_INDEX_BASE_ZERO, F, CKK,
            ptrB.data(), ptrE.data(), wcols.data(), wvals.data());
        assert(status == SPARSE_STATUS_SUCCESS);
    }
    
    void compute() {
        auto t1 = steady_clock::now();
        DimIdx<4> dData {N, C, H+2, W+2};
        DimIdx<6> dScratch {N, C, K, K, H, W};
        auto aData = dData.bind(data);
        auto aScratch = dScratch.bind(scratch);
        #pragma omp parallel for
        for (int in = 0; in < N; in++)
        for (int ic = 0; ic < C; ic++)
        for (int ih = 0; ih < H; ih++)
        for (int iw = 0; iw < W; iw++)
        for (int kh = 0; kh < K; kh++)
        for (int kw = 0; kw < K; kw++)
            aScratch(in, ic, kh, kw, ih, iw) = aData(in, ic, ih + kh, iw + kw);

        auto t2 = steady_clock::now();
        int CKK = C * K * K, HW = H * W;
        for (int in = 0; in < N; in++) {
            auto status = mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1, weight,
                {SPARSE_MATRIX_TYPE_GENERAL}, SPARSE_LAYOUT_ROW_MAJOR,
                scratch.data() + in * CKK * HW, HW, HW,
                0, result.data() + in * F * HW, HW);
            assert (status == SPARSE_STATUS_SUCCESS);
        }

        auto t3 = steady_clock::now();
        time_convert = time_diff(t2, t1);
        time_gemm = time_diff(t3, t2);
        std::cout << "conv,sparse.8," << F << ',' << H
                  << ',' << time_convert * 1000
                  << ',' << time_gemm * 1000
                  << std::endl;
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
    tensor_t indata(nbatch * 64 * 258 * 258);  // padded
    init_rand(indata);
    for (int i = 0; i < cnt_data_sets; i++) {
        int Co, Ci, Kh, Kw, total, HW;
        infmt >> Co >> Ci >> Kh >> Kw;
        HW = 64 * 256 / Co;
        total = Co * Ci * Kh * Kw;
        tensor_t weight(total);
        read_binary(weightfile, weight);
        SolverNCHW solver_nchw(
            weight, {Co, Ci, Kh, Kw}, indata, {nbatch, Ci, HW+2, HW+2});
        SolverNHWC solver_nhwc(
            weight, {Co, Ci, Kh, Kw}, indata, {nbatch, Ci, HW+2, HW+2});
        SolverSparse solver_sparse(
            weight, {Co, Ci, Kh, Kw}, indata, {nbatch, Ci, HW+2, HW+2});
        for (int r = 0; r < 10; r++)
        {
            solver_nchw.compute();
            solver_nhwc.compute();
            solver_sparse.compute();
        }
        auto retA = solver_nchw.rebuild();
        auto retB = solver_nhwc.rebuild();
        float diff = square_diff(retA, retB);
        std::cout << "diff = " << diff << std::endl;
    }
    return 0;
}
