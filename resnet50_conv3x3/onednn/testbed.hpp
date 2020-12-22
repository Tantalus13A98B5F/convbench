#ifndef _TESTBED_H_
#define _TESTBED_H_
#include "dimidx.hpp"
#include "tensorutils.hpp"
#include <memory>
#include <mkl.h>
#include <mkl_spblas.h>
using DI::Range;
using DI::DimIdx;


#define CONSTSTR(name,str) const char* name() { return str; }
#define FOR1(idx, start, stop) for (int idx = start; idx < stop; idx++)


class CaseProvider {
    const tensor_t &data, &weight;
    DimIdx<4> dData, dWeight;

public:
    CaseProvider(const tensor_t &data,   const DimIdx<4> &dData,
                 const tensor_t &weight, const DimIdx<4> &dWeight)
        : data(data), weight(weight), dData(dData), dWeight(dWeight)
    {}

    template <typename ConvClass>
    std::unique_ptr<ConvClass> newConv() {
        auto ret = new ConvClass;
        ret->check_size(dData, dWeight);
        ret->prepare_data(data, weight);
        return std::unique_ptr<ConvClass>(ret);
    }
};


class NCHWDirectConv {
protected:
    int N, C, H, W, F, K;
    tensor_t data, weight, result;

    virtual CONSTSTR(fmt, "NCHW")
    virtual CONSTSTR(alg, "direct")
    virtual CONSTSTR(impl, "raw")
    virtual CONSTSTR(spfmt, "none")
    virtual float sparsity() { return 0; }

    virtual void im2col() {}

    virtual void compute_kernel() {
        auto aData = DimIdx<4>{N, C, H+2, W+2}.bind(data);
        auto aWeight = DimIdx<4>{F, C, K, K}.bind(weight);
        auto aRet = DimIdx<4>{N, F, H, W}.bind(result);
        #pragma omp parallel for collapse(2)
        FOR1 (in, 0, N)
        FOR1 (jf, 0, F)
        FOR1 (ih, 0, H)
        FOR1 (iw, 0, W)
        {
            tensor_t::value_type sum = 0;
            FOR1 (ic, 0, C)
            FOR1 (kh, 0, K)
            FOR1 (kw, 0, K)
            {
                sum += aData(in, ic, ih+kh, iw+kw) * aWeight(jf, ic, kh, kw);
            }
            aRet(in, jf, ih, iw) = sum;
        }
    }

public:
    virtual ~NCHWDirectConv() {}

    void check_size(const DimIdx<4> &dData, const DimIdx<4> &dWeight) {
        dData.unpack(N, C, H, W);
        dWeight.unpack(F, DI::None, K, DI::None);
        assert (dWeight.validate(DI::Any, C, 3, K));
        result.resize(N * F * H * W);
    }

    virtual void prepare_data(const tensor_t &data, const tensor_t &weight) {
        auto aOrig = DimIdx<4>{N, C, H, W}.bind(data);
        auto aNew = DimIdx<4>{N, C, H+2, W+2}.bind<true>(this->data);
        FOR1 (in, 0, N)
        FOR1 (ic, 0, C)
        FOR1 (ih, 0, H)
        FOR1 (iw, 0, W)
            aNew(in, ic, ih+1, iw+1) = aOrig(in, ic, ih, iw);
        
        this->weight = weight;
    }

    void compute() {
        auto t1 = steady_clock::now();
        im2col();
        auto t2 = steady_clock::now();
        compute_kernel();
        auto t3 = steady_clock::now();
        auto time_convert = time_diff(t2, t1) * 1000;
        auto time_compute = time_diff(t3, t2) * 1000;
        std::cout << "conv," << fmt() << ',' << alg() << ',' << impl()
                  << ',' << spfmt() << ',' << sparsity()
                  << ',' << F << ',' << H
                  << ',' << time_convert << ',' << time_compute
                  << std::endl;
    }

    virtual tensor_t get_result() {
        return result;
    }
};


class NCHWMklGemmConv: public NCHWDirectConv {
protected:
    tensor_t scratch;

    CONSTSTR(alg, "gemm")
    CONSTSTR(impl, "mkl")

    void im2col() {
        auto aData = DimIdx<4>{N, C, H+2, W+2}.bind(data);
        auto aScratch = DimIdx<6>{N, H, W, C, K, K}.bind<true>(scratch);
        FOR1 (in, 0, N)
        FOR1 (ic, 0, C)
        FOR1 (ih, 0, H)
        FOR1 (iw, 0, W)
        FOR1 (kh, 0, K)
        FOR1 (kw, 0, K)
            aScratch(in, ih, iw, ic, kh, kw) = aData(in, ic, ih + kh, iw + kw);
    }

    void compute_kernel() {
        int CKK = C * K * K, HW = H * W;
        FOR1 (in, 0, N) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    F, HW, CKK, 1, weight.data(), CKK,
                    scratch.data() + in * CKK * HW, CKK,
                    0, result.data() + in * F * HW, HW);
        }
    }
};


class NHWCMklGemmConv: public NCHWMklGemmConv {
protected:
    CONSTSTR(fmt, "NHWc")

    void im2col() {
        auto aScratch = DimIdx<6>{N, H, W, K, K, C}.bind<true>(scratch);
        auto aData = DimIdx<4>{N, H+2, W+2, C}.bind(data);
        FOR1 (in, 0, N)
        FOR1 (ih, 0, H)
        FOR1 (iw, 0, W)
        FOR1 (kh, 0, K)
        FOR1 (kw, 0, K)
        FOR1 (ic, 0, C)
            aScratch(in, ih, iw, kh, kw, ic) = aData(in, ih + kh, iw + kw, ic);
    }

    void compute_kernel() {
        int CKK = C * K * K, NHW = N * H * W;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                NHW, F, CKK, 1, scratch.data(), CKK,
                weight.data(), CKK,
                0, result.data(), F);
    }

public:
    void prepare_data(const tensor_t &data, const tensor_t &weight) {
        auto dOrig = DimIdx<4>{N, C, H, W}.bind(data);
        auto dNew = DimIdx<4>{N, H+2, W+2, C}.bind<true>(this->data);
        FOR1 (in, 0, N)
        FOR1 (ic, 0, C)
        FOR1 (ih, 0, H)
        FOR1 (iw, 0, W)
            dNew(in, ih+1, iw+1, ic) = dOrig(in, ic, ih, iw);
        
        auto wOrig = DimIdx<4>{F, C, K, K}.bind(weight);
        auto wNew = DimIdx<4>{F, K, K, C}.bind<true>(this->weight);
        FOR1 (jf, 0, F)
        FOR1 (ic, 0, C)
        FOR1 (kh, 0, K)
        FOR1 (kw, 0, K)
            wNew(jf, kh, kw, ic) = wOrig(jf, ic, kh, kw);
    }

    tensor_t get_result() {
        tensor_t nchwresult;
        auto aNHWC = DimIdx<4>{N, H, W, F}.bind(result);
        auto aNCHW = DimIdx<4>{N, F, H, W}.bind<true>(nchwresult);
        FOR1 (in, 0, N)
        FOR1 (ih, 0, H)
        FOR1 (iw, 0, W)
        FOR1 (ic, 0, F)
            aNCHW(in, ic, ih, iw) = aNHWC(in, ih, iw, ic);
        return nchwresult;
    }
};

class NCHWMklSpGemmConv: public NCHWMklGemmConv {
protected:
    std::vector<int> ptrB, ptrE, wcols;
    std::unique_ptr<sparse_matrix_t> spweight;
    tensor_t wvals;
    float sprate;

    CONSTSTR(spfmt, "csr")
    float sparsity() { return sprate; }

    void im2col() {
        auto aScratch = DimIdx<6>{N, C, K, K, H, W}.bind<true>(scratch);
        auto aData = DimIdx<4>{N, C, H+2, W+2}.bind(data);
        FOR1 (in, 0, N)
        FOR1 (ic, 0, C)
        FOR1 (ih, 0, H)
        FOR1 (iw, 0, W)
        FOR1 (kh, 0, K)
        FOR1 (kw, 0, K)
            aScratch(in, ic, kh, kw, ih, iw) = aData(in, ic, ih + kh, iw + kw);
    }

    void compute_kernel() {
        int CKK = C * K * K, HW = H * W;
        FOR1 (in, 0, N) {
            auto status = mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1, *(spweight.get()),
                {SPARSE_MATRIX_TYPE_GENERAL}, SPARSE_LAYOUT_ROW_MAJOR,
                scratch.data() + in * CKK * HW, HW, HW,
                0, result.data() + in * F * HW, HW);
            assert (status == SPARSE_STATUS_SUCCESS);
        }
    }

public:
    void sparsity(float s) {
        sprate = s;
        spweight.reset(new sparse_matrix_t());
        ptrB.clear(); ptrE.clear(); wcols.clear();
        wvals.clear();
        int CKK = C * K * K;
        FOR1 (jf, 0, F) {
            ptrB.push_back(wcols.size());
            tensor_t currow(weight.begin() + CKK*jf,
                            weight.begin() + CKK*jf + CKK);
            std::sort(currow.begin(), currow.end());
            auto flag = currow[sprate * CKK];
            FOR1 (jckk, 0, CKK) {
                auto curval = weight[jf * CKK + jckk];
                if (curval > flag) {
                    wvals.push_back(curval);
                    wcols.push_back(jckk);
                }
            }
            ptrE.push_back(wcols.size());
        }
        auto status = mkl_sparse_s_create_csr(
            spweight.get(), SPARSE_INDEX_BASE_ZERO, F, CKK,
            ptrB.data(), ptrE.data(), wcols.data(), wvals.data());
        assert(status == SPARSE_STATUS_SUCCESS);
    }
};


#undef FOR1
#undef CONSTSTR

#endif  // _TESTBED_H_