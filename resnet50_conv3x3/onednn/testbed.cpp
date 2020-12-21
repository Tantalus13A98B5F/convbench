#include "dimidx.hpp"
#include "tensorutils.hpp"


class NCHWDirectConv {
protected:
    int N, C, H, W, F, K;
    tensor_t data, weight, result;

    const char* fmt() { return "NCHW"; }
    const char* alg() { return "direct"; }
    const char* impl() { return "raw"; }
    const char* spfmt() { return "none"; }
    float sparsity() { return 0; }

    void check_size(const DimIdx<4> &dData, const DimIdx<4> &dWeight) {
        dData.unpack(N, C, H, W);
        dWeight.unpack(F, DI::None, K, DI::None);
        assert (dWeight.validate(DI::Any, C, 3, K));
        size_alloc();
    }

    void size_alloc() {
        result.resize(N * F * H * W);
    }

    void format_convert(const tensor_t &data, const tensor_t &weight) {
        auto aOrig = DI::DimIdx<4>{N, C, H, W}.bind(data);
        auto aNew = DI::DimIdx<4>{N, C, H+2, W+2}.bind<true>(this->data);
        for (auto in: Range<>(0, N))
        for (auto ic: Range<>(0, C))
        for (auto ih: Range<>(0, H))
        for (auto iw: Range<>(0, W))
            aNew(in, ic, ih, iw) = aOrig(in, ic, ih, iw);
        
        this->weight = weight;
    }

    void im2col() {}

    void compute_kernel() {
        auto aData = DI::DimIdx<4>{N, C, H+2, W+2}.bind(data);
        auto aWeight = DI::DimIdx<4>{F, C, K, K}.bind(weight);
        auto aRet = DI::DimIdx<4>{N, F, H, W}.bind(result);
        for (auto in: Range<>(0, N))
        for (auto jf: Range<>(0, F))
        for (auto ih: Range<>(0, H))
        for (auto iw: Range<>(0, W))
        {
            tensor_t::value_type sum = 0;
            for (auto ic: Range<>(0, C))
            for (auto kh: Range<>(0, K))
            for (auto kw: Range<>(0, K))
            {
                sum += aData(in, ic, ih+kh, iw+kw) * aWeight(jf, ic, kh, kw);
            }
            aRet(in, jf, ih, iw) = sum;
        }
    }

public:
    NCHWDirectConv(const tensor_t &data,   const DI::DimIdx<4> &dData,
                   const tensor_t &weight, const DI::DimIdx<4> &dWeight)
    {
        check_size(dData, dWeight);
        format_convert(data, weight);
    }

    virtual ~NCHWDirectConv() {}

    void compute() {
        auto t1 = steady_clock::now();
        im2col();
        auto t2 = steady_clock::now();
        compute_kernel();
        auto t3 = steady_clock::now();
        auto time_convert = time_diff(t2, t1) * 1000;
        auto time_compute = time_diff(t3, t2) * 1000;
        std::cout << "conv," << fmt() << ',' << alg() << ',' << impl()
                  << ',' << spfmt() << ',' << sparisity()
                  << ',' << F << ',' << H
                  << ',' << time_convert << ',' << time_compute
                  << std::endl;
    }

    tensor_t result() {
        return result;
    }
};


class NCHWMklGemmConv: public NCHWDirectConv {
protected:
    tensor_t scratch;

    const char* alg() { return "gemm"; }
    const char* impl() { return "mkl"; }

    void size_alloc() {
        NCHWDirectConv::size_alloc();
        scratch.resize(N * H * W * C * K * K);
    }

    void im2col() {
        auto aData = DimIdx<4>{N, C, H+2, W+2}.bind(data);
        auto aScratch = DimIdx<6>{N, H, W, C, K, K}.bind(scratch);
        for (auto in: Range<>(0, N))
        for (auto ic: Range<>(0, C))
        for (auto ih: Range<>(0, H))
        for (auto iw: Range<>(0, W))
        for (auto kh: Range<>(0, K))
        for (auto kw: Range<>(0, K))
            aScratch(in, ih, iw, ic, kh, kw) = aData(in, ic, ih + kh, iw + kw);
    }

    void compute_kernel() {
        int CKK = C * K * K, HW = H * W;
        for (auto in: Range<>(0, N)) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    F, HW, CKK, 1, weight.data(), CKK,
                    scratch.data() + in * CKK * HW, CKK,
                    0, result.data() + in * F * HW, HW);
        }
    }
};


class NHWCMklGemmConv: public NCHWMklGemmConv {
protected:
    const char* fmt() { return "NHWc"; }

    void format_convert(const tensor_t &data, const tensor_t &weight) {
        auto dOrig = DI::DimIdx<4>{N, C, H, W}.bind(data);
        auto dNew = DI::DimIdx<4>{N, H+2, W+2, C}.bind<true>(this->data);
        for (auto in: Range<>(0, N))
        for (auto ic: Range<>(0, C))
        for (auto ih: Range<>(0, H))
        for (auto iw: Range<>(0, W))
            dNew(in, ih, iw, ic) = dOrig(in, ic, ih, iw);
        
        auto wOrig = DI::DimIdx<4>{F, C, K, K}.bind(weight);
        auto wNew = DI::DimIdx<4>{F, K, K, C}.bind<true>(this->weight);
        for (auto jf: Range<>(0, F))
        for (auto ic: Range<>(0, C))
        for (auto kh: Range<>(0, K))
        for (auto kw: Range<>(0, K))
            wNew(jf, kh, kw, ic) = wOrig(jf, ic, kh, kw);
    }

    void im2col() {
        auto aScratch = DimIdx<6>{N, H, W, K, K, C}.bind(scratch);
        auto aData = DimIdx<4>{N, H+2, W+2, C}.bind(data);
        for (auto in: Range<>(0, N))
        for (auto ih: Range<>(0, H))
        for (auto iw: Range<>(0, W))
        for (auto kh: Range<>(0, K))
        for (auto kw: Range<>(0, K))
        for (auto ic: Range<>(0, C))
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
    tensor_t result() {
        tensor_t nchwresult;
        auto aNHWC = DimIdx<4>{N, H, W, F}.bind(result);
        auto aNCHW = DimIdx<4>{N, F, H, W}.bind<true>(nchwresult);
        for (auto in: Range<>(0, N))
        for (auto ih: Range<>(0, H))
        for (auto iw: Range<>(0, W))
        for (auto ic: Range<>(0, F))
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

    const char* spfmt() { return "csr"; }
    float sparsity() { return sprate; }

    void im2col() {
        auto aScratch = DimIdx<6>{N, C, K, K, H, W}.bind(scratch);
        auto aData = DimIdx<4>{N, C, H+2, W+2}.bind(data);
        for (auto in: Range<>(0, N))
        for (auto ih: Range<>(0, H))
        for (auto iw: Range<>(0, W))
        for (auto kh: Range<>(0, K))
        for (auto kw: Range<>(0, K))
        for (auto ic: Range<>(0, C))
            aScratch(in, ih, iw, kh, kw, ic) = aData(in, ih + kh, iw + kw, ic);
    }

    void compute_kernel() {
        int CKK = C * K * K, HW = H * W;
        for (int in = 0; in < N; in++) {
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
        for (auto jf: Range<>(0, F)) {
            ptrB.push_back(wcols.size());
            tensor_t currow(weight.begin() + CKK*jf,
                            weight.begin() + CKK*jf + CKK);
            std::sort(currow.begin(), currow.end());
            auto flag = currow[sprate * CKK];
            for (auto jckk: Range<>(0, CKK)) {
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