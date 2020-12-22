#include <vector>
#include <fstream>
#include <iostream>
#include <cassert>
#include "dimidx.hpp"
#include "tensorutils.hpp"
#include "testbed.hpp"


int main() {
    std::ifstream infmt("../fmt.txt");
    std::ifstream weightfile("../dat.bin", std::ios::binary);
    int cnt_data_sets; infmt >> cnt_data_sets;
    int nbatch = 10;
    tensor_t indata(nbatch * 64 * 256 * 256);
    init_rand(indata);
    for (auto i: Range<>(0, cnt_data_sets)) {
        int Co, Ci, Kh, Kw; infmt >> Co >> Ci >> Kh >> Kw;
        DimIdx<4> dWeight {Co, Ci, Kh, Kw};
        tensor_t weight(dWeight.totalsize);
        read_binary(weightfile, weight);
        int HW = 64 * 256 / Co;
        CaseProvider cp(indata, {nbatch, Ci, HW, HW}, weight, dWeight);
        tensor_t ret1, ret2;
        {
            auto nchw = cp.newConv<NCHWMklGemmConv>();
            for (auto r: Range<>(0, 10))
                nchw->compute();
            ret1 = nchw->get_result();
        }
        {
            auto nhwc = cp.newConv<NHWCMklGemmConv>();
            for (auto r: Range<>(0, 10))
                nhwc->compute();
            ret2 = nhwc->get_result();
        }
        float diff = square_diff(ret1, ret2);
        std::cout << "diff," << diff << std::endl;
        {
            auto nchwcsr = cp.newConv<NCHWMklSpGemmConv>();
            for (auto i: Range<>(0, 5)) {
                nchwcsr->sparsity(0.35 + i * 0.15);
                for (auto r: Range<>(0, 10))
                    nchwcsr->compute();
            }
        }
    }
    return 0;
}
