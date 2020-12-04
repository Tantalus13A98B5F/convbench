#include <iostream>
#include <vector>
#include <cassert>
#include "dimidx.hpp"
using DI::DimIdx;

int main() {
    DimIdx<4> dImg{100, 3, 512, 512}, dConv{64, 3, 3, 3};
    size_t N, C, H, W, K, O;
    dImg.unpack(N, C, H, W);
    dConv.unpack(O, DI::None, K, DI::None);
    assert (dConv.validate(DI::Any, C, 3, K));
    std::cout << dImg.totalsize << std::endl
              << dConv.totalsize << std::endl;

    // test access
    DimIdx<4> dOut{N, O, H, W};
    do {
        const std::vector<int> vb(dImg.totalsize, 1), vc(dConv.totalsize, 2);
        std::vector<int> va(dOut.totalsize, 0);
        dOut(va, 0, 0, 0, 0) = dImg(vb, 0, 0, 0, 0) + dConv(vc, 0, 0, 0, 0);
        std::cout << dOut(va, 0, 0, 0, 0) << std::endl;
    } while (0);
    do {
        const int *ab = new int[dImg.totalsize]{1}, *ac = new int[dConv.totalsize]{2};
        int *aa = new int[dOut.totalsize];
        dOut(aa, 0, 0, 0, 0) = dImg(ab, 0, 0, 0, 0) + dConv(ac, 0, 0, 0, 0);
        std::cout << dOut(aa, 0, 0, 0, 0) << std::endl;
        delete[] ab; delete[] ac; delete[] aa;
    } while (0);
    return 0;
}
