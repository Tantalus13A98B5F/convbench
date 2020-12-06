#ifndef _TENSORUTILS_HPP_
#define _TENSORUTILS_HPP_
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
using namespace std::chrono;
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

template <typename TPoint, typename retTy = double>
retTy time_diff(TPoint t2, TPoint t1) {
    return duration_cast<duration<retTy>>(t2 - t1).count();
}
#endif  // _TENSORUTILS_HPP_