#include <iostream>
#include <fstream>
#include <chrono>


template <int S1, int S2, int S3, int S4, typename dtype = float>
class Tensor4D {
    typedef dtype arrty[S2][S3][S4];

public:
    arrty *buffer;

public:
    Tensor4D(): buffer(new arrty[S1]) { }

    ~Tensor4D() { delete[] buffer; }

    std::istream& load(std::istream& is) {
        for (int i = 0; i < S1; i++) {
            is.read((char*)buffer[i], sizeof(buffer[i]));
        }
        return is;
    }

    void load(const std::string& filename) {
        std::ifstream datafile;
        datafile.open(filename.c_str(), std::ifstream::binary);
        load(datafile);
        datafile.close();
    }

    dtype& operator() (int i1, int i2, int i3, int i4) {
        return buffer[i1][i2][i3][i4];
    }

    const dtype& operator() (int i1, int i2, int i3, int i4) const {
        return buffer[i1][i2][i3][i4];
    }
};


template <int N, int Cin, int H, int W, int Cout, int K>
Tensor4D<N, Cout, H, W> conv2d(const Tensor4D<N, Cin, H, W> &input,
                            const Tensor4D<Cout, Cin, K, K> &weight,
                            const Tensor4D<1, 1, 1, Cout> &bias) {
    int halfK = K / 2;
    Tensor4D<N, Cout, H, W> ret;
    for (int in = 0; in < N; in++) {
        const int bh = 2, bw = 10;
        for (int nh0 = 0; nh0 < H; nh0 += bh) {
        for (int nw0 = 0; nw0 < W; nw0 += bw) {
            const int nh1 = std::min(H, nh0 + bh);
            const int nw1 = std::min(W, nw0 + bw);
            for (int ifil = 0; ifil < Cout; ifil++) {
            for (int iker = 0; iker < Cin; iker++) {
            for (int nh = nh0; nh < nh1; nh++) {
            for (int nw = nw0; nw < nw1; nw++) {

                if (iker == 0) ret(in, ifil, nh, nw) = bias(0, 0, 0, ifil);
                float tmp = 0;
                for (int ki = 0; ki < K; ki++) {
                for (int kj = 0; kj < K; kj++) {
                    int ih = nh + ki - halfK, iw = nw + kj - halfK;
                    if (0 <= ih && ih < H && 0 <= iw && iw < W) {
                        tmp += input(in, iker, ih, iw) * weight(ifil, iker, ki, kj);
                    }
                }
                }
                ret(in, ifil, nh, nw) += tmp;

            }
            }
            }
            }
        }
        }
    }
    return ret;
}


int main() {
    Tensor4D<64, 3, 3, 3> weight;
    Tensor4D<1, 1, 1, 64> bias;
    Tensor4D<20, 3, 1000, 2000> input;
    weight.load("weight.dat");
    bias.load("bias.dat");
    input.load("input.dat");

    using namespace std::chrono;
    std::cout << "start" << std::endl;
    auto t0 = steady_clock::now();
    auto ret = conv2d(input, weight, bias);
    auto t1 = steady_clock::now();
    auto span = duration_cast<duration<double> >(t1 - t0);
    std::cout << "time: " << span.count() << std::endl;

    std::ofstream ofs("output2.dat", std::ios::binary);
    ofs.write((char*)ret.buffer, sizeof(ret.buffer[0]));

    return 0;
}
