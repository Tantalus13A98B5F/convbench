#include <iostream>
#include <fstream>
#include <chrono>


class Tensor4D {
public:
    const int S1, S2, S3, S4;
    const long long S34, S234, S1234;
    typedef float dtype;
    dtype *buffer;

public:
    Tensor4D(int s1, int s2, int s3, int s4)
        : S1(s1), S2(s2), S3(s3), S4(s4),
          S34(S3 * S4), S234(S2 * S34),
          S1234(S1 * S234) {
        buffer = new dtype[S1234];
    }

    ~Tensor4D() {
        delete[] buffer;
    }

    std::istream& load(std::istream& is) {
        for (int i = 0; i < S1234; i++) {
            is.read((char*)(buffer + i), 4);
        }
        return is;
    }

    void load(const std::string& filename) {
        std::ifstream datafile;
        datafile.open(filename.c_str(), std::ifstream::binary);
        load(datafile);
        datafile.close();
    }

    void unpack_size(int &s1, int &s2, int &s3, int &s4) const {
        s1 = S1; s2 = S2; s3 = S3; s4 = S4;
    }

    dtype& operator() (int i1, int i2, int i3, int i4) {
        return buffer[i1 * S234 + i2 * S34 + i3 * S4 + i4];
    }

    const dtype& operator() (int i1, int i2, int i3, int i4) const {
        return buffer[i1 * S234 + i2 * S34 + i3 * S4 + i4];
    }
};


Tensor4D conv2d(const Tensor4D& input, const Tensor4D& weight, const Tensor4D& bias) {
    int N, Cin, Cout, H, W, K;
    input.unpack_size(N, Cin, H, W);
    weight.unpack_size(Cout, Cin, K, K);
    //Cout = bias.S4;
    int halfK = K / 2;
    Tensor4D ret(N, Cout, H, W);
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
    Tensor4D weight(64, 3, 3, 3), bias(1, 1, 1, 64);
    Tensor4D input(20, 3, 1000, 2000);
    weight.load("weight.dat");
    bias.load("bias.dat");
    input.load("input.dat");

    using namespace std::chrono;
    std::cout << "start" << std::endl;
    auto t0 = steady_clock::now();
    Tensor4D ret = conv2d(input, weight, bias);
    auto t1 = steady_clock::now();
    auto span = duration_cast<duration<double> >(t1 - t0);
    std::cout << "time: " << span.count() << std::endl;

    std::ofstream ofs("output2.dat", std::ios::binary);
    ofs.write((char*)&ret(0, 0, 0, 0), sizeof(Tensor4D::dtype) * ret.S234);
    return 0;
}
