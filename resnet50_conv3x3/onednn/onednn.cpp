#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <random>
#include "dnnl.hpp"
using namespace dnnl;

typedef std::unordered_map<int, memory> primargs_t;
typedef std::vector<float> tensor_t;

template <typename dtype>
void write_to_dnnl_memory(dtype *handle, dnnl::memory &mem) {
    auto handle2 = (uint8_t*)handle;
    dnnl::engine eng = mem.get_engine();
    size_t bytes = mem.get_desc().get_size();

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
        for (size_t i = 0; i < bytes; ++i)
            dst[i] = handle2[i];
    }
}


struct runner {
    using tag = memory::format_tag;
    using dt = memory::data_type;

    std::vector<std::pair<primitive, primargs_t>> prims;
    engine eng;
    stream st;

    runner(): eng(engine::kind::cpu, 0), st(eng) { }

    void test_conv(int N, int C, int HW,
            const tensor_t &weights, const tensor_t &image) {
        memory::dims src_tz = {N, C, HW, HW};
        memory::dims weights_tz = {C, C, 3, 3};
        memory::dims bias_tz = {C};
        memory::dims dst_tz = {N, C, HW, HW};
        memory::dims strides = {1, 1};
        memory::dims padding = {1, 1};

        // create memory for user data
        const tensor_t bias(C);
        auto user_input_memory = memory({{src_tz}, dt::f32, tag::nchw}, eng);
        auto user_weights_memory = memory({{weights_tz}, dt::f32, tag::oihw}, eng);
        auto user_bias_memory = memory({{bias_tz}, dt::f32, tag::x}, eng);
        write_to_dnnl_memory(image.data(), user_input_memory);
        write_to_dnnl_memory(weights.data(), user_weights_memory);
        write_to_dnnl_memory(bias.data(), user_bias_memory);

        // create memory descriptors for convolution data w/ no specified format
        auto src_md = memory::desc({src_tz}, dt::f32, tag::nChw16c);
        auto bias_md = memory::desc({bias_tz}, dt::f32, tag::any);
        auto weights_md = memory::desc({weights_tz}, dt::f32, tag::any);
        auto dst_md = memory::desc({dst_tz}, dt::f32, tag::any);

        // create a convolution
        auto desc = convolution_forward::desc(prop_kind::forward_inference,
                algorithm::convolution_direct, src_md, weights_md,
                bias_md, dst_md, strides, padding, padding);
        auto prim_desc = convolution_forward::primitive_desc(desc, eng);

        auto src_memory = user_input_memory;
        if (prim_desc.src_desc() != user_input_memory.get_desc()) {
            src_memory = memory(prim_desc.src_desc(), eng);
            reorder(user_input_memory, src_memory).execute(st, 
                    {{DNNL_ARG_FROM, user_input_memory},
                     {DNNL_ARG_TO, src_memory}});
        }

        auto weights_memory = user_weights_memory;
        if (prim_desc.weights_desc() != user_weights_memory.get_desc()) {
            weights_memory = memory(prim_desc.weights_desc(), eng);
            reorder(user_weights_memory, weights_memory).execute(st,
                    {{DNNL_ARG_FROM, user_weights_memory},
                     {DNNL_ARG_TO, weights_memory}});
        }

        st.wait();
        auto dst_memory = memory(prim_desc.dst_desc(), eng);

        // create convolution primitive and add it to net
        prims.push_back({
            convolution_forward(prim_desc),
            {{DNNL_ARG_SRC, src_memory},
             {DNNL_ARG_WEIGHTS, weights_memory},
             {DNNL_ARG_BIAS, user_bias_memory},
             {DNNL_ARG_DST, dst_memory}}
        });
    }

    void exec(int times) {
        for (int i = 0; i < times; i++) {
            for (auto &it: prims) {
                it.first.execute(st, it.second);
            }
        }
        st.wait();
    }
};


int main() {
    std::ifstream infmt("../fmt.txt");
    std::ifstream weightfile("../dat.bin", std::ios::binary);
    int cnt_data_sets;
    infmt >> cnt_data_sets;
    int nbatch = 10;
    tensor_t indata(nbatch * 64 * 256 * 256);
    std::minstd_rand0 randgen(0);
    for (int i = 0; i < indata.size(); i++) {
        indata[i] = randgen();
    }
    runner robj;
    for (int i = 0; i < cnt_data_sets; i++) {
        int Co, Ci, Kh, Kw, total, HW;
        infmt >> Co >> Ci >> Kh >> Kw;
        HW = 64 * 256 / Co;
        total = Co * Ci * Kh * Kw;
        tensor_t weight(total);
        weightfile.read((char*) weight.data(), total * sizeof(tensor_t::value_type));
        robj.test_conv(nbatch, Co, HW, weight, indata);
    }
    robj.exec(10);
    return 0;
}
