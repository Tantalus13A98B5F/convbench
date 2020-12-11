CXX := icpc
DNNLPATH := ${HOME}/dnnl_lnx_1.7.0_cpu_iomp
CFLAGS := -O3 -xHost -std=c++11
DNNLLD := -L${DNNLPATH}/lib -Wl,-rpath=${DNNLPATH}/lib -ldnnl
DNNLCF := -I${DNNLPATH}/include
MKLCF := -mkl=parallel
MKLLD := -liomp5 -lpthread -lm -ldl -qopenmp
TARGETS := onednn.x gemm.x

all: ${TARGETS}

onednn.x: onednn.cpp tensorutils.hpp
	${CXX} ${CFLAGS} ${DNNLCF} $< -o $@ ${DNNLLD}

gemm.x: gemm.cpp tensorutils.hpp dimidx.hpp
	${CXX} ${CFLAGS} ${MKLCF} $< -o $@ ${MKLLD}

clean:
	rm -f ${TARGETS}

.PHONY: clean
