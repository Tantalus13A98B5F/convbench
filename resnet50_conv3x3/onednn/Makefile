CXX := icpc
DNNLPATH := ${HOME}/dnnl_lnx_1.7.0_cpu_iomp
CFLAGS := -O2 -std=c++11
DNNLLD := -L${DNNLPATH}/lib -Wl,-rpath=${DNNLPATH}/lib -ldnnl
DNNLCF := -I${DNNLPATH}/include
MKLCF := -mkl=parallel
MKLLD := -liomp5 -lpthread -lm -ldl

all: gemm.x

onednn.x: onednn.cpp
	${CXX} ${CFLAGS} ${DNNLCF} $^ -o $@ ${DNNLLD}

gemm.x: gemm.cpp
	${CXX} ${CFLAGS} ${MKLCF} $^ -o $@ ${MKLLD}

clean:
	rm -f onednn.x

.PHONY: clean
