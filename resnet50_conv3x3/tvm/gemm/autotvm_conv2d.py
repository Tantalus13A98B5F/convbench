from tvm import autotvm, te, tir
import logging
import sys


def idxsplit(idx, dim, *dim2):
    if dim2:
        idx, *lower = idxsplit(idx, *dim2)
    else:
        lower = []
    return (idx // dim, idx % dim, *lower)


class te_compute_by_func:
    def __init__(self, shape, **kwargs):
        self.shape = shape
        self.kwargs = kwargs
       
    def __call__(self, func):
        return te.compute(self.shape, func, **self.kwargs)


@autotvm.template('conv2d_3x3_gemm')
def conv2d_3x3_gemm(N, H, W, CI, CO, dtype='float32'):
    Y, X, K = N*H*W, CO, 9*CI
    cfg = autotvm.get_config()
    cfg.define_split("tile_y", Y, num_outputs=3)
    cfg.define_split("tile_x", X, num_outputs=3)
    cfg.define_split("tile_k", K, num_outputs=2)
    if cfg.is_fallback:
        pass

    data = te.placeholder((N, H, W, CI), dtype=dtype)
    weight = te.placeholder((X, K), dtype=dtype)

    @te_compute_by_func((Y, K), name='im2col')
    def im2col(row, col):
        jn, jh, jw = idxsplit(row, H, W)
        kh, kw, jc = idxsplit(col, 3, CI)
        ih, iw = jh + kh - 1, jw + kw - 1
        return tir.if_then_else(
            tir.all(0 <= ih, ih < H, 0 <= iw, iw < W),
            data[jn, ih, iw, jc], 0)
    
    packw_bn = cfg["tile_x"].size[-1]
    packw = te.compute((X//packw_bn, K, packw_bn),
        lambda xo, k, xi: weight[xo * packw_bn + xi, k],
        name="packed_weight")
    
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((Y, X),
        lambda y, x: te.sum(im2col[y, k] * packw[x//packw_bn, k, x%packw_bn], axis=k),
        name="dense_pack")
    
    s = te.create_schedule(C.op)
    CC = s.cache_write(C, "global")
    
    y, x = s[C].op.axis
    yt, yo, yi = cfg["tile_y"].apply(s, C, y)
    xt, xo, xi = cfg["tile_x"].apply(s, C, x)
    s[C].reorder(yt, xt, yo, xo, yi, xi)
    #xyt = s[C].fuse(yt, xt)
    #s[C].parallel(xyt)
    #xyo = s[C].fuse(yo, xo)
    s[C].unroll(yi)
    s[C].vectorize(xi)

    s[CC].compute_at(s[C], xo)
    yi, xi = s[CC].op.axis
    (k,) = s[CC].op.reduce_axis
    ko, ki = cfg["tile_k"].apply(s, CC, k)
    s[CC].reorder(ko, ki, yi, xi)
    s[CC].vectorize(xi)
    s[CC].unroll(yi)
    s[CC].unroll(ki)
    
    s[im2col].compute_at(s[C], yo)
    yi, k = s[im2col].op.axis
    ko, ki = s[im2col].split(k, factor=CI)
    s[im2col].vectorize(ki)
    #s[im2col].unroll(yi)

    xo, k, xi = s[packw].op.axis
    s[packw].reorder(xo, xi, k)
    #s[packw].parallel(xo)
    return s, [data, weight, C]


if __name__ == '__main__':
    task = autotvm.task.create('conv2d_3x3_gemm', args=(10, 256, 256, 64, 64, 'float32'), target="llvm -mcpu=cascadelake")
    print(task.config_space)

    logging.getLogger("autotvm").setLevel(logging.DEBUG)
    logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(number=4, repeat=3, timeout=20))
    tuner = autotvm.tuner.GATuner(task)
    tuner.tune(
        n_trial=5000,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file("conv2d_3x3_gemm.log")],
    )