from tvm import autotvm, te, tir
import scipy.sparse
import numpy as np
import logging
import sys


def make_bsr_sparse(dense, sprate, blocksize):
    bsrdata = scipy.sparse.bsr_matrix(dense, blocksize=blocksize)
    # find partition value
    summed = bsrdata.data.sum((1, 2))
    idx = int(sprate * len(summed) + 0.5)
    val = np.partition(summed, idx)[idx]
    # filter the data
    data, indices, indptr, bsrWid = [], [], [], bsrdata.indptr[1]
    for idx, (block, indval) in enumerate(zip(bsrdata.data, bsrdata.indices)):
        if idx % bsrWid == 0:
            indptr.append(len(data))
        if block.sum() >= val:
            data.append(block)
            indices.append(indval)
    indptr.append(len(data))
    # convert format
    bsrdata2 = tuple([np.array(i) for i in [data, indices, indptr]])
    return scipy.sparse.bsr_matrix(bsrdata2, shape=dense.shape)


def unpack_bsr(bsrdata):
    return bsrdata.data, bsrdata.indices, bsrdata.indptr


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


@autotvm.template('spconv2d_3x3_gemm')
def spconv2d_3x3_gemm(N, H, W, CI, CO, nElems, bsrR, bsrC, dtype='float32'):
    Y, X, K = N*H*W, CO, 9*CI
    cfg = autotvm.get_config()
    cfg.define_split("tile_y", Y, num_outputs=3)
    cfg.define_split("tile_x", X // bsrR, num_outputs=2)
    cfg.add_flop(Y * (nElems * bsrC * bsrR * 2 - X))
    #cfg.define_split("tile_k", K, num_outputs=2)
    if cfg.is_fallback:
        cfg['tile_y'] = autotvm.task.space.SplitEntity([-1, 160, 8])
        cfg['tile_x'] = autotvm.task.space.SplitEntity([-1, 4])
    
    Data = te.placeholder((N, H, W, CI), dtype=dtype, name='Data')
    Wdat = te.placeholder((nElems, bsrR, bsrC), name='Wdat')
    Wind = te.placeholder((nElems,), dtype='int', name='Wind')
    Wptr = te.placeholder((X // bsrR + 1,), dtype='int', name='Wptr')

    @te_compute_by_func((Y, K), name='Im2Col')
    def Im2Col(row, col):
        jn, jh, jw = idxsplit(row, H, W)
        kh, kw, jc = idxsplit(col, 3, CI)
        ih, iw = jh + kh - 1, jw + kw - 1
        return tir.if_then_else(
            tir.all(0 <= ih, ih < H, 0 <= iw, iw < W),
            Data[jn, ih, iw, jc], 0)
    
    @te_compute_by_func((Y, X // bsrR, bsrR, bsrC), name='CC')
    def CC(drow, wrow, brow, bcol):
        row_start, row_end = Wptr[wrow], Wptr[wrow+1]
        elem_idx = te.reduce_axis((0, row_end - row_start), name='elem_idx')
        elem = row_start + elem_idx
        return te.sum(Im2Col[drow, Wind[elem]*bsrC + bcol] * Wdat[elem, brow, bcol], axis=elem_idx)

    k = te.reduce_axis((0, bsrC), name='k')
    C = te.compute((Y, X), lambda y, x: te.sum(CC[y, x // bsrR, x % bsrR, k], axis=k), name='C')
    
    s = te.create_schedule(C.op)
    y, x = s[C].op.axis
    yt, yo, yi = cfg['tile_y'].apply(s, C, y)
    xo, xi = s[C].split(x, factor=bsrR)
    xt, xo = cfg['tile_x'].apply(s, C, xo)
    (k,) = s[C].op.reduce_axis
    s[C].reorder(yt, xt, yo, xo, yi, xi, k)
    s[C].unroll(k)
    s[C].vectorize(xi)
    s[C].unroll(yi)

    s[CC].compute_at(s[C], xo)
    yi, xi, r, c = s[CC].op.axis
    (k,) = s[CC].op.reduce_axis
    s[CC].reorder(xi, k, yi, r, c)
    s[CC].unroll(c)
    s[CC].vectorize(r)
    s[CC].unroll(yi)
    
    s[Im2Col].compute_at(s[C], yo)
    yi, k = s[Im2Col].op.axis
    ko, ki = s[Im2Col].split(k, factor=CI)
    s[Im2Col].vectorize(ki)
    #s[Im2Col].unroll(yi)
    return s, [Data, Wdat, Wind, Wptr, C]


if __name__ == '__main__':
    nhwc_data = np.random.randint(0, 256, (10, 256, 256, 64)).astype('float32')
    weight_ohwi = np.random.rand(64, 3*3*64).astype('float32')
    spweight_ohwi = make_bsr_sparse(weight_ohwi, 0.6, (16, 1))
    ret = np.zeros((10*256*256, 64), dtype='float32')
    
    args = (10, 256, 256, 64, 64, *spweight_ohwi.data.shape, 'float32')
    task = autotvm.task.create('spconv2d_3x3_gemm', args=args, target="llvm -mcpu=cascadelake")
    print(task.config_space)

    logging.getLogger("autotvm").setLevel(logging.DEBUG)
    logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

    runner = autotvm.LocalRunner(number=4, repeat=3, timeout=20)
    runner.ref_input = [nhwc_data, *unpack_bsr(spweight_ohwi), ret]
    measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(), runner=runner)
    tuner = autotvm.tuner.GATuner(task)
    tuner.tune(
        n_trial=1000,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file("spconv2d_3x3_gemm.log")],
    )