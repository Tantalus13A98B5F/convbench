from tvm import autotvm, te, tir
from functools import partial, reduce


'''# topi.x86.conv2d_NCHWc (AutoTVM)

    {N, OC, H}:para, ow =>
        IC, kh, iic, kw:unroll, iw:unroll, oc:vec =>
            @CC = N, OC, H, {ow, iw}/W, oc  // {IC, iic}/ic, kh, kw
        iw, oc:vec =>
            @CO = N, OC, H, {ow, iw}/W, oc

    {N, OC, H}:para, ow =>
        IC, kh, kw, iic, iw:unroll, oc:vec =>
            @CC = N, OC, H, {ow, iw}/W, oc  // {IC, iic}/ic, kh, kw
        iw, oc:vec =>
            @CO = N, OC, H, {ow, iw}/W, oc
'''


@autotvm.template('conv2d_3x3_gemm')
def conv2d_3x3_gemm(N, H, W, CI, CO, dtype='float32'):
    '''# My Conv2d_3x3_gemm
                
        yt, xt, yo =>
            yi, k9, ci:vec =>
                @im2col = {yt, yo, yi}/y, {k9, ci}/k
            xo =>
                ko, ki:unroll, yi:unroll, xi:vec =>
                    @ccache = {yt, yo, yi}/y, {xt, xo, xi}/x  // {ko, ki}k
                yi:unroll, xi:vec =>
                    @cout = {yt, yo, yi}/y, {xt, xo, xi}/x
    '''
    Y, X, K = N*H*W, CO, 9*CI
    cfg = autotvm.get_config()
    cfg.define_split("tile_y", Y, num_outputs=3)
    cfg.define_split("tile_x", X, num_outputs=3)
    cfg.define_split("tile_k", K, num_outputs=2)
    if cfg.is_fallback:
        pass

    data = te.placeholder((N, H, W, CI), dtype=dtype)
    weight = te.placeholder((X, K), dtype=dtype)
    idxsplit = lambda x,y: reduce(lambda a,b: a[:-1]+[a[-1]%b,a[-1]//b], y, [x])

    @partial(te.compute, (Y, K), name='im2col')
    def im2col(row, col):
        jw, jh, jn = idxsplit(row, [W, H])
        jc, kw, kh = idxsplit(col, [CI, 3])
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


@autotvm.template('spconv2d_3x3_gemm')
def spconv2d_3x3_gemm(N, H, W, CI, CO, nElems, bsrR, bsrC, dtype='float32'):
    '''# My SpConv2d_3x3_gemm

        yt, xt, yo =>
            yi, k9, ci:vec =>
                @im2col = {yt, yo, yi}/y, {k9, ci}/k
            xo =>
                x1:1, ko:dyn(xr), yi:unroll, xi:vec, ki:unroll =>
                    @CC = {yt, yo, yi}/y, {xt, xo, x1}/xr, xi, ki  // ko
                yi:unroll, xi:vec, ki:unroll =>
                    @C = {yt, yo, yi}/y, {xt, xo, xi}/x  // ki
    '''
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
    idxsplit = lambda x,y: reduce(lambda a,b: a[:-1]+[a[-1]%b,a[-1]//b], y, [x])

    @partial(te.compute, (Y, K), name='Im2Col')
    def Im2Col(row, col):
        jw, jh, jn = idxsplit(row, [W, H])
        jc, kw, kh = idxsplit(col, [CI, 3])
        ih, iw = jh + kh - 1, jw + kw - 1
        return tir.if_then_else(
            tir.all(0 <= ih, ih < H, 0 <= iw, iw < W),
            Data[jn, ih, iw, jc], 0)
    
    @partial(te.compute, (Y, X // bsrR, bsrR, bsrC), name='CC')
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