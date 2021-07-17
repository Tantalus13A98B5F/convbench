import numpy as np
from scipy import sparse
from tvm import topi, autotvm, target
import tvm
from conv2d_gemm import spconv2d_3x3_gemm

tgtstr = 'llvm -mcpu=cascadelake'

chann_ = None
for nf in [16, 8, 4]:
    npz = np.load(f'resnet56dat/f{nf}/conv.npz')
    for name, arr in npz.items():
        sparr = sparse.bsr_matrix(arr, blocksize=(nf, 1))
        nnz = sparr.data.shape[0]
        chann = arr.shape[0]
        image = 64 * 8 // chann
        shape = 10, image, image, chann
        with autotvm.apply_history_best('spconv56.best.log'), \
                target.Target(tgtstr):
            s, params = spconv2d_3x3_gemm(*shape, chann, nnz, nf, 1, 'float32')
            func = tvm.build(s, params, target=tgtstr, name='spconv2d')
            dev = tvm.device(tgtstr, 0)
            args = [
                tvm.nd.array(np.random.rand(*shape).astype('float32'), dev),
                tvm.nd.array(sparr.data, dev),
                tvm.nd.array(sparr.indices, dev),
                tvm.nd.array(sparr.indptr, dev),
                tvm.nd.empty((10*image*image, chann), device=dev)
            ]
            evt = func.time_evaluator(func.entry_name, dev, number=1, repeat=100)
            if chann != chann_:
                print(nf, chann, nnz, sep='\t')
            print(min(evt(*args).results))
            chann_ = chann

