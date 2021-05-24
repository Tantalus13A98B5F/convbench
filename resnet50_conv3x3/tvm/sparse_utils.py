import scipy.sparse
import numpy as np
import tvm
from tvm.rpc import RPCSession
from contextlib import contextmanager


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


def hook_method(obj, attr):
    def real_decorator(func):
        orig = getattr(obj, attr)
        func.orig = orig
        func.perform = lambda: setattr(obj, attr, func)
        func.revert = lambda: setattr(obj, attr, orig)
        func.perform()
        return func
    return real_decorator


class NonRandomFill:
    srclst_ = []
    
    @classmethod
    def set_srclst(cls, srclst):
        cls.srclst_ = [tvm.nd.array(it) for it in srclst]

    def __init__(self):
        self.srclst = iter(self.srclst_)
    
    def __call__(self, tgt):
        src = next(self.srclst)
        tgt.copyfrom(src)

    @classmethod
    @contextmanager
    def mock(cls, srclst):
        cls.set_srclst(srclst)
        yield
        cls.set_srclst([])


@hook_method(RPCSession, 'get_function')
def new_get_function(self, funcname):
    random_fill = 'tvm.contrib.random.random_fill'
    if funcname == random_fill and NonRandomFill.srclst_:
        return NonRandomFill()
    else:
        return new_get_function.orig(self, funcname)