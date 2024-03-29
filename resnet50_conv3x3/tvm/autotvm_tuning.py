from contextlib import contextmanager
import logging
import os
from tvm import autotvm, te
import numpy as np
from sparse_utils import *


tgtstr = 'llvm -mcpu=cascadelake'


from collections import namedtuple
ResNet56Conv = namedtuple('ResNet56Conv',
        'nfilters channels image nnz')
with open('fmt.txt') as f:
    caselist = [
            ResNet56Conv(*[int(i) for i in l.strip().split(',')])
            for l in f if l.strip()]


class Conv2dCase:
    def iter_shape(self):
        # resnet56
        N, H, W = 10, 32, 32
        for C in [16, 32, 64]:
            yield (N, C, H * 16 // C, W * 16 // C)
        return
        # resnet50
        N, H, W = 10, 512, 512
        for C in [64, 128, 256, 512]:
            yield (N, C, H * 32 // C, W * 32 // C)
        return

    def prepare_data(self, *args):
        return None

    def configure_task(self, N, C, H, W, data):
        Csplit = 16
        Cgroups = C // Csplit
        A = te.placeholder((N, Cgroups, H, W, Csplit), name='A')
        B = te.placeholder((Cgroups, Cgroups, 3, 3, Csplit, Csplit), name='B')
        import tvm.topi.x86.conv2d
        return autotvm.task.create(
            'conv2d_NCHWc.x86', target=tgtstr,
            args=(A, B, 1, 1, 1, 'NCHWc', 'NCHWc', 'float32'),
        )

    @contextmanager
    def configure_data(self, runner, data):
        yield


class MMConv2dCase(Conv2dCase):
    def configure_task(self, N, C, H, W, data):
        import conv2d_gemm
        return autotvm.task.create(
            'conv2d_3x3_gemm', target=tgtstr,
            args=(N, H, W, C, C, 'float32'),
        )


class SpConv2dCase(Conv2dCase):
    def iter_shape(self):
        #yield from (
        #    (*shape, sprate, veclen)
        #    for shape in super().iter_shape()
        #    for sprate in [0.5, 0.6, 0.7, 0.8, 0.9]
        #    for veclen in [4, 8, 16]
        #)
        for case in caselist:
            yield (10, case.channels, case.image, case.image,
                   case.nnz, case.nfilters)
    
    def prepare_data(self, N, C, H, W, nnz, veclen):
        nhwc_data = np.random.randint(0, 256, (N, H, W, C)).astype('float32')
        #weight_ohwi = np.random.rand(C, 3*3*C).astype('float32')
        #spweight_ohwi = make_bsr_sparse(weight_ohwi, sprate, (veclen, 1))
        spweight_ohwi = random_bsr_sparse((C, 9*C), (veclen, 1), nnz)
        ret = np.zeros((N*H*W, C), dtype='float32')
        return [nhwc_data, *unpack_bsr(spweight_ohwi), ret]

    def configure_task(self, N, C, H, W, NNZ, veclen, data):
        #NNZ = data[1].shape[0]
        import conv2d_gemm
        return autotvm.task.create(
            'spconv2d_3x3_gemm', target=tgtstr,
            args=(N, H, W, C, C, NNZ, veclen, 1, 'float32'),
        )
    
    @contextmanager
    def configure_data(self, runner, data):
        # runner.ref_input = data
        # yield
        with NonRandomFill.mock(data):
            yield


def tune(case, n_trial, result_file, best_file=None):
    logger = logging.getLogger("autotvm")
    logger.setLevel(logging.DEBUG)
    for shape in case.iter_shape():
        data = case.prepare_data(*shape)
        task = case.configure_task(*shape, data)
        logging.info('task: %s, %s, %s', task.name, task.args, len(task.config_space))
        runner = autotvm.LocalRunner(number=4, repeat=3, timeout=20)
        mopt = autotvm.measure_option(builder='local', runner=runner)
        with case.configure_data(runner, data):
            tuner = autotvm.tuner.GATuner(task)
            tuner.tune(n_trial=n_trial, measure_option=mopt, callbacks=[
                autotvm.callback.log_to_file(result_file),
                autotvm.callback.progress_bar(n_trial)
            ])
    if best_file:
        autotvm.record.pick_best(result_file, best_file)


if __name__ == '__main__':
    # logging
    logging.basicConfig(level=logging.INFO)
    logging.captureWarnings(True)
    # tuning
    os.environ['OMP_NUM_THREADS'] = '1'
    tune(Conv2dCase(), 200, 'spconv56.log', 'spconv56.best.log')
    tune(SpConv2dCase(), 200, 'spconv56.log', 'spconv56.best.log')
