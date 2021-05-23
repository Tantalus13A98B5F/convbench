from tvm import topi, autotvm, te
import logging
import sys

A = te.placeholder((655360, 576), name='A')
B = te.placeholder((64, 576), name='B')
task = autotvm.task.create('dense_pack.x86', args=(A, B), target="llvm -mcpu=cascadelake")
print(task.config_space)

# logging config (for printing tuning log to the screen)
logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

# There are two steps for measuring a config: build and run.
# By default, we use all CPU cores to compile program. Then measure them sequentially.
# We measure 5 times and take average to reduce variance.
measure_option = autotvm.measure_option(builder="local", runner=autotvm.LocalRunner(number=4, repeat=3, timeout=20))

# Begin tuning with RandomTuner, log records to file `matmul.log`
# You can use alternatives like XGBTuner.
tuner = autotvm.tuner.RandomTuner(task)
tuner.tune(
    n_trial=500,
    measure_option=measure_option,
    callbacks=[autotvm.callback.log_to_file("dense_pack.log")],
)