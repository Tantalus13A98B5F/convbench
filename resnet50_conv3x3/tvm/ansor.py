import tvm
from tvm import te, auto_scheduler


@auto_scheduler.register_workload
def sparse_dense(M, N, K, bsrR, bsrC, nElems, dtype='float32'):
    Data = te.placeholder((M, K), dtype=dtype, name='Data')
    Wdat = te.placeholder((nElems, bsrR, bsrC), dtype=dtype, name='Wdat')
    Wind = te.placeholder((nElems,), dtype='int', name='Wind')
    Wptr = te.placeholder((N // bsrR + 1,), dtype='int', name='Wptr')
    
    def bsr_gemm_kernel(drow, wrow, brow, bcol):
        row_start, row_end = Wptr[wrow], Wptr[wrow+1]
        elem_idx = te.reduce_axis((0, row_end - row_start), name='elem_idx')
        elem = row_start + elem_idx
        return te.sum(Data[drow, Wind[elem]*bsrC + bcol] * Wdat[elem, brow, bcol], axis=elem_idx)

    CC = te.compute((M, N // bsrR, bsrR, bsrC), bsr_gemm_kernel, name='CC')
    k = te.reduce_axis((0, bsrC), name='k')
    C = te.compute((M, N), lambda m, n: te.sum(CC[m, n // bsrR, n % bsrR, k], axis=k), name='C')
    return [Data, Wdat, Wind, Wptr, C]


target = tvm.target.Target("llvm -mcpu=cascadelake")
task = auto_scheduler.SearchTask(func=sparse_dense, args=(N, L, M, "float32"), target=target)

# Inspect the computational graph
print("Computational DAG:")
print(task.compute_dag)

log_file = "spdense.json"
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=10,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

# Run auto-tuning (search)
task.tune(tune_option)
# Apply the best schedule
sch, args = task.apply_best(log_file)

print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))