import gc, sys, os, tempfile
from mlir.ir import *
from mlir.passmanager import *
from mlir.execution_engine import *
from mlir.runtime import *
import mlir.dialects.arith as arith
import mlir.dialects.builtin as builtin
import mlir.dialects.func as func
import mlir.dialects.vector as vector


def log(*args):
    print(*args, file=sys.stderr)
    sys.stderr.flush()


def lowerToLLVM(module):
    pm = PassManager.parse(
        "builtin.module(convert-vector-to-scf,convert-scf-to-cf,convert-cf-to-llvm,convert-vector-to-llvm,convert-index-to-llvm{index-bitwidth=32},finalize-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)"
    )
    pm.run(module.operation)
    print(module)
    return module


# Construct vector module programmatically. The vector operations below are from
#   https://mlir.llvm.org/docs/Dialects/Vector/
# there are value builders which make these much simpler, and other abstractions
# one can add around these to give a _much_ better user experience.
def testInvokeVectorAdd():
    with Context(), Location.unknown():
        module = Module.parse(
            r"""
func.func @add(%arg0: vector<32xf32>, %arg1: vector<32xf32>) -> vector<32xf32> attributes { llvm.emit_c_interface } {
  %add = arith.addf %arg0, %arg1 : vector<32xf32>
  return %add : vector<32xf32>
}
    """)
    # module = Module.create()
    #    vecT = VectorType.get((32,), F32Type.get())
    #    with InsertionPoint(module.body):

    #        @func.FuncOp.from_py_func(vecT, vecT)
    #        def add(lhs, rhs):
    #            res = arith.AddFOp(lhs, rhs)
    #            return res # vector.PrintOp(source=res)

        execution_engine = ExecutionEngine(lowerToLLVM(module))
        # Prepare arguments: two input floats and one result.
        # Arguments must be passed as pointers.
        c_float_p = ctypes.c_float * 32
        py_values = [42] * 32
        arg0 = c_float_p(*py_values)
        arg1 = c_float_p(*py_values)
        res = c_float_p(*py_values)
        execution_engine.invoke("add", arg0, arg1, res)
        # CHECK: 42.0 + 2.0 = 44.0
        log("{0} + {1} = {2}".format(arg0[0], arg1[0], res[0]))


testInvokeVectorAdd()
