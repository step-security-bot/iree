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


# Simple little wrapper. More full fledged helpers here are in Nelli and other examples.
class Vec:
  def __init__(self, val):
    self.val = val

  def __str__(self):
    return "Vec({0})".format(self.val)

  def __add__(self, other):
    return Vec(arith.AddFOp(self.val, other.val))

  @staticmethod
  def vecT():
    return VectorType.get((32,), F32Type.get())

  @staticmethod
  def const(x):
    val = DenseElementsAttr.get(np.array(x, dtype=np.float32),
                                type=Vec.vecT())
    return Vec(arith.ConstantOp(result=Vec.vecT(), value=val))


# This is definitely where things aren't just push a button :) For better or
# worse. This is where one could customize, but the e
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
        module = Module.create()
        vecT = Vec.vecT()

        with InsertionPoint(module.body):

            @func.FuncOp.from_py_func(vecT, vecT)
            def add(lhs, rhs):
                res = arith.AddFOp(lhs, rhs)
                return res

            @func.FuncOp.from_py_func(vecT)
            def add2(arg):
                res = Vec(arg) + Vec.const([2]*32)
                return res.val

        # This is also boilerplate which if this were tool rather than toolbox
        # would be hidden.
        SymbolTable(module.operation)["add"].attributes["llvm.emit_c_interface"] = UnitAttr.get()
        SymbolTable(module.operation)["add2"].attributes["llvm.emit_c_interface"] = UnitAttr.get()

        # Showing flow below ... except it is wrong :) The ABI that execution
        # engine expects doesn't support vectors natively (it's a testing tool
        # for simple code really rather than full fledged but shows flow).
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
