module {
  llvm.func @add(%arg0: vector<32xf32>, %arg1: vector<32xf32>) -> vector<32xf32> attributes {llvm.emit_c_interface} {
    %0 = llvm.fadd %arg0, %arg1  : vector<32xf32>
    llvm.return %0 : vector<32xf32>
  }
  llvm.func @_mlir_ciface_add(%arg0: vector<32xf32>, %arg1: vector<32xf32>) -> vector<32xf32> attributes {llvm.emit_c_interface} {
    %0 = llvm.call @add(%arg0, %arg1) : (vector<32xf32>, vector<32xf32>) -> vector<32xf32>
    llvm.return %0 : vector<32xf32>
  }
  llvm.func @add2(%arg0: vector<32xf32>) -> vector<32xf32> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(dense<2.000000e+00> : vector<32xf32>) : vector<32xf32>
    %1 = llvm.fadd %arg0, %0  : vector<32xf32>
    llvm.return %1 : vector<32xf32>
  }
  llvm.func @_mlir_ciface_add2(%arg0: vector<32xf32>) -> vector<32xf32> attributes {llvm.emit_c_interface} {
    %0 = llvm.call @add2(%arg0) : (vector<32xf32>) -> vector<32xf32>
    llvm.return %0 : vector<32xf32>
  }
}
