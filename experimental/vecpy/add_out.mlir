module {
  llvm.func @add(%arg0: vector<32xf32>, %arg1: vector<32xf32>) -> vector<32xf32> attributes {llvm.emit_c_interface} {
    %0 = llvm.fadd %arg0, %arg1  : vector<32xf32>
    llvm.return %0 : vector<32xf32>
  }
  llvm.func @_mlir_ciface_add(%arg0: vector<32xf32>, %arg1: vector<32xf32>) -> vector<32xf32> attributes {llvm.emit_c_interface} {
    %0 = llvm.call @add(%arg0, %arg1) : (vector<32xf32>, vector<32xf32>) -> vector<32xf32>
    llvm.return %0 : vector<32xf32>
  }
}
