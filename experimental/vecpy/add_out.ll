; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

define <32 x float> @add(<32 x float> %0, <32 x float> %1) {
  %3 = fadd <32 x float> %0, %1
  ret <32 x float> %3
}

define <32 x float> @_mlir_ciface_add(<32 x float> %0, <32 x float> %1) {
  %3 = call <32 x float> @add(<32 x float> %0, <32 x float> %1)
  ret <32 x float> %3
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
