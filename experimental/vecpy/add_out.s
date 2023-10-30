	.text
	.file	"LLVMDialectModule"
	.globl	add                             # -- Begin function add
	.p2align	4, 0x90
	.type	add,@function
add:                                    # @add
	.cfi_startproc
# %bb.0:
	pushl	%ebp
	.cfi_def_cfa_offset 8
	.cfi_offset %ebp, -8
	movl	%esp, %ebp
	.cfi_def_cfa_register %ebp
	andl	$-64, %esp
	subl	$64, %esp
	vaddps	%zmm2, %zmm0, %zmm0
	vaddps	8(%ebp), %zmm1, %zmm1
	movl	%ebp, %esp
	popl	%ebp
	.cfi_def_cfa %esp, 4
	retl
.Lfunc_end0:
	.size	add, .Lfunc_end0-add
	.cfi_endproc
                                        # -- End function
	.globl	_mlir_ciface_add                # -- Begin function _mlir_ciface_add
	.p2align	4, 0x90
	.type	_mlir_ciface_add,@function
_mlir_ciface_add:                       # @_mlir_ciface_add
	.cfi_startproc
# %bb.0:
	pushl	%ebp
	.cfi_def_cfa_offset 8
	.cfi_offset %ebp, -8
	movl	%esp, %ebp
	.cfi_def_cfa_register %ebp
	andl	$-64, %esp
	subl	$128, %esp
	vmovaps	8(%ebp), %zmm3
	vmovaps	%zmm3, (%esp)
	calll	add@PLT
	movl	%ebp, %esp
	popl	%ebp
	.cfi_def_cfa %esp, 4
	retl
.Lfunc_end1:
	.size	_mlir_ciface_add, .Lfunc_end1-_mlir_ciface_add
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
