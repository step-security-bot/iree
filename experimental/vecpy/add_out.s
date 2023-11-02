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
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2                               # -- Begin function add2
.LCPI2_0:
	.long	0x40000000                      # float 2
	.text
	.globl	add2
	.p2align	4, 0x90
	.type	add2,@function
add2:                                   # @add2
	.cfi_startproc
# %bb.0:
	vbroadcastss	.LCPI2_0, %zmm2         # zmm2 = [2.0E+0,2.0E+0,2.0E+0,2.0E+0,2.0E+0,2.0E+0,2.0E+0,2.0E+0,2.0E+0,2.0E+0,2.0E+0,2.0E+0,2.0E+0,2.0E+0,2.0E+0,2.0E+0]
	vaddps	%zmm2, %zmm0, %zmm0
	vaddps	%zmm2, %zmm1, %zmm1
	retl
.Lfunc_end2:
	.size	add2, .Lfunc_end2-add2
	.cfi_endproc
                                        # -- End function
	.globl	_mlir_ciface_add2               # -- Begin function _mlir_ciface_add2
	.p2align	4, 0x90
	.type	_mlir_ciface_add2,@function
_mlir_ciface_add2:                      # @_mlir_ciface_add2
	.cfi_startproc
# %bb.0:
	subl	$12, %esp
	.cfi_def_cfa_offset 16
	calll	add2@PLT
	addl	$12, %esp
	.cfi_def_cfa_offset 4
	retl
.Lfunc_end3:
	.size	_mlir_ciface_add2, .Lfunc_end3-_mlir_ciface_add2
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
