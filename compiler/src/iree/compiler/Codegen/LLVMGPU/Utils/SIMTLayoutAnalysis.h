// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_SIMTLAYOUTANALYSIS_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_SIMTLAYOUTANALYSIS_H_

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"

namespace mlir {
namespace iree_compiler {

/// Forward decleration for analysis.
class PropagateLayout;
class EnforceLayout;

using HighDimLayout = IREE::VectorExt::LayoutAttr;
SmallVector<int64_t> getSIMTVectorShape(IREE::VectorExt::LayoutAttr layout);

class DistributionLayout : public AnalysisState {
public:
  explicit DistributionLayout(Value val) : AnalysisState(val) {}

  Value getValue() const {
    ProgramPoint point = getPoint();
    assert(isa<Value>(point) && "expected program point to be a value");
    return cast<Value>(point);
  }

  /// TODO: This currently, creates a new value but doesn't replace it with the
  /// current value, because that would be wrong. Find a way to take care of
  /// that better.
  ChangeResult resolveWithPossibleConflict(const DistributionLayout *rhs,
                                           OpOperand &operand);
  ChangeResult resolveWithPossibleConflict(const HighDimLayout &rhs,
                                           OpOperand &operand);

  ChangeResult resolve(const DistributionLayout *rhs);
  ChangeResult resolve(const HighDimLayout &rhs);

  HighDimLayout getInnerLayout() const { return vectorLayout; }

  bool isUninitialized() const { return !vectorLayout; }
  bool hasLayout() const { return !isUninitialized(); }

  /// Compare two states.
  bool operator==(const DistributionLayout &rhs) const {
    return vectorLayout == rhs.vectorLayout;
  }
  bool operator!=(const DistributionLayout &rhs) const {
    return !(*this == rhs);
  }

  void print(raw_ostream &os) const override;

  /// Subscribe an analysis to updates of the lattice. When the lattice
  /// changes, subscribed analyses are re-invoked. This is more efficient than
  /// relying on the dependency map.
  void subscribePropagation(PropagateLayout *analysis) {
    propagation = analysis;
  }
  void subscribeEnforcement(EnforceLayout *analysis) { enforcement = analysis; }

private:
  /// When the lattice gets updated, propagate an update to users of the value
  /// using its use-def chain to subscribed analyses.
  void onUpdate(DataFlowSolver *solver) const override;

  /// The result of a resolution.
  /// Change: The layout was changed.
  /// Conflict: The layout was not changed because there was a conflict.
  /// NoChange: The layout was not changed because it was already the same.
  enum ResolutionResult {
    Change,
    Conflict,
    NoChange,
  };

  /// Attempt to resolve the current lattice with the given lattice. Returns if
  /// the current layout was not changed, changed or if there was a layout
  /// conflict.
  ResolutionResult doResolution(const HighDimLayout &rhs);

  /// Set the layout for this lattice element to the given layout. This function
  /// should only be used when you know there will be no layout conflicts.
  /// Otherwise, the resolve-like functions should be used.
  void setInnerLayout(const HighDimLayout &layout) { vectorLayout = layout; }

  /// The layout of the vector SSA Value.
  HighDimLayout vectorLayout;

  /// Each lattice element stores a pointer to the analysis that work on it so
  /// it can notify them when it changes.
  PropagateLayout *propagation = nullptr;
  EnforceLayout *enforcement = nullptr;
};

class PropagateLayout : public DataFlowAnalysis {
public:
  explicit PropagateLayout(DataFlowSolver &solver, MLIRContext *ctx)
      : DataFlowAnalysis(solver), ctx(ctx) {}

  LogicalResult initialize(Operation *op) override;

  LogicalResult visit(ProgramPoint point) override;

  /// Register a new value to be part of the dataflow analysis. The value should
  /// not be part of the analysis already. This is used for new values that are
  /// created.
  void registerNewValue(Value val, const HighDimLayout &layout);

  friend class DistributionLayout;

private:
  void visitOperation(Operation *op);

  void visitRegionSuccessors(RegionBranchOpInterface branch,
                             RegionBranchPoint branchPoint,
                             OperandRange operands);

  DistributionLayout *getLatticeElement(Value val);

  MLIRContext *ctx;
};

class EnforceLayout : public DataFlowAnalysis {
public:
  explicit EnforceLayout(DataFlowSolver &solver, MLIRContext *ctx)
      : DataFlowAnalysis(solver), ctx(ctx) {}

  LogicalResult initialize(Operation *op) override;

  LogicalResult visit(ProgramPoint point) override;

  void registerNewValue(Value val, const HighDimLayout &layout);

  friend class DistributionLayout;

private:
  void visitOperation(Operation *op);

  void visitRegionSuccessors(RegionBranchOpInterface branch,
                             RegionBranchPoint branchPoint,
                             MutableArrayRef<OpOperand> operands);

  DistributionLayout *getLatticeElement(Value val);

  MLIRContext *ctx;
};

}; // namespace iree_compiler
}; // namespace mlir

#endif // IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_SIMTLAYOUTANALYSIS_H_
