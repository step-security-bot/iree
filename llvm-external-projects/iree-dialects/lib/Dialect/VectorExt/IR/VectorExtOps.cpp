// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"
#include <numeric>

using namespace mlir;
using namespace mlir::iree_compiler::IREE::VectorExt;

//===----------------------------------------------------------------------===//
// LayoutConflictResolutionOp
//===----------------------------------------------------------------------===//

LogicalResult validateLayout(Operation *op, StringRef label,
                             VectorLayoutInterface layout,
                             ArrayRef<int64_t> inputShape) {
  if (!layout.isValidLayout(inputShape)) {
    return op->emitError(
        "The " + label +
        " layout shape cannot be distributed over the given vector shape.");
  }
  return success();
}

// Validate that the desired layout has the same shape as the input.
LogicalResult LayoutConflictResolutionOp::verify() {
  Operation *op = getOperation();
  ArrayRef<int64_t> inputShape =
      cast<VectorType>(getInput().getType()).getShape();
  if (succeeded(validateLayout(op, "source", getSourceLayout(), inputShape)))
    return validateLayout(op, "desired", getDesiredLayout(), inputShape);
  return failure();
}

// to_simd -> to_simt
OpFoldResult ToSIMDOp::fold(FoldAdaptor) {
  if (auto simtOp = getOperand().getDefiningOp<ToSIMTOp>()) {
    return simtOp.getOperand();
  }
  return {};
}

// to_simt -> to_simd
OpFoldResult ToSIMTOp::fold(FoldAdaptor) {
  if (auto simdOp = getOperand().getDefiningOp<ToSIMDOp>()) {
    return simdOp.getOperand();
  }
  return {};
}

void LayoutIterator::maybeFreezeAndConcatenate(
    const LayoutIterator::State &frozenState) {
  for (auto &[frozenDim, frozenIt] : frozenState.iterators) {
    if (!state.contains(frozenDim)) {
      frozenDimensions.insert(frozenDim);
      state[frozenDim] = frozenIt;
    }
  }
}

static bool isLaneDimension(LayoutDimension dim) {
  return (dim == LayoutDimension::LANEX) || (dim == LayoutDimension::LANEY) ||
         (dim == LayoutDimension::LANEZ);
}

void LayoutIterator::initialize(const PerDimLayoutAttr &attr,
                                DenseMap<LayoutDimension, int64_t> strides,
                                std::optional<int64_t> simdIndex) {
  auto reversedLabels = llvm::reverse(attr.getLabels());
  auto reversedShapes = llvm::reverse(attr.getShapes());
  for (auto [nameAttr, shape] : llvm::zip(reversedLabels, reversedShapes)) {
    LayoutDimension dim = nameAttr.getValue();
    if (isLaneDimension(dim))
      continue;
    int64_t stride = strides.contains(dim) ? strides[dim] : 1;
    ranges[dim] = DimensionalRange(0, shape, stride);
    state.iterators[dim] = ranges[dim].begin();
    maxIterations *= shape / stride;
    if (simdIndex) {
      int64_t index = simdIndex.value();
      if (!state.simdToLayoutDim.contains(index))
        state.simdToLayoutDim[index] = {};
      state.simdToLayoutDim[index].insert(dim);
    }
  }
}

void LayoutIterator::erase(LayoutDimension dim) {
  if (state.contains(dim))
    state.erase(dim);
}

/// Returns the iterators that are associated with the provided simdIndex
/// preserving the original order.
LayoutIterator::State
LayoutIterator::State::getProjectedState(int64_t simdIndex) const {
  LayoutIterator::State projectedState = *this;
  for (auto [dim, it] : iterators) {
    if (!simdToLayoutDim.at(simdIndex).contains(dim)) {
      projectedState.erase(dim);
    }
  }
  return projectedState;
}

LayoutIterator LayoutIterator::getBatchIterator() const {
  LayoutIterator projectedIterator = *this;
  for (auto [dim, it] : state.iterators) {
    if (!isBatch(dim)) {
      projectedIterator.erase(dim);
    }
  }
  return projectedIterator;
}

LayoutIterator::LayoutIterator(LayoutAttr &attr,
                               DenseMap<LayoutDimension, int64_t> strides) {
  for (auto perDimAttr : llvm::enumerate(attr.getLayouts())) {
    initialize(perDimAttr.value(), strides, perDimAttr.index());
  }
}

LayoutIterator::LayoutIterator(LayoutAttr &attr) {
  DenseMap<LayoutDimension, int64_t> strides;
  for (auto perDimAttr : llvm::enumerate(attr.getLayouts())) {
    initialize(perDimAttr.value(), strides, perDimAttr.index());
  }
}

LayoutIterator::LayoutIterator(LayoutAttr &attr,
                               DenseMap<LayoutDimension, int64_t> strides,
                               int simtIndex) {
  for (auto perDimAttr : llvm::enumerate(attr.getLayouts())) {
    if (perDimAttr.index() != simtIndex)
      continue;
    initialize(perDimAttr.value(), strides, perDimAttr.index());
  }
}

LayoutIterator::LayoutIterator(LayoutAttr &attr, int simtIndex) {
  DenseMap<LayoutDimension, int64_t> strides;
  for (auto perDimAttr : llvm::enumerate(attr.getLayouts())) {
    if (perDimAttr.index() != simtIndex)
      continue;
    initialize(perDimAttr.value(), strides, perDimAttr.index());
  }
}

LayoutIterator::LayoutIterator(PerDimLayoutAttr &attr,
                               DenseMap<LayoutDimension, int64_t> strides) {
  initialize(attr, strides, std::nullopt);
}

LayoutIterator &LayoutIterator::operator++() {
  for (auto &[dim, it] : state.iterators) {
    if (frozenDimensions.contains(dim))
      continue;
    ++it;
    if (it == ranges[dim].end()) {
      it = ranges[dim].begin();
      continue;
    }
    break;
  }
  iterations++;
  return *this;
}

/// The iterator is done when all the loops are complete.
bool LayoutIterator::iterationComplete() { return iterations == maxIterations; }

void LayoutIterator::apply(
    std::function<void(const LayoutIterator::State &)> callback) {
  for (; !iterationComplete(); ++(*this)) {
    callback(state);
  }
}

// clang-format off
#define GET_OP_CLASSES
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.cpp.inc" // IWYU pragma: keep
// clang-format: on
