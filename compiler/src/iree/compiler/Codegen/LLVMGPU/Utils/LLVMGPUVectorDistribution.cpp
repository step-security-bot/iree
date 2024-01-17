// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/VectorLayoutProvider.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"

#define DEBUG_TYPE "iree-codegen-llvmgpu-vector-distribution"

using namespace mlir::iree_compiler::IREE::VectorExt;

namespace mlir::iree_compiler {

static bool canDistribute(Operation *op, VectorLayoutAnalysis &analysis) {
  bool needsDistribution = false;
  // Check if this operation has any operands with a vector type. If so,
  // then they need to have a layout.
  for (Value operand : op->getOperands()) {
    if (isa<VectorType>(operand.getType())) {
      needsDistribution = true;
      if (!analysis.getLayout<Attribute>(operand)) {
        return false;
      }
    }
  }

  // Check if this operation has any results with a vector type. If so,
  // then they need to have a layout.
  for (OpResult result : op->getResults()) {
    if (isa<VectorType>(result.getType())) {
      needsDistribution = true;
      if (!analysis.getLayout<Attribute>(result)) {
        return false;
      }
    }
  }

  return needsDistribution;
}

TypedValue<VectorType> getDistributed(RewriterBase &rewriter,
                                      TypedValue<VectorType> value,
                                      LayoutProvider *provider) {
  // If this is a result of a "to_simd" op, use the source value of it.
  if (auto toSIMD = value.getDefiningOp<IREE::VectorExt::ToSIMDOp>()) {
    value = cast<TypedValue<VectorType>>(toSIMD.getInput());
    return value;
  }
  // Create a "to_simt" op to convert the value to the distributed layout.
  SmallVector<int64_t> distributedShape = provider->getDistributedShape(value);
  VectorType distributedType =
      VectorType::get(distributedShape, value.getType().getElementType());
  auto toSIMT = rewriter.create<IREE::VectorExt::ToSIMTOp>(
      value.getLoc(), distributedType, value);
  return toSIMT.getResult();
}

void replaceOpWithDistributedValues(RewriterBase &rewriter, Operation *op,
                                    LayoutProvider *provider,
                                    ValueRange values) {
  // Replace all OpResults with the given values.
  SmallVector<Value> replacements;
  for (OpResult opResult : op->getOpResults()) {
    Value replacement = values[opResult.getResultNumber()];
    // If this value is a vector type, it must be converted back to simd.
    if (isa<VectorType>(replacement.getType())) {
      auto oldResult = cast<TypedValue<VectorType>>(opResult);
      // Create a toSIMD op to convert the value back to the simd.
      rewriter.setInsertionPointAfterValue(oldResult);
      auto toSIMD = rewriter.create<IREE::VectorExt::ToSIMDOp>(
          oldResult.getLoc(), oldResult.getType(), replacement);
      // Clone the layout to the new value.
      provider->getAnalysis().cloneLayoutInformationToNewValue(
          oldResult, toSIMD.getResult());
      // Add to replacements.
      replacement = toSIMD.getResult();
    }
    replacements.push_back(replacement);
  }

  rewriter.replaceOp(op, replacements);
}

namespace {

class DistributionRewriter : public IRRewriter, public RewriterBase::Listener {
public:
  DistributionRewriter(MLIRContext *ctx, DenseSet<Operation *> &erasedOps,
                       SmallVector<Operation *> &worklist,
                       VectorLayoutAnalysis &analysis)
      : IRRewriter(ctx), erasedOps(erasedOps), worklist(worklist),
        analysis(analysis) {
    setListener(this);
  }

protected:
  void notifyOperationRemoved(Operation *op) override { erasedOps.insert(op); }

  void notifyOperationInserted(Operation *op) override {
    // For now, do nothing.
    // TODO: Check if the operation can still be distributed and try to
    // distribute it.
  }

private:
  // A reference to the set of operations that have been erased.
  DenseSet<Operation *> &erasedOps;
  // A reference to the worklist of operations that need to be distributed.
  SmallVector<Operation *> &worklist;
  // A reference to the analysis that provides the layout information.
  VectorLayoutAnalysis &analysis;
};

class VectorDistribution {
public:
  VectorDistribution(func::FuncOp root, LayoutProvider *provider,
                     VectorLayoutAnalysis &analysis)
      : root(root), analysis(analysis), provider(provider) {
    provider->setAnchorOps();
    if (failed(analysis.run()))
      return;
    LLVM_DEBUG(llvm::dbgs() << "Layout Analysis Completed Successfully :\n");
    LLVM_DEBUG(analysis.print(llvm::dbgs()));
  }

  void distribute() {
    SmallVector<Operation *> worklist;
    DenseSet<Operation *> erasedOps;
    DistributionRewriter rewriter(root.getContext(), erasedOps, worklist,
                                  analysis);

    // 1: Collect all operations that need to be distributed.
    LLVM_DEBUG(llvm::dbgs() << "Step 1: Collecting operations to distribute\n");
    root->walk([&](Operation *op) {
      if (canDistribute(op, analysis)) {
        worklist.push_back(op);
      }
    });

    // 2. Set the insertion point to the beginning of the root, and build the
    // thread grid.
    LLVM_DEBUG(llvm::dbgs() << "Step 2: Getting thread grid\n");
    rewriter.setInsertionPointToStart(&root.getBody().getBlocks().front());
    threadGrid = provider->getThreadGrid(rewriter);

    // 3. Distribute all operations in the worklist until we reach a fixed
    // point.
    LLVM_DEBUG(llvm::dbgs()
               << "Step 3: Starting distributing collected operations\n");
    bool changed = true;
    while (changed) {
      changed = false;
      for (unsigned i = 0; i < worklist.size(); ++i) {
        Operation *op = worklist[i];
        if (erasedOps.count(op))
          continue;

        rewriter.setInsertionPoint(op);

        LLVM_DEBUG(llvm::dbgs() << "Trying to distribute: ");
        LLVM_DEBUG(op->print(llvm::dbgs(), OpPrintingFlags().skipRegions()));
        LLVM_DEBUG(llvm::dbgs() << "\n");

        if (provider->specializedDistribution(rewriter, op).succeeded()) {
          LLVM_DEBUG(llvm::dbgs() << "Specialized Distribution Successful\n");
          changed = true;
          continue;
        }

        LogicalResult distributed =
            TypeSwitch<Operation *, LogicalResult>(op)
                .Case<vector::TransferReadOp>([&](auto transferReadOp) {
                  return distributeTransferReads(rewriter, transferReadOp);
                })
                .Case<vector::TransferWriteOp>([&](auto transferWriteOp) {
                  return distributeTransferWrites(rewriter, transferWriteOp);
                })
                .Case<arith::ConstantOp>([&](auto constantOp) {
                  return distributeConstants(rewriter, constantOp);
                })
                .Case<IREE::VectorExt::LayoutConflictResolutionOp>(
                    [&](auto resolutionOp) {
                      return distributeResolutions(rewriter, resolutionOp);
                    })
                .Case<scf::ForOp>([&](auto forOp) {
                  return distributeScfFor(rewriter, forOp);
                })
                .Case<scf::YieldOp>([&](auto yieldOp) {
                  return distributeScfYield(rewriter, yieldOp);
                })
                .Case<vector::MultiDimReductionOp>([&](auto reductionOp) {
                  return distributeReductions(rewriter, reductionOp);
                })
                .Case<vector::BroadcastOp>([&](auto broadcastOp) {
                  return distributeBroadcasts(rewriter, broadcastOp);
                })
                .Default([&](Operation *op) { return failure(); });

        // If the operation was distributed, continue with the next one.
        if (distributed.succeeded()) {
          LLVM_DEBUG(llvm::dbgs() << "Distribution Successful\n");
          changed = true;
          continue;
        }

        if (OpTrait::hasElementwiseMappableTraits(op)) {
          if (distributeElementwise(rewriter, op).succeeded()) {
            LLVM_DEBUG(llvm::dbgs() << "Distribution Successful\n");
            changed = true;
          }
          continue;
        }

        LLVM_DEBUG(llvm::dbgs() << "Distribution Failed\n");
      }
    }

    // 4. Ideally, we should error out here if everything was not distributed.
    // Currently, I'm not adding it for debugging purposes.
    // TODO: Add a check here if something was not distributed.
    LLVM_DEBUG(llvm::dbgs() << "Distribution Finished\n");
  }

private:
  LogicalResult
  distributeTransferWrites(RewriterBase &rewriter,
                           vector::TransferWriteOp transferWriteOp);

  LogicalResult distributeTransferReads(RewriterBase &rewriter,
                                        vector::TransferReadOp transferReadOp);

  LogicalResult distributeConstants(RewriterBase &rewriter,
                                    arith::ConstantOp constantOp);

  LogicalResult distributeElementwise(RewriterBase &rewriter, Operation *op);

  LogicalResult distributeResolutions(
      RewriterBase &rewriter,
      IREE::VectorExt::LayoutConflictResolutionOp resolutionOp);

  LogicalResult distributeTripToSharedMem(
      RewriterBase &rewriter,
      IREE::VectorExt::LayoutConflictResolutionOp resolutionOp);

  LogicalResult distributeScfFor(RewriterBase &rewriter, scf::ForOp forOp);

  LogicalResult distributeScfYield(RewriterBase &rewriter,
                                   scf::YieldOp yieldOp);

  LogicalResult distributeReductions(RewriterBase &rewriter,
                                     vector::MultiDimReductionOp reductionOp);

  LogicalResult distributeBroadcasts(RewriterBase &rewriter,
                                     vector::BroadcastOp broadcastOp);

  TypedValue<VectorType> reshapeVector(RewriterBase &rewriter,
                                       TypedValue<VectorType> src,
                                       LayoutAttr &currentLayout,
                                       LayoutAttr &targetLayout,
                                       Type elementType);

  SmallVector<Value> getIndices(RewriterBase &rewriter, LayoutAttr &layout,
                                const LayoutIterator::State &state,
                                SmallVector<Value> indices,
                                AffineMap permutationMap, Location loc);

  ArrayRef<Value> getThreadIds() { return threadGrid; }

  func::FuncOp root;
  VectorLayoutAnalysis &analysis;
  LayoutProvider *provider;
  SmallVector<Value> threadGrid;
}; // namespace

} // namespace

SmallVector<Value>
VectorDistribution::getIndices(RewriterBase &rewriter, LayoutAttr &layout,
                               const LayoutIterator::State &state,
                               SmallVector<Value> indices,
                               AffineMap permutationMap, Location loc) {
  ArrayRef<Value> threadIds = getThreadIds();
  SmallVector<Value> laneIds;
  for (PerDimLayoutAttr dimLayout : layout.getLayouts()) {
    for (LayoutDimensionAttr label : dimLayout.getLabels()) {
      switch (label.getValue()) {
      case LayoutDimension::LANEX:
        laneIds.push_back(threadIds[0]);
        break;
      case LayoutDimension::LANEY:
        laneIds.push_back(threadIds[1]);
        break;
      case LayoutDimension::LANEZ:
        laneIds.push_back(threadIds[2]);
        break;
      default:
        break;
      }
    }
  }

  // Initialize the SIMD indices with the original indices to capture the
  // unit dims.
  SmallVector<Value> simdIndices = indices;

  unsigned unitLeadingDims =
      permutationMap.getNumDims() - permutationMap.getNumResults();
  for (auto [i, dim] : llvm::enumerate(permutationMap.getResults())) {
    auto dimExpr = dyn_cast<AffineDimExpr>(dim);
    if (!dimExpr) {
      llvm::report_fatal_error("permutation map is not a projected "
                               "permutation.");
    }
    unsigned pos = dimExpr.getPosition();
    unsigned reducedPos = pos - unitLeadingDims;
    unsigned projectedI = i + unitLeadingDims;

    PerDimLayoutAttr perDimLayout = layout.getDimLayout(reducedPos);
    LayoutIterator::State projectedState = state.getProjectedState(reducedPos);
    Value index = rewriter.create<affine::AffineApplyOp>(
        loc, computeSIMDIndex(projectedState, perDimLayout),
        laneIds[reducedPos]);
    index = rewriter.create<arith::AddIOp>(loc, index, simdIndices[projectedI]);
    simdIndices[projectedI] = index;
  }

  return simdIndices;
}

/// Given a projected permutation, get a reduced permutation, i.e. without
/// the projected dimensions.
static SmallVector<int64_t> getReducedPermutation(AffineMap permutationMap) {
  assert(permutationMap.isProjectedPermutation() &&
         "permutation map should be a projected permutation.");

  SmallVector<int64_t> permutation;
  permutation.reserve(permutationMap.getNumResults());

  unsigned leadingUnitDims =
      permutationMap.getNumDims() - permutationMap.getNumResults();
  for (AffineExpr dim : permutationMap.getResults()) {
    // Get this dim's position in the permutation map.
    auto dimExpr = dyn_cast<AffineDimExpr>(dim);
    if (!dimExpr) {
      llvm::report_fatal_error("permutation map is not a projected "
                               "permutation.");
    }

    unsigned pos = dimExpr.getPosition();
    assert(pos >= leadingUnitDims && "invalid permutation map");
    pos -= leadingUnitDims;
    permutation.push_back(pos);
  }
  return permutation;
}

LogicalResult VectorDistribution::distributeTransferReads(
    RewriterBase &rewriter, vector::TransferReadOp transferReadOp) {
  TypedValue<VectorType> result = transferReadOp.getResult();
  Value source = transferReadOp.getSource();
  Type elementType = llvm::cast<ShapedType>(source.getType()).getElementType();
  auto vectorType =
      VectorType::get(provider->getDistributedShape(result), elementType);
  Location loc = transferReadOp.getLoc();
  Value vector = rewriter.create<arith::ConstantOp>(
      loc, vectorType, rewriter.getZeroAttr(vectorType));
  LayoutAttr layout = analysis.getLayout<LayoutAttr>(result);
  int64_t loadElements = layout.getTransferElements(
      getReducedPermutation(transferReadOp.getPermutationMap()));
  auto loadFn = [&](const LayoutIterator::State &state) {
    SmallVector<Value> simdIndices =
        getIndices(rewriter, layout, state, transferReadOp.getIndices(),
                   transferReadOp.getPermutationMap(), loc);

    SmallVector<int64_t> simtIndices =
        state.computeSIMTIndex(provider->getSIMTLabels(layout));

    auto vectorType = VectorType::get({loadElements}, elementType);
    if (loadElements == 1) {
      Value element = rewriter.create<memref::LoadOp>(loc, source, simdIndices);
      Value broadcasted =
          rewriter.create<vector::BroadcastOp>(loc, vectorType, element);
      vector = rewriter.create<vector::InsertStridedSliceOp>(
          loc, broadcasted, vector, simtIndices, SmallVector<int64_t>{1});
    } else {
      Value element =
          rewriter.create<vector::LoadOp>(loc, vectorType, source, simdIndices);
      vector = rewriter.create<vector::InsertStridedSliceOp>(
          loc, element, vector, simtIndices, SmallVector<int64_t>{1});
    }
  };
  DenseMap<LayoutDimension, int64_t> steps;
  steps[LayoutDimension::VECTORX] = loadElements;
  LayoutIterator iterator(layout, steps);
  iterator.apply(loadFn);
  replaceOpWithDistributedValues(rewriter, transferReadOp, provider, vector);
  return success();
}

LogicalResult VectorDistribution::distributeTransferWrites(
    RewriterBase &rewriter, vector::TransferWriteOp transferWriteOp) {
  TypedValue<VectorType> vector = transferWriteOp.getVector();
  Value source = transferWriteOp.getSource();
  LayoutAttr layout = analysis.getLayout<LayoutAttr>(vector);
  Location loc = transferWriteOp.getLoc();
  int64_t storeElements = layout.getTransferElements(
      getReducedPermutation(transferWriteOp.getPermutationMap()));
  auto storeFn = [&](const LayoutIterator::State &state) {
    SmallVector<Value> simdIndices =
        getIndices(rewriter, layout, state, transferWriteOp.getIndices(),
                   transferWriteOp.getPermutationMap(), loc);

    SmallVector<int64_t> simtIndices =
        state.computeSIMTIndex(provider->getSIMTLabels(layout));

    if (storeElements == 1) {
      Value result = rewriter.create<vector::ExtractOp>(
          loc, getDistributed(rewriter, vector, provider), simtIndices);
      rewriter.create<memref::StoreOp>(loc, result, source, simdIndices);
    } else {
      SmallVector<int64_t> strides(simtIndices.size(), 1);
      SmallVector<int64_t> shapes(simtIndices.size(), 1);
      shapes[shapes.size() - 1] = storeElements;
      Value result = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, getDistributed(rewriter, vector, provider), simtIndices, shapes,
          strides);
      result = rewriter.create<vector::ExtractOp>(
          loc, result, SmallVector<int64_t>(simtIndices.size() - 1, 0));
      rewriter.create<vector::StoreOp>(loc, result, source, simdIndices);
    }
  };
  DenseMap<LayoutDimension, int64_t> steps;
  // TODO: This is wrong. We should be using the innermost dimension for
  // calculating this step instead.
  steps[LayoutDimension::VECTORX] = storeElements;
  LayoutIterator iterator(layout, steps);
  iterator.apply(storeFn);
  replaceOpWithDistributedValues(rewriter, transferWriteOp, provider,
                                 ValueRange());
  return success();
}

LogicalResult
VectorDistribution::distributeConstants(RewriterBase &rewriter,
                                        arith::ConstantOp constantOp) {
  Value constantResult = constantOp.getResult();
  if (!isa<VectorType>(constantResult.getType()))
    return failure();
  auto constant = cast<TypedValue<VectorType>>(constantResult);
  auto attr = llvm::cast<DenseElementsAttr>(constantOp.getValue());
  // Only handle splat values for now
  if (!attr.isSplat())
    return failure();
  Type elementType = constant.getType().getElementType();
  auto vectorType =
      VectorType::get(provider->getDistributedShape(constant), elementType);
  replaceOpWithNewDistributedOp<arith::ConstantOp>(
      provider, rewriter, constantOp, vectorType,
      DenseElementsAttr::get(vectorType, attr.getSplatValue<APFloat>()));
  return success();
}

LogicalResult VectorDistribution::distributeElementwise(RewriterBase &rewriter,
                                                        Operation *op) {
  assert(OpTrait::hasElementwiseMappableTraits(op) &&
         "expected elementwise op");
  // Replace vector operands with their distributed counter-parts.
  SmallVector<Value> operands;
  for (Value operand : op->getOperands()) {
    if (auto vectorOperand = dyn_cast<TypedValue<VectorType>>(operand)) {
      operand = getDistributed(rewriter, vectorOperand, provider);
    }
    operands.push_back(operand);
  }

  // Replace vector types with their distributed counter-parts.
  SmallVector<Type> resultTypes;
  for (Value result : op->getResults()) {
    if (auto vectorResult = dyn_cast<TypedValue<VectorType>>(result)) {
      // Distribute vector result types.
      auto newType =
          VectorType::get(provider->getDistributedShape(vectorResult),
                          vectorResult.getType().getElementType());
      resultTypes.push_back(newType);
    } else {
      resultTypes.push_back(result.getType());
    }
  }

  // Replace the original op with the distributed op.
  Operation *distributedOp =
      rewriter.create(op->getLoc(), op->getName().getIdentifier(), operands,
                      resultTypes, op->getAttrs());
  replaceOpWithDistributedValues(rewriter, op, provider,
                                 distributedOp->getResults());
  return success();
}

TypedValue<VectorType> VectorDistribution::reshapeVector(
    RewriterBase &rewriter, TypedValue<VectorType> src,
    LayoutAttr &currentLayout, LayoutAttr &targetLayout, Type elementType) {
  SmallVector<int64_t> targetShape =
      targetLayout.getSIMTVectorShape(provider->getSIMTLabels(targetLayout));
  auto newVectorType = VectorType::get(targetShape, elementType);
  Location loc = root->getLoc();
  arith::ConstantOp constantOp = rewriter.create<arith::ConstantOp>(
      loc, newVectorType, rewriter.getZeroAttr(newVectorType));
  auto newVector = cast<TypedValue<VectorType>>(constantOp.getResult());

  SmallVector<int64_t> currentShape =
      currentLayout.getSIMTVectorShape(provider->getSIMTLabels(currentLayout));
  int64_t innermostDim = targetShape.size() - 1;
  int64_t step =
      std::min(targetShape[innermostDim], currentShape[innermostDim]);
  DenseMap<LayoutDimension, int64_t> steps;
  LayoutDimension vecDim = provider->getInnerMostVecDim();
  steps[vecDim] = step;
  LayoutIterator srcIterator(currentLayout, steps);
  LayoutIterator targetIterator(targetLayout, steps);
  for (;
       !srcIterator.iterationComplete() && !targetIterator.iterationComplete();
       ++srcIterator, ++targetIterator) {
    auto srcOffset = srcIterator.getState().computeSIMTIndex(
        provider->getSIMTLabels(currentLayout));
    SmallVector<int64_t> sliceSize(srcOffset.size(), 1);
    SmallVector<int64_t> sliceStride(srcOffset.size(), 1);
    sliceSize[sliceSize.size() - 1] = step;
    Value slice = rewriter.create<vector::ExtractStridedSliceOp>(
        loc, src, srcOffset, sliceSize, sliceStride);
    auto targetOffset = targetIterator.getState().computeSIMTIndex(
        provider->getSIMTLabels(targetLayout));
    newVector = rewriter.create<vector::InsertStridedSliceOp>(
        loc, slice, newVector, targetOffset, sliceStride);
  }
  return newVector;
}

LogicalResult VectorDistribution::distributeResolutions(
    RewriterBase &rewriter,
    IREE::VectorExt::LayoutConflictResolutionOp resolutionOp) {
  TypedValue<VectorType> vector = resolutionOp.getInput();
  TypedValue<VectorType> result = resolutionOp.getOutput();
  LayoutAttr currentLayout = cast<LayoutAttr>(resolutionOp.getSourceLayout());
  LayoutAttr targetLayout = cast<LayoutAttr>(resolutionOp.getDesiredLayout());
  if (currentLayout.hasLaneConflictWith(targetLayout)) {
    return distributeTripToSharedMem(rewriter, resolutionOp);
  }
  SmallVector<int64_t> currentVecShape =
      currentLayout.getSIMTVectorShape(provider->getSIMTLabels(currentLayout));
  SmallVector<int64_t> targetVecShape =
      targetLayout.getSIMTVectorShape(provider->getSIMTLabels(targetLayout));
  if (currentVecShape.size() != targetVecShape.size()) {
    return failure();
  }
  auto numElements = [](ArrayRef<int64_t> vector) {
    return std::accumulate(vector.begin(), vector.end(), 1,
                           std::multiplies<int64_t>());
  };
  if (numElements(currentVecShape) != numElements(targetVecShape)) {
    return failure();
  }
  Type elementType = llvm::cast<VectorType>(result.getType()).getElementType();
  Value newVector =
      reshapeVector(rewriter, getDistributed(rewriter, vector, provider),
                    currentLayout, targetLayout, elementType);
  replaceOpWithDistributedValues(rewriter, resolutionOp, provider, newVector);
  return success();
}

/// Helper function for loop distribution. Given a list of bbArgs of the new
/// (distributed) loop op, wrap the distributed vector args (now distributed)
/// into ToSIMDOps, so that the block body can be moved over to the new op.
static SmallVector<Value> getBbArgsReplacements(RewriterBase &rewriter,
                                                Block::BlockArgListType bbArgs,
                                                ValueRange oldInits) {
  SmallVector<Value> replacements;
  for (auto [bbArg, oldInit] : llvm::zip_equal(bbArgs, oldInits)) {
    Value val = bbArg;
    if (auto oldVectorInit = dyn_cast<TypedValue<VectorType>>(oldInit)) {
      val = rewriter.create<IREE::VectorExt::ToSIMDOp>(
          oldVectorInit.getLoc(), oldVectorInit.getType(), val);
    }
    replacements.push_back(val);
  }
  return replacements;
}

LogicalResult VectorDistribution::distributeScfFor(RewriterBase &rewriter,
                                                   scf::ForOp forOp) {
  Block *oldLoopBody = forOp.getBody();

  // The new vector init_args of the loop.
  SmallVector<Value> newInitArgs;
  for (Value initArg : forOp.getInitArgs()) {
    if (auto vectorInitArg = dyn_cast<TypedValue<VectorType>>(initArg)) {
      initArg = getDistributed(rewriter, vectorInitArg, provider);
    }
    newInitArgs.push_back(initArg);
  }

  auto newForOp = rewriter.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newInitArgs);
  newForOp->setAttrs(forOp->getAttrs());
  Block *loopBody = newForOp.getBody();

  // Set up new iter_args. The loop body uses SIMD, so wrap the SIMD iter_args
  // of the new loop op into ToSIMDOps.
  rewriter.setInsertionPointToStart(loopBody);
  SmallVector<Value> iterArgs = getBbArgsReplacements(
      rewriter, newForOp.getRegionIterArgs(), forOp.getInitArgs());
  iterArgs.insert(iterArgs.begin(), newForOp.getInductionVar());

  // Move loop body to new loop.
  rewriter.mergeBlocks(oldLoopBody, loopBody, iterArgs);

  // Rpleace loop results.
  replaceOpWithDistributedValues(rewriter, forOp, provider,
                                 newForOp.getResults());
  return success();
}

LogicalResult VectorDistribution::distributeScfYield(RewriterBase &rewriter,
                                                     scf::YieldOp yieldOp) {
  SmallVector<Value> newOperands;
  for (Value operand : yieldOp.getOperands()) {
    if (auto vectorOperand = dyn_cast<TypedValue<VectorType>>(operand)) {
      operand = getDistributed(rewriter, vectorOperand, provider);
    }
    newOperands.push_back(operand);
  }
  rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, newOperands);
  return success();
}

LogicalResult VectorDistribution::distributeReductions(
    RewriterBase &rewriter, vector::MultiDimReductionOp reductionOp) {
  Location loc = reductionOp.getLoc();
  auto reductionDims = llvm::to_vector<4>(
      reductionOp.getReductionDims().getAsRange<IntegerAttr>());
  // Only support reduction on one dimension
  if (reductionDims.size() > 1)
    return failure();
  int reductionDim = reductionDims[0].getInt();
  // TODO: The following assumes only a 2D tensor
  int parallelDim = !reductionDim;
  vector::CombiningKind combiningKind = reductionOp.getKind();
  auto acc = cast<TypedValue<VectorType>>(reductionOp.getAcc());
  TypedValue<VectorType> source = reductionOp.getSource();

  Type elementType = llvm::cast<ShapedType>(acc.getType()).getElementType();
  int bitWidth = elementType.getIntOrFloatBitWidth();
  LayoutAttr layout = analysis.getLayout<LayoutAttr>(source);

  // Get information for reduction from layout
  uint64_t offset{0};
  auto maybeReductionLaneDim = layout.getLaneId(reductionDim);
  if (!maybeReductionLaneDim)
    return failure();
  int64_t laneSize = *layout.getShape(*maybeReductionLaneDim);
  switch (*maybeReductionLaneDim) {
  case LayoutDimension::LANEX:
    offset = 1;
    break;
  case LayoutDimension::LANEY:
    offset = *layout.getShape(LayoutDimension::LANEX);
    break;
  case LayoutDimension::LANEZ:
    offset = (*layout.getShape(LayoutDimension::LANEX)) *
             (*layout.getShape(LayoutDimension::LANEY));
    break;
  default:
    break;
  }

  auto resultVec = dyn_cast<TypedValue<VectorType>>(reductionOp.getResult());
  // For now, fail if result is not a vector.
  // TODO: Support this.
  if (!resultVec) {
    return failure();
  }
  auto storeVectorType =
      VectorType::get(provider->getDistributedShape(resultVec), elementType);
  Value storeVec = rewriter.create<arith::ConstantOp>(
      loc, storeVectorType, rewriter.getZeroAttr(storeVectorType));

  auto reduceFn = [&](const LayoutIterator::State &state) {
    SmallVector<int64_t> parallelSimtIndices =
        state.computeSIMTIndex(provider->getSIMTLabels(layout));
    Value mEmpty = rewriter.create<vector::ExtractOp>(
        loc, getDistributed(rewriter, acc, provider), parallelSimtIndices);
    int count{0};
    Value tmp, result, mask;
    if (bitWidth == 32) {
      tmp = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(VectorType::get({1}, elementType)));
    } else {
      tmp = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(VectorType::get({2}, elementType)));
    }

    auto reduceLocalFn = [&](const LayoutIterator::State &state) {
      SmallVector<int64_t> indices =
          state.computeSIMTIndex(provider->getSIMTLabels(layout));
      Value x = rewriter.create<vector::ExtractOp>(
          loc, getDistributed(rewriter, source, provider), indices);
      if (bitWidth == 32) {
        tmp = rewriter.create<vector::InsertOp>(loc, x, tmp,
                                                SmallVector<int64_t>{0});
      } else {
        int index = (count++) % 2;
        tmp = rewriter.create<vector::InsertOp>(loc, x, tmp,
                                                SmallVector<int64_t>{index});
        if (index == 0)
          return;
      }
      result = !result ? tmp
                       : makeArithReduction(rewriter, loc, combiningKind,
                                            result, tmp, nullptr, mask);
    };

    LayoutIterator reductionIterator(layout, reductionDim);
    reductionIterator.maybeFreezeAndConcatenate(state);
    reductionIterator.apply(reduceLocalFn);

    auto reduceGlobalFn = [&]() {
      uint32_t size{32};
      for (uint64_t i = offset; i < offset * laneSize; i <<= 1) {
        Value packed = packVectorToSupportedWidth(loc, rewriter, result);
        auto shuffleOp = rewriter.create<gpu::ShuffleOp>(loc, packed, i, size,
                                                         gpu::ShuffleMode::XOR);
        Value unpacked =
            unpackToVector(loc, rewriter, shuffleOp.getShuffleResult(),
                           result.getType().cast<VectorType>());
        result = makeArithReduction(rewriter, loc, combiningKind, unpacked,
                                    result, nullptr, mask);
      }

      // Convert to f16 or f32
      if (bitWidth == 32) {
        Value v0 = rewriter.create<vector::ExtractOp>(loc, result,
                                                      SmallVector<int64_t>{0});
        result = makeArithReduction(rewriter, loc, combiningKind, v0, mEmpty,
                                    nullptr, mask);
      } else {
        Value v0 = rewriter.create<vector::ExtractOp>(loc, result,
                                                      SmallVector<int64_t>{0});
        Value v1 = rewriter.create<vector::ExtractOp>(loc, result,
                                                      SmallVector<int64_t>{1});
        result = makeArithReduction(rewriter, loc, combiningKind, v0, v1,
                                    nullptr, mask);
        result = makeArithReduction(rewriter, loc, combiningKind, result,
                                    mEmpty, nullptr, mask);
      }
    };
    reduceGlobalFn();
    storeVec = rewriter.create<vector::InsertOp>(loc, result, storeVec,
                                                 parallelSimtIndices);
  };

  LayoutIterator parallelIterator(layout, parallelDim);
  parallelIterator.apply(reduceFn);
  replaceOpWithDistributedValues(rewriter, reductionOp, provider, storeVec);
  return success();
}

void distributeVectors(RewriterBase &rewriter, func::FuncOp funcOp) {
  VectorLayoutAnalysis analysis(funcOp);
  AMDCDNAGPULayoutProvider layoutProvider(analysis, funcOp);
  VectorDistribution distribution(funcOp, &layoutProvider, analysis);
  distribution.distribute();
}

// Distributed a broadcast + transpose which returns a 1D vector
// to its original 2D shape (prior to be folded along the reduction dimension).
LogicalResult
VectorDistribution::distributeBroadcasts(RewriterBase &rewriter,
                                         vector::BroadcastOp broadcastOp) {
  // Check if this is a broadcast with one use that is a transpose.
  if (!broadcastOp.getResult().hasOneUse())
    return failure();
  vector::TransposeOp transposeOp;
  for (Operation *user : broadcastOp.getResult().getUsers()) {
    transposeOp = dyn_cast<vector::TransposeOp>(user);
    if (transposeOp) {
      break;
    }
  }

  // For broadcast + transpose, reduction dimension = 1.
  // For broadcast, reduction dimension = 0.
  int64_t reductionDim;
  int64_t parallelDim;
  TypedValue<VectorType> result;
  Operation *opToReplace;
  if (!transposeOp) {
    result = broadcastOp.getResult();
    reductionDim = 0;
    parallelDim = 1;
    opToReplace = broadcastOp;
  } else {
    result = transposeOp.getResult();
    reductionDim = 1;
    parallelDim = 0;
    opToReplace = transposeOp;
  }

  TypedValue<VectorType> source =
      cast<TypedValue<VectorType>>(broadcastOp.getSource());
  auto layout = analysis.getLayout<LayoutAttr>(result);
  Type elementType = llvm::cast<ShapedType>(result.getType()).getElementType();
  auto vectorType =
      VectorType::get(provider->getDistributedShape(result), elementType);
  Location loc = broadcastOp.getLoc();
  Value broadcastedVector = rewriter.create<arith::ConstantOp>(
      loc, vectorType, rewriter.getZeroAttr(vectorType));

  auto broadcastFn = [&](const LayoutIterator::State &state) {
    SmallVector<int64_t> parallelSimtIndices =
        state.computeSIMTIndex(provider->getSIMTLabels(layout));
    Value value = rewriter.create<vector::ExtractOp>(
        loc, getDistributed(rewriter, source, provider), parallelSimtIndices);

    auto broadcastValue = [&](const LayoutIterator::State &state) {
      SmallVector<int64_t> indices =
          state.computeSIMTIndex(provider->getSIMTLabels(layout));

      broadcastedVector = rewriter.create<vector::InsertOp>(
          loc, value, broadcastedVector, indices);
    };

    // Mark parallel dims to have a step of their size, because we iterate over
    // them once only.
    LayoutIterator reductionIterator(layout, reductionDim);
    reductionIterator.maybeFreezeAndConcatenate(state);
    reductionIterator.apply(broadcastValue);
  };
  LayoutIterator parallelIterator(layout, parallelDim);
  parallelIterator.apply(broadcastFn);
  replaceOpWithDistributedValues(rewriter, opToReplace, provider,
                                 broadcastedVector);
  return success();
}

LogicalResult VectorDistribution::distributeTripToSharedMem(
    RewriterBase &rewriter,
    IREE::VectorExt::LayoutConflictResolutionOp resolutionOp) {
  TypedValue<VectorType> vector = resolutionOp.getInput();
  TypedValue<VectorType> result = resolutionOp.getOutput();
  VectorLayoutInterface targetLayout =
      analysis.getLayout<VectorLayoutInterface>(result);

  // We do a SIMD -> SIMD rewrite, and then distribute the result to get
  // SIMD -> SIMT.
  // TODO: Ideally, the distribution should know what is SIMD and what is SIMT.
  // So we should just be able to do a SIMD -> SIMD rewrite here and the
  // distribution should pickup on that.

  // Create a alloca which can store the vector in shared memory.
  // The alloca shape can be taken from the type of the output.
  ArrayRef<int64_t> shape = result.getType().getShape();
  Type elementType = result.getType().getElementType();
  auto memSpace =
      rewriter.getAttr<gpu::AddressSpaceAttr>(gpu::AddressSpace::Workgroup);
  auto memType = MemRefType::get(shape, elementType,
                                 MemRefLayoutAttrInterface{}, memSpace);
  auto allocaOp =
      rewriter.create<memref::AllocOp>(resolutionOp.getLoc(), memType);

  // Create a transfer_write to this new alloca.
  auto zeroIndex =
      rewriter.create<arith::ConstantIndexOp>(resolutionOp.getLoc(), 0);
  SmallVector<Value> indices(shape.size(), zeroIndex);
  auto transferWriteOp = rewriter.create<vector::TransferWriteOp>(
      resolutionOp.getLoc(), vector, allocaOp.getResult(), indices);

  // Create a barrier over this transfer_read.
  rewriter.create<gpu::BarrierOp>(resolutionOp.getLoc());

  // Create a transfer_write from this new alloca.
  SmallVector<VectorLayoutInterface> readLayouts = {targetLayout};
  auto transferReadOp = createSIMDOp<vector::TransferReadOp>(
      provider, rewriter, ArrayRef(readLayouts), resolutionOp.getLoc(),
      result.getType(), allocaOp.getResult(), indices);

  // Replace with the conflict result with the transfer_read result.
  rewriter.replaceOp(resolutionOp, transferReadOp.getResult());

  // Distribute the transfer_write.
  rewriter.setInsertionPoint(transferWriteOp);
  (void)distributeTransferWrites(rewriter, transferWriteOp);

  // Distribute the transfer_read.
  rewriter.setInsertionPoint(transferReadOp);
  (void)distributeTransferReads(rewriter, transferReadOp);

  return success();
}

void static prefetchLoadsInForLoop(RewriterBase &rewriter, scf::ForOp forOp) {
  // Find all vector.transfer_reads in the loop body.
  SmallVector<vector::TransferReadOp> loadOps;
  forOp.getBody()->walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<vector::TransferReadOp>(op)) {
      loadOps.push_back(loadOp);
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  int numPrefetched = loadOps.size();

  // Create a IR Mapping.
  IRMapping irMapping;

  // Map the loop induction variable to to lower bound of the loop.
  irMapping.map(forOp.getInductionVar(), forOp.getLowerBound());

  // Clone the load ops and map the indices to the loop induction variable.
  rewriter.setInsertionPoint(forOp);
  SmallVector<Value> prefetch0;
  for (vector::TransferReadOp loadOp : loadOps) {
    Operation *newOp = rewriter.clone(*loadOp, irMapping);
    prefetch0.push_back(newOp->getResult(0));
  }

  // Add prefetch0 to loop iter_args. Also replace init args usage inside the
  // loop.
  auto maybeNewFor =
      forOp.replaceWithAdditionalIterOperands(rewriter, prefetch0, false);
  scf::ForOp newForOp = cast<scf::ForOp>(*maybeNewFor);

  // Refresh the load ops since a new forOp was created.
  loadOps.clear();
  newForOp.getBody()->walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<vector::TransferReadOp>(op)) {
      loadOps.push_back(loadOp);
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  // Move the load ops after their first use to reduce register pressure.
  for (vector::TransferReadOp loadOp : loadOps) {
    Operation *firstUser = loadOp.getResult().use_begin()->getOwner();
    loadOp->moveAfter(firstUser);
  }

  // newFor.upper_bound = newFor.upper_bound - newFor.step
  rewriter.setInsertionPoint(newForOp);
  Value newUpperBound = rewriter.create<arith::SubIOp>(
      newForOp.getLoc(), newForOp.getUpperBound(), newForOp.getStep());
  newForOp.getUpperBoundMutable().assign(newUpperBound);

  // Move inside the loop now.
  rewriter.setInsertionPointToStart(newForOp.getBody());

  Value ivPlusOne = rewriter.create<arith::AddIOp>(
      newForOp.getLoc(), newForOp.getInductionVar(), newForOp.getStep());

  // Create an ir mapping from iv -> iv + 1
  IRMapping irMappingLoop;
  irMappingLoop.map(newForOp.getInductionVar(), ivPlusOne);

  // Get the last prefetch1.size() yield values and replace them with
  // loadOps.
  MutableArrayRef<OpOperand> yields =
      newForOp.getYieldedValuesMutable().take_back(numPrefetched);
  for (auto [newPrefetched, oldPrefetched] : llvm::zip_equal(loadOps, yields)) {
    oldPrefetched.set(newPrefetched);
  }

  // Replace all uses of the loadOps with last numPrefetched iter_args except
  // the yield operation.
  ArrayRef<BlockArgument> prefetchIterArgs =
      newForOp.getRegionIterArgs().take_back(numPrefetched);
  for (auto [load, iterArg] : llvm::zip_equal(loadOps, prefetchIterArgs)) {
    rewriter.replaceAllUsesExcept(load.getResult(), iterArg,
                                  newForOp.getBody()->getTerminator());
  }

  // Replace the loadOps with new loadOps with the new irMapping.
  for (vector::TransferReadOp &loadOp : loadOps) {
    rewriter.setInsertionPoint(loadOp);
    Operation *mappedLoadOp = rewriter.clone(*loadOp, irMappingLoop);
    rewriter.replaceOp(loadOp, mappedLoadOp);
    loadOp = cast<vector::TransferReadOp>(mappedLoadOp);
  }

  // Generate the final loop iteration without the loadOps.
  rewriter.setInsertionPointAfter(newForOp);
  IRMapping irMappingFinal;
  // Map iv -> newForOp.upperBound
  irMappingFinal.map(newForOp.getInductionVar(), newForOp.getUpperBound());

  // Map the iter_args to newFor return values.
  for (auto [iterArg, yield] :
       llvm::zip_equal(newForOp.getRegionIterArgs(), newForOp.getResults())) {
    irMappingFinal.map(iterArg, yield);
  }

  // Save all uses of the newForOp to replace after this cloning.
  SmallVector<OpOperand *> uses;
  for (Value res : newForOp.getResults()) {
    for (OpOperand &use : res.getUses()) {
      uses.push_back(&use);
    }
  }

  // Clone the loop body with the new irMapping but without the loadOps.
  for (Operation &op : newForOp.getBody()->getOperations()) {
    // Check if loadOps contains the current op.
    if (llvm::is_contained(loadOps, &op)) {
      continue;
    }

    // Do not clone the yield operation, but add a mapping for the newfor
    // results to the yield operands.
    if (isa<scf::YieldOp>(op)) {
      for (auto [result, operand] :
           llvm::zip_equal(newForOp.getResults(), op.getOperands())) {
        irMappingFinal.map(result, irMappingFinal.lookupOrNull(operand));
      }
      break;
    }

    // Map as we clone.
    Operation *newOp = rewriter.clone(op, irMappingFinal);
    for (auto [oldResult, newResult] :
         llvm::zip_equal(op.getResults(), newOp->getResults())) {
      irMappingFinal.map(oldResult, newResult);
    }
  }

  // Replace all uses of the newForOp with irMappingFinal.
  for (OpOperand *use : uses) {
    use->assign(irMappingFinal.lookup(use->get()));
  }
}

void prefetchLoads(RewriterBase &rewriter, func::FuncOp funcOp) {
  // Find all scf.for loop in funcOp.
  SmallVector<scf::ForOp> forOps;
  funcOp.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });
  for (scf::ForOp forOp : forOps) {
    prefetchLoadsInForLoop(rewriter, forOp);
  }
}

} // namespace mlir::iree_compiler
