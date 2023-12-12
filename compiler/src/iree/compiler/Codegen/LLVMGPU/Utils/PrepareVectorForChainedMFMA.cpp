// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace {

struct DoubleTranspose : public OpRewritePattern<vector::TransposeOp> {
  using OpRewritePattern<vector::TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    if (auto parentTransposeOp = transposeOp.getVector().getDefiningOp<vector::TransposeOp>()) {
       rewriter.replaceAllUsesWith(transposeOp, parentTransposeOp.getVector());
       return success();
    }
    return failure();
  }
};

struct FoldTransposeContract : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {
    auto lhs = contractOp.getLhs();
    auto rhs = contractOp.getRhs();
    vector::TransposeOp rhsDefOp = rhs.getDefiningOp<vector::TransposeOp>();
    vector::TransferReadOp lhsDefOp = lhs.getDefiningOp<vector::TransferReadOp>();
    if (!lhsDefOp && !rhsDefOp)
      return failure();
    AffineMap permutationMap = lhsDefOp.getPermutationMap();
    if (permutationMap.isIdentity())
      return failure();
    AffineExpr m, n, k;
    bindDims(rewriter.getContext(), m, n, k);
    auto indexingMaps = contractOp.getIndexingMapsArray();
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
    SmallVector<AffineMap> newIndexingMaps = infer({{m, k}, {k, n}, {m, n}});
    if (newIndexingMaps == indexingMaps)
        return failure();
    Value newResult = rewriter.create<vector::ContractionOp>(contractOp.getLoc(),
        lhs, rhsDefOp.getVector(), contractOp.getAcc(),
        rewriter.getAffineMapArrayAttr(newIndexingMaps),
        contractOp.getIteratorTypesAttr());
    rewriter.replaceOp(contractOp, newResult);
    return success();
  }
};


struct SwapContractOperands : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {
    auto lhs = contractOp.getLhs();
    auto rhs = contractOp.getRhs();
    auto acc = contractOp.getAcc();
    //auto lhsDefOp = lhs.getDefiningOp<vector::TransferReadOp>();
    //auto rhsDefOp = rhs.getDefiningOp<vector::TransferReadOp>();
    //if (!lhsDefOp || !rhsDefOp)
    //    return failure();
    AffineExpr m, n, k;
    bindDims(rewriter.getContext(), m, n, k);
    auto indexingMaps = contractOp.getIndexingMapsArray();
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
    SmallVector<AffineMap> newIndexingMaps = infer({{n, k}, {m, k}, {n, m}});
    if (indexingMaps == newIndexingMaps) return failure();
    Value transposedAcc = rewriter.create<vector::TransposeOp>(contractOp.getLoc(), acc, SmallVector<int64_t>{1, 0});
    Value newResult = rewriter.create<vector::ContractionOp>(contractOp.getLoc(),
        rhs, lhs, transposedAcc,
        rewriter.getAffineMapArrayAttr(newIndexingMaps),
        contractOp.getIteratorTypesAttr());
    transposedAcc = rewriter.create<vector::TransposeOp>(contractOp.getLoc(), newResult, SmallVector<int64_t>{1, 0});
    rewriter.replaceAllUsesWith(contractOp.getResult(), transposedAcc);
    return success();
  }

};

struct PropagateTransposeThroughReduction : public OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern<vector::MultiDimReductionOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::MultiDimReductionOp reductionOp,
                                PatternRewriter &rewriter) const override {
    Value source = reductionOp.getSource();
    auto definingOp = source.getDefiningOp();
    if (auto transposeOp = dyn_cast<vector::TransposeOp>(definingOp)) {
       SmallVector<bool> reductionMask;
       for (auto val : transposeOp.getPermutation()) {
         reductionMask.push_back(val ? true : false);
       }
       rewriter.replaceOpWithNewOp<vector::MultiDimReductionOp>(reductionOp,
            transposeOp.getVector(), reductionOp.getAcc(), reductionMask,
            reductionOp.getKind());
       return success();
    }
    return failure();
  }
};

template<typename T>
struct PropagateTransposeElementwiseConstant : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T elementwiseOp,
                                PatternRewriter &rewriter) const override {
    if (!OpTrait::hasElementwiseMappableTraits(elementwiseOp))
      return failure();

    // First get permutation and transposed operands in right spot
    ArrayRef<int64_t> permutation;
    SmallVector<Value> newOperands;
    for (auto operand : elementwiseOp->getOperands()) {
      if (auto transposeOp = operand.template getDefiningOp<vector::TransposeOp>()) {
        permutation = transposeOp.getPermutation();
        newOperands.push_back(transposeOp.getVector());
      } else {
        bool isConstant = matchPattern(operand, m_Constant());
        if (!isConstant) return failure();
        newOperands.push_back(Value());
      }
    }

    if (permutation.empty())
      return failure();

    for (auto operand : llvm::enumerate(elementwiseOp->getOperands())) {
      if (!newOperands[operand.index()])
      newOperands[operand.index()] = rewriter.create<vector::TransposeOp>(elementwiseOp.getLoc(),
        operand.value(), permutation);
    }

    auto resultType = VectorType::get(cast<ShapedType>(newOperands[0].getType()).getShape(),
                                      cast<ShapedType>(elementwiseOp.getResult().getType()).getElementType());
    Operation *newOp = rewriter.create(elementwiseOp.getLoc(), elementwiseOp->getName().getIdentifier(),
      newOperands, resultType, elementwiseOp->getAttrs());
    rewriter.replaceOpWithNewOp<vector::TransposeOp>(elementwiseOp, newOp->getResult(0), permutation);
    return success();
  }
};

template<typename T>
struct PropagateTransposeElementwise : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T elementwiseOp,
                                PatternRewriter &rewriter) const override {
    if (!OpTrait::hasElementwiseMappableTraits(elementwiseOp))
      return failure();
    bool allOperandsTransposed{true};
    SmallVector<Value> newOperands;
    ArrayRef<int64_t> permutation;
    for (auto operand : elementwiseOp->getOperands()) {
      if (auto transposeOp = operand.template getDefiningOp<vector::TransposeOp>()) {
        allOperandsTransposed &= true;
        newOperands.push_back(transposeOp.getVector());
        permutation = transposeOp.getPermutation();
      } else {
        allOperandsTransposed &= false;
        break;
      }
    }
    if (allOperandsTransposed) {
      auto resultType = VectorType::get(cast<ShapedType>(newOperands[0].getType()).getShape(),
                                        cast<ShapedType>(elementwiseOp.getResult().getType()).getElementType());
      Operation *newOp = rewriter.create(elementwiseOp.getLoc(), elementwiseOp->getName().getIdentifier(),
        newOperands, resultType, elementwiseOp->getAttrs());
      rewriter.replaceOpWithNewOp<vector::TransposeOp>(elementwiseOp, newOp->getResult(0), permutation);
      return success();
    }
    return failure();
  }
};


struct CanonicalizeForOpInductionVarShape final
    : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  Value FoldCarryDep(scf::ForOp forOp, Operation *ivDef) const {
    if (auto transposeOp = dyn_cast<vector::TransposeOp>(ivDef)) {
       return transposeOp.getVector();
    }
    return Value();
  }

  // Transfer the body of `source` into `dest` and update the terminator of
  // `dest` to use the specified results. The result list also replaces
  // any block arguments from `source` with the corresponding block argument
  // in `dest` and returns the updated result list.
  SmallVector<Value> transferBody(Block *source, Block *dest,
                                  ArrayRef<Value> results,
                                  PatternRewriter &rewriter) const {
    // Collect the old block arguments before merging.
    SmallVector<std::optional<int64_t>> maybeBlockArgNum;
    for (auto res : results) {
      if (auto blockArg = dyn_cast<BlockArgument>(res)) {
        maybeBlockArgNum.push_back(blockArg.getArgNumber());
      } else {
        maybeBlockArgNum.push_back(std::nullopt);
      }
    }
    // Move all operations to the destination block.
    rewriter.mergeBlocks(source, dest, dest->getArguments());
    // Replace the yield op by one that returns only the used values.
    auto yieldOp = cast<scf::YieldOp>(dest->getTerminator());
    // Create a new result set with the updated block arguments.
    SmallVector<Value> newResults;
    for (auto [index, argNum] : llvm::enumerate(maybeBlockArgNum)) {
      if (argNum) {
        newResults.push_back(dest->getArgument(*argNum));
      } else {
        newResults.push_back(results[index]);
      }
    }
    rewriter.updateRootInPlace(
        yieldOp, [&]() { yieldOp.getOperation()->setOperands(newResults); });
    return newResults;
  }

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {

    Value candidate;
    int64_t candidateIndex;
    vector::TransposeOp transposeOp;
    auto terminator = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    for (auto result : llvm::enumerate(terminator.getOperands())) {
      transposeOp = result.value().getDefiningOp<vector::TransposeOp>();
      if (transposeOp) {
        candidate = result.value();
        candidateIndex = result.index();
        break;
      }
    }
    if (!candidate) return failure();

    SmallVector<unsigned, 8> iteratorFolded;
    SmallVector<Operation *, 8> resultOps;
    auto returnValues = llvm::to_vector<8>(terminator.getOperands());
    auto initArgs = llvm::to_vector<8>(forOp.getInitArgs());
    for (auto [index, iterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
      if (index == candidateIndex) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(forOp);
        Value initArg = forOp.getInitArgs()[candidateIndex];
        Value transposedInitArg = rewriter.create<vector::TransposeOp>(forOp.getLoc(), initArg, transposeOp.getPermutation());
        iteratorFolded.push_back(index);
        returnValues[index] = transposeOp.getVector();
        IRMapping mapping;
        initArgs[index] = transposedInitArg;
      }
    }
    if (iteratorFolded.empty())
      return failure();
    auto newLoop = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), initArgs);
    SmallVector<Value> newReturnVals = transferBody(
        forOp.getBody(), newLoop.getBody(), returnValues, rewriter);
    Value bbArg = newLoop.getRegionIterArg(candidateIndex);
    for (Operation *user : bbArg.getUsers()) {
      rewriter.setInsertionPoint(user);
      Value transposed = rewriter.create<vector::TransposeOp>(user->getLoc(), bbArg, transposeOp.getPermutation());
      IRMapping mapping;
      mapping.map(bbArg, transposed);
      rewriter.replaceOp(user, rewriter.clone(*user, mapping)->getResult(0));
    }

    // Replace the operation by the new one.
    SmallVector<Value, 8> repResults;
    for (auto result : llvm::enumerate(forOp.getResults())) {
      if (result.index() == candidateIndex) {
        rewriter.setInsertionPointAfter(forOp);
        Value transposedResult = rewriter.create<vector::TransposeOp>(forOp.getLoc(),
                                        newLoop.getResult(result.index()),
                                        transposeOp.getPermutation());
        repResults.push_back(transposedResult);
        continue;
      }
      repResults.push_back(newLoop.getResult(result.index()));
    }
    rewriter.replaceOp(forOp, repResults);
    return success();
  }
};

struct CombineTransposeTransferWriteOp
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp transferWriteOp,
                                PatternRewriter &rewriter) const override {
    // Look through integer extend ops.
    Value vector = transferWriteOp.getVector();

    auto op = vector.getDefiningOp<vector::TransposeOp>();
    if (!op)
      return failure();

    if (transferWriteOp.getMask() || transferWriteOp.hasOutOfBoundsDim())
      return rewriter.notifyMatchFailure(op, "not inbounds transfer write");

    AffineMap permutationMap =
        AffineMap::getPermutationMap(op.getPermutation(), op.getContext());
    AffineMap newMap =
        permutationMap.compose(transferWriteOp.getPermutationMap());

     rewriter
         .replaceOpWithNewOp<vector::TransferWriteOp>(
             transferWriteOp, op.getVector(), transferWriteOp.getSource(),
             transferWriteOp.getIndices(), AffineMapAttr::get(newMap),
             transferWriteOp.getMask(), transferWriteOp.getInBoundsAttr());

    return success();
  }
};

}

namespace mlir {
namespace iree_compiler {

void populatePrepareVectorForChainedMFMAPatterns(RewritePatternSet &patterns) {
  patterns.add<SwapContractOperands>(patterns.getContext());
}

void populateTransposePropagationPatterns(RewritePatternSet &patterns) {
  patterns.add<PropagateTransposeThroughReduction,
               PropagateTransposeElementwise<arith::SubFOp>,
               PropagateTransposeElementwise<arith::MulFOp>,
               PropagateTransposeElementwise<math::Exp2Op>,
               PropagateTransposeElementwise<arith::TruncFOp>,
               PropagateTransposeElementwiseConstant<arith::DivFOp>,
               DoubleTranspose,
               CanonicalizeForOpInductionVarShape
               >(patterns.getContext());
}

void populateFoldTransposeContractPatterns(RewritePatternSet &patterns) {
  patterns.add<FoldTransposeContract
               >(patterns.getContext());
}

void populateTransferWritePatterns(RewritePatternSet &patterns) {
  patterns.add<CombineTransposeTransferWriteOp
               >(patterns.getContext());
}

} // namespace iree_compiler
} // namespace mlir