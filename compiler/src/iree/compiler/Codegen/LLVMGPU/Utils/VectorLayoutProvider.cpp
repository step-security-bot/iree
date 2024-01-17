// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Utils/VectorLayoutProvider.h"
#include "VectorLayoutProvider.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir::iree_compiler::IREE::VectorExt;
using namespace mlir::iree_compiler;
using namespace mlir;

static std::tuple<uint32_t, uint32_t, uint32_t>
getCanonicalDims(AMDCDNAGPULayoutProvider::MFMAType type) {
  switch (type) {
  case AMDCDNAGPULayoutProvider::MFMAType::F16_16x16x16_F32:
    return {16, 16, 16};
  case AMDCDNAGPULayoutProvider::MFMAType::F16_32x32x8_F32:
    return {32, 32, 8};
  default:
    return {0, 0, 0};
  }
}

static SmallVector<int64_t> getCanonicalShape(int64_t M, int64_t N, int64_t K,
                                              ContractMatrixType matrixType,
                                              ContractType contractType) {
  SmallVector<int64_t> shape;
  switch (matrixType) {
  case ContractMatrixType::A:
    shape = contractType == ContractType::MTM ? SmallVector<int64_t>{K, M}
                                              : SmallVector<int64_t>{M, K};
    break;
  case ContractMatrixType::B:
    shape = contractType == ContractType::MMT ? SmallVector<int64_t>{N, K}
                                              : SmallVector<int64_t>{K, N};
    break;
  default:
    shape = {M, N};
  }
  return shape;
}

static PerDimLayoutAttr createPerDimLayout(MLIRContext *ctx,
                                           ArrayRef<LayoutDimension> dims,
                                           ArrayRef<int64_t> shapes) {
  SmallVector<LayoutDimensionAttr> dimAttrs;
  for (auto dim : dims)
    dimAttrs.push_back(LayoutDimensionAttr::get(ctx, dim));
  return PerDimLayoutAttr::get(ctx, dimAttrs, shapes);
}

LayoutAttr AMDCDNAGPULayoutProvider::getCanonicalMFMALayout(
    TypedValue<VectorType> value, ContractMatrixType matrixType,
    int64_t numElements, ContractType contractType,
    SmallVector<int64_t> permutation) {
  MLIRContext *ctx = root->getContext();

  // Get MMA matrix info.
  auto [M, N, K] = getCanonicalDims(mfmaType);
  ArrayRef<int64_t> matrixShape = value.getType().getShape();
  SmallVector<int64_t> canonicalShape =
      getCanonicalShape(M, N, K, matrixType, contractType);

  // Get batch sizes.
  int64_t batchRow = matrixShape[0] / canonicalShape[0];
  // If we load more elements, corresponds to smaller batch size.
  int64_t multiplier = numElements >= 4 ? numElements / 4 : 1;
  int64_t batchCol = matrixShape[1] / canonicalShape[1] / multiplier;

  auto rowLayout = [&](LayoutDimension batchDim,
                       int64_t batch) -> PerDimLayoutAttr {
    if (mfmaType == AMDCDNAGPULayoutProvider::MFMAType::F16_32x32x8_F32)
      return createPerDimLayout(ctx, {batchDim, LayoutDimension::LANEX},
                              {batch, 32});
    return createPerDimLayout(ctx, {batchDim, LayoutDimension::LANEX},
                            {batch, 16});
  };

  auto colLayout = [&](LayoutDimension batchDim, int64_t batch,
                       int64_t numElements) -> PerDimLayoutAttr {
    if (mfmaType == AMDCDNAGPULayoutProvider::MFMAType::F16_32x32x8_F32) {
      if ((matrixType == ContractMatrixType::C) || (matrixType == ContractMatrixType::D))
        return createPerDimLayout(ctx, {batchDim, LayoutDimension::VECTORY, LayoutDimension::LANEY, LayoutDimension::VECTORX},
                              {batch, 4, 2, numElements});
      return createPerDimLayout(
        ctx, {batchDim, LayoutDimension::LANEY, LayoutDimension::VECTORX},
        {batch, 2, numElements});
    }
    return createPerDimLayout(
      ctx, {batchDim, LayoutDimension::LANEY, LayoutDimension::VECTORX},
      {batch, 4, numElements});
  };

  // For MMT contract, the layout of B is same as layout of A.
  if (contractType == ContractType::MMT &&
      matrixType == ContractMatrixType::B) {
    matrixType = ContractMatrixType::A;
  }

  // Set layout based on matrix type.
  LayoutAttr layout;
  switch (matrixType) {
  case ContractMatrixType::A:
    layout = LayoutAttr::get(
        ctx, {rowLayout(LayoutDimension::BATCHX, batchRow),
              colLayout(LayoutDimension::BATCHY, batchCol, numElements)});
    break;
  case ContractMatrixType::B:
    layout = LayoutAttr::get(
        ctx, {colLayout(LayoutDimension::BATCHX, batchRow, numElements),
              rowLayout(LayoutDimension::BATCHY, batchCol)});
    break;
  default:
    layout =
        LayoutAttr::get(ctx, {colLayout(LayoutDimension::BATCHX, batchRow, 4),
                              rowLayout(LayoutDimension::BATCHY, batchCol)});
  }

  if (!permutation.empty()) {
    layout = cast<LayoutAttr>(layout.permute(permutation));
  }
  return layout;
}

bool AMDCDNAGPULayoutProvider::hasCanonicalShape(ContractMatrixType matrixType,
                                                 ArrayRef<int64_t> shape) {
  SmallVector<int64_t> Ashape, Bshape, Cshape;
  switch (mfmaType) {
  case AMDCDNAGPULayoutProvider::MFMAType::F16_16x16x16_F32:
    Ashape = Bshape = Cshape = {16, 16};
    break;
  case AMDCDNAGPULayoutProvider::MFMAType::F16_32x32x8_F32:
    Ashape = {32, 8};
    Bshape = {8, 32};
    Cshape = {32, 32};
    break;
  default:
    return false;
  }

  switch (matrixType) {
  case ContractMatrixType::A:
    return Ashape == shape;
  case ContractMatrixType::B:
    return Bshape == shape;
  case ContractMatrixType::C:
  case ContractMatrixType::D:
    return Cshape == shape;
  default:
    return false;
  }
}

static SmallVector<int64_t> getPermutation(MLIRContext *ctx, AffineMap inputMap,
                                           ContractMatrixType matrixType,
                                           ContractType contractType) {
  AffineExpr d0, d1, d2;
  bindDims(ctx, d0, d1, d2);
  SmallVector<AffineMap> validMaps;
  switch (matrixType) {
  case ContractMatrixType::A:
  case ContractMatrixType::B:
    if (contractType == ContractType::MMT) {
      validMaps.push_back(AffineMap::get(3, 0, {d0, d2}, ctx));
      validMaps.push_back(AffineMap::get(3, 0, {d1, d2}, ctx));
    } else {
      // Contract Type MM
      validMaps.push_back(AffineMap::get(3, 0, {d0, d2}, ctx));
      validMaps.push_back(AffineMap::get(3, 0, {d2, d1}, ctx));
    }
    break;
  default:
    validMaps.push_back(AffineMap::get(3, 0, {d0, d1}, ctx));
    validMaps.push_back(AffineMap::get(3, 0, {d1, d0}, ctx));
  }
  for (AffineMap map : validMaps) {
    if (inputMap == map)
      return SmallVector<int64_t>{0, 1};
  }
  return SmallVector<int64_t>{1, 0};
}

static ContractType inferContractType(MLIRContext *ctx,
                                      SmallVector<AffineMap> &indexingMaps) {
  SmallVector<bool> operandsTransposed(3, false);
  AffineExpr d0, d1, d2;
  bindDims(ctx, d0, d1, d2);
  for (int i = 0; i < indexingMaps.size(); i++) {
    if ((i == 0) || (i == 1)) {
      auto validMap = AffineMap::get(3, 0, {d0, d2}, ctx);
      auto validMapT = AffineMap::get(3, 0, {d1, d2}, ctx);
      if ((indexingMaps[i] != validMap) && (indexingMaps[i] != validMapT))
        operandsTransposed[i] = true;
    }
  }
  if (!operandsTransposed[0] && !operandsTransposed[1])
    return ContractType::MMT;
  if (operandsTransposed[0] && !operandsTransposed[1])
    return ContractType::MTMT;
  if (!operandsTransposed[0] && operandsTransposed[1])
    return ContractType::MM;
  return ContractType::MTM;
}


void AMDCDNAGPULayoutProvider::setAnchorOps() {
  root->walk([&](Operation *op) {
    MLIRContext *ctx = op->getContext();
    TypeSwitch<Operation *, void>(op)
        .Case<vector::ContractionOp>([&](vector::ContractionOp contractOp) {
          AffineMap mapA = contractOp.getIndexingMapsArray()[0];
          SmallVector<AffineMap> indexingMaps(contractOp.getIndexingMapsArray());
          ContractType contractType = inferContractType(ctx, indexingMaps);
          LayoutAttr layoutA = getCanonicalMFMALayout(
              contractOp.getLhs(), ContractMatrixType::A, 4,
              contractType, getPermutation(ctx, mapA, ContractMatrixType::A, contractType));
          analysis.setAnchorForOperand(contractOp->getOpOperand(0), layoutA);
          AffineMap mapB = contractOp.getIndexingMapsArray()[1];
          LayoutAttr layoutB = getCanonicalMFMALayout(
              contractOp.getRhs(), ContractMatrixType::B, 4,
              contractType, getPermutation(ctx, mapB, ContractMatrixType::B, contractType));
          analysis.setAnchorForOperand(contractOp->getOpOperand(1), layoutB);
          // Do not allow scalar result.
          if (isa<VectorType>(contractOp.getAccType()) &&
              isa<VectorType>(contractOp.getResultType())) {
            auto acc = cast<TypedValue<VectorType>>(contractOp.getAcc());
            auto result = cast<TypedValue<VectorType>>(contractOp.getResult());
            AffineMap mapC = contractOp.getIndexingMapsArray()[2];
            LayoutAttr layoutC = getCanonicalMFMALayout(
                acc, ContractMatrixType::C, 1,
                contractType, getPermutation(ctx, mapC, ContractMatrixType::C, contractType));
            analysis.setAnchorForOperand(contractOp->getOpOperand(2), layoutC);
            analysis.setAnchorForValue(result, layoutC);
          }
        })
        .Case<vector::TransferReadOp>([&](auto transferReadOp) {
          for (OpOperand &use : transferReadOp.getVector().getUses()) {
            // If this use is a scf.for op, then find the corressponding
            // iter_arg and find the use.
            Operation *user = use.getOwner();
            TypedValue<VectorType> vector = transferReadOp.getVector();
            if (auto forOp = dyn_cast<scf::ForOp>(user)) {
              int operandNum = use.getOperandNumber();
              user = forOp.getRegionIterArgs()[operandNum - 3]
                         .getUses()
                         .begin()
                         ->getOwner();
              vector = cast<TypedValue<VectorType>>(forOp.getRegionIterArgs()[operandNum - 3]);
              user = *vector.getUsers().begin();
            }

            if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
              int operandNum = use.getOperandNumber();
              scf::ForOp forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
              if (forOp) {
                user = forOp.getRegionIterArgs()[operandNum]
                           .getUses()
                           .begin()
                           ->getOwner();
                vector = cast<TypedValue<VectorType>>(forOp.getRegionIterArgs()[operandNum]);
                user = *vector.getUsers().begin();
              }
            }

            if (auto contractOp = dyn_cast<vector::ContractionOp>(user)) {
              ArrayRef<int64_t> lhsShape =
                  cast<ShapedType>(contractOp.getLhs().getType()).getShape();
              ArrayRef<int64_t> rhsShape =
                  cast<ShapedType>(contractOp.getRhs().getType()).getShape();
              if (hasCanonicalShape(ContractMatrixType::A, lhsShape) ||
                  hasCanonicalShape(ContractMatrixType::B, rhsShape))
                return;
              SmallVector<AffineMap> indexingMaps(contractOp.getIndexingMapsArray());
              ContractType contractType = inferContractType(ctx, indexingMaps);
              if (vector == contractOp.getLhs()) {
                vector = transferReadOp.getVector();
                LayoutAttr layoutLhs =
                    getCanonicalMFMALayout(vector, ContractMatrixType::A, 8, contractType);
                analysis.setAnchorForValue(vector, layoutLhs);
              }
              if (vector == contractOp.getRhs()) {
                vector = transferReadOp.getVector();
                LayoutAttr layoutRhs =
                    getCanonicalMFMALayout(vector, ContractMatrixType::B, 8, contractType);
                analysis.setAnchorForValue(vector, layoutRhs);
              }
              break;
            }
          }
        })
        .Default([&](Operation *op) { return; });
  });
}

SmallVector<int64_t>
AMDCDNAGPULayoutProvider::getDistributedShape(TypedValue<VectorType> value) {
  auto layout = analysis.getLayout<LayoutAttr>(value);
  return layout.getSIMTVectorShape(simtLabels);
}

SmallVector<int64_t> AMDCDNAGPULayoutProvider::getContractIndices(
    ContractType contractType, ContractMatrixType matrixType, int i, int j) {
  if (matrixType == ContractMatrixType::A) {
    switch (contractType) {
    case ContractType::MM:
    case ContractType::MMT:
      return SmallVector<int64_t>{i, j};
    case ContractType::MTM:
    case ContractType::MTMT:
      return SmallVector<int64_t>{j, i};
    }
  }

  if (matrixType == ContractMatrixType::B) {
    switch (contractType) {
    case ContractType::MM:
    case ContractType::MTM:
      return SmallVector<int64_t>{i, j};
    case ContractType::MMT:
    case ContractType::MTMT:
      return SmallVector<int64_t>{j, i};
    }
  }
  return SmallVector<int64_t>{i, j};
}

Value AMDCDNAGPULayoutProvider::computeMMA(Value a, Value b, Value c,
                                           Location loc, OpBuilder &rewriter) {
  uint32_t m, n, k, blks;
  if (mfmaType == AMDCDNAGPULayoutProvider::MFMAType::F16_16x16x16_F32) {
    m = n = k = 16;
  } else if (mfmaType == AMDCDNAGPULayoutProvider::MFMAType::F16_32x32x8_F32) {
    m = n = 32;
    k = 8;
  }
  blks = 1;
  return rewriter.create<amdgpu::MFMAOp>(loc, c.getType(), m, n, k, blks, a, b,
                                         c);
}

int64_t AMDCDNAGPULayoutProvider::getKDimension(int64_t rowBatch,
                                                int64_t colBatch,
                                                ContractType contractType) {
  if (contractType == ContractType::MTM) {
    return rowBatch;
  }
  return colBatch;
}


static LogicalResult
distributeContractionsToMFMA(RewriterBase &rewriter,
                             AMDCDNAGPULayoutProvider *provider,
                             vector::ContractionOp contractOp) {
  VectorLayoutAnalysis &analysis = provider->getAnalysis();
  TypedValue<VectorType> lhs = contractOp.getLhs();
  TypedValue<VectorType> rhs = contractOp.getRhs();
  Value accVal = contractOp.getAcc();
  if (!isa<VectorType>(accVal.getType()))
    return failure();
  TypedValue<VectorType> acc = cast<TypedValue<VectorType>>(accVal);
  Location loc = contractOp.getLoc();
  TypedValue<VectorType> result =
      cast<TypedValue<VectorType>>(contractOp.getResult());
  LayoutAttr layout = analysis.getLayout<LayoutAttr>(result);
  LayoutAttr lhsLayout = analysis.getLayout<LayoutAttr>(lhs);
  Type elementType = llvm::cast<ShapedType>(acc.getType()).getElementType();
  SmallVector<int64_t> vectorShape = provider->getDistributedShape(result);
  auto vectorType = VectorType::get(vectorShape, elementType);
  Value vector = rewriter.create<arith::ConstantOp>(
      loc, vectorType, rewriter.getZeroAttr(vectorType));
  // Determine contraction type from indexing maps
  SmallVector<AffineMap> indexingMaps = contractOp.getIndexingMapsArray();
  ContractType contractType =
      inferContractType(contractOp.getContext(), indexingMaps);
  int K = provider->getKDimension(lhsLayout.getBatchDim(0),
                                  lhsLayout.getBatchDim(1), contractType);
  auto contractFn = [&](const LayoutIterator::State &state) {
    SmallVector<int64_t> simtIndices = state.computeIteratorProjectedSIMTIndex(
        provider->getSIMTLabels(layout));
    Value dMatrix = rewriter.create<vector::ExtractOp>(
        loc, getDistributed(rewriter, acc, provider), simtIndices);
    for (int k = 0; k < K; k++) {
      Value aMatrix = rewriter.create<vector::ExtractOp>(
          loc, getDistributed(rewriter, lhs, provider),
          provider->getContractIndices(contractType, ContractMatrixType::A,
                                       simtIndices[0], k));
      Value bMatrix = rewriter.create<vector::ExtractOp>(
          loc, getDistributed(rewriter, rhs, provider),
          provider->getContractIndices(contractType, ContractMatrixType::B, k,
                                       simtIndices[1]));
      dMatrix = provider->computeMMA(aMatrix, bMatrix, dMatrix, loc, rewriter);
    }
    vector =
        rewriter.create<vector::InsertOp>(loc, dMatrix, vector, simtIndices);
  };
  DenseMap<LayoutDimension, int64_t> strides;
  LayoutIterator iterator(layout, strides);
  LayoutIterator batchIterator = iterator.getBatchIterator();
  batchIterator.apply(contractFn);
  replaceOpWithDistributedValues(rewriter, contractOp, provider, vector);
  return success();
}

LogicalResult
AMDCDNAGPULayoutProvider::specializedDistribution(RewriterBase &rewriter,
                                                  Operation *op) {
  // Do specialized mfma distribution.
  if (auto contractOp = dyn_cast<vector::ContractionOp>(op)) {
    return distributeContractionsToMFMA(rewriter, this, contractOp);
  }
  return failure();
}
