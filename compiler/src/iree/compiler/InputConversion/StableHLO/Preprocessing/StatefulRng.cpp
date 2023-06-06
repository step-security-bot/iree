// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/MLIRContext.h>

#include <numeric>
#include <random>

#include "iree/compiler/InputConversion/StableHLO/Preprocessing/Passes.h"
#include "iree/compiler/InputConversion/StableHLO/Preprocessing/Rewriters.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_STATEFULRNG
#include "iree/compiler/InputConversion/StableHLO/Preprocessing/Passes.h.inc"

namespace {

using GlobalFn = std::function<ml_program::GlobalOp()>;

class ExpandRngUniform : public OpRewritePattern<::mlir::stablehlo::RngOp> {
 public:
  ExpandRngUniform(MLIRContext *context, GlobalFn &getGlobal)
      : OpRewritePattern<::mlir::stablehlo::RngOp>::OpRewritePattern(context),
        getGlobal(getGlobal){};

  LogicalResult matchAndRewrite(::mlir::stablehlo::RngOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getRngDistribution() != ::mlir::stablehlo::RngDistribution::UNIFORM)
      return failure();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ml_program::GlobalOp global = getGlobal();
    auto symbol = SymbolRefAttr::get(global.getSymNameAttr());
    auto val = b.create<ml_program::GlobalLoadOp>(global.getType(), symbol);
    auto algo = ::mlir::stablehlo::RngAlgorithm::THREE_FRY;
    auto rngOp = b.create<::mlir::stablehlo::RngBitGeneratorOp>(
        TypeRange{val.getType(), op.getType()}, algo, val);
    b.create<ml_program::GlobalStoreOp>(symbol, rngOp.getOutputState());

    rewriter.replaceOp(op, rngOp.getOutput());
    return success();
  }

  GlobalFn &getGlobal;
};

struct StatefulRngPass : public impl::StatefulRngBase<StatefulRngPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ml_program::MLProgramDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    ml_program::GlobalOp global;

    auto getGlobal = [&]() {
      if (global) return global;

      ModuleOp module = getOperation();
      OpBuilder globalBuilder(module.getBodyRegion());

      // Use an arbitrary seed value. This value is public so that if desired
      // the seed could be set on lowered program.
      std::vector<uint32_t> vals({12, 34, 56, 78});
      RankedTensorType ty =
          RankedTensorType::get(4, globalBuilder.getIntegerType(32));
      auto initValue = DenseIntElementsAttr::get(ty, vals);

      global = globalBuilder.create<ml_program::GlobalOp>(
          module.getLoc(), "global_hlo_rng_state", ty,
          /*is_mutable=*/true, initValue,
          /*visibility=*/globalBuilder.getStringAttr("public"));
      return global;
    };
    GlobalFn g = getGlobal;

    patterns.insert<ExpandRngUniform>(context, g);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createStatefulRngPreprocessingPass() {
  return std::make_unique<StatefulRngPass>();
}

}  // namespace mlir::iree_compiler::stablehlo
