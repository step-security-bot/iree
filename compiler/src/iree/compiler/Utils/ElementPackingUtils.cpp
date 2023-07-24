// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/ElementPackingUtils.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace iree_compiler {

/// Returns true if the given |bitWidth|, if appearing at runtime-kernel
/// interface, is less than a byte that should be tightly packed together.
static bool needToPackSubByteElementBitWidth(unsigned bitWidth) {
  // Require the original bit width to be some power of two for now to avoid
  // trickiness and weirdness of packing and cross-byte access.
  // Also disallow boolean values for now--they may require separate interface
  // choices.
  return bitWidth < 8 && llvm::isPowerOf2_32(bitWidth) && bitWidth != 1;
}

Type legalizeStorageElementType(Type elementType) {
  // Only handle integers; floats in MLIR all have aligned widths (today).
  auto intType = dyn_cast<IntegerType>(elementType);
  if (!intType)
    return elementType;

  // For sub-byte elements, default to pack them into bytes.
  unsigned bitWidth = intType.getWidth();
  if (needToPackSubByteElementBitWidth(bitWidth))
    return elementType;

  // Otherwise, extend them to the next power-of-two bit width.
  unsigned alignedBitWidth =
      IREE::Util::getRoundedElementByteWidth(intType) * 8;
  if (alignedBitWidth == bitWidth)
    return elementType;
  return IntegerType::get(elementType.getContext(), alignedBitWidth,
                          intType.getSignedness());
}

} // namespace iree_compiler
} // namespace mlir
