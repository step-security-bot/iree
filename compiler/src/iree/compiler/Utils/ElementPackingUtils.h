// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_TYPEUTILS_H_
#define IREE_COMPILER_UTILS_TYPEUTILS_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace iree_compiler {

/// Legalizes the given |elementType| for storage.
///
/// In IREE, if compiling from the same source model, we control both the
/// runtime and kernel. For such cases, we perform tight packing for supported
/// sub-byte elements, and expand to the next power-of-two bit width for other
/// cases.
Type legalizeStorageElementType(Type elementType);

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_UTILS_TYPEUTILS_H_
