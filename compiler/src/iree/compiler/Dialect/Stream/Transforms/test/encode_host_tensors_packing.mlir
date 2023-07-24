// RUN: iree-opt --split-input-file --mlir-print-local-scope --iree-stream-encode-host-tensors --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @denseTensorConstantI2()
func.func @denseTensorConstantI2() -> !stream.resource<constant> {
  // CHECK: %[[STATIC_SIZE:.+]] = arith.constant 4 : index
  // CHECK: %[[RESULT:.+]] = stream.async.constant : !stream.resource<constant>{%[[STATIC_SIZE]]} =
  // CHECK-SAME: dense<[0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1]> : tensor<16xi2>
  %0 = stream.tensor.constant : tensor<16xi2> in !stream.resource<constant> = dense<[
    0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
  ]> : tensor<16xi2>
  // CHECK: return %[[RESULT]]
  return %0 : !stream.resource<constant>
}

// -----

// Ensures that a non-power-of-two type (i3) constant is stored as packed values
// in i8 that have 2 bits of zero padding per i8.

// CHECK: func.func @denseTensorConstantI3()
func.func @denseTensorConstantI3() -> !stream.resource<constant> {
  // CHECK: %[[STATIC_SIZE:.+]] = arith.constant 2 : index
  // CHECK: %[[RESULT:.+]] = stream.async.constant : !stream.resource<constant>{%[[STATIC_SIZE]]} = dense<[0, -1, 2, -3]> : tensor<4xi3>
  %0 = stream.tensor.constant : tensor<4xi3> in !stream.resource<constant> = dense<[0, -1, 2, -3]> : tensor<4xi3>
  // CHECK: return %[[RESULT]]
  return %0 : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @denseTensorConstantI4
func.func @denseTensorConstantI4() -> !stream.resource<constant> {
  // CHECK: %[[STATIC_SIZE:.+]] = arith.constant 4 : index
  // CHECK: %[[RESULT:.+]] = stream.async.constant : !stream.resource<constant>{%[[STATIC_SIZE]]} = dense<[5, -1, 0, 3, 1, 7, -8, 4]> : tensor<8xi4>
  %0 = stream.tensor.constant : tensor<8xi4> in !stream.resource<constant> = dense<[5, 15, 0, 3, 1, 7, 8, 4]> : tensor<8xi4>
  // CHECK: return %[[RESULT]]
  return %0 : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @denseTensorConstantPaddedI4
func.func @denseTensorConstantPaddedI4() -> !stream.resource<constant> {
  // CHECK: %[[STATIC_SIZE:.+]] = arith.constant 3 : index
  // CHECK: %[[RESULT:.+]] = stream.async.constant : !stream.resource<constant>{%[[STATIC_SIZE]]} = dense<[5, -1, 0, 3, 1]> : tensor<5xi4>
  %0 = stream.tensor.constant : tensor<5xi4> in !stream.resource<constant> = dense<[5, -1, 0, 3, 1]> : tensor<5xi4>
  // CHECK: return %[[RESULT]]
  return %0 : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @denseTensorConstantI8
func.func @denseTensorConstantI8() -> !stream.resource<constant> {
  // CHECK: %[[STATIC_SIZE:.+]] = arith.constant 8 : index
  // CHECK: %[[RESULT:.+]] = stream.async.constant : !stream.resource<constant>{%[[STATIC_SIZE]]} = dense<[5, 15, 0, 3, 1, 7, 8, 4]> : tensor<8xi8>
  %0 = stream.tensor.constant : tensor<8xi8> in !stream.resource<constant> = dense<[5, 15, 0, 3, 1, 7, 8, 4]> : tensor<8xi8>
  // CHECK: return %[[RESULT]]
  return %0 : !stream.resource<constant>
}

// -----

// CHECK-LABEL: @denseTensorSizeOfStatic
func.func @denseTensorSizeOfStatic() -> index {
  // CHECK-DAG: %[[STATIC_SIZE:.+]] = arith.constant 6 : index
  %0 = stream.tensor.sizeof tensor<12xi4> : index
  // CHECK: return %[[STATIC_SIZE]]
  return %0 : index
}

// -----

// CHECK-LABEL: @denseTensorSizeOfStaticPadded
func.func @denseTensorSizeOfStaticPadded() -> index {
  // CHECK-DAG: %[[STATIC_SIZE:.+]] = arith.constant 6 : index
  %0 = stream.tensor.sizeof tensor<11xi4> : index
  // CHECK: return %[[STATIC_SIZE]]
  return %0 : index
}

// -----

// CHECK-LABEL: @denseTensorSizeOfDynamic
func.func @denseTensorSizeOfDynamic(%arg0: index) -> index {
  // CHECK: %[[DYNAMIC_SIZE:.+]] = affine.apply affine_map<()[s0] -> ((s0 * 5) ceildiv 2)>()[%arg0]
  %0 = stream.tensor.sizeof tensor<?x5xi4>{%arg0} : index
  // CHECK: return %[[DYNAMIC_SIZE]]
  return %0 : index
}

// -----

// Checks that stream.tensor.load with sub-byte packing is not supported right now.

// CHECK-LABEL: @denseTensorLoad
func.func @denseTensorLoad(%arg0: !stream.resource<staging>, %arg1: index, %arg2: index, %arg3: index) -> i4 {
  %c0 = arith.constant 0 : index
  // CHECK: stream.tensor.load
  %0 = stream.tensor.load %arg0[%arg3] : tensor<?xi4>{%arg1} in !stream.resource<staging>{%arg2} -> i4
  return %0 : i4
}

// -----

// Checks that stream.tensor.store with sub-byte packing is not supported right now.

// CHECK-LABEL: @denseTensorStore
func.func @denseTensorStore(%arg0: !stream.resource<staging>, %arg1: index, %arg2: index, %arg3: i4) -> !stream.resource<staging> {
  %c0 = arith.constant 0 : index
  // CHECK: stream.tensor.store
  %0 = stream.tensor.store %arg3, %arg0[%c0] : i4 -> tensor<?xi4>{%arg1} in %arg0 as !stream.resource<staging>{%arg2}
  return %0 : !stream.resource<staging>
}

// -----

// CHECK-LABEL: @denseTensorSplatI2
func.func @denseTensorSplatI2(%arg0: i2, %arg1: index, %arg2: index) -> !stream.resource<*> {
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i8
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i8
  // CHECK-DAG: %[[C6:.+]] = arith.constant 6 : i8
  // CHECK: %[[PART:.+]] = arith.extui %arg0 : i2 to i8
  // CHECK: %[[SHL0:.+]] = arith.shli %[[PART]], %[[C2]] : i8
  // CHECK: %[[OR0:.+]] = arith.ori %[[PART]], %[[SHL0]] : i8
  // CHECK: %[[SHL1:.+]] = arith.shli %[[PART]], %[[C4]] : i8
  // CHECK: %[[OR1:.+]] = arith.ori %[[OR0]], %[[SHL1]] : i8
  // CHECK: %[[SH2:.+]] = arith.shli %[[PART]], %[[C6]] : i8
  // CHECK: %[[FULL:.+]] = arith.ori %[[OR1]], %[[SH2]] : i8
  // CHECK: %[[SPLAT:.+]] = stream.async.splat %[[FULL]] : i8 -> !stream.resource<*>{%arg2}
  %0 = stream.tensor.splat %arg0 : i2 -> tensor<?x1x16xi2>{%arg1} in !stream.resource<*>{%arg2}
  // CHECK: return %[[SPLAT]] : !stream.resource<*>
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorFillI4
func.func @denseTensorFillI4(%arg0: i4, %arg1: !stream.resource<*>, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index) -> !stream.resource<*> {
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i8
  // CHECK: %[[PART:.+]] = arith.extui %arg0 : i4 to i8
  // CHECK: %[[SHL:.+]] = arith.shli %[[PART]], %[[C4]] : i8
  // CHECK: %[[FULL:.+]] = arith.ori %[[PART]], %[[SHL]] : i8
  // CHECK: %[[OFFSET:.+]] = affine.apply affine_map<()[s0, s1] -> ((s0 + s1 * 16) ceildiv 2)>()[%arg5, %arg4]
  // CHECK: %[[LENGTH:.+]] = affine.apply affine_map<()[s0, s1] -> ((s0 + s1 * 16) ceildiv 2)>()[%arg7, %arg6]
  // CHECK: %[[END:.+]] = affine.apply affine_map<()[s0, s1, s2, s3] -> ((s0 + s1 * 16) ceildiv 2 + (s2 + s3 * 16) ceildiv 2)>()[%arg5, %arg4, %arg7, %arg6]
  // CHECK: %[[FILL:.+]] = stream.async.fill %[[FULL]], %arg1[%[[OFFSET]] to %[[END]] for %[[LENGTH]]] : i8 -> %arg1 as !stream.resource<*>{%arg3}
  %0 = stream.tensor.fill %arg0, %arg1[%arg4, %arg5 for %arg6, %arg7] : i4 -> tensor<?x16xi4>{%arg2} in %arg1 as !stream.resource<*>{%arg3}
  // CHECK: return %[[FILL]]
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorSliceI2
func.func @denseTensorSliceI2(%arg0: !stream.resource<*>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> !stream.resource<*> {
  %c2 = arith.constant 2 : index
  // CHECK: %[[OFFSET:.+]] = affine.apply affine_map<()[s0, s1] -> ((s0 + s1 * 8) ceildiv 4)>()[%arg6, %arg5]
  // CHECK: %[[LENGTH:.+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 + (s1 + s2 * 8) ceildiv 4)>()[%arg4, %arg6, %arg5]
  // CHECK: %[[SLICE:.+]] = stream.async.slice %arg0[%[[OFFSET]] to %[[LENGTH]]] : !stream.resource<*>{%arg2} -> !stream.resource<*>{%arg4}
  %0 = stream.tensor.slice %arg0[%arg5, %arg6 for %arg3, %c2] : tensor<?x8xi2>{%arg1} in !stream.resource<*>{%arg2} -> tensor<?x2xi2>{%arg3} in !stream.resource<*>{%arg4}
  // CHECK: return %[[SLICE]] : !stream.resource<*>
  return %0 : !stream.resource<*>
}

// -----

// Ensures that a non-power-of-two type (i3) slice is expanded to a full byte
// because we don't currently do unaligned sub-byte packing.

// CHECK-LABEL: @denseTensorSliceI3
func.func @denseTensorSliceI3(%arg0: !stream.resource<*>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> !stream.resource<*> {
  %c2 = arith.constant 2 : index
  // CHECK: %[[OFFSET:.+]] = affine.apply affine_map<()[s0, s1] -> ((s0 + s1 * 8) ceildiv 2)>()[%arg6, %arg5]
  // CHECK: %[[LENGTH:.+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 + (s1 + s2 * 8) ceildiv 2)>()[%arg4, %arg6, %arg5]
  // CHECK: %[[SLICE:.+]] = stream.async.slice %arg0[%[[OFFSET]] to %[[LENGTH]]] : !stream.resource<*>{%arg2} -> !stream.resource<*>{%arg4}
  %0 = stream.tensor.slice %arg0[%arg5, %arg6 for %arg3, %c2] : tensor<?x8xi3>{%arg1} in !stream.resource<*>{%arg2} -> tensor<?x2xi3>{%arg3} in !stream.resource<*>{%arg4}
  // CHECK: return %[[SLICE]] : !stream.resource<*>
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorUpdateI3
func.func @denseTensorUpdateI3(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.resource<*>, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> !stream.resource<*> {
  // CHECK: %[[OFFSET:.+]] = affine.apply affine_map<()[s0, s1] -> ((s0 + s1 * 4) ceildiv 2)>()[%arg6, %arg5]
  // CHECK: %[[LENGTH:.+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 + (s1 + s2 * 4) ceildiv 2)>()[%arg1, %arg6, %arg5]
  // CHECK: %[[UPDATE:.+]] = stream.async.update %arg0, %arg2[%[[OFFSET]] to %[[LENGTH]]] : !stream.resource<*>{%arg1} -> %arg2 as !stream.resource<*>{%arg4}
  %0 = stream.tensor.update %arg0, %arg2[%arg5, %arg6] : tensor<8x4xi3> in !stream.resource<*>{%arg1} -> tensor<?x4xi3>{%arg3} in %arg2 as !stream.resource<*>{%arg4}
  // CHECK: return %[[UPDATE]] : !stream.resource<*>
  return %0 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @denseTensorUpdateI4
func.func @denseTensorUpdateI4(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.resource<*>, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> !stream.resource<*> {
  // CHECK: %[[OFFSET:.+]] = affine.apply affine_map<()[s0, s1] -> ((s0 + s1 * 4) ceildiv 2)>()[%arg6, %arg5]
  // CHECK: %[[LENGTH:.+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 + (s1 + s2 * 4) ceildiv 2)>()[%arg1, %arg6, %arg5]
  // CHECK: %[[UPDATE:.+]] = stream.async.update %arg0, %arg2[%[[OFFSET]] to %[[LENGTH]]] : !stream.resource<*>{%arg1} -> %arg2 as !stream.resource<*>{%arg4}
  %0 = stream.tensor.update %arg0, %arg2[%arg5, %arg6] : tensor<8x4xi4> in !stream.resource<*>{%arg1} -> tensor<?x4xi4>{%arg3} in %arg2 as !stream.resource<*>{%arg4}
  // CHECK: return %[[UPDATE]] : !stream.resource<*>
  return %0 : !stream.resource<*>
}
