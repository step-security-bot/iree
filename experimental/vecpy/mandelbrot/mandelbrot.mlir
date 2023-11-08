// Compile and run: 
// mlir-opt -one-shot-bufferize="bufferize-function-boundaries" -convert-scf-to-cf -convert-index-to-llvm -test-lower-to-llvm mandelbrot.mlir |\
// mlir-cpu-runner --entry-point-result=void -e main --shared-libs=lib/libmlir_c_runner_utils.dylib,lib/libmlir_runner_utils.dylib


module {

// Main entry point
func.func @main() -> () {
  // TODO: These should be parameters. ATM just match the Python inputs.
  %x0 = arith.constant -1.0 : f32
  %y0 = arith.constant -1.0 : f32
  %x1 = arith.constant 1.0 : f32
  %y1 = arith.constant 1.0 : f32

  %height = arith.constant 100.0 : f32
  %width = arith.constant 100.0 : f32
  // TODO: Should be a cast of the above
  %height_i = arith.constant 100 : index
  %width_i = arith.constant 100 : index

  %max_iters = arith.constant 1000 : i32

  // Compute the Mandelbrot set
  %res = func.call @mandelbrot(
    %x0, %y0,
    %x1, %y1,
    %width, %height,
    %max_iters) : (f32, f32, f32, f32, f32, f32, i32) -> (tensor<100x100xi32>)

  // Print the computed Mandelbrot set
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %j = %c0 to %height_i step %c1 {
    scf.for %i = %c0 to %width_i step %c1 {
      %val = tensor.extract %res[%i, %j] : tensor<100x100xi32>
      vector.print %val : i32 punctuation <comma>
    }
    vector.print punctuation <newline>
  }
  return
}

//   Computes the Mandelbrot recurrence equation:
//
//       z_{n+1} = z_{n}^2 + c,
//
//   where z_0 = c. Returns either max_iters or the number of iterations
//   requires to reach MAX_NORM_SQUARED.
func.func @mandel_recurrence(%z_re_init : f32, %z_im_init : f32, %max_iters : i32)  -> i32 {
  %c0  = arith.constant 0 : index
  %c0_i32  = arith.constant 0 : i32
  %c1  = arith.constant 1 : i32
  %c2  = arith.constant 2.0 : f32
  // Arbitrary value - update
  %MAX_NORM_SQUARED  = arith.constant 4.0 : f32

  %iters_res, %re_res, %im_res = scf.while (%iters = %c0_i32, %z_re = %z_re_init, %z_im = %z_im_init) : (i32, f32, f32) -> (i32, f32, f32) {
    // Compute the euclidean norm squared
    %pow_1 = arith.mulf %z_re, %z_re : f32
    %pow_2 = arith.mulf %z_im, %z_im : f32
    %euc_norm_squared = arith.addf %pow_1, %pow_2 : f32

    // Check whether the current value is within the bounds
    %cond_val = arith.cmpf olt, %euc_norm_squared, %MAX_NORM_SQUARED : f32

    // Check whether the number of iterations is within the bounds
    %cond_iter = arith.cmpi "slt", %iters, %max_iters : i32

    %cond = arith.andi %cond_iter, %cond_val : i1
    scf.condition(%cond) %iters, %z_re, %z_im : i32, f32, f32
  } do {
  ^bb0(%iters : i32, %z_re : f32, %z_im : f32):
    %iters_new = arith.addi %c1, %iters : i32

    // Compute the "real" part of the new value:
    //   z_re*z_re - z_im*z_im
    %new_re_1 = arith.mulf %z_re, %z_re : f32
    %new_re_2 = arith.mulf %z_im, %z_im : f32
    %new_re_3 = arith.subf %new_re_1, %new_re_2 : f32

    // Compute the "imaginary" part of the new value:
    //    2 * a * b
    %new_im_1 = arith.mulf %c2, %z_re : f32
    %new_im_2 = arith.mulf %new_im_1, %z_im : f32

    // Compute the new value:
    //    z_{n+1} = z_n^2 + c
    %new_re = arith.addf %new_re_3, %z_re_init : f32
    %new_im = arith.addf %new_im_2, %z_im_init : f32

    scf.yield %iters_new, %new_re, %new_im : i32, f32, f32
  }

  return %iters_res : i32
}


//   Computes the Mandelbrot set for a square plane between (x0, y0) and (x1,
//   y1). The number of points in X and Y axis are equal `width` and `height`,
//   respectively.
func.func @mandelbrot(%x0: f32, %y0: f32, %x1 : f32, %y1 : f32, %width : f32, %height : f32, %max_iters : i32) -> (tensor<100x100xi32> ) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index

    %dx_1 = arith.subf %x1, %x0 : f32
    %dx = arith.divf %dx_1, %width : f32

    %dy_1 = arith.subf %y1, %y0 : f32
    %dy = arith.divf %dy_1, %height : f32

    %output = tensor.empty() : tensor<100x100xi32>

    // TODO: Can we do better instead of just casting?
    %height_i32 = arith.fptosi %height : f32 to i32
    %width_i32 = arith.fptosi %width : f32 to i32
    %height_i = arith.index_cast %height_i32 : i32 to index
    %width_i = arith.index_cast %width_i32 : i32 to index

    // Iterate over all points on the input plane and compute the Mandelbrot
    // recurrence equation for each.
    scf.for %j = %c0 to %height_i step %c1 {
      scf.for %i = %c0 to %width_i step %c1 {
        %i_i32 = arith.index_cast %i : index to i32
        %j_i32 = arith.index_cast %j : index to i32
        %i_f32 = arith.uitofp %i_i32 : i32 to f32
        %j_f32 = arith.uitofp %j_i32 : i32 to f32

        %x_1 = arith.mulf %i_f32, %dx : f32
        %x = arith.addf %x0, %x_1 : f32
        %y_1 = arith.mulf %j_f32, %dy : f32
        %y = arith.addf %y0, %y_1 : f32

        %new_val = func.call @mandel_recurrence(%x, %y, %max_iters) : (f32, f32, i32) -> (i32)
        tensor.insert %new_val into %output[%i, %j] : tensor<100x100xi32>
      }
    }

    return %output : tensor<100x100xi32>
  }
}
