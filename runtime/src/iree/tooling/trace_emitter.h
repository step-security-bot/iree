// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_TRACE_EMITTER_H_
#define IREE_TOOLING_TRACE_EMITTER_H_

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_trace_emitter_t
//===----------------------------------------------------------------------===//
//
// ██╗░░░██╗███╗░░██╗░██████╗████████╗░█████╗░██████╗░██╗░░░░░███████╗
// ██║░░░██║████╗░██║██╔════╝╚══██╔══╝██╔══██╗██╔══██╗██║░░░░░██╔════╝
// ██║░░░██║██╔██╗██║╚█████╗░░░░██║░░░███████║██████╦╝██║░░░░░█████╗░░
// ██║░░░██║██║╚████║░╚═══██╗░░░██║░░░██╔══██║██╔══██╗██║░░░░░██╔══╝░░
// ╚██████╔╝██║░╚███║██████╔╝░░░██║░░░██║░░██║██████╦╝███████╗███████╗
// ░╚═════╝░╚═╝░░╚══╝╚═════╝░░░░╚═╝░░░╚═╝░░╚═╝╚═════╝░╚══════╝╚══════╝
//
// Low-level streaming emitter for YAML trace files. These files contain a full
// sequence of VM context creation, module loading, and function calls within
// those modules as well as some utility operations to simulate what hosting
// applications may do with their buffer views.
//
// The trace file consists of zero or more YAML documents separated by `---`.
// Each document represents an event of a certain type with its own structure.
// See iree-run-trace --help for more information on the format.
//
// This emitter is intended to directly model the trace format and as such does
// not provide stateful tracking of runtime resources. Binding layers that
// integrate the emitter can perform the tracking in whatever way is natural
// for them and may handle automatic input/output/blackboard assignment.
//
// Related tools:
//   iree-run-trace:
//     Like iree-run-module but takes one or more trace files as positional
//     arguments. --input= flags are available in the trace via the `!input.*`
//     macros and --output= flags are available via `!output.*` macros.
//     Rudimentary testing is available via --expected_output= flags.
//     HAL --device= flags are available for overriding trace device selection.
//   iree-benchmark-trace:
//     Like iree-benchmark-module but takes one or more trace files as
//     positional arguments and benchmarks each of them. --input= and --device=
//     flags are available as with iree-run-trace.
//
// Future changes to the API/format:
// - encode the context used with relevant events
// - ways to specify device sets for the HAL module
// - ways to specify which device buffers get allocated on
// - ways to specify fences for async execution
// - some basic copy/fill transfer ops for intra-/inter-device movement
// - more data types (!vm.buffer, etc)
// - loops?
//
// Example usage (all these should be status checked!):
//   // Open emitter file for overwriting.
//   iree_trace_emitter_t* emitter = NULL;
//   iree_trace_emitter_open(..., "trace.yml", /*append=*/false, ..., &emitter);
//
//   // Create a new context.
//   iree_trace_emit_context_load(emitter, context);
//
//   // Load the HAL module and a VMFB from a trace file relative path.
//   iree_trace_emit_module_load_builtin(emitter, context, IREE_SV("hal"));
//   iree_trace_emit_module_load_bytecode_from_path(
//       emitter, context, IREE_SV("my_module"), IREE_SV("../my_module.vmfb"));
//
//   // Load a numpy array into blackboard[0].
//   iree_trace_emit_numpy_load_array(
//       emitter, IREE_SV("../input.npy"),
//       iree_trace_emitter_target_blackboard_set(0));
//
//   // Make a call to a function:
//   //   outputs.push(fn(blackboard[0], constant_buffer_view))
//   iree_trace_emitter_source_t args[2] = {
//     iree_trace_emitter_source_blackboard_get(0),
//     iree_trace_emitter_source_ref(buffer_view_ref),
//   };
//   iree_trace_emitter_source_t results[1] = {
//     iree_trace_emitter_target_output_push(),
//   };
//   iree_trace_emit_call(
//       emitter, context, function,
//       IREE_ARRAYSIZE(args), args, IREE_ARRAYSIZE(results), results);
//
//   // Incrementally build another call to the same function.
//   iree_trace_emitter_event_t event = 0;
//   iree_trace_emit_call_begin(emitter, context, function, &event);
//   iree_trace_emit_call_append_argument(
//       emitter, event, iree_trace_emitter_source_blackboard_get(0));
//   iree_trace_emit_call_append_argument(
//       emitter, event, iree_trace_emitter_source_ref(buffer_view_ref));
//   iree_trace_emit_call_append_result(
//       emitter, event, iree_trace_emitter_target_output_push());
//   iree_trace_emit_call_end(emitter, event);
//
//   // Flush to ensure everything ends up in the file.
//   iree_trace_emitter_flush(emitter);
//
//   // Free the emitter and close the file.
//   iree_trace_emitter_free(emitter);

// Streaming YAML trace file emitter.
typedef struct iree_trace_emitter_t iree_trace_emitter_t;

enum iree_trace_emitter_flag_bits_t {
  IREE_TRACE_EMITTER_FLAG_NONE = 0u,
};
typedef uint32_t iree_trace_emitter_flags_t;

// Opaque event identifier. Only valid for the active event and closed when
// the emitter is flushed or another event is opened.
typedef uintptr_t iree_trace_emitter_event_t;

// Input/output and blackboard slot ordinal.
typedef uint32_t iree_trace_emitter_ordinal_t;

// Allocates a new trace emitter streaming to the given |file|.
// If |owns_file| is true the file will be closed when the emitter is freed.
// |out_emitter| must be freed by the caller with iree_trace_emitter_free.
// Emitted results must be flushed with iree_trace_emitter_flush.
iree_status_t iree_trace_emitter_allocate(iree_trace_emitter_flags_t flags,
                                          FILE* file, bool owns_file,
                                          iree_allocator_t host_allocator,
                                          iree_trace_emitter_t** out_emitter);

// Opens a new trace emitter streaming to the given file |path|.
// If |append| is true then the file will be opened for append.
// |out_emitter| must be freed by the caller with iree_trace_emitter_free.
// Emitted results must be flushed with iree_trace_emitter_flush.
iree_status_t iree_trace_emitter_open(iree_trace_emitter_flags_t flags,
                                      const char* path, bool append,
                                      iree_allocator_t host_allocator,
                                      iree_trace_emitter_t** out_emitter);

// Frees an allocated trace emitter **without flushing**.
// Callers must use iree_trace_emitter_flush to ensure all emitted data has been
// flushed to the underlying file.
void iree_trace_emitter_free(iree_trace_emitter_t* emitter);

// Flushes a trace emitter to the output stream.
iree_status_t iree_trace_emitter_flush(iree_trace_emitter_t* emitter);

// Acquires a new unique blackboard slot and returns its ordinal.
// Reset to 0 with iree_trace_emit_blackboard_clear.
iree_trace_emitter_ordinal_t iree_trace_emitter_acquire_blackboard_slot(
    iree_trace_emitter_t* emitter);

//===----------------------------------------------------------------------===//
// iree_trace_emitter_source_t
//===----------------------------------------------------------------------===//

enum iree_trace_emitter_source_type_e {
  // Undefined value; ignored during emission.
  IREE_TRACE_BUILDER_SOURCE_TYPE_UNDEFINED = 0u,
  // Scalar value, !hal.buffer, !hal.buffer_view, etc embedded in the trace.
  IREE_TRACE_BUILDER_SOURCE_TYPE_CONSTANT,
  // !input.get <ordinal>
  IREE_TRACE_BUILDER_SOURCE_TYPE_INPUT_GET,
  // !input.take <ordinal>
  IREE_TRACE_BUILDER_SOURCE_TYPE_INPUT_TAKE,
  // !output.get <ordinal>
  IREE_TRACE_BUILDER_SOURCE_TYPE_OUTPUT_GET,
  // !output.get <ordinal>
  IREE_TRACE_BUILDER_SOURCE_TYPE_OUTPUT_TAKE,
  // !blackboard.get <ordinal>
  IREE_TRACE_BUILDER_SOURCE_TYPE_BLACKBOARD_GET,
  // !blackboard.take <ordinal>
  IREE_TRACE_BUILDER_SOURCE_TYPE_BLACKBOARD_TAKE,
};
typedef uint32_t iree_trace_emitter_source_type_t;

// A source value specifying either the constant value or where to obtain the
// value from in the pipeline.
typedef struct iree_trace_emitter_source_t {
  iree_trace_emitter_source_type_t type;
  iree_vm_variant_t variant;
  iree_trace_emitter_ordinal_t ordinal;
} iree_trace_emitter_source_t;

// Returns an undefined value that can be ignored during emission.
static inline iree_trace_emitter_source_t iree_trace_emitter_source_undefined(
    void) {
  iree_trace_emitter_source_t source = {
      IREE_TRACE_BUILDER_SOURCE_TYPE_UNDEFINED,
  };
  return source;
}

// Returns true if |source| is undefined.
static inline bool iree_trace_emitter_source_is_undefined(
    iree_trace_emitter_source_t source) {
  return source.type == IREE_TRACE_BUILDER_SOURCE_TYPE_UNDEFINED;
}

// Specifies a source scalar value, !hal.buffer, !hal.buffer_view, etc
// embedded in the trace. The |variant| contents are written to the stream and
// need not live beyond the emission call.
static inline iree_trace_emitter_source_t iree_trace_emitter_source_constant(
    iree_vm_variant_t variant) {
  iree_trace_emitter_source_t source;
  source.type = IREE_TRACE_BUILDER_SOURCE_TYPE_CONSTANT;
  source.variant = variant;
  source.ordinal = 0;
  return source;
}

// Specifies a source VM reference object embedded in the trace.
static inline iree_trace_emitter_source_t iree_trace_emitter_source_ref(
    iree_vm_ref_t ref) {
  iree_trace_emitter_source_t source;
  source.type = IREE_TRACE_BUILDER_SOURCE_TYPE_CONSTANT;
  source.variant = iree_vm_make_variant_ref_assign(ref);
  source.ordinal = 0;
  return source;
}

// Specifies the value should be retrieved from the pipeline input |ordinal|.
// The value will remain in the input list for future access.
static inline iree_trace_emitter_source_t iree_trace_emitter_source_input_get(
    iree_trace_emitter_ordinal_t ordinal) {
  iree_trace_emitter_source_t source;
  source.type = IREE_TRACE_BUILDER_SOURCE_TYPE_INPUT_GET;
  source.variant = iree_vm_variant_empty();
  source.ordinal = ordinal;
  return source;
}

// Specifies the value should be taken from the pipeline input |ordinal|.
// The value will be reset in the input list and consumed by the event.
static inline iree_trace_emitter_source_t iree_trace_emitter_source_input_take(
    iree_trace_emitter_ordinal_t ordinal) {
  iree_trace_emitter_source_t source;
  source.type = IREE_TRACE_BUILDER_SOURCE_TYPE_INPUT_TAKE;
  source.variant = iree_vm_variant_empty();
  source.ordinal = ordinal;
  return source;
}

// Specifies the value should be retrieved from the pipeline output |ordinal|.
// The value will remain in the output list for future access.
static inline iree_trace_emitter_source_t iree_trace_emitter_source_output_get(
    iree_trace_emitter_ordinal_t ordinal) {
  iree_trace_emitter_source_t source;
  source.type = IREE_TRACE_BUILDER_SOURCE_TYPE_OUTPUT_GET;
  source.variant = iree_vm_variant_empty();
  source.ordinal = ordinal;
  return source;
}

// Specifies the value should be taken from the pipeline output |ordinal|.
// The value will be reset in the output list and consumed by the event.
static inline iree_trace_emitter_source_t iree_trace_emitter_source_output_take(
    iree_trace_emitter_ordinal_t ordinal) {
  iree_trace_emitter_source_t source;
  source.type = IREE_TRACE_BUILDER_SOURCE_TYPE_OUTPUT_TAKE;
  source.variant = iree_vm_variant_empty();
  source.ordinal = ordinal;
  return source;
}

// Specifies the value should be retrieved from the blackboard slot |ordinal|.
// The value will remain in the blackboard list for future access.
static inline iree_trace_emitter_source_t
iree_trace_emitter_source_blackboard_get(iree_trace_emitter_ordinal_t ordinal) {
  iree_trace_emitter_source_t source;
  source.type = IREE_TRACE_BUILDER_SOURCE_TYPE_BLACKBOARD_GET;
  source.variant = iree_vm_variant_empty();
  source.ordinal = ordinal;
  return source;
}

// Specifies the value should be taken from the blackboard slot |ordinal|.
// The value will be reset in the blackboard list and consumed by the event.
static inline iree_trace_emitter_source_t
iree_trace_emitter_source_blackboard_take(
    iree_trace_emitter_ordinal_t ordinal) {
  iree_trace_emitter_source_t source;
  source.type = IREE_TRACE_BUILDER_SOURCE_TYPE_BLACKBOARD_TAKE;
  source.variant = iree_vm_variant_empty();
  source.ordinal = ordinal;
  return source;
}

//===----------------------------------------------------------------------===//
// iree_trace_emitter_target_t
//===----------------------------------------------------------------------===//

enum iree_trace_emitter_target_type_e {
  // Undefined value; ignored during emission.
  IREE_TRACE_BUILDER_TARGET_TYPE_UNDEFINED = 0u,
  // Scalar value, !hal.buffer, !hal.buffer_view, etc embedded in the trace.
  IREE_TRACE_BUILDER_TARGET_TYPE_CONSTANT,
  // !output.set <ordinal>
  IREE_TRACE_BUILDER_TARGET_TYPE_OUTPUT_SET,
  // !output.push
  IREE_TRACE_BUILDER_TARGET_TYPE_OUTPUT_PUSH,
  // !blackboard.set <ordinal>
  IREE_TRACE_BUILDER_TARGET_TYPE_BLACKBOARD_SET,
  // !blackboard.push
  IREE_TRACE_BUILDER_TARGET_TYPE_BLACKBOARD_PUSH,
};
typedef uint32_t iree_trace_emitter_target_type_t;

// A target value specifying either the expected constant value or where to
// route the value in the pipeline.
typedef struct iree_trace_emitter_target_t {
  iree_trace_emitter_target_type_t type;
  iree_vm_variant_t variant;
  iree_trace_emitter_ordinal_t ordinal;
} iree_trace_emitter_target_t;

// Returns an undefined value that can be ignored during emission.
static inline iree_trace_emitter_target_t iree_trace_emitter_target_undefined(
    void) {
  iree_trace_emitter_target_t target = {
      IREE_TRACE_BUILDER_TARGET_TYPE_UNDEFINED,
  };
  return target;
}

// Returns true if |target| is undefined.
static inline bool iree_trace_emitter_target_is_undefined(
    iree_trace_emitter_target_t target) {
  return target.type == IREE_TRACE_BUILDER_TARGET_TYPE_UNDEFINED;
}

// Specifies a target expected scalar value, !hal.buffer, !hal.buffer_view, etc
// embedded in the trace. These may be used for output verification and diffing
// tools but are generally not required. The |variant| contents are written to
// the stream and need not live beyond the emission call.
static inline iree_trace_emitter_target_t iree_trace_emitter_target_constant(
    iree_vm_variant_t variant) {
  iree_trace_emitter_target_t target;
  target.type = IREE_TRACE_BUILDER_TARGET_TYPE_CONSTANT;
  target.variant = variant;
  target.ordinal = 0;
  return target;
}

// Specifies a target VM reference object embedded in the trace.
static inline iree_trace_emitter_target_t iree_trace_emitter_target_ref(
    iree_vm_ref_t ref) {
  iree_trace_emitter_target_t target;
  target.type = IREE_TRACE_BUILDER_TARGET_TYPE_CONSTANT;
  target.variant = iree_vm_make_variant_ref_assign(ref);
  target.ordinal = 0;
  return target;
}

// Specifies the value should be set as pipeline output |ordinal|.
// The value will be retained by the output until it is taken or reset.
// The output list capacity will be grown if needed.
static inline iree_trace_emitter_target_t iree_trace_emitter_target_output_set(
    iree_trace_emitter_ordinal_t ordinal) {
  iree_trace_emitter_target_t target;
  target.type = IREE_TRACE_BUILDER_TARGET_TYPE_OUTPUT_SET;
  target.variant = iree_vm_variant_empty();
  target.ordinal = ordinal;
  return target;
}

// Specifies the value should be pushed onto the target pipeline output list.
// The value will be retained by the output until it is taken or reset.
static inline iree_trace_emitter_target_t iree_trace_emitter_target_output_push(
    void) {
  iree_trace_emitter_target_t target;
  target.type = IREE_TRACE_BUILDER_TARGET_TYPE_OUTPUT_PUSH;
  target.variant = iree_vm_variant_empty();
  target.ordinal = 0;
  return target;
}

// Specifies the value should be set as blackboard slot |ordinal|.
// The value will be retained by the blackboard until it is taken or reset.
// The blackboard slot capacity will be grown if needed.
static inline iree_trace_emitter_target_t
iree_trace_emitter_target_blackboard_set(iree_trace_emitter_ordinal_t ordinal) {
  iree_trace_emitter_target_t target;
  target.type = IREE_TRACE_BUILDER_TARGET_TYPE_BLACKBOARD_SET;
  target.variant = iree_vm_variant_empty();
  target.ordinal = ordinal;
  return target;
}

// Specifies the value should be pushed onto the target blackboard list.
// The value will be retained by the blackboard until it is taken or reset.
static inline iree_trace_emitter_target_t
iree_trace_emitter_target_blackboard_push(void) {
  iree_trace_emitter_target_t target;
  target.type = IREE_TRACE_BUILDER_TARGET_TYPE_BLACKBOARD_PUSH;
  target.variant = iree_vm_variant_empty();
  target.ordinal = 0;
  return target;
}

//===----------------------------------------------------------------------===//
// iree_trace_emitter_t event emission
//===----------------------------------------------------------------------===//

//----------------------------------------------------------------------------//
// `context_load`
//----------------------------------------------------------------------------//

// Emits a `context_load` event indicating a new context has been created.
// The new context will have no modules registered and no state.
iree_status_t iree_trace_emit_context_load(iree_trace_emitter_t* emitter,
                                           iree_vm_context_t* context);

//----------------------------------------------------------------------------//
// `module_load`
//----------------------------------------------------------------------------//

// Emits a `module_load` event for a built-in named module such as the HAL.
iree_status_t iree_trace_emit_module_load_builtin(iree_trace_emitter_t* emitter,
                                                  iree_vm_context_t* context,
                                                  iree_string_view_t name);

// TODO(benvanik): iree_trace_emit_module_load_hal with device fields.

// Emits a `module_load` event for a bytecode module read from stdin at runtime.
// Only one module may be read from stdin per trace.
iree_status_t iree_trace_emit_module_load_bytecode_from_stdin(
    iree_trace_emitter_t* emitter, iree_vm_context_t* context,
    iree_string_view_t name);

// Emits a `module_load` event for a bytecode module with the given file path.
// Prefer paths relative to the trace file.
iree_status_t iree_trace_emit_module_load_bytecode_from_path(
    iree_trace_emitter_t* emitter, iree_vm_context_t* context,
    iree_string_view_t name, iree_string_view_t path);

// Emits a `module_load` event for a bytecode module embedded in the trace.
// |contents| will be immediately written to the stream and need not remain
// live past the call.
iree_status_t iree_trace_emit_module_load_bytecode_from_data(
    iree_trace_emitter_t* emitter, iree_vm_context_t* context,
    iree_string_view_t name, iree_const_byte_span_t contents);

//----------------------------------------------------------------------------//
// `blackboard_clear`
//----------------------------------------------------------------------------//

// Emits a `blackboard_clear` event that resets the blackboard to 0 slots.
// All previously acquired blackboard slots are invalidated and new ones must
// be obtained with iree_trace_emitter_acquire_blackboard_slot.
iree_status_t iree_trace_emit_blackboard_clear(iree_trace_emitter_t* emitter);

//----------------------------------------------------------------------------//
// `assign`
//----------------------------------------------------------------------------//

// Emits a single entry `assign` event from |from| to |to|.
iree_status_t iree_trace_emit_assign(iree_trace_emitter_t* emitter,
                                     iree_trace_emitter_source_t from,
                                     iree_trace_emitter_target_t to);

// Begins emitting an `assign` event. One or more from values can be specified
// with iree_trace_emit_assign_append_from followed by one or more to values
// with iree_trace_emit_assign_append_to. When done the event is ended with
// iree_trace_emit_assign_end.
//
// Example:
//   iree_trace_emit_assign_begin(emitter, &event);
//   iree_trace_emit_assign_append_from(emitter, event, source_0);
//   iree_trace_emit_assign_append_from(emitter, event, source_1);
//   iree_trace_emit_assign_append_to(emitter, event, target_0);
//   iree_trace_emit_assign_append_to(emitter, event, target_1);
//   iree_trace_emit_assign_end(emitter, event);
iree_status_t iree_trace_emit_assign_begin(
    iree_trace_emitter_t* emitter, iree_trace_emitter_event_t* out_event);
// Emits an entry in the `assign` event `from` sequence.
iree_status_t iree_trace_emit_assign_append_from(
    iree_trace_emitter_t* emitter, iree_trace_emitter_event_t event,
    iree_trace_emitter_source_t source);
// Emits an entry in the `assign` event `to` sequence.
iree_status_t iree_trace_emit_assign_append_to(
    iree_trace_emitter_t* emitter, iree_trace_emitter_event_t event,
    iree_trace_emitter_target_t target);
// Ends emitting an `assign` event.
iree_status_t iree_trace_emit_assign_end(iree_trace_emitter_t* emitter,
                                         iree_trace_emitter_event_t event);

//----------------------------------------------------------------------------//
// `numpy_load`
//----------------------------------------------------------------------------//

// Emits a `numpy_load` event of the given .npy |path| to the given |array|.
iree_status_t iree_trace_emit_numpy_load_array(
    iree_trace_emitter_t* emitter, iree_string_view_t path,
    const iree_trace_emitter_target_t array);

// Emits a `numpy_load` event of the given .npy |path| to the given |arrays|.
iree_status_t iree_trace_emit_numpy_load_arrays(
    iree_trace_emitter_t* emitter, iree_string_view_t path,
    iree_host_size_t array_count, const iree_trace_emitter_target_t* arrays);

// Begins emitting a `numpy_load` event from the given .npy |path|.
// One or more target array values can be specified with
// iree_trace_emit_numpy_load_append_array. When done the event is ended with
// iree_trace_emit_numpy_load_end.
//
// Example:
//   iree_trace_emit_numpy_load_begin(emitter, npy_path, &event);
//   iree_trace_emit_numpy_load_append_array(emitter, event, target_0);
//   iree_trace_emit_numpy_load_append_array(emitter, event, target_1);
//   iree_trace_emit_numpy_load_end(emitter, event);
iree_status_t iree_trace_emit_numpy_load_begin(
    iree_trace_emitter_t* emitter, iree_string_view_t path,
    iree_trace_emitter_event_t* out_event);
// Emits an entry in the `numpy_load` event `arrays` sequence.
iree_status_t iree_trace_emit_numpy_load_append_array(
    iree_trace_emitter_t* emitter, iree_trace_emitter_event_t event,
    iree_trace_emitter_target_t target);
// Ends emitting a `numpy_load` event.
iree_status_t iree_trace_emit_numpy_load_end(iree_trace_emitter_t* emitter,
                                             iree_trace_emitter_event_t event);

//----------------------------------------------------------------------------//
// `numpy_save`
//----------------------------------------------------------------------------//

// Emits a `numpy_save` event of the given |array| to the given .npy |path|.
// If |append| is true the array will be concatenated with existing ones in the
// file.
iree_status_t iree_trace_emit_numpy_save_array(
    iree_trace_emitter_t* emitter, iree_string_view_t path, bool append,
    iree_trace_emitter_source_t array);

// Emits a `numpy_save` event of the given |arrays| to the given .npy |path|.
// If |append| is true the arrays will be concatenated with existing ones in the
// file.
iree_status_t iree_trace_emit_numpy_save_arrays(
    iree_trace_emitter_t* emitter, iree_string_view_t path, bool append,
    iree_host_size_t array_count, const iree_trace_emitter_source_t* arrays);

// Begins emitting a `numpy_save` event to the given .npy |path|.
// If |append| is true the arrays will be concatenated with existing ones in the
// file. One or more source array values can be specified with
// iree_trace_emit_numpy_save_append_array. When done the event is ended with
// iree_trace_emit_numpy_save_end.
//
// Example:
//   iree_trace_emit_numpy_save_begin(emitter, npy_path, false, &event);
//   iree_trace_emit_numpy_save_append_array(emitter, event, source_0);
//   iree_trace_emit_numpy_save_append_array(emitter, event, source_1);
//   iree_trace_emit_numpy_save_end(emitter, event);
iree_status_t iree_trace_emit_numpy_save_begin(
    iree_trace_emitter_t* emitter, iree_string_view_t path, bool append,
    iree_trace_emitter_event_t* out_event);
// Emits an entry in the `numpy_save` event `arrays` sequence.
iree_status_t iree_trace_emit_numpy_save_append_array(
    iree_trace_emitter_t* emitter, iree_trace_emitter_event_t event,
    iree_trace_emitter_source_t source);
// Ends emitting a `numpy_save` event.
iree_status_t iree_trace_emit_numpy_save_end(iree_trace_emitter_t* emitter,
                                             iree_trace_emitter_event_t event);

//----------------------------------------------------------------------------//
// `call`
//----------------------------------------------------------------------------//

// Emits a `call` event to the given |function| with the given args/results.
iree_status_t iree_trace_emit_call(iree_trace_emitter_t* emitter,
                                   iree_vm_context_t* context,
                                   iree_vm_function_t function,
                                   iree_host_size_t arg_count,
                                   const iree_trace_emitter_source_t* args,
                                   iree_host_size_t result_count,
                                   const iree_trace_emitter_target_t* results);

// Begins emitting a `call` event to the given |function|.
// Zero or more source argument values can be specified with
// iree_trace_emit_call_append_argument followed by zero or more target result
// values specified with iree_trace_emit_call_append_result. When done the event
// is ended with iree_trace_emit_call_end.
//
// Example:
//   iree_trace_emit_call_begin(emitter, context, function, &event);
//   iree_trace_emit_call_append_argument(emitter, event, source_0);
//   iree_trace_emit_call_append_argument(emitter, event, source_1);
//   iree_trace_emit_call_append_result(emitter, event, target_0);
//   iree_trace_emit_call_end(emitter, event);
iree_status_t iree_trace_emit_call_begin(iree_trace_emitter_t* emitter,
                                         iree_vm_context_t* context,
                                         iree_vm_function_t function,
                                         iree_trace_emitter_event_t* out_event);
// Emits an entry in the `call` event `args` sequence.
iree_status_t iree_trace_emit_call_append_argument(
    iree_trace_emitter_t* emitter, iree_trace_emitter_event_t event,
    iree_trace_emitter_source_t source);
// Emits an entry in the `call` event `results` sequence.
iree_status_t iree_trace_emit_call_append_result(
    iree_trace_emitter_t* emitter, iree_trace_emitter_event_t event,
    iree_trace_emitter_target_t target);
// Ends emitting a `call` event.
iree_status_t iree_trace_emit_call_end(iree_trace_emitter_t* emitter,
                                       iree_trace_emitter_event_t event);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_TRACE_EMITTER_H_
