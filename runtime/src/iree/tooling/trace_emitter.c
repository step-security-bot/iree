// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/trace_emitter.h"

#include <varargs.h>

#include "iree/modules/hal/types.h"

//===----------------------------------------------------------------------===//
// iree_trace_emitter_t
//===----------------------------------------------------------------------===//

enum iree_trace_emitter_state_bits_t {
  IREE_TRACE_EMITTER_STATE_EMPTY = 0u,
  IREE_TRACE_EMITTER_STATE_OPEN = 1u << 0,
  IREE_TRACE_EMITTER_STATE_ASSIGN_BEGIN = 1u << 1,
  IREE_TRACE_EMITTER_STATE_ASSIGN_FROM = 1u << 2,
  IREE_TRACE_EMITTER_STATE_ASSIGN_TO = 1u << 3,
  IREE_TRACE_EMITTER_STATE_NUMPY_LOAD_BEGIN = 1u << 4,
  IREE_TRACE_EMITTER_STATE_NUMPY_LOAD_ARRAYS = 1u << 5,
  IREE_TRACE_EMITTER_STATE_NUMPY_SAVE_BEGIN = 1u << 6,
  IREE_TRACE_EMITTER_STATE_NUMPY_SAVE_ARRAYS = 1u << 7,
  IREE_TRACE_EMITTER_STATE_CALL_BEGIN = 1u << 8,
  IREE_TRACE_EMITTER_STATE_CALL_ARGUMENTS = 1u << 9,
  IREE_TRACE_EMITTER_STATE_CALL_RESULTS = 1u << 10,
};
typedef uint32_t iree_trace_emitter_state_t;

typedef struct iree_trace_emitter_t {
  iree_allocator_t host_allocator;
  iree_trace_emitter_flags_t flags;
  FILE* file;
  bool owns_file;
  iree_trace_emitter_state_t state;
  iree_trace_emitter_ordinal_t next_blackboard_slot;
} iree_trace_emitter_t;

iree_status_t iree_trace_emitter_allocate(iree_trace_emitter_flags_t flags,
                                          FILE* file, bool owns_file,
                                          iree_allocator_t host_allocator,
                                          iree_trace_emitter_t** out_emitter) {
  IREE_ASSERT_ARGUMENT(file);
  IREE_ASSERT_ARGUMENT(out_emitter);
  *out_emitter = NULL;
  iree_trace_emitter_t* emitter = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*emitter), (void**)&emitter);
  if (iree_status_is_ok(status)) {
    emitter->host_allocator = host_allocator;
    emitter->flags = flags;
    emitter->file = file;
    emitter->owns_file = owns_file;
    emitter->state = IREE_TRACE_EMITTER_STATE_EMPTY;
    emitter->next_blackboard_slot = 0;
    *out_emitter = emitter;
  } else {
    if (owns_file) fclose(file);
  }
  return status;
}

iree_status_t iree_trace_emitter_open(iree_trace_emitter_flags_t flags,
                                      const char* path, bool append,
                                      iree_allocator_t host_allocator,
                                      iree_trace_emitter_t** out_emitter) {
  IREE_ASSERT_ARGUMENT(path);
  IREE_ASSERT_ARGUMENT(out_emitter);
  *out_emitter = NULL;
  FILE* file = fopen(path, append ? "ab" : "wb");
  if (!file) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to open file '%s' for writing", path);
  }
  return iree_trace_emitter_allocate(flags, file, /*owns_file=*/true,
                                     host_allocator, out_emitter);
}

void iree_trace_emitter_free(iree_trace_emitter_t* emitter) {
  if (!emitter) return;
  iree_allocator_t host_allocator = emitter->host_allocator;
  if (emitter->owns_file) {
    fclose(emitter->file);
  }
  iree_allocator_free(host_allocator, emitter);
}

iree_status_t iree_trace_emitter_flush(iree_trace_emitter_t* emitter) {
  IREE_ASSERT_ARGUMENT(emitter);
  int ret = fflush(emitter->file);
  return ret != 0 ? iree_make_status(IREE_STATUS_DATA_LOSS,
                                     "failed to fflush emitter file")
                  : iree_ok_status();
}

iree_trace_emitter_ordinal_t iree_trace_emitter_acquire_blackboard_slot(
    iree_trace_emitter_t* emitter) {
  IREE_ASSERT_ARGUMENT(emitter);
  return emitter->next_blackboard_slot++;
}

// fprintf but with a status result.
static iree_status_t IREE_PRINTF_ATTRIBUTE(2, 3)
    iree_trace_emitter_printf(iree_trace_emitter_t* emitter, const char* format,
                              ...) {
  va_list varargs;
  va_start(varargs, format);

  iree_status_t status = iree_ok_status();
  int ret = vfprintf(emitter->file, format, varargs);
  if (ret < 0) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "failed fprintf'ing to the file");
  }

  va_end(varargs);
  return status;
}

static iree_status_t iree_trace_emitter_print_variant(
    iree_trace_emitter_t* emitter, int indent, iree_vm_variant_t variant);

static iree_status_t iree_trace_emitter_print_scalar(
    iree_trace_emitter_t* emitter, int indent, iree_vm_variant_t variant) {
  IREE_RETURN_IF_ERROR(iree_trace_emitter_printf(emitter, "type: value\n"));
  switch (variant.type.value_type) {
    case IREE_VM_VALUE_TYPE_I8:
      return iree_trace_emitter_printf(emitter, "%*si8: %" PRIi8, indent, "",
                                       variant.i8);
    case IREE_VM_VALUE_TYPE_I16:
      return iree_trace_emitter_printf(emitter, "%*si16: %" PRIi16, indent, "",
                                       variant.i16);
    case IREE_VM_VALUE_TYPE_I32:
      return iree_trace_emitter_printf(emitter, "%*si32: %" PRIi32, indent, "",
                                       variant.i32);
    case IREE_VM_VALUE_TYPE_I64:
      return iree_trace_emitter_printf(emitter, "%*si64: %" PRIi64, indent, "",
                                       variant.i64);
    case IREE_VM_VALUE_TYPE_F32:
      return iree_trace_emitter_printf(emitter, "%*sf32: %f", indent, "",
                                       variant.f32);
    case IREE_VM_VALUE_TYPE_F64:
      return iree_trace_emitter_printf(emitter, "%*sf64: %f", indent, "",
                                       variant.f64);
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported value type");
  }
}

static iree_status_t iree_trace_emitter_print_vm_list(
    iree_trace_emitter_t* emitter, int indent, iree_vm_list_t* list) {
  IREE_RETURN_IF_ERROR(iree_trace_emitter_printf(emitter, "type: vm.list\n"));
  IREE_RETURN_IF_ERROR(
      iree_trace_emitter_printf(emitter, "%*sitems:\n", indent, ""));
  for (iree_host_size_t i = 0; i < iree_vm_list_size(list); ++i) {
    iree_vm_variant_t variant = iree_vm_variant_empty();
    IREE_RETURN_IF_ERROR(iree_vm_list_get_variant_assign(list, i, &variant));
    IREE_RETURN_IF_ERROR(
        iree_trace_emitter_printf(emitter, "%*s- ", indent, ""));
    IREE_RETURN_IF_ERROR(
        iree_trace_emitter_print_variant(emitter, indent + 2, variant));
    IREE_RETURN_IF_ERROR(iree_trace_emitter_printf(emitter, "\n"));
  }
  return iree_ok_status();
}

static bool iree_trace_emitter_should_inline_buffer(
    iree_device_size_t buffer_size) {
  // TODO(benvanik): support base64 encoding; today we always emit inline.
  return true;
}

static iree_status_t iree_trace_emitter_print_hal_buffer(
    iree_trace_emitter_t* emitter, int indent, iree_hal_buffer_t* buffer) {
  // If the buffer is small then we can emit it inline in a nice form.
  // e.g. !hal.buffer 4xi8=[0 1 2 3]
  iree_device_size_t buffer_size = iree_hal_buffer_byte_length(buffer);
  if (iree_trace_emitter_should_inline_buffer(buffer_size)) {
    // TODO(benvanik): don't require the buffer view wrapping; this is just a
    // lazy way to avoid exposing more fprint methods that don't (one day)
    // require big allocations of the output contents.
    IREE_RETURN_IF_ERROR(iree_trace_emitter_printf(emitter, "!hal.buffer "));
    iree_hal_dim_t dims[1] = {0};
    iree_hal_buffer_view_t* buffer_view = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create(
        buffer, IREE_ARRAYSIZE(dims), dims, IREE_HAL_ELEMENT_TYPE_INT_8,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, emitter->host_allocator,
        &buffer_view));
    iree_status_t status = iree_hal_buffer_view_fprint(
        emitter->file, buffer_view, IREE_HOST_SIZE_MAX,
        emitter->host_allocator);
    iree_hal_buffer_view_release(buffer_view);
    return status;
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "base64 buffer encoding not yet implemented");
}

static iree_status_t iree_trace_emitter_print_hal_buffer_view(
    iree_trace_emitter_t* emitter, int indent,
    iree_hal_buffer_view_t* buffer_view) {
  // If the buffer is small then we can emit it inline in a nice form.
  // e.g. !hal.buffer_view 4xf32=[0.0 1.0 2.0 3.0]
  iree_device_size_t buffer_size =
      iree_hal_buffer_view_byte_length(buffer_view);
  if (iree_trace_emitter_should_inline_buffer(buffer_size)) {
    IREE_RETURN_IF_ERROR(
        iree_trace_emitter_printf(emitter, "!hal.buffer_view "));
    return iree_hal_buffer_view_fprint(emitter->file, buffer_view,
                                       IREE_HOST_SIZE_MAX,
                                       emitter->host_allocator);
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "base64 buffer view encoding not yet implemented");
}

static iree_status_t iree_trace_emitter_print_variant(
    iree_trace_emitter_t* emitter, int indent, iree_vm_variant_t variant) {
  if (iree_vm_variant_is_empty(variant)) {
    return iree_trace_emitter_printf(emitter, "type: null\n");
  } else if (iree_vm_variant_is_value(variant)) {
    return iree_trace_emitter_print_scalar(emitter, indent, variant);
  } else if (iree_vm_list_isa(variant.ref)) {
    return iree_trace_emitter_print_vm_list(emitter, indent,
                                            iree_vm_list_deref(variant.ref));
  } else if (iree_hal_buffer_isa(variant.ref)) {
    return iree_trace_emitter_print_hal_buffer(
        emitter, indent, iree_hal_buffer_deref(variant.ref));
  } else if (iree_hal_buffer_view_isa(variant.ref)) {
    return iree_trace_emitter_print_hal_buffer_view(
        emitter, indent, iree_hal_buffer_view_deref(variant.ref));
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "unsupported variant type");
}

static iree_status_t iree_trace_emitter_print_source(
    iree_trace_emitter_t* emitter, iree_trace_emitter_source_t source) {
  switch (source.type) {
    default:
    case IREE_TRACE_BUILDER_SOURCE_TYPE_UNDEFINED:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "cannot emit an undefined value");
    case IREE_TRACE_BUILDER_SOURCE_TYPE_CONSTANT:
      return iree_trace_emitter_print_variant(emitter, /*indent=*/0,
                                              source.variant);
    case IREE_TRACE_BUILDER_SOURCE_TYPE_INPUT_GET:
      return iree_trace_emitter_printf(emitter, "!input.get %u",
                                       source.ordinal);
    case IREE_TRACE_BUILDER_SOURCE_TYPE_INPUT_TAKE:
      return iree_trace_emitter_printf(emitter, "!input.take %u",
                                       source.ordinal);
    case IREE_TRACE_BUILDER_SOURCE_TYPE_OUTPUT_GET:
      return iree_trace_emitter_printf(emitter, "!output.get %u",
                                       source.ordinal);
    case IREE_TRACE_BUILDER_SOURCE_TYPE_OUTPUT_TAKE:
      return iree_trace_emitter_printf(emitter, "!output.take %u",
                                       source.ordinal);
    case IREE_TRACE_BUILDER_SOURCE_TYPE_BLACKBOARD_GET:
      return iree_trace_emitter_printf(emitter, "!blackboard.get %u",
                                       source.ordinal);
    case IREE_TRACE_BUILDER_SOURCE_TYPE_BLACKBOARD_TAKE:
      return iree_trace_emitter_printf(emitter, "!blackboard.take %u",
                                       source.ordinal);
  }
  return iree_ok_status();
}

static iree_status_t iree_trace_emitter_print_target(
    iree_trace_emitter_t* emitter, iree_trace_emitter_target_t target) {
  switch (target.type) {
    default:
    case IREE_TRACE_BUILDER_TARGET_TYPE_UNDEFINED:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "cannot emit an undefined value");
    case IREE_TRACE_BUILDER_TARGET_TYPE_CONSTANT:
      return iree_trace_emitter_print_variant(emitter, /*indent=*/0,
                                              target.variant);
    case IREE_TRACE_BUILDER_TARGET_TYPE_OUTPUT_SET:
      return iree_trace_emitter_printf(emitter, "!output.set %u",
                                       target.ordinal);
    case IREE_TRACE_BUILDER_TARGET_TYPE_OUTPUT_PUSH:
      return iree_trace_emitter_printf(emitter, "!output.push");
    case IREE_TRACE_BUILDER_TARGET_TYPE_BLACKBOARD_SET:
      return iree_trace_emitter_printf(emitter, "!blackboard.set %u",
                                       target.ordinal);
    case IREE_TRACE_BUILDER_TARGET_TYPE_BLACKBOARD_PUSH:
      return iree_trace_emitter_printf(emitter, "!blackboard.push");
  }
  return iree_ok_status();
}

// Emits a single |source| sequence item: `- <value>\n`.
static iree_status_t iree_trace_emitter_print_sequence_source(
    iree_trace_emitter_t* emitter, iree_trace_emitter_source_t source) {
  IREE_RETURN_IF_ERROR(iree_trace_emitter_printf(emitter, "- "));
  IREE_RETURN_IF_ERROR(iree_trace_emitter_print_source(emitter, source));
  IREE_RETURN_IF_ERROR(iree_trace_emitter_printf(emitter, "\n"));
  return iree_ok_status();
}

// Emits a single |target| sequence item: `- <value>\n`.
static iree_status_t iree_trace_emitter_print_sequence_target(
    iree_trace_emitter_t* emitter, iree_trace_emitter_target_t target) {
  IREE_RETURN_IF_ERROR(iree_trace_emitter_printf(emitter, "- "));
  IREE_RETURN_IF_ERROR(iree_trace_emitter_print_target(emitter, target));
  IREE_RETURN_IF_ERROR(iree_trace_emitter_printf(emitter, "\n"));
  return iree_ok_status();
}

// Prepares a new subdocument by adding a `---` document separator if needed.
static iree_status_t iree_trace_emitter_event(iree_trace_emitter_t* emitter) {
  if (emitter->state == IREE_TRACE_EMITTER_STATE_OPEN) {
    // Have emitted something and are starting a new event, split with `---`.
    IREE_RETURN_IF_ERROR(iree_trace_emitter_printf(emitter, "---\n"));
  }
  emitter->state = IREE_TRACE_EMITTER_STATE_OPEN;
  return iree_ok_status();
}

// Prepares a new multi-step event by preparing a new subdocument and setting
// the emitter to |new_state|.
static iree_status_t iree_trace_emitter_begin_event(
    iree_trace_emitter_t* emitter, iree_trace_emitter_state_t new_state) {
  // Cleanup prior state and transition to the specified new state.
  IREE_RETURN_IF_ERROR(iree_trace_emitter_event(emitter));
  emitter->state = new_state;
  return iree_ok_status();
}

// Updates a multi-step event sequence in |new_state| by possibly starting a
// new sequence with |sequence_key|: `<sequence_key>:\n`.
// Callers must specify the `-` prefix on each sequence item they add.
static iree_status_t iree_trace_emitter_update_event_sequence(
    iree_trace_emitter_t* emitter, iree_trace_emitter_state_t allowed_states,
    iree_trace_emitter_state_t new_state, iree_string_view_t sequence_key) {
  if (iree_all_bits_set(emitter->state, new_state)) {
    // Already in the sequence, continue emitting.
    return iree_ok_status();
  }
  if (iree_any_bit_set(emitter->state, allowed_states)) {
    // In a compatible prior state, transition and emit the sequence key.
    emitter->state = new_state;
    return iree_trace_emitter_printf(emitter, "%.*s:\n", (int)sequence_key.size,
                                     sequence_key.data);
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "failed emitting sequence '%.*s' as emitter is not in an allowed state; "
      "check for proper emission order",
      (int)sequence_key.size, sequence_key.data);
}

// Ends a multi-step event.
static iree_status_t iree_trace_emitter_end_event(
    iree_trace_emitter_t* emitter) {
  // Moving to open lets us know we've emitted an event (not empty) but don't
  // expect any more multi-step updates.
  emitter->state = IREE_TRACE_EMITTER_STATE_OPEN;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_trace_emitter_t event emission
//===----------------------------------------------------------------------===//

//----------------------------------------------------------------------------//
// `context_load`
//----------------------------------------------------------------------------//

iree_status_t iree_trace_emit_context_load(iree_trace_emitter_t* emitter,
                                           iree_vm_context_t* context) {
  IREE_ASSERT_ARGUMENT(emitter);
  IREE_ASSERT_ARGUMENT(context);
  IREE_RETURN_IF_ERROR(iree_trace_emitter_event(emitter));
  IREE_RETURN_IF_ERROR(
      iree_trace_emitter_printf(emitter, "type: context_load\n"));
  return iree_ok_status();
}

//----------------------------------------------------------------------------//
// `module_load`
//----------------------------------------------------------------------------//

// NOTE: callers must indent all lines 2 spaces.
static iree_status_t iree_trace_emit_module_load_header(
    iree_trace_emitter_t* emitter, iree_vm_context_t* context,
    iree_string_view_t type, iree_string_view_t name) {
  IREE_RETURN_IF_ERROR(iree_trace_emitter_event(emitter));
  IREE_RETURN_IF_ERROR(iree_trace_emitter_printf(emitter,
                                                 "type: module_load\n"
                                                 "module:\n"
                                                 "  type: %.*s\n"
                                                 "  name: %.*s\n",
                                                 (int)type.size, type.data,
                                                 (int)name.size, name.data));
  return iree_ok_status();
}

iree_status_t iree_trace_emit_module_load_builtin(iree_trace_emitter_t* emitter,
                                                  iree_vm_context_t* context,
                                                  iree_string_view_t name) {
  IREE_ASSERT_ARGUMENT(emitter);
  IREE_ASSERT_ARGUMENT(context);
  IREE_RETURN_IF_ERROR(iree_trace_emit_module_load_header(
      emitter, context, IREE_SV("builtin"), name));
  return iree_ok_status();
}

iree_status_t iree_trace_emit_module_load_bytecode_from_stdin(
    iree_trace_emitter_t* emitter, iree_vm_context_t* context,
    iree_string_view_t name) {
  IREE_ASSERT_ARGUMENT(emitter);
  IREE_ASSERT_ARGUMENT(context);
  IREE_RETURN_IF_ERROR(iree_trace_emit_module_load_header(
      emitter, context, IREE_SV("bytecode"), name));
  IREE_RETURN_IF_ERROR(iree_trace_emitter_printf(emitter, "  path: <stdin>\n"));
  return iree_ok_status();
}

iree_status_t iree_trace_emit_module_load_bytecode_from_path(
    iree_trace_emitter_t* emitter, iree_vm_context_t* context,
    iree_string_view_t name, iree_string_view_t path) {
  IREE_ASSERT_ARGUMENT(emitter);
  IREE_ASSERT_ARGUMENT(context);
  IREE_RETURN_IF_ERROR(iree_trace_emit_module_load_header(
      emitter, context, IREE_SV("bytecode"), name));
  IREE_RETURN_IF_ERROR(iree_trace_emitter_printf(emitter, "  path: %.*s\n",
                                                 (int)path.size, path.data));
  return iree_ok_status();
}

iree_status_t iree_trace_emit_module_load_bytecode_from_data(
    iree_trace_emitter_t* emitter, iree_vm_context_t* context,
    iree_string_view_t name, iree_const_byte_span_t contents) {
  IREE_ASSERT_ARGUMENT(emitter);
  IREE_ASSERT_ARGUMENT(context);
  IREE_RETURN_IF_ERROR(iree_trace_emit_module_load_header(
      emitter, context, IREE_SV("bytecode"), name));
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "embedded data not yet implemented");
}

//----------------------------------------------------------------------------//
// `blackboard_clear`
//----------------------------------------------------------------------------//

iree_status_t iree_trace_emit_blackboard_clear(iree_trace_emitter_t* emitter) {
  IREE_ASSERT_ARGUMENT(emitter);
  IREE_RETURN_IF_ERROR(iree_trace_emitter_event(emitter));
  IREE_RETURN_IF_ERROR(
      iree_trace_emitter_printf(emitter, "type: blackboard_clear\n"));
  emitter->next_blackboard_slot = 0;
  return iree_ok_status();
}

//----------------------------------------------------------------------------//
// `assign`
//----------------------------------------------------------------------------//

iree_status_t iree_trace_emit_assign(iree_trace_emitter_t* emitter,
                                     iree_trace_emitter_source_t from,
                                     iree_trace_emitter_target_t to) {
  IREE_ASSERT_ARGUMENT(emitter);
  iree_trace_emitter_event_t event = 0;
  IREE_RETURN_IF_ERROR(iree_trace_emit_assign_begin(emitter, &event));
  IREE_RETURN_IF_ERROR(
      iree_trace_emit_assign_append_from(emitter, event, from));
  IREE_RETURN_IF_ERROR(iree_trace_emit_assign_append_to(emitter, event, to));
  return iree_trace_emit_assign_end(emitter, event);
}

iree_status_t iree_trace_emit_assign_begin(
    iree_trace_emitter_t* emitter, iree_trace_emitter_event_t* out_event) {
  IREE_ASSERT_ARGUMENT(emitter);
  IREE_ASSERT_ARGUMENT(out_event);
  *out_event = 0;
  IREE_RETURN_IF_ERROR(iree_trace_emitter_begin_event(
      emitter, IREE_TRACE_EMITTER_STATE_ASSIGN_BEGIN));
  IREE_RETURN_IF_ERROR(iree_trace_emitter_printf(emitter, "type: assign\n"));
  return iree_ok_status();
}

iree_status_t iree_trace_emit_assign_append_from(
    iree_trace_emitter_t* emitter, iree_trace_emitter_event_t event,
    iree_trace_emitter_source_t source) {
  IREE_ASSERT_ARGUMENT(emitter);
  if (iree_trace_emitter_source_is_undefined(source)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_trace_emitter_update_event_sequence(
      emitter, IREE_TRACE_EMITTER_STATE_ASSIGN_BEGIN,
      IREE_TRACE_EMITTER_STATE_ASSIGN_FROM, IREE_SV("from")));
  IREE_RETURN_IF_ERROR(
      iree_trace_emitter_print_sequence_source(emitter, source));
  return iree_ok_status();
}

iree_status_t iree_trace_emit_assign_append_to(
    iree_trace_emitter_t* emitter, iree_trace_emitter_event_t event,
    iree_trace_emitter_target_t target) {
  IREE_ASSERT_ARGUMENT(emitter);
  if (iree_trace_emitter_target_is_undefined(target)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_trace_emitter_update_event_sequence(
      emitter,
      IREE_TRACE_EMITTER_STATE_ASSIGN_BEGIN |
          IREE_TRACE_EMITTER_STATE_ASSIGN_FROM,
      IREE_TRACE_EMITTER_STATE_ASSIGN_TO, IREE_SV("to")));
  IREE_RETURN_IF_ERROR(
      iree_trace_emitter_print_sequence_target(emitter, target));
  return iree_ok_status();
}

iree_status_t iree_trace_emit_assign_end(iree_trace_emitter_t* emitter,
                                         iree_trace_emitter_event_t event) {
  IREE_ASSERT_ARGUMENT(emitter);
  return iree_trace_emitter_end_event(emitter);
}

//----------------------------------------------------------------------------//
// `numpy_load`
//----------------------------------------------------------------------------//

iree_status_t iree_trace_emit_numpy_load_array(
    iree_trace_emitter_t* emitter, iree_string_view_t path,
    const iree_trace_emitter_target_t array) {
  return iree_trace_emit_numpy_load_arrays(emitter, path, 1, &array);
}

iree_status_t iree_trace_emit_numpy_load_arrays(
    iree_trace_emitter_t* emitter, iree_string_view_t path,
    iree_host_size_t array_count, const iree_trace_emitter_target_t* arrays) {
  IREE_ASSERT_ARGUMENT(emitter);
  iree_trace_emitter_event_t event = 0;
  IREE_RETURN_IF_ERROR(iree_trace_emit_numpy_load_begin(emitter, path, &event));
  for (iree_host_size_t i = 0; i < array_count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_trace_emit_numpy_load_append_array(emitter, event, arrays[i]));
  }
  return iree_trace_emit_numpy_load_end(emitter, event);
}

iree_status_t iree_trace_emit_numpy_load_begin(
    iree_trace_emitter_t* emitter, iree_string_view_t path,
    iree_trace_emitter_event_t* out_event) {
  IREE_ASSERT_ARGUMENT(emitter);
  IREE_ASSERT_ARGUMENT(out_event);
  *out_event = 0;
  IREE_RETURN_IF_ERROR(iree_trace_emitter_begin_event(
      emitter, IREE_TRACE_EMITTER_STATE_NUMPY_LOAD_BEGIN));
  IREE_RETURN_IF_ERROR(iree_trace_emitter_printf(emitter,
                                                 "type: numpy_load\n"
                                                 "path: %.*s\n",
                                                 (int)path.size, path.data));
  return iree_ok_status();
}

iree_status_t iree_trace_emit_numpy_load_append_array(
    iree_trace_emitter_t* emitter, iree_trace_emitter_event_t event,
    iree_trace_emitter_target_t target) {
  IREE_ASSERT_ARGUMENT(emitter);
  if (iree_trace_emitter_target_is_undefined(target)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_trace_emitter_update_event_sequence(
      emitter, IREE_TRACE_EMITTER_STATE_NUMPY_LOAD_BEGIN,
      IREE_TRACE_EMITTER_STATE_NUMPY_LOAD_ARRAYS, IREE_SV("arrays")));
  IREE_RETURN_IF_ERROR(
      iree_trace_emitter_print_sequence_target(emitter, target));
  return iree_ok_status();
}

iree_status_t iree_trace_emit_numpy_load_end(iree_trace_emitter_t* emitter,
                                             iree_trace_emitter_event_t event) {
  IREE_ASSERT_ARGUMENT(emitter);
  return iree_trace_emitter_end_event(emitter);
}

//----------------------------------------------------------------------------//
// `numpy_save`
//----------------------------------------------------------------------------//

iree_status_t iree_trace_emit_numpy_save_array(
    iree_trace_emitter_t* emitter, iree_string_view_t path, bool append,
    iree_trace_emitter_source_t array) {
  return iree_trace_emit_numpy_save_arrays(emitter, path, append, 1, &array);
}

iree_status_t iree_trace_emit_numpy_save_arrays(
    iree_trace_emitter_t* emitter, iree_string_view_t path, bool append,
    iree_host_size_t array_count, const iree_trace_emitter_source_t* arrays) {
  IREE_ASSERT_ARGUMENT(emitter);
  iree_trace_emitter_event_t event = 0;
  IREE_RETURN_IF_ERROR(
      iree_trace_emit_numpy_save_begin(emitter, path, append, &event));
  for (iree_host_size_t i = 0; i < array_count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_trace_emit_numpy_save_append_array(emitter, event, arrays[i]));
  }
  return iree_trace_emit_numpy_save_end(emitter, event);
}

iree_status_t iree_trace_emit_numpy_save_begin(
    iree_trace_emitter_t* emitter, iree_string_view_t path, bool append,
    iree_trace_emitter_event_t* out_event) {
  IREE_ASSERT_ARGUMENT(emitter);
  IREE_ASSERT_ARGUMENT(out_event);
  *out_event = 0;
  IREE_RETURN_IF_ERROR(iree_trace_emitter_begin_event(
      emitter, IREE_TRACE_EMITTER_STATE_NUMPY_SAVE_BEGIN));
  IREE_RETURN_IF_ERROR(iree_trace_emitter_printf(emitter,
                                                 "type: numpy_save\n"
                                                 "path: %.*s\n"
                                                 "append: %s\n",
                                                 (int)path.size, path.data,
                                                 append ? "true" : "false"));
  return iree_ok_status();
}

iree_status_t iree_trace_emit_numpy_save_append_array(
    iree_trace_emitter_t* emitter, iree_trace_emitter_event_t event,
    iree_trace_emitter_source_t source) {
  IREE_ASSERT_ARGUMENT(emitter);
  if (iree_trace_emitter_source_is_undefined(source)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_trace_emitter_update_event_sequence(
      emitter, IREE_TRACE_EMITTER_STATE_NUMPY_SAVE_BEGIN,
      IREE_TRACE_EMITTER_STATE_NUMPY_SAVE_ARRAYS, IREE_SV("arrays")));
  IREE_RETURN_IF_ERROR(
      iree_trace_emitter_print_sequence_source(emitter, source));
  return iree_ok_status();
}

iree_status_t iree_trace_emit_numpy_save_end(iree_trace_emitter_t* emitter,
                                             iree_trace_emitter_event_t event) {
  IREE_ASSERT_ARGUMENT(emitter);
  return iree_trace_emitter_end_event(emitter);
}

//----------------------------------------------------------------------------//
// `call`
//----------------------------------------------------------------------------//

iree_status_t iree_trace_emit_call(iree_trace_emitter_t* emitter,
                                   iree_vm_context_t* context,
                                   iree_vm_function_t function,
                                   iree_host_size_t arg_count,
                                   const iree_trace_emitter_source_t* args,
                                   iree_host_size_t result_count,
                                   const iree_trace_emitter_target_t* results) {
  IREE_ASSERT_ARGUMENT(emitter);
  iree_trace_emitter_event_t event = 0;
  IREE_RETURN_IF_ERROR(
      iree_trace_emit_call_begin(emitter, context, function, &event));
  for (iree_host_size_t i = 0; i < arg_count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_trace_emit_call_append_argument(emitter, event, args[i]));
  }
  for (iree_host_size_t i = 0; i < result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_trace_emit_call_append_result(emitter, event, results[i]));
  }
  return iree_trace_emit_call_end(emitter, event);
}

iree_status_t iree_trace_emit_call_begin(
    iree_trace_emitter_t* emitter, iree_vm_context_t* context,
    iree_vm_function_t function, iree_trace_emitter_event_t* out_event) {
  IREE_ASSERT_ARGUMENT(emitter);
  IREE_ASSERT_ARGUMENT(out_event);
  *out_event = 0;
  IREE_RETURN_IF_ERROR(iree_trace_emitter_begin_event(
      emitter, IREE_TRACE_EMITTER_STATE_CALL_BEGIN));
  iree_string_view_t module_name = iree_vm_module_name(function.module);
  iree_string_view_t function_name = iree_vm_function_name(&function);
  IREE_RETURN_IF_ERROR(
      iree_trace_emitter_printf(emitter,
                                "type: call\n"
                                "function: %.*s.%.*s\n",
                                (int)module_name.size, module_name.data,
                                (int)function_name.size, function_name.data));
  return iree_ok_status();
}

iree_status_t iree_trace_emit_call_append_argument(
    iree_trace_emitter_t* emitter, iree_trace_emitter_event_t event,
    iree_trace_emitter_source_t source) {
  IREE_ASSERT_ARGUMENT(emitter);
  if (iree_trace_emitter_source_is_undefined(source)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_trace_emitter_update_event_sequence(
      emitter, IREE_TRACE_EMITTER_STATE_CALL_BEGIN,
      IREE_TRACE_EMITTER_STATE_CALL_ARGUMENTS, IREE_SV("args")));
  IREE_RETURN_IF_ERROR(
      iree_trace_emitter_print_sequence_source(emitter, source));
  return iree_ok_status();
}

iree_status_t iree_trace_emit_call_append_result(
    iree_trace_emitter_t* emitter, iree_trace_emitter_event_t event,
    iree_trace_emitter_target_t target) {
  IREE_ASSERT_ARGUMENT(emitter);
  if (iree_trace_emitter_target_is_undefined(target)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_trace_emitter_update_event_sequence(
      emitter,
      IREE_TRACE_EMITTER_STATE_CALL_BEGIN |
          IREE_TRACE_EMITTER_STATE_CALL_ARGUMENTS,
      IREE_TRACE_EMITTER_STATE_CALL_RESULTS, IREE_SV("results")));
  IREE_RETURN_IF_ERROR(
      iree_trace_emitter_print_sequence_target(emitter, target));
  return iree_ok_status();
}

iree_status_t iree_trace_emit_call_end(iree_trace_emitter_t* emitter,
                                       iree_trace_emitter_event_t event) {
  IREE_ASSERT_ARGUMENT(emitter);
  return iree_trace_emitter_end_event(emitter);
}
