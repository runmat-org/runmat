mod blocking_cpu;
mod callback_batch;
mod capture;
pub use callback_batch::SelectedWorkerCallable;
pub(crate) use callback_batch::{CallbackBatch, CallbackParallelMode};
mod completion;
mod cooperation;
mod errors;
mod future_handle;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod native_task;
mod owned_builtin;
mod phases;
pub use phases::after_result;
mod result_loop;
pub(crate) use result_loop::loop_result;
mod retaining;
pub(crate) use retaining::with_retained_owner;
mod program_capture;
mod retained_value;
pub(crate) mod rooted_continuation;
mod running_future;
pub(crate) mod scoped_callable;
pub use future_handle::{FutureCompletionRef, FutureExecutionHandle, FutureExecutionIdentity};
pub(crate) mod object;
pub(crate) mod object_array;
mod services;
pub(crate) mod staged_file;
mod storable;
pub(crate) mod structure;
pub use object::construct_object_in_context;
pub use program_capture::{
    capture_deferred_environment, capture_deferred_program, DeferredProgramCapture,
};
pub(crate) use structure::construct_struct_from_fields_in_optional_context;
pub use structure::{
    construct_struct_array_from_columns_in_context, construct_struct_array_from_values_in_context,
    construct_struct_from_fields_in_context, construct_struct_in_context,
};
pub mod value_codec;

#[cfg(not(target_arch = "wasm32"))]
pub use blocking_cpu::native_default_blocking_cpu_service;
pub use blocking_cpu::{OwnedCpuJob, OwnedCpuSubmission, RuntimeBlockingCpuService};
pub use capture::{validate_spawn_capture, validate_spawn_captures};
pub(crate) use completion::{
    ArgumentProjection, RootedValueAdmissionError, RootedValueChildren, RootedValueHandles,
    RootedValueMember, RootedValuesIteration,
};
pub use completion::{
    RootedResult, RootedValue, RootedValueLease, RootedValueSequence, RootedValues,
    RootedValuesBuilder, RootedValuesIntoIter,
};
pub use cooperation::{CpuCooperationBudget, CpuCooperationPolicy, OwnedCpuPayload};
pub use errors::{
    decode_runtime_failure, decode_runtime_failure_with_source_issuer_in_context,
    encode_runtime_failure, encode_runtime_failure_in_context,
    encode_runtime_failure_with_source_issuer_in_context, preserve_source_call_stack_in_context,
    render_source_call_stack_in_context, ExecutionServiceError,
};
pub use owned_builtin::{
    admit_owned_builtin_work, execute_owned_builtin_work, quote_owned_builtin_continuation,
    RuntimeOwnedBuiltinWorkService,
};
pub use retained_value::RetainedValueReadLease;
pub use rooted_continuation::{
    admit_domain_continuation_with, admit_dynamic_function_load_with,
    admit_dynamic_workspace_sequence, admit_executable_sequence, admit_executable_sequence_with,
    admit_executable_sequence_with_error, admit_execution_supervisor_with,
    admit_native_continuation_with, admit_native_hostcall_with,
    admit_native_hostcall_with_contextual_error, admit_optional_hostcall_with,
    admit_optional_rooted_result_with, admit_rooted_result_with, admit_stack_mutation,
    AdmittedExecutionSupervisor, RuntimeContinuationFuture,
};
// Numerical phases borrow their domain's original roots and mutable owners.
pub(crate) use rooted_continuation::admit_optional_hostcall_with_contextual_error as admit_borrowed_operation_with_contextual_error;
pub use running_future::RunningFutureLease;
pub use services::{
    AdmittedCallableReentry, AwaitAction, DeferredCall, DeferredGraphCapture, DeferredGraphProgram,
    DeferredGraphSnapshot, DeferredInvocation, DeferredProgram, DeferredSelectedCodeCapture,
    DurableJobOptions, GraphResultReceipt, RestoredCallableGraph, RuntimeExecutionService,
    RuntimeExecutionServices, SpmdExecutionMode, SpmdGangCall, SpmdGangResults, SpmdRankResult,
    TaskRetention,
};
pub use storable::validate_storable_value;
