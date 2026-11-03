#![allow(clippy::result_large_err)]

pub mod accel;
pub(crate) mod bytecode;
pub(crate) mod call;
pub(crate) mod compiler;
mod graph_schema;
pub(crate) mod instr {
    pub use crate::bytecode::instr::Instr;
}
pub(crate) mod interpreter;
pub(crate) mod layout;
pub(crate) mod object;
pub(crate) mod ops;
mod program_execution;
pub(crate) mod runtime;

pub use bytecode::program::{
    instruction_reference_sites, remap_reference_sites, FunctionReferenceKind,
};
pub use bytecode::program::{
    SelectedExecutableAttachmentError, SelectedExecutableCode, SelectedExecutableProduct,
    SelectedLinkedNativeSite, SelectedStaticCapture, SelectedStaticProductDependency,
    SelectedStaticProductTarget, SelectedStaticSiteOwner,
};
pub use graph_schema::{
    selected_callable_capture_required_schema, selected_graph_code_required_schema,
    selected_graph_execution_required_schema,
};

pub use bytecode::{
    compile, compile_semantic_function_registry, compile_semantic_function_registry_with_analysis,
    compile_with_analysis, validate_compiled_parallel_contracts,
};

/// Retain typed class declarations in a product without a callable or script
/// entrypoint, using the same HIR/MIR lowering as compiled function prologues.
/// A selected source keeps companion declarations out of the primary bytecode.
pub fn compile_class_declarations(
    hir: &runmat_hir::HirAssembly,
    mir: &runmat_mir::MirAssembly,
    source: Option<runmat_types::ProgramSourceId>,
) -> Vec<Instr> {
    compiler::core::class_declaration_instructions(hir, mir, source)
}
pub use bytecode::{
    decode_interpreter_program_v2, decode_interpreter_script_v2, encode_interpreter_program_v2,
    encode_interpreter_script_v2, InterpreterPayloadError, InterpreterPayloadForm,
    InterpreterRevisionField,
};
pub use bytecode::{
    rebind_session_source, remap_closed_session_function, remap_decorated_session_definitions,
    remap_session_entrypoint, remap_session_function, SessionCallableBinding,
};
pub use bytecode::{
    semantic_definition_fingerprint, AsyncMetadata, AwaitSite, Bytecode,
    BytecodeCodistributedOverload, BytecodeCollectiveOp, BytecodeDistributedBuildValidation,
    BytecodeDistributedOp, BytecodeDottedInvokeArgument, BytecodeParallelVariable,
    BytecodeParforRegion, BytecodeRegion, BytecodeRegionBoundary, BytecodeSourceRef,
    BytecodeSpmdHeader, BytecodeSpmdRegion, BytecodeSubscriptSelector, BytecodeSubscriptStep,
    EmitLabel, FunctionBytecode, FunctionRegistry, FunctionSourceProjectionError,
    InheritedStorageKind, InheritedStorageSlot, Instr, PrivateSourceAlias, PrivateSourceDirectory,
    PublishedNameResolutionView, SpawnSite, StackEffect, StatementPcProjectionError,
    StatementPcRange, StochasticSourceOperand, BYTECODE_SCHEMA_VERSION,
    FUNCTION_REGISTRY_SCHEMA_VERSION,
};
#[cfg(feature = "native-accel")]
pub use bytecode::{
    FusionCandidateGroup, FusionInstructionKind, FusionInstructionWindow, FusionMetadata,
    ImageCheckpoint, ImageLiveOut, ImageLiveOutSource, ImageNormalizePublicationJoin,
    ImageRegionStore, ImageStoreDestination, ImageStoreOrdinal, ProspectiveMultiStoreRegion,
    ProspectiveSemanticMultiStoreRegion, ProspectiveStore,
};
pub use call::builtins::{
    compile_dynamic_workspace_eval_product, evaluate_dynamic_workspace_in_context,
    evaluate_numeric_text_source_in_context, evaluate_session_source_in_context,
    evaluate_session_source_with_dynamic_admission_in_context, push_dynamic_eval_options,
    set_dynamic_eval_options, DynamicEvalCompileProduct, DynamicEvalResult,
    DynamicSessionEvalAdmission, EvalSelectedSource,
};
pub use compiler::CompileError;
pub use interpreter::dynamic_admission::{
    proves_no_graph_demand, AdmittedDynamicEval, AdmittedPendingClosure, DynamicEvalAdmissionPort,
    DynamicEvalProductAdmission, DynamicEvalPublication, PendingClosureAdmissionRequest,
};
pub use interpreter::runner::{
    interpret, interpret_function, interpret_function_with_counts, interpret_workspace,
    invoke_semantic_function_value, invoke_semantic_function_value_in_context,
    invoke_semantic_function_value_with_dynamic_admission_in_context,
    invoke_semantic_function_value_with_environment_in_context,
    invoke_semantic_function_with_rooted_arguments_and_dynamic_admission_in_context,
    WorkspaceInterpreterOutcome,
};
pub use interpreter::runner::{
    interpret_resume_in_context, interpret_resume_with_dynamic_admission_in_context,
    interpret_resume_with_environment_in_context, prepare_admitted_execution_metadata,
    prepare_admitted_execution_metadata_batch, prepare_admitted_execution_metadata_for_class_batch,
    prepare_native_execution_metadata, unresolved_admitted_class_parents,
    unresolved_admitted_class_parents_for, DynamicResumeSelection, SelectedClassProduct,
    SelectedDeclarationProduct,
};
pub use interpreter::state::{
    InterpreterCompletionBoundary, InterpreterOutcome, InterpreterResumeState, InterpreterState,
};
pub use interpreter::vars::{InterpreterValueRef, InterpreterVars};
pub use interpreter::CompletedFrameProjection;
pub use interpreter::PersistentBinding;
pub use layout::{
    derive_layout, remap_layout_function_ids, LayoutError, VmAssemblyLayout, VmEntrypointLayout,
    VmFrameAbi, VmFunctionLayout, VmSlotId, VM_LAYOUT_SCHEMA_VERSION,
};
pub use program_execution::callable_admission::{
    select_retained_worker_callable, select_worker_callable,
};
pub use program_execution::{
    admit_executable_unit, prepare_retained_callable_units, selected_graph_callable_facts,
    selected_graph_callable_facts_for_owner, AdmittedExecutableUnit, ExecutableUnitAdmissionError,
    GraphCodecAdmission, PreparedRetainedCallableUnits, RetainedCallableStageError,
    RetainedCallableStageLimits, RetainedExecutableSourceOwner, RuntimeBoundExecutableUnit,
    SelectedFingerprintInvariant, SelectedGraphCallableFacts, SelectedUnitInteropReceipt,
};
pub use program_execution::{
    compiled_graph_execution_capabilities, encode_portable_region_sources_v1,
    execute_deferred_program_in_context, execute_graph_request, execute_graph_request_with_context,
    execute_graph_request_with_context_recording,
    execute_graph_request_with_selected_foreign_ports_recording, execute_prepared_graph_request,
    execute_program_request, execute_program_request_with_context,
    execute_program_request_with_context_and_bundle, materialize_deferred_call,
    materialize_deferred_graph_call, prepare_graph_request_with_selected_foreign_ports_recording,
    runtime_failure_response, GraphFailureSourceRecorder, PreparedSelectedGraph,
    RestoredSelectedGraph, SelectedGraphCleanupContexts, SelectedGraphForeignPorts,
};
pub use runtime::workspace::{
    take_updated_workspace_assigned_report, take_updated_workspace_state, workspace_admission,
    workspace_admission_with_global_bindings, PendingWorkspaceAdmission, WorkspaceAssignedReport,
    WorkspaceExecutionCompletion,
};

#[doc(hidden)]
pub fn reset_thread_state_for_tests() {
    runmat_runtime::debug_context::reset_for_tests();
    runmat_runtime::builtins::introspection::debugging::reset_lock_registry_for_tests();
    runtime::globals::reset_thread_state_for_tests();
    runtime::workspace::reset_thread_state_for_tests();
}
