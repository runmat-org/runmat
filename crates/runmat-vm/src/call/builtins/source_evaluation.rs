//! Executor-owned source evaluation for base-workspace callbacks and isolated text.

use super::{
    compile_dynamic_workspace_eval_inner, compile_dynamic_workspace_eval_inner_typed,
    parse_dynamic_eval, DynamicEvalCompileRequest, DynamicEvalCompiled, DynamicEvalResultCapture,
    DynamicTerminalOutput, FunctionRegistry,
};
use crate::interpreter::api::InterpreterOutcome;
use crate::interpreter::dynamic_admission::DynamicEvalAdmissionPort;
use crate::interpreter::runner::interpret_workspace;
use runmat_runtime::RuntimeError;
use runmat_value::Value;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

#[path = "source_evaluation/numeric_text.rs"]
mod numeric_text;
pub use numeric_text::evaluate_numeric_text_source_in_context;

/// Execute executor-owned source text against the session's canonical base
/// workspace. Asynchronous callbacks enter here without borrowing an ambient
/// caller frame or synthesizing source identity.
pub async fn evaluate_session_source_in_context(
    runtime: runmat_runtime::context::RuntimeContext,
    source: String,
    source_label: String,
    source_ref: runmat_runtime::source_context::RuntimeSourceRef,
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
) -> Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
    evaluate_session_source_with_dynamic_admission_in_context(
        runtime,
        source,
        source_label,
        source_ref,
        requested_outputs,
        function_registry,
        None,
    )
    .await
}

pub struct DynamicSessionEvalAdmission {
    pub source_lease: runmat_runtime::source_context::RuntimeSourceLease,
    pub port: DynamicEvalAdmissionPort,
}

pub async fn evaluate_session_source_with_dynamic_admission_in_context(
    runtime: runmat_runtime::context::RuntimeContext,
    source: String,
    source_label: String,
    source_ref: runmat_runtime::source_context::RuntimeSourceRef,
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
    selected: Option<DynamicSessionEvalAdmission>,
) -> Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
    if selected
        .as_ref()
        .is_some_and(|selected| selected.source_lease.source_ref() != source_ref)
    {
        return Err(runmat_runtime::build_runtime_error(
            "dynamic source lease does not match the admitted source reference",
        )
        .with_identifier("RunMat:SourceEvaluation:SourceMismatch")
        .build());
    }
    if requested_outputs > 1 {
        return Err(runmat_runtime::build_runtime_error(format!(
            "source evaluation supports at most one output, but {requested_outputs} were requested"
        ))
        .with_identifier("RunMat:SourceEvaluation:TooManyOutputs")
        .build());
    }
    let base = runtime
        .service_ports()
        .require_base_workspace("evaluate session source")
        .map_err(runmat_runtime::context::RuntimeCapabilityError::into_runtime_error)?
        .owner();
    let projection = runmat_runtime::workspace::BaseWorkspaceProjectionService::new_for_context(
        base.begin_projection()
            .map_err(runmat_runtime::workspace::base_workspace_error_to_runtime)?,
        &runtime,
    );
    let generation = runmat_runtime::context::RuntimeWorkspaceService::capture_workspace_generation_with_context(
        &projection, &runtime,
    )?.ok_or_else(|| runmat_runtime::build_runtime_error("session source has no canonical workspace generation")
        .with_identifier("RunMat:SourceEvaluation:WorkspaceGeneration").build())?;
    let mut names = HashMap::new();
    let mut assigned = HashSet::new();
    for (slot, (name, _)) in generation.iter().enumerate() {
        names.insert(name.to_owned(), slot);
        assigned.insert(name.to_owned());
    }
    let global_aliases = runmat_runtime::workspace::bindings::FrameGlobalBindings::default();
    for (name, target) in generation.global_bindings() {
        if let Some(slot) = names.get(name).copied() {
            let prepared = global_aliases
                .prepare_target(&runtime, slot, target)
                .map_err(|cause| {
                    crate::runtime::workspace::frame_allocation_error_in_context(
                        Some(&runtime),
                        cause,
                    )
                })?;
            prepared.checkpoint().map_err(|cause| {
                crate::runtime::workspace::frame_allocation_error_in_context(Some(&runtime), cause)
            })?;
            prepared.commit();
        }
    }
    let requires_product = |canonical: &crate::bytecode::Bytecode, inherited: &FunctionRegistry| {
        selected.as_ref().map_or(Ok(false), |selected| {
            selected
                .port
                .host()
                .requires_selected_product(canonical, inherited)
        })
    };
    let ast = parse_dynamic_eval("source evaluation", &source)?;
    let Some(compiled) = compile_dynamic_workspace_eval_inner(DynamicEvalCompileRequest {
        builtin: "source evaluation",
        ast: &ast,
        workspace_bindings: &names,
        workspace_value_count: names.len(),
        lowering_bindings: &names,
        function_registry,
        source_ref: Some(source_ref),
        requested_outputs,
        result_capture: DynamicEvalResultCapture::TerminalExpression(
            DynamicTerminalOutput::Preserve,
        ),
        select_product: selected.as_ref().map(|_| {
            &requires_product
                as &dyn Fn(
                    &crate::bytecode::Bytecode,
                    &FunctionRegistry,
                ) -> Result<bool, RuntimeError>
        }),
        selected_sources: super::eval_static_selection::EvalSelectedSources::default(),
    })?
    else {
        return runmat_runtime::execution::RootedValueSequence::new(
            runmat_value::ValueSequence::empty(),
        )
        .map_err(|error| error.into_runtime_error());
    };
    let (bytecode, result, runtime, _publication) = match compiled {
        DynamicEvalCompiled::Ordinary(bytecode, result) => (
            *bytecode,
            result,
            runtime.without_selected_graph_program(),
            None,
        ),
        DynamicEvalCompiled::Selected(product) => {
            let selected = selected.as_ref().expect("selected product has a port");
            let admitted = selected.port.host().admit(
                &runtime,
                &selected.source_lease,
                &source,
                None,
                *product,
                function_registry,
            )?;
            (
                admitted.execution_bytecode,
                admitted.result,
                admitted.runtime,
                Some(admitted.publication),
            )
        }
    };

    let projection = Rc::new(projection);
    let services = runtime
        .service_ports()
        .clone()
        .with_workspace(projection.clone());
    let execution_runtime = runtime.with_service_ports(services);
    let mut publication = projection.terminal_publication(&execution_runtime)?;
    let frame = crate::InterpreterVars::from_workspace_generation(
        &execution_runtime,
        &generation,
        names.iter().map(|(name, slot)| (name.as_str(), *slot)),
        bytecode.var_count,
    )?;
    let admission = crate::runtime::workspace::detached_workspace_admission_with_global_bindings(
        names,
        assigned,
        global_aliases,
    );
    let execution = interpret_workspace(
        &bytecode,
        frame,
        Some(&source_label),
        execution_runtime.clone(),
        Some(admission),
        selected.as_ref().map(|selected| selected.port.clone()),
    )
    .await;

    publication.publish_pending()?;
    let InterpreterOutcome::Completed(completion) = execution.outcome?;
    if requested_outputs == 0 {
        return runmat_runtime::execution::RootedValueSequence::new(
            runmat_value::ValueSequence::empty(),
        )
        .map_err(|error| error.into_runtime_error());
    }
    completion.try_map_owned(
        |completion| {
            let value = result
                .and_then(|result| {
                    completion
                        .values
                        .get(result.slot)
                        .map(|value| value.clone())
                })
                .unwrap_or(Value::Num(0.0));
            runmat_value::ValueSequence::single(value)
                .map_err(runmat_runtime::sequence::sequence_error_to_runtime)
        },
        |error| error.into_runtime_error(),
    )
}
