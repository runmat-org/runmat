//! Executor-owned source evaluation against a session base workspace.

use super::{compile_dynamic_workspace_eval, FunctionRegistry};
use crate::interpreter::api::InterpreterOutcome;
use crate::interpreter::runner::interpret_with_workspace_in_context;
use runmat_runtime::RuntimeError;
use runmat_value::Value;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

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
) -> Result<runmat_value::ValueSequence, RuntimeError> {
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
    let projection = runmat_runtime::workspace::BaseWorkspaceProjectionService::new(
        base.begin_projection()
            .map_err(runmat_runtime::workspace::base_workspace_error_to_runtime)?,
    );
    let snapshot = projection.snapshot()?;
    let mut names = HashMap::with_capacity(snapshot.len());
    let mut assigned = HashSet::with_capacity(snapshot.len());
    let mut values = Vec::with_capacity(snapshot.len());
    for (slot, (name, value)) in snapshot.into_iter().enumerate() {
        names.insert(name.clone(), slot);
        assigned.insert(name);
        values.push(value);
    }
    let global_aliases = runmat_runtime::workspace::bindings::FrameGlobalBindings::default();
    for name in
        runmat_runtime::context::RuntimeWorkspaceService::base_global_binding_names(&projection)?
    {
        let Some(value) = runtime.session_global_value(&name)? else {
            continue;
        };
        let slot = if let Some(slot) = names.get(&name).copied() {
            values[slot] = value;
            slot
        } else {
            let slot = values.len();
            names.insert(name.clone(), slot);
            values.push(value);
            slot
        };
        assigned.insert(name.clone());
        global_aliases.declare_named(slot, name);
    }
    let Some((bytecode, result_slot)) = compile_dynamic_workspace_eval(
        "source evaluation",
        &source,
        &names,
        function_registry,
        Some(source_ref),
        requested_outputs,
    )?
    else {
        return Ok(runmat_value::ValueSequence::empty());
    };

    let projection = Rc::new(projection);
    let services = runtime
        .service_ports()
        .clone()
        .with_workspace(projection.clone());
    let execution_runtime = runtime.with_service_ports(services);
    let admission = crate::runtime::workspace::workspace_admission_with_global_bindings(
        names,
        assigned,
        global_aliases,
    );
    let execution = interpret_with_workspace_in_context(
        &bytecode,
        &mut values,
        Some(&source_label),
        execution_runtime,
        Some(admission),
    )
    .await;

    if projection.has_pending_changes()? {
        projection.publish()?;
    }
    let InterpreterOutcome::Completed(completion) = execution.outcome?;
    if requested_outputs == 0 {
        return Ok(runmat_value::ValueSequence::empty());
    }
    let value = result_slot
        .and_then(|slot| completion.values.get(slot).cloned())
        .unwrap_or(Value::Num(0.0));
    runmat_value::ValueSequence::single(value)
        .map_err(runmat_runtime::sequence::sequence_error_to_runtime)
}
