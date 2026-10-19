//! Typed builtin call boundaries used by executors and multi-output consumers.

use super::{call_builtin_sequence_async_impl, workspace_effect};
use crate::context::RuntimeContext;
use crate::execution::RootedValueSequence;
use crate::RuntimeError;
use runmat_value::Value;

/// Embedding call without session binding authority.
pub async fn call_builtin_sequence_async(
    name: &str,
    args: &[Value],
) -> Result<RootedValueSequence, RuntimeError> {
    call_builtin_sequence_async_impl(name, args, None, None, None).await
}

/// Session-owned binding selection, execution and workspace reconciliation.
pub async fn call_builtin_sequence_async_in_context(
    context: &RuntimeContext,
    name: &str,
    args: &[Value],
) -> Result<RootedValueSequence, RuntimeError> {
    call_in_context(context, name, args, None).await
}

pub async fn call_builtin_sequence_async_with_outputs(
    name: &str,
    args: &[Value],
    outputs: usize,
) -> Result<RootedValueSequence, RuntimeError> {
    call_builtin_sequence_async_impl(name, args, Some(outputs), None, None).await
}

pub async fn call_builtin_sequence_async_with_outputs_in_context(
    context: &RuntimeContext,
    name: &str,
    args: &[Value],
    outputs: usize,
) -> Result<RootedValueSequence, RuntimeError> {
    call_in_context(context, name, args, Some(outputs)).await
}

async fn call_in_context(
    context: &RuntimeContext,
    name: &str,
    args: &[Value],
    outputs: Option<usize>,
) -> Result<RootedValueSequence, RuntimeError> {
    let fence = workspace_effect::PostCallWorkspaceEffectFence::capture(Some(context));
    let observation = super::direct_call_observation::CallObservation::capture(Some(context))
        .map(std::rc::Rc::new);
    let result = context
        .scope(call_builtin_sequence_async_impl(
            name,
            args,
            outputs,
            Some(context.clone()),
            observation.clone(),
        ))
        .await;
    let (result, outcome) = fence.finish_owned_sequence_observed(result);
    if let Some(observation) = observation {
        observation.finish_completed(outcome);
    }
    result
}
