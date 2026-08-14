use futures::executor::block_on;
use runmat_mir::{MirCall, MirCallArg, MirCallee};
use runmat_runtime::call::descriptor::{CallableCallKind, CallableDescriptor};
use runmat_value::Value;

use crate::{JitError, JitResult};

use super::operand::materialize_operand;
use super::state::HostState;

pub(super) fn evaluate(state: &mut HostState, call: &MirCall) -> JitResult<Vec<Value>> {
    let arguments = call
        .args
        .iter()
        .map(|argument| match argument {
            MirCallArg::Single(operand) => materialize_operand(state, operand),
            MirCallArg::Expansion { .. } => Err(JitError::UnsupportedSite(
                "comma-separated-list call expansion requires the call-shape cohort".into(),
            )),
        })
        .collect::<JitResult<Vec<_>>>()?;
    let requested_outputs = call.requested_outputs.fixed_count();
    let result =
        match &call.callee {
            MirCallee::Static(identity) => {
                let descriptor = CallableDescriptor::resolved(
                    identity.clone(),
                    arguments,
                    requested_outputs,
                    call.fallback_policy,
                    CallableCallKind::Direct,
                );
                block_on(state.runtime.scope(
                    runmat_runtime::call::descriptor::execute_callable_descriptor(descriptor),
                ))
            }
            MirCallee::Dynamic(operand) => {
                let target = materialize_operand(state, operand)?;
                block_on(
                    state
                        .runtime
                        .scope(runmat_runtime::call_feval_async_with_outputs(
                            target,
                            &arguments,
                            requested_outputs,
                        )),
                )
            }
            other => {
                return Err(JitError::UnsupportedSite(format!(
                    "callee {other:?} requires the super-dispatch cohort"
                )))
            }
        }
        .map_err(JitError::from)?;
    normalize_outputs(result, requested_outputs)
}

pub(super) fn builtin(
    state: &mut HostState,
    name: &str,
    arguments: Vec<Value>,
    requested_outputs: usize,
) -> JitResult<Vec<Value>> {
    let result = block_on(
        state
            .runtime
            .scope(runmat_runtime::call_builtin_async_with_outputs(
                name,
                &arguments,
                requested_outputs,
            )),
    )
    .map_err(JitError::from)?;
    normalize_outputs(result, requested_outputs)
}

fn normalize_outputs(result: Value, requested_outputs: usize) -> JitResult<Vec<Value>> {
    if requested_outputs == 0 {
        return Ok(Vec::new());
    }
    match result {
        Value::OutputList(values) if values.len() == requested_outputs => Ok(values),
        Value::OutputList(values) => Err(JitError::Host(format!(
            "runtime returned {} outputs for a {}-output call",
            values.len(),
            requested_outputs
        ))),
        value if requested_outputs == 1 => Ok(vec![value]),
        _ => Err(JitError::Host(format!(
            "runtime did not return an output list for a {requested_outputs}-output call"
        ))),
    }
}
