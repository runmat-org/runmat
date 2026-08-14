use runmat_mir::{MirCall, MirCallArg, MirCallee};
use runmat_runtime::call::arguments::MaterializedArgument;
use runmat_runtime::call::descriptor::{CallableCallKind, CallableDescriptor};
use runmat_value::Value;

use crate::{JitError, JitResult};

use super::operand::materialize_operand;
use super::state::HostState;

pub(super) fn evaluate(state: &mut HostState, call: &MirCall) -> JitResult<Vec<Value>> {
    let arguments = materialize_arguments(state, &call.args)?;
    let requested_outputs = call.requested_outputs.fixed_count();
    let result = match &call.callee {
        MirCallee::Static(identity) => {
            let descriptor = CallableDescriptor::resolved(
                identity.clone(),
                arguments,
                requested_outputs,
                call.fallback_policy,
                CallableCallKind::Direct,
            );
            super::sync::complete(
                &state.runtime,
                runmat_runtime::call::descriptor::execute_callable_descriptor(descriptor),
                "resolved call",
            )
        }
        MirCallee::Dynamic(operand) => {
            let target = materialize_operand(state, operand)?;
            super::sync::complete(
                &state.runtime,
                runmat_runtime::call_feval_async_with_outputs(
                    target,
                    &arguments,
                    requested_outputs,
                ),
                "dynamic call",
            )
        }
        MirCallee::SuperConstructor {
            current_class,
            super_class,
        } => {
            let _outputs = runmat_runtime::output_context::push_output_count(requested_outputs);
            super::sync::complete(
                &state.runtime,
                runmat_runtime::call_super_constructor(
                    current_class.clone(),
                    super_class.clone(),
                    arguments,
                ),
                "superclass constructor call",
            )
        }
        MirCallee::SuperMethod {
            current_class,
            super_class,
            method,
        } => {
            let _outputs = runmat_runtime::output_context::push_output_count(requested_outputs);
            super::sync::complete(
                &state.runtime,
                runmat_runtime::call_super_method(
                    current_class.clone(),
                    super_class.clone(),
                    method.clone(),
                    arguments,
                ),
                "superclass method call",
            )
        }
    }?;
    normalize_outputs(result, requested_outputs)
}

pub(super) fn materialize_arguments(
    state: &mut HostState,
    arguments: &[MirCallArg],
) -> JitResult<Vec<Value>> {
    let materialized = arguments
        .iter()
        .map(|argument| materialize_argument(state, argument))
        .collect::<JitResult<Vec<_>>>()?;
    super::sync::complete(
        &state.runtime,
        runmat_runtime::call::arguments::expand_arguments(materialized),
        "call argument expansion",
    )
}

fn materialize_argument(
    state: &mut HostState,
    argument: &MirCallArg,
) -> JitResult<MaterializedArgument> {
    match argument {
        MirCallArg::Single(operand) => {
            materialize_operand(state, operand).map(MaterializedArgument::Single)
        }
        MirCallArg::Expansion {
            base,
            indices,
            expand_all,
        } => Ok(MaterializedArgument::Expansion {
            base: materialize_operand(state, base)?,
            indices: indices
                .iter()
                .map(|index| materialize_operand(state, index))
                .collect::<JitResult<Vec<_>>>()?,
            expand_all: *expand_all,
        }),
    }
}

pub(super) fn builtin(
    state: &mut HostState,
    name: &str,
    arguments: Vec<Value>,
    requested_outputs: usize,
) -> JitResult<Vec<Value>> {
    let result = super::sync::complete(
        &state.runtime,
        runmat_runtime::call_builtin_async_with_outputs(name, &arguments, requested_outputs),
        "builtin call",
    )?;
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
