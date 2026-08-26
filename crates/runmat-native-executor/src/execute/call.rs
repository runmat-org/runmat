use runmat_mir::{MirCall, MirCallArg, MirCallee};
use runmat_runtime::call::arguments::MaterializedArgument;
use runmat_runtime::call::descriptor::{CallableCallKind, CallableDescriptor};
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::operand::materialize_operand;
use super::state::HostState;

pub(super) fn evaluate(state: &mut HostState, call: &MirCall) -> NativeExecutorResult<Vec<Value>> {
    if let Some(outputs) = super::call_suspension::take_completed(state)? {
        return Ok(outputs);
    }
    let mut arguments = materialize_arguments(state, &call.args)?;
    let requested_outputs = call.requested_outputs.fixed_count();
    if matches!(
        call.syntax,
        runmat_hir::CallSyntax::Method | runmat_hir::CallSyntax::DottedInvoke
    ) {
        if let MirCallee::Static(identity) = &call.callee {
            if !matches!(
                identity,
                runmat_hir::CallableIdentity::BoundFunction(_)
                    | runmat_hir::CallableIdentity::ExternalFunction { .. }
            ) {
                if arguments.is_empty() {
                    return Err(NativeExecutorError::Host(
                        "method/member-index call requires a base receiver".into(),
                    ));
                }
                let base = arguments.remove(0);
                let _outputs = runmat_runtime::output_context::push_output_count(requested_outputs);
                let caller = state.function.name.clone();
                let class_context =
                    runmat_runtime::class_registry::class_context_for_function(&caller);
                let _access = class_context
                    .map(|class_name| runmat_runtime::push_class_access_context(Some(class_name)));
                let value = complete_call(
                    state,
                    {
                        let identity = identity.clone();
                        let fallback_policy = call.fallback_policy;
                        async move {
                            runmat_runtime::object::dispatch::call_method_or_member_index_with_outputs(
                                base,
                                identity,
                                arguments,
                                requested_outputs,
                                (!caller.is_empty()).then_some(caller.as_str()),
                                fallback_policy,
                            )
                            .await
                        }
                    },
                    requested_outputs,
                )?;
                return Ok(value);
            }
        }
    }
    let result = match &call.callee {
        MirCallee::Static(identity) => {
            if let Some(function) = local_program_function(identity)? {
                if let Some(captures) = state.lexical_captures(function)? {
                    let invoker =
                        runmat_runtime::user_functions::current_lexical_function_invoker()
                            .ok_or_else(|| {
                                NativeExecutorError::Host(
                                    "native nested call has no lexical function invoker".into(),
                                )
                            })?;
                    let runtime = state.runtime.clone();
                    let result = super::sync::complete(
                        &runtime,
                        invoker(runmat_runtime::call::lexical::LexicalCall {
                            function: function.0 as usize,
                            captures,
                            arguments,
                            requested_outputs,
                        }),
                        "nested lexical call",
                    )?;
                    state.apply_lexical_captures(result.captures)?;
                    return normalize_outputs(result.value, requested_outputs);
                }
            }
            let descriptor = CallableDescriptor::resolved(
                identity.clone(),
                arguments,
                requested_outputs,
                call.fallback_policy,
                CallableCallKind::Direct,
            );
            return complete_call(
                state,
                runmat_runtime::call::descriptor::execute_callable_descriptor(descriptor),
                requested_outputs,
            );
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

fn local_program_function(
    identity: &runmat_hir::CallableIdentity,
) -> NativeExecutorResult<Option<runmat_types::ProgramFunctionId>> {
    let function = match identity {
        runmat_hir::CallableIdentity::BoundFunction(function)
        | runmat_hir::CallableIdentity::AnonymousFunction(function) => function.0,
        _ => return Ok(None),
    };
    u32::try_from(function)
        .map(runmat_types::ProgramFunctionId)
        .map(Some)
        .map_err(|_| {
            NativeExecutorError::Host("semantic function identity exceeds native schema".into())
        })
}

pub(super) fn materialize_arguments(
    state: &mut HostState,
    arguments: &[MirCallArg],
) -> NativeExecutorResult<Vec<Value>> {
    let materialized = arguments
        .iter()
        .map(|argument| materialize_argument(state, argument))
        .collect::<NativeExecutorResult<Vec<_>>>()?;
    super::sync::complete(
        &state.runtime,
        runmat_runtime::call::arguments::expand_arguments(materialized),
        "call argument expansion",
    )
}

fn materialize_argument(
    state: &mut HostState,
    argument: &MirCallArg,
) -> NativeExecutorResult<MaterializedArgument> {
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
                .collect::<NativeExecutorResult<Vec<_>>>()?,
            expand_all: *expand_all,
        }),
    }
}

pub(super) fn builtin(
    state: &mut HostState,
    name: &str,
    arguments: Vec<Value>,
    requested_outputs: usize,
) -> NativeExecutorResult<Vec<Value>> {
    if let Some(outputs) = super::call_suspension::take_completed(state)? {
        return Ok(outputs);
    }
    complete_call(
        state,
        {
            let name = name.to_owned();
            async move {
                runmat_runtime::call_builtin_async_with_outputs(
                    &name,
                    &arguments,
                    requested_outputs,
                )
                .await
            }
        },
        requested_outputs,
    )
}

fn complete_call(
    state: &mut HostState,
    future: impl std::future::Future<Output = Result<Value, runmat_runtime::RuntimeError>> + 'static,
    requested_outputs: usize,
) -> NativeExecutorResult<Vec<Value>> {
    let runtime = state.runtime.clone();
    super::call_suspension::begin(
        state,
        Box::pin(async move {
            let value = runtime.scope(future).await?;
            normalize_outputs(value, requested_outputs)
        }),
    )
}

fn normalize_outputs(result: Value, requested_outputs: usize) -> NativeExecutorResult<Vec<Value>> {
    if requested_outputs == 0 {
        return Ok(Vec::new());
    }
    match result {
        Value::OutputList(values) if values.len() == requested_outputs => Ok(values),
        Value::OutputList(values) => Err(NativeExecutorError::Host(format!(
            "runtime returned {} outputs for a {}-output call",
            values.len(),
            requested_outputs
        ))),
        value if requested_outputs == 1 => Ok(vec![value]),
        _ => Err(NativeExecutorError::Host(format!(
            "runtime did not return an output list for a {requested_outputs}-output call"
        ))),
    }
}
