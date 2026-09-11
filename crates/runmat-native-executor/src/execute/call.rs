use runmat_mir::{MirCall, MirCallArg, MirCallee, MirExpansionSource, MirSequenceLocalId};
use runmat_runtime::call::arguments::{MaterializedArgument, MaterializedExpansionSource};
use runmat_runtime::call::descriptor::{CallableCallKind, CallableDescriptor};
use runmat_runtime::sequence::ResolveValueSequence;
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::operand::materialize_operand;
use super::state::HostState;

pub(super) fn evaluate(
    state: &mut HostState,
    call: &MirCall,
    requested_outputs: usize,
) -> NativeExecutorResult<Vec<Value>> {
    let embedded = state.enter_embedded_call();
    let result = evaluate_inner(state, call, requested_outputs, embedded.as_ref());
    if let Ok(outputs) = &result {
        state.cache_embedded_call(embedded.as_ref(), outputs);
    }
    let finished = state.finish_embedded_call(embedded.as_ref());
    match (result, finished) {
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Ok(outputs), Ok(())) => Ok(outputs),
    }
}

fn evaluate_inner(
    state: &mut HostState,
    call: &MirCall,
    requested_outputs: usize,
    embedded: Option<&super::state::EmbeddedOperationIdentity>,
) -> NativeExecutorResult<Vec<Value>> {
    if let Some(outputs) = super::call_suspension::take_completed(state, embedded)? {
        return Ok(outputs);
    }
    let mut arguments = materialize_arguments(state, &call.args)?;
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
                    embedded.cloned(),
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
    match &call.callee {
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
                    return normalize_sequence_outputs(result.outputs, requested_outputs);
                }
            }
            let descriptor = CallableDescriptor::resolved(
                identity.clone(),
                arguments,
                requested_outputs,
                call.fallback_policy,
                CallableCallKind::Direct,
            );
            complete_call(
                state,
                embedded.cloned(),
                runmat_runtime::call::descriptor::execute_callable_descriptor(descriptor),
                requested_outputs,
            )
        }
        MirCallee::Dynamic(operand) => {
            let target = materialize_operand(state, operand)?;
            let value = super::sync::complete(
                &state.runtime,
                runmat_runtime::call_feval_async_with_outputs(
                    target,
                    &arguments,
                    requested_outputs,
                ),
                "dynamic call",
            )?;
            normalize_legacy_outputs(value, requested_outputs)
        }
        MirCallee::SuperConstructor {
            current_class,
            super_class,
        } => {
            let _outputs = runmat_runtime::output_context::push_output_count(requested_outputs);
            let value = super::sync::complete(
                &state.runtime,
                runmat_runtime::call_super_constructor(
                    current_class.clone(),
                    super_class.clone(),
                    arguments,
                ),
                "superclass constructor call",
            )?;
            normalize_legacy_outputs(value, requested_outputs)
        }
        MirCallee::SuperMethod {
            current_class,
            super_class,
            method,
        } => {
            let _outputs = runmat_runtime::output_context::push_output_count(requested_outputs);
            let value = super::sync::complete(
                &state.runtime,
                runmat_runtime::call_super_method(
                    current_class.clone(),
                    super_class.clone(),
                    method.clone(),
                    arguments,
                ),
                "superclass method call",
            )?;
            normalize_legacy_outputs(value, requested_outputs)
        }
    }
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
        runmat_runtime::call::arguments::expand_arguments(&state.runtime, materialized),
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
        MirCallArg::Expansion(source) => Ok(MaterializedArgument::Expansion(
            materialize_expansion_source(state, source)?,
        )),
        MirCallArg::CapturedSequence(sequence) => {
            let values = take_captured_sequence(state, *sequence)?;
            Ok(MaterializedArgument::Sequence(
                runmat_runtime::sequence::ValueSequence::comma_separated(values)
                    .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?,
            ))
        }
    }
}

pub(super) fn capture_sequence(
    state: &mut HostState,
    destination: MirSequenceLocalId,
    source: &MirExpansionSource,
) -> NativeExecutorResult<()> {
    if state.captured_sequences.contains_key(&destination) {
        return Err(NativeExecutorError::Host(format!(
            "native sequence local {} was overwritten before consumption",
            destination.0
        )));
    }
    let values = if let MirExpansionSource::SubscriptChain(chain) = source {
        super::subscript_path::read(state, chain, 1)?
    } else {
        let source = materialize_expansion_source(state, source)?;
        let sequence = super::sync::complete(
            &state.runtime,
            runmat_runtime::call::arguments::materialize_expansion(&state.runtime, source),
            "sequence capture",
        )?;
        sequence.resolve(
            runmat_types::SequenceUse::ExpandAll,
            runmat_runtime::sequence::SequenceResolutionContext::default(),
        )?
    };
    let existing_roots = state
        .roots
        .len()
        .checked_add(
            state
                .captured_sequences
                .values()
                .try_fold(0usize, |total, values| total.checked_add(values.len()))
                .ok_or_else(|| {
                    NativeExecutorError::Host("native sequence root cardinality overflow".into())
                })?,
        )
        .and_then(|total| total.checked_add(values.len()))
        .ok_or_else(|| {
            NativeExecutorError::Host("native sequence root cardinality overflow".into())
        })?;
    u32::try_from(existing_roots).map_err(|_| {
        NativeExecutorError::Host("native sequence root cardinality exceeds the ABI".into())
    })?;
    let references = values
        .into_iter()
        .map(|value| state.arena.insert(value))
        .collect();
    state.captured_sequences.insert(destination, references);
    Ok(())
}

pub(super) fn take_captured_sequence(
    state: &mut HostState,
    sequence: MirSequenceLocalId,
) -> NativeExecutorResult<Vec<Value>> {
    let references = state.captured_sequences.remove(&sequence).ok_or_else(|| {
        NativeExecutorError::Host(format!(
            "native sequence local {} was read before capture or after consumption",
            sequence.0
        ))
    })?;
    references
        .into_iter()
        .map(|reference| state.arena.get(reference).cloned())
        .collect()
}

fn materialize_expansion_source(
    state: &mut HostState,
    source: &MirExpansionSource,
) -> NativeExecutorResult<MaterializedExpansionSource> {
    Ok(match source {
        runmat_mir::MirExpansionSource::SubscriptChain(_) => {
            return Err(NativeExecutorError::Host(
                "subscript-chain expansion must use typed sequence capture".into(),
            ));
        }
        runmat_mir::MirExpansionSource::CellContents { base, indexing } => {
            let base = materialize_operand(state, base)?;
            super::indexing::materialize_cell_expansion_source(state, base, indexing)?
        }
        runmat_mir::MirExpansionSource::ReturnedOutputs(base) => {
            MaterializedExpansionSource::ReturnedOutputs(
                runmat_runtime::call::arguments::adapt_legacy_builtin_result(materialize_operand(
                    state, base,
                )?)?,
            )
        }
        runmat_mir::MirExpansionSource::Member { base, member } => {
            MaterializedExpansionSource::Member {
                base: materialize_operand(state, base)?,
                member: member.clone(),
            }
        }
        runmat_mir::MirExpansionSource::DynamicMember { base, member } => {
            MaterializedExpansionSource::DynamicMember {
                base: materialize_operand(state, base)?,
                member: materialize_operand(state, member)?,
            }
        }
    })
}

pub(super) fn builtin(
    state: &mut HostState,
    name: &str,
    arguments: Vec<Value>,
    requested_outputs: usize,
) -> NativeExecutorResult<Vec<Value>> {
    let embedded = state.enter_embedded_call();
    let result = builtin_inner(state, name, arguments, requested_outputs, embedded.as_ref());
    if let Ok(outputs) = &result {
        state.cache_embedded_call(embedded.as_ref(), outputs);
    }
    let finished = state.finish_embedded_call(embedded.as_ref());
    match (result, finished) {
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Ok(outputs), Ok(())) => Ok(outputs),
    }
}

fn builtin_inner(
    state: &mut HostState,
    name: &str,
    arguments: Vec<Value>,
    requested_outputs: usize,
    embedded: Option<&super::state::EmbeddedOperationIdentity>,
) -> NativeExecutorResult<Vec<Value>> {
    if let Some(outputs) = super::call_suspension::take_completed(state, embedded)? {
        return Ok(outputs);
    }
    complete_call(
        state,
        embedded.cloned(),
        {
            let name = name.to_owned();
            async move {
                let value = runmat_runtime::call_builtin_async_with_outputs(
                    &name,
                    &arguments,
                    requested_outputs,
                )
                .await?;
                runmat_runtime::call::arguments::adapt_legacy_builtin_result(value)
            }
        },
        requested_outputs,
    )
}

fn complete_call(
    state: &mut HostState,
    embedded: Option<super::state::EmbeddedOperationIdentity>,
    future: impl std::future::Future<
            Output = Result<runmat_value::ValueSequence, runmat_runtime::RuntimeError>,
        > + 'static,
    requested_outputs: usize,
) -> NativeExecutorResult<Vec<Value>> {
    let runtime = state.runtime.clone();
    super::call_suspension::begin(
        state,
        embedded,
        Box::pin(async move {
            let sequence = runtime.scope(future).await?;
            normalize_sequence_outputs(sequence, requested_outputs)
        }),
    )
}

fn normalize_legacy_outputs(
    result: Value,
    requested_outputs: usize,
) -> NativeExecutorResult<Vec<Value>> {
    let sequence = runmat_runtime::call::arguments::adapt_legacy_builtin_result(result)?;
    normalize_sequence_outputs(sequence, requested_outputs)
}

fn normalize_sequence_outputs(
    sequence: runmat_value::ValueSequence,
    requested_outputs: usize,
) -> NativeExecutorResult<Vec<Value>> {
    let values = sequence.into_values();
    if requested_outputs == 0 {
        return Ok(Vec::new());
    }
    if values.len() == requested_outputs {
        Ok(values)
    } else {
        Err(NativeExecutorError::Host(format!(
            "runtime returned {} outputs for a {}-output call",
            values.len(),
            requested_outputs
        )))
    }
}
