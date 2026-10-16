use std::future::Future;

use runmat_mir::{MirCall, MirCallArg, MirCallee, MirExpansionSource, MirSequenceLocalId};
use runmat_runtime::call::arguments::{MaterializedArgument, MaterializedExpansionSource};
use runmat_runtime::call::descriptor::{CallableCallKind, CallableDescriptor};
use runmat_runtime::callsite::CallsiteInfo;
use runmat_runtime::execution::RootedValueLease;
use runmat_runtime::sequence::ResolveValueSequence;
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::operand::materialize_operand;
use super::state::HostState;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "call/poll_scope_tests.rs"]
mod poll_scope_tests;
#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "call/sequence_capture_tests.rs"]
mod sequence_capture_tests;

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
    let source_id = state
        .runtime
        .program_source_id(state.function.source)
        .ok_or_else(|| {
            NativeExecutorError::Host("native source is absent from invocation binding".into())
        })?;
    let input_callsite = CallsiteInfo {
        source_id: Some(source_id),
        arg_spans: call.arg_spans.clone(),
    };
    if matches!(
        call.syntax,
        runmat_types::CallSyntax::Method | runmat_types::CallSyntax::DottedInvoke
    ) {
        if let MirCallee::Static(identity) = &call.callee {
            if !matches!(
                identity,
                runmat_types::CallableIdentity::BoundFunction(_)
                    | runmat_types::CallableIdentity::ExternalFunction { .. }
            ) {
                if arguments.is_empty() {
                    return Err(NativeExecutorError::Host(
                        "method/member-index call requires a base receiver".into(),
                    ));
                }
                let base = arguments.remove(0);
                let caller = state.function.name.clone();
                let class_context =
                    runmat_runtime::class_registry::class_context_for_function(&caller);
                let runtime = state.runtime.clone();
                let value = complete_rooted_call(
                    state,
                    embedded.cloned(),
                    {
                        let identity = identity.clone();
                        let fallback_policy = call.fallback_policy;
                        with_output_count(
                            requested_outputs,
                            with_class_access_context(class_context, async move {
                                runmat_runtime::object::dispatch::call_method_or_member_index_with_outputs_in_context(
                                &runtime,
                                base,
                                identity,
                                arguments,
                                requested_outputs,
                                (!caller.is_empty()).then_some(caller.as_str()),
                                fallback_policy,
                            )
                            .await
                            }),
                        )
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
                    let lexical_call = runmat_runtime::call::lexical::LexicalCall {
                        function: function.0 as usize,
                        captures,
                        environment: state.lexical_environment.clone(),
                        arguments,
                        requested_outputs,
                    };
                    let _call_roots =
                        runmat_runtime::execution::RootedValueLease::for_traced(&lexical_call)
                            .map_err(|error| error.into_runtime_error())?;
                    let result = super::sync::complete(
                        &runtime,
                        runmat_runtime::callsite::with_function_input_callsite(
                            Some(input_callsite.clone()),
                            invoker(Some(runtime.clone()), lexical_call),
                        ),
                        "nested lexical call",
                    )
                    .finish(state)?;
                    return result.with_owned(|result| {
                        state.apply_lexical_captures(result.captures)?;
                        normalize_sequence_outputs(result.outputs, requested_outputs)?
                            .into_iter()
                            .map(|value| {
                                let reference = state.arena.insert(value);
                                state.arena.get(reference).cloned()
                            })
                            .collect()
                    });
                }
            }
            let descriptor = CallableDescriptor::resolved(
                identity.clone(),
                arguments,
                requested_outputs,
                call.fallback_policy,
                CallableCallKind::Direct,
            )
            .with_function_input_callsite(input_callsite);
            let runtime = state.runtime.clone();
            complete_rooted_call(
                state,
                embedded.cloned(),
                async move {
                    runmat_runtime::call::descriptor::execute_callable_descriptor_in_context(
                        &runtime, descriptor,
                    )
                    .await
                },
                requested_outputs,
            )
        }
        MirCallee::Dynamic(operand) => {
            let target = materialize_operand(state, operand)?;
            let runtime = state.runtime.clone();
            complete_rooted_call(
                state,
                embedded.cloned(),
                async move {
                    runmat_runtime::call_feval_sequence_async_with_outputs_and_callsite_in_context(
                        &runtime,
                        target,
                        &arguments,
                        requested_outputs,
                        Some(input_callsite),
                    )
                    .await
                },
                requested_outputs,
            )
        }
        MirCallee::SuperConstructor {
            current_class,
            super_class,
        } => {
            let current_class = current_class.clone();
            let super_class = super_class.clone();
            let runtime = state.runtime.clone();
            complete_rooted_call(
                state,
                embedded.cloned(),
                with_output_count(requested_outputs, async move {
                    runmat_runtime::call_super_constructor(
                        &runtime,
                        current_class,
                        super_class,
                        arguments,
                    )
                    .await?
                    .try_map_owned(
                        runmat_runtime::call::arguments::adapt_legacy_builtin_result,
                        |error| error.into_runtime_error(),
                    )
                }),
                requested_outputs,
            )
        }
        MirCallee::SuperMethod {
            current_class,
            super_class,
            method,
        } => {
            let current_class = current_class.clone();
            let super_class = super_class.clone();
            let method = method.clone();
            let runtime = state.runtime.clone();
            complete_rooted_call(
                state,
                embedded.cloned(),
                with_output_count(requested_outputs, async move {
                    runmat_runtime::call_super_method_in_context(
                        &runtime,
                        current_class,
                        super_class,
                        method,
                        arguments,
                    )
                    .await?
                    .try_map_owned(
                        runmat_runtime::call::arguments::adapt_legacy_builtin_result,
                        |error| error.into_runtime_error(),
                    )
                }),
                requested_outputs,
            )
        }
    }
}

fn local_program_function(
    identity: &runmat_types::CallableIdentity,
) -> NativeExecutorResult<Option<runmat_types::ProgramFunctionId>> {
    let function = match identity {
        runmat_types::CallableIdentity::BoundFunction(function)
        | runmat_types::CallableIdentity::AnonymousFunction(function) => function.0,
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
    let embedded = state.enter_embedded_call();
    let result = materialize_arguments_inner(state, arguments, embedded.as_ref());
    if let Ok(values) = &result {
        state.cache_embedded_call(embedded.as_ref(), values);
    }
    let finished = state.finish_embedded_call(embedded.as_ref());
    match (result, finished) {
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Ok(values), Ok(())) => Ok(values),
    }
}

fn materialize_arguments_inner(
    state: &mut HostState,
    arguments: &[MirCallArg],
    embedded: Option<&super::state::EmbeddedOperationIdentity>,
) -> NativeExecutorResult<Vec<Value>> {
    if let Some(values) =
        super::call_suspension::take_completed_expanded_call_arguments(state, embedded)?
    {
        return Ok(values);
    }
    if arguments
        .iter()
        .all(|argument| matches!(argument, MirCallArg::Single(_)))
    {
        // Single operands are already arena-rooted and need no asynchronous
        // expansion. This structural path keeps ordinary calls synchronous.
        let values = arguments
            .iter()
            .map(|argument| match argument {
                MirCallArg::Single(operand) => materialize_operand(state, operand),
                _ => unreachable!("all arguments were checked as Single"),
            })
            .collect::<NativeExecutorResult<Vec<_>>>()?;
        state.reconcile_external_workspace()?;
        return Ok(values);
    }
    let materialized = arguments
        .iter()
        .map(|argument| materialize_argument(state, argument))
        .collect::<NativeExecutorResult<Vec<_>>>()?;
    let runtime = state.runtime.clone();
    super::call_suspension::begin_expanded_call_arguments(
        state,
        embedded.cloned(),
        Box::pin(async move {
            runtime
                .scope(runmat_runtime::call::arguments::expand_arguments(
                    &runtime,
                    materialized,
                ))
                .await
        }),
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
        runmat_runtime::execution::RootedValues::new(super::subscript_path::read(state, chain, 1)?)
            .map_err(|error| NativeExecutorError::from(error.into_runtime_error()))?
    } else {
        let stage = super::call_suspension::SequenceReadStage::Capture { destination };
        let sequence = if let Some(sequence) =
            super::call_suspension::take_completed_sequence_read(state, stage)?
        {
            sequence
        } else {
            let source = materialize_expansion_source(state, source)?;
            let roots = root_expansion_source(&source)?;
            let runtime = state.runtime.clone();
            super::call_suspension::begin_sequence_read(
                state,
                stage,
                Box::pin(async move {
                    let sequence = runtime
                        .scope(runmat_runtime::call::arguments::materialize_expansion(
                            &runtime, source,
                        ))
                        .await?;
                    drop(roots);
                    Ok(sequence)
                }),
            )?
        };
        sequence.with_owned_sequence(|sequence| {
            let values = sequence.resolve(
                runmat_types::SequenceUse::ExpandAll,
                runmat_runtime::sequence::SequenceResolutionContext::default(),
            )?;
            runmat_runtime::execution::RootedValues::new(values)
                .map_err(|error| error.into_runtime_error())
        })?
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
    let references = values.with_owned_values(|values| {
        values
            .into_iter()
            .map(|value| state.arena.insert(value))
            .collect()
    });
    state.captured_sequences.insert(destination, references);
    Ok(())
}

fn root_expansion_source(
    source: &MaterializedExpansionSource,
) -> NativeExecutorResult<RootedValueLease> {
    let roots = match source {
        MaterializedExpansionSource::CellContents { base, indices, .. } => {
            RootedValueLease::for_values(std::iter::once(base).chain(indices.iter()))
        }
        MaterializedExpansionSource::ReturnedOutputs(sequence) => {
            RootedValueLease::for_sequence(sequence)
        }
        MaterializedExpansionSource::Member { base, .. } => {
            RootedValueLease::for_values(std::iter::once(base))
        }
        MaterializedExpansionSource::DynamicMember { base, member } => {
            RootedValueLease::for_values([base, member].into_iter())
        }
    };
    roots.map_err(|error| NativeExecutorError::from(error.into_runtime_error()))
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
    complete_rooted_call(
        state,
        embedded.cloned(),
        {
            let name = name.to_owned();
            let runtime = state.runtime.clone();
            async move {
                runmat_runtime::call_builtin_sequence_async_with_outputs_in_context(
                    &runtime,
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

async fn with_output_count<T>(count: usize, future: impl Future<Output = T>) -> T {
    let mut future = std::pin::pin!(future);
    std::future::poll_fn(|context| {
        let _outputs = runmat_runtime::output_context::push_output_count(count);
        future.as_mut().poll(context)
    })
    .await
}

async fn with_class_access_context<T>(
    class_context: Option<runmat_types::ClassIdentity>,
    future: impl std::future::Future<Output = T>,
) -> T {
    let mut future = std::pin::pin!(future);
    std::future::poll_fn(|context| {
        let _access = class_context
            .clone()
            .map(|class_name| runmat_runtime::push_class_access_context(Some(class_name)));
        future.as_mut().poll(context)
    })
    .await
}

pub(super) fn normalize_sequence_outputs(
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

fn complete_rooted_call(
    state: &mut HostState,
    embedded: Option<super::state::EmbeddedOperationIdentity>,
    future: impl std::future::Future<
            Output = Result<
                runmat_runtime::execution::RootedValueSequence,
                runmat_runtime::RuntimeError,
            >,
        > + 'static,
    requested_outputs: usize,
) -> NativeExecutorResult<Vec<Value>> {
    let runtime = state.runtime.clone();
    super::call_suspension::begin_rooted_sequence(
        state,
        embedded,
        Box::pin(async move { runtime.scope(future).await }),
        requested_outputs,
    )
}
