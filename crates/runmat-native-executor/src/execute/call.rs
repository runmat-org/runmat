use std::future::Future;

use runmat_mir::{MirCall, MirCallArg, MirExpansionSource, MirSequenceLocalId};
use runmat_runtime::call::arguments::{MaterializedArgument, MaterializedExpansionSource};
use runmat_runtime::callsite::CallsiteInfo;
use runmat_runtime::execution::RootedValueLease;
use runmat_value::{OutputListStorage, Value};

use crate::{NativeExecutorError, NativeExecutorResult};

use super::operand::materialize_operand;
use super::state::HostState;

#[path = "call/dispatch.rs"]
mod dispatch;
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
) -> NativeExecutorResult<OutputListStorage> {
    let embedded = state.enter_embedded_call()?;
    let result = evaluate_inner(state, call, requested_outputs, embedded.as_ref());
    state.settle_embedded_result(embedded.as_ref(), result, OutputListStorage::as_slice)
}

fn evaluate_inner(
    state: &mut HostState,
    call: &MirCall,
    requested_outputs: usize,
    embedded: Option<&super::state::EmbeddedOperationIdentity>,
) -> NativeExecutorResult<OutputListStorage> {
    if let Some(outputs) = super::call_suspension::take_completed(state, embedded)? {
        return Ok(OutputListStorage::unbounded(outputs));
    }
    let arguments = materialize_arguments(state, &call.args)?;
    let source_id = state
        .runtime
        .program_source_id(state.function.source)
        .ok_or_else(|| {
            NativeExecutorError::Host("native source is absent from invocation binding".into())
        })?;
    let input_callsite =
        CallsiteInfo::from_borrowed_in_context(Some(source_id), &call.arg_spans, &state.runtime)?;
    dispatch::prepared(
        state,
        call,
        requested_outputs,
        embedded,
        arguments,
        input_callsite,
    )
}

pub(super) fn materialize_arguments(
    state: &mut HostState,
    arguments: &[MirCallArg],
) -> NativeExecutorResult<Vec<Value>> {
    let embedded = state.enter_embedded_call()?;
    let result = materialize_arguments_inner(state, arguments, embedded.as_ref());
    state.settle_embedded_result(embedded.as_ref(), result, Vec::as_slice)
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
    let mut materialized =
        runmat_runtime::call::arguments::prepare_materialized_arguments_in_context(
            &state.runtime,
            arguments.len(),
        )?;
    for argument in arguments {
        materialized
            .try_push(materialize_argument(state, argument)?)
            .map_err(|_| {
                NativeExecutorError::Host("native arguments exceeded admitted source width".into())
            })?;
    }
    let materialized = materialized.try_finish().map_err(|_| {
        NativeExecutorError::Host(
            "native arguments did not fill their admitted source width".into(),
        )
    })?;
    let roots = runmat_runtime::call::arguments::retain_materialized_arguments_in_context(
        &state.runtime,
        materialized.as_slice(),
    )?;
    let materialized = materialized.try_into_owned().map_err(|_| {
        NativeExecutorError::Host("native prepared arguments lost unique source ownership".into())
    })?;
    let runtime = state.runtime.clone();
    super::call_suspension::begin_expanded_call_arguments(state, embedded.cloned(), || async move {
        let _roots = roots;
        runtime
            .scope(runmat_runtime::call::arguments::expand_arguments(
                &runtime,
                materialized,
            ))
            .await
    })
}

pub(super) fn materialize_argument(
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
        MirCallArg::CapturedSequence(sequence) => Ok(MaterializedArgument::Sequence(
            state.take_captured_sequence_owner(*sequence)?,
        )),
    }
}

pub(super) fn capture_sequence(
    state: &mut HostState,
    destination: MirSequenceLocalId,
    source: &MirExpansionSource,
) -> NativeExecutorResult<()> {
    // Table growth is admitted before expansion callbacks/effects. The actual
    // output's own backing/roots travel intact; there is no ref Vec mirror.
    state.prepare_captured_sequence_slot(destination)?;
    let sequence = if let MirExpansionSource::SubscriptChain(chain) = source {
        // The legacy subscript ABI returns a Vec. Its separate rooting adapter
        // is unchanged here; ordinary sequence callbacks already own roots.
        let values = runmat_runtime::execution::RootedValues::new(super::subscript_path::read(
            state, chain, 1,
        )?)
        .map_err(|error| NativeExecutorError::from(error.into_runtime_error()))?;
        values.with_owned_output_storage(|values| {
            let sequence = runmat_runtime::sequence::ValueSequence::from_output_storage(values)
                .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?;
            runmat_runtime::execution::RootedValueSequence::in_context(sequence, &state.runtime)
                .map_err(NativeExecutorError::from)
        })?
    } else {
        let stage = super::call_suspension::SequenceReadStage::Capture { destination };
        let sequence = if let Some(sequence) =
            super::call_suspension::take_completed_sequence_read(state, stage)?
        {
            sequence
        } else {
            let source = materialize_expansion_source(state, source)?;
            let roots = root_expansion_source(&source, &state.runtime)?;
            let runtime = state.runtime.clone();
            super::call_suspension::begin_sequence_read(state, stage, || async move {
                let sequence = runtime
                    .scope(runmat_runtime::call::arguments::materialize_expansion(
                        &runtime, source,
                    ))
                    .await?;
                drop(roots);
                Ok(sequence)
            })?
        };
        // ExpandAll is already the materializer's canonical sequence result:
        // retain its inline/shared/unique backing and original registration.
        sequence
    };
    state.publish_captured_sequence(destination, sequence)
}

fn root_expansion_source(
    source: &MaterializedExpansionSource,
    runtime: &runmat_runtime::context::RuntimeContext,
) -> NativeExecutorResult<RootedValueLease> {
    let roots = match source {
        MaterializedExpansionSource::CellContents { base, indices, .. } => {
            RootedValueLease::for_values_iterator_in_context(
                std::iter::once(base).chain(indices.iter()),
                runtime,
            )
        }
        MaterializedExpansionSource::ReturnedOutputs(_) => {
            // This canonical source already owns its direct roots.
            RootedValueLease::for_values_iterator_in_context(std::iter::empty::<&Value>(), runtime)
        }
        MaterializedExpansionSource::Member { base, .. } => {
            RootedValueLease::for_values_iterator_in_context(std::iter::once(base), runtime)
        }
        MaterializedExpansionSource::DynamicMember { base, member } => {
            RootedValueLease::for_values_iterator_in_context([base, member].into_iter(), runtime)
        }
    };
    roots.map_err(NativeExecutorError::from)
}

pub(super) fn take_captured_sequence(
    state: &mut HostState,
    sequence: MirSequenceLocalId,
) -> NativeExecutorResult<runmat_runtime::execution::RootedValuesIntoIter> {
    state
        .take_captured_sequence_owner(sequence)?
        .try_into_values()
        .map_err(Into::into)
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
            let base = if super::indexing::indexing_requires_retention(indexing) {
                super::operand::materialize_retained_operand(state, base)?.1
            } else {
                materialize_operand(state, base)?
            };
            super::indexing::materialize_cell_expansion_source(state, base, indexing)?
        }
        runmat_mir::MirExpansionSource::ReturnedOutputs(base) => {
            let sequence = runmat_runtime::call::arguments::adapt_legacy_builtin_result(
                materialize_operand(state, base)?,
            )?;
            MaterializedExpansionSource::ReturnedOutputs(
                runmat_runtime::execution::RootedValueSequence::in_context(
                    sequence,
                    &state.runtime,
                )?,
            )
        }
        runmat_mir::MirExpansionSource::Member { base, member } => {
            MaterializedExpansionSource::Member {
                base: materialize_operand(state, base)?,
                member: runmat_runtime::call::argument_text::copy_str(
                    &member.0,
                    Some(&state.runtime),
                )?,
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
) -> NativeExecutorResult<OutputListStorage> {
    let embedded = state.enter_embedded_call()?;
    let result = builtin_inner(state, name, arguments, requested_outputs, embedded.as_ref());
    state.settle_embedded_result(embedded.as_ref(), result, OutputListStorage::as_slice)
}

fn builtin_inner(
    state: &mut HostState,
    name: &str,
    arguments: Vec<Value>,
    requested_outputs: usize,
    embedded: Option<&super::state::EmbeddedOperationIdentity>,
) -> NativeExecutorResult<OutputListStorage> {
    if let Some(outputs) = super::call_suspension::take_completed(state, embedded)? {
        return Ok(OutputListStorage::unbounded(outputs));
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

async fn with_output_count<T>(
    count: usize,
    future: impl Future<Output = Result<T, runmat_runtime::RuntimeError>>,
) -> Result<T, runmat_runtime::RuntimeError> {
    let mut future = std::pin::pin!(future);
    std::future::poll_fn(|context| {
        let _outputs = match runmat_runtime::output_context::push_output_count(count) {
            Ok(guard) => guard,
            Err(error) => return std::task::Poll::Ready(Err(error)),
        };
        future.as_mut().poll(context)
    })
    .await
}

async fn with_class_access_context<T>(
    class_context: Option<runmat_types::ClassIdentity>,
    future: impl std::future::Future<Output = Result<T, runmat_runtime::RuntimeError>>,
) -> Result<T, runmat_runtime::RuntimeError> {
    let mut future = std::pin::pin!(future);
    std::future::poll_fn(|context| {
        let access = class_context
            .clone()
            .map(|class_name| runmat_runtime::push_class_access_context(Some(class_name)))
            .transpose();
        let _access = match access {
            Ok(guard) => guard,
            Err(error) => return std::task::Poll::Ready(Err(error)),
        };
        future.as_mut().poll(context)
    })
    .await
}

pub(super) fn normalize_sequence_outputs(
    sequence: runmat_value::ValueSequence,
    requested_outputs: usize,
) -> NativeExecutorResult<OutputListStorage> {
    let values = sequence.into_output_storage();
    validate_output_arity(values.len(), requested_outputs)?;
    if requested_outputs == 0 {
        return Ok(OutputListStorage::empty());
    }
    Ok(values)
}

/// Exact ordinary call arity, not comma-separated prefix selection.
pub(super) fn validate_output_arity(actual: usize, requested: usize) -> NativeExecutorResult<()> {
    if requested == 0 || actual == requested {
        return Ok(());
    }
    Err(NativeExecutorError::Host(format!(
        "runtime returned {actual} outputs for a {requested}-output call"
    )))
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
) -> NativeExecutorResult<OutputListStorage> {
    let runtime = state.runtime.clone();
    super::call_suspension::begin_rooted_sequence(
        state,
        embedded,
        move || async move { runtime.scope(future).await },
        requested_outputs,
    )
    .map(OutputListStorage::unbounded)
}
