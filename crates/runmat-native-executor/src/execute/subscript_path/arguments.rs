use runmat_mir::{MirCallArgumentRegion, MirDottedInvokeArgument};
use runmat_runtime::execution::RootedValues;
use runmat_runtime::object::indexing::ObjectIndexComponent;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::super::state::HostState;

pub(super) fn input_callsite(
    state: &HostState,
    arguments: &[MirDottedInvokeArgument],
) -> NativeExecutorResult<Option<runmat_runtime::callsite::CallsiteInfo>> {
    if arguments
        .iter()
        .any(|argument| !matches!(argument, MirDottedInvokeArgument::Call(_)))
    {
        return Ok(None);
    }
    let source = state
        .runtime
        .program_source_id(state.function.source)
        .ok_or_else(|| {
            NativeExecutorError::Host(
                "native argument source is absent from invocation binding".into(),
            )
        })?;
    runmat_runtime::callsite::CallsiteInfo::from_projected_in_context(
        Some(source),
        arguments,
        &state.runtime,
        |argument| match argument {
            MirDottedInvokeArgument::Call(region) => region.span,
            MirDottedInvokeArgument::Colon | MirDottedInvokeArgument::Contextual(_) => {
                unreachable!("non-call selectors were rejected before metadata projection")
            }
        },
    )
    .map(Some)
    .map_err(Into::into)
}

pub(super) fn materialize(
    state: &mut HostState,
    arguments: &[MirDottedInvokeArgument],
    prepared: &runmat_runtime::object::protocol::PreparedSubscriptReceiver,
) -> NativeExecutorResult<runmat_value::ExactArrayStorage<ObjectIndexComponent>> {
    let runtime = state.runtime.clone();
    let mut chunks = runmat_runtime::context::host_allocation::prepare_host_array::<RootedValues>(
        &runtime,
        arguments.len(),
    )?;
    let mut width = 0usize;
    for (component, argument) in arguments.iter().enumerate() {
        let values = match argument {
            MirDottedInvokeArgument::Colon => RootedValues::try_initialize_in_context(
                0,
                &runtime,
                |_| -> NativeExecutorResult<runmat_value::Value> {
                    unreachable!("colon has no numeric payload")
                },
            )?,
            MirDottedInvokeArgument::Contextual(region) => {
                RootedValues::try_initialize_in_context(1, &runtime, |_| {
                    state.subscript_end_receivers.push((
                        prepared.clone(),
                        component,
                        arguments.len(),
                    ));
                    let result =
                        super::super::indexing::materialize_expression_region(state, region, 1);
                    state.subscript_end_receivers.pop();
                    result
                })?
            }
            MirDottedInvokeArgument::Call(region) => materialize_call_region(state, region)?,
        };
        width = width
            .checked_add(if matches!(argument, MirDottedInvokeArgument::Colon) {
                1
            } else {
                values.values().len()
            })
            .ok_or_else(|| NativeExecutorError::Host("dotted argument width overflow".into()))?;
        chunks.try_push(values).map_err(|_| {
            NativeExecutorError::Host("dotted chunks exceeded admitted source width".into())
        })?;
    }
    let chunks = chunks.try_finish().map_err(|_| {
        NativeExecutorError::Host("dotted chunks did not fill admitted source width".into())
    })?;
    let chunks = chunks
        .try_into_owned()
        .map_err(|_| NativeExecutorError::Host("dotted chunks lost unique ownership".into()))?;
    let mut components =
        runmat_runtime::object::indexing::prepare_components_in_context(&runtime, width)?;
    for (argument, values) in arguments.iter().zip(chunks) {
        if matches!(argument, MirDottedInvokeArgument::Colon) {
            components
                .try_push(ObjectIndexComponent::Colon)
                .map_err(|_| {
                    NativeExecutorError::Host("dotted output exceeded admitted width".into())
                })?;
        } else {
            for value in values.try_into_values()? {
                components
                    .try_push(ObjectIndexComponent::Value(value))
                    .map_err(|_| {
                        NativeExecutorError::Host("dotted output exceeded admitted width".into())
                    })?;
            }
        }
    }
    components
        .try_finish()
        .map_err(|_| NativeExecutorError::Host("dotted output did not fill admitted width".into()))
}

pub(super) fn materialize_call_region(
    state: &mut HostState,
    region: &MirCallArgumentRegion,
) -> NativeExecutorResult<RootedValues> {
    let (ordinal, next_step, completed) = state.enter_contextual_region()?;
    if let Some(references) = completed {
        let runtime = state.runtime.clone();
        return RootedValues::try_initialize_in_context(
            references.as_slice().len(),
            &runtime,
            |index| state.arena.get(references.as_slice()[index]).cloned(),
        );
    }
    let result = (|| {
        super::super::indexing::execute_preparation_steps(
            state,
            ordinal,
            &region.steps,
            next_step,
        )?;
        let argument_step = region.steps.len();
        state.enter_contextual_step(ordinal, argument_step)?;
        let values = materialize_region_argument(state, &region.argument)?;
        state.finish_contextual_step(ordinal, argument_step, argument_step + 1)?;
        // The canonical rooted output remains owned by this result. Separate
        // arena references exist only to replay this exact completed region;
        // transfer those references to its existing site-retirement owner.
        state.finish_owned_contextual_values(ordinal, values.values())?;
        Ok(values)
    })();
    if result.is_err() && !matches!(&result, Err(NativeExecutorError::CallSuspended)) {
        state.abandon_contextual_region(region.steps.iter().filter_map(|step| match step {
            runmat_mir::MirExpressionStep::CaptureSequence { destination, .. } => {
                Some(*destination)
            }
            runmat_mir::MirExpressionStep::Let { .. } => None,
        }))?;
    }
    result
}

/// The existing argument materializer and Runtime expansion evaluator, with
/// their final rooted output retained across the same exact embedded site.
fn materialize_region_argument(
    state: &mut HostState,
    argument: &runmat_mir::MirCallArg,
) -> NativeExecutorResult<RootedValues> {
    use super::super::call_suspension as suspension;
    let embedded = state.enter_embedded_call()?;
    let result = (|| {
        if let Some(values) = suspension::take_completed_semantic_operation(
            state,
            embedded.as_ref(),
            suspension::SemanticOperationKind::ArgumentRegion,
        )? {
            return Ok(values);
        }
        let mut inputs =
            runmat_runtime::call::arguments::prepare_materialized_arguments_in_context(
                &state.runtime,
                1,
            )?;
        inputs
            .try_push(super::super::call::materialize_argument(state, argument)?)
            .map_err(|_| {
                NativeExecutorError::Host(
                    "argument region exceeded its admitted source width".into(),
                )
            })?;
        let inputs = inputs.try_finish().map_err(|_| {
            NativeExecutorError::Host(
                "argument region did not fill its admitted source width".into(),
            )
        })?;
        let roots = runmat_runtime::call::arguments::retain_materialized_arguments_in_context(
            &state.runtime,
            inputs.as_slice(),
        )?;
        let inputs = inputs.try_into_owned().map_err(|_| {
            NativeExecutorError::Host("argument region lost its unique admitted ledger".into())
        })?;
        let runtime = state.runtime.clone();
        suspension::begin_admitted_semantic_operation_with(
            state,
            embedded.clone(),
            suspension::SemanticOperationKind::ArgumentRegion,
            || async move {
                let _roots = roots;
                runtime
                    .scope(runmat_runtime::call::arguments::expand_arguments(
                        &runtime, inputs,
                    ))
                    .await
                    .map_err(Into::into)
            },
        )
    })();
    state.settle_embedded_result(embedded.as_ref(), result, RootedValues::values)
}
