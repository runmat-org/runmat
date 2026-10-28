use runmat_mir::{MirSubscriptChain, MirSubscriptStep};
use runmat_runtime::object::indexing::{ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath};
use runmat_value::Value;

use super::operand::{materialize_operand, materialize_retained_operand};
use super::state::HostState;
use crate::{NativeExecutorError, NativeExecutorResult};
use runmat_runtime::execution::RootedValueLease;
mod access;
mod arguments;
mod end;
mod roots;
mod selectors;
pub(super) use access::{frame_access, frame_access_runtime};
pub(super) use end::resolve_end;
use selectors::materialize_selectors;

/// Reuse the ordered read-selector region owner, but admit the target context
/// independently. No fixed-width mask or new expansion evaluator is involved.
pub(super) fn materialize_target_selectors(
    state: &mut HostState,
    base: &Value,
    indexing: &runmat_mir::MirIndexing,
) -> NativeExecutorResult<
    runmat_value::ExactArrayStorage<runmat_runtime::object::indexing::ObjectIndexComponent>,
> {
    indexing.validate_target_selectors().map_err(|error| {
        NativeExecutorError::Host(format!("invalid MIR target selectors: {error}"))
    })?;
    let access = frame_access(state)?;
    Ok(materialize_selectors(state, 0, base, &[], indexing, &access, true)?.components)
}

pub(super) fn read(
    state: &mut HostState,
    chain: &MirSubscriptChain,
    requested_outputs: usize,
) -> NativeExecutorResult<Vec<Value>> {
    chain.validate().map_err(|error| {
        NativeExecutorError::Host(format!("invalid MIR subscript chain: {error}"))
    })?;
    let embedded = state.enter_embedded_call()?;
    let result = read_inner(state, chain, requested_outputs, embedded.as_ref());
    state.settle_embedded_result(embedded.as_ref(), result, Vec::as_slice)
}

fn read_inner(
    state: &mut HostState,
    chain: &MirSubscriptChain,
    requested_outputs: usize,
    embedded: Option<&super::state::EmbeddedOperationIdentity>,
) -> NativeExecutorResult<Vec<Value>> {
    if let Some(outputs) = super::call_suspension::take_completed_subscript_path(state, embedded)? {
        return Ok(outputs);
    }
    let retains_inputs = chain.steps.iter().any(|step| match step {
        MirSubscriptStep::PreparedDynamicMember(_) => true,
        MirSubscriptStep::Index(indexing) => indexing.components.iter().any(|component| {
            matches!(
                component,
                runmat_mir::MirIndexComponent::ContextualExpr(_)
                    | runmat_mir::MirIndexComponent::PreparedExpr(_)
                    | runmat_mir::MirIndexComponent::Expansion(_)
            )
        }),
        MirSubscriptStep::DottedInvoke { arguments, .. } => arguments
            .iter()
            .any(|argument| !matches!(argument, runmat_mir::MirDottedInvokeArgument::Colon)),
        MirSubscriptStep::Member(_) | MirSubscriptStep::DynamicMember(_) => false,
    });
    let mut root = if retains_inputs {
        materialize_retained_operand(state, &chain.root)?.1
    } else {
        materialize_operand(state, &chain.root)?
    };
    let access = frame_access(state)?;
    let runtime = state.runtime.clone();
    let mut steps = prepare_segment(&runtime, &chain.steps, 0)?;
    for (step_index, step) in chain.steps.iter().enumerate() {
        match step {
            MirSubscriptStep::Member(member) => {
                append_steps(
                    &mut steps,
                    [ObjectSubscript::member_retained(
                        runmat_runtime::call::argument_text::copy_str(&member.0, Some(&runtime))?,
                    )],
                )?;
            }
            MirSubscriptStep::DynamicMember(member) => {
                let member_value = if retains_inputs {
                    materialize_retained_operand(state, member)?.1
                } else {
                    materialize_operand(state, member)?
                };
                append_steps(
                    &mut steps,
                    [retained_dynamic_member(&runtime, &member_value)?],
                )?;
            }
            MirSubscriptStep::PreparedDynamicMember(region) => {
                let value = super::indexing::materialize_prepared_expression_region(state, region)?;
                append_steps(&mut steps, [retained_dynamic_member(&runtime, &value)?])?;
            }
            MirSubscriptStep::Index(indexing) => {
                // Keep the original prefix alive through its preparation, but
                // admit the replacement segment before any receiver/End effect.
                let replacement = if starts_segment(step) && !steps.initialized().is_empty() {
                    Some(prepare_segment(&runtime, &chain.steps, step_index)?)
                } else {
                    None
                };
                let materialized = materialize_selectors(
                    state,
                    step_index,
                    &root,
                    steps.initialized(),
                    indexing,
                    &access,
                    retains_inputs,
                )?;
                if let Some(receiver) = materialized.prepared_receiver {
                    root = receiver;
                    if let Some(replacement) = replacement {
                        steps = replacement;
                    }
                }
                let selector = ObjectIndexSelector::IndexValues {
                    components: materialized.components,
                };
                append_steps(
                    &mut steps,
                    [match indexing.kind {
                        runmat_types::IndexKind::Paren => ObjectSubscript::parentheses(selector),
                        runmat_types::IndexKind::Brace => ObjectSubscript::braces(selector),
                    }
                    .with_input_callsite(super::indexing::index_callsite(state, indexing)?)],
                )?;
            }
            MirSubscriptStep::DottedInvoke {
                member,
                arguments,
                semantic_fallback,
            } => {
                // Resolve the preceding path before evaluating this call's
                // arguments. The cached prepared receiver survives suspension
                // and prevents replay of earlier invocation effects.
                let prefix = if steps.initialized().is_empty() {
                    None
                } else {
                    let replacement = prepare_segment(&runtime, &chain.steps, step_index)?;
                    let prefix = std::mem::replace(&mut steps, replacement);
                    Some(ObjectSubscriptPath::from_storage(
                        prefix.finish_initialized(),
                    )?)
                };
                let prepared = selectors::prepare_dotted_receiver(
                    state,
                    step_index,
                    root,
                    prefix,
                    access.clone(),
                )?;
                root = prepared.receiver().clone();
                let components = arguments::materialize(state, arguments, &prepared)?;
                let input_callsite = arguments::input_callsite(state, arguments)?;
                let [member_step, paren_step] =
                    ObjectSubscript::dotted_invoke_retained_with_fallback(
                        runmat_runtime::call::argument_text::copy_str(
                            &member.0,
                            Some(&state.runtime),
                        )?,
                        components,
                        semantic_fallback.clone(),
                    );
                append_steps(
                    &mut steps,
                    [member_step, paren_step.with_input_callsite(input_callsite)],
                )?;
            }
        }
    }
    let path = ObjectSubscriptPath::from_storage(steps.finish_initialized())?;
    let sequence_context = match chain.sequence_use {
        runmat_types::SequenceUse::SelectCurrentFunctionOutputs => {
            runmat_runtime::sequence::SequenceResolutionContext::current_function_outputs(
                requested_outputs,
            )
        }
        runmat_types::SequenceUse::SelectDestinationCardinality => {
            runmat_runtime::sequence::SequenceResolutionContext::destination_cardinality(
                requested_outputs,
            )
        }
        _ => Default::default(),
    };
    let request = runmat_runtime::object::protocol::SubscriptReadRequest {
        sequence_use: chain.sequence_use,
        sequence_context,
        indexing_context: chain.context,
        access,
    };
    let runtime = state.runtime.clone();
    let selection = chain.sequence_use;
    let roots = RootedValueLease::for_values_iterator_in_context(
        roots::subscript_values(&root, Some(&path)),
        &runtime,
    )?;
    super::call_suspension::begin_selected_sequence(state, embedded.cloned(), move || async move {
        let _roots = roots;
        let sequence = runtime
                .scope(
                    runmat_runtime::object::protocol::read_subscript_path_sequence_with_access_in_context(
                        &runtime, root, path, request,
                    ),
                )
                .await?;
        sequence.resolve_sequence(selection, sequence_context)
    })
}

type PathBuilder = runmat_runtime::context::host_allocation::PreparedHostArray<ObjectSubscript>;

fn starts_segment(step: &MirSubscriptStep) -> bool {
    match step {
        MirSubscriptStep::DottedInvoke { .. } => true,
        MirSubscriptStep::Index(indexing) => indexing.components.iter().any(|component| {
            matches!(component, runmat_mir::MirIndexComponent::ContextualExpr(region)
                if region.contains_contextual_end())
        }),
        _ => false,
    }
}

/// Each MIR step is counted once in its segment. The boundary step starts the
/// next segment because its predecessor prefix is resolved before it appends
/// its own index or dotted pair.
fn prepare_segment(
    runtime: &runmat_runtime::context::RuntimeContext,
    source: &[MirSubscriptStep],
    start: usize,
) -> NativeExecutorResult<PathBuilder> {
    let mut width = 0usize;
    for (offset, step) in source[start..].iter().enumerate() {
        if offset != 0 && starts_segment(step) {
            break;
        }
        width = width
            .checked_add(if matches!(step, MirSubscriptStep::DottedInvoke { .. }) {
                2
            } else {
                1
            })
            .ok_or_else(|| NativeExecutorError::Host("subscript path width overflow".into()))?;
    }
    runmat_runtime::object::indexing::prepare_subscripts_in_context(runtime, width)
        .map_err(Into::into)
}

fn append_steps<const N: usize>(
    steps: &mut PathBuilder,
    appended: [ObjectSubscript; N],
) -> NativeExecutorResult<()> {
    for step in appended {
        steps.try_push(step).map_err(|_| {
            NativeExecutorError::Host(
                "subscript path exceeded its statically admitted segment width".into(),
            )
        })?;
    }
    Ok(())
}

fn retained_dynamic_member(
    runtime: &runmat_runtime::context::RuntimeContext,
    value: &Value,
) -> NativeExecutorResult<ObjectSubscript> {
    let runmat_runtime::sequence::AssignmentStepSpec::Member(name) =
        runmat_runtime::sequence::AssignmentStepSpec::dynamic_member_in_context(runtime, value)?
    else {
        unreachable!("dynamic member factory")
    };
    Ok(ObjectSubscript::member_retained(name))
}
