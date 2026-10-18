use runmat_mir::{MirSubscriptChain, MirSubscriptStep};
use runmat_runtime::execution::RootedValues;
use runmat_runtime::object::indexing::{ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath};
use runmat_runtime::sequence::ResolveValueSequence;
use runmat_value::Value;

use super::operand::{materialize_operand, materialize_retained_operand};
use super::state::HostState;
use crate::memory::ScopedValueRoots;
use crate::{NativeExecutorError, NativeExecutorResult};
mod access;
mod arguments;
mod end;
mod roots;
mod selectors;
use access::frame_access;
pub(super) use end::resolve_end;
use selectors::materialize_selectors;

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
    let mut steps = Vec::new();
    for (step_index, step) in chain.steps.iter().enumerate() {
        match step {
            MirSubscriptStep::Member(member) => {
                steps.push(ObjectSubscript::member(member.clone()));
            }
            MirSubscriptStep::DynamicMember(member) => {
                let member_value = if retains_inputs {
                    materialize_retained_operand(state, member)?.1
                } else {
                    materialize_operand(state, member)?
                };
                let member = String::try_from(&member_value).map_err(|error| {
                    NativeExecutorError::from(runmat_runtime::runtime_error::semantic_error(
                        "DynamicFieldName",
                        error,
                    ))
                })?;
                steps.push(ObjectSubscript::member(member));
            }
            MirSubscriptStep::PreparedDynamicMember(region) => {
                let value = super::indexing::materialize_prepared_expression_region(state, region)?;
                let member = String::try_from(&value).map_err(|error| {
                    NativeExecutorError::from(runmat_runtime::runtime_error::semantic_error(
                        "DynamicFieldName",
                        error,
                    ))
                })?;
                steps.push(ObjectSubscript::member(member));
            }
            MirSubscriptStep::Index(indexing) => {
                let materialized = materialize_selectors(
                    state,
                    step_index,
                    &root,
                    &steps,
                    indexing,
                    &access,
                    retains_inputs,
                )?;
                if let Some(receiver) = materialized.prepared_receiver {
                    root = receiver;
                    steps.clear();
                }
                let selector = ObjectIndexSelector::IndexValues {
                    components: materialized.components,
                };
                steps.push(match indexing.kind {
                    runmat_types::IndexKind::Paren => ObjectSubscript::parentheses(selector),
                    runmat_types::IndexKind::Brace => ObjectSubscript::braces(selector),
                });
            }
            MirSubscriptStep::DottedInvoke {
                member,
                arguments,
                semantic_fallback,
            } => {
                // Resolve the preceding path before evaluating this call's
                // arguments. The cached prepared receiver survives suspension
                // and prevents replay of earlier invocation effects.
                let prefix = if steps.is_empty() {
                    None
                } else {
                    Some(ObjectSubscriptPath::new(std::mem::take(&mut steps))?)
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
                steps.extend(ObjectSubscript::dotted_invoke_components_with_fallback(
                    member.clone(),
                    components,
                    semantic_fallback.clone(),
                ));
            }
        }
    }
    let path = ObjectSubscriptPath::new(steps)?;
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
    let roots = ScopedValueRoots::register(
        roots::subscript_roots(&root, Some(&path)),
        "native_pending_subscript_path",
    )?;
    super::call_suspension::begin_rooted_arguments(
        state,
        embedded.cloned(),
        Box::pin(async move {
            let _roots = roots;
            let sequence = runtime
                .scope(
                    runmat_runtime::object::protocol::read_subscript_path_sequence_with_access_in_context(
                        &runtime, root, path, request,
                    ),
                )
                .await?;
            sequence.with_owned_sequence(|sequence| {
                let values = sequence.resolve(selection, sequence_context)?;
                RootedValues::new(values).map_err(|error| error.into_runtime_error())
            })
        }),
    )
}
