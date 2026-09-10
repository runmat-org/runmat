use runmat_mir::{MirSubscriptChain, MirSubscriptStep};
use runmat_runtime::object::indexing::{ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath};
use runmat_value::Value;

use super::operand::materialize_operand;
use super::state::HostState;
use crate::memory::ScopedValueRoots;
use crate::{NativeExecutorError, NativeExecutorResult};
mod access;
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
    let embedded = state.enter_embedded_call();
    let result = read_inner(state, chain, requested_outputs, embedded.as_ref());
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

fn read_inner(
    state: &mut HostState,
    chain: &MirSubscriptChain,
    requested_outputs: usize,
    embedded: Option<&super::state::EmbeddedOperationIdentity>,
) -> NativeExecutorResult<Vec<Value>> {
    if let Some(outputs) = super::call_suspension::take_completed_subscript_path(state, embedded)? {
        return Ok(outputs);
    }
    let mut root = materialize_operand(state, &chain.root)?;
    let access = frame_access(state)?;
    let mut steps = Vec::new();
    for (step_index, step) in chain.steps.iter().enumerate() {
        match step {
            MirSubscriptStep::Member(member) => {
                steps.push(ObjectSubscript::member(member.clone()));
            }
            MirSubscriptStep::DynamicMember(member) => {
                let member =
                    String::try_from(&materialize_operand(state, member)?).map_err(|error| {
                        NativeExecutorError::from(runmat_runtime::runtime_error::semantic_error(
                            "DynamicFieldName",
                            error,
                        ))
                    })?;
                steps.push(ObjectSubscript::member(member));
            }
            MirSubscriptStep::Index(indexing) => {
                let materialized =
                    materialize_selectors(state, step_index, &root, &steps, indexing, &access)?;
                if let Some(receiver) = materialized.prepared_receiver {
                    root = receiver;
                    steps.clear();
                }
                let selector = ObjectIndexSelector::IndexValues {
                    components: materialized.components,
                };
                steps.push(match indexing.kind {
                    runmat_hir::IndexKind::Paren => ObjectSubscript::parentheses(selector),
                    runmat_hir::IndexKind::Brace => ObjectSubscript::braces(selector),
                });
            }
            MirSubscriptStep::DottedInvoke { member, indexing } => {
                let materialized =
                    materialize_selectors(state, step_index, &root, &steps, indexing, &access)?;
                if let Some(receiver) = materialized.prepared_receiver {
                    root = receiver;
                    steps.clear();
                }
                steps.extend(ObjectSubscript::dotted_invoke_components(
                    member.clone(),
                    materialized.components,
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
    super::call_suspension::begin(
        state,
        embedded.cloned(),
        Box::pin(async move {
            let _roots = roots;
            let sequence = runtime
                .scope(
                    runmat_runtime::object::protocol::read_subscript_path_sequence_with_access(
                        root, path, request,
                    ),
                )
                .await?;
            sequence
                .resolve(selection, sequence_context)
                .map_err(Into::into)
        }),
    )
}
