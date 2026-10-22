use runmat_mir::{MirCallArgumentRegion, MirDottedInvokeArgument};
use runmat_runtime::object::indexing::ObjectIndexComponent;
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::super::state::HostState;

pub(super) fn materialize(
    state: &mut HostState,
    arguments: &[MirDottedInvokeArgument],
    prepared: &runmat_runtime::object::protocol::PreparedSubscriptReceiver,
) -> NativeExecutorResult<Vec<ObjectIndexComponent>> {
    let mut components = Vec::new();
    for (component, argument) in arguments.iter().enumerate() {
        match argument {
            MirDottedInvokeArgument::Colon => components.push(ObjectIndexComponent::Colon),
            MirDottedInvokeArgument::Contextual(region) => {
                state
                    .subscript_end_receivers
                    .push((prepared.clone(), component, arguments.len()));
                let result =
                    super::super::indexing::materialize_expression_region(state, region, 1);
                state.subscript_end_receivers.pop();
                components.push(ObjectIndexComponent::Value(result?));
            }
            MirDottedInvokeArgument::Call(region) => {
                components.extend(
                    materialize_call_region(state, region)?
                        .into_iter()
                        .map(ObjectIndexComponent::Value),
                );
            }
        }
    }
    Ok(components)
}

pub(super) fn materialize_call_region(
    state: &mut HostState,
    region: &MirCallArgumentRegion,
) -> NativeExecutorResult<Vec<Value>> {
    let (ordinal, next_step, completed) = state.enter_contextual_region()?;
    if let Some(references) = completed {
        return references
            .as_slice()
            .iter()
            .map(|reference| state.arena.get(*reference).cloned())
            .collect();
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
        let values = super::super::call::materialize_arguments(
            state,
            std::slice::from_ref(&region.argument),
        )?;
        state.finish_contextual_step(ordinal, argument_step, argument_step + 1)?;
        // These references exist only to replay the completed argument region;
        // the returned Values are detached clones. Transfer their initial arena
        // ownership to the expression-region state for exact site retirement.
        state.finish_owned_contextual_values(ordinal, &values)?;
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
