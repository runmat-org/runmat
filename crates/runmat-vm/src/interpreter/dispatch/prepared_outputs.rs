use crate::interpreter::vars::InterpreterVars;
use runmat_runtime::object::indexing::ObjectIndexComponent;
use runmat_runtime::sequence::{AssignmentStepSpec, SequenceEndpointSpec};
use runmat_runtime::RuntimeError;
use runmat_value::{IntValue, Value};

use super::{PreparedOutputTarget, SequenceState};

mod commit;
mod legacy;
mod support;
use support::{
    empty_cell_root, finish_destination, pop_selector_components, root, stack_underflow,
};

pub(super) struct PreparedOutputContext<'a> {
    pub authority: super::slot_authority::AssignmentSlotAuthority<'a>,
}

pub(super) async fn dispatch(
    instruction: &crate::bytecode::Instr,
    stack: &mut Vec<Value>,
    vars: &mut InterpreterVars,
    state: &mut SequenceState,
    context: PreparedOutputContext<'_>,
    before_overwrite: &mut dyn FnMut(usize),
    after_store: &mut dyn FnMut(usize),
) -> Result<bool, RuntimeError> {
    if legacy::dispatch(instruction, stack, vars, state).await? {
        return Ok(true);
    }
    use crate::bytecode::Instr;
    match instruction {
        Instr::BeginOutputAssignment { target_count } => {
            state.begin_output_assignment(*target_count)?
        }
        Instr::PrepareFixedOutputTarget => state.push_output_target(PreparedOutputTarget::Fixed),
        Instr::PrepareDiscardOutputTarget => {
            state.push_output_target(PreparedOutputTarget::Discard)
        }
        Instr::BeginSequenceOutputTarget {
            root_slot,
            root_creation,
        } => {
            let root_presence = context.authority.refresh(vars, *root_slot)?;
            let absent_root = if *root_creation == runmat_mir::SequenceRootCreation::EmptyCell
                && root_presence.is_absent()
            {
                Some(empty_cell_root())
            } else {
                None
            };
            let original = root(vars, *root_slot)?;
            match absent_root {
                Some(seed) => state.begin_destination(
                    *root_slot,
                    &seed,
                    Some(Box::new(seed.clone())),
                    *root_creation,
                )?,
                None => state.begin_destination(*root_slot, &original, None, *root_creation)?,
            }
        }
        Instr::PrepareMemberPathStep(member) => {
            state
                .destination_builder_mut()?
                .push_step(
                    AssignmentStepSpec::Member(member.clone()),
                    Some(context.authority.current_function_name),
                )
                .await?;
        }
        Instr::PrepareDynamicMemberPathStep => {
            let member = runmat_types::MemberName(String::try_from(
                &stack.pop().ok_or_else(stack_underflow)?,
            )?);
            state
                .destination_builder_mut()?
                .push_step(
                    AssignmentStepSpec::Member(member),
                    Some(context.authority.current_function_name),
                )
                .await?;
        }
        Instr::BeginPreparedIndexSelectors { component_count } => {
            state.begin_index_selectors(*component_count)?;
        }
        Instr::LoadPreparedIndexEnd { component } => {
            let component_count = state.index_selector_count()?;
            let extent = state
                .destination_builder_mut()?
                .selector_extent(component_count, *component)?;
            stack.push(Value::Num(extent as f64));
        }
        Instr::BeginContextualIndexSelectors { component_count } => {
            let base = stack.last().ok_or_else(stack_underflow)?;
            let shape = runmat_runtime::builtins::common::shape::value_dimensions(base).await?;
            state.begin_contextual_index(*component_count, shape)?;
        }
        Instr::LoadContextualIndexEnd { component } => {
            let extent = state.contextual_index_extent(*component)?;
            stack.push(Value::Num(extent as f64));
        }
        Instr::FinishContextualIndexSelectors { component_count } => {
            state.finish_contextual_index(*component_count)?;
        }
        Instr::PrepareParenthesesPathStep {
            component_count,
            selectors,
        } => {
            let selectors = pop_selector_components(stack, selectors)?;
            state.finish_index_selectors(*component_count)?;
            state
                .destination_builder_mut()?
                .push_step(
                    AssignmentStepSpec::Parentheses { selectors },
                    Some(context.authority.current_function_name),
                )
                .await?;
        }
        Instr::PrepareBracesPathStep {
            component_count,
            selectors,
            expand_all,
        } => {
            let mut indices = pop_selector_components(stack, selectors)?;
            if *expand_all {
                indices.push(ObjectIndexComponent::Colon);
            }
            state.finish_index_selectors(*component_count)?;
            state
                .destination_builder_mut()?
                .push_step(
                    AssignmentStepSpec::Braces(indices),
                    Some(context.authority.current_function_name),
                )
                .await?;
        }
        Instr::FinishMemberSequenceOutputTarget(member) => {
            finish_destination(state, SequenceEndpointSpec::Member(member.clone())).await?;
        }
        Instr::FinishDynamicMemberSequenceOutputTarget => {
            let member = runmat_types::MemberName(String::try_from(
                &stack.pop().ok_or_else(stack_underflow)?,
            )?);
            finish_destination(state, SequenceEndpointSpec::Member(member)).await?;
        }
        Instr::FinishCellContentsSequenceOutputTarget {
            component_count,
            selectors,
            expand_all,
        } => {
            let mut indices = pop_selector_components(stack, selectors)?;
            if *expand_all {
                indices.push(ObjectIndexComponent::Colon);
            }
            state.finish_index_selectors(*component_count)?;
            finish_destination(state, SequenceEndpointSpec::CellContents(indices)).await?;
        }
        Instr::LoadPreparedOutputCardinality => {
            let total = u64::try_from(state.output_layout()?.total()).map_err(|_| {
                crate::interpreter::errors::mex(
                    "DestinationCardinalityOverflow",
                    "output destination cardinality exceeds the supported limit",
                )
            })?;
            stack.push(Value::Int(IntValue::U64(total)));
        }
        Instr::CommitPreparedOutputTargets { retained_outputs } => {
            commit::commit(
                stack,
                vars,
                state,
                *retained_outputs,
                &context.authority,
                before_overwrite,
                after_store,
            )
            .await?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}
