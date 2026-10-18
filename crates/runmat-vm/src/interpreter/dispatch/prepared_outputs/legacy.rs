use crate::interpreter::vars::InterpreterVars;
use runmat_runtime::object::indexing::ObjectIndexComponent;
use runmat_runtime::sequence::{
    AssignmentStepSpec, PreparedSequenceDestination, SequenceEndpointSpec,
};
use runmat_runtime::RuntimeError;
use runmat_value::Value;

use super::super::{PreparedOutputTarget, SequenceState};
use super::support::{pop_values, root, stack_underflow};

pub(super) async fn dispatch(
    instruction: &crate::bytecode::Instr,
    stack: &mut crate::interpreter::stack::InterpreterStack,
    vars: &InterpreterVars,
    state: &mut SequenceState,
) -> Result<bool, RuntimeError> {
    use crate::bytecode::Instr;
    let (root_slot, steps, endpoint) = match instruction {
        Instr::PrepareMemberSequenceOutputTarget { root_slot, member } => (
            *root_slot,
            Vec::new(),
            SequenceEndpointSpec::Member(member.clone()),
        ),
        Instr::PrepareMemberDynamicSequenceOutputTarget { root_slot } => {
            let member = dynamic_member(stack)?;
            (*root_slot, Vec::new(), SequenceEndpointSpec::Member(member))
        }
        Instr::PrepareIndexedMemberSequenceOutputTarget {
            root_slot,
            member,
            num_indices,
        } => (
            *root_slot,
            vec![parentheses(pop_values(stack, *num_indices)?)],
            SequenceEndpointSpec::Member(member.clone()),
        ),
        Instr::PrepareIndexedMemberDynamicSequenceOutputTarget {
            root_slot,
            num_indices,
        } => {
            let member = dynamic_member(stack)?;
            (
                *root_slot,
                vec![parentheses(pop_values(stack, *num_indices)?)],
                SequenceEndpointSpec::Member(member),
            )
        }
        Instr::PrepareCellContentsSequenceOutputTarget {
            root_slot,
            num_indices,
            expand_all,
        } => {
            let mut components = values(pop_values(stack, *num_indices)?);
            if *expand_all {
                components.push(ObjectIndexComponent::Colon);
            }
            (
                *root_slot,
                Vec::new(),
                SequenceEndpointSpec::CellContents(components),
            )
        }
        _ => return Ok(false),
    };
    // Detach the root before object/index preparation can suspend or invoke a
    // user protocol. A shared InterpreterVars read guard must never span that
    // boundary. `prepare_owned` consumes this same required snapshot, so this
    // does not add a second Value clone.
    let root = root(vars, root_slot)?.clone();
    let endpoint = PreparedSequenceDestination::prepare_owned(root, steps, endpoint).await?;
    state.push_output_target(PreparedOutputTarget::Sequence {
        root_slot,
        endpoint,
        absent_root: None,
        root_creation: runmat_mir::SequenceRootCreation::Existing,
    });
    Ok(true)
}

fn dynamic_member(
    stack: &mut crate::interpreter::stack::InterpreterStack,
) -> Result<runmat_types::MemberName, RuntimeError> {
    let value = stack.pop().ok_or_else(stack_underflow)?;
    Ok(runmat_types::MemberName(String::try_from(&value)?))
}

fn parentheses(indices: impl IntoIterator<Item = Value>) -> AssignmentStepSpec {
    AssignmentStepSpec::Parentheses {
        selectors: values(indices),
    }
}

fn values(values: impl IntoIterator<Item = Value>) -> Vec<ObjectIndexComponent> {
    values
        .into_iter()
        .map(ObjectIndexComponent::Value)
        .collect()
}
