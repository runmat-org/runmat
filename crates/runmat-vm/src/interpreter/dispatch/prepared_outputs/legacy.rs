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
    stack: &mut Vec<Value>,
    vars: &[Value],
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
    let endpoint =
        PreparedSequenceDestination::prepare(root(vars, root_slot)?, steps, endpoint).await?;
    state.push_output_target(PreparedOutputTarget::Sequence {
        root_slot,
        endpoint,
    });
    Ok(true)
}

fn dynamic_member(stack: &mut Vec<Value>) -> Result<runmat_types::MemberName, RuntimeError> {
    let value = stack.pop().ok_or_else(stack_underflow)?;
    Ok(runmat_types::MemberName(String::try_from(&value)?))
}

fn parentheses(indices: Vec<Value>) -> AssignmentStepSpec {
    AssignmentStepSpec::Parentheses {
        selectors: values(indices),
    }
}

fn values(values: Vec<Value>) -> Vec<ObjectIndexComponent> {
    values
        .into_iter()
        .map(ObjectIndexComponent::Value)
        .collect()
}
