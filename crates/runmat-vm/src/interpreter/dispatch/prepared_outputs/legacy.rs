use crate::interpreter::vars::InterpreterVars;
use runmat_runtime::object::indexing::ObjectIndexComponent;
use runmat_runtime::sequence::{
    AssignmentStepSpec, SequenceDestinationBuilder, SequenceEndpointSpec,
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
    runtime: &runmat_runtime::context::RuntimeContext,
    bytecode: &crate::bytecode::Bytecode,
    caller: Option<&str>,
) -> Result<bool, RuntimeError> {
    use crate::bytecode::Instr;
    let (root_slot, steps, endpoint) = match instruction {
        Instr::PrepareMemberSequenceOutputTarget { root_slot, member } => (
            *root_slot,
            Vec::new(),
            SequenceEndpointSpec::Member(member.0.clone().into()),
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
            SequenceEndpointSpec::Member(member.0.clone().into()),
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
                SequenceEndpointSpec::CellContents(
                    runmat_value::ExactArrayStorage::shared_unbounded(components),
                ),
            )
        }
        _ => return Ok(false),
    };
    // Detach the root before object/index preparation can suspend or invoke a
    // user protocol. A shared InterpreterVars read guard must never span that
    // boundary. `prepare_owned` consumes this same required snapshot, so this
    // does not add a second Value clone.
    let root = root(vars, root_slot)?.clone();
    let builder = SequenceDestinationBuilder::from_owned_in_context(runtime, root)?;
    let mut builder = if bytecode.active_class_method_owner.is_some() {
        builder.with_method_frame_factory_in_context(runtime, || {
            super::super::object::frame_access(bytecode, runtime)
        })?
    } else {
        builder
    };
    for step in steps {
        builder.push_step_in_context(runtime, step, caller).await?;
    }
    let endpoint = builder.finish_in_context(runtime, endpoint).await?;
    state.push_output_target(PreparedOutputTarget::Sequence {
        root_slot,
        endpoint,
        absent_root: None,
        root_creation: runmat_mir::SequenceRootCreation::Existing,
        prepared_root: None,
    })?;
    Ok(true)
}

fn dynamic_member(
    stack: &mut crate::interpreter::stack::InterpreterStack,
) -> Result<runmat_value::Utf8Text, RuntimeError> {
    let value = stack.pop().ok_or_else(stack_underflow)?;
    Ok(String::try_from(&value)?.into())
}

fn parentheses(indices: impl IntoIterator<Item = Value>) -> AssignmentStepSpec {
    AssignmentStepSpec::Parentheses {
        selectors: runmat_value::ExactArrayStorage::shared_unbounded(values(indices)),
    }
}

fn values(values: impl IntoIterator<Item = Value>) -> Vec<ObjectIndexComponent> {
    values
        .into_iter()
        .map(ObjectIndexComponent::Value)
        .collect()
}
