use crate::interpreter::vars::{InterpreterValueRef, InterpreterVars};
use runmat_runtime::object::indexing::ObjectIndexComponent;
use runmat_runtime::sequence::SequenceEndpointSpec;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

use super::super::{PendingSequenceDestination, PreparedOutputTarget, SequenceState};

pub(super) fn pop_selector_components(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    selectors: &[crate::bytecode::BytecodeSubscriptSelector],
) -> Result<Vec<ObjectIndexComponent>, RuntimeError> {
    let value_count = selectors
        .iter()
        .filter(|selector| matches!(selector, crate::bytecode::BytecodeSubscriptSelector::Value))
        .count();
    let mut values = pop_values(stack, value_count)?.into_iter();
    selectors
        .iter()
        .map(|selector| match selector {
            crate::bytecode::BytecodeSubscriptSelector::Colon => Ok(ObjectIndexComponent::Colon),
            crate::bytecode::BytecodeSubscriptSelector::Value => values
                .next()
                .map(ObjectIndexComponent::Value)
                .ok_or_else(stack_underflow),
        })
        .collect()
}

pub(super) async fn finish_destination(
    state: &mut SequenceState,
    endpoint: SequenceEndpointSpec,
) -> Result<(), RuntimeError> {
    let PendingSequenceDestination {
        root_slot,
        builder,
        absent_root,
        root_creation,
    } = state.take_destination_builder()?;
    let endpoint = builder.finish(endpoint).await?;
    state.push_output_target(PreparedOutputTarget::Sequence {
        root_slot,
        endpoint,
        absent_root,
        root_creation,
    });
    Ok(())
}

pub(super) fn empty_cell_root() -> Value {
    super::super::slot_authority::empty_cell_root()
}

pub(super) fn root(
    vars: &InterpreterVars,
    slot: usize,
) -> Result<InterpreterValueRef<'_>, RuntimeError> {
    vars.get(slot).ok_or_else(|| {
        crate::interpreter::errors::mex(
            "OutputTargetSlotOutOfBounds",
            "sequence output target root is out of bounds",
        )
    })
}

pub(super) fn stack_underflow() -> RuntimeError {
    crate::interpreter::errors::mex("StackUnderflow", "stack underflow")
}

pub(super) fn pop_values(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    count: usize,
) -> Result<runmat_value::OwnedOutputListValues, RuntimeError> {
    let start = stack.len().checked_sub(count).ok_or_else(stack_underflow)?;
    stack
        .try_extract_output(start)?
        .try_into_owned()
        .map_err(|_| {
            crate::interpreter::errors::mex(
                "SharedOutputStorage",
                "prepared output operands unexpectedly have shared storage",
            )
        })
}
