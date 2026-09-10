use runmat_runtime::object::indexing::ObjectIndexComponent;
use runmat_runtime::sequence::SequenceEndpointSpec;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

use super::super::{PreparedOutputTarget, SequenceState};

pub(super) fn pop_selector_components(
    stack: &mut Vec<Value>,
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
    let (root_slot, builder) = state.take_destination_builder()?;
    let endpoint = builder.finish(endpoint).await?;
    state.push_output_target(PreparedOutputTarget::Sequence {
        root_slot,
        endpoint,
    });
    Ok(())
}

pub(super) fn root(vars: &[Value], slot: usize) -> Result<&Value, RuntimeError> {
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

pub(super) fn pop_values(stack: &mut Vec<Value>, count: usize) -> Result<Vec<Value>, RuntimeError> {
    let start = stack.len().checked_sub(count).ok_or_else(stack_underflow)?;
    Ok(stack.split_off(start))
}
