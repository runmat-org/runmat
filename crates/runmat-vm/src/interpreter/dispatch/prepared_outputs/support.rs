use crate::interpreter::vars::{InterpreterValueRef, InterpreterVars};
use runmat_runtime::object::indexing::ObjectIndexComponent;
use runmat_runtime::sequence::SequenceEndpointSpec;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

use super::super::{PendingSequenceDestination, PreparedOutputTarget, SequenceState};

pub(super) async fn pop_selector_components(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    selectors: &[crate::bytecode::BytecodeSubscriptSelector],
    expand_all: bool,
    runtime: &runmat_runtime::context::RuntimeContext,
    state: &mut SequenceState,
) -> Result<
    (
        runmat_value::ExactArrayStorage<ObjectIndexComponent>,
        crate::runtime::workspace::FrameVec<runmat_runtime::execution::RootedValueLease>,
    ),
    RuntimeError,
> {
    use crate::bytecode::BytecodeSubscriptSelector as Selector;
    state.validate_capture_context(runtime)?;
    let inputs = runmat_runtime::execution::RootedValueLease::for_value_slices_in_context(
        &[stack.as_slice()],
        runtime,
    )?;
    let specs = selectors.iter().filter_map(|selector| match selector {
        Selector::Colon => None,
        Selector::Value => Some(&runmat_runtime::call::arguments::ArgumentSpec::Single),
        Selector::Expansion(spec) => Some(spec),
    });
    let mut arguments =
        crate::call::shared::take_materialized_args_from_specs(stack, specs, state, runtime)?
            .into_iter();
    let root_count = selectors.len().checked_add(2).ok_or_else(|| {
        crate::interpreter::errors::mex(
            "RunMat:SubscriptPathState",
            "selector root count overflows",
        )
    })?;
    let mut roots = state.frame_vector(root_count)?;
    roots.push(inputs);
    let components = super::super::object::materialize_ordered_selector_arguments(
        runtime,
        &mut arguments,
        selectors
            .iter()
            .map(super::super::object::selector_source)
            .chain(expand_all.then_some(super::super::object::PreparedSelectorSource::Colon)),
        &mut roots,
        state,
    )
    .await?;
    if arguments.next().is_some() {
        return Err(crate::interpreter::errors::mex(
            "RunMat:SubscriptPathState",
            "prepared selectors have excess operands",
        ));
    }
    Ok((components, roots))
}

pub(super) async fn finish_destination(
    state: &mut SequenceState,
    runtime: &runmat_runtime::context::RuntimeContext,
    endpoint: SequenceEndpointSpec,
) -> Result<(), RuntimeError> {
    let PendingSequenceDestination {
        root_slot,
        builder,
        absent_root,
        root_creation,
        prepared_root,
    } = state.take_destination_builder()?;
    let endpoint = builder.finish_in_context(runtime, endpoint).await?;
    state.push_output_target(PreparedOutputTarget::Sequence {
        root_slot,
        endpoint,
        absent_root,
        root_creation,
        prepared_root,
    })?;
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
