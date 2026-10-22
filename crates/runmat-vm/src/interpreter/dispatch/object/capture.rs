use super::ObjectDispatchContext;
use runmat_runtime::RuntimeError;

pub(super) async fn dispatch(
    instr: &crate::bytecode::Instr,
    stack: &mut crate::interpreter::stack::InterpreterStack,
    sequence_register: &mut super::super::SequenceState,
    context: ObjectDispatchContext<'_>,
) -> Result<bool, RuntimeError> {
    use crate::bytecode::Instr;
    use runmat_runtime::call::arguments::MaterializedExpansionSource as Source;
    use runmat_runtime::execution::RootedValueLease;
    let runtime = context.runtime;
    sequence_register.validate_capture_context(runtime)?;
    match instr {
        Instr::MoveCapturedSequence {
            source_slot,
            destination_slot,
        } => {
            sequence_register.move_capture(*source_slot, *destination_slot)?;
            Ok(true)
        }
        Instr::CaptureCallOutputSequence => {
            sequence_register.capture_call_outputs()?;
            Ok(true)
        }
        Instr::CaptureScalarSequence => {
            sequence_register.capture_scalar_top(stack)?;
            Ok(true)
        }
        Instr::CaptureMemberSequence {
            member,
            sequence_slot,
        } => {
            let base = pop(stack)?;
            let _roots = RootedValueLease::for_value_slices_in_context(
                &[std::slice::from_ref(&base)],
                runtime,
            )?;
            let member = runmat_runtime::call::argument_text::copy_str(&member.0, Some(runtime))?;
            let values = runmat_runtime::call::arguments::materialize_expansion(
                runtime,
                Source::Member { base, member },
            )
            .await?;
            sequence_register.capture_rooted(*sequence_slot, stack, values)?;
            Ok(true)
        }
        Instr::CaptureMemberDynamicSequence { sequence_slot } => {
            let member = pop(stack)?;
            let base = pop(stack)?;
            let _roots = RootedValueLease::for_value_slices_in_context(
                &[std::slice::from_ref(&base), std::slice::from_ref(&member)],
                runtime,
            )?;
            let values = runmat_runtime::call::arguments::materialize_expansion(
                runtime,
                Source::DynamicMember { base, member },
            )
            .await?;
            sequence_register.capture_rooted(*sequence_slot, stack, values)?;
            Ok(true)
        }
        Instr::CaptureCellContentsSequence {
            sequence_slot,
            num_indices,
            expand_all,
        } => {
            let start = stack.len().checked_sub(*num_indices).ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "cell expansion indices are missing",
                )
            })?;
            let indices = stack.try_extract_output_range(start, stack.len())?;
            let base = pop(stack)?;
            let _roots = RootedValueLease::for_value_slices_in_context(
                &[std::slice::from_ref(&base), indices.as_slice()],
                runtime,
            )?;
            let values = runmat_runtime::call::arguments::materialize_expansion(
                runtime,
                Source::CellContents {
                    base,
                    indices,
                    expand_all: *expand_all,
                },
            )
            .await?;
            sequence_register.capture_rooted(*sequence_slot, stack, values)?;
            Ok(true)
        }
        Instr::CaptureReturnedOutputsSequence { sequence_slot } => {
            let value = pop(stack)?;
            let _roots = RootedValueLease::for_value_slices_in_context(
                &[std::slice::from_ref(&value)],
                runtime,
            )?;
            let sequence = runmat_runtime::execution::RootedValueSequence::in_context(
                runmat_runtime::call::arguments::adapt_legacy_builtin_result(value)?,
                runtime,
            )?;
            let values = runmat_runtime::call::arguments::materialize_expansion(
                runtime,
                Source::ReturnedOutputs(sequence),
            )
            .await?;
            sequence_register.capture_rooted(*sequence_slot, stack, values)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn pop(
    stack: &mut crate::interpreter::stack::InterpreterStack,
) -> Result<runmat_value::Value, RuntimeError> {
    stack.pop().ok_or_else(|| {
        crate::interpreter::errors::mex("StackUnderflow", "capture operand is missing")
    })
}
