use super::{write_sequence_register, ObjectDispatchContext};
use runmat_runtime::sequence::ResolveValueSequence;
use runmat_runtime::RuntimeError;

pub(super) async fn dispatch(
    instr: &crate::bytecode::Instr,
    stack: &mut crate::interpreter::stack::InterpreterStack,
    sequence_register: &mut super::super::SequenceState,
    context: ObjectDispatchContext<'_>,
) -> Result<bool, RuntimeError> {
    let runtime = context.runtime;
    match instr {
        crate::bytecode::Instr::CaptureCallOutputSequence => {
            sequence_register.capture_call_outputs()?;
            Ok(true)
        }
        crate::bytecode::Instr::CaptureScalarSequence => {
            let value = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _value_roots = crate::call::root_values([&value])?;
            write_sequence_register(stack, sequence_register, vec![value])?;
            Ok(true)
        }
        crate::bytecode::Instr::CaptureMemberSequence {
            member,
            sequence_slot,
        } => {
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _base_roots = crate::call::root_values([&base])?;
            let values = runmat_runtime::call::arguments::materialize_expansion(
                runtime,
                runmat_runtime::call::arguments::MaterializedExpansionSource::Member {
                    base,
                    member: member.clone(),
                },
            )
            .await?;
            values.with_owned_sequence(|sequence| {
                let values =
                    sequence.resolve(runmat_types::SequenceUse::ExpandAll, Default::default())?;
                sequence_register.capture(*sequence_slot, stack, values)
            })?;
            Ok(true)
        }
        crate::bytecode::Instr::CaptureMemberDynamicSequence { sequence_slot } => {
            let member = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _member_roots = crate::call::root_values([&member])?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _base_roots = crate::call::root_values([&base])?;
            let values = runmat_runtime::call::arguments::materialize_expansion(
                runtime,
                runmat_runtime::call::arguments::MaterializedExpansionSource::DynamicMember {
                    base,
                    member,
                },
            )
            .await?;
            values.with_owned_sequence(|sequence| {
                let values =
                    sequence.resolve(runmat_types::SequenceUse::ExpandAll, Default::default())?;
                sequence_register.capture(*sequence_slot, stack, values)
            })?;
            Ok(true)
        }
        crate::bytecode::Instr::CaptureCellContentsSequence {
            sequence_slot,
            num_indices,
            expand_all,
        } => {
            let mut indices = Vec::with_capacity(*num_indices);
            for _ in 0..*num_indices {
                indices.push(stack.pop().ok_or(crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "stack underflow",
                ))?);
            }
            indices.reverse();
            let _indices_roots = crate::call::root_values(&indices)?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _base_roots = crate::call::root_values([&base])?;
            let values = runmat_runtime::call::arguments::materialize_expansion(
                runtime,
                runmat_runtime::call::arguments::MaterializedExpansionSource::CellContents {
                    base,
                    indices,
                    expand_all: *expand_all,
                },
            )
            .await?;
            values.with_owned_sequence(|sequence| {
                let values =
                    sequence.resolve(runmat_types::SequenceUse::ExpandAll, Default::default())?;
                sequence_register.capture(*sequence_slot, stack, values)
            })?;
            Ok(true)
        }
        crate::bytecode::Instr::CaptureReturnedOutputsSequence { sequence_slot } => {
            let value = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _value_roots = crate::call::root_values([&value])?;
            let values = runmat_runtime::call::arguments::materialize_expansion(
                runtime,
                runmat_runtime::call::arguments::MaterializedExpansionSource::ReturnedOutputs(
                    runmat_runtime::call::arguments::adapt_legacy_builtin_result(value)?,
                ),
            )
            .await?;
            values.with_owned_sequence(|sequence| {
                let values =
                    sequence.resolve(runmat_types::SequenceUse::ExpandAll, Default::default())?;
                sequence_register.capture(*sequence_slot, stack, values)
            })?;
            Ok(true)
        }
        _ => Ok(false),
    }
}
