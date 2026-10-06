use super::{
    execute_subscript_path, frame_access, materialize_subscript_path, write_sequence_register,
    ObjectDispatchContext, SubscriptPathReadContext,
};
use runmat_runtime::RuntimeError;
use runmat_value::Value;

pub(super) async fn dispatch(
    instr: &crate::bytecode::Instr,
    stack: &mut Vec<Value>,
    sequence_register: &mut super::super::SequenceState,
    context: ObjectDispatchContext<'_>,
) -> Result<bool, RuntimeError> {
    let runtime = context.runtime;
    let bytecode = context.bytecode;
    let invocation_arity = context.invocation_arity;
    match instr {
        crate::bytecode::Instr::BeginSubscriptEndReceiver { prefix } => {
            let operand_count = prefix
                .iter()
                .map(crate::bytecode::BytecodeSubscriptStep::operand_count)
                .sum::<usize>();
            let start = stack.len().checked_sub(operand_count + 1).ok_or_else(|| {
                crate::interpreter::errors::mex("StackUnderflow", "subscript path stack underflow")
            })?;
            let (root, path) = materialize_subscript_path(stack[start..].to_vec(), prefix)?;
            let prepared = runmat_runtime::object::protocol::prepare_subscript_receiver_in_context(
                runtime,
                root,
                path,
                frame_access(bytecode)?,
            )
            .await?;
            let stack_index = stack.len();
            stack.push(prepared.receiver().clone());
            sequence_register.push_subscript_receiver(prepared, start, stack_index);
            Ok(true)
        }
        crate::bytecode::Instr::LoadSubscriptEnd {
            component,
            component_count,
        } => {
            let value = runmat_runtime::object::protocol::resolve_subscript_end_in_context(
                runtime,
                sequence_register.subscript_receiver()?,
                *component,
                *component_count,
            )
            .await?;
            stack.push(value);
            Ok(true)
        }
        crate::bytecode::Instr::FinishSubscriptEndReceiver {
            selector_count,
            prefix_operand_count,
        } => {
            let (prepared, prefix_start, expected_receiver) =
                sequence_register.pop_subscript_receiver()?;
            let receiver = stack.len().checked_sub(selector_count + 1).ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "subscript receiver stack underflow",
                )
            })?;
            if receiver != expected_receiver {
                return Err(crate::interpreter::errors::mex(
                    "RunMat:SubscriptPathState",
                    "subscript receiver root moved before completion",
                ));
            }
            if receiver.checked_sub(prefix_start) != Some(*prefix_operand_count + 1) {
                return Err(crate::interpreter::errors::mex(
                    "RunMat:SubscriptPathState",
                    "prepared subscript prefix operand count is inconsistent",
                ));
            }
            stack.remove(receiver);
            if prefix_start > receiver {
                return Err(crate::interpreter::errors::mex(
                    "RunMat:SubscriptPathState",
                    "subscript prefix root follows its prepared receiver",
                ));
            }
            stack.drain(prefix_start..receiver);
            stack.insert(prefix_start, prepared.receiver().clone());
            Ok(true)
        }
        crate::bytecode::Instr::FinishDottedInvokeReceiver {
            prefix_operand_count,
            argument_operand_count,
        } => {
            sequence_register.finish_dotted_invoke_receiver(
                stack,
                *prefix_operand_count,
                *argument_operand_count,
            )?;
            Ok(true)
        }
        crate::bytecode::Instr::ReadSubscriptPath {
            steps,
            selection,
            context,
            to_sequence_register,
        } => {
            let values = execute_subscript_path(
                runtime,
                stack,
                steps,
                SubscriptPathReadContext {
                    selection: *selection,
                    indexing: *context,
                    access: frame_access(bytecode)?,
                    invocation_arity,
                },
                sequence_register,
            )
            .await?;
            if *to_sequence_register {
                write_sequence_register(stack, sequence_register, values)?;
            } else {
                stack.extend(values);
            }
            Ok(true)
        }
        crate::bytecode::Instr::CaptureSubscriptPath {
            steps,
            sequence_slot,
            context,
        } => {
            let values = execute_subscript_path(
                runtime,
                stack,
                steps,
                SubscriptPathReadContext {
                    selection: runmat_types::SequenceUse::ExpandAll,
                    indexing: *context,
                    access: frame_access(bytecode)?,
                    invocation_arity,
                },
                sequence_register,
            )
            .await?;
            sequence_register.capture(*sequence_slot, stack, values)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}
