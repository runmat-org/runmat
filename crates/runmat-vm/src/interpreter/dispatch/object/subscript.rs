use super::{
    execute_subscript_path, frame_access, materialize_subscript_path,
    write_rooted_sequence_register, Request, SubscriptPathReadContext,
};
use runmat_runtime::RuntimeError;

#[inline(never)]
pub(super) async fn begin_subscript_end_receiver(
    request: Request<'_>,
) -> Result<bool, RuntimeError> {
    let Request {
        instruction: instr,
        stack,
        sequence_register,
        context,
        ..
    } = request;
    let runtime = context.runtime;
    let bytecode = context.bytecode;
    match instr {
        crate::bytecode::Instr::BeginSubscriptEndReceiver { prefix } => {
            sequence_register.prepare_subscript_receiver()?;
            let operand_count = prefix
                .iter()
                .map(crate::bytecode::BytecodeSubscriptStep::operand_count)
                .sum::<usize>();
            let start = stack.len().checked_sub(operand_count + 1).ok_or_else(|| {
                crate::interpreter::errors::mex("StackUnderflow", "subscript path stack underflow")
            })?;
            let operands = runmat_runtime::execution::RootedValues::try_initialize_in_context(
                stack.len() - start,
                runtime,
                |index| Ok::<_, RuntimeError>(stack.as_slice()[start + index].clone()),
            )?;
            let (root, path) = operands.with_owned_output_storage(|storage| {
                let values = storage.try_into_owned().map_err(|_| {
                    crate::interpreter::errors::mex(
                        "RunMat:SubscriptPathState",
                        "fresh subscript snapshot is unexpectedly shared",
                    )
                })?;
                materialize_subscript_path(runtime, values, prefix)
            })?;
            let prepared = runmat_runtime::object::protocol::prepare_subscript_receiver_in_context(
                runtime,
                root,
                path,
                frame_access(bytecode, runtime)?,
            )
            .await?;
            let stack_index = stack.len();
            stack.try_push(prepared.receiver().clone())?;
            sequence_register.push_subscript_receiver(prepared, start, stack_index)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

#[inline(never)]
pub(super) async fn load_subscript_end(request: Request<'_>) -> Result<bool, RuntimeError> {
    let Request {
        instruction: instr,
        stack,
        sequence_register,
        context,
        ..
    } = request;
    let runtime = context.runtime;
    match instr {
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
            value.with_owned(|value| stack.try_push(value))?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

#[inline(never)]
pub(super) async fn finish_subscript_end_receiver(
    request: Request<'_>,
) -> Result<bool, RuntimeError> {
    let Request {
        instruction: instr,
        stack,
        sequence_register,
        ..
    } = request;
    match instr {
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
            stack.try_insert(prefix_start, prepared.receiver().clone())?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

#[inline(never)]
pub(super) async fn finish_dotted_invoke_receiver(
    request: Request<'_>,
) -> Result<bool, RuntimeError> {
    let Request {
        instruction: instr,
        stack,
        sequence_register,
        ..
    } = request;
    match instr {
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
        _ => Ok(false),
    }
}

#[inline(never)]
pub(super) async fn read_subscript_path(request: Request<'_>) -> Result<bool, RuntimeError> {
    let Request {
        instruction: instr,
        stack,
        sequence_register,
        context,
        ..
    } = request;
    let runtime = context.runtime;
    let bytecode = context.bytecode;
    let invocation_arity = context.invocation_arity;
    match instr {
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
                    access: frame_access(bytecode, runtime)?,
                    invocation_arity,
                },
                sequence_register,
            )
            .await?;
            if *to_sequence_register {
                write_rooted_sequence_register(stack, sequence_register, values)?;
            } else {
                stack.try_extend(values.try_into_values()?)?;
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

#[inline(never)]
pub(super) async fn capture_subscript_path(request: Request<'_>) -> Result<bool, RuntimeError> {
    let Request {
        instruction: instr,
        stack,
        sequence_register,
        context,
        ..
    } = request;
    let runtime = context.runtime;
    let bytecode = context.bytecode;
    let invocation_arity = context.invocation_arity;
    match instr {
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
                    access: frame_access(bytecode, runtime)?,
                    invocation_arity,
                },
                sequence_register,
            )
            .await?;
            sequence_register.capture_rooted_values(*sequence_slot, stack, values)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}
