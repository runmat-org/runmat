use super::{take_sequence_register, ObjectDispatchContext};
use runmat_runtime::object::resolve as obj_resolve;
use runmat_runtime::RuntimeError;

pub(super) async fn dispatch(
    instr: &crate::bytecode::Instr,
    stack: &mut crate::interpreter::stack::InterpreterStack,
    sequence_register: &mut super::super::SequenceState,
    context: ObjectDispatchContext<'_>,
) -> Result<bool, RuntimeError> {
    let runtime = context.runtime;
    let caller_function_name =
        (!context.current_function_name.is_empty()).then_some(context.current_function_name);
    match instr {
        crate::bytecode::Instr::StoreMember(field) => {
            let rhs = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _rhs_roots = crate::call::root_values([&rhs])?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _base_roots = crate::call::root_values([&base])?;
            let value = obj_resolve::store_member_traced_in_context(
                runtime,
                base,
                field.0.clone(),
                rhs,
                false,
                caller_function_name,
            )
            .await?;
            value.with_owned(|value| stack.try_push(value))?;
            Ok(true)
        }
        crate::bytecode::Instr::StoreMemberOrInit(field) => {
            let rhs = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _rhs_roots = crate::call::root_values([&rhs])?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _base_roots = crate::call::root_values([&base])?;
            let value = obj_resolve::store_member_traced_in_context(
                runtime,
                base,
                field.0.clone(),
                rhs,
                true,
                caller_function_name,
            )
            .await?;
            value.with_owned(|value| stack.try_push(value))?;
            Ok(true)
        }
        crate::bytecode::Instr::StoreMemberDynamic => {
            let rhs = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _rhs_roots = crate::call::root_values([&rhs])?;
            let name_val = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _base_roots = crate::call::root_values([&base])?;
            let name: String = (&name_val).try_into()?;
            let value = obj_resolve::store_member_dynamic_traced_in_context(
                runtime,
                base,
                name,
                rhs,
                false,
                caller_function_name,
            )
            .await?;
            value.with_owned(|value| stack.try_push(value))?;
            Ok(true)
        }
        crate::bytecode::Instr::StoreMemberDynamicOrInit => {
            let rhs = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _rhs_roots = crate::call::root_values([&rhs])?;
            let name_val = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _base_roots = crate::call::root_values([&base])?;
            let name: String = (&name_val).try_into()?;
            let value = obj_resolve::store_member_dynamic_traced_in_context(
                runtime,
                base,
                name,
                rhs,
                true,
                caller_function_name,
            )
            .await?;
            value.with_owned(|value| stack.try_push(value))?;
            Ok(true)
        }
        crate::bytecode::Instr::StoreMemberSequence(field) => {
            let values = runmat_runtime::sequence::into_unbounded_values(take_sequence_register(
                stack,
                sequence_register,
            )?)?;
            let _values_roots = crate::call::root_values(&values)?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _base_roots = crate::call::root_values([&base])?;
            let updated = obj_resolve::store_member_sequence_traced_in_context(
                runtime,
                base,
                field.0.clone(),
                values,
                caller_function_name,
            )
            .await?;
            updated.with_owned(|value| stack.try_push(value))?;
            Ok(true)
        }
        crate::bytecode::Instr::StoreMemberDynamicSequence => {
            let values = runmat_runtime::sequence::into_unbounded_values(take_sequence_register(
                stack,
                sequence_register,
            )?)?;
            let _values_roots = crate::call::root_values(&values)?;
            let name: String = (&stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?)
                .try_into()?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _base_roots = crate::call::root_values([&base])?;
            let updated = obj_resolve::store_member_sequence_traced_in_context(
                runtime,
                base,
                name,
                values,
                caller_function_name,
            )
            .await?;
            updated.with_owned(|value| stack.try_push(value))?;
            Ok(true)
        }
        _ => Ok(false),
    }
}
