use super::{take_sequence_register, Request};
use runmat_runtime::object::resolve as obj_resolve;
use runmat_runtime::RuntimeError;

#[inline(never)]
pub(super) async fn store_member(request: Request<'_>) -> Result<bool, RuntimeError> {
    let Request {
        instruction: instr,
        stack,
        context,
        ..
    } = request;
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
            let access = super::frame_access(context.bytecode, runtime)?;
            let value = obj_resolve::store_member_traced_with_access(
                runtime,
                base,
                field.0.clone(),
                rhs,
                caller_function_name,
                &access,
            )
            .await?;
            value.with_owned(|value| stack.try_push(value))?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

#[inline(never)]
pub(super) async fn store_member_or_init(request: Request<'_>) -> Result<bool, RuntimeError> {
    let Request {
        instruction: instr,
        stack,
        context,
        ..
    } = request;
    let runtime = context.runtime;
    let caller_function_name =
        (!context.current_function_name.is_empty()).then_some(context.current_function_name);
    match instr {
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
            let access = super::frame_access(context.bytecode, runtime)?;
            let value = obj_resolve::store_member_traced_with_access(
                runtime,
                base,
                field.0.clone(),
                rhs,
                caller_function_name,
                &access,
            )
            .await?;
            value.with_owned(|value| stack.try_push(value))?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

#[inline(never)]
pub(super) async fn store_member_dynamic(request: Request<'_>) -> Result<bool, RuntimeError> {
    let Request {
        instruction: instr,
        stack,
        context,
        ..
    } = request;
    let runtime = context.runtime;
    let caller_function_name =
        (!context.current_function_name.is_empty()).then_some(context.current_function_name);
    match instr {
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
            let access = super::frame_access(context.bytecode, runtime)?;
            let value = obj_resolve::store_member_traced_with_access(
                runtime,
                base,
                name,
                rhs,
                caller_function_name,
                &access,
            )
            .await?;
            value.with_owned(|value| stack.try_push(value))?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

#[inline(never)]
pub(super) async fn store_member_dynamic_or_init(
    request: Request<'_>,
) -> Result<bool, RuntimeError> {
    let Request {
        instruction: instr,
        stack,
        context,
        ..
    } = request;
    let runtime = context.runtime;
    let caller_function_name =
        (!context.current_function_name.is_empty()).then_some(context.current_function_name);
    match instr {
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
            let access = super::frame_access(context.bytecode, runtime)?;
            let value = obj_resolve::store_member_traced_with_access(
                runtime,
                base,
                name,
                rhs,
                caller_function_name,
                &access,
            )
            .await?;
            value.with_owned(|value| stack.try_push(value))?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

#[inline(never)]
pub(super) async fn store_member_sequence(request: Request<'_>) -> Result<bool, RuntimeError> {
    let Request {
        instruction: instr,
        stack,
        sequence_register,
        context,
        ..
    } = request;
    let runtime = context.runtime;
    let caller_function_name =
        (!context.current_function_name.is_empty()).then_some(context.current_function_name);
    match instr {
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
            let access = super::frame_access(context.bytecode, runtime)?;
            let updated = obj_resolve::store_member_sequence_traced_with_access(
                runtime,
                base,
                field.0.clone(),
                values,
                caller_function_name,
                &access,
            )
            .await?;
            updated.with_owned(|value| stack.try_push(value))?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

#[inline(never)]
pub(super) async fn store_member_dynamic_sequence(
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
    let caller_function_name =
        (!context.current_function_name.is_empty()).then_some(context.current_function_name);
    match instr {
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
            let access = super::frame_access(context.bytecode, runtime)?;
            let updated = obj_resolve::store_member_sequence_traced_with_access(
                runtime,
                base,
                name,
                values,
                caller_function_name,
                &access,
            )
            .await?;
            updated.with_owned(|value| stack.try_push(value))?;
            Ok(true)
        }
        _ => Ok(false),
    }
}
