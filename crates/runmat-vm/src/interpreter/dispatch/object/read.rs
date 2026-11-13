use super::{write_sequence_values, Request};
use runmat_runtime::object::resolve as obj_resolve;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

#[inline(never)]
pub(super) async fn load_member(request: Request<'_>) -> Result<bool, RuntimeError> {
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
        crate::bytecode::Instr::LoadMember(field) => {
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _base_roots = crate::call::root_values([&base])?;
            let access = super::frame_access(context.bytecode, runtime)?;
            let value = obj_resolve::load_member_with_access(
                Some(runtime),
                base,
                field.0.clone(),
                false,
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
pub(super) async fn load_member_or_init(request: Request<'_>) -> Result<bool, RuntimeError> {
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
        crate::bytecode::Instr::LoadMemberOrInit(field) => {
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _base_roots = crate::call::root_values([&base])?;
            let access = super::frame_access(context.bytecode, runtime)?;
            let value = obj_resolve::load_member_with_access(
                Some(runtime),
                base,
                field.0.clone(),
                true,
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
pub(super) async fn load_member_dynamic(request: Request<'_>) -> Result<bool, RuntimeError> {
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
        crate::bytecode::Instr::LoadMemberDynamic => {
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
            let value = obj_resolve::load_member_with_access(
                Some(runtime),
                base,
                name,
                false,
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
pub(super) async fn load_member_dynamic_or_init(
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
        crate::bytecode::Instr::LoadMemberDynamicOrInit => {
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
            let value = obj_resolve::load_member_with_access(
                Some(runtime),
                base,
                name,
                true,
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
pub(super) async fn load_member_sequence(request: Request<'_>) -> Result<bool, RuntimeError> {
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
        crate::bytecode::Instr::LoadMemberSequence { member, selection } => {
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _base_roots = crate::call::root_values([&base])?;
            let access = super::frame_access(context.bytecode, runtime)?;
            let values = obj_resolve::read_member_sequence_with_access(
                Some(runtime),
                base,
                member.0.clone(),
                false,
                caller_function_name,
                &access,
            )
            .await?;
            stack.try_extend(values.resolve_values(*selection, Default::default())?)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

#[inline(never)]
pub(super) async fn load_member_dynamic_sequence(
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
        crate::bytecode::Instr::LoadMemberDynamicSequence { selection } => {
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
            let values = obj_resolve::read_member_sequence_with_access(
                Some(runtime),
                base,
                name,
                false,
                caller_function_name,
                &access,
            )
            .await?;
            stack.try_extend(values.resolve_values(*selection, Default::default())?)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

#[inline(never)]
pub(super) async fn member_sequence_cardinality(
    request: Request<'_>,
) -> Result<bool, RuntimeError> {
    let Request {
        instruction: instr,
        stack,
        ..
    } = request;
    match instr {
        crate::bytecode::Instr::MemberSequenceCardinality => {
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _base_roots = crate::call::root_values([&base])?;
            let count = obj_resolve::member_sequence_cardinality(&base)?;
            stack.try_push(Value::Num(count as f64))?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

#[inline(never)]
pub(super) async fn load_member_sequence_using_output_slot(
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
    let vars = context.vars;
    let caller_function_name =
        (!context.current_function_name.is_empty()).then_some(context.current_function_name);
    match instr {
        crate::bytecode::Instr::LoadMemberSequenceUsingOutputSlot {
            member,
            output_count_slot,
        } => {
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let _base_roots = crate::call::root_values([&base])?;
            let count = super::super::requested_outputs_from_slot(vars, *output_count_slot)?;
            let access = super::frame_access(context.bytecode, runtime)?;
            let sequence = obj_resolve::read_member_sequence_with_access(
                Some(runtime),
                base,
                member.0.clone(),
                false,
                caller_function_name,
                &access,
            )
            .await?;
            let values = sequence.resolve_values(
                runmat_types::SequenceUse::SelectPrefix { count },
                Default::default(),
            )?;
            write_sequence_values(stack, sequence_register, values)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

#[inline(never)]
pub(super) async fn load_member_dynamic_sequence_using_output_slot(
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
    let vars = context.vars;
    let caller_function_name =
        (!context.current_function_name.is_empty()).then_some(context.current_function_name);
    match instr {
        crate::bytecode::Instr::LoadMemberDynamicSequenceUsingOutputSlot { output_count_slot } => {
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
            let count = super::super::requested_outputs_from_slot(vars, *output_count_slot)?;
            let access = super::frame_access(context.bytecode, runtime)?;
            let sequence = obj_resolve::read_member_sequence_with_access(
                Some(runtime),
                base,
                name,
                false,
                caller_function_name,
                &access,
            )
            .await?;
            let values = sequence.resolve_values(
                runmat_types::SequenceUse::SelectPrefix { count },
                Default::default(),
            )?;
            write_sequence_values(stack, sequence_register, values)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}
