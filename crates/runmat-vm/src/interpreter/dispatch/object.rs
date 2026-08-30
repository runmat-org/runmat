use runmat_runtime::object::resolve as obj_resolve;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

pub async fn dispatch_object(
    instr: &crate::bytecode::Instr,
    stack: &mut Vec<Value>,
    runtime: &runmat_runtime::context::RuntimeContext,
    current_function_name: &str,
) -> Result<bool, RuntimeError> {
    let caller_function_name = if current_function_name.is_empty() {
        None
    } else {
        Some(current_function_name)
    };
    match instr {
        crate::bytecode::Instr::LoadMember(field) => {
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let value = obj_resolve::load_member_with_context(
                Some(runtime),
                base,
                field.0.clone(),
                false,
                caller_function_name,
            )
            .await?;
            stack.push(value);
            Ok(true)
        }
        crate::bytecode::Instr::LoadMemberOrInit(field) => {
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let value = obj_resolve::load_member_with_context(
                Some(runtime),
                base,
                field.0.clone(),
                true,
                caller_function_name,
            )
            .await?;
            stack.push(value);
            Ok(true)
        }
        crate::bytecode::Instr::LoadMemberDynamic => {
            let name_val = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let name: String = (&name_val).try_into()?;
            let value = obj_resolve::load_member_dynamic_with_context(
                Some(runtime),
                base,
                name,
                false,
                caller_function_name,
            )
            .await?;
            stack.push(value);
            Ok(true)
        }
        crate::bytecode::Instr::LoadMemberDynamicOrInit => {
            let name_val = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let name: String = (&name_val).try_into()?;
            let value = obj_resolve::load_member_dynamic_with_context(
                Some(runtime),
                base,
                name,
                true,
                caller_function_name,
            )
            .await?;
            stack.push(value);
            Ok(true)
        }
        crate::bytecode::Instr::StoreMember(field) => {
            let rhs = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let value = obj_resolve::store_member_traced(
                base,
                field.0.clone(),
                rhs,
                false,
                caller_function_name,
            )
            .await?;
            stack.push(value);
            Ok(true)
        }
        crate::bytecode::Instr::StoreMemberOrInit(field) => {
            let rhs = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let value = obj_resolve::store_member_traced(
                base,
                field.0.clone(),
                rhs,
                true,
                caller_function_name,
            )
            .await?;
            stack.push(value);
            Ok(true)
        }
        crate::bytecode::Instr::StoreMemberDynamic => {
            let rhs = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let name_val = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let name: String = (&name_val).try_into()?;
            let value = obj_resolve::store_member_dynamic_traced(
                base,
                name,
                rhs,
                false,
                caller_function_name,
            )
            .await?;
            stack.push(value);
            Ok(true)
        }
        crate::bytecode::Instr::StoreMemberDynamicOrInit => {
            let rhs = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let name_val = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let name: String = (&name_val).try_into()?;
            let value = obj_resolve::store_member_dynamic_traced(
                base,
                name,
                rhs,
                true,
                caller_function_name,
            )
            .await?;
            stack.push(value);
            Ok(true)
        }
        _ => Ok(false),
    }
}
