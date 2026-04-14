use crate::object::resolve as obj_resolve;
use runmat_builtins::Value;
use runmat_gc::gc_record_write;
use runmat_runtime::RuntimeError;

pub async fn dispatch_object(instr: &crate::bytecode::Instr, stack: &mut Vec<Value>) -> Result<bool, RuntimeError> {
    match instr {
        crate::bytecode::Instr::LoadMember(field) => {
            let base = stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
            let value = obj_resolve::load_member(base, field.clone(), false).await?;
            stack.push(value);
            Ok(true)
        }
        crate::bytecode::Instr::LoadMemberOrInit(field) => {
            let base = stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
            let value = obj_resolve::load_member(base, field.clone(), true).await?;
            stack.push(value);
            Ok(true)
        }
        crate::bytecode::Instr::LoadMemberDynamic => {
            let name_val = stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
            let base = stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
            let name: String = (&name_val).try_into()?;
            let value = obj_resolve::load_member_dynamic(base, name, false).await?;
            stack.push(value);
            Ok(true)
        }
        crate::bytecode::Instr::LoadMemberDynamicOrInit => {
            let name_val = stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
            let base = stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
            let name: String = (&name_val).try_into()?;
            let value = obj_resolve::load_member_dynamic(base, name, true).await?;
            stack.push(value);
            Ok(true)
        }
        crate::bytecode::Instr::StoreMember(field) => {
            let rhs = stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
            let base = stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
            let value = obj_resolve::store_member(base, field.clone(), rhs, false, |oldv, newv| {
                gc_record_write(oldv, newv);
            })
            .await?;
            stack.push(value);
            Ok(true)
        }
        crate::bytecode::Instr::StoreMemberOrInit(field) => {
            let rhs = stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
            let base = stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
            let value = obj_resolve::store_member(base, field.clone(), rhs, true, |oldv, newv| {
                gc_record_write(oldv, newv);
            })
            .await?;
            stack.push(value);
            Ok(true)
        }
        crate::bytecode::Instr::StoreMemberDynamic => {
            let rhs = stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
            let name_val = stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
            let base = stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
            let name: String = (&name_val).try_into()?;
            let value = obj_resolve::store_member_dynamic(base, name, rhs, false, |oldv, newv| {
                gc_record_write(oldv, newv);
            })
            .await?;
            stack.push(value);
            Ok(true)
        }
        crate::bytecode::Instr::StoreMemberDynamicOrInit => {
            let rhs = stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
            let name_val = stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
            let base = stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
            let name: String = (&name_val).try_into()?;
            let value = obj_resolve::store_member_dynamic(base, name, rhs, true, |oldv, newv| {
                gc_record_write(oldv, newv);
            })
            .await?;
            stack.push(value);
            Ok(true)
        }
        _ => Ok(false),
    }
}
