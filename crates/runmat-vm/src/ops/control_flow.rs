use crate::interpreter::stack::pop_value;
use runmat_builtins::Value;
use runmat_runtime::RuntimeError;

pub enum ControlFlowAction {
    Next,
    Jump(usize),
    Return,
}

#[inline]
pub fn and_and(stack: &mut Vec<Value>, target: usize) -> Result<ControlFlowAction, RuntimeError> {
    let lhs: f64 = (&pop_value(stack)?).try_into()?;
    if lhs == 0.0 {
        Ok(ControlFlowAction::Jump(target))
    } else {
        Ok(ControlFlowAction::Next)
    }
}

#[inline]
pub fn or_or(stack: &mut Vec<Value>, target: usize) -> Result<ControlFlowAction, RuntimeError> {
    let lhs: f64 = (&pop_value(stack)?).try_into()?;
    if lhs != 0.0 {
        Ok(ControlFlowAction::Jump(target))
    } else {
        Ok(ControlFlowAction::Next)
    }
}

#[inline]
pub fn jump_if_false(cond: bool, target: usize) -> ControlFlowAction {
    if cond {
        ControlFlowAction::Next
    } else {
        ControlFlowAction::Jump(target)
    }
}

#[inline]
pub fn jump(target: usize) -> ControlFlowAction {
    ControlFlowAction::Jump(target)
}

#[inline]
pub fn enter_try(
    try_stack: &mut Vec<(usize, Option<usize>)>,
    catch_pc: usize,
    catch_var: Option<usize>,
) {
    try_stack.push((catch_pc, catch_var));
}

#[inline]
pub fn pop_try(try_stack: &mut Vec<(usize, Option<usize>)>) {
    let _ = try_stack.pop();
}

#[inline]
pub fn enter_scope(locals: &mut Vec<Value>, local_count: usize) {
    for _ in 0..local_count {
        locals.push(Value::Num(0.0));
    }
}

pub fn exit_scope<OnPop>(locals: &mut Vec<Value>, local_count: usize, mut on_pop: OnPop)
where
    OnPop: FnMut(&Value),
{
    for _ in 0..local_count {
        if let Some(value) = locals.pop() {
            on_pop(&value);
        }
    }
}

#[inline]
pub fn return_value(stack: &mut Vec<Value>) -> Result<ControlFlowAction, RuntimeError> {
    let value = pop_value(stack)?;
    stack.push(value);
    Ok(ControlFlowAction::Return)
}

#[inline]
pub fn return_void() -> ControlFlowAction {
    ControlFlowAction::Return
}
