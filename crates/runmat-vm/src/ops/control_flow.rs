pub enum ControlFlowAction {
    Next,
    Jump(usize),
    Return,
}

#[inline]
pub fn and_and(lhs_truth: bool, target: usize) -> ControlFlowAction {
    if lhs_truth {
        ControlFlowAction::Next
    } else {
        ControlFlowAction::Jump(target)
    }
}

#[inline]
pub fn or_or(lhs_truth: bool, target: usize) -> ControlFlowAction {
    if lhs_truth {
        ControlFlowAction::Jump(target)
    } else {
        ControlFlowAction::Next
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
use crate::interpreter::stack::pop_value;
use runmat_builtins::Value;
use runmat_runtime::RuntimeError;
