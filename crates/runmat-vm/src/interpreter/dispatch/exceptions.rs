use crate::interpreter::state::ActiveTryHandler;
use crate::runtime::workspace::mark_workspace_assigned;
use runmat_runtime::RuntimeError;
use runmat_value::{MException, Value};

pub enum ExceptionHandling {
    Caught,
    Uncaught(Box<RuntimeError>),
}

pub fn redirect_exception_to_catch(
    err: RuntimeError,
    try_stack: &mut Vec<ActiveTryHandler>,
    vars: &mut Vec<Value>,
    last_exception: &mut Option<MException>,
    pc: &mut usize,
    refresh_vars: impl Fn(&[Value]),
) -> ExceptionHandling {
    if let Some(handler) = try_stack.pop() {
        if let Some(var_idx) = handler.catch_var {
            if var_idx >= vars.len() {
                vars.resize(var_idx + 1, Value::Num(0.0));
                refresh_vars(vars);
            }
            let mex = runmat_runtime::runtime_error::exception_from_error(&err);
            *last_exception = Some(mex.clone());
            vars[var_idx] = Value::MException(mex);
            refresh_vars(vars);
            mark_workspace_assigned(var_idx);
        }
        *pc = handler.catch_pc;
        ExceptionHandling::Caught
    } else {
        ExceptionHandling::Uncaught(Box::new(err))
    }
}
