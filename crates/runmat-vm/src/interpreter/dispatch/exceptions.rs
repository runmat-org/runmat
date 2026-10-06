use crate::interpreter::state::ActiveTryHandler;
use crate::runtime::workspace::admit_workspace_slot_write;
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
    _refresh_vars: impl Fn(&[Value]),
) -> ExceptionHandling {
    if let Some(handler) = try_stack.pop() {
        let mex = runmat_runtime::runtime_error::exception_from_error(err);
        *last_exception = Some(mex.clone());
        if let Some(var_idx) = handler.catch_var {
            if var_idx >= vars.len() {
                vars.resize(var_idx + 1, Value::Num(0.0));
            }
            let value = Value::MException(mex);
            if let Err(error) = admit_workspace_slot_write(var_idx, &value) {
                return ExceptionHandling::Uncaught(Box::new(error));
            }
            vars[var_idx] = value;
        }
        *pc = handler.catch_pc;
        ExceptionHandling::Caught
    } else {
        ExceptionHandling::Uncaught(Box::new(err))
    }
}
