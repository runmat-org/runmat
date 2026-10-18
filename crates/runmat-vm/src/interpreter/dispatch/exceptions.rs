use crate::interpreter::state::ActiveTryHandler;
use crate::interpreter::vars::InterpreterVars;
use runmat_runtime::RuntimeError;
use runmat_value::{MException, Value};

pub enum ExceptionHandling {
    Caught,
    Uncaught(Box<RuntimeError>),
}

pub fn redirect_exception_to_catch(
    err: RuntimeError,
    try_stack: &mut Vec<ActiveTryHandler>,
    vars: &mut InterpreterVars,
    last_exception: &mut Option<MException>,
    pc: &mut usize,
) -> ExceptionHandling {
    if let Some(handler) = try_stack.pop() {
        let mex = runmat_runtime::runtime_error::exception_from_error(err);
        *last_exception = Some(mex.clone());
        if let Some(var_idx) = handler.catch_var {
            let value = Value::MException(mex);
            if let Err(error) =
                crate::runtime::workspace::commit_workspace_slot_value(vars, var_idx, None, value)
            {
                return ExceptionHandling::Uncaught(Box::new(error));
            }
        }
        *pc = handler.catch_pc;
        ExceptionHandling::Caught
    } else {
        ExceptionHandling::Uncaught(Box::new(err))
    }
}
