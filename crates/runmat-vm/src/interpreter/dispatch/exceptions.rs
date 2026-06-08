use crate::runtime::call_stack::error_namespace;
use crate::runtime::workspace::mark_workspace_assigned;
use runmat_builtins::{MException, Value};
use runmat_runtime::RuntimeError;

pub enum ExceptionHandling {
    Caught,
    Uncaught(Box<RuntimeError>),
}

pub fn parse_exception(err: &RuntimeError) -> MException {
    if let Some(identifier) = err.identifier() {
        return MException::new(identifier.to_string(), err.message().to_string());
    }
    let message = err.message();
    if let Some(idx) = message.rfind(": ") {
        let (id, msg) = message.split_at(idx);
        let message = msg.trim_start_matches(':').trim().to_string();
        let ident = if id.trim().is_empty() {
            format!("{}:error", error_namespace())
        } else {
            id.trim().to_string()
        };
        return MException::new(ident, message);
    }
    if let Some(idx) = message.rfind(':') {
        let (id, msg) = message.split_at(idx);
        let message = msg.trim_start_matches(':').trim().to_string();
        let ident = if id.trim().is_empty() {
            format!("{}:error", error_namespace())
        } else {
            id.trim().to_string()
        };
        return MException::new(ident, message);
    }
    MException::new(format!("{}:error", error_namespace()), message.to_string())
}

pub fn redirect_exception_to_catch(
    err: RuntimeError,
    try_stack: &mut Vec<(usize, Option<usize>)>,
    vars: &mut Vec<Value>,
    last_exception: &mut Option<MException>,
    pc: &mut usize,
    refresh_vars: impl Fn(&[Value]),
) -> ExceptionHandling {
    if let Some((catch_pc, catch_var)) = try_stack.pop() {
        if let Some(var_idx) = catch_var {
            if var_idx >= vars.len() {
                vars.resize(var_idx + 1, Value::Num(0.0));
                refresh_vars(vars);
            }
            let mex = parse_exception(&err);
            *last_exception = Some(mex.clone());
            vars[var_idx] = Value::MException(mex);
            mark_workspace_assigned(var_idx);
        }
        *pc = catch_pc;
        ExceptionHandling::Caught
    } else {
        ExceptionHandling::Uncaught(Box::new(err))
    }
}
