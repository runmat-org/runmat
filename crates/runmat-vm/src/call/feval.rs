use crate::call::user::try_builtin_fallback_single;
use runmat_builtins::{Closure, Value};
use runmat_runtime::RuntimeError;

pub fn closure_call_args(closure: &Closure, args: Vec<Value>) -> (String, Vec<Value>) {
    let name = closure.function_name.clone();
    let mut call_args = closure.captures.clone();
    call_args.extend(args);
    (name, call_args)
}

pub async fn forward_builtin_feval(
    func_value: Value,
    args: Vec<Value>,
) -> Result<Value, RuntimeError> {
    let mut argv = Vec::with_capacity(1 + args.len());
    argv.push(func_value);
    argv.extend(args);
    runmat_runtime::call_builtin_async("feval", &argv).await
}

pub async fn try_closure_builtin_fallback(
    name: &str,
    call_args: &[Value],
) -> Result<Option<Value>, RuntimeError> {
    try_builtin_fallback_single(name, call_args).await
}
