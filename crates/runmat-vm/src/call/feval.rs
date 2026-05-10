use crate::bytecode::{SemanticFunctionRegistry, UserFunction};
use crate::call::user::try_builtin_fallback_single;
use crate::interpreter::runner::dynamic_user_functions_snapshot;
use runmat_builtins::{Closure, Value};
use runmat_runtime::RuntimeError;
use std::collections::HashMap;

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
    call_runtime_feval(&argv).await
}

pub async fn call_runtime_feval(args: &[Value]) -> Result<Value, RuntimeError> {
    runmat_runtime::call_builtin_async("feval", args).await
}

pub async fn try_closure_builtin_fallback(
    name: &str,
    call_args: &[Value],
) -> Result<Option<Value>, RuntimeError> {
    try_builtin_fallback_single(name, call_args).await
}

pub enum FevalDispatch {
    Completed(Value),
    InvokeUser {
        name: String,
        args: Vec<Value>,
        functions: HashMap<String, UserFunction>,
    },
}

pub async fn execute_feval(
    func_val: Value,
    args: Vec<Value>,
    requested_outputs: usize,
    context_functions: &HashMap<String, UserFunction>,
    bytecode_functions: &HashMap<String, UserFunction>,
    semantic_registry: &SemanticFunctionRegistry,
) -> Result<FevalDispatch, RuntimeError> {
    match func_val {
        Value::Closure(c) => {
            let (name, call_args) = closure_call_args(&c, args);
            if let Some(function) = c.semantic_function {
                if let Some(result) = runmat_runtime::user_functions::try_call_semantic_function(
                    function,
                    &call_args,
                    requested_outputs,
                )
                .await
                {
                    return Ok(FevalDispatch::Completed(result?));
                }
            }
            if let Some(result) = try_closure_builtin_fallback(&name, &call_args).await? {
                return Ok(FevalDispatch::Completed(result));
            }
            let mut functions = bytecode_functions.clone();
            for (k, v) in context_functions {
                functions.insert(k.clone(), v.clone());
            }
            for (k, v) in dynamic_user_functions_snapshot() {
                functions.insert(k, v);
            }
            Ok(FevalDispatch::InvokeUser {
                name,
                args: call_args,
                functions,
            })
        }
        Value::FunctionHandle(name) => {
            if let Some(function) = semantic_registry.resolve_name(&name) {
                if let Some(result) = runmat_runtime::user_functions::try_call_semantic_function(
                    function.0,
                    &args,
                    requested_outputs,
                )
                .await
                {
                    return Ok(FevalDispatch::Completed(result?));
                }
            }
            Ok(FevalDispatch::Completed(
                forward_builtin_feval(Value::FunctionHandle(name), args).await?,
            ))
        }
        other => Ok(FevalDispatch::Completed(
            forward_builtin_feval(other, args).await?,
        )),
    }
}
