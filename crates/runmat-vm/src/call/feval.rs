use crate::bytecode::SemanticFunctionRegistry;
use crate::call::descriptor::{execute_callable_descriptor, CallableDescriptor};
use runmat_builtins::Value;
use runmat_runtime::RuntimeError;

pub async fn forward_builtin_feval(
    func_value: Value,
    args: Vec<Value>,
    requested_outputs: usize,
) -> Result<Value, RuntimeError> {
    let mut argv = Vec::with_capacity(1 + args.len());
    argv.push(func_value);
    argv.extend(args);
    if requested_outputs == 1 {
        call_runtime_feval(&argv).await
    } else {
        runmat_runtime::call_builtin_async_with_outputs("feval", &argv, requested_outputs).await
    }
}

pub async fn call_runtime_feval(args: &[Value]) -> Result<Value, RuntimeError> {
    runmat_runtime::call_feval_async(args).await
}

pub enum FevalDispatch {
    Completed(Value),
}

pub async fn execute_feval(
    func_val: Value,
    args: Vec<Value>,
    requested_outputs: usize,
    semantic_registry: &SemanticFunctionRegistry,
) -> Result<FevalDispatch, RuntimeError> {
    let descriptor =
        CallableDescriptor::from_feval_value(func_val, args, requested_outputs, semantic_registry);
    Ok(FevalDispatch::Completed(
        execute_callable_descriptor(descriptor).await?,
    ))
}
