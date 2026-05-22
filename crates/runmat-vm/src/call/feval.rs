use crate::bytecode::FunctionRegistry;
use crate::call::descriptor::{execute_callable_descriptor, CallableDescriptor};
use runmat_builtins::Value;
use runmat_runtime::RuntimeError;

pub async fn forward_builtin_feval(
    func_value: Value,
    args: Vec<Value>,
    requested_outputs: usize,
) -> Result<Value, RuntimeError> {
    runmat_runtime::call_feval_async_with_outputs(func_value, &args, requested_outputs).await
}

pub enum FevalDispatch {
    Completed(Value),
}

pub async fn execute_feval(
    func_val: Value,
    args: Vec<Value>,
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
) -> Result<FevalDispatch, RuntimeError> {
    let descriptor =
        CallableDescriptor::from_feval_value(func_val, args, requested_outputs, function_registry);
    Ok(FevalDispatch::Completed(
        execute_callable_descriptor(descriptor).await?,
    ))
}
