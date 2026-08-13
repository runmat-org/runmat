use crate::bytecode::FunctionRegistry;
use runmat_runtime::call::descriptor::{execute_callable_descriptor, CallableDescriptor};
use runmat_runtime::RuntimeError;
use runmat_value::Value;

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
