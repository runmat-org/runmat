use crate::bytecode::FunctionRegistry;
use crate::call::execute_callable_descriptor;
use runmat_runtime::call::descriptor::CallableDescriptor;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

pub enum FevalDispatch {
    Completed(runmat_runtime::execution::RootedValueSequence),
}

pub async fn execute_feval(
    runtime: &runmat_runtime::context::RuntimeContext,
    func_val: Value,
    args: Vec<Value>,
    requested_outputs: usize,
    function_registry: &FunctionRegistry,
) -> Result<FevalDispatch, RuntimeError> {
    let _roots = crate::call::root_values(args.iter().chain(std::iter::once(&func_val)))?;
    let descriptor =
        CallableDescriptor::from_feval_value(func_val, args, requested_outputs, function_registry);
    Ok(FevalDispatch::Completed(
        execute_callable_descriptor(runtime, descriptor).await?,
    ))
}
