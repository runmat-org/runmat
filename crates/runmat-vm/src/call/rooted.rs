//! Managed roots follow call-owned values after they leave the VM frame.
use crate::runtime::gc::ManagedRoots;
use runmat_runtime::call::descriptor::{CallableDescriptor, CallableTarget};
use runmat_runtime::RuntimeError;
use runmat_value::{Value, ValueSequence};

pub(crate) fn root_values<'a>(
    values: impl IntoIterator<Item = &'a Value>,
) -> Result<ManagedRoots, RuntimeError> {
    ManagedRoots::from_values(values).map_err(RuntimeError::new)
}

pub(crate) async fn execute_callable_descriptor(
    descriptor: CallableDescriptor,
) -> Result<ValueSequence, RuntimeError> {
    let callable = match &descriptor.target {
        CallableTarget::FevalForward(value) => Some(value),
        CallableTarget::Resolved { .. } => None,
    };
    let _roots = root_values(descriptor.args.iter().chain(callable))?;
    runmat_runtime::call::descriptor::execute_callable_descriptor(descriptor).await
}

pub(crate) async fn call_builtin_async(name: &str, args: &[Value]) -> Result<Value, RuntimeError> {
    let _roots = root_values(args)?;
    runmat_runtime::call_builtin_async(name, args).await
}

pub(crate) async fn call_builtin_async_with_outputs(
    name: &str,
    args: &[Value],
    outputs: usize,
) -> Result<Value, RuntimeError> {
    let _roots = root_values(args)?;
    runmat_runtime::call_builtin_async_with_outputs(name, args, outputs).await
}
