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
    runtime: &runmat_runtime::context::RuntimeContext,
    descriptor: CallableDescriptor,
) -> Result<ValueSequence, RuntimeError> {
    let callable = match &descriptor.target {
        CallableTarget::FevalForward(value) => Some(value),
        // Bound authority retains definition leases, not captured GC values.
        CallableTarget::Bound(_) => None,
        CallableTarget::Resolved { .. } => None,
    };
    let _roots = root_values(descriptor.args.iter().chain(callable))?;
    runmat_runtime::call::descriptor::execute_callable_descriptor_in_context(runtime, descriptor)
        .await
}

pub(crate) async fn call_builtin_async(
    runtime: &runmat_runtime::context::RuntimeContext,
    name: &str,
    args: &[Value],
) -> Result<Value, RuntimeError> {
    let _roots = root_values(args)?;
    runmat_runtime::call_builtin_async_in_context(runtime, name, args).await
}

pub(crate) async fn call_builtin_async_with_outputs(
    runtime: &runmat_runtime::context::RuntimeContext,
    name: &str,
    args: &[Value],
    outputs: usize,
) -> Result<Value, RuntimeError> {
    let _roots = root_values(args)?;
    runmat_runtime::call_builtin_async_with_outputs_in_context(runtime, name, args, outputs).await
}
