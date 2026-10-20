use runmat_value::Value;

/// Root the fixed callback target before option parsing or provider awaits.
/// Tail inputs already carry their original root forest.
pub(crate) fn root_callable_value(
    value: &Value,
    context: Option<&crate::context::RuntimeContext>,
) -> crate::BuiltinResult<crate::execution::RootedValueLease> {
    match context {
        Some(context) => crate::execution::RootedValueLease::for_values_iterator_in_context(
            std::iter::once(value),
            context,
        ),
        None => {
            use crate::context::host_allocation::{
                host_allocation_runtime_error, HostAllocationError, HostAllocationPolicy,
                HostAllocationReceipt,
            };
            let allocation = |source| {
                host_allocation_runtime_error(
                    None,
                    "RunMat:MappedCallable:HostAllocation",
                    "mapped callback roots",
                    source,
                )
            };
            if HostAllocationReceipt::policy(None).map_err(allocation)?
                != HostAllocationPolicy::Unbounded
            {
                return Err(allocation(HostAllocationError::MissingManagedOwner));
            }
            crate::execution::RootedValueLease::for_values(std::iter::once(value))
                .map_err(|error| error.into_runtime_error())
        }
    }
}
