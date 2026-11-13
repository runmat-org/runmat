//! Construct directly in the final rooted collection's admitted backing.

use runmat_value::{exact_capacity_vec, OutputListInitializationError, OutputListStorage, Value};

use crate::context::{
    host_allocation::{
        host_allocation_runtime_error, HostAllocationError, HostAllocationPlan,
        HostAllocationPolicy, HostAllocationReceipt,
    },
    RuntimeContext,
};
use crate::RuntimeError;

use super::RootedValues;

impl RootedValues {
    /// Admit the final physical collection before invoking `initialize`.
    /// Each returned Value moves directly into its final slot, without an
    /// intermediate Value vector or per-element root registrations.
    ///
    /// The caller MUST keep every source/returned managed Value independently
    /// rooted until this method returns, including across subsequent initializer
    /// calls. This method registers the completed collection's roots only after
    /// all elements have been prepared. It is not a per-child decoding/rooting
    /// API for newly created, otherwise unprotected GC handles.
    /// Allocation failures enter the caller's existing error domain; element
    /// failures are returned unchanged, including executor control transfers.
    pub fn try_initialize_in_context<E: From<RuntimeError>>(
        count: usize,
        runtime: &RuntimeContext,
        mut initialize: impl FnMut(usize) -> Result<Value, E>,
    ) -> Result<Self, E> {
        let allocation = |source| {
            E::from(host_allocation_runtime_error(
                Some(runtime),
                "RunMat:Call:Arguments",
                "rooted collection initialization failed",
                source,
            ))
        };
        let raw = OutputListStorage::requested_bytes(count)
            .map_err(|source| allocation(HostAllocationError::Numeric(source)))?;
        let raw = u64::try_from(raw).map_err(|_| allocation(HostAllocationError::Overflow))?;
        let retained =
            HostAllocationPlan::retained_exact_output_bytes(raw, usize::from(count != 0))
                .map_err(allocation)?;
        let plan = HostAllocationPlan::new(0, retained).map_err(allocation)?;
        let receipt = HostAllocationReceipt::admit(Some(runtime), plan).map_err(allocation)?;

        let check_unbounded = || match HostAllocationReceipt::policy(Some(runtime))
            .map_err(allocation)?
        {
            HostAllocationPolicy::Unbounded => Ok(()),
            HostAllocationPolicy::Managed => Err(allocation(HostAllocationError::StaleAuthority)),
        };
        let storage = if let Some(receipt) = &receipt {
            let permit = receipt.output_permit(runtime).map_err(allocation)?;
            let result = OutputListStorage::try_initialize(count, &permit, |index| {
                receipt.checkpoint(runtime).map_err(allocation)?;
                let value = initialize(index)?;
                receipt.checkpoint(runtime).map_err(allocation)?;
                Ok(value)
            });
            match result {
                Ok(storage) => storage,
                Err(OutputListInitializationError::Element(error)) => return Err(error),
                Err(OutputListInitializationError::Allocation(source)) => {
                    return Err(allocation(
                        receipt
                            .checkpoint(runtime)
                            .err()
                            .unwrap_or(HostAllocationError::Numeric(source)),
                    ));
                }
            }
        } else {
            check_unbounded()?;
            let mut values = exact_capacity_vec(count)
                .map_err(|source| allocation(HostAllocationError::Numeric(source)))?;
            for index in 0..count {
                check_unbounded()?;
                let value = initialize(index)?;
                check_unbounded()?;
                values.push(value);
            }
            OutputListStorage::shared_unbounded(values)
        };
        if let Some(receipt) = &receipt {
            receipt.checkpoint(runtime).map_err(allocation)?;
        } else {
            check_unbounded()?;
        }
        let result = Self::from_output_storage_in_context(storage, runtime).map_err(E::from)?;
        if let Some(receipt) = &receipt {
            receipt.checkpoint(runtime).map_err(allocation)?;
        } else {
            check_unbounded()?;
        }
        Ok(result)
    }
}
