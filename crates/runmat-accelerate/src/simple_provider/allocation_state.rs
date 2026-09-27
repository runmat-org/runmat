//! The single InProcess payload map and its owning allocation associations.
//! Pending publication and live leases share this map; there is no ID-to-owner
//! side registry. Typed guards borrow payloads and never copy them.
use super::InProcessNumericData;
use runmat_accelerate_api::allocation::{
    AllocationIdentity, AllocationRetirementOutcome, GpuPhysicalDescriptor, WeakAllocationLease,
};
use std::collections::HashMap;
use std::sync::{Arc, LockResult, Mutex};

mod guards;
mod ownership;
mod publication;
mod payload;
mod registry;
pub(super) use guards::{F64Registry, IntegerRegistry, NumericRegistryGuard};
pub(super) use publication::PendingAllocation;
pub(super) use registry::{integer_registry, numeric_registry, registry};

enum Phase {
    Pending,
    Live(WeakAllocationLease),
    Retiring,
    Failed(AllocationRetirementOutcome),
}

struct Record {
    identity: AllocationIdentity,
    physical: GpuPhysicalDescriptor,
    producer_logical: bool,
    payload: Option<InProcessNumericData>,
    phase: Phase,
}

#[derive(Default)]
pub(super) struct NumericAllocationState {
    allocations: Mutex<HashMap<u64, Record>>,
}

impl NumericAllocationState {
    pub(super) fn lock(&self) -> LockResult<NumericRegistryGuard<'_>> {
        self.allocations
            .lock()
            .map(|guard| NumericRegistryGuard::new(guard, true))
            .map_err(|error| {
                std::sync::PoisonError::new(NumericRegistryGuard::new(error.into_inner(), false))
            })
    }
}

#[cfg(test)]
mod tests;
