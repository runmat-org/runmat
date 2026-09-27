//! The single InProcess numeric allocation table and its typed views.
//!
//! This module preserves current explicit-free semantics. It does not introduce
//! handle leases; later ownership changes must use this same allocation owner.
use super::InProcessNumericData;
use once_cell::sync::OnceCell;
use runmat_accelerate_api::HostIntegerDataOwned;
use std::collections::HashMap;
use std::sync::{Arc, LockResult, Mutex, MutexGuard, PoisonError};

#[derive(Default)]
pub(super) struct NumericAllocationState {
    allocations: Mutex<HashMap<u64, InProcessNumericData>>,
}

impl NumericAllocationState {
    pub(super) fn lock(&self) -> LockResult<MutexGuard<'_, HashMap<u64, InProcessNumericData>>> {
        self.allocations.lock()
    }

    /// Detach under the map lock, then transfer physical teardown to the caller.
    /// No payload destructor or metadata callback runs while the lock is held.
    pub(super) fn take(&self, id: u64) -> Option<InProcessNumericData> {
        let mut allocations = self.lock().unwrap_or_else(|error| error.into_inner());
        allocations.remove(&id)
    }
}

static NUMERIC_REGISTRY: OnceCell<Arc<NumericAllocationState>> = OnceCell::new();

pub(super) fn numeric_registry() -> &'static Arc<NumericAllocationState> {
    NUMERIC_REGISTRY.get_or_init(|| Arc::new(NumericAllocationState::default()))
}

pub(super) struct F64Registry;

pub(super) struct F64RegistryGuard<'a>(MutexGuard<'a, HashMap<u64, InProcessNumericData>>);

impl F64Registry {
    pub(super) fn lock(&self) -> LockResult<F64RegistryGuard<'_>> {
        numeric_registry()
            .lock()
            .map(F64RegistryGuard)
            .map_err(|error| PoisonError::new(F64RegistryGuard(error.into_inner())))
    }
}

impl F64RegistryGuard<'_> {
    pub(super) fn get(&self, id: &u64) -> Option<&Vec<f64>> {
        match self.0.get(id) {
            Some(InProcessNumericData::F64(values)) => Some(values),
            Some(InProcessNumericData::F32(_) | InProcessNumericData::Integer(_)) | None => None,
        }
    }

    pub(super) fn get_mut(&mut self, id: &u64) -> Option<&mut Vec<f64>> {
        match self.0.get_mut(id) {
            Some(InProcessNumericData::F64(values)) => Some(values),
            Some(InProcessNumericData::F32(_) | InProcessNumericData::Integer(_)) | None => None,
        }
    }

    pub(super) fn insert(&mut self, id: u64, values: Vec<f64>) -> Option<Vec<f64>> {
        match self.0.insert(id, InProcessNumericData::F64(values)) {
            Some(InProcessNumericData::F64(previous)) => Some(previous),
            Some(InProcessNumericData::F32(_) | InProcessNumericData::Integer(_)) | None => None,
        }
    }

    #[cfg(test)]
    pub(super) fn contains_key(&self, id: &u64) -> bool {
        matches!(self.0.get(id), Some(InProcessNumericData::F64(_)))
    }
}

static F64_REGISTRY_VIEW: F64Registry = F64Registry;

pub(super) fn registry() -> &'static F64Registry {
    &F64_REGISTRY_VIEW
}

pub(super) struct IntegerRegistry;

pub(super) struct IntegerRegistryGuard<'a>(MutexGuard<'a, HashMap<u64, InProcessNumericData>>);

impl IntegerRegistry {
    pub(super) fn lock(&self) -> LockResult<IntegerRegistryGuard<'_>> {
        numeric_registry()
            .lock()
            .map(IntegerRegistryGuard)
            .map_err(|error| PoisonError::new(IntegerRegistryGuard(error.into_inner())))
    }
}

impl IntegerRegistryGuard<'_> {
    pub(super) fn get(&self, id: &u64) -> Option<&HostIntegerDataOwned> {
        match self.0.get(id) {
            Some(InProcessNumericData::Integer(values)) => Some(values),
            Some(InProcessNumericData::F64(_) | InProcessNumericData::F32(_)) | None => None,
        }
    }

    pub(super) fn get_mut(&mut self, id: &u64) -> Option<&mut HostIntegerDataOwned> {
        match self.0.get_mut(id) {
            Some(InProcessNumericData::Integer(values)) => Some(values),
            Some(InProcessNumericData::F64(_) | InProcessNumericData::F32(_)) | None => None,
        }
    }

    pub(super) fn insert(
        &mut self,
        id: u64,
        values: HostIntegerDataOwned,
    ) -> Option<HostIntegerDataOwned> {
        match self.0.insert(id, InProcessNumericData::Integer(values)) {
            Some(InProcessNumericData::Integer(previous)) => Some(previous),
            Some(InProcessNumericData::F64(_) | InProcessNumericData::F32(_)) | None => None,
        }
    }
}

static INTEGER_REGISTRY_VIEW: IntegerRegistry = IntegerRegistry;

pub(super) fn integer_registry() -> &'static IntegerRegistry {
    &INTEGER_REGISTRY_VIEW
}

#[cfg(test)]
mod tests;
