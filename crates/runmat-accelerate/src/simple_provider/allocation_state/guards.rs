use super::{numeric_registry, InProcessNumericData, Phase, Record};
use runmat_accelerate_api::allocation::AllocationLease;
use runmat_accelerate_api::{GpuTensorHandle, HostIntegerDataOwned};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{LockResult, MutexGuard, PoisonError};

pub(in crate::simple_provider) struct NumericRegistryGuard<'a> {
    // Unlock before retained leases can retire and re-enter this owner.
    records: MutexGuard<'a, HashMap<u64, Record>>,
    retained: RefCell<Vec<AllocationLease>>,
    healthy: bool,
}
impl<'a> NumericRegistryGuard<'a> {
    pub(super) fn new(guard: MutexGuard<'a, HashMap<u64, Record>>, healthy: bool) -> Self {
        Self {
            records: guard,
            retained: RefCell::new(Vec::new()),
            healthy,
        }
    }

    fn live_record(&self, handle: &GpuTensorHandle) -> Option<&Record> {
        if !self.healthy {
            return None;
        }
        let record = self.records.get(&handle.buffer_id())?;
        if record.identity != handle.allocation_identity() {
            return None;
        }
        let Phase::Live(weak) = &record.phase else {
            return None;
        };
        // The argument's strong lease prevents a last-reference destructor
        // while this short validation reference is released under the map lock.
        let lease = weak.acquire()?;
        if !lease.shares_allocation_with(handle.allocation_lease()) {
            return None;
        }
        let mut retained = self.retained.try_borrow_mut().ok()?;
        if !retained
            .iter()
            .any(|existing| existing.shares_allocation_with(&lease))
        {
            retained.push(handle.allocation_lease().clone());
        }
        Some(record)
    }

    pub(in crate::simple_provider) fn get(
        &self,
        handle: &GpuTensorHandle,
    ) -> Option<&InProcessNumericData> {
        self.live_record(handle)?.payload.as_ref()
    }

    fn writable(
        &mut self,
        handle: &GpuTensorHandle,
        logical_values: bool,
    ) -> Option<&mut InProcessNumericData> {
        let record = self.live_record(handle)?;
        if record.producer_logical && !logical_values {
            return None;
        }
        self.records.get_mut(&handle.buffer_id())?.payload.as_mut()
    }
}

pub(in crate::simple_provider) struct F64Registry;
pub(in crate::simple_provider) struct F64RegistryGuard<'a>(NumericRegistryGuard<'a>);
impl F64Registry {
    pub(in crate::simple_provider) fn lock(&self) -> LockResult<F64RegistryGuard<'_>> {
        numeric_registry()
            .lock()
            .map(F64RegistryGuard)
            .map_err(|error| PoisonError::new(F64RegistryGuard(error.into_inner())))
    }
}
impl F64RegistryGuard<'_> {
    pub(in crate::simple_provider) fn get(&self, handle: &GpuTensorHandle) -> Option<&Vec<f64>> {
        match self.0.get(handle)? {
            InProcessNumericData::F64(data) => Some(data),
            _ => None,
        }
    }
    pub(in crate::simple_provider) fn get_mut_for_write(
        &mut self,
        handle: &GpuTensorHandle,
        values: &[f64],
    ) -> Option<&mut Vec<f64>> {
        let logical = values.iter().all(|value| *value == 0.0 || *value == 1.0);
        match self.0.writable(handle, logical)? {
            InProcessNumericData::F64(data) => Some(data),
            _ => None,
        }
    }
}

pub(in crate::simple_provider) struct IntegerRegistry;
pub(in crate::simple_provider) struct IntegerRegistryGuard<'a>(NumericRegistryGuard<'a>);
impl IntegerRegistry {
    pub(in crate::simple_provider) fn lock(&self) -> LockResult<IntegerRegistryGuard<'_>> {
        numeric_registry()
            .lock()
            .map(IntegerRegistryGuard)
            .map_err(|error| PoisonError::new(IntegerRegistryGuard(error.into_inner())))
    }
}
impl IntegerRegistryGuard<'_> {
    pub(in crate::simple_provider) fn get(
        &self,
        handle: &GpuTensorHandle,
    ) -> Option<&HostIntegerDataOwned> {
        match self.0.get(handle)? {
            InProcessNumericData::Integer(data) => Some(data),
            _ => None,
        }
    }
    pub(in crate::simple_provider) fn get_mut_for_write(
        &mut self,
        handle: &GpuTensorHandle,
        values: &HostIntegerDataOwned,
    ) -> Option<&mut HostIntegerDataOwned> {
        // The only admitted integer logical encoding is U8.
        let logical =
            matches!(values, HostIntegerDataOwned::U8(data) if data.iter().all(|v| *v <= 1));
        match self.0.writable(handle, logical)? {
            InProcessNumericData::Integer(data) => Some(data),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests;
