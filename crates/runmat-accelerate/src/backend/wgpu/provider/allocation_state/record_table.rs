//! Physical records have two table domains. Ordinary unbounded registrations
//! retain their existing table policy; selected finite registrations use a
//! separately funded table so an unrelated insertion cannot consume a
//! pre-effect reservation or replace its retained backing claim.

use super::{AllocationStateError, Record};
use runmat_execution::resource::RetainedHostAllocationClaim;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[derive(Default)]
pub(super) struct RecordTable {
    ordinary: HashMap<u64, Record>,
    managed: hashbrown::HashMap<u64, Record>,
    managed_claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
    outstanding: Arc<AtomicUsize>,
}

/// Old table backing remains owned until it is dropped outside the records
/// lock. The allocation claim must outlive the table it funded.
pub(super) struct RetiredManagedBacking {
    _table: hashbrown::HashMap<u64, Record>,
    _claim: Arc<dyn RetainedHostAllocationClaim>,
}

impl RecordTable {
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.ordinary.len() + self.managed.len()
    }

    pub(super) fn values(&self) -> impl Iterator<Item = &Record> {
        self.ordinary.values().chain(self.managed.values())
    }
    pub(super) fn has_scoped_records(
        &self,
        scope: runmat_accelerate_api::DeviceAllocationScope,
    ) -> bool {
        self.values()
            .any(|record| record.scope.is_some_and(|created| scope.covers(created)))
    }
    pub(super) fn scoped_count(
        &self,
        scope: runmat_accelerate_api::DeviceAllocationScope,
    ) -> usize {
        self.values()
            .filter(|record| record.scope.is_some_and(|created| scope.covers(created)))
            .count()
    }
    pub(super) fn scoped_ids(
        &self,
        scope: runmat_accelerate_api::DeviceAllocationScope,
    ) -> anyhow::Result<Vec<u64>> {
        let mut ids = runmat_value::exact_capacity_vec(self.scoped_count(scope))?;
        for (id, record) in self.ordinary.iter().chain(self.managed.iter()) {
            if record.scope.is_some_and(|created| scope.covers(created)) {
                ids.push(*id);
            }
        }
        Ok(ids)
    }

    pub(super) fn get(&self, id: &u64) -> Option<&Record> {
        self.managed.get(id).or_else(|| self.ordinary.get(id))
    }

    pub(super) fn get_mut(&mut self, id: &u64) -> Option<&mut Record> {
        if self.managed.contains_key(id) {
            self.managed.get_mut(id)
        } else {
            self.ordinary.get_mut(id)
        }
    }

    pub(super) fn contains_key(&self, id: &u64) -> bool {
        self.managed.contains_key(id) || self.ordinary.contains_key(id)
    }

    pub(super) fn remove(&mut self, id: &u64) -> Option<Record> {
        let mut record = self
            .managed
            .remove(id)
            .or_else(|| self.ordinary.remove(id))?;
        // Return reserved capacity while removal is still under the table
        // lock. A later admission cannot take an uncounted retirement gap.
        if let Some(mut slot) = record.registration_slot.take() {
            slot.release();
        }
        Some(record)
    }

    pub(super) fn insert(&mut self, id: u64, record: Record) -> Option<Record> {
        self.ordinary.insert(id, record)
    }

    pub(super) fn insert_managed(&mut self, id: u64, record: Record) -> Option<Record> {
        self.managed.insert(id, record)
    }

    pub(super) fn managed_slot_is_prepared(&self) -> bool {
        self.managed
            .len()
            .checked_add(self.outstanding.load(Ordering::Acquire))
            .is_some_and(|needed| needed <= self.managed.capacity())
    }

    pub(super) fn try_reserve(
        &mut self,
        additional: usize,
    ) -> Result<(), std::collections::TryReserveError> {
        self.ordinary.try_reserve(additional)
    }

    pub(super) fn into_records(
        self,
    ) -> (
        impl Iterator<Item = (u64, Record)>,
        Option<Arc<dyn RetainedHostAllocationClaim>>,
    ) {
        (
            self.ordinary.into_iter().chain(self.managed),
            self.managed_claim,
        )
    }

    /// A pinned hashbrown 0.16 table uses a power-of-two bucket count with
    /// 7/8 maximum occupancy. Its RawTable layout is bucket bytes, aligned
    /// control offset, then bucket-count plus one SIMD group of control bytes.
    /// Using the largest supported group width (16) is a checked upper bound
    /// on both native and portable WASM configurations, not an RSS estimate.
    fn backing_upper_for_entries(entries: usize) -> Option<u64> {
        if entries == 0 {
            return Some(0);
        }
        let buckets = if entries < 4 {
            4usize
        } else if entries < 8 {
            8
        } else if entries < 15 {
            16
        } else {
            entries
                .checked_mul(8)?
                .checked_div(7)?
                .checked_add(1)?
                .checked_next_power_of_two()?
        };
        let align = std::mem::align_of::<(u64, Record)>().max(16);
        let bucket_bytes = buckets.checked_mul(std::mem::size_of::<(u64, Record)>())?;
        let control_offset = bucket_bytes.checked_add(align - 1)? & !(align - 1);
        let total = control_offset.checked_add(buckets)?.checked_add(16)?;
        u64::try_from(total).ok()
    }

    pub(super) fn managed_backing_upper(&self, additional: usize) -> Option<u64> {
        let required = self
            .managed
            .len()
            .checked_add(self.outstanding.load(Ordering::Acquire))?
            .checked_add(additional)?;
        if required <= self.managed.capacity() && self.managed_claim.is_some() {
            return Some(0);
        }
        if required <= self.managed.capacity() {
            return u64::try_from(self.managed.allocation_size()).ok();
        }
        Self::backing_upper_for_entries(required)
    }

    /// The caller obtained `backing_claim` before any table allocation. If
    /// another request changed the table after the quote, fail pre-effect and
    /// let placement replan; never allocate above the admitted bound.
    pub(super) fn reserve_managed(
        &mut self,
        additional: usize,
        charged_upper: u64,
        backing_claim: &Option<Arc<dyn RetainedHostAllocationClaim>>,
    ) -> Result<(Arc<AtomicUsize>, Option<RetiredManagedBacking>), AllocationStateError> {
        let current_upper = self
            .managed_backing_upper(additional)
            .ok_or(AllocationStateError::Capacity)?;
        if current_upper > charged_upper || (current_upper > 0 && backing_claim.is_none()) {
            return Err(AllocationStateError::Capacity);
        }
        let outstanding = self.outstanding.load(Ordering::Acquire);
        let required = self
            .managed
            .len()
            .checked_add(outstanding)
            .and_then(|count| count.checked_add(additional))
            .ok_or(AllocationStateError::Capacity)?;
        // A used table can have tombstones. `try_reserve` on it may grow to
        // max(required, old_full_capacity + 1), larger than the clean-table
        // quote. Allocate a fresh table under the new claim instead, then
        // move records only after its measured allocation fits that claim.
        let retired = if required > self.managed.capacity() {
            let mut replacement = hashbrown::HashMap::new();
            replacement
                .try_reserve(required)
                .map_err(|_| AllocationStateError::Capacity)?;
            if u64::try_from(replacement.allocation_size())
                .map_err(|_| AllocationStateError::Capacity)?
                > charged_upper
            {
                return Err(AllocationStateError::Capacity);
            }
            for (id, record) in self.managed.drain() {
                replacement.insert(id, record);
            }
            let old_table = std::mem::replace(&mut self.managed, replacement);
            self.managed_claim
                .replace(
                    backing_claim
                        .as_ref()
                        .ok_or(AllocationStateError::Capacity)?
                        .clone(),
                )
                .map(|claim| RetiredManagedBacking {
                    _table: old_table,
                    _claim: claim,
                })
        } else {
            if current_upper > 0 {
                self.managed_claim = backing_claim.clone();
            }
            None
        };
        self.outstanding
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(additional)
            })
            .map_err(|_| AllocationStateError::Capacity)?;
        Ok((self.outstanding.clone(), retired))
    }

    pub(super) fn detach_empty_managed(&mut self) -> Option<RetiredManagedBacking> {
        if !self.managed.is_empty() || self.outstanding.load(Ordering::Acquire) != 0 {
            return None;
        }
        let claim = self.managed_claim.take()?;
        Some(RetiredManagedBacking {
            _table: std::mem::take(&mut self.managed),
            _claim: claim,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_table_layout_bound_covers_ordinary_slot_counts() {
        for count in [1usize, 3, 4, 7, 8, 14, 15, 16, 64, 128] {
            let mut table = hashbrown::HashMap::<u64, Record>::new();
            table.try_reserve(count).unwrap();
            assert!(
                u64::try_from(table.allocation_size()).unwrap()
                    <= RecordTable::backing_upper_for_entries(count).unwrap()
            );
        }
    }

    #[test]
    fn clean_replacement_bound_survives_tombstone_churn() {
        let mut used = hashbrown::HashMap::<u64, u64>::new();
        used.try_reserve(64).unwrap();
        for id in 0..64_u64 {
            used.insert(id, id);
        }
        for id in 0..60_u64 {
            used.remove(&id);
        }
        let required = used.len() + 128;
        let mut replacement = hashbrown::HashMap::<u64, Record>::new();
        replacement.try_reserve(required).unwrap();
        assert!(
            u64::try_from(replacement.allocation_size()).unwrap()
                <= RecordTable::backing_upper_for_entries(required).unwrap()
        );
        // The selected owner never asks the churned table to grow: the
        // replacement is prepared and charged before any record is moved.
        assert!(used.capacity() < required);
    }
}
