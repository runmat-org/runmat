//! Source-ordered continuation entries with an admitted lookup index.
//!
//! The index never owns a second semantic value. Growing both physical blocks
//! is prepared against the current operation before any entry is moved, and
//! each block keeps its own collection claim after the permit is uninstalled.

use allocator_api2::vec::Vec as AllocatedVec;
use hashbrown::HashMap;
use runmat_runtime::context::host_allocation::{
    collection_block_retained_bytes, frame_hash_table_retained_bytes, FrameCollectionGate,
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt, OperationCollectionAllocator,
};
use runmat_runtime::context::RuntimeContext;
use runmat_value::ExactHostNumericAllocationError;
use std::alloc::Layout;
use std::collections::hash_map::RandomState;
use std::hash::Hash;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::continuation_admission_error;

type Index<K> = HashMap<K, usize, RandomState, OperationCollectionAllocator>;

pub(super) fn prepare_vector_push<T>(
    values: &mut AllocatedVec<T, OperationCollectionAllocator>,
    runtime: &RuntimeContext,
    gate: &FrameCollectionGate,
) -> NativeExecutorResult<()> {
    if values.len() < values.capacity() {
        return Ok(());
    }
    let target = doubled_capacity(values.capacity(), values.len())?;
    let layout = Layout::array::<T>(target).map_err(|_| {
        NativeExecutorError::Host("native continuation vector layout overflow".into())
    })?;
    let retained = collection_block_retained_bytes(layout)
        .map_err(|cause| continuation_admission_error(runtime, cause))?;
    with_collection_growth(runtime, gate, retained, |allocator| {
        let mut replacement = AllocatedVec::new_in(allocator);
        replacement
            .try_reserve_exact(target)
            .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)?;
        for value in values.drain(..) {
            replacement.push(value);
        }
        *values = replacement;
        Ok(())
    })
}

fn doubled_capacity(capacity: usize, len: usize) -> NativeExecutorResult<usize> {
    let target = capacity
        .saturating_mul(2)
        .max(4)
        .max(len.checked_add(1).ok_or_else(|| {
            NativeExecutorError::Host("native continuation capacity overflow".into())
        })?);
    Ok(target)
}

fn with_collection_growth<T>(
    runtime: &RuntimeContext,
    gate: &FrameCollectionGate,
    retained: u64,
    build: impl FnOnce(OperationCollectionAllocator) -> Result<T, ExactHostNumericAllocationError>,
) -> NativeExecutorResult<T> {
    let plan = HostAllocationPlan::new(0, retained)
        .map_err(|cause| continuation_admission_error(runtime, cause))?;
    let receipt = HostAllocationReceipt::admit(Some(runtime), plan)
        .map_err(|cause| continuation_admission_error(runtime, cause))?;
    let _installed = match &receipt {
        Some(receipt) => gate.install(runtime, receipt),
        None => gate.install_unbounded(),
    }
    .map_err(|cause| continuation_admission_error(runtime, cause))?;
    let result = build(gate.allocator()).map_err(|cause| {
        continuation_admission_error(runtime, HostAllocationError::Numeric(cause))
    })?;
    if let Some(receipt) = &receipt {
        receipt
            .checkpoint(runtime)
            .map_err(|cause| continuation_admission_error(runtime, cause))?;
    }
    Ok(result)
}

pub(super) struct OrderedCache<K, V> {
    entries: AllocatedVec<(K, V), OperationCollectionAllocator>,
    index: Index<K>,
}

impl<K, V> Drop for OrderedCache<K, V> {
    fn drop(&mut self) {
        // Entries may release foreign owners. Their observable retirement is
        // source-ordered even when this cache is abandoned without a site
        // finish; the derived hash index has no semantic values to retire.
        for entry in self.entries.drain(..) {
            drop(entry);
        }
    }
}

impl<K: Eq + Hash + Clone, V> OrderedCache<K, V> {
    pub(super) fn new(allocator: OperationCollectionAllocator) -> Self {
        Self {
            entries: AllocatedVec::new_in(allocator.clone()),
            index: HashMap::with_hasher_in(RandomState::new(), allocator),
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(super) fn get(&self, key: &K) -> Option<&V> {
        self.index
            .get(key)
            .and_then(|index| self.entries.get(*index).map(|(_, value)| value))
    }

    pub(super) fn values(&self) -> impl Iterator<Item = &V> {
        self.entries.iter().map(|(_, value)| value)
    }

    pub(super) fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let index = *self.index.get(key)?;
        self.entries.get_mut(index).map(|(_, value)| value)
    }

    pub(super) fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.entries.iter_mut().map(|(_, value)| value)
    }

    pub(super) fn clear_in_source_order(&mut self) {
        for entry in self.entries.drain(..) {
            drop(entry);
        }
        self.index.clear();
    }

    pub(super) fn insert(
        &mut self,
        runtime: &RuntimeContext,
        gate: &FrameCollectionGate,
        key: K,
        value: V,
    ) -> NativeExecutorResult<Option<V>> {
        if let Some(index) = self.index.get(&key) {
            return Ok(Some(std::mem::replace(&mut self.entries[*index].1, value)));
        }
        self.prepare_one_more(runtime, gate)?;
        Ok(self.insert_ready(key, value))
    }

    pub(super) fn insert_ready(&mut self, key: K, value: V) -> Option<V> {
        if let Some(index) = self.index.get(&key) {
            return Some(std::mem::replace(&mut self.entries[*index].1, value));
        }
        debug_assert!(self.entries.len() < self.entries.capacity());
        debug_assert!(self.index.len() < self.index.capacity());
        let position = self.entries.len();
        self.index.insert(key.clone(), position);
        self.entries.push((key, value));
        None
    }

    pub(super) fn prepare_one_more(
        &mut self,
        runtime: &RuntimeContext,
        gate: &FrameCollectionGate,
    ) -> NativeExecutorResult<()> {
        if self.entries.len() < self.entries.capacity() && self.index.len() < self.index.capacity()
        {
            return Ok(());
        }
        let target = doubled_capacity(self.entries.capacity(), self.entries.len())?;
        let layout = Layout::array::<(K, V)>(target).map_err(|_| {
            NativeExecutorError::Host("native continuation cache layout overflow".into())
        })?;
        let entry_bytes = collection_block_retained_bytes(layout)
            .map_err(|cause| continuation_admission_error(runtime, cause))?;
        let index_bytes = frame_hash_table_retained_bytes::<K, usize>(target)
            .map_err(|cause| continuation_admission_error(runtime, cause))?;
        let retained = entry_bytes
            .checked_add(index_bytes)
            .ok_or_else(|| continuation_admission_error(runtime, HostAllocationError::Overflow))?;
        let (mut entries, mut index) =
            with_collection_growth(runtime, gate, retained, |allocator| {
                let mut entries = AllocatedVec::new_in(allocator.clone());
                entries
                    .try_reserve_exact(target)
                    .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)?;
                let mut index = HashMap::with_hasher_in(RandomState::new(), allocator);
                index
                    .try_reserve(target)
                    .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)?;
                Ok((entries, index))
            })?;
        for (key, value) in self.entries.drain(..) {
            index.insert(key.clone(), entries.len());
            entries.push((key, value));
        }
        self.entries = entries;
        self.index = index;
        Ok(())
    }
}
