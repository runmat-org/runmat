//! Actual hash/vector allocation layouts, under one captured operation owner.
use super::*;
use crate::context::host_allocation::{
    collection_block_retained_bytes, frame_hash_table_retained_bytes, FrameCollectionGate,
    HostAllocationCheckpoint, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
    OperationCollectionAllocator,
};
use std::{alloc::Layout, collections::hash_map::RandomState, hash::Hash};
pub(in crate::builtins::math::optim) type Map<K, V> =
    hashbrown::HashMap<K, V, RandomState, OperationCollectionAllocator>;
pub(in crate::builtins::math::optim) type Set<K> =
    hashbrown::HashSet<K, RandomState, OperationCollectionAllocator>;
pub(in crate::builtins::math::optim) type Vector<T> =
    allocator_api2::vec::Vec<T, OperationCollectionAllocator>;

pub(in crate::builtins::math::optim) struct Collections {
    runtime: Option<RuntimeContext>,
    checkpoint: Option<HostAllocationCheckpoint>,
    policy: HostAllocationPolicy,
    gate: FrameCollectionGate,
}
impl Collections {
    pub fn new(cooperation: &ScalarCooperation) -> BuiltinResult<Self> {
        cooperation.check_allocation_epoch()?;
        let runtime = cooperation.runtime().cloned();
        let context = runtime.as_ref();
        let checkpoint = HostAllocationReceipt::capture_owner(context)
            .map_err(|source| allocation(context, source))?;
        let policy =
            HostAllocationReceipt::policy(context).map_err(|source| allocation(context, source))?;
        let gate = match policy {
            HostAllocationPolicy::Managed => {
                let plan = HostAllocationPlan::new(
                    0,
                    FrameCollectionGate::retained_bytes()
                        .map_err(|source| allocation(context, source))?,
                )
                .map_err(|source| allocation(context, source))?;
                let receipt = HostAllocationReceipt::admit(context, plan)
                    .map_err(|source| allocation(context, source))?
                    .ok_or_else(|| allocation(context, HostAllocationError::StaleAuthority))?;
                FrameCollectionGate::prepare(context.expect("managed original context"), &receipt)
                    .map_err(|source| allocation(context, source))?
            }
            HostAllocationPolicy::Unbounded => FrameCollectionGate::unbounded(),
        };
        let owner = Self {
            runtime,
            checkpoint,
            policy,
            gate,
        };
        owner.check()?;
        cooperation.check_allocation_epoch()?;
        Ok(owner)
    }
    pub fn check(&self) -> BuiltinResult<()> {
        let context = self.runtime.as_ref();
        if let Some(checkpoint) = &self.checkpoint {
            checkpoint
                .check(context.expect("managed original context"))
                .map_err(|source| allocation(context, source))?;
        } else if HostAllocationReceipt::policy(context)
            .map_err(|source| allocation(context, source))?
            != self.policy
        {
            return Err(allocation(context, HostAllocationError::StaleAuthority));
        }
        if context.is_some_and(RuntimeContext::is_cancelled) {
            return Err(crate::execution::ExecutionServiceError::Cancelled
                .into_runtime_error_in_context(context));
        }
        Ok(())
    }
    pub fn map<K, V>(&self) -> Map<K, V> {
        Map::with_hasher_in(RandomState::new(), self.gate.allocator())
    }
    pub fn set<K>(&self) -> Set<K> {
        Set::with_hasher_in(RandomState::new(), self.gate.allocator())
    }
    pub fn vector<T>(&self) -> Vector<T> {
        Vector::new_in(self.gate.allocator())
    }
    fn grow<T>(&self, bytes: u64, allocate: impl FnOnce() -> Result<T, ()>) -> BuiltinResult<T> {
        self.check()?;
        let context = self.runtime.as_ref();
        let receipt = HostAllocationReceipt::admit(
            context,
            HostAllocationPlan::new(0, bytes).map_err(|source| allocation(context, source))?,
        )
        .map_err(|source| allocation(context, source))?;
        let _installed = match &receipt {
            Some(receipt) => self
                .gate
                .install(context.expect("managed original context"), receipt),
            None => self.gate.install_unbounded(),
        }
        .map_err(|source| allocation(context, source))?;
        let result = allocate();
        if let Some(receipt) = receipt.as_ref() {
            receipt
                .checkpoint(context.expect("managed original context"))
                .map_err(|source| allocation(context, source))?;
        }
        self.check()?;
        result.map_err(|_| {
            allocation(
                context,
                HostAllocationError::Numeric(
                    runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                ),
            )
        })
    }
    pub fn reserve_map<K: Eq + Hash, V>(
        &self,
        values: &mut Map<K, V>,
        additional: usize,
    ) -> BuiltinResult<()> {
        let count = values
            .len()
            .checked_add(additional)
            .ok_or_else(|| allocation(self.runtime.as_ref(), HostAllocationError::Overflow))?;
        if count <= values.capacity() {
            return self.check();
        }
        let bytes = frame_hash_table_retained_bytes::<K, V>(count)
            .map_err(|source| allocation(self.runtime.as_ref(), source))?;
        self.grow(bytes, || values.try_reserve(additional).map_err(|_| ()))
    }
    pub fn reserve_set<K: Eq + Hash>(
        &self,
        values: &mut Set<K>,
        additional: usize,
    ) -> BuiltinResult<()> {
        let count = values
            .len()
            .checked_add(additional)
            .ok_or_else(|| allocation(self.runtime.as_ref(), HostAllocationError::Overflow))?;
        if count <= values.capacity() {
            return self.check();
        }
        let bytes = frame_hash_table_retained_bytes::<K, ()>(count)
            .map_err(|source| allocation(self.runtime.as_ref(), source))?;
        self.grow(bytes, || values.try_reserve(additional).map_err(|_| ()))
    }
    pub fn reserve_vector<T>(
        &self,
        values: &mut Vector<T>,
        additional: usize,
    ) -> BuiltinResult<()> {
        let count = values
            .len()
            .checked_add(additional)
            .ok_or_else(|| allocation(self.runtime.as_ref(), HostAllocationError::Overflow))?;
        if count <= values.capacity() {
            return self.check();
        }
        let capacity = values
            .capacity()
            .max(2)
            .checked_mul(2)
            .map(|capacity| capacity.max(count))
            .ok_or_else(|| allocation(self.runtime.as_ref(), HostAllocationError::Overflow))?;
        let layout = Layout::array::<T>(capacity)
            .map_err(|_| allocation(self.runtime.as_ref(), HostAllocationError::Overflow))?;
        let bytes = collection_block_retained_bytes(layout)
            .map_err(|source| allocation(self.runtime.as_ref(), source))?;
        self.grow(bytes, || {
            values
                .try_reserve_exact(capacity - values.len())
                .map_err(|_| ())
        })
    }
}
