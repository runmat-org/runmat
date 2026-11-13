//! Decoder-local control storage. Every managed physical block uses the
//! canonical collection allocator; decoded language payloads never pass here.

use allocator_api2::{boxed::Box as AllocatedBox, vec::Vec as AllocatedVec};
use std::{alloc::Layout, fmt, ops::Deref};

use crate::context::{
    host_allocation::{
        collection_block_retained_bytes, FrameCollectionGate, HostAllocationCheckpoint,
        HostAllocationError, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
        OperationCollectionAllocator,
    },
    RuntimeContext,
};

use super::ValueCodecError;

#[derive(Clone)]
pub(super) struct ControlOwner<'a> {
    gate: FrameCollectionGate,
    runtime: Option<&'a RuntimeContext>,
    policy: HostAllocationPolicy,
    checkpoint: Option<HostAllocationCheckpoint>,
}

pub(super) type ControlBox<T> = AllocatedBox<T, OperationCollectionAllocator>;
pub(super) type ControlIntoIter<T> = allocator_api2::vec::IntoIter<T, OperationCollectionAllocator>;

impl<'a> ControlOwner<'a> {
    pub(super) fn new(runtime: Option<&'a RuntimeContext>) -> Result<Self, ValueCodecError> {
        let allocation = |source| ValueCodecError::allocation("$", source);
        let checkpoint = HostAllocationReceipt::capture_owner(runtime).map_err(allocation)?;
        let policy = if checkpoint.is_some() {
            HostAllocationPolicy::Managed
        } else {
            HostAllocationPolicy::Unbounded
        };
        let gate = match policy {
            HostAllocationPolicy::Unbounded => FrameCollectionGate::unbounded(),
            HostAllocationPolicy::Managed => {
                let runtime = runtime.expect("managed policy has a runtime");
                let bytes = FrameCollectionGate::retained_bytes().map_err(allocation)?;
                let plan = HostAllocationPlan::new(0, bytes).map_err(allocation)?;
                let receipt = HostAllocationReceipt::admit(Some(runtime), plan)
                    .map_err(allocation)?
                    .ok_or_else(|| allocation(HostAllocationError::StaleAuthority))?;
                FrameCollectionGate::prepare(runtime, &receipt).map_err(allocation)?
            }
        };
        if let Some(checkpoint) = &checkpoint {
            checkpoint
                .check(runtime.expect("managed control owner has a runtime"))
                .map_err(allocation)?;
        }
        Ok(Self {
            gate,
            runtime,
            policy,
            checkpoint,
        })
    }

    fn allocate<T>(
        &self,
        layout: Layout,
        path: &str,
        operation: impl FnOnce() -> Result<T, ()>,
    ) -> Result<T, ValueCodecError> {
        let allocation = |source| ValueCodecError::allocation(path, source);
        if let Some(checkpoint) = &self.checkpoint {
            checkpoint
                .check(self.runtime.expect("managed control owner has a runtime"))
                .map_err(allocation)?;
        } else if HostAllocationReceipt::policy(self.runtime).map_err(allocation)? != self.policy {
            return Err(allocation(HostAllocationError::StaleAuthority));
        }
        let failure = HostAllocationError::Numeric(
            runmat_value::ExactHostNumericAllocationError::AllocationFailed,
        );
        match self.policy {
            HostAllocationPolicy::Unbounded => {
                let _permit = self.gate.install_unbounded().map_err(allocation)?;
                operation().map_err(|()| allocation(failure))
            }
            HostAllocationPolicy::Managed => {
                let runtime = self.runtime.expect("managed policy has a runtime");
                let bytes = collection_block_retained_bytes(layout).map_err(allocation)?;
                let plan = HostAllocationPlan::new(0, bytes).map_err(allocation)?;
                let receipt = HostAllocationReceipt::admit(Some(runtime), plan)
                    .map_err(allocation)?
                    .ok_or_else(|| allocation(HostAllocationError::StaleAuthority))?;
                let _permit = self.gate.install(runtime, &receipt).map_err(allocation)?;
                let result = operation();
                // Preserve cancellation/revocation instead of translating the
                // allocator's deliberately narrow error into a capacity result.
                receipt.checkpoint(runtime).map_err(allocation)?;
                self.checkpoint
                    .as_ref()
                    .expect("managed control owner checkpoint")
                    .check(runtime)
                    .map_err(allocation)?;
                result.map_err(|()| allocation(failure))
            }
        }
    }

    pub(super) fn vector<T>(
        &self,
        capacity: usize,
        path: &str,
    ) -> Result<ControlVec<'a, T>, ValueCodecError> {
        let mut result = ControlVec {
            values: AllocatedVec::new_in(self.gate.allocator()),
            owner: self.clone(),
        };
        result.reserve_capacity(capacity, path)?;
        Ok(result)
    }

    pub(super) fn boxed<T>(&self, value: T, path: &str) -> Result<ControlBox<T>, ValueCodecError> {
        self.allocate(Layout::new::<T>(), path, || {
            AllocatedBox::try_new_in(value, self.gate.allocator()).map_err(|_| ())
        })
    }

    pub(super) fn path(&self, path: &str) -> Result<ControlPath<'a>, ValueCodecError> {
        let mut bytes = self.vector(path.len(), path)?;
        bytes.values.extend_from_slice(path.as_bytes());
        Ok(ControlPath { bytes })
    }
}

pub(super) struct ControlVec<'a, T> {
    values: AllocatedVec<T, OperationCollectionAllocator>,
    owner: ControlOwner<'a>,
}

impl<T> ControlVec<'_, T> {
    fn reserve_capacity(&mut self, capacity: usize, path: &str) -> Result<(), ValueCodecError> {
        if capacity <= self.values.capacity() {
            return Ok(());
        }
        let layout = Layout::array::<T>(capacity)
            .map_err(|_| ValueCodecError::allocation(path, HostAllocationError::Overflow))?;
        let additional = capacity - self.values.len();
        self.owner.allocate(layout, path, || {
            self.values.try_reserve_exact(additional).map_err(|_| ())
        })
    }

    pub(super) fn push(&mut self, value: T, path: &str) -> Result<(), ValueCodecError> {
        if self.values.len() == self.values.capacity() {
            let capacity = self
                .values
                .len()
                .checked_add(1)
                .and_then(|minimum| minimum.checked_next_power_of_two())
                .ok_or_else(|| ValueCodecError::allocation(path, HostAllocationError::Overflow))?;
            self.reserve_capacity(capacity, path)?;
        }
        self.values.push(value);
        Ok(())
    }

    pub(super) fn pop(&mut self) -> Option<T> {
        self.values.pop()
    }
}

impl<T> Deref for ControlVec<'_, T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl<T> IntoIterator for ControlVec<'_, T> {
    type Item = T;
    type IntoIter = ControlIntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

pub(super) struct ControlPath<'a> {
    bytes: ControlVec<'a, u8>,
}

impl<'a> ControlPath<'a> {
    pub(super) fn owner(&self) -> &ControlOwner<'a> {
        &self.bytes.owner
    }

    pub(super) fn try_clone(&self) -> Result<Self, ValueCodecError> {
        self.owner().path(self)
    }

    /// Count formatting without allocating, then write once into exact admitted
    /// backing. Parent diagnostic bytes are copied; source/value payloads are not.
    pub(super) fn extended(&self, suffix: fmt::Arguments<'_>) -> Result<Self, ValueCodecError> {
        struct Count(Option<usize>);
        impl fmt::Write for Count {
            fn write_str(&mut self, value: &str) -> fmt::Result {
                self.0 = self.0.and_then(|count| count.checked_add(value.len()));
                self.0.map(|_| ()).ok_or(fmt::Error)
            }
        }
        let mut count = Count(Some(self.len()));
        fmt::write(&mut count, suffix)
            .map_err(|_| ValueCodecError::allocation(self, HostAllocationError::Overflow))?;
        let mut result = Self {
            bytes: self.owner().vector(count.0.expect("counted path"), self)?,
        };
        result.bytes.values.extend_from_slice(self.as_bytes());
        fmt::write(&mut result, suffix)
            .map_err(|_| ValueCodecError::allocation(self, HostAllocationError::Overflow))?;
        Ok(result)
    }
}

impl Deref for ControlPath<'_> {
    type Target = str;
    fn deref(&self) -> &str {
        std::str::from_utf8(&self.bytes).expect("decoder paths contain only UTF-8 text")
    }
}

impl fmt::Display for ControlPath<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self)
    }
}

impl fmt::Write for ControlPath<'_> {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        if value.len() > self.bytes.values.capacity() - self.bytes.values.len() {
            return Err(fmt::Error);
        }
        self.bytes.values.extend_from_slice(value.as_bytes());
        Ok(())
    }
}
