//! CUDA allocation records and their retained primary-context authority.
//! Handle ownership is unchanged in this preparatory slice.
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::{anyhow, Result};
use runmat_accelerate_api::allocation::catch_allocation_cleanup_unwind;
use runmat_accelerate_api::{GpuTensorDescriptor, GpuTensorHandle};

use super::driver::{CudaDriver, DevicePointer};

mod context;
mod retirement;
#[cfg(test)]
mod tests;

/// Narrow lifecycle seam; numerical operations still use the real driver.
trait ContextDriver: Send + Sync {
    fn push(&self, context: usize) -> Result<()>;
    fn pop(&self) -> Result<()>;
    fn free(&self, pointer: DevicePointer) -> Result<()>;
    fn release(&self, device: i32) -> Result<()>;
}

impl ContextDriver for CudaDriver {
    fn push(&self, context: usize) -> Result<()> {
        self.push_context(context)
    }
    fn pop(&self) -> Result<()> {
        self.pop_context()
    }
    fn free(&self, pointer: DevicePointer) -> Result<()> {
        self.free(pointer)
    }
    fn release(&self, device: i32) -> Result<()> {
        self.release_primary_context(device)
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Allocation {
    pub(super) pointer: DevicePointer,
    pub(super) byte_length: usize,
    pub(super) descriptor: GpuTensorDescriptor,
}

#[derive(Debug)]
enum Phase {
    Pending,
    Live,
    Retiring,
    EntryFailed(String),
    Uncertain(String),
    FreedExitFailed(String),
}

#[derive(Debug)]
struct Record {
    allocation: Option<Allocation>,
    phase: Phase,
}

pub(super) struct AllocationState {
    driver: Arc<dyn ContextDriver>,
    cuda_device: i32,
    context: usize,
    device_id: u32,
    next_buffer: AtomicU64,
    allocations: Mutex<HashMap<u64, Record>>,
    unavailable: Mutex<Option<context::ContextUnavailable>>,
}

impl AllocationState {
    pub(super) fn new(
        driver: Arc<CudaDriver>,
        cuda_device: i32,
        context: usize,
        device_id: u32,
    ) -> Self {
        Self::with_driver(driver, cuda_device, context, device_id)
    }

    fn with_driver(
        driver: Arc<dyn ContextDriver>,
        cuda_device: i32,
        context: usize,
        device_id: u32,
    ) -> Self {
        Self {
            driver,
            cuda_device,
            context,
            device_id,
            next_buffer: AtomicU64::new(1),
            allocations: Mutex::new(HashMap::new()),
            unavailable: Mutex::new(None),
        }
    }

    // Recover only to preserve bookkeeping. Public operations reject a poisoned
    // owner through ensure_available; recovery never claims a healthy context.
    fn records(&self) -> MutexGuard<'_, HashMap<u64, Record>> {
        self.allocations
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    pub(super) fn context_identity(&self) -> Result<usize> {
        self.ensure_available()?;
        Ok(self.context)
    }

    pub(super) fn allocation(&self, handle: &GpuTensorHandle) -> Result<Allocation> {
        self.validate_owner(handle)?;
        self.ensure_available()?;
        let records = self.records();
        let record = records
            .get(&handle.buffer_id)
            .ok_or_else(|| anyhow!("CUDA handle is stale or unknown"))?;
        if !matches!(record.phase, Phase::Live) {
            return Err(anyhow!("CUDA allocation is not live: {:?}", record.phase));
        }
        let allocation = record
            .allocation
            .ok_or_else(|| anyhow!("CUDA allocation has no payload"))?;
        if allocation.descriptor.element_type != handle.descriptor.element_type
            || allocation.descriptor.storage != handle.descriptor.storage
        {
            return Err(anyhow!(
                "CUDA handle descriptor does not match its allocation"
            ));
        }
        Ok(allocation)
    }

    fn validate_owner(&self, handle: &GpuTensorHandle) -> Result<()> {
        if handle.device_id != self.device_id {
            return Err(anyhow!("CUDA handle belongs to a different provider"));
        }
        Ok(())
    }

    /// Publish the driver pointer in the existing map before context exit can
    /// fail. An unpublished allocation is therefore still accounted for.
    pub(super) fn allocate(
        &self,
        shape: Vec<usize>,
        descriptor: GpuTensorDescriptor,
        byte_length: usize,
        allocate: impl FnOnce() -> Result<DevicePointer>,
        initialize: impl FnOnce(DevicePointer) -> Result<()>,
    ) -> Result<GpuTensorHandle> {
        self.ensure_available()?;
        let buffer_id = self
            .next_buffer
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| anyhow!("CUDA buffer identity space is exhausted"))?;
        self.records().insert(
            buffer_id,
            Record {
                allocation: None,
                phase: Phase::Pending,
            },
        );
        let registration = PendingRegistration {
            owner: self,
            id: buffer_id,
            active: true,
        };
        let result = self.with_context(|| {
            let pointer = allocate()?;
            let allocation = Allocation {
                pointer,
                byte_length,
                descriptor,
            };
            self.records().insert(
                buffer_id,
                Record {
                    allocation: Some(allocation),
                    phase: Phase::Live,
                },
            );
            initialize(pointer)
        });
        result?;
        // Retain the unpublished record if quarantine was observed after the
        // context scope completed. Publication does not guarantee future health.
        self.ensure_available()?;
        registration.publish();
        Ok(GpuTensorHandle {
            shape,
            device_id: self.device_id,
            buffer_id,
            descriptor,
        })
    }
}

/// The raw allocator must return every successfully created pointer; an error
/// or unwind must not conceal an allocation. A slot with no returned pointer
/// is reservation bookkeeping, not evidence of a live physical allocation.
struct PendingRegistration<'a> {
    owner: &'a AllocationState,
    id: u64,
    active: bool,
}
impl PendingRegistration<'_> {
    fn publish(mut self) {
        self.active = false;
    }
}
impl Drop for PendingRegistration<'_> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let needs_cleanup = {
            let mut records = self.owner.records();
            if records
                .get(&self.id)
                .is_some_and(|record| matches!(record.phase, Phase::Pending))
            {
                records.remove(&self.id);
                false
            } else {
                records.contains_key(&self.id)
            }
        };
        if needs_cleanup {
            // The invocation context scope has already exited, including on
            // initializer unwind. Never perform driver work under the map lock.
            let _ = catch_allocation_cleanup_unwind(|| {
                let handle = GpuTensorHandle::new(vec![], self.owner.device_id, self.id);
                if let Err(cleanup) = self.owner.free(&handle) {
                    log::error!(
                        "CUDA unpublished allocation remains accounted after failure: {cleanup}"
                    );
                }
            });
        }
    }
}
