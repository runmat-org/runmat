use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BufferUsageClass {
    Generic,
    MatmulPartial,
    MatmulOut,
    SyrkOut,
    FusionOut,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResidencyKey {
    usage: BufferUsageClass,
    allocated_bytes: u64,
}

impl ResidencyKey {
    fn new(usage: BufferUsageClass, allocated_bytes: u64) -> Self {
        Self {
            usage,
            allocated_bytes,
        }
    }
}

impl Hash for ResidencyKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.usage.hash(state);
        self.allocated_bytes.hash(state);
    }
}

pub struct BufferResidency {
    pools: Mutex<HashMap<ResidencyKey, VecDeque<Arc<wgpu::Buffer>>>>,
    max_per_key: usize,
}

/// Optional buffer reuse was declined; this is not a device/free failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolRejectionReason {
    Empty,
    Full,
    Poisoned,
    BookkeepingCapacity,
}

pub struct RejectedBuffer {
    pub reason: PoolRejectionReason,
    pub buffer: Arc<wgpu::Buffer>,
}

impl Drop for BufferResidency {
    fn drop(&mut self) {
        let pools = std::mem::take(self.pools.get_mut().unwrap_or_else(|e| e.into_inner()));
        for buffers in pools.into_values() {
            for buffer in buffers {
                // Final pool disposal, not an uncertain driver free retry.
                if runmat_accelerate_api::allocation::catch_allocation_cleanup_unwind(|| {
                    drop(buffer)
                })
                .is_err()
                {
                    let _ =
                        runmat_accelerate_api::allocation::catch_allocation_cleanup_unwind(|| {
                            log::error!(
                                "WGPU pool terminal reference release unwound; no retry attempted"
                            );
                        });
                }
            }
        }
    }
}

impl BufferResidency {
    pub fn new(max_per_key: usize) -> Self {
        Self {
            pools: Mutex::new(HashMap::new()),
            max_per_key,
        }
    }

    pub fn acquire(
        &self,
        device: &wgpu::Device,
        usage: BufferUsageClass,
        len: usize,
        element_size: usize,
        label: &str,
    ) -> (Arc<wgpu::Buffer>, bool) {
        if len == 0 {
            return (
                Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: element_size.max(1) as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })),
                false,
            );
        }

        let size_bytes = (len as u64).max(1) * element_size as u64;
        let key = ResidencyKey::new(usage, size_bytes);
        let reused = self
            .pools
            .lock()
            .ok()
            .and_then(|mut pools| pools.get_mut(&key).and_then(VecDeque::pop_front));
        if let Some(buffer) = reused {
            log::trace!(
                "buffer_residency: reuse {:?} len={} ptr={:p}",
                usage,
                len,
                Arc::as_ptr(&buffer)
            );
            return (buffer, true);
        }

        let buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        log::trace!(
            "buffer_residency: new {:?} len={} ptr={:p}",
            usage,
            len,
            Arc::as_ptr(&buffer)
        );
        (buffer, false)
    }

    pub fn release(
        &self,
        usage: BufferUsageClass,
        allocated_bytes: u64,
        buffer: Arc<wgpu::Buffer>,
    ) -> Result<(), RejectedBuffer> {
        release_into(
            &self.pools,
            self.max_per_key,
            ResidencyKey::new(usage, allocated_bytes),
            buffer,
        )
        .map_err(|(reason, buffer)| RejectedBuffer { reason, buffer })
    }

    pub fn pooled_bytes(&self, _element_size: usize) -> u64 {
        self.pools
            .lock()
            .map(|pools| {
                pools.iter().fold(0_u64, |total, (key, buffers)| {
                    let bytes_each = key.allocated_bytes;
                    total.saturating_add(
                        bytes_each.saturating_mul(u64::try_from(buffers.len()).unwrap_or(u64::MAX)),
                    )
                })
            })
            .unwrap_or(0)
    }
}

fn release_into<T>(
    pools: &Mutex<HashMap<ResidencyKey, VecDeque<T>>>,
    max_per_key: usize,
    key: ResidencyKey,
    buffer: T,
) -> Result<(), (PoolRejectionReason, T)> {
    if key.allocated_bytes == 0 {
        return Err((PoolRejectionReason::Empty, buffer));
    }
    if max_per_key == 0 {
        return Err((PoolRejectionReason::Full, buffer));
    }
    let mut pools = match pools.lock() {
        Ok(pools) => pools,
        Err(_) => return Err((PoolRejectionReason::Poisoned, buffer)),
    };
    if let Some(queue) = pools.get_mut(&key) {
        if queue.len() >= max_per_key {
            return Err((PoolRejectionReason::Full, buffer));
        }
        if queue.try_reserve(1).is_err() {
            return Err((PoolRejectionReason::BookkeepingCapacity, buffer));
        }
        queue.push_back(buffer);
    } else {
        // Complete fallible bookkeeping before transferring the resource.
        let mut queue = VecDeque::new();
        if pools.try_reserve(1).is_err() || queue.try_reserve(1).is_err() {
            return Err((PoolRejectionReason::BookkeepingCapacity, buffer));
        }
        queue.push_back(buffer);
        pools.insert(key, queue);
    }
    Ok(())
}

#[cfg(test)]
#[path = "residency/tests.rs"]
mod tests;
