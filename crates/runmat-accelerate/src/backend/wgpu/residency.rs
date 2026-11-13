use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

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
    scope: Option<runmat_accelerate_api::DeviceAllocationScope>,
}

impl ResidencyKey {
    fn new(usage: BufferUsageClass, allocated_bytes: u64) -> Self {
        Self {
            usage,
            allocated_bytes,
            scope: None,
        }
    }
}

impl Hash for ResidencyKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.usage.hash(state);
        self.allocated_bytes.hash(state);
        self.scope.hash(state);
    }
}

pub struct BufferResidency {
    pools: Mutex<HashMap<ResidencyKey, VecDeque<runmat_accelerate_api::WgpuBufferOwner>>>,
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
    pub buffer: runmat_accelerate_api::WgpuBufferOwner,
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
    /// Exact buffer extent used by `acquire`, including its nonzero empty
    /// binding. This is also the provider's symbolic FusionOut projection.
    pub(crate) fn requested_bytes(len: usize, element_size: usize) -> Option<u64> {
        if element_size == 0 {
            return None;
        }
        if len == 0 {
            u64::try_from(element_size.max(1)).ok()
        } else {
            u64::try_from(len)
                .ok()?
                .checked_mul(u64::try_from(element_size).ok()?)
        }
    }

    pub fn new(max_per_key: usize) -> Self {
        Self {
            pools: Mutex::new(HashMap::new()),
            max_per_key,
        }
    }

    pub fn acquire(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        usage: BufferUsageClass,
        len: usize,
        element_size: usize,
        label: &str,
    ) -> anyhow::Result<(runmat_accelerate_api::WgpuBufferOwner, bool)> {
        let size_bytes = Self::requested_bytes(len, element_size)
            .ok_or_else(|| anyhow::anyhow!("{label}: allocation byte extent overflow"))?;
        if len == 0 {
            return Ok((
                runmat_accelerate_api::WgpuBufferOwner::new(emission.device().create_buffer(
                    &wgpu::BufferDescriptor {
                        label: Some(label),
                        size: size_bytes,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    },
                )),
                false,
            ));
        }

        let mut key = ResidencyKey::new(usage, size_bytes);
        key.scope = runmat_accelerate_api::current_device_allocation_scope();
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
                runmat_accelerate_api::WgpuBufferOwner::as_ptr(&buffer)
            );
            return Ok((buffer, true));
        }

        let buffer = runmat_accelerate_api::WgpuBufferOwner::new(emission.device().create_buffer(
            &wgpu::BufferDescriptor {
                label: Some(label),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        ));
        log::trace!(
            "buffer_residency: new {:?} len={} ptr={:p}",
            usage,
            len,
            runmat_accelerate_api::WgpuBufferOwner::as_ptr(&buffer)
        );
        Ok((buffer, false))
    }

    pub fn release(
        &self,
        usage: BufferUsageClass,
        allocated_bytes: u64,
        buffer: runmat_accelerate_api::WgpuBufferOwner,
    ) -> Result<(), RejectedBuffer> {
        let mut key = ResidencyKey::new(usage, allocated_bytes);
        key.scope = buffer.creation_scope();
        release_into(&self.pools, self.max_per_key, key, buffer)
            .map_err(|(reason, buffer)| RejectedBuffer { reason, buffer })
    }

    /// Detach one scope's reusable buffers, then destroy without the pool lock.
    pub(super) fn has_scope(
        &self,
        scope: runmat_accelerate_api::DeviceAllocationScope,
    ) -> anyhow::Result<bool> {
        Ok(self
            .pools
            .lock()
            .map_err(|_| anyhow::anyhow!("WGPU scoped pool query is poisoned"))?
            .iter()
            .any(|(key, buffers)| {
                !buffers.is_empty() && key.scope.is_some_and(|created| scope.covers(created))
            }))
    }

    pub(super) fn reset_selection_bytes(
        &self,
        scope: runmat_accelerate_api::DeviceAllocationScope,
        retiring_records: usize,
    ) -> Option<u64> {
        let pools = self.pools.lock().ok()?;
        let count = pools
            .keys()
            .filter(|key| key.scope.is_some_and(|created| scope.covers(created)))
            .count();
        // A concurrent final lease can move a quoted record into one new pool
        // key before the fence settles. Fund that bounded transition too.
        u64::try_from(
            count
                .checked_add(retiring_records)?
                .checked_mul(std::mem::size_of::<ResidencyKey>())?,
        )
        .ok()
    }

    pub(super) async fn destroy_scope(
        &self,
        scope: runmat_accelerate_api::DeviceAllocationScope,
        access: &runmat_accelerate_api::WgpuDeviceAccess,
    ) -> anyhow::Result<()> {
        let keys = {
            let pools = self
                .pools
                .lock()
                .map_err(|_| anyhow::anyhow!("WGPU scoped pool reset is poisoned"))?;
            let count = pools
                .keys()
                .filter(|key| key.scope.is_some_and(|created| scope.covers(created)))
                .count();
            let mut keys = runmat_value::exact_capacity_vec(count)?;
            keys.extend(
                pools
                    .keys()
                    .filter(|key| key.scope.is_some_and(|created| scope.covers(created)))
                    .copied(),
            );
            keys
        };
        for key in keys {
            loop {
                tokio::task::yield_now().await;
                let candidate = {
                    let pools = self
                        .pools
                        .lock()
                        .map_err(|_| anyhow::anyhow!("WGPU scoped pool reset is poisoned"))?;
                    pools.get(&key).and_then(|buffers| buffers.front().cloned())
                };
                let Some(buffer) = candidate else {
                    break;
                };
                // A rejected cleanup leaves every owner in the existing pool.
                // No pool lock crosses a driver call and no queue is detached on
                // failure, so a retained reset failure cannot fabricate release.
                access.with_cleanup(|cleanup| cleanup.destroy_buffer(&buffer))?;
                let removed = {
                    let mut pools = self
                        .pools
                        .lock()
                        .map_err(|_| anyhow::anyhow!("WGPU scoped pool reset is poisoned"))?;
                    let Some(buffers) = pools.get_mut(&key) else {
                        continue;
                    };
                    let removed = if buffers.front().is_some_and(|front| {
                        runmat_accelerate_api::WgpuBufferOwner::ptr_eq(front, &buffer)
                    }) {
                        buffers.pop_front()
                    } else {
                        None
                    };
                    if buffers.is_empty() {
                        pools.remove(&key);
                    }
                    removed
                };
                drop(removed);
                drop(buffer);
            }
        }
        Ok(())
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
