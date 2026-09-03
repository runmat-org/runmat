use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BufferUsageClass {
    Generic,
    Readback,
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
    state: Mutex<ResidencyState>,
    max_per_key: usize,
    max_total: usize,
}

struct ResidencyState {
    pools: HashMap<ResidencyKey, VecDeque<PooledBuffer>>,
    release_order: VecDeque<(ResidencyKey, u64)>,
    next_release_id: u64,
    total: usize,
}

struct PooledBuffer {
    release_id: u64,
    buffer: Arc<wgpu::Buffer>,
}

impl BufferResidency {
    pub fn new(max_per_key: usize, max_total: usize) -> Self {
        Self {
            state: Mutex::new(ResidencyState {
                pools: HashMap::new(),
                release_order: VecDeque::new(),
                next_release_id: 0,
                total: 0,
            }),
            max_per_key,
            max_total,
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
        self.acquire_bytes(
            device,
            usage,
            size_bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            label,
        )
    }

    pub fn acquire_readback(
        &self,
        device: &wgpu::Device,
        size_bytes: u64,
        label: &str,
    ) -> (Arc<wgpu::Buffer>, bool) {
        self.acquire_bytes(
            device,
            BufferUsageClass::Readback,
            size_bytes.max(1),
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            label,
        )
    }

    pub fn acquire_storage_bytes(
        &self,
        device: &wgpu::Device,
        size_bytes: u64,
        label: &str,
    ) -> (Arc<wgpu::Buffer>, bool) {
        self.acquire_bytes(
            device,
            BufferUsageClass::Generic,
            size_bytes.max(1),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            label,
        )
    }

    fn acquire_bytes(
        &self,
        device: &wgpu::Device,
        usage: BufferUsageClass,
        size_bytes: u64,
        buffer_usages: wgpu::BufferUsages,
        label: &str,
    ) -> (Arc<wgpu::Buffer>, bool) {
        let key = ResidencyKey::new(usage, size_bytes);
        if let Ok(mut state) = self.state.lock() {
            let pooled = state.pools.get_mut(&key).and_then(VecDeque::pop_front);
            if let Some(pooled) = pooled {
                state.total = state.total.saturating_sub(1);
                if let Some(position) = state
                    .release_order
                    .iter()
                    .position(|entry| *entry == (key, pooled.release_id))
                {
                    state.release_order.remove(position);
                }
                if state.pools.get(&key).is_some_and(VecDeque::is_empty) {
                    state.pools.remove(&key);
                }
                let buffer = pooled.buffer;
                log::trace!(
                    "buffer_residency: reuse {:?} bytes={} ptr={:p}",
                    usage,
                    size_bytes,
                    Arc::as_ptr(&buffer)
                );
                return (buffer, true);
            }
        }

        let buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size_bytes,
            usage: buffer_usages,
            mapped_at_creation: false,
        }));
        log::trace!(
            "buffer_residency: new {:?} bytes={} ptr={:p}",
            usage,
            size_bytes,
            Arc::as_ptr(&buffer)
        );
        (buffer, false)
    }

    pub fn release(
        &self,
        usage: BufferUsageClass,
        allocated_bytes: u64,
        buffer: Arc<wgpu::Buffer>,
    ) {
        if allocated_bytes == 0 {
            return;
        }

        let key = ResidencyKey::new(usage, allocated_bytes);
        if self.max_per_key == 0 || self.max_total == 0 {
            return;
        }

        if let Ok(mut state) = self.state.lock() {
            let key_len = state.pools.get(&key).map_or(0, VecDeque::len);
            if key_len < self.max_per_key {
                while state.total >= self.max_total {
                    let Some((evicted_key, evicted_id)) = state.release_order.pop_front() else {
                        break;
                    };
                    let mut evicted = None;
                    let mut remove_key = false;
                    if let Some(queue) = state.pools.get_mut(&evicted_key) {
                        if let Some(position) = queue
                            .iter()
                            .position(|entry| entry.release_id == evicted_id)
                        {
                            evicted = queue.remove(position);
                        }
                        remove_key = queue.is_empty();
                    }
                    if remove_key {
                        state.pools.remove(&evicted_key);
                    }
                    if let Some(evicted) = evicted {
                        state.total = state.total.saturating_sub(1);
                        log::trace!(
                            "buffer_residency: evict {:?} bytes={} ptr={:p} (global pool full)",
                            evicted_key.usage,
                            evicted_key.allocated_bytes,
                            Arc::as_ptr(&evicted.buffer)
                        );
                    }
                }

                let release_id = state.next_release_id;
                state.next_release_id = state.next_release_id.wrapping_add(1);
                log::trace!(
                    "buffer_residency: release {:?} bytes={} ptr={:p}",
                    usage,
                    allocated_bytes,
                    Arc::as_ptr(&buffer)
                );
                state
                    .pools
                    .entry(key)
                    .or_insert_with(VecDeque::new)
                    .push_back(PooledBuffer { release_id, buffer });
                state.release_order.push_back((key, release_id));
                state.total += 1;
            } else {
                log::trace!(
                    "buffer_residency: drop {:?} bytes={} ptr={:p} (pool full)",
                    usage,
                    allocated_bytes,
                    Arc::as_ptr(&buffer)
                );
            }
        }
    }

    pub fn pooled_bytes(&self, _element_size: usize) -> u64 {
        self.state
            .lock()
            .map(|state| {
                state.pools.iter().fold(0_u64, |total, (key, buffers)| {
                    let bytes_each = key.allocated_bytes;
                    total.saturating_add(
                        bytes_each.saturating_mul(u64::try_from(buffers.len()).unwrap_or(u64::MAX)),
                    )
                })
            })
            .unwrap_or(0)
    }

    #[cfg(test)]
    pub fn pooled_count(&self) -> usize {
        self.state.lock().map(|state| state.total).unwrap_or(0)
    }

    #[cfg(test)]
    pub fn max_total(&self) -> usize {
        self.max_total
    }
}
