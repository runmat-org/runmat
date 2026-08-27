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
    pools: Mutex<HashMap<ResidencyKey, VecDeque<Arc<wgpu::Buffer>>>>,
    max_per_key: usize,
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
        if let Ok(mut guard) = self.pools.lock() {
            if let Some(queue) = guard.get_mut(&key) {
                if let Some(buffer) = queue.pop_front() {
                    log::trace!(
                        "buffer_residency: reuse {:?} bytes={} ptr={:p}",
                        usage,
                        size_bytes,
                        Arc::as_ptr(&buffer)
                    );
                    return (buffer, true);
                }
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
        if let Ok(mut guard) = self.pools.lock() {
            let queue = guard.entry(key).or_insert_with(VecDeque::new);
            if queue.len() < self.max_per_key {
                log::trace!(
                    "buffer_residency: release {:?} bytes={} ptr={:p}",
                    usage,
                    allocated_bytes,
                    Arc::as_ptr(&buffer)
                );
                queue.push_back(buffer);
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
