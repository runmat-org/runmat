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
    len: usize,
}

impl ResidencyKey {
    fn new(usage: BufferUsageClass, len: usize) -> Self {
        Self { usage, len }
    }
}

impl Hash for ResidencyKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.usage.hash(state);
        self.len.hash(state);
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
    ) -> Arc<wgpu::Buffer> {
        if len == 0 {
            return Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: element_size.max(1) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        let key = ResidencyKey::new(usage, len);
        if let Ok(mut guard) = self.pools.lock() {
            if let Some(queue) = guard.get_mut(&key) {
                if let Some(buffer) = queue.pop_front() {
                    log::trace!(
                        "buffer_residency: reuse {:?} len={} ptr={:p}",
                        usage,
                        len,
                        Arc::as_ptr(&buffer)
                    );
                    return buffer;
                }
            }
        }

        let size_bytes = (len as u64).max(1) * element_size as u64;
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
        buffer
    }

    pub fn release(&self, usage: BufferUsageClass, len: usize, buffer: Arc<wgpu::Buffer>) {
        if len == 0 {
            return;
        }

        let key = ResidencyKey::new(usage, len);
        if let Ok(mut guard) = self.pools.lock() {
            let queue = guard.entry(key).or_insert_with(VecDeque::new);
            if queue.len() < self.max_per_key {
                log::trace!(
                    "buffer_residency: release {:?} len={} ptr={:p}",
                    usage,
                    len,
                    Arc::as_ptr(&buffer)
                );
                queue.push_back(buffer);
            } else {
                log::trace!(
                    "buffer_residency: drop {:?} len={} ptr={:p} (pool full)",
                    usage,
                    len,
                    Arc::as_ptr(&buffer)
                );
            }
        }
    }
}
