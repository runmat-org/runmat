use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum UniformBufferKey {
    MatmulParams,
    SyrkParams,
    ImageNormalizeUniforms,
}

pub struct KernelResourceRegistry {
    uniform_buffers: Mutex<HashMap<UniformBufferKey, Arc<wgpu::Buffer>>>,
}

impl KernelResourceRegistry {
    pub fn new() -> Self {
        Self {
            uniform_buffers: Mutex::new(HashMap::new()),
        }
    }

    pub fn uniform_buffer(
        &self,
        device: &wgpu::Device,
        key: UniformBufferKey,
        size: u64,
        label: &'static str,
    ) -> Arc<wgpu::Buffer> {
        if let Some(existing) = self
            .uniform_buffers
            .lock()
            .expect("uniform buffer registry poisoned")
            .get(&key)
            .cloned()
        {
            return existing;
        }

        let buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let mut guard = self
            .uniform_buffers
            .lock()
            .expect("uniform buffer registry poisoned");
        guard.entry(key).or_insert_with(|| buffer.clone());
        buffer
    }
}
