use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use runmat_accelerate_api::GpuTensorHandle;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum UniformBufferKey {
    MatmulParams,
    SyrkParams,
    ImageNormalizeUniforms,
    CenteredGramParamsF32,
    CenteredGramParamsF64,
    QrPowerIterParams,
    LenOpParams,
    BinaryBroadcastParams,
    ScalarParamsF32,
    ScalarParamsF64,
    ReductionParams,
    ReductionPass1Params,
    ReductionPass2Params,
}

pub struct KernelResourceRegistry {
    uniform_buffers: Mutex<HashMap<UniformBufferKey, Arc<wgpu::Buffer>>>,
    matmul_sources: Mutex<HashMap<u64, (GpuTensorHandle, GpuTensorHandle)>>,
    scratch_buffers: Mutex<HashMap<(ScratchBufferKind, u64), Arc<wgpu::Buffer>>>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ScratchBufferKind {
    QrGram,
    QrR,
    QrRInv,
    SyrkOut,
    ReductionOut,
    ReductionPartials,
}

impl KernelResourceRegistry {
    pub fn new() -> Self {
        Self {
            uniform_buffers: Mutex::new(HashMap::new()),
            matmul_sources: Mutex::new(HashMap::new()),
            scratch_buffers: Mutex::new(HashMap::new()),
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

    pub fn scratch_storage_buffer(
        &self,
        device: &wgpu::Device,
        kind: ScratchBufferKind,
        size: u64,
        label: &'static str,
    ) -> Arc<wgpu::Buffer> {
        let key = (kind, size);
        if let Some(existing) = self
            .scratch_buffers
            .lock()
            .expect("scratch buffer registry poisoned")
            .get(&key)
            .cloned()
        {
            return existing;
        }
        let buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let mut guard = self
            .scratch_buffers
            .lock()
            .expect("scratch buffer registry poisoned");
        guard.entry(key).or_insert_with(|| buffer.clone());
        buffer
    }

    pub fn remember_matmul_sources(
        &self,
        product: &GpuTensorHandle,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
    ) {
        if let Ok(mut map) = self.matmul_sources.lock() {
            map.insert(product.buffer_id, (lhs.clone(), rhs.clone()));
        }
    }

    pub fn take_matmul_sources(
        &self,
        product: &GpuTensorHandle,
    ) -> Option<(GpuTensorHandle, GpuTensorHandle)> {
        self.matmul_sources
            .lock()
            .ok()
            .and_then(|mut map| map.remove(&product.buffer_id))
    }

    pub fn clear_matmul_source(&self, product_id: u64) {
        if let Ok(mut map) = self.matmul_sources.lock() {
            map.remove(&product_id);
        }
    }
}
