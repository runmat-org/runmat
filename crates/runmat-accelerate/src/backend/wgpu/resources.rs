use std::collections::HashMap;
use std::sync::Mutex;

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
    uniform_buffers: Mutex<HashMap<UniformBufferKey, runmat_accelerate_api::WgpuBufferOwner>>,
    matmul_sources: Mutex<
        HashMap<
            runmat_accelerate_api::allocation::TensorViewIdentity,
            (GpuTensorHandle, GpuTensorHandle),
        >,
    >,
    scratch_buffers:
        Mutex<HashMap<ScratchBufferKind, (u64, runmat_accelerate_api::WgpuBufferOwner)>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KernelProvenanceLockPoisoned;

impl std::fmt::Display for KernelProvenanceLockPoisoned {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("WGPU matmul provenance lock poisoned")
    }
}

impl std::error::Error for KernelProvenanceLockPoisoned {}

impl Default for KernelResourceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ScratchBufferKind {
    QrGram,
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
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        key: UniformBufferKey,
        size: u64,
        label: &'static str,
    ) -> anyhow::Result<runmat_accelerate_api::WgpuBufferOwner> {
        let should_cache = !matches!(
            key,
            UniformBufferKey::MatmulParams
                | UniformBufferKey::LenOpParams
                | UniformBufferKey::BinaryBroadcastParams
                | UniformBufferKey::ScalarParamsF32
                | UniformBufferKey::ScalarParamsF64
                | UniformBufferKey::ReductionParams
                | UniformBufferKey::ReductionPass1Params
                | UniformBufferKey::ReductionPass2Params
        );
        if should_cache {
            if let Some(existing) = self
                .uniform_buffers
                .lock()
                .expect("uniform buffer registry poisoned")
                .get(&key)
                .cloned()
            {
                return Ok(existing);
            }
            super::cache::scope::allow_creation(emission.mode())?;
        }

        const UNIFORM_ALIGN: u64 = 256;
        let alloc_size = if size == 0 {
            UNIFORM_ALIGN
        } else {
            size.div_ceil(UNIFORM_ALIGN) * UNIFORM_ALIGN
        };
        let buffer = runmat_accelerate_api::WgpuBufferOwner::new(emission.device().create_buffer(
            &wgpu::BufferDescriptor {
                label: Some(label),
                size: alloc_size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        ));

        if should_cache {
            let mut guard = self
                .uniform_buffers
                .lock()
                .expect("uniform buffer registry poisoned");
            guard.entry(key).or_insert_with(|| buffer.clone());
        }
        Ok(buffer)
    }

    pub fn scratch_storage_buffer(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        kind: ScratchBufferKind,
        size: u64,
        label: &'static str,
    ) -> anyhow::Result<runmat_accelerate_api::WgpuBufferOwner> {
        let guard = self
            .scratch_buffers
            .lock()
            .expect("scratch buffer registry poisoned");
        if let Some((capacity, existing)) = guard.get(&kind) {
            if *capacity >= size {
                return Ok(existing.clone());
            }
        }
        drop(guard);
        super::cache::scope::allow_creation(emission.mode())?;
        let alloc_size = size.max(1);
        let buffer = runmat_accelerate_api::WgpuBufferOwner::new(emission.device().create_buffer(
            &wgpu::BufferDescriptor {
                label: Some(label),
                size: alloc_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        ));
        let mut guard = self
            .scratch_buffers
            .lock()
            .expect("scratch buffer registry poisoned");
        if let Some((capacity, existing)) = guard.get(&kind) {
            if *capacity >= size {
                return Ok(existing.clone());
            }
        }
        let replaced = guard.insert(kind, (alloc_size, buffer.clone()));
        drop(guard);
        drop(replaced);
        Ok(buffer)
    }

    pub fn remember_matmul_sources(
        &self,
        product: &GpuTensorHandle,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
    ) {
        let replaced =
            self.matmul_sources.lock().ok().and_then(|mut map| {
                map.insert(product.view_identity(), (lhs.clone(), rhs.clone()))
            });
        drop(replaced);
    }

    pub fn take_matmul_sources(
        &self,
        product: &GpuTensorHandle,
    ) -> Option<(GpuTensorHandle, GpuTensorHandle)> {
        self.matmul_sources
            .lock()
            .ok()
            .and_then(|mut map| map.remove(&product.view_identity()))
    }

    pub(crate) fn clear_matmul_source_checked(&self, product_id: u64) -> anyhow::Result<()> {
        let removed = {
            let mut map = self
                .matmul_sources
                .lock()
                .map_err(|_| KernelProvenanceLockPoisoned)?;
            let mut keys = Vec::new();
            keys.try_reserve(map.len())?;
            keys.extend(
                map.keys()
                    .filter(|key| key.allocation().allocation_id() == product_id)
                    .copied(),
            );
            let mut removed = Vec::new();
            removed.try_reserve(keys.len())?;
            for key in keys {
                if let Some(value) = map.remove(&key) {
                    removed.push(value);
                }
            }
            removed
        };
        drop(removed);
        Ok(())
    }
}
