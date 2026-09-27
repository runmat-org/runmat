use runmat_accelerate_api::{WgpuAccessError, WgpuBufferRef, WgpuEmission};
use std::sync::Arc;

/// A renderer-owned buffer or an authenticated provider export.
/// Clones of a provider input retain its physical allocation authority, including
/// a distinct allocation created when the provider materializes a lazy view.
#[derive(Clone)]
pub enum GpuInputBuffer {
    Renderer(Arc<wgpu::Buffer>),
    Exported(WgpuBufferRef),
}

impl From<Arc<wgpu::Buffer>> for GpuInputBuffer {
    fn from(buffer: Arc<wgpu::Buffer>) -> Self {
        Self::Renderer(buffer)
    }
}

impl From<WgpuBufferRef> for GpuInputBuffer {
    fn from(export: WgpuBufferRef) -> Self {
        Self::Exported(export)
    }
}

impl GpuInputBuffer {
    /// Borrow backing only during the caller's lexical emission. Renderer
    /// buffers are created in that renderer's device domain; provider exports
    /// additionally authenticate the exact access authority and live lease.
    pub fn buffer<'a>(
        &'a self,
        emission: &'a WgpuEmission<'_>,
    ) -> Result<&'a wgpu::Buffer, WgpuAccessError> {
        match self {
            Self::Renderer(buffer) => Ok(buffer),
            Self::Exported(export) => export.buffer(emission),
        }
    }
}

impl std::fmt::Debug for GpuInputBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuInputBuffer")
            .field("provider_owned", &matches!(self, Self::Exported(_)))
            .finish()
    }
}
