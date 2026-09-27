use runmat_accelerate_api::WgpuBufferRef;
use std::{ops::Deref, sync::Arc};

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

impl Deref for GpuInputBuffer {
    type Target = Arc<wgpu::Buffer>;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Renderer(buffer) => buffer,
            Self::Exported(export) => &export.buffer,
        }
    }
}

impl std::fmt::Debug for GpuInputBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuInputBuffer")
            .field("buffer", &**self)
            .field("provider_owned", &matches!(self, Self::Exported(_)))
            .finish()
    }
}
