use std::sync::Arc;

/// GPU context used by plot elements to build viewport-aware GPU geometry.
///
/// On wasm, some geometry generation is deferred until render-time so we can incorporate the
/// current viewport pixel size (e.g. to convert `LineWidth` pixels → data units).
pub struct GpuPackContext<'a> {
    pub device: &'a Arc<wgpu::Device>,
    pub queue: &'a Arc<wgpu::Queue>,
}

