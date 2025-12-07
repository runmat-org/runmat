use once_cell::sync::OnceCell;
use std::sync::Arc;

/// Shared WGPU instance/device/queue triple exported by a host acceleration provider.
#[derive(Clone)]
pub struct SharedWgpuContext {
    pub instance: Arc<wgpu::Instance>,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter: Arc<wgpu::Adapter>,
    pub adapter_info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
    pub features: wgpu::Features,
}

static GLOBAL_CONTEXT: OnceCell<SharedWgpuContext> = OnceCell::new();

/// Install a shared context that other subsystems (GUI, exporters, web) can reuse.
pub fn install_shared_wgpu_context(context: SharedWgpuContext) {
    let _ = GLOBAL_CONTEXT.set(context);
}

/// Retrieve the shared context if one has been installed.
pub fn shared_wgpu_context() -> Option<SharedWgpuContext> {
    GLOBAL_CONTEXT.get().cloned()
}
