#[cfg(not(target_arch = "wasm32"))]
use once_cell::sync::OnceCell;
#[cfg(target_arch = "wasm32")]
use once_cell::unsync::OnceCell;
#[cfg(target_arch = "wasm32")]
use runmat_thread_local::runmat_thread_local;
#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
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

#[cfg(not(target_arch = "wasm32"))]
static GLOBAL_CONTEXT: OnceCell<SharedWgpuContext> = OnceCell::new();

#[cfg(target_arch = "wasm32")]
runmat_thread_local! {
    static GLOBAL_CONTEXT: RefCell<OnceCell<SharedWgpuContext>> = RefCell::new(OnceCell::new());
}

/// Install a shared context that other subsystems (GUI, exporters, web) can reuse.
pub fn install_shared_wgpu_context(context: SharedWgpuContext) {
    #[cfg(not(target_arch = "wasm32"))]
    let _ = GLOBAL_CONTEXT.set(context);
    #[cfg(target_arch = "wasm32")]
    GLOBAL_CONTEXT.with(|cell| {
        let _ = cell.borrow_mut().set(context);
    });
}

/// Retrieve the shared context if one has been installed.
pub fn shared_wgpu_context() -> Option<SharedWgpuContext> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        GLOBAL_CONTEXT.get().cloned()
    }
    #[cfg(target_arch = "wasm32")]
    {
        GLOBAL_CONTEXT.with(|cell| cell.borrow().get().cloned())
    }
}
