#[cfg(target_arch = "wasm32")]
use runmat_thread_local::runmat_thread_local;
#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{OnceLock, RwLock};

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
static GLOBAL_CONTEXT: OnceLock<RwLock<Option<SharedWgpuContext>>> = OnceLock::new();

#[cfg(target_arch = "wasm32")]
runmat_thread_local! {
    static GLOBAL_CONTEXT: RefCell<Option<SharedWgpuContext>> = RefCell::new(None);
}

#[cfg(not(target_arch = "wasm32"))]
fn global_context() -> &'static RwLock<Option<SharedWgpuContext>> {
    GLOBAL_CONTEXT.get_or_init(|| RwLock::new(None))
}

/// Install a shared context that other subsystems (GUI, exporters, web) can reuse.
pub fn install_shared_wgpu_context(context: SharedWgpuContext) {
    #[cfg(not(target_arch = "wasm32"))]
    if let Ok(mut slot) = global_context().write() {
        *slot = Some(context);
    }
    #[cfg(target_arch = "wasm32")]
    GLOBAL_CONTEXT.with(|cell| {
        *cell.borrow_mut() = Some(context);
    });
}

/// Retrieve the shared context if one has been installed.
pub fn shared_wgpu_context() -> Option<SharedWgpuContext> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        global_context()
            .read()
            .ok()
            .and_then(|context| context.clone())
    }
    #[cfg(target_arch = "wasm32")]
    {
        GLOBAL_CONTEXT.with(|cell| cell.borrow().clone())
    }
}
