use anyhow::Result;
use once_cell::sync::OnceCell;

pub use crate::backend::wgpu::provider_impl::{WgpuProvider, WgpuProviderOptions};

pub fn register_wgpu_provider(opts: WgpuProviderOptions) -> Result<&'static WgpuProvider> {
    static INSTANCE: OnceCell<&'static WgpuProvider> = OnceCell::new();
    let leaked: &'static WgpuProvider = *INSTANCE.get_or_try_init(move || {
        let provider = WgpuProvider::new(opts)?;
        let leaked: &'static WgpuProvider = Box::leak(Box::new(provider));
        // Register on first creation
        unsafe { runmat_accelerate_api::register_provider(leaked) };
        Ok::<&'static WgpuProvider, anyhow::Error>(leaked)
    })?;
    // Reinstall the WGPU provider reference (same singleton) to ensure it is the active global.
    unsafe { runmat_accelerate_api::register_provider(leaked) };
    runmat_accelerate_api::set_thread_provider(Some(leaked));
    Ok(leaked)
}

pub fn ensure_wgpu_provider() -> Result<Option<&'static WgpuProvider>> {
    match register_wgpu_provider(WgpuProviderOptions::default()) {
        Ok(p) => Ok(Some(p)),
        Err(e) => {
            log::warn!("RunMat Accelerate: wgpu provider initialization failed: {e}");
            Ok(None)
        }
    }
}
