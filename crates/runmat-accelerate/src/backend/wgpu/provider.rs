#[cfg(target_arch = "wasm32")]
use anyhow::anyhow;
use anyhow::Result;
use once_cell::sync::OnceCell;

pub use crate::backend::wgpu::provider_impl::{WgpuProvider, WgpuProviderOptions};

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(target_arch = "wasm32")]
pub fn register_wgpu_provider(_opts: WgpuProviderOptions) -> Result<&'static WgpuProvider> {
    Err(anyhow!(
        "RunMat Accelerate: synchronous WGPU initialization is unavailable on wasm targets. Call register_wgpu_provider_async instead."
    ))
}

#[cfg(target_arch = "wasm32")]
pub async fn register_wgpu_provider_async(
    opts: WgpuProviderOptions,
) -> Result<&'static WgpuProvider> {
    static INSTANCE: OnceCell<Box<WgpuProvider>> = OnceCell::new();
    if INSTANCE.get().is_none() {
        let provider = Box::new(WgpuProvider::new_async(opts).await?);
        if INSTANCE.set(provider).is_err() {
            log::warn!("RunMat Accelerate: WGPU provider was initialized concurrently; reusing existing instance");
        }
    }
    let leaked: &'static WgpuProvider = INSTANCE
        .get()
        .map(|boxed| boxed.as_ref())
        .ok_or_else(|| anyhow!("wgpu provider failed to initialize"))?;
    unsafe { runmat_accelerate_api::register_provider(leaked) };
    runmat_accelerate_api::set_thread_provider(Some(leaked));
    Ok(leaked)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn ensure_wgpu_provider() -> Result<Option<&'static WgpuProvider>> {
    match register_wgpu_provider(WgpuProviderOptions::default()) {
        Ok(p) => Ok(Some(p)),
        Err(e) => {
            log::warn!("RunMat Accelerate: wgpu provider initialization failed: {e}");
            Ok(None)
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub async fn ensure_wgpu_provider_async() -> Result<Option<&'static WgpuProvider>> {
    match register_wgpu_provider_async(WgpuProviderOptions::default()).await {
        Ok(p) => Ok(Some(p)),
        Err(e) => {
            log::warn!("RunMat Accelerate: wgpu provider initialization failed: {e}");
            Ok(None)
        }
    }
}
