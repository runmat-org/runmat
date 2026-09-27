#[cfg(target_arch = "wasm32")]
use anyhow::anyhow;
use anyhow::Result;
#[cfg(not(target_arch = "wasm32"))]
use once_cell::sync::OnceCell;

mod backend;

#[cfg(test)]
pub(crate) use backend::backend_shared::host_tensor_from_value;
pub use backend::backend_types::{WgpuProvider, WgpuProviderOptions};

#[cfg(not(target_arch = "wasm32"))]
pub fn register_wgpu_provider(opts: WgpuProviderOptions) -> Result<&'static WgpuProvider> {
    static INSTANCE: OnceCell<&'static WgpuProvider> = OnceCell::new();
    let leaked: &'static WgpuProvider = *INSTANCE.get_or_try_init(move || {
        let provider = WgpuProvider::new(opts)?;
        let leaked: &'static WgpuProvider = Box::leak(Box::new(provider));
        #[cfg(not(test))]
        unsafe {
            runmat_accelerate_api::register_provider(leaked)
        };
        Ok::<&'static WgpuProvider, anyhow::Error>(leaked)
    })?;
    #[cfg(not(test))]
    unsafe {
        // Reinstall the WGPU provider reference (same singleton) to ensure it is the active global.
        runmat_accelerate_api::register_provider(leaked)
    };
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
    thread_local! {
        // The public provider ABI retains its façade for this worker's lifetime;
        // individual payloads still retire through their allocation leases.
        static INSTANCE: std::cell::Cell<Option<&'static WgpuProvider>> = const { std::cell::Cell::new(None) };
    }
    let leaked = if let Some(existing) = INSTANCE.with(std::cell::Cell::get) {
        existing
    } else {
        let provider = Box::new(WgpuProvider::new_async(opts).await?);
        // Another task in this worker may have initialized during the await.
        if let Some(existing) = INSTANCE.with(std::cell::Cell::get) {
            drop(provider);
            existing
        } else {
            let retained: &'static WgpuProvider = Box::leak(provider);
            INSTANCE.with(|cell| cell.set(Some(retained)));
            retained
        }
    };
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
