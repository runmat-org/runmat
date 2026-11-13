#[cfg(target_arch = "wasm32")]
use anyhow::anyhow;
use anyhow::Result;
#[cfg(not(target_arch = "wasm32"))]
use once_cell::sync::OnceCell;

#[cfg(not(target_arch = "wasm32"))]
static INSTANCE: OnceCell<&'static WgpuProvider> = OnceCell::new();
#[cfg(target_arch = "wasm32")]
thread_local! {
    // The registered façade belongs to this browser worker for its lifetime.
    static INSTANCE: std::cell::Cell<Option<&'static WgpuProvider>> = const { std::cell::Cell::new(None) };
}

mod backend;

#[cfg(test)]
pub(crate) use backend::backend_shared::host_tensor_from_value;
pub use backend::backend_types::{WgpuPrecisionPolicy, WgpuProvider, WgpuProviderOptions};

/// Read the existing registered façade without initializing a device or
/// changing ambient provider selection.
pub fn registered_wgpu_provider() -> Option<&'static WgpuProvider> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        INSTANCE.get().copied()
    }
    #[cfg(target_arch = "wasm32")]
    {
        INSTANCE.with(std::cell::Cell::get)
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn register_wgpu_provider(opts: WgpuProviderOptions) -> Result<&'static WgpuProvider> {
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod test_session {
    use std::ops::Deref;
    use std::sync::{LazyLock, Mutex, MutexGuard};

    use runmat_accelerate_api::ThreadProviderGuard;

    use super::{register_wgpu_provider, WgpuProvider, WgpuProviderOptions};

    static TEST_PROVIDER_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    /// Exclusive access to a scoped test provider with independent state, so
    /// one workload cannot contaminate the next.
    pub(crate) struct WgpuTestSession {
        provider: WgpuProvider,
        _guard: MutexGuard<'static, ()>,
    }

    impl Deref for WgpuTestSession {
        type Target = WgpuProvider;

        fn deref(&self) -> &Self::Target {
            &self.provider
        }
    }

    impl WgpuTestSession {
        pub(crate) fn provider(&self) -> &WgpuProvider {
            self
        }
    }

    pub(crate) fn register_test_wgpu_provider(
        options: WgpuProviderOptions,
    ) -> anyhow::Result<WgpuTestSession> {
        let guard = TEST_PROVIDER_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Ok(WgpuTestSession {
            provider: WgpuProvider::new(options)?,
            _guard: guard,
        })
    }

    /// Tests of registered-owner semantics share the production singleton,
    /// exclusively with both retained and independently scoped test devices.
    pub(crate) struct RetainedWgpuTestSession {
        provider: &'static WgpuProvider,
        // Restore thread selection before releasing exclusive test access.
        _selection: ThreadProviderGuard,
        _guard: MutexGuard<'static, ()>,
    }

    impl RetainedWgpuTestSession {
        pub(crate) fn provider(&self) -> &'static WgpuProvider {
            self.provider
        }
    }

    pub(crate) fn register_retained_test_wgpu_provider(
        options: WgpuProviderOptions,
    ) -> anyhow::Result<RetainedWgpuTestSession> {
        let guard = TEST_PROVIDER_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Registration selects its singleton on this thread, including under
        // cfg(test). Capture the old selection first, also covering failures.
        let selection = ThreadProviderGuard::set(None);
        let provider = register_wgpu_provider(options)?;
        // SAFETY: the production singleton retains this same physical owner
        // for process lifetime. No scoped provider or borrowed façade is registered.
        unsafe { runmat_accelerate_api::register_device_provider(provider) };
        Ok(RetainedWgpuTestSession {
            provider,
            _selection: selection,
            _guard: guard,
        })
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) use test_session::{
    register_retained_test_wgpu_provider, register_test_wgpu_provider, WgpuTestSession,
};
