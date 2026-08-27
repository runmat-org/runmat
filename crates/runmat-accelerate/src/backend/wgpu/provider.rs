#[cfg(target_arch = "wasm32")]
use anyhow::anyhow;
use anyhow::Result;
use once_cell::sync::OnceCell;

mod backend;

#[cfg(test)]
pub(crate) use backend::backend_shared::host_tensor_from_value;
pub use backend::backend_types::{WgpuProvider, WgpuProviderOptions};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod test_session {
    use std::ops::Deref;
    use std::sync::{LazyLock, Mutex, MutexGuard};

    use super::{register_wgpu_provider, WgpuProvider, WgpuProviderOptions};

    static TEST_PROVIDER_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    /// Exclusive access to the process-wide test provider.
    ///
    /// Production keeps the WGPU provider alive for the runtime session. Unit
    /// tests instead execute many independent workloads through that singleton;
    /// this scope prevents those workloads from overlapping and releases every
    /// provider-owned test buffer when each workload ends.
    pub(crate) struct WgpuTestSession {
        provider: &'static WgpuProvider,
        _guard: MutexGuard<'static, ()>,
    }

    impl Deref for WgpuTestSession {
        type Target = WgpuProvider;

        fn deref(&self) -> &Self::Target {
            self.provider
        }
    }

    impl WgpuTestSession {
        pub(crate) fn provider(&self) -> &'static WgpuProvider {
            self.provider
        }
    }

    impl Drop for WgpuTestSession {
        fn drop(&mut self) {
            self.provider.clear_test_state();
            runmat_accelerate_api::set_thread_provider(None);
        }
    }

    pub(crate) fn register_test_wgpu_provider(
        options: WgpuProviderOptions,
    ) -> anyhow::Result<WgpuTestSession> {
        let guard = TEST_PROVIDER_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let provider = register_wgpu_provider(options)?;
        provider.clear_test_state();
        Ok(WgpuTestSession {
            provider,
            _guard: guard,
        })
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) use test_session::{register_test_wgpu_provider, WgpuTestSession};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod test_session_tests {
    use runmat_accelerate_api::{AccelProvider as _, HostTensorView};

    use super::{register_test_wgpu_provider, WgpuProviderOptions};

    #[test]
    fn test_sessions_release_provider_owned_buffers() {
        let Ok(first) = register_test_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let handle = first
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[2, 1],
            })
            .expect("upload test buffer");
        assert!(first.test_buffer_count() > 0);
        drop(handle);
        drop(first);

        let second = register_test_wgpu_provider(WgpuProviderOptions::default())
            .expect("reopen test provider session");
        assert_eq!(second.test_buffer_count(), 0);
    }
}

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
