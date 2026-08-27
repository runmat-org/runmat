#[cfg(target_arch = "wasm32")]
use anyhow::anyhow;
use anyhow::Result;
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod test_session {
    use std::ops::Deref;
    use std::sync::{LazyLock, Mutex, MutexGuard};

    use once_cell::sync::OnceCell;

    use super::{WgpuProvider, WgpuProviderOptions};

    static TEST_PROVIDER_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));
    static TEST_GPU_RUNTIME: OnceCell<WgpuProvider> = OnceCell::new();

    /// Exclusive access to a scoped test provider.
    ///
    /// Production keeps the WGPU provider alive for the runtime session. Unit
    /// tests need independent provider state so one workload cannot contaminate
    /// the next. The physical device remains stable across sessions because
    /// repeatedly creating native devices can itself exhaust a backend.
    pub(crate) struct WgpuTestSession {
        provider: Option<WgpuProvider>,
        _guard: MutexGuard<'static, ()>,
    }

    impl Deref for WgpuTestSession {
        type Target = WgpuProvider;

        fn deref(&self) -> &Self::Target {
            self.provider
                .as_ref()
                .expect("test provider session is active")
        }
    }

    impl WgpuTestSession {
        pub(crate) fn provider(&self) -> &WgpuProvider {
            self
        }
    }

    impl Drop for WgpuTestSession {
        fn drop(&mut self) {
            if let Some(provider) = self.provider.take() {
                let device = provider.test_device_handle();
                drop(provider);
                device.poll(wgpu::Maintain::Wait);
            }
        }
    }

    pub(crate) fn register_test_wgpu_provider(
        options: WgpuProviderOptions,
    ) -> anyhow::Result<WgpuTestSession> {
        let guard = TEST_PROVIDER_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let runtime = TEST_GPU_RUNTIME.get_or_try_init(|| WgpuProvider::new(options))?;
        let provider = runtime.new_session();
        Ok(WgpuTestSession {
            provider: Some(provider),
            _guard: guard,
        })
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) use test_session::{register_test_wgpu_provider, WgpuTestSession};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod test_session_tests {
    use runmat_accelerate_api::AccelProvider as _;

    use super::{register_test_wgpu_provider, WgpuProviderOptions};

    #[test]
    fn test_sessions_use_fresh_provider_state() {
        // Use a size dedicated to this lifecycle test so unrelated tests cannot
        // pre-populate the same physical residency key.
        const TEST_LEN: usize = 7_919;
        let Ok(first) = register_test_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let handle = first
            .zeros(&[TEST_LEN, 1])
            .expect("allocate first session buffer through provider residency");
        let first_buffer_ptr = first
            .test_buffer_ptr(&handle)
            .expect("first session owns uploaded buffer");
        assert_eq!(first.test_buffer_count(), 1);
        drop(handle);
        drop(first);

        let second = register_test_wgpu_provider(WgpuProviderOptions::default())
            .expect("reopen test provider session");
        assert_eq!(second.test_buffer_count(), 0);
        let second_handle = second
            .zeros(&[TEST_LEN, 1])
            .expect("allocate second session buffer through provider residency");
        assert_eq!(second.test_buffer_count(), 1);
        assert_eq!(
            second.test_buffer_ptr(&second_handle),
            Some(first_buffer_ptr),
            "logical sessions should reuse released physical storage"
        );
        drop(second_handle);
    }
}
