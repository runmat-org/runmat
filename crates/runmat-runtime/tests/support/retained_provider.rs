//! Retain one test provider per native process or browser worker for the
//! thread-provider facade, whose guard borrows a `'static` provider.
//!
//! Native providers retain their thread-safe OnceLock storage. Browser
//! providers hold worker-local allocation leases, so their one-time fixture
//! allocation must not be placed in a Sync static or recreated per call.

macro_rules! retained_test_provider {
    ($name:ident: $provider:ty = $init:expr) => {{
        #[cfg(not(target_arch = "wasm32"))]
        {
            static $name: std::sync::OnceLock<$provider> = std::sync::OnceLock::new();
            $name.get_or_init(|| $init)
        }
        #[cfg(target_arch = "wasm32")]
        {
            runmat_thread_local::runmat_thread_local! {
                static $name: std::cell::OnceCell<&'static $provider> = std::cell::OnceCell::new();
            }
            $name.with(|slot| {
                *slot.get_or_init(|| {
                    let provider: &'static $provider = Box::leak(Box::new($init));
                    provider
                })
            })
        }
    }};
}

pub(crate) use retained_test_provider;
