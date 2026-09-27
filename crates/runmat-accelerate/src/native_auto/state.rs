//! Offload configuration follows the provider registry's execution scope.
use super::NativeAutoOffload;
#[cfg(not(target_arch = "wasm32"))]
pub type AutoOffloadHandle = std::sync::Arc<NativeAutoOffload>;
#[cfg(target_arch = "wasm32")]
pub type AutoOffloadHandle = std::rc::Rc<NativeAutoOffload>;

#[cfg(not(target_arch = "wasm32"))]
static GLOBAL: once_cell::sync::OnceCell<Option<AutoOffloadHandle>> =
    once_cell::sync::OnceCell::new();

#[cfg(target_arch = "wasm32")]
thread_local! {
    static GLOBAL: std::cell::OnceCell<Option<AutoOffloadHandle>> = const {
        std::cell::OnceCell::new()
    };
}

pub(super) fn cached() -> Option<Option<AutoOffloadHandle>> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        GLOBAL.get().cloned()
    }
    #[cfg(target_arch = "wasm32")]
    {
        GLOBAL.with(|state| state.get().cloned())
    }
}

pub(super) fn initialize(value: Option<NativeAutoOffload>) {
    let value = value.map(AutoOffloadHandle::new);
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = GLOBAL.set(value);
    }
    #[cfg(target_arch = "wasm32")]
    {
        GLOBAL.with(|state| {
            let _ = state.set(value);
        });
    }
}
