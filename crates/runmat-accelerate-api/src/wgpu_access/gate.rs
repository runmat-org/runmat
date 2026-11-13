//! No recursive acquisition: nested helpers receive the existing permit.
use super::WgpuAccessError;
#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::RwLock;

#[derive(Default)]
pub(super) struct Gate {
    #[cfg(not(target_arch = "wasm32"))]
    state: RwLock<()>,
    #[cfg(not(target_arch = "wasm32"))]
    fenced: AtomicBool,
    #[cfg(target_arch = "wasm32")]
    state: RefCell<()>,
}

impl Gate {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn fence_emission(&self) {
        let _guard = self
            .state
            .write()
            .unwrap_or_else(|error| error.into_inner());
        self.fenced.store(true, Ordering::Release);
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn is_fenced(&self) -> bool {
        self.fenced.load(Ordering::Acquire)
    }
    /// Recovery is teardown-only; never clear poison or admit new emission.
    pub fn cleanup<T>(&self, operation: impl FnOnce() -> T) -> Result<T, WgpuAccessError> {
        #[cfg(not(target_arch = "wasm32"))]
        let _guard = self
            .state
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        #[cfg(target_arch = "wasm32")]
        let _guard = self.state.try_borrow().map_err(|_| WgpuAccessError::Busy)?;
        Ok(operation())
    }

    pub fn shared<T>(&self, operation: impl FnOnce() -> T) -> Result<T, WgpuAccessError> {
        #[cfg(not(target_arch = "wasm32"))]
        let _guard = self.state.read().map_err(|_| WgpuAccessError::Poisoned)?;
        #[cfg(target_arch = "wasm32")]
        let _guard = self.state.try_borrow().map_err(|_| WgpuAccessError::Busy)?;
        #[cfg(not(target_arch = "wasm32"))]
        if self.is_fenced() {
            return Err(WgpuAccessError::Poisoned);
        }
        Ok(operation())
    }

    pub fn exclusive<T>(&self, operation: impl FnOnce() -> T) -> Result<T, WgpuAccessError> {
        #[cfg(not(target_arch = "wasm32"))]
        let _guard = self.state.write().map_err(|_| WgpuAccessError::Poisoned)?;
        #[cfg(target_arch = "wasm32")]
        let _guard = self
            .state
            .try_borrow_mut()
            .map_err(|_| WgpuAccessError::Busy)?;
        #[cfg(not(target_arch = "wasm32"))]
        if self.is_fenced() {
            return Err(WgpuAccessError::Poisoned);
        }
        Ok(operation())
    }
}
