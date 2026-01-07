#[cfg(target_arch = "wasm32")]
use std::cell::UnsafeCell;

#[cfg(target_arch = "wasm32")]
pub struct WasmTlsCell<T> {
    init: fn() -> T,
    value: UnsafeCell<Option<T>>,
}

#[cfg(target_arch = "wasm32")]
impl<T> WasmTlsCell<T> {
    pub const fn new(init: fn() -> T) -> Self {
        Self {
            init,
            value: UnsafeCell::new(None),
        }
    }

    fn ensure(&self) -> *mut T {
        unsafe {
            let slot = &mut *self.value.get();
            if slot.is_none() {
                *slot = Some((self.init)());
            }
            slot.as_mut().expect("runmat wasm TLS slot init failed") as *mut T
        }
    }

    pub fn with<R>(&self, f: impl FnOnce(&T) -> R) -> R {
        let ptr = self.ensure();
        unsafe { f(&*ptr) }
    }

    pub fn with_mut<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        let ptr = self.ensure();
        unsafe { f(&mut *ptr) }
    }
}

#[cfg(target_arch = "wasm32")]
unsafe impl<T> Sync for WasmTlsCell<T> {}

#[cfg(not(target_arch = "wasm32"))]
#[macro_export]
macro_rules! runmat_thread_local {
    ($(#[$meta:meta])* static $name:ident : $ty:ty = const { $init:expr }; $($rest:tt)*) => {
        thread_local! {
            $(#[$meta])*
            static $name: $ty = const { $init };
        }
        $crate::runmat_thread_local! { $($rest)* }
    };
    ($(#[$meta:meta])* static $name:ident : $ty:ty = $init:expr; $($rest:tt)*) => {
        thread_local! {
            $(#[$meta])*
            static $name: $ty = $init;
        }
        $crate::runmat_thread_local! { $($rest)* }
    };
    () => {};
}

#[cfg(target_arch = "wasm32")]
#[macro_export]
macro_rules! runmat_thread_local {
    ($(#[$meta:meta])* static $name:ident : $ty:ty = const { $init:expr }; $($rest:tt)*) => {
        $crate::runmat_thread_local! {
            $(#[$meta])*
            static $name : $ty = $init;
            $($rest)*
        }
    };
    ($(#[$meta:meta])* static $name:ident : $ty:ty = $init:expr; $($rest:tt)*) => {
        $(#[$meta])*
        static $name: $crate::WasmTlsCell<$ty> = {
            fn init() -> $ty {
                $init
            }
            $crate::WasmTlsCell::new(init)
        };
        $crate::runmat_thread_local! { $($rest)* }
    };
    () => {};
}
