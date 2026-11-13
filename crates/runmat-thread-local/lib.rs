#[cfg(target_arch = "wasm32")]
use std::cell::UnsafeCell;

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
compile_error!(
    "runmat_thread_local wasm TLS shim assumes single-threaded wasm; use native thread_local semantics for wasm atomics builds"
);

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

    /// Matches native TLS access without a thread-destruction failure phase.
    /// Supported WASM cells are single-threaded statics and live for the instance.
    /// Panics from initialization or the callback propagate normally.
    pub fn try_with<R>(
        &'static self,
        f: impl FnOnce(&T) -> R,
    ) -> Result<R, std::thread::AccessError> {
        Ok(self.with(f))
    }
}

#[cfg(target_arch = "wasm32")]
unsafe impl<T> Sync for WasmTlsCell<T> {}

#[cfg(not(target_arch = "wasm32"))]
#[macro_export]
macro_rules! runmat_thread_local {
    ($(#[$meta:meta])* $vis:vis static $name:ident : $ty:ty = const { $init:expr }; $($rest:tt)*) => {
        thread_local! {
            $(#[$meta])*
            $vis static $name: $ty = const { $init };
        }
        $crate::runmat_thread_local! { $($rest)* }
    };
    ($(#[$meta:meta])* $vis:vis static $name:ident : $ty:ty = $init:expr; $($rest:tt)*) => {
        thread_local! {
            $(#[$meta])*
            $vis static $name: $ty = $init;
        }
        $crate::runmat_thread_local! { $($rest)* }
    };
    () => {};
}

#[cfg(target_arch = "wasm32")]
#[macro_export]
macro_rules! runmat_thread_local {
    ($(#[$meta:meta])* $vis:vis static $name:ident : $ty:ty = const { $init:expr }; $($rest:tt)*) => {
        $crate::runmat_thread_local! {
            $(#[$meta])*
            $vis static $name : $ty = $init;
            $($rest)*
        }
    };
    ($(#[$meta:meta])* $vis:vis static $name:ident : $ty:ty = $init:expr; $($rest:tt)*) => {
        $(#[$meta])*
        $vis static $name: $crate::WasmTlsCell<$ty> = {
            fn init() -> $ty {
                $init
            }
            $crate::WasmTlsCell::new(init)
        };
        $crate::runmat_thread_local! { $($rest)* }
    };
    () => {};
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    crate::runmat_thread_local! {
        static VALUE: Cell<u32> = const { Cell::new(7) };
    }

    #[test]
    fn fallible_and_infallible_access_share_storage() {
        VALUE.with(|value| value.set(11));
        assert_eq!(VALUE.try_with(Cell::get).expect("live TLS access"), 11);
        VALUE
            .try_with(|value| value.set(19))
            .expect("live TLS access");
        assert_eq!(VALUE.with(Cell::get), 19);
    }
}
