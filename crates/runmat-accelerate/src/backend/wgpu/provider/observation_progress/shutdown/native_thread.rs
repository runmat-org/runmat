//! OS thread identity remains available after Rust thread-local destruction.
//! Exit callbacks must not initialize or access Rust's current-thread state.

use std::thread::JoinHandle;

#[cfg(unix)]
pub(super) fn is_current<T>(thread: &JoinHandle<T>) -> bool {
    use std::os::unix::thread::JoinHandleExt;

    // SAFETY: the borrowed join handle owns a live, joinable pthread identity.
    // pthread_self/equal do not depend on Rust thread-local initialization.
    unsafe { libc::pthread_equal(thread.as_pthread_t(), libc::pthread_self()) != 0 }
}

#[cfg(windows)]
pub(super) fn is_current<T>(thread: &JoinHandle<T>) -> bool {
    use std::{ffi::c_void, os::windows::io::AsRawHandle};

    #[link(name = "kernel32")]
    extern "system" {
        fn GetThreadId(thread: *mut c_void) -> u32;
        fn GetCurrentThreadId() -> u32;
    }

    // SAFETY: the borrowed join handle keeps the native thread handle valid.
    // These functions query OS identity and neither consume nor close it.
    unsafe { GetThreadId(thread.as_raw_handle()) == GetCurrentThreadId() }
}
