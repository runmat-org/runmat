//! Native per-thread CPU clock. The process-wide `cputime` clock is separate.

use runmat_execution::profiling::CommandThreadCpuSampler;
use std::sync::Arc;
#[cfg(any(target_os = "linux", target_os = "macos", windows))]
use std::sync::OnceLock;
#[cfg(any(target_os = "linux", target_os = "macos", windows))]
use std::time::Duration;

#[cfg(any(target_os = "linux", target_os = "macos", windows))]
struct PlatformThreadCpuSampler;

pub(super) fn platform_sampler() -> Option<Arc<dyn CommandThreadCpuSampler>> {
    #[cfg(any(target_os = "linux", target_os = "macos", windows))]
    {
        static SAMPLER: OnceLock<Arc<dyn CommandThreadCpuSampler>> = OnceLock::new();
        Some(
            SAMPLER
                .get_or_init(|| Arc::new(PlatformThreadCpuSampler))
                .clone(),
        )
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    {
        None
    }
}

#[cfg(any(target_os = "linux", target_os = "macos", windows))]
impl CommandThreadCpuSampler for PlatformThreadCpuSampler {
    fn sample_current_thread(&self) -> Option<Duration> {
        sample_current_thread()
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn sample_current_thread() -> Option<Duration> {
    let mut time = std::mem::MaybeUninit::<libc::timespec>::uninit();
    // SAFETY: clock_gettime writes the valid timespec pointer on success.
    if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, time.as_mut_ptr()) } != 0 {
        return None;
    }
    let time = unsafe { time.assume_init() };
    let seconds = u64::try_from(time.tv_sec).ok()?;
    let nanos = u32::try_from(time.tv_nsec).ok()?;
    (nanos < 1_000_000_000).then_some(Duration::new(seconds, nanos))
}

#[cfg(windows)]
fn sample_current_thread() -> Option<Duration> {
    use std::ffi::c_void;
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct FileTime {
        low_date_time: u32,
        high_date_time: u32,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThread() -> *mut c_void;
        fn GetThreadTimes(
            thread: *mut c_void,
            creation: *mut FileTime,
            exit: *mut FileTime,
            kernel: *mut FileTime,
            user: *mut FileTime,
        ) -> i32;
    }
    let mut creation = std::mem::MaybeUninit::<FileTime>::uninit();
    let mut exit = std::mem::MaybeUninit::<FileTime>::uninit();
    let mut kernel = std::mem::MaybeUninit::<FileTime>::uninit();
    let mut user = std::mem::MaybeUninit::<FileTime>::uninit();
    // SAFETY: the pseudo-handle names this thread and all pointers are valid
    // writable storage; success initializes kernel and user counters.
    if unsafe {
        GetThreadTimes(
            GetCurrentThread(),
            creation.as_mut_ptr(),
            exit.as_mut_ptr(),
            kernel.as_mut_ptr(),
            user.as_mut_ptr(),
        )
    } == 0
    {
        return None;
    }
    let kernel = unsafe { kernel.assume_init() };
    let user = unsafe { user.assume_init() };
    let ticks =
        |value: FileTime| (u64::from(value.high_date_time) << 32) | u64::from(value.low_date_time);
    let total = ticks(kernel).checked_add(ticks(user))?;
    Some(Duration::from_nanos(total.checked_mul(100)?))
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos", windows)))]
mod tests {
    use super::*;

    #[test]
    fn platform_counter_is_available_and_nondecreasing_on_this_thread() {
        let sampler = platform_sampler().expect("native sampler");
        let start = sampler.sample_current_thread().expect("thread counter");
        let end = sampler.sample_current_thread().expect("thread counter");
        assert!(end >= start);
    }
}
