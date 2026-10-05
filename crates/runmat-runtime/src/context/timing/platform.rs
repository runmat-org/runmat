use super::RuntimeTimingService;
use crate::context::RuntimeServiceFuture;
use runmat_time::Instant;
use std::time::Duration;

pub(super) struct PlatformTimingService {
    origin: Instant,
}

impl PlatformTimingService {
    pub(super) fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl RuntimeTimingService for PlatformTimingService {
    fn monotonic_elapsed(&self) -> Duration {
        Instant::now()
            .checked_duration_since(self.origin)
            .unwrap_or(Duration::ZERO)
    }

    fn process_cpu_seconds(&self) -> Option<f64> {
        process_cpu_seconds()
    }

    fn sleep(&self, duration: Duration) -> RuntimeServiceFuture<Result<(), crate::RuntimeError>> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Box::pin(async move {
                async_io::Timer::after(duration).await;
                Ok(())
            })
        }
        #[cfg(target_arch = "wasm32")]
        {
            Box::pin(browser_sleep(duration))
        }
    }
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
fn process_cpu_seconds() -> Option<f64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: the pointer is valid writable storage; success initializes it.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    let usage = unsafe { usage.assume_init() };
    let seconds = |value: libc::timeval| value.tv_sec as f64 + value.tv_usec as f64 / 1_000_000.0;
    Some(seconds(usage.ru_utime) + seconds(usage.ru_stime))
}

#[cfg(windows)]
fn process_cpu_seconds() -> Option<f64> {
    use std::ffi::c_void;
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct FileTime {
        low_date_time: u32,
        high_date_time: u32,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn GetProcessTimes(
            process: *mut c_void,
            creation_time: *mut FileTime,
            exit_time: *mut FileTime,
            kernel_time: *mut FileTime,
            user_time: *mut FileTime,
        ) -> i32;
    }
    let mut creation = std::mem::MaybeUninit::<FileTime>::uninit();
    let mut exit = std::mem::MaybeUninit::<FileTime>::uninit();
    let mut kernel = std::mem::MaybeUninit::<FileTime>::uninit();
    let mut user = std::mem::MaybeUninit::<FileTime>::uninit();
    // SAFETY: GetCurrentProcess returns a process pseudo-handle and all output
    // pointers designate writable storage; success initializes kernel/user.
    if unsafe {
        GetProcessTimes(
            GetCurrentProcess(),
            creation.as_mut_ptr(),
            exit.as_mut_ptr(),
            kernel.as_mut_ptr(),
            user.as_mut_ptr(),
        )
    } == 0
    {
        return None;
    }
    let ticks =
        |value: FileTime| ((value.high_date_time as u64) << 32) | value.low_date_time as u64;
    let (kernel, user) = unsafe { (kernel.assume_init(), user.assume_init()) };
    Some((ticks(kernel) + ticks(user)) as f64 / 10_000_000.0)
}

#[cfg(target_arch = "wasm32")]
fn process_cpu_seconds() -> Option<f64> {
    None
}

#[cfg(target_arch = "wasm32")]
async fn browser_sleep(duration: Duration) -> Result<(), crate::RuntimeError> {
    use js_sys::{Function, Promise, Reflect};
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;

    let global = js_sys::global();
    let set_timeout = Reflect::get(&global, &wasm_bindgen::JsValue::from_str("setTimeout"))
        .map_err(|_| crate::RuntimeError::new("pause: setTimeout unavailable"))?
        .dyn_into::<Function>()
        .map_err(|_| crate::RuntimeError::new("pause: setTimeout unavailable"))?;
    let millis = duration.as_millis().min(i32::MAX as u128) as i32;
    let promise = Promise::new(&mut |resolve, _reject| {
        let resolve: Function = resolve.unchecked_into();
        let _ = set_timeout.call2(
            &global,
            &resolve.into(),
            &wasm_bindgen::JsValue::from_f64(millis as f64),
        );
    });
    JsFuture::from(promise)
        .await
        .map_err(|error| crate::RuntimeError::new(format!("pause: timer failed ({error:?})")))?;
    Ok(())
}
