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

    fn wall_time_since_epoch(&self) -> Duration {
        runmat_time::duration_since_epoch()
    }

    fn process_cpu_seconds(&self) -> Option<f64> {
        process_cpu_seconds()
    }

    fn sleep(&self, duration: Duration) -> RuntimeServiceFuture<Result<(), crate::RuntimeError>> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Box::pin(async move {
                if Instant::now().checked_add(duration).is_none() {
                    return Err(crate::RuntimeError::new(
                        "timing: duration exceeds host clock range",
                    ));
                }
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
    Some(ticks(kernel) as f64 / 10_000_000.0 + ticks(user) as f64 / 10_000_000.0)
}

#[cfg(target_arch = "wasm32")]
fn process_cpu_seconds() -> Option<f64> {
    None
}

#[cfg(target_arch = "wasm32")]
async fn browser_sleep(duration: Duration) -> Result<(), crate::RuntimeError> {
    let mut remaining = duration.as_millis() + u128::from(duration.subsec_nanos() % 1_000_000 != 0);
    loop {
        let chunk = remaining.min(i32::MAX as u128) as i32;
        browser_sleep_chunk(chunk).await?;
        if remaining <= chunk as u128 {
            break;
        }
        remaining -= chunk as u128;
    }
    Ok(())
}

#[cfg(target_arch = "wasm32")]
async fn browser_sleep_chunk(millis: i32) -> Result<(), crate::RuntimeError> {
    use futures::channel::oneshot;
    use js_sys::{Function, Reflect};
    use wasm_bindgen::{closure::Closure, JsCast, JsValue};

    struct ScheduledTimeout {
        global: JsValue,
        clear: Function,
        id: JsValue,
        _callback: Closure<dyn FnMut()>,
    }

    impl Drop for ScheduledTimeout {
        fn drop(&mut self) {
            let _ = self.clear.call1(&self.global, &self.id);
        }
    }

    let global = js_sys::global();
    let timer = |name: &str| -> Result<Function, crate::RuntimeError> {
        Reflect::get(&global, &JsValue::from_str(name))
            .map_err(|_| crate::RuntimeError::new(format!("pause: {name} unavailable")))?
            .dyn_into::<Function>()
            .map_err(|_| crate::RuntimeError::new(format!("pause: {name} unavailable")))
    };
    let set = timer("setTimeout")?;
    let clear = timer("clearTimeout")?;
    let (sender, receiver) = oneshot::channel::<()>();
    let mut sender = Some(sender);
    let callback = Closure::wrap(Box::new(move || {
        if let Some(sender) = sender.take() {
            let _ = sender.send(());
        }
    }) as Box<dyn FnMut()>);
    let id = set
        .call2(
            &global,
            callback.as_ref(),
            &JsValue::from_f64(millis as f64),
        )
        .map_err(|error| {
            crate::RuntimeError::new(format!("pause: timer setup failed ({error:?})"))
        })?;
    let scheduled = ScheduledTimeout {
        global,
        clear,
        id,
        _callback: callback,
    };
    receiver
        .await
        .map_err(|_| crate::RuntimeError::new("pause: timer callback was dropped"))?;
    drop(scheduled);
    Ok(())
}
