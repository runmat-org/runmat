mod calendar;
mod thread_cpu;

use super::{RuntimeCalendarInstant, RuntimeTimingService};
use crate::context::RuntimeServiceFuture;
use calendar::CapturedCalendar;
use runmat_execution::profiling::CommandThreadCpuSampler;
use runmat_time::Instant;
use std::sync::Arc;
use std::time::Duration;

pub(super) struct PlatformTimingService {
    origin: Instant,
    calendar: Option<CapturedCalendar>,
}

impl PlatformTimingService {
    pub(super) fn new() -> Self {
        Self {
            origin: Instant::now(),
            calendar: CapturedCalendar::try_system().ok(),
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

    fn local_datetime_to_instant(
        &self,
        local: chrono::NaiveDateTime,
    ) -> Option<RuntimeCalendarInstant> {
        self.calendar.as_ref()?.local_to_instant(local)
    }

    fn local_datetime_at_instant(
        &self,
        instant: RuntimeCalendarInstant,
    ) -> Option<chrono::NaiveDateTime> {
        self.calendar.as_ref()?.local_at_instant(instant)
    }

    fn process_cpu_seconds(&self) -> Option<f64> {
        process_cpu_seconds()
    }

    fn command_thread_cpu_sampler(&self) -> Option<Arc<dyn CommandThreadCpuSampler>> {
        thread_cpu::platform_sampler()
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
    let mut remaining =
        duration.as_millis() + u128::from(!duration.subsec_nanos().is_multiple_of(1_000_000));
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
        global: global.into(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn unavailable_calendar_does_not_fall_back_to_utc() {
        let timing = PlatformTimingService {
            origin: Instant::now(),
            calendar: None,
        };
        let local = NaiveDate::from_ymd_opt(2024, 1, 1)
            .expect("test date")
            .and_hms_opt(0, 0, 0)
            .expect("test time");

        assert_eq!(timing.local_datetime_to_instant(local), None);
        assert_eq!(
            timing.local_datetime_at_instant(
                RuntimeCalendarInstant::from_unix_timestamp(0, 0).unwrap()
            ),
            None
        );
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test]
    fn browser_platform_calendar_matches_intl_zone_and_round_trips() {
        use wasm_bindgen::JsValue;

        let format =
            js_sys::Intl::DateTimeFormat::new(&js_sys::Array::new(), &js_sys::Object::new());
        let options = format.resolved_options();
        let zone_name = js_sys::Reflect::get(&options, &JsValue::from_str("timeZone"))
            .expect("Intl resolvedOptions.timeZone access")
            .as_string()
            .filter(|name| !name.is_empty())
            .expect("browser-selected IANA time zone");

        let timing = PlatformTimingService::new();
        let epoch = Duration::new(1_705_320_000, 123_000_000);
        let instant = RuntimeCalendarInstant::from_epoch_duration(epoch).unwrap();
        let date = js_sys::Date::new(&JsValue::from_f64(epoch.as_millis() as f64));
        let expected = NaiveDate::from_ymd_opt(
            i32::try_from(date.get_full_year()).expect("browser year"),
            date.get_month() + 1,
            date.get_date(),
        )
        .and_then(|date_value| {
            date_value.and_hms_milli_opt(
                date.get_hours(),
                date.get_minutes(),
                date.get_seconds(),
                date.get_milliseconds(),
            )
        })
        .unwrap_or_else(|| panic!("browser date in {zone_name} must be representable"));

        let first = timing
            .local_datetime_at_instant(instant)
            .unwrap_or_else(|| panic!("captured browser calendar unavailable for {zone_name}"));
        assert_eq!(first, expected, "captured calendar must match browser zone");
        assert_eq!(
            timing.local_datetime_to_instant(first),
            Some(instant),
            "paired conversion must use the same captured rules"
        );
        assert_eq!(
            timing.local_datetime_at_instant(instant),
            Some(first),
            "the service must retain its captured calendar policy"
        );
    }
}
