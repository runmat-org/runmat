//! Cross-process monotonic observations, comparable only within a live owner epoch.
//!
//! The caller must bind both readings to one freshly generated owner nonce. A
//! persisted reading alone is not a boot identifier and cannot survive an
//! owner restart as evidence of a continuous reservation interval.

use serde::{Deserialize, Serialize};

use crate::{ProcessHostError, ProcessHostResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostMonotonicReading {
    pub ticks: u64,
    pub ticks_per_second: u64,
}

impl HostMonotonicReading {
    pub fn elapsed_millis_ceil(self, end: Self) -> ProcessHostResult<u64> {
        if self.ticks_per_second == 0
            || self.ticks_per_second != end.ticks_per_second
            || end.ticks < self.ticks
        {
            return Err(invalid_clock());
        }
        let ticks = u128::from(end.ticks - self.ticks);
        let millis = ticks
            .checked_mul(1000)
            .and_then(|n| n.checked_add(u128::from(self.ticks_per_second) - 1))
            .map(|n| n / u128::from(self.ticks_per_second))
            .ok_or_else(invalid_clock)?;
        u64::try_from(millis).map_err(|_| invalid_clock())
    }
}

#[cfg(unix)]
pub fn observe_monotonic() -> ProcessHostResult<HostMonotonicReading> {
    let mut stamp = std::mem::MaybeUninit::<libc::timespec>::uninit();
    // Linux CLOCK_MONOTONIC excludes system suspension. A retained wall-time
    // reservation includes that interval, so use its suspend-aware boot clock.
    // Darwin CLOCK_MONOTONIC already includes sleep (unlike CLOCK_UPTIME_RAW).
    #[cfg(target_os = "linux")]
    let clock_id = libc::CLOCK_BOOTTIME;
    #[cfg(not(target_os = "linux"))]
    let clock_id = libc::CLOCK_MONOTONIC;
    // SAFETY: clock_gettime initializes the timespec on success.
    if unsafe { libc::clock_gettime(clock_id, stamp.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    // SAFETY: success above initialized the value.
    let stamp = unsafe { stamp.assume_init() };
    let seconds = u64::try_from(stamp.tv_sec).map_err(|_| invalid_clock())?;
    let nanos = u64::try_from(stamp.tv_nsec).map_err(|_| invalid_clock())?;
    if nanos >= 1_000_000_000 {
        return Err(invalid_clock());
    }
    Ok(HostMonotonicReading {
        ticks: seconds
            .checked_mul(1_000_000_000)
            .and_then(|value| value.checked_add(nanos))
            .ok_or_else(invalid_clock)?,
        ticks_per_second: 1_000_000_000,
    })
}

#[cfg(windows)]
pub fn observe_monotonic() -> ProcessHostResult<HostMonotonicReading> {
    use windows_sys::Win32::System::Performance::{
        QueryPerformanceCounter, QueryPerformanceFrequency,
    };

    let mut ticks = 0_i64;
    let mut frequency = 0_i64;
    // QPC is system-wide and has a boot-fixed frequency; the caller's owner
    // nonce is still required to reject observations after an agent restart.
    if unsafe { QueryPerformanceFrequency(&mut frequency) } == 0
        || unsafe { QueryPerformanceCounter(&mut ticks) } == 0
    {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(HostMonotonicReading {
        ticks: u64::try_from(ticks).map_err(|_| invalid_clock())?,
        ticks_per_second: u64::try_from(frequency).map_err(|_| invalid_clock())?,
    })
}

fn invalid_clock() -> ProcessHostError {
    ProcessHostError::Protocol("monotonic observation is not comparable".into())
}
