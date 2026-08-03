use tokio::process::Command;

use crate::{ProcessHostError, ProcessHostResult};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResourceLimits {
    pub memory_bytes: Option<u64>,
    pub cpu_seconds: Option<u64>,
    pub process_count: Option<u64>,
}

impl ResourceLimits {
    pub fn validate(self) -> ProcessHostResult<Self> {
        if self.memory_bytes == Some(0)
            || self.cpu_seconds == Some(0)
            || self.process_count == Some(0)
        {
            return Err(ProcessHostError::Configuration(
                "child resource limits must be greater than zero".into(),
            ));
        }
        Ok(self)
    }
}

#[cfg(unix)]
pub(super) fn configure(command: &mut Command, limits: ResourceLimits) {
    unsafe {
        command.pre_exec(move || {
            #[cfg(any(target_os = "linux", target_os = "android"))]
            set(limits.memory_bytes, libc::RLIMIT_AS)?;
            set(limits.cpu_seconds, libc::RLIMIT_CPU)?;
            #[cfg(any(target_os = "linux", target_os = "android"))]
            set(limits.process_count, libc::RLIMIT_NPROC)?;
            Ok(())
        });
    }
}

#[cfg(unix)]
#[cfg(any(target_os = "linux", target_os = "android"))]
type RlimitResource = libc::__rlimit_resource_t;

#[cfg(unix)]
#[cfg(not(any(target_os = "linux", target_os = "android")))]
type RlimitResource = libc::c_int;

#[cfg(unix)]
fn set(value: Option<u64>, resource: RlimitResource) -> std::io::Result<()> {
    let Some(value) = value else {
        return Ok(());
    };
    let mut limit: libc::rlimit = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrlimit(resource, &mut limit) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    limit.rlim_cur = (value as libc::rlim_t).min(limit.rlim_max);
    if unsafe { libc::setrlimit(resource, &limit) } == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(not(unix))]
pub(super) fn configure(_: &mut Command, _: ResourceLimits) {}
