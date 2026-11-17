//! The memory this process may use, measured once: what a native process sizes its part cache by
//! ([`super::part_cache`]).
//!
//! The capacity is the first of these that is set: the cgroup memory limit (a Linux container),
//! the process's address-space limit (`RLIMIT_AS`) when it is finite, else the machine's physical
//! memory. A target that can measure none of them (wasm) has no capacity here; its host measures
//! its own memory and sets the cache's budget.
use std::sync::OnceLock;

/// A limit at or past 2^62 bytes is the platform's "unlimited" marker (cgroup v1 and
/// `RLIM_INFINITY` both use one), never a real limit.
const UNLIMITED: u64 = 1 << 62;

/// The process's memory capacity, measured on first use.
pub fn process_memory_capacity() -> Option<u64> {
    static CAPACITY: OnceLock<Option<u64>> = OnceLock::new();
    *CAPACITY
        .get_or_init(|| capacity_from(cgroup_limit(), address_space_limit(), physical_memory()))
}

/// The capacity rule over the measured limits: the cgroup limit, else a finite address-space
/// limit, else physical memory.
pub(crate) fn capacity_from(
    cgroup: Option<u64>,
    address_space: Option<u64>,
    physical: Option<u64>,
) -> Option<u64> {
    let set = |limit: Option<u64>| limit.filter(|&bytes| bytes > 0 && bytes < UNLIMITED);
    set(cgroup).or(set(address_space)).or(set(physical))
}

#[cfg(target_os = "linux")]
fn cgroup_limit() -> Option<u64> {
    // cgroup v2, then v1; "max" (v2) and the v1 sentinel are no limit.
    [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ]
    .iter()
    .find_map(|path| {
        std::fs::read_to_string(path)
            .ok()?
            .trim()
            .parse::<u64>()
            .ok()
    })
}

#[cfg(not(target_os = "linux"))]
fn cgroup_limit() -> Option<u64> {
    None
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn address_space_limit() -> Option<u64> {
    #[repr(C)]
    struct RLimit {
        current: u64,
        maximum: u64,
    }
    #[cfg(target_os = "linux")]
    const RLIMIT_AS: std::ffi::c_int = 9;
    #[cfg(target_os = "macos")]
    const RLIMIT_AS: std::ffi::c_int = 5;
    unsafe extern "C" {
        fn getrlimit(resource: std::ffi::c_int, limit: *mut RLimit) -> std::ffi::c_int;
    }
    let mut limit = RLimit {
        current: 0,
        maximum: 0,
    };
    // SAFETY: getrlimit writes one `struct rlimit` (two 64-bit `rlim_t` on these targets).
    (unsafe { getrlimit(RLIMIT_AS, &mut limit) } == 0).then_some(limit.current)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn address_space_limit() -> Option<u64> {
    None
}

#[cfg(target_os = "linux")]
fn physical_memory() -> Option<u64> {
    let info = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = info.lines().find(|l| l.starts_with("MemTotal:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    kib.checked_mul(1024)
}

#[cfg(target_os = "macos")]
fn physical_memory() -> Option<u64> {
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const std::ffi::c_char,
            value: *mut std::ffi::c_void,
            size: *mut usize,
            new_value: *mut std::ffi::c_void,
            new_size: usize,
        ) -> std::ffi::c_int;
    }
    let mut bytes: u64 = 0;
    let mut size = std::mem::size_of::<u64>();
    // SAFETY: hw.memsize is one 64-bit integer; `size` names the buffer's length.
    let status = unsafe {
        sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&mut bytes as *mut u64).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (status == 0).then_some(bytes)
}

#[cfg(windows)]
fn physical_memory() -> Option<u64> {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_physical: u64,
        available_physical: u64,
        total_page_file: u64,
        available_page_file: u64,
        total_virtual: u64,
        available_virtual: u64,
        available_extended_virtual: u64,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
    }
    let mut status = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        memory_load: 0,
        total_physical: 0,
        available_physical: 0,
        total_page_file: 0,
        available_page_file: 0,
        total_virtual: 0,
        available_virtual: 0,
        available_extended_virtual: 0,
    };
    // SAFETY: `length` is set to the struct's size, as GlobalMemoryStatusEx requires.
    (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status.total_physical)
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
fn physical_memory() -> Option<u64> {
    None
}
