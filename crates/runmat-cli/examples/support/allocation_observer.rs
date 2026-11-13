//! Process-local allocation accounting for the meshing verification executable.

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const REPORT_DIRECTORY_ENV: &str = "RUNMAT_MESH_ALLOCATION_REPORT_DIR";

struct CountingAllocator;

static ALLOCATION_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static REPORT_WRITTEN: AtomicBool = AtomicBool::new(false);

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(to_u64(layout.size()), Ordering::Relaxed);
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !new_pointer.is_null() {
            LIVE_BYTES.fetch_sub(to_u64(layout.size()), Ordering::Relaxed);
            record_allocation(new_size);
        }
        new_pointer
    }
}

fn record_allocation(size: usize) {
    let size = to_u64(size);
    ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
    ALLOCATED_BYTES.fetch_add(size, Ordering::Relaxed);
    let live = LIVE_BYTES
        .fetch_add(size, Ordering::Relaxed)
        .saturating_add(size);
    let mut peak = PEAK_LIVE_BYTES.load(Ordering::Relaxed);
    while live > peak {
        match PEAK_LIVE_BYTES.compare_exchange_weak(
            peak,
            live,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(observed) => peak = observed,
        }
    }
}

fn to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

pub struct ReportGuard {
    enabled: bool,
}

impl ReportGuard {
    pub fn from_environment() -> Self {
        Self {
            enabled: std::env::var_os(REPORT_DIRECTORY_ENV).is_some(),
        }
    }
}

impl Drop for ReportGuard {
    fn drop(&mut self) {
        if self.enabled {
            write_environment_report();
        }
    }
}

pub fn write_environment_report() {
    if REPORT_WRITTEN.swap(true, Ordering::AcqRel) {
        return;
    }
    let Some(directory) = std::env::var_os(REPORT_DIRECTORY_ENV).map(PathBuf::from) else {
        return;
    };
    let allocation_count = ALLOCATION_COUNT.load(Ordering::Relaxed);
    let allocated_bytes = ALLOCATED_BYTES.load(Ordering::Relaxed);
    let peak_live_bytes = PEAK_LIVE_BYTES.load(Ordering::Relaxed);
    let peak_rss_bytes = process_peak_rss_bytes();
    let process_id = std::process::id();
    if std::fs::create_dir_all(&directory).is_err() {
        return;
    }
    let record = format!(
        "{{\"schema_version\":1,\"process_id\":{process_id},\"allocation_count\":{allocation_count},\"allocated_bytes\":{allocated_bytes},\"peak_live_bytes\":{peak_live_bytes},\"peak_rss_bytes\":{peak_rss_bytes}}}\n"
    );
    let _ = std::fs::write(directory.join(format!("process-{process_id}.json")), record);
}

#[cfg(unix)]
fn process_peak_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `usage` points to writable storage for exactly one `rusage` value.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return 0;
    }
    // SAFETY: a successful `getrusage` initialized the complete value.
    let maximum = unsafe { usage.assume_init() }.ru_maxrss;
    let maximum = u64::try_from(maximum).unwrap_or(0);
    if cfg!(target_os = "macos") {
        maximum
    } else {
        maximum.saturating_mul(1024)
    }
}

#[cfg(not(unix))]
fn process_peak_rss_bytes() -> u64 {
    0
}
