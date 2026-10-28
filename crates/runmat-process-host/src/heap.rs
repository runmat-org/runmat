//! Return the parent's freed heap pages to the operating system before it launches a child.
//!
//! A host process often builds large transient structures (compiled models, plans, encoded task
//! inputs) and frees them just before it starts the worker that does the heavy work. The system
//! allocator keeps freed pages resident for reuse, so without an explicit release the idle
//! parent keeps holding gigabytes while its child runs, and the two together exceed a memory
//! ceiling that the child alone fits. Releasing at spawn hands that headroom to the child.

/// Release free pages retained by the process allocator. Live allocations are untouched.
pub fn release_retained_heap_pages() {
    #[cfg(target_os = "macos")]
    {
        extern "C" {
            // libmalloc: frees empty regions and cached large blocks in every zone (zone = NULL)
            // up to `goal` bytes; a goal of 0 releases as much as possible.
            fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize;
        }
        // SAFETY: a NULL zone selects all registered zones; the call only returns free pages.
        unsafe {
            malloc_zone_pressure_relief(std::ptr::null_mut(), 0);
        }
    }
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        // SAFETY: glibc trims free memory from every arena; live allocations are untouched.
        unsafe {
            libc::malloc_trim(0);
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::release_retained_heap_pages;

    fn physical_footprint_bytes() -> u64 {
        let mut info = std::mem::MaybeUninit::<libc::rusage_info_v2>::zeroed();
        // SAFETY: the buffer is a rusage_info_v2, matching the requested flavor.
        let status = unsafe {
            libc::proc_pid_rusage(
                std::process::id() as libc::c_int,
                libc::RUSAGE_INFO_V2,
                info.as_mut_ptr().cast(),
            )
        };
        assert_eq!(status, 0, "proc_pid_rusage failed");
        // SAFETY: proc_pid_rusage succeeded and filled the structure.
        unsafe { info.assume_init() }.ri_phys_footprint
    }

    #[test]
    fn freed_heap_pages_leave_the_parent_footprint_at_release() {
        const BLOCK: usize = 16 * 1024;
        const TOTAL: usize = 768 * 1024 * 1024;
        let mut blocks = Vec::with_capacity(TOTAL / BLOCK);
        for index in 0..TOTAL / BLOCK {
            blocks.push(vec![(index % 251) as u8 + 1; BLOCK]);
        }
        let held = physical_footprint_bytes();
        drop(blocks);
        let retained = physical_footprint_bytes();
        release_retained_heap_pages();
        let released = physical_footprint_bytes();
        assert!(
            retained.saturating_sub(released) >= (TOTAL / 2) as u64,
            "footprint held {held}, after free {retained}, after release {released}"
        );
    }
}
