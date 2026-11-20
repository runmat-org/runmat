//! Live heap accounting. Linear memory only grows, so its length is a high-water
//! mark; the allocator reuses what was freed. Free heap is the instance budget
//! less the bytes this counter holds live.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Bytes the program holds allocated now (requested sizes, without allocator
/// headers or fragmentation).
pub(crate) fn live_heap_bytes() -> u64 {
    LIVE_BYTES.load(Ordering::Relaxed) as u64
}

pub(crate) struct CountingAllocator;

// SAFETY: every call forwards to the system allocator unchanged; the counter only
// records the sizes of successful allocations and releases.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = System.alloc(layout);
        if !pointer.is_null() {
            LIVE_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = System.alloc_zeroed(layout);
        if !pointer.is_null() {
            LIVE_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        System.dealloc(pointer, layout);
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let moved = System.realloc(pointer, layout, new_size);
        if !moved.is_null() {
            if new_size >= layout.size() {
                LIVE_BYTES.fetch_add(new_size - layout.size(), Ordering::Relaxed);
            } else {
                LIVE_BYTES.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        moved
    }
}

#[cfg(target_arch = "wasm32")]
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_follows_allocation_growth_shrink_and_release() {
        let layout = Layout::from_size_align(64, 8).unwrap();
        let start = live_heap_bytes();
        unsafe {
            let pointer = CountingAllocator.alloc(layout);
            assert!(!pointer.is_null());
            assert_eq!(live_heap_bytes() - start, 64);
            let grown = CountingAllocator.realloc(pointer, layout, 256);
            assert_eq!(live_heap_bytes() - start, 256);
            let grown_layout = Layout::from_size_align(256, 8).unwrap();
            let shrunk = CountingAllocator.realloc(grown, grown_layout, 16);
            assert_eq!(live_heap_bytes() - start, 16);
            CountingAllocator.dealloc(shrunk, Layout::from_size_align(16, 8).unwrap());
        }
        assert_eq!(live_heap_bytes(), start);
    }
}
