//! Success-path allocation check in its own test binary; custom serializers may still allocate.
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

use runmat_canonical_codec::{serialized_materialization_resident_bytes_estimate, CanonicalLimits};
use serde::Serialize;

thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocated() {
    let _ = TRACKING.try_with(|enabled| {
        if enabled.get() {
            let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
        }
    });
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        allocated();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        allocated();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        allocated();
        unsafe { System.realloc(pointer, layout, size) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[test]
fn borrowed_struct_and_numeric_arrays_do_not_allocate() {
    #[derive(Serialize)]
    struct Fixture {
        name: String,
        values: Vec<f64>,
        enabled: bool,
        optional: Option<String>,
    }
    let fixture = Fixture {
        name: "thermal-field".into(),
        values: vec![293.15; 10_000],
        enabled: true,
        optional: Some("present".into()),
    };
    ALLOCATIONS.with(|count| count.set(0));
    TRACKING.with(|enabled| enabled.set(true));
    let result = serialized_materialization_resident_bytes_estimate(
        &fixture,
        CanonicalLimits::new(1_000_000, 20_000, 4096, 64),
    );
    TRACKING.with(|enabled| enabled.set(false));
    let allocations = ALLOCATIONS.with(Cell::get);
    assert!(result.is_ok());
    assert_eq!(allocations, 0);
}
