//! Focused GC soundness tests.
//!
//! These tests intentionally avoid threads, I/O, and provider state so they can
//! also run under Miri:
//!
//! ```text
//! cargo miri test -p runmat-gc --test soundness
//! ```

use runmat_builtins::{CellArray, HandleRef, Value};
use runmat_gc::*;

#[test]
fn allocation_returns_value_aligned_handles() {
    gc_test_context(|| {
        let align = std::mem::align_of::<Value>();

        for i in 0..64 {
            let handle = gc_allocate(Value::Num(i as f64)).expect("allocation should succeed");
            assert_eq!(handle.addr() % align, 0, "GC handle is not Value-aligned");
        }
    });
}

#[test]
fn read_guard_rejects_unowned_address() {
    gc_test_context(|| {
        let boxed = Box::new(Value::Num(1.0));
        let raw = Box::into_raw(boxed);
        let handle = unsafe {
            runmat_gc::GcHandle::from_addr_unchecked(
                std::num::NonZeroUsize::new(raw as usize).expect("box pointer should be non-null"),
            )
        };

        assert!(matches!(
            gc_read_value(&handle),
            Err(GcError::InvalidPointer(_))
        ));

        unsafe {
            drop(Box::from_raw(raw));
        }
    });
}

#[test]
fn mutable_guard_is_exclusive_and_updates_value() {
    gc_test_context(|| {
        let handle = gc_allocate(Value::Num(1.0)).expect("allocation should succeed");

        {
            let mut guard = gc_write_value(&handle).expect("write guard should succeed");
            *guard = Value::Num(2.0);
            assert!(matches!(
                gc_write_value(&handle),
                Err(GcError::SyncError(_))
            ));
        }

        assert_eq!(
            gc_clone_value(&handle).expect("mutated handle should remain valid"),
            Value::Num(2.0)
        );
    });
}

#[test]
fn root_traversal_keeps_handle_targets_alive() {
    gc_test_context(|| {
        let target = gc_allocate(Value::String("target".to_string()))
            .expect("target allocation should succeed");
        let holder = Value::HandleObject(HandleRef {
            class_name: "Thing".to_string(),
            target,
            valid: true,
        });
        let rooted =
            gc_allocate_rooted(holder).expect("rooted handle-object allocation should succeed");

        gc_collect_minor().expect("minor collection should succeed");

        let target_handle = match &*gc_read_value(&rooted.handle()).expect("root should be live") {
            Value::HandleObject(handle) => handle.target,
            other => panic!("expected handle object, got {other:?}"),
        };

        assert_eq!(
            gc_clone_value(&target_handle).expect("target should remain live through root trace"),
            Value::String("target".to_string())
        );
    });
}

#[test]
fn root_traversal_scans_owned_cell_values() {
    gc_test_context(|| {
        let target =
            gc_allocate(Value::Num(42.0)).expect("nested target allocation should succeed");
        let cell = Value::Cell(
            CellArray::new(
                vec![Value::HandleObject(HandleRef {
                    class_name: "Thing".to_string(),
                    target,
                    valid: true,
                })],
                1,
                1,
            )
            .expect("cell construction should succeed"),
        );
        let rooted = gc_allocate_rooted(cell).expect("rooted cell allocation should succeed");

        gc_collect_minor().expect("minor collection should succeed");

        let target_handle = match &*gc_read_value(&rooted.handle()).expect("root should be live") {
            Value::Cell(cell) => match &cell.data[0] {
                Value::HandleObject(handle) => handle.target,
                other => panic!("expected handle object in cell, got {other:?}"),
            },
            other => panic!("expected cell, got {other:?}"),
        };

        assert_eq!(
            gc_clone_value(&target_handle).expect("cell target should remain live through trace"),
            Value::Num(42.0)
        );
    });
}
