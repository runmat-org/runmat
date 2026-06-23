//! Miri-focused GC soundness tests.
//!
//! This crate intentionally depends only on `runmat-gc` and `runmat-builtins`
//! so it avoids the heavier VM/runtime dev-dependency graph. Run with:
//!
//! ```text
//! cargo miri test -p runmat-gc-miri-tests
//! ```

#[cfg(test)]
mod tests {
    use runmat_builtins::{
        CellArray, Closure, HandleRef, Listener, ObjectInstance, StructValue, Value,
    };
    use runmat_gc::*;
    use std::ptr::NonNull;

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
                runmat_gc::GcHandle::from_ptr_unchecked(
                    NonNull::new(raw.cast()).expect("box pointer should be non-null"),
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

            let target_handle =
                match &*gc_read_value(&rooted.handle()).expect("root should be live") {
                    Value::HandleObject(handle) => handle.target,
                    other => panic!("expected handle object, got {other:?}"),
                };

            assert_eq!(
                gc_clone_value(&target_handle)
                    .expect("target should remain live through root trace"),
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

            let target_handle =
                match &*gc_read_value(&rooted.handle()).expect("root should be live") {
                    Value::Cell(cell) => match &cell.data[0] {
                        Value::HandleObject(handle) => handle.target,
                        other => panic!("expected handle object in cell, got {other:?}"),
                    },
                    other => panic!("expected cell, got {other:?}"),
                };

            assert_eq!(
                gc_clone_value(&target_handle)
                    .expect("cell target should remain live through trace"),
                Value::Num(42.0)
            );
        });
    }

    #[test]
    fn root_traversal_scans_nested_trace_variants() {
        gc_test_context(|| {
            let target = gc_allocate(Value::String("target".to_string()))
                .expect("target allocation should succeed");
            let callback = gc_allocate(Value::String("callback".to_string()))
                .expect("callback allocation should succeed");

            let handle = Value::HandleObject(HandleRef {
                class_name: "Thing".to_string(),
                target,
                valid: true,
            });
            let listener = Value::Listener(Listener {
                id: 1,
                target,
                event_name: "Changed".to_string(),
                callback,
                enabled: true,
                valid: true,
            });
            let closure = Value::Closure(Closure {
                function_name: "callback".to_string(),
                bound_function: None,
                captures: vec![Value::OutputList(vec![handle, listener])],
            });

            let mut object = ObjectInstance::new("Owner".to_string());
            object.properties.insert("capture".to_string(), closure);

            let mut root_struct = StructValue::new();
            root_struct
                .fields
                .insert("object".to_string(), Value::Object(object));

            let rooted = gc_allocate_rooted(Value::Struct(root_struct))
                .expect("rooted nested value should allocate");

            gc_collect_minor().expect("minor collection should succeed");

            assert!(matches!(
                &*gc_read_value(&rooted.handle()).expect("root should remain live"),
                Value::Struct(_)
            ));
            assert_eq!(
                gc_clone_value(&target).expect("listener target should remain live"),
                Value::String("target".to_string())
            );
            assert_eq!(
                gc_clone_value(&callback).expect("listener callback should remain live"),
                Value::String("callback".to_string())
            );
        });
    }
}
