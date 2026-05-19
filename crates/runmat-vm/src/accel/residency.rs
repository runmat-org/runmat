use runmat_builtins::Value;
#[cfg(feature = "native-accel")]
use std::collections::HashSet;

#[cfg(feature = "native-accel")]
pub fn clear_value(value: &Value) {
    clear_handles_in_value(value);
}

#[cfg(not(feature = "native-accel"))]
pub fn clear_value(_value: &Value) {}

#[cfg(feature = "native-accel")]
pub fn clear_value_excluding(current: &Value, incoming: &Value) {
    let mut keep_ids = HashSet::new();
    collect_gpu_buffer_ids(incoming, &mut keep_ids);
    clear_handles_in_value_excluding(current, &keep_ids);
}

#[cfg(feature = "native-accel")]
fn clear_handles_in_value(value: &Value) {
    clear_handles_in_value_excluding(value, &HashSet::new());
}

#[cfg(feature = "native-accel")]
fn clear_handles_in_value_excluding(value: &Value, keep_ids: &HashSet<u64>) {
    let mut visited_handle_targets = HashSet::new();
    clear_handles_in_value_excluding_with_visited(value, keep_ids, &mut visited_handle_targets);
}

#[cfg(feature = "native-accel")]
fn clear_handles_in_value_excluding_with_visited(
    value: &Value,
    keep_ids: &HashSet<u64>,
    visited_handle_targets: &mut HashSet<usize>,
) {
    match value {
        Value::GpuTensor(handle) => {
            if !keep_ids.contains(&handle.buffer_id) {
                if let Some(provider) = runmat_accelerate_api::provider_for_handle(handle) {
                    let _ = provider.free(handle);
                }
                runmat_accelerate::fusion_residency::clear(handle);
                runmat_accelerate_api::clear_handle_logical(handle);
                runmat_accelerate_api::clear_handle_storage(handle);
                runmat_accelerate_api::clear_handle_transpose(handle);
            }
        }
        Value::Cell(cell) => {
            for elem in &cell.data {
                clear_handles_in_value_excluding_with_visited(
                    elem,
                    keep_ids,
                    visited_handle_targets,
                );
            }
        }
        Value::Struct(struct_value) => {
            for elem in struct_value.fields.values() {
                clear_handles_in_value_excluding_with_visited(
                    elem,
                    keep_ids,
                    visited_handle_targets,
                );
            }
        }
        Value::Object(object_value) => {
            for elem in object_value.properties.values() {
                clear_handles_in_value_excluding_with_visited(
                    elem,
                    keep_ids,
                    visited_handle_targets,
                );
            }
        }
        Value::Closure(closure) => {
            for capture in &closure.captures {
                clear_handles_in_value_excluding_with_visited(
                    capture,
                    keep_ids,
                    visited_handle_targets,
                );
            }
        }
        Value::OutputList(values) => {
            for elem in values {
                clear_handles_in_value_excluding_with_visited(
                    elem,
                    keep_ids,
                    visited_handle_targets,
                );
            }
        }
        Value::HandleObject(handle) => {
            let raw_target = unsafe { handle.target.as_raw() } as usize;
            if visited_handle_targets.insert(raw_target) {
                let target = unsafe { &*handle.target.as_raw() };
                clear_handles_in_value_excluding_with_visited(
                    target,
                    keep_ids,
                    visited_handle_targets,
                );
            }
        }
        Value::Int(_)
        | Value::Num(_)
        | Value::Complex(_, _)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_)
        | Value::Tensor(_)
        | Value::ComplexTensor(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::SemanticFunctionHandle { .. }
        | Value::ClassRef(_)
        | Value::MException(_) => {}
    }
}

#[cfg(feature = "native-accel")]
fn collect_gpu_buffer_ids(value: &Value, output: &mut HashSet<u64>) {
    let mut visited_handle_targets = HashSet::new();
    collect_gpu_buffer_ids_with_visited(value, output, &mut visited_handle_targets);
}

#[cfg(feature = "native-accel")]
fn collect_gpu_buffer_ids_with_visited(
    value: &Value,
    output: &mut HashSet<u64>,
    visited_handle_targets: &mut HashSet<usize>,
) {
    match value {
        Value::GpuTensor(handle) => {
            output.insert(handle.buffer_id);
        }
        Value::Cell(cell) => {
            for elem in &cell.data {
                collect_gpu_buffer_ids_with_visited(elem, output, visited_handle_targets);
            }
        }
        Value::Struct(struct_value) => {
            for elem in struct_value.fields.values() {
                collect_gpu_buffer_ids_with_visited(elem, output, visited_handle_targets);
            }
        }
        Value::Object(object_value) => {
            for elem in object_value.properties.values() {
                collect_gpu_buffer_ids_with_visited(elem, output, visited_handle_targets);
            }
        }
        Value::Closure(closure) => {
            for capture in &closure.captures {
                collect_gpu_buffer_ids_with_visited(capture, output, visited_handle_targets);
            }
        }
        Value::OutputList(values) => {
            for elem in values {
                collect_gpu_buffer_ids_with_visited(elem, output, visited_handle_targets);
            }
        }
        Value::HandleObject(handle) => {
            let raw_target = unsafe { handle.target.as_raw() } as usize;
            if visited_handle_targets.insert(raw_target) {
                let target = unsafe { &*handle.target.as_raw() };
                collect_gpu_buffer_ids_with_visited(target, output, visited_handle_targets);
            }
        }
        Value::Int(_)
        | Value::Num(_)
        | Value::Complex(_, _)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_)
        | Value::Tensor(_)
        | Value::ComplexTensor(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::SemanticFunctionHandle { .. }
        | Value::ClassRef(_)
        | Value::MException(_) => {}
    }
}

#[cfg(all(test, feature = "native-accel"))]
mod tests {
    use super::{clear_value, clear_value_excluding};
    use futures::executor::block_on;
    use once_cell::sync::Lazy;
    use runmat_accelerate::fusion_residency;
    use runmat_accelerate::simple_provider::InProcessProvider;
    use runmat_accelerate_api::{
        AccelProvider, GpuTensorHandle, HostTensorView, ThreadProviderGuard,
    };
    use runmat_builtins::{CellArray, Closure, HandleRef, StructValue, Value};

    static TEST_PROVIDER: Lazy<InProcessProvider> = Lazy::new(InProcessProvider::new);

    fn upload_handle(data: Vec<f64>, shape: Vec<usize>) -> GpuTensorHandle {
        TEST_PROVIDER
            .upload(&HostTensorView {
                data: &data,
                shape: &shape,
            })
            .expect("upload should succeed")
    }

    #[test]
    fn clear_value_releases_nested_gpu_handles_in_cells() {
        let handle = GpuTensorHandle {
            shape: vec![1],
            device_id: 7,
            buffer_id: 7001,
        };
        fusion_residency::mark(&handle);
        assert!(fusion_residency::is_resident(&handle));

        let value = Value::Cell(
            CellArray::new(vec![Value::GpuTensor(handle.clone())], 1, 1).expect("cell"),
        );
        clear_value(&value);
        assert!(
            !fusion_residency::is_resident(&handle),
            "nested cell GPU handles should clear residency"
        );
    }

    #[test]
    fn clear_value_releases_nested_gpu_handles_in_closure_captures() {
        let handle = GpuTensorHandle {
            shape: vec![1],
            device_id: 8,
            buffer_id: 8001,
        };
        fusion_residency::mark(&handle);
        assert!(fusion_residency::is_resident(&handle));

        let value = Value::Closure(Closure {
            function_name: "worker".to_string(),
            semantic_function: None,
            captures: vec![Value::GpuTensor(handle.clone())],
        });
        clear_value(&value);
        assert!(
            !fusion_residency::is_resident(&handle),
            "closure-captured GPU handles should clear residency"
        );
    }

    #[test]
    fn clear_value_excluding_preserves_shared_handles() {
        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let shared = upload_handle(vec![1.0], vec![1]);
        let old_only = upload_handle(vec![2.0], vec![1]);
        fusion_residency::mark(&shared);
        fusion_residency::mark(&old_only);
        assert!(fusion_residency::is_resident(&shared));
        assert!(fusion_residency::is_resident(&old_only));
        assert!(block_on(TEST_PROVIDER.download(&shared)).is_ok());
        assert!(block_on(TEST_PROVIDER.download(&old_only)).is_ok());

        let current = Value::OutputList(vec![
            Value::GpuTensor(shared.clone()),
            Value::GpuTensor(old_only.clone()),
        ]);
        let incoming = Value::GpuTensor(shared.clone());
        clear_value_excluding(&current, &incoming);

        assert!(
            fusion_residency::is_resident(&shared),
            "shared handle should remain resident across overwrite"
        );
        assert!(
            !fusion_residency::is_resident(&old_only),
            "non-shared handle should clear residency across overwrite"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&shared)).is_ok(),
            "shared handle should remain available in provider storage"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&old_only)).is_err(),
            "dropped handle should be released from provider storage"
        );
        fusion_residency::clear(&shared);
        let _ = TEST_PROVIDER.free(&shared);
    }

    #[test]
    fn clear_value_releases_provider_storage_for_dropped_handle() {
        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_handle(vec![3.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);

        clear_value(&Value::GpuTensor(handle.clone()));

        assert!(
            !fusion_residency::is_resident(&handle),
            "cleared handle should no longer be marked resident"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "cleared handle should be released from provider storage"
        );
    }

    #[test]
    fn clear_value_releases_gpu_handles_nested_in_handle_object_target() {
        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let handle = upload_handle(vec![4.0], vec![1]);
        assert!(block_on(TEST_PROVIDER.download(&handle)).is_ok());
        fusion_residency::mark(&handle);
        let mut payload = StructValue::new();
        payload
            .fields
            .insert("nested".to_string(), Value::GpuTensor(handle.clone()));
        let gc_target =
            runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        let value = Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target: gc_target,
            valid: true,
        });

        clear_value(&value);

        assert!(
            !fusion_residency::is_resident(&handle),
            "nested handle-object payload should clear GPU residency"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&handle)).is_err(),
            "nested handle-object payload should release provider storage"
        );
    }

    #[test]
    fn clear_value_excluding_preserves_handles_referenced_in_handle_object_target() {
        let _provider_guard = ThreadProviderGuard::set(Some(&*TEST_PROVIDER));
        let shared = upload_handle(vec![5.0], vec![1]);
        let old_only = upload_handle(vec![6.0], vec![1]);
        fusion_residency::mark(&shared);
        fusion_residency::mark(&old_only);
        assert!(block_on(TEST_PROVIDER.download(&shared)).is_ok());
        assert!(block_on(TEST_PROVIDER.download(&old_only)).is_ok());

        let current = Value::OutputList(vec![
            Value::GpuTensor(shared.clone()),
            Value::GpuTensor(old_only.clone()),
        ]);
        let mut incoming_payload = StructValue::new();
        incoming_payload
            .fields
            .insert("nested".to_string(), Value::GpuTensor(shared.clone()));
        let incoming_gc =
            runmat_gc::gc_allocate(Value::Struct(incoming_payload)).expect("gc allocate incoming");
        let incoming = Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target: incoming_gc,
            valid: true,
        });

        clear_value_excluding(&current, &incoming);

        assert!(
            fusion_residency::is_resident(&shared),
            "shared handle in handle-object target should remain resident across overwrite"
        );
        assert!(
            !fusion_residency::is_resident(&old_only),
            "non-shared handle should clear residency across overwrite"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&shared)).is_ok(),
            "shared handle in handle-object target should remain available in provider storage"
        );
        assert!(
            block_on(TEST_PROVIDER.download(&old_only)).is_err(),
            "dropped handle should be released from provider storage"
        );
        fusion_residency::clear(&shared);
        let _ = TEST_PROVIDER.free(&shared);
    }
}
