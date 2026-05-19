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
    match value {
        Value::GpuTensor(handle) => {
            if !keep_ids.contains(&handle.buffer_id) {
                runmat_accelerate::fusion_residency::clear(handle);
            }
        }
        Value::Cell(cell) => {
            for elem in &cell.data {
                clear_handles_in_value_excluding(elem, keep_ids);
            }
        }
        Value::Struct(struct_value) => {
            for elem in struct_value.fields.values() {
                clear_handles_in_value_excluding(elem, keep_ids);
            }
        }
        Value::Object(object_value) => {
            for elem in object_value.properties.values() {
                clear_handles_in_value_excluding(elem, keep_ids);
            }
        }
        Value::Closure(closure) => {
            for capture in &closure.captures {
                clear_handles_in_value_excluding(capture, keep_ids);
            }
        }
        Value::OutputList(values) => {
            for elem in values {
                clear_handles_in_value_excluding(elem, keep_ids);
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
        | Value::HandleObject(_)
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
    match value {
        Value::GpuTensor(handle) => {
            output.insert(handle.buffer_id);
        }
        Value::Cell(cell) => {
            for elem in &cell.data {
                collect_gpu_buffer_ids(elem, output);
            }
        }
        Value::Struct(struct_value) => {
            for elem in struct_value.fields.values() {
                collect_gpu_buffer_ids(elem, output);
            }
        }
        Value::Object(object_value) => {
            for elem in object_value.properties.values() {
                collect_gpu_buffer_ids(elem, output);
            }
        }
        Value::Closure(closure) => {
            for capture in &closure.captures {
                collect_gpu_buffer_ids(capture, output);
            }
        }
        Value::OutputList(values) => {
            for elem in values {
                collect_gpu_buffer_ids(elem, output);
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
        | Value::HandleObject(_)
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
    use runmat_accelerate::fusion_residency;
    use runmat_accelerate_api::GpuTensorHandle;
    use runmat_builtins::{CellArray, Closure, Value};

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
        let shared = GpuTensorHandle {
            shape: vec![1],
            device_id: 9,
            buffer_id: 9001,
        };
        let old_only = GpuTensorHandle {
            shape: vec![1],
            device_id: 9,
            buffer_id: 9002,
        };
        fusion_residency::mark(&shared);
        fusion_residency::mark(&old_only);
        assert!(fusion_residency::is_resident(&shared));
        assert!(fusion_residency::is_resident(&old_only));

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
        fusion_residency::clear(&shared);
    }
}
