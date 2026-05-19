use runmat_builtins::Value;

#[cfg(feature = "native-accel")]
pub fn clear_value(value: &Value) {
    clear_handles_in_value(value);
}

#[cfg(not(feature = "native-accel"))]
pub fn clear_value(_value: &Value) {}

#[cfg(feature = "native-accel")]
fn clear_handles_in_value(value: &Value) {
    match value {
        Value::GpuTensor(handle) => runmat_accelerate::fusion_residency::clear(handle),
        Value::Cell(cell) => {
            for elem in &cell.data {
                clear_handles_in_value(elem);
            }
        }
        Value::Struct(struct_value) => {
            for elem in struct_value.fields.values() {
                clear_handles_in_value(elem);
            }
        }
        Value::Object(object_value) => {
            for elem in object_value.properties.values() {
                clear_handles_in_value(elem);
            }
        }
        Value::Closure(closure) => {
            for capture in &closure.captures {
                clear_handles_in_value(capture);
            }
        }
        Value::OutputList(values) => {
            for elem in values {
                clear_handles_in_value(elem);
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

pub fn same_gpu_handle(lhs: &Value, rhs: &Value) -> bool {
    matches!(
        (lhs, rhs),
        (Value::GpuTensor(left), Value::GpuTensor(right)) if left.buffer_id == right.buffer_id
    )
}

#[cfg(all(test, feature = "native-accel"))]
mod tests {
    use super::clear_value;
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
}
