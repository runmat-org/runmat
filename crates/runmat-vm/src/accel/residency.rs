use runmat_builtins::Value;

#[cfg(feature = "native-accel")]
pub fn clear_value(value: &Value) {
    if let Value::GpuTensor(handle) = value {
        runmat_accelerate::fusion_residency::clear(handle);
    }
}

#[cfg(not(feature = "native-accel"))]
pub fn clear_value(_value: &Value) {}

pub fn same_gpu_handle(lhs: &Value, rhs: &Value) -> bool {
    matches!(
        (lhs, rhs),
        (Value::GpuTensor(left), Value::GpuTensor(right)) if left.buffer_id == right.buffer_id
    )
}

#[cfg(feature = "native-accel")]
pub fn mark(handle: &runmat_accelerate_api::GpuTensorHandle) {
    runmat_accelerate::fusion_residency::mark(handle);
}

#[cfg(not(feature = "native-accel"))]
pub fn mark(_handle: &runmat_accelerate_api::GpuTensorHandle) {}
