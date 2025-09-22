//! Acceleration builtins: gpuArray, gather, gpuDevice
//!
//! These builtins provide explicit GPU array support using the runmat-accelerate-api
//! provider interface. If no provider is registered, calls will return an error.

use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

#[runtime_builtin(
    name = "gpuArray",
    category = "acceleration/gpu",
    summary = "Move data to GPU and return a gpuArray handle.",
    examples = "G = gpuArray(A)",
    keywords = "gpuArray,gpu,accelerate"
)]
fn gpu_array_builtin(x: Value) -> Result<Value, String> {
    let provider = runmat_accelerate_api::provider()
        .ok_or_else(|| "gpuArray: no acceleration provider registered".to_string())?;

    match x {
        Value::Tensor(t) => {
            let view = runmat_accelerate_api::HostTensorView {
                data: &t.data,
                shape: &t.shape,
            };
            let handle = provider.upload(&view).map_err(|e| e.to_string())?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Num(n) => {
            let t = Tensor::new_2d(vec![n], 1, 1).map_err(|e| format!("gpuArray: {e}"))?;
            let view = runmat_accelerate_api::HostTensorView {
                data: &t.data,
                shape: &t.shape,
            };
            let handle = provider.upload(&view).map_err(|e| e.to_string())?;
            Ok(Value::GpuTensor(handle))
        }
        Value::Int(i) => {
            let t = Tensor::new_2d(vec![i.to_f64()], 1, 1).map_err(|e| format!("gpuArray: {e}"))?;
            let view = runmat_accelerate_api::HostTensorView {
                data: &t.data,
                shape: &t.shape,
            };
            let handle = provider.upload(&view).map_err(|e| e.to_string())?;
            Ok(Value::GpuTensor(handle))
        }
        Value::LogicalArray(la) => {
            // Convert logical to numeric tensor (0/1) then upload
            let data: Vec<f64> = la.data.iter().map(|&b| if b != 0 { 1.0 } else { 0.0 }).collect();
            let t = Tensor::new(data, la.shape.clone()).map_err(|e| format!("gpuArray: {e}"))?;
            let view = runmat_accelerate_api::HostTensorView {
                data: &t.data,
                shape: &t.shape,
            };
            let handle = provider.upload(&view).map_err(|e| e.to_string())?;
            Ok(Value::GpuTensor(handle))
        }
        other => Err(format!(
            "gpuArray: unsupported input type for GPU transfer: {other:?}"
        )),
    }
}

#[runtime_builtin(
    name = "gather",
    category = "acceleration/gpu",
    summary = "Bring data from GPU back to host memory.",
    examples = "A = gather(G)",
    keywords = "gather,gpu,accelerate"
)]
fn gather_builtin(x: Value) -> Result<Value, String> {
    match x {
        Value::GpuTensor(h) => {
            let provider = runmat_accelerate_api::provider()
                .ok_or_else(|| "gather: no acceleration provider registered".to_string())?;
            let host = provider.download(&h).map_err(|e| e.to_string())?;
            let t = Tensor::new(host.data, host.shape).map_err(|e| format!("gather: {e}"))?;
            Ok(Value::Tensor(t))
        }
        // Pass-through for non-GPU values
        v => Ok(v),
    }
}

#[runtime_builtin(
    name = "gpuDevice",
    category = "acceleration/gpu",
    summary = "Return information about the active GPU device.",
    examples = "info = gpuDevice()",
    keywords = "gpu,device,info"
)]
fn gpu_device_builtin() -> Result<Value, String> {
    if let Some(p) = runmat_accelerate_api::provider() {
        // Prefer structured info when available
        let info = p.device_info_struct();
        let mut fields = std::collections::HashMap::new();
        fields.insert("device_id".to_string(), Value::Int(runmat_builtins::IntValue::U32(info.device_id)));
        fields.insert("name".to_string(), Value::String(info.name));
        fields.insert("vendor".to_string(), Value::String(info.vendor));
        if let Some(bytes) = info.memory_bytes {
            fields.insert("memory_bytes".to_string(), Value::Int(runmat_builtins::IntValue::U64(bytes)));
        }
        if let Some(backend) = info.backend {
            fields.insert("backend".to_string(), Value::String(backend));
        }
        Ok(Value::Struct(runmat_builtins::StructValue { fields }))
    } else {
        Ok(Value::String("no provider".to_string()))
    }
}

#[runtime_builtin(
    name = "gpuInfo",
    category = "acceleration/gpu",
    summary = "Pretty string describing the active GPU device.",
    examples = "disp(gpuInfo())",
    keywords = "gpu,device,info"
)]
fn gpu_info_builtin() -> Result<Value, String> {
    if let Some(p) = runmat_accelerate_api::provider() {
        let info = p.device_info_struct();
        let mut parts: Vec<String> = Vec::new();
        parts.push(format!("device_id={}", info.device_id));
        if !info.name.is_empty() {
            parts.push(format!("name='{}'", info.name));
        }
        if !info.vendor.is_empty() {
            parts.push(format!("vendor='{}'", info.vendor));
        }
        if let Some(bytes) = info.memory_bytes {
            parts.push(format!("memory_bytes={}", bytes));
        }
        if let Some(backend) = info.backend {
            parts.push(format!("backend='{}'", backend));
        }
        Ok(Value::String(format!("GPU[{}]", parts.join(", "))))
    } else {
        Ok(Value::String("GPU[no provider]".to_string()))
    }
}


