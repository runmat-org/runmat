use super::*;

pub(super) async fn plus_gpu_pair(
    lhs: GpuTensorHandle,
    rhs: GpuTensorHandle,
) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if lhs.shape == rhs.shape {
            if let Ok(handle) = provider.elem_add(&lhs, &rhs).await {
                return Ok(gpu_helpers::resident_gpu_value(handle));
            }
        }
        // Attempt N-D broadcast via repmat to keep computation on device
        if let Some((out_shape, reps_l, reps_r)) = broadcast_reps(&lhs.shape, &rhs.shape) {
            let made_left = reps_l.iter().any(|&r| r != 1);
            let made_right = reps_r.iter().any(|&r| r != 1);
            let left_expanded = if made_left {
                provider
                    .repmat(&lhs, &reps_l)
                    .map_err(|e| builtin_error(format!("plus: {e}")))?
            } else {
                lhs.clone()
            };
            let right_expanded = if made_right {
                provider
                    .repmat(&rhs, &reps_r)
                    .map_err(|e| builtin_error(format!("plus: {e}")))?
            } else {
                rhs.clone()
            };
            let result = provider
                .elem_add(&left_expanded, &right_expanded)
                .await
                .map_err(|e| builtin_error(format!("plus: {e}")));
            if made_left {
                let _ = provider.free(&left_expanded);
            }
            if made_right {
                let _ = provider.free(&right_expanded);
            }
            if let Ok(handle) = result {
                if handle.shape == out_shape {
                    return Ok(gpu_helpers::resident_gpu_value(handle));
                } else {
                    let _ = provider.free(&handle);
                }
            }
        }
        if is_scalar_shape(&lhs.shape) {
            if let Some(scalar) = gpu_scalar_value(&lhs).await? {
                if let Ok(handle) = provider.scalar_add(&rhs, scalar) {
                    return Ok(gpu_helpers::resident_gpu_value(handle));
                }
            }
        }
        if is_scalar_shape(&rhs.shape) {
            if let Some(scalar) = gpu_scalar_value(&rhs).await? {
                if let Ok(handle) = provider.scalar_add(&lhs, scalar) {
                    return Ok(gpu_helpers::resident_gpu_value(handle));
                }
            }
        }
    }
    let left = gpu_helpers::gather_value_async(&Value::GpuTensor(lhs))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let right = gpu_helpers::gather_value_async(&Value::GpuTensor(rhs))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    plus_host(left, right)
}

fn broadcast_reps(a: &[usize], b: &[usize]) -> Option<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    let rank = a.len().max(b.len()).max(1);
    let mut out = vec![1usize; rank];
    let aa = crate::builtins::common::broadcast::align_shape(a, rank);
    let bb = crate::builtins::common::broadcast::align_shape(b, rank);
    for i in 0..rank {
        let (ad, bd) = (aa[i], bb[i]);
        if ad == bd {
            out[i] = ad;
        } else if ad == 1 {
            out[i] = bd;
        } else if bd == 1 {
            out[i] = ad;
        } else {
            return None;
        }
    }
    let reps_a: Vec<usize> = (0..rank)
        .map(|i| if aa[i] == out[i] { 1 } else { out[i] })
        .collect();
    let reps_b: Vec<usize> = (0..rank)
        .map(|i| if bb[i] == out[i] { 1 } else { out[i] })
        .collect();
    Some((out, reps_a, reps_b))
}

pub(super) async fn plus_gpu_host_left(lhs: GpuTensorHandle, rhs: Value) -> BuiltinResult<Value> {
    if is_real_integer_operand(&rhs) {
        let host_lhs = gpu_helpers::gather_value_async(&Value::GpuTensor(lhs.clone()))
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
        let host = plus_host(host_lhs, rhs)?;
        return gpu_helpers::restore_class_preserving_value(&lhs, host, BUILTIN_NAME);
    }
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Some(scalar) = extract_scalar_f64(&rhs)? {
            if let Some(uploaded) =
                gpu_helpers::upload_exact_integer_scalar_like(provider, &lhs, scalar)
            {
                let result = provider.elem_add(&lhs, &uploaded).await;
                let _ = provider.free(&uploaded);
                if let Ok(handle) = result {
                    return Ok(gpu_helpers::resident_gpu_value(handle));
                }
            }
            if let Ok(handle) = provider.scalar_add(&lhs, scalar) {
                return Ok(gpu_helpers::resident_gpu_value(handle));
            }
        }
    }
    let host_lhs = gpu_helpers::gather_value_async(&Value::GpuTensor(lhs))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    plus_host(host_lhs, rhs)
}

pub(super) async fn plus_gpu_host_right(lhs: Value, rhs: GpuTensorHandle) -> BuiltinResult<Value> {
    if is_real_integer_operand(&lhs) {
        let host_rhs = gpu_helpers::gather_value_async(&Value::GpuTensor(rhs.clone()))
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
        let host = plus_host(lhs, host_rhs)?;
        return gpu_helpers::restore_class_preserving_value(&rhs, host, BUILTIN_NAME);
    }
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Some(scalar) = extract_scalar_f64(&lhs)? {
            if let Some(uploaded) =
                gpu_helpers::upload_exact_integer_scalar_like(provider, &rhs, scalar)
            {
                let result = provider.elem_add(&uploaded, &rhs).await;
                let _ = provider.free(&uploaded);
                if let Ok(handle) = result {
                    return Ok(gpu_helpers::resident_gpu_value(handle));
                }
            }
            if let Ok(handle) = provider.scalar_add(&rhs, scalar) {
                return Ok(gpu_helpers::resident_gpu_value(handle));
            }
        }
    }
    let host_rhs = gpu_helpers::gather_value_async(&Value::GpuTensor(rhs))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    plus_host(lhs, host_rhs)
}
fn extract_scalar_f64(value: &Value) -> BuiltinResult<Option<f64>> {
    match value {
        Value::Num(n) => Ok(Some(*n)),
        Value::Bool(b) => Ok(Some(if *b { 1.0 } else { 0.0 })),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => Ok(Some(tensor::tensor_value_f64(t, 0))),
        Value::LogicalArray(l) if l.data.len() == 1 => {
            Ok(Some(if l.data[0] != 0 { 1.0 } else { 0.0 }))
        }
        Value::CharArray(ca) if ca.rows * ca.cols == 1 => Ok(Some(
            ca.data.first().map(|&ch| ch as u32 as f64).unwrap_or(0.0),
        )),
        _ => Ok(None),
    }
}

fn is_scalar_shape(shape: &[usize]) -> bool {
    shape.iter().copied().product::<usize>() <= 1
}

async fn gpu_scalar_value(handle: &GpuTensorHandle) -> BuiltinResult<Option<f64>> {
    if !is_scalar_shape(&handle.shape) {
        return Ok(None);
    }
    let tensor = gpu_helpers::gather_tensor_async(handle).await?;
    Ok(tensor::tensor_values_f64(&tensor).first().copied())
}
