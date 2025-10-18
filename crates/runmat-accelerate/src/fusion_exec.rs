use anyhow::{anyhow, Result};

use crate::fusion::FusionGroupPlan;
use crate::fusion_residency;
use crate::graph::ShapeInfo;
use runmat_accelerate_api::{provider, GpuTensorHandle, HostTensorView, ProviderPrecision};
use runmat_builtins::Value;

struct PreparedInput {
    handle: GpuTensorHandle,
    owned: Option<GpuTensorHandle>,
}

pub struct FusionExecutionRequest<'a> {
    pub plan: &'a FusionGroupPlan,
    pub inputs: Vec<Value>,
}

pub fn execute_elementwise(request: FusionExecutionRequest<'_>) -> Result<Value> {
    crate::ensure_residency_hooks();
    if !request.plan.group.kind.is_elementwise() {
        return Err(anyhow!("unsupported fusion kind"));
    }
    let provider = provider().ok_or_else(|| anyhow!("no acceleration provider registered"))?;
    if !request.plan.kernel.supported {
        return Err(anyhow!("fusion kernel not supported for this plan"));
    }
    if request.inputs.len() != request.plan.inputs.len() {
        return Err(anyhow!(
            "fusion input mismatch: expected {}, got {}",
            request.plan.inputs.len(),
            request.inputs.len()
        ));
    }
    let len = request
        .plan
        .element_count()
        .ok_or_else(|| anyhow!("fusion: unknown element count"))?;
    if len == 0 {
        return Err(anyhow!("fusion: zero-length execution not supported"));
    }
    let constant_shape = request.plan.constant_shape(len);
    let mut prepared = Vec::with_capacity(request.inputs.len());
    let mut temp_scalars: Vec<Vec<f64>> = Vec::new();
    for value in &request.inputs {
        match value {
            Value::GpuTensor(handle) => prepared.push(PreparedInput {
                handle: handle.clone(),
                owned: None,
            }),
            Value::Tensor(t) => {
                let view = HostTensorView {
                    data: &t.data,
                    shape: &t.shape,
                };
                let handle = provider.upload(&view)?;
                prepared.push(PreparedInput {
                    handle: handle.clone(),
                    owned: Some(handle),
                });
            }
            Value::Num(n) => {
                temp_scalars.push(vec![*n; len]);
                let data = temp_scalars.last().unwrap();
                let view = HostTensorView {
                    data,
                    shape: &constant_shape,
                };
                let handle = provider.upload(&view)?;
                prepared.push(PreparedInput {
                    handle: handle.clone(),
                    owned: Some(handle),
                });
            }
            Value::Int(i) => {
                temp_scalars.push(vec![i.to_f64(); len]);
                let data = temp_scalars.last().unwrap();
                let view = HostTensorView {
                    data,
                    shape: &constant_shape,
                };
                let handle = provider.upload(&view)?;
                prepared.push(PreparedInput {
                    handle: handle.clone(),
                    owned: Some(handle),
                });
            }
            _ => {
                return Err(anyhow!("fusion: unsupported value type"));
            }
        }
    }

    let scalar_ty = match provider.precision() {
        ProviderPrecision::F32 => "f32",
        ProviderPrecision::F64 => "f64",
    };
    let shader = request
        .plan
        .generate_wgsl(scalar_ty)
        .ok_or_else(|| anyhow!("fusion: WGSL generation failed"))?;

    let handles: Vec<GpuTensorHandle> = prepared.iter().map(|p| p.handle.clone()).collect();
    let output_shape = infer_output_shape(&request.plan.group, len, &handles)?;
    let output = provider.fused_elementwise(&shader, &handles, &output_shape, len)?;
    fusion_residency::mark(&output);

    // Clean up temporary uploads
    for input in prepared {
        if let Some(handle) = input.owned {
            let _ = provider.free(&handle);
        }
    }

    Ok(Value::GpuTensor(output))
}

pub fn execute_reduction(
    request: FusionExecutionRequest<'_>,
    reduce_len: usize,
    num_slices: usize,
    workgroup_size: u32,
) -> Result<Value> {
    crate::ensure_residency_hooks();
    if !request.plan.group.kind.is_reduction() {
        return Err(anyhow!("unsupported fusion kind"));
    }
    let provider = provider().ok_or_else(|| anyhow!("no acceleration provider registered"))?;
    if !request.plan.kernel.supported {
        return Err(anyhow!("fusion kernel not supported for this plan"));
    }
    if request.inputs.len() != request.plan.inputs.len() {
        return Err(anyhow!(
            "fusion input mismatch: expected {}, got {}",
            request.plan.inputs.len(),
            request.inputs.len()
        ));
    }
    let len = reduce_len * num_slices;
    if len == 0 {
        return Err(anyhow!("fusion: zero-length execution not supported"));
    }
    let constant_shape = request.plan.constant_shape(len);
    let mut prepared = Vec::with_capacity(request.inputs.len());
    let mut temp_scalars: Vec<Vec<f64>> = Vec::new();
    for value in &request.inputs {
        match value {
            Value::GpuTensor(handle) => prepared.push(PreparedInput {
                handle: handle.clone(),
                owned: None,
            }),
            Value::Tensor(t) => {
                let view = HostTensorView {
                    data: &t.data,
                    shape: &t.shape,
                };
                let handle = provider.upload(&view)?;
                prepared.push(PreparedInput {
                    handle: handle.clone(),
                    owned: Some(handle),
                });
            }
            Value::Num(n) => {
                temp_scalars.push(vec![*n; len]);
                let data = temp_scalars.last().unwrap();
                let view = HostTensorView {
                    data,
                    shape: &constant_shape,
                };
                let handle = provider.upload(&view)?;
                prepared.push(PreparedInput {
                    handle: handle.clone(),
                    owned: Some(handle),
                });
            }
            Value::Int(i) => {
                temp_scalars.push(vec![i.to_f64(); len]);
                let data = temp_scalars.last().unwrap();
                let view = HostTensorView {
                    data,
                    shape: &constant_shape,
                };
                let handle = provider.upload(&view)?;
                prepared.push(PreparedInput {
                    handle: handle.clone(),
                    owned: Some(handle),
                });
            }
            _ => return Err(anyhow!("fusion: unsupported value type")),
        }
    }

    let handles: Vec<GpuTensorHandle> = prepared.iter().map(|p| p.handle.clone()).collect();
    let output_shape = vec![num_slices];

    let scalar_ty = match provider.precision() {
        ProviderPrecision::F32 => "f32",
        ProviderPrecision::F64 => "f64",
    };
    let shader = request
        .plan
        .generate_reduction_wgsl(scalar_ty)
        .ok_or_else(|| anyhow!("fusion: reduction WGSL generation failed"))?;

    let output = provider.fused_reduction(
        &shader,
        &handles,
        &output_shape,
        reduce_len,
        num_slices,
        workgroup_size,
    )?;
    fusion_residency::mark(&output);

    for input in prepared {
        if let Some(handle) = input.owned {
            let _ = provider.free(&handle);
        }
    }

    Ok(Value::GpuTensor(output))
}

fn infer_output_shape(
    group: &crate::fusion::FusionGroup,
    len: usize,
    inputs: &[GpuTensorHandle],
) -> Result<Vec<usize>> {
    if let ShapeInfo::Tensor(dims) = &group.shape {
        if !dims.is_empty() {
            let mut resolved = Vec::with_capacity(dims.len());
            for (idx, dim) in dims.iter().enumerate() {
                if let Some(v) = dim {
                    resolved.push(*v);
                } else if let Some(first) = inputs.first() {
                    resolved.push(*first.shape.get(idx).unwrap_or(&1));
                } else {
                    resolved.push(1);
                }
            }
            return Ok(resolved);
        }
    }
    // Fallback: preserve the shape of the first non-empty input (avoids scalars/constants)
    for handle in inputs {
        if !handle.shape.is_empty() {
            return Ok(handle.shape.clone());
        }
    }
    Ok(vec![len, 1])
}
