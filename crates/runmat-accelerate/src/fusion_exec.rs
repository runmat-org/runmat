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
                temp_scalars.push(vec![*n]);
                let data = temp_scalars.last().unwrap();
                let shape = [1usize];
                let view = HostTensorView {
                    data,
                    shape: &shape,
                };
                let handle = provider.upload(&view)?;
                prepared.push(PreparedInput {
                    handle: handle.clone(),
                    owned: Some(handle),
                });
            }
            Value::Int(i) => {
                temp_scalars.push(vec![i.to_f64()]);
                let data = temp_scalars.last().unwrap();
                let shape = [1usize];
                let view = HostTensorView {
                    data,
                    shape: &shape,
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

    let len = request
        .plan
        .element_count()
        .ok_or_else(|| anyhow!("fusion: unknown element count"))?;
    if len == 0 {
        return Err(anyhow!("fusion: zero-length execution not supported"));
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
    Ok(vec![len, 1])
}
