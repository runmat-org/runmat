use anyhow::{anyhow, Result};

use crate::fusion::FusionGroupPlan;
use crate::graph;
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
    // Determine output shape from the fusion plan and derive the element count from it.
    let output_shape = match &request.plan.group.shape {
        ShapeInfo::Tensor(dims) if !dims.is_empty() => {
            let resolved: Vec<usize> = dims.iter().map(|d| d.unwrap_or(1)).collect();
            resolved
        }
        _ => return Err(anyhow!("fusion: unknown output shape")),
    };
    let len: usize = output_shape.iter().copied().product();
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

    let wg = if workgroup_size == 0 {
        provider.default_reduction_workgroup_size()
    } else {
        workgroup_size
    };
    let output =
        provider.fused_reduction(&shader, &handles, &output_shape, reduce_len, num_slices, wg)?;
    fusion_residency::mark(&output);

    for input in prepared {
        if let Some(handle) = input.owned {
            let _ = provider.free(&handle);
        }
    }

    Ok(Value::GpuTensor(output))
}

pub fn execute_matmul_epilogue(request: FusionExecutionRequest<'_>) -> Result<Value> {
    crate::ensure_residency_hooks();
    if request.plan.group.kind != crate::fusion::FusionKind::MatmulEpilogue {
        return Err(anyhow!("unsupported fusion kind"));
    }
    let prov = provider().ok_or_else(|| anyhow!("no acceleration provider registered"))?;

    // Map ValueId -> prepared GpuTensorHandle
    let mut prepared: Vec<(graph::ValueId, GpuTensorHandle, Option<GpuTensorHandle>)> = Vec::new();
    let mut owned: Vec<GpuTensorHandle> = Vec::new();
    for (idx, vid) in request.plan.inputs.iter().copied().enumerate() {
        let v = request
            .inputs
            .get(idx)
            .ok_or_else(|| anyhow!("fusion: missing input value"))?;
        let handle = match v {
            Value::GpuTensor(h) => h.clone(),
            Value::Tensor(t) => {
                let view = HostTensorView { data: &t.data, shape: &t.shape };
                let h = prov.upload(&view)?;
                owned.push(h.clone());
                h
            }
            _ => return Err(anyhow!("matmul_epilogue: unsupported input value kind")),
        };
        prepared.push((vid, handle.clone(), None));
    }

    // Helper: find handle by ValueId
    let find_handle = |vid: graph::ValueId| -> Option<GpuTensorHandle> {
        prepared.iter().find_map(|(v, h, _)| if *v == vid { Some(h.clone()) } else { None })
    };

    // Find matmul op and its output
    let mut cur_out: Option<graph::ValueId> = None;
    let mut a_vid: Option<graph::ValueId> = None;
    let mut b_vid: Option<graph::ValueId> = None;
    for op in &request.plan.operations {
        if let crate::fusion::FusionOp::Builtin { name, inputs, output } = op {
            if name.eq_ignore_ascii_case("mtimes") {
                a_vid = inputs.get(0).copied();
                b_vid = inputs.get(1).copied();
                cur_out = *output;
                break;
            }
        }
    }
    let (a_vid, b_vid, mut cur) = (a_vid.ok_or_else(|| anyhow!("mtimes not found"))?, b_vid.ok_or_else(|| anyhow!("mtimes not found"))?, cur_out.ok_or_else(|| anyhow!("mtimes output missing"))?);

    // Derive epilogue (alpha, beta, row/col scale with mul/div flags) by walking subsequent ops that consume cur
    let mut alpha: f64 = 1.0;
    let mut beta: f64 = 0.0;
    let mut row_scale: Option<GpuTensorHandle> = None;
    let mut col_scale: Option<GpuTensorHandle> = None;
    let mut row_div = false;
    let mut col_div = false;

    for op in &request.plan.operations {
        let (op_kind, inputs, output) = match op {
            crate::fusion::FusionOp::Primitive { op, inputs, output } => (Some(*op), inputs, output),
            _ => continue,
        };
        if let Some(out) = output {
            if !inputs.contains(&cur) { continue; }
            // Determine the other input
            let other = if inputs[0] == cur { inputs[1] } else { inputs[0] };
            let const_opt = request.plan.const_values.get(&other).cloned();
            match op_kind.unwrap() {
                crate::graph::PrimitiveOp::Mul | crate::graph::PrimitiveOp::ElemMul => {
                    if let Some(Value::Num(s)) = const_opt {
                        alpha *= s;
                    } else if let Some(Value::Int(i)) = const_opt {
                        alpha *= i.to_f64();
                    } else if row_scale.is_none() || col_scale.is_none() {
                        // classify vector shape via handle shape
                        if let Some(h) = find_handle(other) {
                            // Heuristic: [m,1] => row_scale; [1,n] => col_scale
                            let r = h.shape.get(0).copied().unwrap_or(1);
                            let c = h.shape.get(1).copied().unwrap_or(1);
                            if c == 1 && row_scale.is_none() {
                                row_scale = Some(h);
                                row_div = false;
                            } else if r == 1 && col_scale.is_none() {
                                col_scale = Some(h);
                                col_div = false;
                            }
                        }
                    }
                }
                crate::graph::PrimitiveOp::Div | crate::graph::PrimitiveOp::ElemDiv => {
                    if let Some(Value::Num(s)) = const_opt {
                        if s != 0.0 { alpha *= 1.0 / s; }
                    } else if let Some(Value::Int(i)) = const_opt {
                        let s = i.to_f64(); if s != 0.0 { alpha *= 1.0 / s; }
                    } else if row_scale.is_none() || col_scale.is_none() {
                        if let Some(h) = find_handle(other) {
                            let r = h.shape.get(0).copied().unwrap_or(1);
                            let c = h.shape.get(1).copied().unwrap_or(1);
                            if c == 1 && row_scale.is_none() {
                                row_scale = Some(h);
                                row_div = true;
                            } else if r == 1 && col_scale.is_none() {
                                col_scale = Some(h);
                                col_div = true;
                            }
                        }
                    }
                }
                crate::graph::PrimitiveOp::Add => {
                    if let Some(Value::Num(s)) = const_opt { beta += s; }
                    else if let Some(Value::Int(i)) = const_opt { beta += i.to_f64(); }
                }
                crate::graph::PrimitiveOp::Sub => {
                    if let Some(Value::Num(s)) = const_opt { beta -= s; }
                    else if let Some(Value::Int(i)) = const_opt { beta -= i.to_f64(); }
                }
                _ => {}
            }
            cur = *out;
        }
    }

    // Build epilogue descriptor
    let mut ep = runmat_accelerate_api::MatmulEpilogue::noop();
    ep.alpha = alpha;
    ep.beta = beta;
    ep.row_op = if row_div { runmat_accelerate_api::ScaleOp::Divide } else { runmat_accelerate_api::ScaleOp::Multiply };
    ep.col_op = if col_div { runmat_accelerate_api::ScaleOp::Divide } else { runmat_accelerate_api::ScaleOp::Multiply };
    if let Some(h) = row_scale.clone() { ep.row_scale = Some(h); }
    if let Some(h) = col_scale.clone() { ep.col_scale = Some(h); }

    let a = find_handle(a_vid).ok_or_else(|| anyhow!("missing A"))?;
    let b = find_handle(b_vid).ok_or_else(|| anyhow!("missing B"))?;
    let out = prov.matmul_epilogue(&a, &b, &ep)?;
    for h in owned { let _ = prov.free(&h); }
    fusion_residency::mark(&out);
    Ok(Value::GpuTensor(out))
}
