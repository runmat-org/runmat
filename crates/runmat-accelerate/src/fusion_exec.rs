use anyhow::{anyhow, Result};

use crate::fusion::{FusionGroupPlan, FusionKind, FusionPattern};
use crate::fusion_residency;
use crate::graph;
use crate::graph::{ShapeInfo, ValueId};
use runmat_accelerate_api::{
    provider, AccelProvider, CovRows, CovarianceOptions, GpuTensorHandle, HostTensorView,
    PowerStepEpilogue, ProviderPrecision,
};
use runmat_builtins::Value;

struct PreparedInput {
    handle: GpuTensorHandle,
    owned: Option<GpuTensorHandle>,
}

pub struct FusionExecutionRequest<'a> {
    pub plan: &'a FusionGroupPlan,
    pub inputs: Vec<Value>,
}

fn ensure_gpu_tensor(
    provider: &dyn AccelProvider,
    value: &Value,
) -> Result<(GpuTensorHandle, Option<GpuTensorHandle>)> {
    match value {
        Value::GpuTensor(handle) => Ok((handle.clone(), None)),
        Value::Tensor(tensor) => {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view)?;
            Ok((handle.clone(), Some(handle)))
        }
        _ => Err(anyhow!("fusion: expected tensor input")),
    }
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

pub fn execute_centered_gram(request: FusionExecutionRequest<'_>) -> Result<Value> {
    crate::ensure_residency_hooks();
    if request.plan.group.kind != FusionKind::CenteredGram {
        return Err(anyhow!("unsupported fusion kind"));
    }
    let provider = provider().ok_or_else(|| anyhow!("no acceleration provider registered"))?;
    let (matrix_vid, normalization) = match request.plan.pattern.as_ref() {
        Some(FusionPattern::CenteredGram {
            matrix,
            normalization,
        }) => (*matrix, *normalization),
        _ => return Err(anyhow!("centered gram: missing pattern metadata")),
    };

    let matrix_index = request
        .plan
        .inputs
        .iter()
        .position(|vid| *vid == matrix_vid)
        .ok_or_else(|| anyhow!("centered gram: matrix input not found"))?;
    let matrix_value = request
        .inputs
        .get(matrix_index)
        .ok_or_else(|| anyhow!("centered gram: matrix value missing"))?;

    let (matrix_handle, owned_matrix) = ensure_gpu_tensor(provider, matrix_value)?;

    let mut options = CovarianceOptions::default();
    options.normalization = normalization;
    options.rows = CovRows::All;
    options.has_weight_vector = false;

    let output = provider.covariance(&matrix_handle, None, None, &options)?;

    if let Some(temp) = owned_matrix {
        let _ = provider.free(&temp);
    }

    fusion_residency::mark(&output);
    Ok(Value::GpuTensor(output))
}

pub fn execute_power_step_normalize(request: FusionExecutionRequest<'_>) -> Result<Value> {
    crate::ensure_residency_hooks();
    if request.plan.group.kind != FusionKind::PowerStepNormalize {
        return Err(anyhow!("unsupported fusion kind"));
    }
    let provider = provider().ok_or_else(|| anyhow!("no acceleration provider registered"))?;
    let (lhs_vid, rhs_vid, epsilon) = match request.plan.pattern.as_ref() {
        Some(FusionPattern::PowerStepNormalize { lhs, rhs, epsilon }) => (*lhs, *rhs, *epsilon),
        _ => {
            return Err(anyhow!(
                "power-step normalization: missing pattern metadata"
            ))
        }
    };

    let lhs_index = request
        .plan
        .inputs
        .iter()
        .position(|vid| *vid == lhs_vid)
        .ok_or_else(|| anyhow!("power-step normalization: lhs input not found"))?;
    let rhs_index = request
        .plan
        .inputs
        .iter()
        .position(|vid| *vid == rhs_vid)
        .ok_or_else(|| anyhow!("power-step normalization: rhs input not found"))?;

    let lhs_value = request
        .inputs
        .get(lhs_index)
        .ok_or_else(|| anyhow!("power-step normalization: lhs value missing"))?;
    let rhs_value = request
        .inputs
        .get(rhs_index)
        .ok_or_else(|| anyhow!("power-step normalization: rhs value missing"))?;

    let (lhs_handle, lhs_owned) = ensure_gpu_tensor(provider, lhs_value)?;
    let (rhs_handle, rhs_owned) = ensure_gpu_tensor(provider, rhs_value)?;

    let desc = PowerStepEpilogue { epsilon };
    let output = provider.matmul_power_step(&lhs_handle, &rhs_handle, &desc)?;

    if let Some(temp) = lhs_owned {
        let _ = provider.free(&temp);
    }
    if let Some(temp) = rhs_owned {
        let _ = provider.free(&temp);
    }

    fusion_residency::mark(&output);
    Ok(Value::GpuTensor(output))
}

pub fn execute_explained_variance(request: FusionExecutionRequest<'_>) -> Result<Value> {
    crate::ensure_residency_hooks();
    if request.plan.group.kind != FusionKind::ExplainedVariance {
        return Err(anyhow!("unsupported fusion kind"));
    }
    let provider = provider().ok_or_else(|| anyhow!("no acceleration provider registered"))?;
    let (q_vid, g_vid) = match request.plan.pattern.as_ref() {
        Some(FusionPattern::ExplainedVariance { q, g }) => (*q, *g),
        _ => return Err(anyhow!("explained variance: missing pattern metadata")),
    };

    let find_value = |vid: ValueId| -> Result<&Value> {
        if let Some(pos) = request.plan.inputs.iter().position(|id| *id == vid) {
            request
                .inputs
                .get(pos)
                .ok_or_else(|| anyhow!("explained variance: missing runtime value"))
        } else {
            request
                .plan
                .const_values
                .get(&vid)
                .ok_or_else(|| anyhow!("explained variance: value not materialized"))
        }
    };

    let q_value = find_value(q_vid)?;
    let g_value = find_value(g_vid)?;

    let (mut q_handle, q_owned) = ensure_gpu_tensor(provider, q_value)?;
    let (g_handle, g_owned) = ensure_gpu_tensor(provider, g_value)?;

    let q_shape = q_handle.shape.clone();
    if q_shape.len() < 2 {
        return Err(anyhow!("explained variance: Q must be 2-D"));
    }
    let q_rows = q_shape[0];
    let q_cols = q_shape[1];
    if q_rows == 0 || q_cols == 0 {
        return Err(anyhow!("explained variance: zero-sized Q"));
    }

    let g_shape = g_handle.shape.clone();
    if g_shape.len() < 2 {
        return Err(anyhow!("explained variance: G must be 2-D"));
    }
    if g_shape[0] != q_rows || g_shape[1] != q_rows {
        return Err(anyhow!(
            "explained variance: expected G shape [{} x {}], got {:?}",
            q_rows,
            q_rows,
            g_shape
        ));
    }

    // Interpreter's transpose retains the original data layout. Mimic that by
    // reshaping rather than launching a real transpose so downstream matmul
    // observes the same misoriented layout.
    let mut transposed_shape = q_shape.clone();
    transposed_shape.swap(0, 1);
    let q_transposed_view = provider.reshape(&q_handle, &transposed_shape)?;

    let tmp = provider.matmul(&q_transposed_view, &g_handle)?;

    // Restore Q's original shape before the second multiplication.
    q_handle = provider.reshape(&q_handle, &q_shape)?;

    let product = provider.matmul(&tmp, &q_handle)?;

    let diag = provider.diag_extract(&product, 0)?;
    let diag = match diag.shape.as_slice() {
        [len] => provider.reshape(&diag, &[*len, 1])?,
        [_len, 1] => diag,
        _ => diag,
    };

    if std::env::var("RUNMAT_DEBUG_EXPLAINED").is_ok() {
        if let Ok(host) = provider.download(&tmp) {
            println!("tmp runtime shape {:?} data {:?}", host.shape, host.data);
        }
        if let Ok(host) = provider.download(&product) {
            println!("prod runtime shape {:?} data {:?}", host.shape, host.data);
        }
        if let Ok(host) = provider.download(&diag) {
            println!("diag runtime shape {:?} data {:?}", host.shape, host.data);
        }
    }

    let _ = provider.free(&tmp);
    let _ = provider.free(&product);

    if let Some(temp) = q_owned {
        let _ = provider.free(&temp);
    }
    if let Some(temp) = g_owned {
        let _ = provider.free(&temp);
    }

    fusion_residency::mark(&diag);
    Ok(Value::GpuTensor(diag))
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
    let mut clamp_min: Option<f64> = None;
    let mut clamp_max: Option<f64> = None;
    let mut pow_exponent: Option<f64> = None;

    for op in &request.plan.operations {
        match op {
            crate::fusion::FusionOp::Primitive { op, inputs, output } => {
        if let Some(out) = output {
                    if !inputs.contains(&cur) {
                        continue;
                    }
            let other = if inputs[0] == cur { inputs[1] } else { inputs[0] };
            let const_opt = request.plan.const_values.get(&other).cloned();
                    match op {
                crate::graph::PrimitiveOp::Mul | crate::graph::PrimitiveOp::ElemMul => {
                    if let Some(Value::Num(s)) = const_opt {
                        alpha *= s;
                    } else if let Some(Value::Int(i)) = const_opt {
                        alpha *= i.to_f64();
                    } else if row_scale.is_none() || col_scale.is_none() {
                        if let Some(h) = find_handle(other) {
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
                                if s != 0.0 {
                                    alpha *= 1.0 / s;
                                }
                    } else if let Some(Value::Int(i)) = const_opt {
                                let s = i.to_f64();
                                if s != 0.0 {
                                    alpha *= 1.0 / s;
                                }
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
                            if let Some(Value::Num(s)) = const_opt {
                                beta += s;
                            } else if let Some(Value::Int(i)) = const_opt {
                                beta += i.to_f64();
                            }
                }
                crate::graph::PrimitiveOp::Sub => {
                            if let Some(Value::Num(s)) = const_opt {
                                beta -= s;
                            } else if let Some(Value::Int(i)) = const_opt {
                                beta -= i.to_f64();
                            }
                        }
                        crate::graph::PrimitiveOp::Pow | crate::graph::PrimitiveOp::ElemPow => {
                            if inputs
                                .iter()
                                .position(|v| *v == cur)
                                .map(|idx| idx == 0)
                                .unwrap_or(false)
                            {
                                if let Some(exp) = const_opt {
                                    let exponent = match exp {
                                        Value::Num(v) => v,
                                        Value::Int(i) => i.to_f64(),
                                        _ => {
                                            cur = *out;
                                            continue;
                                        }
                                    };
                                    pow_exponent = Some(match pow_exponent {
                                        Some(existing) => existing * exponent,
                                        None => exponent,
                                    });
                                }
                            }
                }
                _ => {}
            }
            cur = *out;
                }
            }
            crate::fusion::FusionOp::Builtin { name, inputs, output } => {
                if let Some(out) = output {
                    if !inputs.contains(&cur) {
                        continue;
                    }
                    let lc = name.to_ascii_lowercase();
                    let other_vals: Vec<_> = inputs.iter().copied().filter(|v| *v != cur).collect();
                    if other_vals.len() == 1 {
                        if let Some(const_val) = request.plan.const_values.get(&other_vals[0]) {
                            let numeric = match const_val {
                                Value::Num(v) => Some(*v),
                                Value::Int(i) => Some(i.to_f64()),
                                _ => None,
                            };
                            if let Some(val) = numeric {
                                if lc == "max" {
                                    clamp_min = Some(clamp_min.map(|existing| existing.max(val)).unwrap_or(val));
                                } else if lc == "min" {
                                    clamp_max = Some(clamp_max.map(|existing| existing.min(val)).unwrap_or(val));
                                }
                            }
                        }
                    }
                    cur = *out;
                }
            }
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
    ep.clamp_min = clamp_min;
    ep.clamp_max = clamp_max;
    ep.pow_exponent = pow_exponent;

    let a = find_handle(a_vid).ok_or_else(|| anyhow!("missing A"))?;
    let b = find_handle(b_vid).ok_or_else(|| anyhow!("missing B"))?;
    let out = prov.matmul_epilogue(&a, &b, &ep)?;
    for h in owned { let _ = prov.free(&h); }
    fusion_residency::mark(&out);
    Ok(Value::GpuTensor(out))
}
