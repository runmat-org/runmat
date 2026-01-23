use anyhow::{anyhow, Result};

use crate::fusion::{FusionGroupPlan, FusionKind, FusionPattern, ImageScalar};
use crate::fusion_residency;
use crate::graph;
use crate::graph::{ShapeInfo, ValueId};
use crate::precision::ensure_provider_supports_dtype;
use log;
use runmat_accelerate_api::{
    provider, AccelProvider, CovRows, CovarianceOptions, GpuTensorHandle, HostTensorView,
    ImageNormalizeDescriptor, PowerStepEpilogue, ProviderPrecision, ReductionFlavor,
};
use runmat_builtins::{NumericDType, Value};
use runmat_runtime::builtins::common::shape::normalize_scalar_shape;
use runmat_runtime::gather_if_needed;
use runmat_time::Instant;
use std::sync::OnceLock;

struct PreparedInput {
    handle: GpuTensorHandle,
    owned: Option<GpuTensorHandle>,
}

pub struct FusionExecutionRequest<'a> {
    pub plan: &'a FusionGroupPlan,
    pub inputs: Vec<Value>,
}

#[inline]
fn fusion_timing_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| match std::env::var("RUNMAT_FUSION_TIMING") {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    })
}

struct FusionStageTimer {
    inner: Option<FusionStageTimerInner>,
}

struct FusionStageTimerInner {
    plan_index: usize,
    kind: &'static str,
    len: usize,
    start: Instant,
    last: Instant,
    stages: Vec<(&'static str, f64)>,
}

impl FusionStageTimer {
    fn new(kind: &'static str, plan_index: usize, len: usize) -> Self {
        if fusion_timing_enabled() && log::log_enabled!(log::Level::Debug) {
            let now = Instant::now();
            Self {
                inner: Some(FusionStageTimerInner {
                    plan_index,
                    kind,
                    len,
                    start: now,
                    last: now,
                    stages: Vec::new(),
                }),
            }
        } else {
            Self { inner: None }
        }
    }

    fn mark(&mut self, label: &'static str) {
        if let Some(inner) = &mut self.inner {
            let now = Instant::now();
            let delta = now.duration_since(inner.last).as_secs_f64() * 1000.0;
            inner.stages.push((label, delta));
            inner.last = now;
        }
    }

    fn finish(self) {
        if let Some(inner) = self.inner {
            let total = inner.start.elapsed().as_secs_f64() * 1000.0;
            let summary = inner
                .stages
                .into_iter()
                .map(|(label, ms)| format!("{label}={ms:.3}ms"))
                .collect::<Vec<_>>()
                .join(" ");
            log::debug!(
                "fusion timing plan={} kind={} len={} {} total={:.3}ms",
                inner.plan_index,
                inner.kind,
                inner.len,
                summary,
                total
            );
        }
    }
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

fn scalar_upload_dtype(provider: &dyn AccelProvider) -> NumericDType {
    match provider.precision() {
        ProviderPrecision::F32 => NumericDType::F32,
        ProviderPrecision::F64 => NumericDType::F64,
    }
}

fn value_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Num(n) => Some(*n),
        Value::Int(i) => Some(i.to_f64()),
        _ => None,
    }
}

fn scalar_from_value(value: &Value) -> Result<f64> {
    if let Some(v) = value_to_f64(value) {
        return Ok(v);
    }
    match value {
        Value::Tensor(t) => {
            if t.data.len() == 1 {
                Ok(t.data[0])
            } else {
                Err(anyhow!(
                    "image normalize: expected scalar tensor, got {} elements",
                    t.data.len()
                ))
            }
        }
        Value::GpuTensor(_) => {
            let gathered = gather_if_needed(value).map_err(|e| anyhow!("image normalize: {e}"))?;
            scalar_from_value(&gathered)
        }
        _ => Err(anyhow!(
            "image normalize: expected numeric scalar value, got {:?}",
            value
        )),
    }
}

fn resolve_image_scalar_value(
    scalar: &ImageScalar,
    plan: &FusionGroupPlan,
    request: &FusionExecutionRequest<'_>,
) -> Result<f64> {
    match scalar {
        ImageScalar::Constant(v) => Ok(*v),
        ImageScalar::Value(vid) => {
            if let Some(value) = plan.const_values.get(vid) {
                return scalar_from_value(value);
            }
            if let Some(idx) = plan.inputs.iter().position(|id| *id == *vid) {
                let runtime_value = request
                    .inputs
                    .get(idx)
                    .ok_or_else(|| anyhow!("image normalize: runtime scalar missing"))?;
                return scalar_from_value(runtime_value);
            }
            Err(anyhow!(
                "image normalize: scalar input {:?} not materialized in plan",
                vid
            ))
        }
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
    // Determine output shape from the fusion plan; if unknown, derive from runtime inputs via broadcasting.
    fn runtime_broadcast_shape(values: &[Value]) -> Option<Vec<usize>> {
        // Collect shapes; scalars map to empty shape which broadcasts to any
        let mut shapes: Vec<Vec<usize>> = Vec::new();
        for v in values {
            match v {
                Value::GpuTensor(h) => shapes.push(h.shape.clone()),
                Value::Tensor(t) => shapes.push(t.shape.clone()),
                Value::Num(_) | Value::Int(_) => shapes.push(Vec::new()),
                _ => return None, // unsupported at runtime for broadcasting
            }
        }
        let rank = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
        let mut out = vec![1usize; rank];
        for shape in shapes {
            let offset = rank.saturating_sub(shape.len());
            for (i, &dim) in shape.iter().enumerate() {
                let j = offset + i;
                let a = out[j];
                let b = dim;
                if a == 1 {
                    out[j] = b.max(1);
                } else if b == 1 || a == b {
                    // keep a
                } else {
                    return None; // incompatible
                }
            }
        }
        Some(out)
    }
    // Determine output shape from the fusion plan and derive the element count from it.
    let mut output_shape = match &request.plan.group.shape {
        ShapeInfo::Tensor(dims) if !dims.is_empty() => {
            let resolved: Vec<usize> = dims.iter().map(|d| d.unwrap_or(1)).collect();
            resolved
        }
        _ => {
            // Fallback to runtime broadcasting inference
            runtime_broadcast_shape(&request.inputs)
                .ok_or_else(|| anyhow!("fusion: unknown output shape"))?
        }
    };
    let mut len: usize = output_shape.iter().copied().product();
    if len == 0 {
        if let Some(rt_shape) = runtime_broadcast_shape(&request.inputs) {
            output_shape = rt_shape;
            len = output_shape.iter().copied().product();
        }
        if len == 0 {
            return Err(anyhow!("fusion: zero-length execution not supported"));
        }
    }
    output_shape = normalize_scalar_shape(&output_shape);
    let mut timer = FusionStageTimer::new("elementwise", request.plan.index, len);
    let scalar_shape = normalize_scalar_shape(&vec![1; output_shape.len()]);
    let mut prepared = Vec::with_capacity(request.inputs.len());
    let mut temp_scalars: Vec<Vec<f64>> = Vec::new();
    let scalar_dtype = scalar_upload_dtype(provider);
    for value in &request.inputs {
        match value {
            Value::GpuTensor(handle) => prepared.push(PreparedInput {
                handle: handle.clone(),
                owned: None,
            }),
            Value::Tensor(t) => {
                if let Err(msg) = ensure_provider_supports_dtype(provider, t.dtype) {
                    return Err(anyhow!(
                        "fusion: tensor input requires unsupported precision ({msg})"
                    ));
                }
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
                if let Err(msg) = ensure_provider_supports_dtype(provider, scalar_dtype) {
                    return Err(anyhow!(
                        "fusion: scalar input requires unsupported precision ({msg})"
                    ));
                }
                let scalar = match provider.precision() {
                    ProviderPrecision::F32 => (*n as f32) as f64,
                    ProviderPrecision::F64 => *n,
                };
                temp_scalars.push(vec![scalar]);
                let data = temp_scalars.last().unwrap();
                let view = HostTensorView {
                    data,
                    shape: &scalar_shape,
                };
                let handle = provider.upload(&view)?;
                prepared.push(PreparedInput {
                    handle: handle.clone(),
                    owned: Some(handle),
                });
            }
            Value::Int(i) => {
                if let Err(msg) = ensure_provider_supports_dtype(provider, scalar_dtype) {
                    return Err(anyhow!(
                        "fusion: scalar input requires unsupported precision ({msg})"
                    ));
                }
                let scalar = match provider.precision() {
                    ProviderPrecision::F32 => (i.to_f64() as f32) as f64,
                    ProviderPrecision::F64 => i.to_f64(),
                };
                temp_scalars.push(vec![scalar]);
                let data = temp_scalars.last().unwrap();
                let view = HostTensorView {
                    data,
                    shape: &scalar_shape,
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
    timer.mark("prepare_inputs");

    let scalar_ty = match provider.precision() {
        ProviderPrecision::F32 => "f32",
        ProviderPrecision::F64 => "f64",
    };
    let shader = request
        .plan
        .generate_wgsl(scalar_ty)
        .ok_or_else(|| anyhow!("fusion: WGSL generation failed"))?;
    timer.mark("generate_wgsl");

    let handles: Vec<GpuTensorHandle> = prepared.iter().map(|p| p.handle.clone()).collect();
    let output = provider.fused_elementwise(&shader, &handles, &output_shape, len)?;
    timer.mark("dispatch");
    fusion_residency::mark(&output);

    // Clean up temporary uploads
    for input in prepared {
        if let Some(handle) = input.owned {
            let _ = provider.free(&handle);
        }
    }
    timer.mark("cleanup");
    timer.finish();

    Ok(Value::GpuTensor(output))
}

pub fn execute_reduction(
    request: FusionExecutionRequest<'_>,
    reduce_len: usize,
    num_slices: usize,
    workgroup_size: u32,
) -> Result<Value> {
    if std::env::var("RUNMAT_DISABLE_FUSED_REDUCTION").is_ok() {
        return Err(anyhow!("fused reduction disabled by env"));
    }
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
    let scalar_shape = {
        let constant_shape = request.plan.constant_shape(len);
        normalize_scalar_shape(&vec![1; constant_shape.len()])
    };
    let mut timer = FusionStageTimer::new("reduction", request.plan.index, len);
    let mut prepared = Vec::with_capacity(request.inputs.len());
    let mut temp_scalars: Vec<Vec<f64>> = Vec::new();
    let scalar_dtype = scalar_upload_dtype(provider);
    for value in &request.inputs {
        match value {
            Value::GpuTensor(handle) => prepared.push(PreparedInput {
                handle: handle.clone(),
                owned: None,
            }),
            Value::Tensor(t) => {
                if let Err(msg) = ensure_provider_supports_dtype(provider, t.dtype) {
                    return Err(anyhow!(
                        "fusion: tensor input requires unsupported precision ({msg})"
                    ));
                }
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
                if let Err(msg) = ensure_provider_supports_dtype(provider, scalar_dtype) {
                    return Err(anyhow!(
                        "fusion: scalar input requires unsupported precision ({msg})"
                    ));
                }
                let scalar = match provider.precision() {
                    ProviderPrecision::F32 => (*n as f32) as f64,
                    ProviderPrecision::F64 => *n,
                };
                temp_scalars.push(vec![scalar]);
                let data = temp_scalars.last().unwrap();
                let view = HostTensorView {
                    data,
                    shape: &scalar_shape,
                };
                let handle = provider.upload(&view)?;
                prepared.push(PreparedInput {
                    handle: handle.clone(),
                    owned: Some(handle),
                });
            }
            Value::Int(i) => {
                if let Err(msg) = ensure_provider_supports_dtype(provider, scalar_dtype) {
                    return Err(anyhow!(
                        "fusion: scalar input requires unsupported precision ({msg})"
                    ));
                }
                let scalar = match provider.precision() {
                    ProviderPrecision::F32 => (i.to_f64() as f32) as f64,
                    ProviderPrecision::F64 => i.to_f64(),
                };
                temp_scalars.push(vec![scalar]);
                let data = temp_scalars.last().unwrap();
                let view = HostTensorView {
                    data,
                    shape: &scalar_shape,
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
    timer.mark("prepare_inputs");

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
    timer.mark("generate_wgsl");
    if std::env::var("RUNMAT_DEBUG_DUMP_FUSED_WGSL").is_ok() {
        println!(
            "---- fused reduction WGSL ----\n{}\n------------------------------",
            shader
        );
    }

    let mut wg = if workgroup_size == 0 {
        provider.default_reduction_workgroup_size()
    } else {
        workgroup_size
    };
    if let Ok(raw) = std::env::var("RUNMAT_FUSED_WG") {
        if let Ok(val) = raw.trim().parse::<u32>() {
            if val > 0 {
                let capped = val.min(provider.default_reduction_workgroup_size());
                wg = capped.max(1);
            }
        }
    }
    let flavor = request
        .plan
        .reduction_flavor
        .unwrap_or(ReductionFlavor::Sum);
    let output = provider.fused_reduction(
        &shader,
        &handles,
        &output_shape,
        reduce_len,
        num_slices,
        wg,
        flavor,
    )?;
    timer.mark("dispatch");
    fusion_residency::mark(&output);

    for input in prepared {
        if let Some(handle) = input.owned {
            let _ = provider.free(&handle);
        }
    }
    timer.mark("cleanup");
    timer.finish();

    Ok(Value::GpuTensor(output))
}

pub async fn execute_centered_gram(request: FusionExecutionRequest<'_>) -> Result<Value> {
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

    let options = CovarianceOptions {
        normalization,
        rows: CovRows::All,
        has_weight_vector: false,
    };

    let output = provider
        .covariance(&matrix_handle, None, None, &options)
        .await?;

    if let Some(temp) = owned_matrix {
        let _ = provider.free(&temp);
    }

    fusion_residency::mark(&output);
    Ok(Value::GpuTensor(output))
}

pub async fn execute_power_step_normalize(request: FusionExecutionRequest<'_>) -> Result<Value> {
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
    let output = provider
        .matmul_power_step(&lhs_handle, &rhs_handle, &desc)
        .await?;

    if let Some(temp) = lhs_owned {
        let _ = provider.free(&temp);
    }
    if let Some(temp) = rhs_owned {
        let _ = provider.free(&temp);
    }

    fusion_residency::mark(&output);
    Ok(Value::GpuTensor(output))
}

pub async fn execute_explained_variance(request: FusionExecutionRequest<'_>) -> Result<Value> {
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

    let debug_explained = std::env::var("RUNMAT_DEBUG_EXPLAINED").is_ok();
    if debug_explained {
        println!(
            "[explained] initial Q shape {:?}, G shape {:?}",
            q_handle.shape, g_handle.shape
        );
        if let Ok(info) = provider.download(&q_handle).await {
            println!(
                "[explained] Q (sample) len={} first=[{:?}]",
                info.data.len(),
                info.data.get(0..4)
            );
        }
    }

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
        return Err(anyhow!("explained variance: G shape mismatch"));
    }

    let mut tmp = provider.matmul(&q_handle, &g_handle).await?;
    let tmp_shape = tmp.shape.clone();
    if tmp_shape.len() < 2 {
        return Err(anyhow!("explained variance: intermediate must be 2-D"));
    }
    if tmp_shape[0] != q_cols {
        return Err(anyhow!(
            "explained variance: expected intermediate rows {}, got {}",
            q_cols,
            tmp_shape[0]
        ));
    }

    if debug_explained {
        println!("[explained] after Q*G tmp shape {:?}", tmp.shape);
    }

    // Interpreter's transpose retains the original data layout. Mimic that by
    // reshaping rather than launching a real transpose so downstream matmul
    // observes the same misoriented layout.
    let mut transposed_shape = q_shape.clone();
    transposed_shape.swap(0, 1);
    let q_transposed_view = provider.reshape(&q_handle, &transposed_shape)?;

    tmp = provider.matmul(&q_transposed_view, &g_handle).await?;

    if debug_explained {
        println!(
            "[explained] after reshape(matmul) tmp shape {:?}",
            tmp.shape
        );
    }

    // Restore Q's original shape before the second multiplication.
    q_handle = provider.reshape(&q_handle, &q_shape)?;

    let product = provider.matmul(&tmp, &q_handle).await?;

    if debug_explained {
        println!("[explained] product shape {:?}", product.shape);
    }

    let diag = provider.diag_extract(&product, 0)?;
    let diag = match diag.shape.as_slice() {
        [len] => provider.reshape(&diag, &[*len, 1])?,
        [_len, 1] => diag,
        _ => diag,
    };

    if debug_explained {
        if let Ok(host) = provider.download(&tmp).await {
            println!("tmp runtime shape {:?} data {:?}", host.shape, host.data);
        }
        if let Ok(host) = provider.download(&product).await {
            println!("prod runtime shape {:?} data {:?}", host.shape, host.data);
        }
        if let Ok(host) = provider.download(&diag).await {
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

pub async fn execute_image_normalize(request: FusionExecutionRequest<'_>) -> Result<Value> {
    crate::ensure_residency_hooks();
    if request.plan.group.kind != FusionKind::ImageNormalize {
        return Err(anyhow!("unsupported fusion kind"));
    }
    let provider = provider().ok_or_else(|| anyhow!("no acceleration provider registered"))?;
    let pattern = match request.plan.pattern.as_ref() {
        Some(FusionPattern::ImageNormalize(p)) => p,
        _ => return Err(anyhow!("image normalize: missing pattern metadata")),
    };
    if log::log_enabled!(log::Level::Debug) {
        log::debug!(
            "execute_image_normalize: plan inputs={:?} stack={:?}",
            request.plan.inputs,
            request.plan.stack_pattern
        );
    }

    let find_value = |vid: ValueId| -> Result<&Value> {
        if let Some(pos) = request.plan.inputs.iter().position(|id| *id == vid) {
            request
                .inputs
                .get(pos)
                .ok_or_else(|| anyhow!("image normalize: runtime value missing"))
        } else {
            request
                .plan
                .const_values
                .get(&vid)
                .ok_or_else(|| anyhow!("image normalize: value {vid:?} not materialized"))
        }
    };

    let input_value = find_value(pattern.input)?;
    let (input_handle, input_owned) = ensure_gpu_tensor(provider, input_value)?;
    let shape = input_handle.shape.clone();
    if shape.len() != 3 {
        return Err(anyhow!(
            "image normalize: expected 3-D input tensor, got shape {:?}",
            shape
        ));
    }
    let batch = shape[0];
    let height = shape[1];
    let width = shape[2];

    let epsilon = resolve_image_scalar_value(&pattern.epsilon, request.plan, &request)?;
    let gain = match &pattern.gain {
        Some(s) => Some(resolve_image_scalar_value(s, request.plan, &request)?),
        None => None,
    };
    let bias = match &pattern.bias {
        Some(s) => Some(resolve_image_scalar_value(s, request.plan, &request)?),
        None => None,
    };
    let gamma = match &pattern.gamma {
        Some(s) => Some(resolve_image_scalar_value(s, request.plan, &request)?),
        None => None,
    };

    let desc = ImageNormalizeDescriptor {
        batch,
        height,
        width,
        epsilon,
        gain,
        bias,
        gamma,
    };
    if log::log_enabled!(log::Level::Trace) {
        log::trace!("execute_image_normalize: desc {:?}", desc);
    }

    let output = provider.image_normalize(&input_handle, &desc).await?;

    if let Some(temp) = input_owned {
        provider.free(&temp).ok();
    }

    fusion_residency::mark(&output);
    Ok(Value::GpuTensor(output))
}

pub async fn execute_matmul_epilogue(request: FusionExecutionRequest<'_>) -> Result<Value> {
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
                let view = HostTensorView {
                    data: &t.data,
                    shape: &t.shape,
                };
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
        prepared
            .iter()
            .find_map(|(v, h, _)| if *v == vid { Some(h.clone()) } else { None })
    };

    // Find matmul op and its output
    let mut cur_out: Option<graph::ValueId> = None;
    let mut a_vid: Option<graph::ValueId> = None;
    let mut b_vid: Option<graph::ValueId> = None;
    for op in &request.plan.operations {
        if let crate::fusion::FusionOp::Builtin {
            name,
            inputs,
            output,
        } = op
        {
            if name.eq_ignore_ascii_case("mtimes") {
                a_vid = inputs.first().copied();
                b_vid = inputs.get(1).copied();
                cur_out = *output;
                break;
            }
        }
    }
    let (a_vid, b_vid, mut cur) = (
        a_vid.ok_or_else(|| anyhow!("mtimes not found"))?,
        b_vid.ok_or_else(|| anyhow!("mtimes not found"))?,
        cur_out.ok_or_else(|| anyhow!("mtimes output missing"))?,
    );

    // Derive epilogue (scale/bias, clamp, pow, diag) by walking subsequent ops that consume cur
    let mut alpha: f64 = 1.0;
    let mut beta: f64 = 0.0;
    let mut row_scale: Option<GpuTensorHandle> = None;
    let mut col_scale: Option<GpuTensorHandle> = None;
    let mut clamp_min: Option<f64> = None;
    let mut clamp_max: Option<f64> = None;
    let mut pow_exponent: Option<f64> = None;
    let mut row_div = false;
    let mut col_div = false;
    let mut diag_vid: Option<graph::ValueId> = None;

    for op in &request.plan.operations {
        match op {
            crate::fusion::FusionOp::Primitive { op, inputs, output } => {
                let Some(out) = output else { continue };
                if !inputs.contains(&cur) {
                    continue;
                }
                let other = if inputs[0] == cur {
                    inputs[1]
                } else {
                    inputs[0]
                };
                let const_opt = request.plan.const_values.get(&other);
                let const_f64 = const_opt.and_then(value_to_f64);
                match op {
                    crate::graph::PrimitiveOp::Mul | crate::graph::PrimitiveOp::ElemMul => {
                        if let Some(val) = const_f64 {
                            alpha *= val;
                        } else if row_scale.is_none() || col_scale.is_none() {
                            if let Some(h) = find_handle(other) {
                                let r = h.shape.first().copied().unwrap_or(1);
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
                        if let Some(val) = const_f64 {
                            if val != 0.0 {
                                alpha *= 1.0 / val;
                            }
                        } else if row_scale.is_none() || col_scale.is_none() {
                            if let Some(h) = find_handle(other) {
                                let r = h.shape.first().copied().unwrap_or(1);
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
                        if let Some(val) = const_f64 {
                            beta += val;
                        }
                    }
                    crate::graph::PrimitiveOp::Sub => {
                        if let Some(val) = const_f64 {
                            beta -= val;
                        }
                    }
                    crate::graph::PrimitiveOp::Pow | crate::graph::PrimitiveOp::ElemPow => {
                        if pow_exponent.is_none() && inputs[0] == cur {
                            pow_exponent = const_f64;
                        }
                    }
                    _ => {}
                }
                cur = *out;
            }
            crate::fusion::FusionOp::Builtin {
                name,
                inputs,
                output,
            } => {
                let Some(out) = output else { continue };
                if !inputs.contains(&cur) {
                    continue;
                }
                let lower = name.to_ascii_lowercase();
                if lower == "max" || lower == "min" {
                    if let Some(&other) = inputs.iter().find(|&&v| v != cur) {
                        if let Some(val) =
                            request.plan.const_values.get(&other).and_then(value_to_f64)
                        {
                            if lower == "max" {
                                clamp_min = Some(clamp_min.map_or(val, |prev| prev.max(val)));
                            } else {
                                clamp_max = Some(clamp_max.map_or(val, |prev| prev.min(val)));
                            }
                        }
                    }
                } else if lower == "pow" && pow_exponent.is_none() {
                    if let Some(&other) = inputs.iter().find(|&&v| v != cur) {
                        if let Some(val) =
                            request.plan.const_values.get(&other).and_then(value_to_f64)
                        {
                            pow_exponent = Some(val);
                        }
                    }
                } else if lower == "diag" {
                    diag_vid = Some(*out);
                }
                cur = *out;
            }
        }
    }

    // Build epilogue descriptor
    let mut ep = runmat_accelerate_api::MatmulEpilogue::noop();
    ep.alpha = alpha;
    ep.beta = beta;
    ep.clamp_min = clamp_min;
    ep.clamp_max = clamp_max;
    ep.pow_exponent = pow_exponent;
    ep.row_op = if row_div {
        runmat_accelerate_api::ScaleOp::Divide
    } else {
        runmat_accelerate_api::ScaleOp::Multiply
    };
    ep.col_op = if col_div {
        runmat_accelerate_api::ScaleOp::Divide
    } else {
        runmat_accelerate_api::ScaleOp::Multiply
    };
    if let Some(h) = row_scale.clone() {
        ep.row_scale = Some(h);
    }
    if let Some(h) = col_scale.clone() {
        ep.col_scale = Some(h);
    }

    let a = find_handle(a_vid).ok_or_else(|| anyhow!("missing A"))?;
    let b = find_handle(b_vid).ok_or_else(|| anyhow!("missing B"))?;

    let mut diag_handle: Option<(graph::ValueId, GpuTensorHandle)> = None;
    if let Some(vid) = diag_vid {
        let diag_len = std::cmp::min(
            a.shape.first().copied().unwrap_or(0),
            b.shape.get(1).copied().unwrap_or(0),
        );
        let mut diag_shape = vec![diag_len, 1];
        if diag_len == 0 {
            diag_shape[1] = 1;
        }
        let handle = prov.zeros(&diag_shape)?;
        ep.diag_output = Some(handle.clone());
        diag_handle = Some((vid, handle));
    }

    let out = prov.matmul_epilogue(&a, &b, &ep).await?;
    for h in owned {
        let _ = prov.free(&h);
    }

    if let Some((_, diag)) = &diag_handle {
        fusion_residency::mark(diag);
    }

    let final_vid = request.plan.output.or(Some(cur));
    let mut result = out.clone();
    let mut free_out = false;
    if let Some((vid, diag)) = &diag_handle {
        if Some(*vid) == final_vid {
            result = diag.clone();
            free_out = true;
        }
    }

    if free_out {
        let _ = prov.free(&out);
    } else {
        fusion_residency::mark(&out);
    }

    fusion_residency::mark(&result);
    Ok(Value::GpuTensor(result))
}
