use anyhow::{anyhow, Result};

use crate::fusion::{
    FusionGroupPlan, FusionKind, FusionPattern, FusionStoreMaterialization, ImageScalar,
};
use crate::fusion_residency;
use crate::graph::{ShapeInfo, ValueId};
use crate::placement::{
    dense_output_representation, fusion_operation_token, plan_local, FusionPlacementObserver,
    LocalPlacementFeedback, LocalPlacementOutcome, LocalPlacementRequest, PlacementCorrelationId,
    PlacementPolicy, PlacementVariant,
};
use crate::precision::ensure_provider_supports_dtype;
use log;
use runmat_accelerate_api::{
    provider, AccelProvider, CovRows, CovarianceOptions, GpuTensorHandle, HostTensorView,
    ImageNormalizeDescriptor, PowerStepEpilogue, ProviderOperationFamily, ProviderPrecision,
    ProviderWorkload, ReductionFlavor,
};
use runmat_execution::{
    CandidateExecutionLocation, CandidateOutputResidency, CandidatePreparationState,
    EstimateConfidence, EstimateSource, ExecutionCandidateDescriptor, ExecutionCandidateKind,
    ExecutionCostComponents, ExecutionCostEstimate,
};
use runmat_runtime::builtins::common::shape::normalize_scalar_shape;
use runmat_runtime::gather_if_needed;
use runmat_time::Instant;
use runmat_value::{NumericDType, Value};
use std::collections::HashMap;
use std::sync::OnceLock;

mod observation;
mod provider_observation;
use provider_observation::FusionPlacementSelection;
mod admission;
use admission::AdmittedBindings;
mod outputs;
mod elementwise;
pub use elementwise::{execute_elementwise, execute_elementwise_placed};
pub use observation::{FusionCpuObservation, FusionExecutionOutcome};

struct PreparedInput {
    handle: GpuTensorHandle,
    owned: Option<GpuTensorHandle>,
}

fn mark_fusion_output(handle: &mut GpuTensorHandle, inputs: &[Value]) {
    let explicit = inputs.iter().any(|value| {
        matches!(value, Value::GpuTensor(input) if runmat_accelerate_api::handle_is_explicit(input))
    });
    if explicit {
        runmat_accelerate_api::mark_handle_explicit(handle);
    } else {
        runmat_accelerate_api::mark_handle_automatic(handle);
    }
    fusion_residency::mark(handle);
}

pub struct FusionExecutionRequest<'a> {
    pub plan: &'a FusionGroupPlan,
    pub inputs: Vec<Value>,
    pub placement: Option<PlacementCorrelationId>,
    pub runtime: Option<runmat_runtime::context::RuntimeContext>,
}

pub struct ElementwiseExecutionResult {
    pub final_value: Value,
    pub materialized_stores: Vec<(FusionStoreMaterialization, Value)>,
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
    observer: FusionPlacementObserver,
) -> Result<(GpuTensorHandle, Option<GpuTensorHandle>)> {
    reject_native_integer_fusion_values(std::slice::from_ref(value), &[])?;
    match value {
        Value::GpuTensor(handle) => Ok((handle.clone(), None)),
        Value::Tensor(tensor) => {
            let data = tensor.materialize_f64();
            let view = HostTensorView {
                data: &data,
                shape: &tensor.shape,
            };
            let started = Instant::now();
            let mut handle = provider.upload(&view)?;
            observer.upload(started, host_f64_bytes(data.len()));
            runmat_accelerate_api::mark_handle_automatic(&mut handle);
            Ok((handle.clone(), Some(handle)))
        }
        _ => Err(anyhow!("fusion: expected tensor input")),
    }
}

fn host_f64_bytes(len: usize) -> Option<u64> {
    u64::try_from(len).ok()?.checked_mul(8)
}

fn fusion_input_elements(values: &[Value]) -> Option<usize> {
    values
        .iter()
        .filter_map(|value| match value {
            Value::Tensor(tensor) => Some(tensor.len()),
            Value::GpuTensor(handle) => Some(handle.shape.iter().copied().product::<usize>()),
            Value::Num(_) => Some(1),
            _ => None,
        })
        .max()
}

struct FusionProviderContract<'a> {
    outputs: Vec<runmat_accelerate_api::ProviderRepresentation>,
    semantic_identity: Option<runmat_execution::Digest>,
    scalar_work_units: Option<u64>,
    matmul: Option<matmul::workload::Selection<'a>>,
    preparation_program: Option<runmat_accelerate_api::ProviderPreparationProgram<'a>>,
}

fn select_fusion_provider(
    request: &FusionExecutionRequest<'_>,
    bindings: &AdmittedBindings,
    provider: &dyn AccelProvider,
    output_shapes: &[Vec<usize>],
) -> Result<FusionExecutionOutcome<FusionPlacementSelection>> {
    select_fusion_provider_with_contract(
        request,
        bindings,
        provider,
        FusionProviderContract {
            outputs: output_shapes
                .iter()
                .map(|shape| dense_output_representation(provider.precision(), shape))
                .collect(),
            semantic_identity: None,
            scalar_work_units: None,
            matmul: None,
            preparation_program: None,
        },
    )
}

fn select_fusion_provider_with_contract(
    request: &FusionExecutionRequest<'_>,
    bindings: &AdmittedBindings,
    provider: &dyn AccelProvider,
    contract: FusionProviderContract<'_>,
) -> Result<FusionExecutionOutcome<FusionPlacementSelection>> {
    let FusionProviderContract {
        outputs,
        semantic_identity,
        scalar_work_units,
        matmul,
        preparation_program,
    } = contract;
    let scalar_work_units = if let Some(contract) = &matmul {
        Some(
            crate::placement::matmul_scalar_work_units(&contract.workload, provider.precision())
                .ok_or_else(|| anyhow!("matrix work estimate overflows"))?,
        )
    } else {
        scalar_work_units
    };
    let observer = FusionPlacementObserver::new(request.placement);
    let typed_elementwise = semantic_identity.is_some() && request.plan.group.kind.is_elementwise();
    let semantic_identity = Some(bindings.identity(semantic_identity)?);
    let input_refs = matmul.as_ref().map_or_else(
        || request.inputs.iter().collect::<Vec<_>>(),
        |contract| contract.inputs.clone(),
    );
    let elements = if let Some(contract) = &matmul {
        contract
            .workload
            .rows
            .checked_mul(contract.workload.columns)
            .and_then(|n| usize::try_from(n).ok())
    } else if typed_elementwise {
        outputs
            .first()
            .and_then(|output| output.checked_element_count())
            .and_then(|count| usize::try_from(count).ok())
    } else {
        fusion_input_elements(&request.inputs)
    };
    let required_upload_bytes = if let Some(contract) = &matmul {
        Some(contract.upload_bytes)
    } else if typed_elementwise {
        let width = match provider.precision() {
            ProviderPrecision::F32 => 4u64,
            ProviderPrecision::F64 => 8u64,
        };
        Some(
            request
                .inputs
                .iter()
                .try_fold(0u64, |bytes, value| {
                    let elements = match value {
                        Value::Num(_) => 1,
                        Value::Tensor(tensor) => u64::try_from(tensor.len()).ok()?,
                        Value::GpuTensor(_) => 0,
                        _ => return None,
                    };
                    bytes.checked_add(elements.checked_mul(width)?)
                })
                .ok_or_else(|| anyhow!("fusion upload byte count overflows"))?,
        )
    } else {
        None
    };
    if request.placement.is_none() {
        let _ = observer.select_provider(
            provider,
            &request.plan.group.kind,
            &input_refs,
            &outputs,
            ProviderWorkload {
                elements: elements.and_then(|n| u64::try_from(n).ok()),
                scalar_work_units,
                matmul: matmul.as_ref().map(|contract| contract.workload.clone()),
                ..Default::default()
            },
        )?;
        observer.selected(PlacementVariant::ProviderFusion, "direct_request", None);
        return Ok(FusionExecutionOutcome::Executed(
            FusionPlacementSelection::new(observer, None, provider, None),
        ));
    }

    let operation = fusion_operation_token(&request.plan.group.kind);
    let element_count = u64::try_from(elements.unwrap_or(1)).unwrap_or(u64::MAX);
    let execution_units = scalar_work_units.unwrap_or(element_count);
    let cpu_cost = ExecutionCostEstimate {
        components: ExecutionCostComponents {
            execution_ns: execution_units.saturating_mul(100).max(100),
            ..ExecutionCostComponents::default()
        },
        scratch_bytes: 0,
        confidence: EstimateConfidence::Low,
        source: EstimateSource::StaticPrior,
    };
    let cpu = ExecutionCandidateDescriptor {
        identity: format!("cpu.shared.{operation}"),
        region: None,
        kind: ExecutionCandidateKind::SharedRuntime,
        execution_location: CandidateExecutionLocation::Host,
        preparation: CandidatePreparationState::Ready,
        cost: cpu_cost,
        output_residency: CandidateOutputResidency::Host,
        guards: Vec::new(),
    };
    let provider_fallback = ExecutionCostEstimate {
        components: ExecutionCostComponents {
            compile_or_prepare_ns: 20_000,
            allocation_ns: 5_000,
            queue_ns: 10_000,
            execution_ns: execution_units.saturating_mul(10).max(100),
            ..ExecutionCostComponents::default()
        },
        scratch_bytes: 0,
        confidence: EstimateConfidence::Prior,
        source: EstimateSource::StaticPrior,
    };
    match plan_local(
        provider,
        LocalPlacementRequest {
            operation,
            semantic_identity,
            family: ProviderOperationFamily::Fusion,
            provider_kind: ExecutionCandidateKind::ProviderFusion,
            inputs: &input_refs,
            outputs,
            workload: ProviderWorkload {
                matmul: matmul.as_ref().map(|contract| contract.workload.clone()),
                elements: Some(element_count),
                flops: None,
                scalar_work_units,
                batch: None,
                numerical_domain: Default::default(),
                requested_outputs: None,
            },
            // Without an exact shader-cache readiness receipt, charge the
            // conservative preparation prior, never invent a warm matrix path.
            preparation: if matmul.is_some() {
                CandidatePreparationState::Cold
            } else {
                CandidatePreparationState::Warm
            },
            preparation_program,
            cpu,
            provider_fallback,
            required_upload_bytes,
            required_download_bytes: 0,
            downstream_materialization: false,
            compare_profitability: true,
            runtime: request.runtime.as_ref(),
        },
        PlacementPolicy::default(),
    )? {
        LocalPlacementOutcome::ProviderRejected { code, .. } => {
            observer.provider_rejected(code);
            Err(anyhow!("fusion provider rejected operation"))
        }
        LocalPlacementOutcome::Selected {
            selected,
            provider_cost,
            feedback,
            provider_query,
            provider_observation,
            ..
        } if selected.kind.is_provider() => {
            observer.provider_candidate();
            observer.selected(
                PlacementVariant::ProviderFusion,
                "complete_cost",
                provider_cost.checked_total_ns(),
            );
            Ok(FusionExecutionOutcome::Executed(
                FusionPlacementSelection::new(
                    observer,
                    feedback.map(|feedback| *feedback),
                    provider,
                    provider_observation.then_some(provider_query.as_ref()),
                ),
            ))
        }
        LocalPlacementOutcome::Selected {
            cpu_cost, feedback, ..
        } => {
            observer.provider_candidate();
            observer.selected(
                PlacementVariant::SharedRuntime,
                "complete_cost",
                cpu_cost.checked_total_ns(),
            );
            Ok(FusionExecutionOutcome::SharedRuntime(
                FusionCpuObservation::new(feedback),
            ))
        }
    }
}

fn scalar_upload_dtype(provider: &dyn AccelProvider) -> NumericDType {
    match provider.precision() {
        ProviderPrecision::F32 => NumericDType::F32,
        ProviderPrecision::F64 => NumericDType::F64,
    }
}

/// Float descriptors and kernels intentionally accept only floating scalar
/// values. Integer values have a distinct provider representation and must not
/// be converted through f64 merely because this particular fusion has no typed
/// integer kernel yet.
fn value_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Num(n) => Some(*n),
        _ => None,
    }
}

fn native_integer_fusion_value_kind(value: &Value) -> Option<&'static str> {
    match value {
        Value::Int(_) => Some("integer scalar"),
        Value::Tensor(t) if t.integer_storage().is_some() => Some("integer tensor"),
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_integer_type(handle).is_some() =>
        {
            Some("native integer gpuArray")
        }
        _ => None,
    }
}

fn reject_native_integer_fusion_values(values: &[Value], constants: &[&Value]) -> Result<()> {
    if let Some(kind) = values
        .iter()
        .chain(constants.iter().copied())
        .find_map(native_integer_fusion_value_kind)
    {
        return Err(anyhow!(
            "fusion: {kind} requires a dedicated typed integer fusion kernel; refusing f64 fallback"
        ));
    }
    Ok(())
}

fn reject_native_integer_fusion_inputs(request: &FusionExecutionRequest<'_>) -> Result<()> {
    let constants: Vec<&Value> = request.plan.const_values.values().collect();
    reject_native_integer_fusion_values(&request.inputs, &constants)
}

fn scalar_from_value(value: &Value, observer: FusionPlacementObserver) -> Result<f64> {
    reject_native_integer_fusion_values(std::slice::from_ref(value), &[])?;
    if let Some(v) = value_to_f64(value) {
        return Ok(v);
    }
    match value {
        Value::Tensor(t) => {
            if t.len() == 1 {
                Ok(t.numeric_value_at(0)
                    .expect("validated scalar tensor storage")
                    .materialize_f64())
            } else {
                Err(anyhow!(
                    "image normalize: expected scalar tensor, got {} elements",
                    t.len()
                ))
            }
        }
        Value::GpuTensor(_) => {
            let started = Instant::now();
            let gathered = gather_if_needed(value).map_err(|e| anyhow!("image normalize: {e}"))?;
            let bytes = match &gathered {
                Value::Tensor(tensor) => host_f64_bytes(tensor.len()),
                _ => None,
            };
            observer.synchronize(started, bytes);
            scalar_from_value(&gathered, observer)
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
    observer: FusionPlacementObserver,
) -> Result<f64> {
    match scalar {
        ImageScalar::Constant(v) => Ok(*v),
        ImageScalar::Value(vid) => {
            if let Some(value) = plan.const_values.get(vid) {
                return scalar_from_value(value, observer);
            }
            if let Some(idx) = plan.inputs.iter().position(|id| *id == *vid) {
                let runtime_value = request
                    .inputs
                    .get(idx)
                    .ok_or_else(|| anyhow!("image normalize: runtime scalar missing"))?;
                return scalar_from_value(runtime_value, observer);
            }
            Err(anyhow!(
                "image normalize: scalar input {:?} not materialized in plan",
                vid
            ))
        }
    }
}

fn runtime_broadcast_shape(values: &[Value]) -> Option<Vec<usize>> {
    let mut shapes: Vec<Vec<usize>> = Vec::new();
    for value in values {
        match value {
            Value::GpuTensor(handle) => shapes.push(handle.shape.clone()),
            Value::Tensor(tensor) => shapes.push(tensor.shape.clone()),
            Value::Num(_) | Value::Int(_) => shapes.push(Vec::new()),
            _ => return None,
        }
    }
    let rank = shapes.iter().map(Vec::len).max().unwrap_or(0);
    let mut output = vec![1usize; rank];
    for mut shape in shapes {
        if shape.len() == 1 && rank >= 2 {
            shape.insert(0, 1);
        }
        for (dimension, &extent) in shape.iter().enumerate() {
            let current = output[dimension];
            if current == extent {
                continue;
            }
            if current == 1 {
                output[dimension] = extent;
            } else if extent != 1 {
                return None;
            }
        }
    }
    Some(output)
}

pub fn execute_reduction(
    request: FusionExecutionRequest<'_>,
    reduce_len: usize,
    num_slices: usize,
    workgroup_size: u32,
) -> Result<Value> {
    match execute_reduction_with_shape_placed(
        request,
        reduce_len,
        num_slices,
        &[num_slices],
        workgroup_size,
    )? {
        FusionExecutionOutcome::Executed(value) => Ok(value),
        FusionExecutionOutcome::SharedRuntime(_) => {
            Err(anyhow!("direct fusion request selected shared runtime"))
        }
    }
}

/// Direct provider execution retaining the established value-returning API.
/// Executors that can run shared bytecode use the placement-aware entrypoint.
pub fn execute_reduction_with_shape(
    request: FusionExecutionRequest<'_>,
    reduce_len: usize,
    num_slices: usize,
    output_shape: &[usize],
    workgroup_size: u32,
) -> Result<Value> {
    match execute_reduction_with_shape_placed(
        request,
        reduce_len,
        num_slices,
        output_shape,
        workgroup_size,
    )? {
        FusionExecutionOutcome::Executed(value) => Ok(value),
        FusionExecutionOutcome::SharedRuntime(_) => {
            Err(anyhow!("direct fusion request selected shared runtime"))
        }
    }
}

pub fn execute_reduction_with_shape_placed(
    request: FusionExecutionRequest<'_>,
    reduce_len: usize,
    num_slices: usize,
    output_shape: &[usize],
    workgroup_size: u32,
) -> Result<FusionExecutionOutcome<Value>> {
    let Some(bindings) = AdmittedBindings::new(&request) else {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    };
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
    reject_native_integer_fusion_inputs(&request)?;
    let len = reduce_len * num_slices;
    if len == 0 {
        return Err(anyhow!("fusion: zero-length execution not supported"));
    }
    let scalar_shape = {
        let constant_shape = request.plan.constant_shape(len);
        normalize_scalar_shape(&vec![1; constant_shape.len()])
    };
    let selection =
        select_fusion_provider(&request, &bindings, provider, &[output_shape.to_vec()])?;
    let selection = match selection {
        FusionExecutionOutcome::Executed(selection) => selection,
        FusionExecutionOutcome::SharedRuntime(observation) => {
            return Ok(FusionExecutionOutcome::SharedRuntime(observation))
        }
    };
    let observer = selection.observer;
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
                if let Err(msg) = ensure_provider_supports_dtype(provider, t.numeric_dtype()) {
                    return Err(anyhow!(
                        "fusion: tensor input requires unsupported precision ({msg})"
                    ));
                }
                let data = t.materialize_f64();
                let view = HostTensorView {
                    data: &data,
                    shape: &t.shape,
                };
                let started = Instant::now();
                let handle = provider.upload(&view)?;
                observer.upload(started, host_f64_bytes(data.len()));
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
                let started = Instant::now();
                let handle = provider.upload(&view)?;
                observer.upload(started, host_f64_bytes(data.len()));
                prepared.push(PreparedInput {
                    handle: handle.clone(),
                    owned: Some(handle),
                });
            }
            Value::Int(_) => unreachable!("integer inputs rejected before fusion upload"),
            _ => return Err(anyhow!("fusion: unsupported value type")),
        }
    }
    timer.mark("prepare_inputs");

    let handles: Vec<GpuTensorHandle> = prepared.iter().map(|p| p.handle.clone()).collect();
    let output_len = output_shape.iter().copied().product::<usize>();
    if output_len != num_slices {
        return Err(anyhow!(
            "fusion: reduction output shape {:?} contains {} elements, expected {}",
            output_shape,
            output_len,
            num_slices
        ));
    }

    let scalar_ty = match provider.precision() {
        ProviderPrecision::F32 => "f32",
        ProviderPrecision::F64 => "f64",
    };
    let started = Instant::now();
    let shader = request
        .plan
        .generate_reduction_wgsl(scalar_ty)
        .ok_or_else(|| anyhow!("fusion: reduction WGSL generation failed"))?;
    observer.compile(started);
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
    let started = Instant::now();
    let mut output = provider.fused_reduction(
        &shader,
        &handles,
        output_shape,
        reduce_len,
        num_slices,
        wg,
        flavor,
    )?;
    observer.queue(started);
    observer.kernel_unmeasured();
    timer.mark("dispatch");
    mark_fusion_output(&mut output, &request.inputs);

    for input in prepared {
        if let Some(handle) = input.owned {
            let _ = provider.free(&handle);
        }
    }
    timer.mark("cleanup");
    timer.finish();

    selection.submitted();
    Ok(FusionExecutionOutcome::Executed(Value::GpuTensor(output)))
}

/// Direct provider execution retaining the established value-returning API.
/// Executors that can run shared bytecode use the placement-aware entrypoint.
pub async fn execute_centered_gram(request: FusionExecutionRequest<'_>) -> Result<Value> {
    match execute_centered_gram_placed(request).await? {
        FusionExecutionOutcome::Executed(value) => Ok(value),
        FusionExecutionOutcome::SharedRuntime(_) => {
            Err(anyhow!("direct fusion request selected shared runtime"))
        }
    }
}

pub async fn execute_centered_gram_placed(
    request: FusionExecutionRequest<'_>,
) -> Result<FusionExecutionOutcome<Value>> {
    let Some(bindings) = AdmittedBindings::new(&request) else {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    };
    crate::ensure_residency_hooks();
    reject_native_integer_fusion_inputs(&request)?;
    if request.plan.group.kind != FusionKind::CenteredGram {
        return Err(anyhow!("unsupported fusion kind"));
    }
    let provider = provider().ok_or_else(|| anyhow!("no acceleration provider registered"))?;
    let selection = select_fusion_provider(&request, &bindings, provider, &[])?;
    let selection = match selection {
        FusionExecutionOutcome::Executed(selection) => selection,
        FusionExecutionOutcome::SharedRuntime(observation) => {
            return Ok(FusionExecutionOutcome::SharedRuntime(observation))
        }
    };
    let observer = selection.observer;
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

    let (matrix_handle, owned_matrix) = ensure_gpu_tensor(provider, matrix_value, observer)?;

    let options = CovarianceOptions {
        normalization,
        rows: CovRows::All,
        has_weight_vector: false,
    };

    observer.queue_unmeasured();
    let mut output = provider
        .covariance(&matrix_handle, None, None, &options)
        .await?;
    observer.kernel_unmeasured();

    if let Some(temp) = owned_matrix {
        let _ = provider.free(&temp);
    }

    mark_fusion_output(&mut output, &request.inputs);
    selection.submitted();
    Ok(FusionExecutionOutcome::Executed(Value::GpuTensor(output)))
}

/// Direct provider execution retaining the established value-returning API.
/// Executors that can run shared bytecode use the placement-aware entrypoint.
pub async fn execute_power_step_normalize(request: FusionExecutionRequest<'_>) -> Result<Value> {
    match execute_power_step_normalize_placed(request).await? {
        FusionExecutionOutcome::Executed(value) => Ok(value),
        FusionExecutionOutcome::SharedRuntime(_) => {
            Err(anyhow!("direct fusion request selected shared runtime"))
        }
    }
}

pub async fn execute_power_step_normalize_placed(
    request: FusionExecutionRequest<'_>,
) -> Result<FusionExecutionOutcome<Value>> {
    let Some(bindings) = AdmittedBindings::new(&request) else {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    };
    crate::ensure_residency_hooks();
    reject_native_integer_fusion_inputs(&request)?;
    if request.plan.group.kind != FusionKind::PowerStepNormalize {
        return Err(anyhow!("unsupported fusion kind"));
    }
    let provider = provider().ok_or_else(|| anyhow!("no acceleration provider registered"))?;
    let selection = select_fusion_provider(&request, &bindings, provider, &[])?;
    let selection = match selection {
        FusionExecutionOutcome::Executed(selection) => selection,
        FusionExecutionOutcome::SharedRuntime(observation) => {
            return Ok(FusionExecutionOutcome::SharedRuntime(observation))
        }
    };
    let observer = selection.observer;
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

    let (lhs_handle, lhs_owned) = ensure_gpu_tensor(provider, lhs_value, observer)?;
    let (rhs_handle, rhs_owned) = ensure_gpu_tensor(provider, rhs_value, observer)?;

    let desc = PowerStepEpilogue { epsilon };
    observer.queue_unmeasured();
    let mut output = provider
        .matmul_power_step(&lhs_handle, &rhs_handle, &desc)
        .await?;
    observer.kernel_unmeasured();

    if let Some(temp) = lhs_owned {
        let _ = provider.free(&temp);
    }
    if let Some(temp) = rhs_owned {
        let _ = provider.free(&temp);
    }

    mark_fusion_output(&mut output, &request.inputs);
    selection.submitted();
    Ok(FusionExecutionOutcome::Executed(Value::GpuTensor(output)))
}

/// Direct provider execution retaining the established value-returning API.
/// Executors that can run shared bytecode use the placement-aware entrypoint.
pub async fn execute_explained_variance(request: FusionExecutionRequest<'_>) -> Result<Value> {
    match execute_explained_variance_placed(request).await? {
        FusionExecutionOutcome::Executed(value) => Ok(value),
        FusionExecutionOutcome::SharedRuntime(_) => {
            Err(anyhow!("direct fusion request selected shared runtime"))
        }
    }
}

pub async fn execute_explained_variance_placed(
    request: FusionExecutionRequest<'_>,
) -> Result<FusionExecutionOutcome<Value>> {
    let Some(bindings) = AdmittedBindings::new(&request) else {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    };
    crate::ensure_residency_hooks();
    reject_native_integer_fusion_inputs(&request)?;
    if request.plan.group.kind != FusionKind::ExplainedVariance {
        return Err(anyhow!("unsupported fusion kind"));
    }
    let provider = provider().ok_or_else(|| anyhow!("no acceleration provider registered"))?;
    let selection = select_fusion_provider(&request, &bindings, provider, &[])?;
    let selection = match selection {
        FusionExecutionOutcome::Executed(selection) => selection,
        FusionExecutionOutcome::SharedRuntime(observation) => {
            return Ok(FusionExecutionOutcome::SharedRuntime(observation))
        }
    };
    let observer = selection.observer;
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

    let (mut q_handle, q_owned) = ensure_gpu_tensor(provider, q_value, observer)?;
    let (g_handle, g_owned) = ensure_gpu_tensor(provider, g_value, observer)?;

    let debug_explained = std::env::var("RUNMAT_DEBUG_EXPLAINED").is_ok();
    if debug_explained {
        println!(
            "[explained] initial Q shape {:?}, G shape {:?}",
            q_handle.shape, g_handle.shape
        );
        let started = Instant::now();
        if let Ok(info) = provider.download(&q_handle).await {
            observer.synchronize(started, host_f64_bytes(info.data.len()));
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

    observer.queue_unmeasured();
    let mut tmp = provider.matmul(&q_handle, &g_handle).await?;
    observer.kernel_unmeasured();
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

    observer.queue_unmeasured();
    tmp = provider.matmul(&q_transposed_view, &g_handle).await?;
    observer.kernel_unmeasured();

    if debug_explained {
        println!(
            "[explained] after reshape(matmul) tmp shape {:?}",
            tmp.shape
        );
    }

    // Restore Q's original shape before the second multiplication.
    q_handle = provider.reshape(&q_handle, &q_shape)?;

    observer.queue_unmeasured();
    let product = provider.matmul(&tmp, &q_handle).await?;
    observer.kernel_unmeasured();

    if debug_explained {
        println!("[explained] product shape {:?}", product.shape);
    }

    let started = Instant::now();
    let diag = provider.diag_extract(&product, 0)?;
    observer.queue(started);
    observer.kernel_unmeasured();
    let mut diag = match diag.shape.as_slice() {
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

    mark_fusion_output(&mut diag, &request.inputs);
    selection.submitted();
    Ok(FusionExecutionOutcome::Executed(Value::GpuTensor(diag)))
}

/// Direct provider execution retaining the established value-returning API.
/// Executors that can run shared bytecode use the placement-aware entrypoint.
pub async fn execute_image_normalize(request: FusionExecutionRequest<'_>) -> Result<Value> {
    match execute_image_normalize_placed(request).await? {
        FusionExecutionOutcome::Executed(value) => Ok(value),
        FusionExecutionOutcome::SharedRuntime(_) => {
            Err(anyhow!("direct fusion request selected shared runtime"))
        }
    }
}

pub async fn execute_image_normalize_placed(
    request: FusionExecutionRequest<'_>,
) -> Result<FusionExecutionOutcome<Value>> {
    let Some(bindings) = AdmittedBindings::new(&request) else {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    };
    crate::ensure_residency_hooks();
    reject_native_integer_fusion_inputs(&request)?;
    if request.plan.group.kind != FusionKind::ImageNormalize {
        return Err(anyhow!("unsupported fusion kind"));
    }
    let provider = provider().ok_or_else(|| anyhow!("no acceleration provider registered"))?;
    let selection = select_fusion_provider(&request, &bindings, provider, &[])?;
    let selection = match selection {
        FusionExecutionOutcome::Executed(selection) => selection,
        FusionExecutionOutcome::SharedRuntime(observation) => {
            return Ok(FusionExecutionOutcome::SharedRuntime(observation))
        }
    };
    let observer = selection.observer;
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
    let (input_handle, input_owned) = ensure_gpu_tensor(provider, input_value, observer)?;
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

    let epsilon = resolve_image_scalar_value(&pattern.epsilon, request.plan, &request, observer)?;
    let gain = match &pattern.gain {
        Some(s) => Some(resolve_image_scalar_value(
            s,
            request.plan,
            &request,
            observer,
        )?),
        None => None,
    };
    let bias = match &pattern.bias {
        Some(s) => Some(resolve_image_scalar_value(
            s,
            request.plan,
            &request,
            observer,
        )?),
        None => None,
    };
    let gamma = match &pattern.gamma {
        Some(s) => Some(resolve_image_scalar_value(
            s,
            request.plan,
            &request,
            observer,
        )?),
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
        clamp_zero: pattern.clamp_zero,
    };
    if log::log_enabled!(log::Level::Trace) {
        log::trace!("execute_image_normalize: desc {:?}", desc);
    }

    observer.queue_unmeasured();
    let mut output = provider.image_normalize(&input_handle, &desc).await?;
    observer.kernel_unmeasured();

    if let Some(temp) = input_owned {
        provider.free(&temp).ok();
    }

    mark_fusion_output(&mut output, &request.inputs);
    selection.submitted();
    Ok(FusionExecutionOutcome::Executed(Value::GpuTensor(output)))
}

mod matmul;
pub use matmul::{execute_matmul_epilogue, execute_matmul_epilogue_placed};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_direct_execution_signatures_remain_value_returning() {
        let _: for<'a, 'b> fn(
            FusionExecutionRequest<'a>,
            usize,
            usize,
            &'b [usize],
            u32,
        ) -> Result<Value> = execute_reduction_with_shape;
        let _: for<'a> fn(FusionExecutionRequest<'a>, usize, usize, u32) -> Result<Value> =
            execute_reduction;
        fn check_elementwise_signature(request: FusionExecutionRequest<'_>) {
            fn require_future(
                _: impl std::future::Future<Output = Result<ElementwiseExecutionResult>>,
            ) {
            }
            require_future(execute_elementwise(request));
        }
        let _: for<'a> fn(FusionExecutionRequest<'a>) = check_elementwise_signature;

        fn require_value_future(_: impl std::future::Future<Output = Result<Value>>) {}
        fn check_async_signatures(request: FusionExecutionRequest<'_>, which: usize) {
            match which {
                0 => require_value_future(execute_centered_gram(request)),
                1 => require_value_future(execute_power_step_normalize(request)),
                2 => require_value_future(execute_explained_variance(request)),
                3 => require_value_future(execute_image_normalize(request)),
                _ => require_value_future(execute_matmul_epilogue(request)),
            }
        }
        // Type-check every public future without invoking a provider or creating
        // a runtime request. Existing direct-GPU parity tests execute the APIs.
        let _: for<'a> fn(FusionExecutionRequest<'a>, usize) = check_async_signatures;
    }
    use runmat_value::IntValue;

    #[test]
    fn fusion_output_inherits_explicit_intent_or_remains_automatic() {
        let explicit_input = GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 901,
            buffer_id: 1,
            descriptor: Default::default(),
        };
        let mut automatic_output = GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 901,
            buffer_id: 2,
            descriptor: Default::default(),
        };
        let mut explicit_output = GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 901,
            buffer_id: 3,
            descriptor: Default::default(),
        };
        let mut explicit_input = explicit_input;
        runmat_accelerate_api::mark_handle_explicit(&mut explicit_input);

        mark_fusion_output(&mut automatic_output, &[Value::Num(1.0)]);
        mark_fusion_output(
            &mut explicit_output,
            &[Value::GpuTensor(explicit_input.clone())],
        );

        assert_eq!(
            runmat_accelerate_api::handle_provenance(&automatic_output),
            Some(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        );
        assert!(runmat_accelerate_api::handle_is_explicit(&explicit_output));
        for handle in [&explicit_input, &automatic_output, &explicit_output] {
            runmat_accelerate_api::clear_handle_metadata(handle);
        }
    }

    #[test]
    fn wide_uint64_scalar_is_rejected_before_float_fusion_conversion() {
        let value = Value::Int(IntValue::U64(u64::MAX));

        assert_eq!(value_to_f64(&value), None);
        let error = reject_native_integer_fusion_values(std::slice::from_ref(&value), &[])
            .expect_err("wide uint64 must not be uploaded through f64");
        assert!(error
            .to_string()
            .contains("dedicated typed integer fusion kernel"));
        assert!(error.to_string().contains("refusing f64 fallback"));
    }

    #[test]
    fn native_integer_scalars_never_use_float_fusion_fallbacks() {
        for value in [
            Value::Int(IntValue::I8(7)),
            Value::Int(IntValue::I16(7)),
            Value::Int(IntValue::I32(7)),
            Value::Int(IntValue::I64(i64::MAX)),
            Value::Int(IntValue::U8(7)),
            Value::Int(IntValue::U16(7)),
            Value::Int(IntValue::U32(7)),
            Value::Int(IntValue::U64(u64::MAX)),
        ] {
            assert_eq!(value_to_f64(&value), None);
            assert!(
                reject_native_integer_fusion_values(std::slice::from_ref(&value), &[])
                    .expect_err("integer scalar must require a typed fusion kernel")
                    .to_string()
                    .contains("refusing f64 fallback")
            );
        }
    }

    #[test]
    fn floating_scalar_remains_eligible_for_float_fusion() {
        let value = Value::Num(1.25);

        assert_eq!(value_to_f64(&value), Some(1.25));
        reject_native_integer_fusion_values(std::slice::from_ref(&value), &[])
            .expect("floating fusion input remains supported");
    }

    #[test]
    fn runtime_broadcast_appends_missing_singletons_and_enforces_zero_rules() {
        let tensor = |shape: Vec<usize>| {
            Value::Tensor(
                runmat_value::Tensor::new(vec![0.0; shape.iter().product()], shape)
                    .expect("test tensor"),
            )
        };

        assert_eq!(
            runtime_broadcast_shape(&[tensor(vec![2, 3]), tensor(vec![2, 3, 4])]),
            Some(vec![2, 3, 4])
        );
        assert_eq!(
            runtime_broadcast_shape(&[tensor(vec![2, 3]), tensor(vec![1, 2, 3])]),
            None
        );
        assert_eq!(
            runtime_broadcast_shape(&[tensor(vec![0, 3]), tensor(vec![1, 3])]),
            Some(vec![0, 3])
        );
        assert_eq!(
            runtime_broadcast_shape(&[tensor(vec![0, 3]), tensor(vec![2, 3])]),
            None
        );
        assert_eq!(
            runtime_broadcast_shape(&[tensor(vec![3]), tensor(vec![1, 3, 2])]),
            Some(vec![1, 3, 2])
        );
    }
}
