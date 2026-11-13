use anyhow::{anyhow, Result};
use runmat_accelerate_api::{ProviderExecutionError, ProviderExecutionResult as ExecutionResult};

use crate::fusion::{
    FusionGroupPlan, FusionKind, FusionPattern, FusionStoreMaterialization, ImageScalar,
};
use crate::fusion_residency;
use crate::graph::{ShapeInfo, ValueId};
use crate::placement::{
    dense_output_representation, fusion_operation_token, FusionPlacementObserver,
    LocalPlacementFeedback, LocalPlacementOutcome, LocalPlacementRequest, PlacementCorrelationId,
    PlacementPolicy, PlacementVariant,
};
use log;
use runmat_accelerate_api::{
    provider, AccelProvider, GpuTensorHandle, HostTensorView, ImageNormalizeDescriptor,
    PowerStepEpilogue, ProviderOperationFamily, ProviderPrecision, ProviderRejectionCode,
    ProviderShaderRequirements, ProviderWorkload,
};
use runmat_execution::{
    CandidateExecutionLocation, CandidateOutputResidency, CandidatePreparationState,
    EstimateConfidence, EstimateSource, ExecutionCandidateDescriptor, ExecutionCandidateKind,
    ExecutionCostComponents, ExecutionCostEstimate,
};
use runmat_runtime::gather_if_needed;
use runmat_time::Instant;
use runmat_value::{NumericDType, Value};
use std::collections::HashMap;
use std::sync::OnceLock;

mod host_input_uses;
pub(crate) mod host_readback;
mod observation;
pub use host_input_uses::{VmHostInputUses, VmHostLoadUse};
pub(crate) use host_readback::checked_extra_host_scratch;
pub use host_readback::PreparedVmHostInputs;
mod provider_observation;
use provider_observation::FusionPlacementSelection;
mod admission;
pub(crate) use admission::binding_storage_bytes;
use admission::AdmittedBindings;
mod reduction;
mod reduction_arithmetic;
mod vm_authority;
pub use reduction::{
    execute_reduction, execute_reduction_with_shape, execute_reduction_with_shape_placed,
    execute_reduction_with_shape_placed_async, execute_reduction_with_shape_placed_async_vm,
    execute_reduction_with_shape_placed_async_vm_decline, CpuReadbackFailure,
    TypedReductionVmOutput,
};
use vm_authority::{VmFusionExecutionRight, VmFusionSelectionRight};
mod reduction_program;
pub use reduction_program::{
    resolve_reduction_program, ReductionDataInput, ResolvedReductionProgram,
};
mod image_output;
mod outputs;
// Staged image arithmetic remains held until exact numerical, resource, and
// source-store publication admission is complete.
mod elementwise;
mod explained_input;
mod explained_output;
pub(crate) mod image_stages;
#[cfg(feature = "wgpu")]
pub use image_stages::ImageVmOriginalRoute;
pub use image_stages::{
    ImageVmCheckpoint, ImageVmPreparation, ImageVmSelectedRoute, ImageVmSourceFallbackRoute,
    ImageVmStages,
};
#[cfg(test)]
mod ownership_tests;
pub(crate) use elementwise::execute_elementwise_native;
pub use elementwise::{
    elementwise_floating_admission, execute_elementwise, execute_elementwise_placed,
    execute_elementwise_placed_vm, execute_elementwise_placed_vm_decline,
    execute_elementwise_placed_vm_inputs, ElementwiseFloatingAdmission,
};
pub use observation::{FusionCpuObservation, FusionExecutionOutcome, FusionProviderCompletion};

struct PreparedInput {
    handle: GpuTensorHandle,
}

fn mark_fusion_output(handle: &mut GpuTensorHandle, inputs: &[Value]) -> Result<()> {
    mark_fusion_output_from_values(handle, inputs.iter())
}

fn mark_fusion_output_from_values<'a>(
    handle: &mut GpuTensorHandle,
    inputs: impl IntoIterator<Item = &'a Value>,
) -> Result<()> {
    let explicit = inputs.into_iter().any(|value| {
        matches!(value, Value::GpuTensor(input) if runmat_accelerate_api::handle_is_explicit(input))
    });
    let provenance = if explicit {
        runmat_accelerate_api::GpuHandleProvenance::Explicit
    } else {
        runmat_accelerate_api::GpuHandleProvenance::Automatic
    };
    *handle = handle.with_provenance(provenance)?;
    fusion_residency::mark(handle);
    Ok(())
}

pub struct FusionExecutionRequest<'a> {
    pub plan: &'a FusionGroupPlan,
    pub inputs: Vec<Value>,
    pub placement: Option<PlacementCorrelationId>,
    pub runtime: Option<runmat_runtime::context::RuntimeContext>,
}

/// The placement-aware executor may choose its shared-runtime continuation on
/// provider absence only for a wholly host-owned request. Direct execution
/// wrappers reject that outcome instead. A resident leaf (including a captured
/// or constant leaf) still requires its exact owner and must never be replayed
/// as an ordinary host input. This does not classify failures after provider
/// discovery as retryable.
fn admit_provider_availability<'p>(
    request: &FusionExecutionRequest<'_>,
    available: Option<&'p dyn AccelProvider>,
) -> ExecutionResult<FusionExecutionOutcome<&'p dyn AccelProvider>> {
    if let Some(provider) = available {
        return Ok(FusionExecutionOutcome::Executed(provider));
    }
    if request
        .inputs
        .iter()
        .chain(unbound_constant_values(request.plan))
        .any(runmat_runtime::dispatcher::value_contains_gpu)
    {
        return Err(anyhow!("no acceleration provider registered").into());
    }
    FusionPlacementObserver::new(request.placement)
        .provider_rejected(ProviderRejectionCode::ProviderUnavailable);
    Ok(FusionExecutionOutcome::SharedRuntime(
        FusionCpuObservation::new(None),
    ))
}

/// Source constants can redundantly describe an invocation input. Such a
/// binding is represented by the supplied value; only unbound constants are
/// independent residency and numeric-classification leaves.
fn unbound_constant_values(plan: &FusionGroupPlan) -> impl Iterator<Item = &Value> {
    plan.const_values
        .iter()
        .filter(move |(id, _)| !plan.inputs.contains(id))
        .map(|(_, value)| value)
        .chain(
            plan.constants
                .iter()
                .filter(|(slot, _)| **slot >= plan.inputs.len())
                .map(|(_, value)| value),
        )
}

pub struct ElementwiseExecutionResult {
    pub final_value: Value,
    pub materialized_stores: Vec<(FusionStoreMaterialization, Value)>,
}

pub struct ElementwiseVmExecutionResult {
    pub result: ElementwiseExecutionResult,
    pub completion: Option<FusionProviderCompletion>,
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
) -> Result<GpuTensorHandle> {
    reject_native_integer_fusion_values(std::slice::from_ref(value), &[])?;
    match value {
        Value::GpuTensor(handle) => Ok(handle.clone()),
        Value::Tensor(tensor) => {
            let data = tensor.materialize_f64();
            let view = HostTensorView {
                data: &data,
                shape: tensor.shape(),
            };
            let started = Instant::now();
            let handle = provider.upload(&view)?;
            observer.upload(started, host_f64_bytes(data.len()));
            Ok(handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)?)
        }
        _ => Err(anyhow!("fusion: expected tensor input")),
    }
}

/// Preserve native single storage for image normalization. The general
/// elementwise upload path intentionally materializes a double computation
/// buffer; a matched `single` chain cannot use that representation.
fn ensure_image_normalize_input(
    provider: &dyn AccelProvider,
    value: &Value,
    observer: FusionPlacementObserver,
) -> ExecutionResult<GpuTensorHandle> {
    match value {
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => {
            use runmat_accelerate_api::{
                upload_numeric_protecting, GpuTensorStorage, HostNumericDataView,
                HostNumericTensorView,
            };
            let data = tensor
                .as_f32_slice()
                .ok_or_else(|| anyhow!("image normalize: native single storage unavailable"))?;
            let started = Instant::now();
            let handle = upload_numeric_protecting(
                provider,
                &HostNumericTensorView {
                    data: HostNumericDataView::F32(data),
                    shape: tensor.shape(),
                    storage: GpuTensorStorage::Real,
                },
                &[],
            )
            .map_err(ProviderExecutionError::from_source)?;
            observer.upload(started, u64::try_from(std::mem::size_of_val(data)).ok());
            handle
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .map_err(ProviderExecutionError::from_source)
        }
        _ => ensure_gpu_tensor(provider, value, observer).map_err(Into::into),
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
            Value::GpuTensor(handle) => Some(handle.shape().iter().copied().product::<usize>()),
            Value::Num(_) => Some(1),
            _ => None,
        })
        .max()
}

struct FusionProviderContract<'a> {
    elementwise_abi: Option<runmat_accelerate_api::ProviderFusedElementwiseWorkload>,
    /// Physical data operands only; logical controls remain in source identity.
    input_indices: Option<&'a [usize]>,
    outputs: Vec<runmat_accelerate_api::ProviderRepresentation>,
    semantic_identity: Option<runmat_execution::Digest>,
    scalar_work_units: Option<u64>,
    /// Original-class host work, independent of emitted provider helpers.
    host_scalar_work_units: Option<u64>,
    matmul: Option<matmul::workload::Selection<'a>>,
    preparation_program: Option<runmat_accelerate_api::ProviderPreparationProgram<'a>>,
    shader_requirements: Option<ProviderShaderRequirements>,
}

fn select_fusion_provider(
    request: &FusionExecutionRequest<'_>,
    bindings: &AdmittedBindings,
    provider: &dyn AccelProvider,
    output_shapes: &[Vec<usize>],
    shader_requirements: Option<ProviderShaderRequirements>,
) -> ExecutionResult<FusionExecutionOutcome<FusionPlacementSelection>> {
    select_fusion_provider_vm_decline(
        request,
        bindings,
        provider,
        output_shapes,
        shader_requirements,
        None,
    )
}

fn select_fusion_provider_vm_decline(
    request: &FusionExecutionRequest<'_>,
    bindings: &AdmittedBindings,
    provider: &dyn AccelProvider,
    output_shapes: &[Vec<usize>],
    shader_requirements: Option<ProviderShaderRequirements>,
    vm_decline: Option<&runmat_runtime::context::VmFusionDeclineRight>,
) -> ExecutionResult<FusionExecutionOutcome<FusionPlacementSelection>> {
    select_fusion_provider_with_contract_vm_cpu(
        request,
        bindings,
        provider,
        FusionProviderContract {
            elementwise_abi: None,
            input_indices: None,
            outputs: output_shapes
                .iter()
                .map(|shape| dense_output_representation(provider.precision(), shape))
                .collect(),
            semantic_identity: None,
            scalar_work_units: None,
            host_scalar_work_units: None,
            matmul: None,
            preparation_program: None,
            shader_requirements,
        },
        vm_decline.map(VmFusionSelectionRight::Decline),
        host_readback::CpuHostPlacement::Legacy,
    )
}

fn select_fusion_provider_with_contract(
    request: &FusionExecutionRequest<'_>,
    bindings: &AdmittedBindings,
    provider: &dyn AccelProvider,
    contract: FusionProviderContract<'_>,
) -> ExecutionResult<FusionExecutionOutcome<FusionPlacementSelection>> {
    select_fusion_provider_with_contract_vm(request, bindings, provider, contract, None)
}

fn select_fusion_provider_with_contract_vm(
    request: &FusionExecutionRequest<'_>,
    bindings: &AdmittedBindings,
    provider: &dyn AccelProvider,
    contract: FusionProviderContract<'_>,
    vm_right: Option<&runmat_runtime::context::VmFusionCompletionRight>,
) -> ExecutionResult<FusionExecutionOutcome<FusionPlacementSelection>> {
    select_fusion_provider_with_contract_vm_cpu(
        request,
        bindings,
        provider,
        contract,
        vm_right.map(VmFusionSelectionRight::Completion),
        host_readback::CpuHostPlacement::Legacy,
    )
}

fn select_fusion_provider_with_contract_vm_cpu(
    request: &FusionExecutionRequest<'_>,
    bindings: &AdmittedBindings,
    provider: &dyn AccelProvider,
    contract: FusionProviderContract<'_>,
    vm_selection: Option<VmFusionSelectionRight<'_>>,
    host: host_readback::CpuHostPlacement<'_>,
) -> ExecutionResult<FusionExecutionOutcome<FusionPlacementSelection>> {
    select_fusion_provider_with_implementation(
        request,
        bindings,
        provider,
        contract,
        vm_selection,
        host,
        crate::placement::LocalProviderAdmission::Available,
    )
}

fn select_fusion_provider_with_implementation(
    request: &FusionExecutionRequest<'_>,
    bindings: &AdmittedBindings,
    provider: &dyn AccelProvider,
    contract: FusionProviderContract<'_>,
    vm_selection: Option<VmFusionSelectionRight<'_>>,
    host: host_readback::CpuHostPlacement<'_>,
    implementation: crate::placement::LocalProviderAdmission,
) -> ExecutionResult<FusionExecutionOutcome<FusionPlacementSelection>> {
    let (vm_right, vm_decline) = match vm_selection {
        Some(VmFusionSelectionRight::Completion(right)) => (Some(right), None),
        Some(VmFusionSelectionRight::Decline(right)) => (None, Some(right)),
        None => (None, None),
    };
    let vm_decline = vm_decline
        .and_then(|right| VmFusionSelectionRight::Decline(right).certified_decline(request));
    let FusionProviderContract {
        elementwise_abi,
        input_indices,
        outputs,
        semantic_identity,
        scalar_work_units,
        host_scalar_work_units,
        matmul,
        preparation_program,
        shader_requirements,
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
    let input_refs = if let Some(contract) = &matmul {
        contract.inputs.clone()
    } else if let Some(indices) = input_indices {
        indices
            .iter()
            .map(|&index| {
                request
                    .inputs
                    .get(index)
                    .ok_or_else(|| anyhow!("fusion physical input projection is out of bounds"))
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        request.inputs.iter().collect::<Vec<_>>()
    };
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
        let width = match elementwise_abi.map_or(provider.precision(), |work| work.abi.precision())
        {
            ProviderPrecision::F32 => 4u64,
            ProviderPrecision::F64 => 8u64,
        };
        Some(
            input_refs
                .iter()
                .copied()
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
    if request.placement.is_none()
        && !matches!(host, host_readback::CpuHostPlacement::Native { .. })
    {
        if matches!(
            implementation,
            crate::placement::LocalProviderAdmission::ImplementationUnavailable
        ) {
            observer.provider_rejected(ProviderRejectionCode::UnsupportedOperation);
            return Ok(FusionExecutionOutcome::SharedRuntime(
                FusionCpuObservation::new(None),
            ));
        }
        if shader_requirements
            .is_some_and(|required| !provider.supports_shader_requirements(required))
        {
            observer.provider_rejected(ProviderRejectionCode::UnsupportedElementType);
            return Ok(FusionExecutionOutcome::SharedRuntime(
                FusionCpuObservation::new(None),
            ));
        }
        let _ = observer.select_provider(
            provider,
            &request.plan.group.kind,
            &input_refs,
            &outputs,
            ProviderWorkload {
                fused_elementwise: elementwise_abi,
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
    let host_available = match host {
        host_readback::CpuHostPlacement::Native { .. } => true,
        host_readback::CpuHostPlacement::Readbacks(_) => true,
        host_readback::CpuHostPlacement::Unavailable => false,
        host_readback::CpuHostPlacement::Legacy => request
            .inputs
            .iter()
            .chain(unbound_constant_values(request.plan))
            .all(|value| !runmat_runtime::dispatcher::value_contains_gpu(value)),
    };
    let host_units = if let Some(contract) = &matmul {
        // Original controls are checked at double precision, independently of
        // a provider's optional narrower implementation.
        crate::placement::matmul_scalar_work_units(&contract.workload, ProviderPrecision::F64)
    } else if typed_elementwise {
        host_scalar_work_units
    } else {
        // Non-scalar legacy domains retain their existing element prior.
        Some(element_count)
    };
    let host_execution_ns = host_units.and_then(|units| units.checked_mul(100));
    if host_available
        && host_execution_ns.is_none()
        && !matches!(host, host_readback::CpuHostPlacement::Native { .. })
    {
        // Ordinary execution remains valid without a comparative cost proof.
        // The caller retains any admitted private-readback continuation. Do
        // not publish a fabricated estimate or timing feedback for this route.
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    }
    let execution_units = scalar_work_units.unwrap_or(element_count);
    let cpu_cost = ExecutionCostEstimate {
        components: ExecutionCostComponents {
            // With no host admission this descriptor is excluded from the
            // candidate graph; its carrier is required by the local API.
            execution_ns: host_execution_ns.map_or(0, |ns| ns.max(100)),
            ..ExecutionCostComponents::default()
        },
        scratch_bytes: Some(0),
        confidence: EstimateConfidence::Low,
        source: EstimateSource::StaticPrior,
    };
    let mut cpu = ExecutionCandidateDescriptor {
        identity: format!("cpu.shared.{operation}"),
        region: None,
        kind: ExecutionCandidateKind::SharedRuntime,
        execution_location: CandidateExecutionLocation::Host,
        preparation: CandidatePreparationState::Ready,
        cost: cpu_cost,
        output_residency: CandidateOutputResidency::Host,
        presentation: runmat_execution::CandidateOutputPresentation::Direct,
        guards: Vec::new(),
    };
    let provider_fallback = ExecutionCostEstimate {
        components: ExecutionCostComponents {
            compile_or_prepare_ns: 20_000,
            allocation_ns: 5_000,
            queue_ns: 10_000,
            execution_ns: execution_units
                .checked_mul(10)
                .ok_or_else(|| anyhow!("provider fusion execution prior overflows"))?
                .max(100),
            ..ExecutionCostComponents::default()
        },
        scratch_bytes: Some(0),
        confidence: EstimateConfidence::Prior,
        source: EstimateSource::StaticPrior,
    };
    let (cpu, cpu_admission) = match host {
        host_readback::CpuHostPlacement::Native {
            cpu: native_cpu,
            right,
            source_host,
        } => crate::placement::LocalCpuCandidate::native_region(
            native_cpu,
            provider,
            right,
            source_host,
        ),
        _ if host_execution_ns.is_none() => {
            (cpu.into(), crate::placement::LocalCpuAdmission::Unavailable)
        }
        host_readback::CpuHostPlacement::Legacy => {
            (cpu.into(), crate::placement::LocalCpuAdmission::Legacy)
        }
        host_readback::CpuHostPlacement::Unavailable => {
            (cpu.into(), crate::placement::LocalCpuAdmission::Unavailable)
        }
        host_readback::CpuHostPlacement::Readbacks(exact) => {
            cpu.presentation = exact.presentation.clone();
            cpu.cost.scratch_bytes = Some(exact.scratch_bytes);
            (
                cpu.into(),
                crate::placement::LocalCpuAdmission::ExactReadbacks {
                    resources: Box::new(exact.resources),
                    download_bytes: exact.download_bytes,
                    readback_count: exact.readback_count,
                },
            )
        }
    };
    let host_unavailable = matches!(
        cpu_admission,
        crate::placement::LocalCpuAdmission::Unavailable
    ) && !matches!(host, host_readback::CpuHostPlacement::Native { .. });
    match crate::placement::plan_local_vm_cpu_with_provider(
        provider,
        LocalPlacementRequest {
            compound: None,
            operation,
            semantic_identity,
            observation_contract: if matches!(host, host_readback::CpuHostPlacement::Native { .. })
            {
                crate::placement::TotalCandidateObservationContract::NativeFullCall
            } else if vm_right.is_some() {
                crate::placement::TotalCandidateObservationContract::FullCall
            } else {
                crate::placement::TotalCandidateObservationContract::PreparationBodyCleanup
            },
            prepared_authority: None,
            source_decline: crate::placement::LocalSourceDecline::None,
            family: ProviderOperationFamily::Fusion,
            provider_kind: ExecutionCandidateKind::ProviderFusion,
            inputs: &input_refs,
            outputs,
            workload: ProviderWorkload {
                gpu_random: None,
                gpu_integer: None,
                gpu_integer_batch: None,
                stochastic_evolution: None,
                fused_elementwise: elementwise_abi,
                real_power_mode: None,
                real_power_scalar: None,
                preparation_route: None,
                matmul: matmul.as_ref().map(|contract| contract.workload.clone()),
                typed_reduction: None,
                sum: None,
                extrema: None,
                image_stages: None,
                convolution: None,
                iir: None,
                fspecial: None,
                imfilter: None,
                transfer_output_bounds: None,
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
            shader_requirements,
            cpu,
            provider_fallback: runmat_execution::CandidateCostEvidence::Estimated(
                provider_fallback,
            ),
            required_upload_bytes,
            required_download_bytes: 0,
            downstream_materialization: false,
            compare_profitability: true,
            runtime: request.runtime.as_ref(),
        },
        PlacementPolicy::default(),
        vm_right,
        vm_decline,
        cpu_admission,
        implementation,
    )
    .map_err(|error| ProviderExecutionError::from_source(*error))?
    {
        LocalPlacementOutcome::ProviderRejected { code, feedback, .. } => {
            observer.provider_rejected(code);
            if host_unavailable {
                if vm_right.is_some()
                    || vm_decline.is_some_and(|right| {
                        request
                            .runtime
                            .as_ref()
                            .is_some_and(|runtime| right.is_current(runtime))
                    })
                {
                    // No fused provider or whole-group host route exists, but
                    // the VM still owns the original ordered instructions.
                    // Decline before effects and without a CPU FullCall claim.
                    return Ok(FusionExecutionOutcome::SharedRuntime(
                        FusionCpuObservation::new(None),
                    ));
                }
                return Err(anyhow!("fusion has neither an admitted provider route nor a complete host readback bound").into());
            }
            Ok(FusionExecutionOutcome::SharedRuntime(
                FusionCpuObservation::new(feedback),
            ))
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
                provider_cost.and_then(|cost| cost.checked_total_ns()),
            );
            let mut ticket = if vm_right.is_some() {
                FusionPlacementSelection::new_vm(
                    observer,
                    feedback.map(|feedback| *feedback),
                    provider,
                    provider_observation.then_some(provider_query.as_ref()),
                )
            } else {
                FusionPlacementSelection::new(
                    observer,
                    feedback.map(|feedback| *feedback),
                    provider,
                    provider_observation.then_some(provider_query.as_ref()),
                )
            };
            if vm_right.is_some() {
                ticket.require_vm_provider_execution(
                    provider.device_id(),
                    provider_query.operation.clone(),
                );
            }
            Ok(FusionExecutionOutcome::Executed(ticket))
        }
        LocalPlacementOutcome::Selected {
            selected,
            cpu_cost,
            feedback,
            ..
        } => {
            observer.provider_candidate();
            observer.selected(
                PlacementVariant::SharedRuntime,
                "complete_cost",
                cpu_cost.estimate().and_then(|cost| cost.checked_total_ns()),
            );
            let native_kind = match selected.kind {
                ExecutionCandidateKind::SpecializedNativeCpu if matches!(host, host_readback::CpuHostPlacement::Native { source_host: Some(_), .. }) => Some(runmat_runtime::numeric_region::service::NativeNumericRegionCpuKind::Generic),
                ExecutionCandidateKind::VectorizedNativeCpu if matches!(host, host_readback::CpuHostPlacement::Native { .. }) => Some(runmat_runtime::numeric_region::service::NativeNumericRegionCpuKind::Vectorized),
                _ => None,
            };
            Ok(FusionExecutionOutcome::SharedRuntime(
                FusionCpuObservation::native_selected(
                    feedback,
                    native_kind,
                    (selected.kind == ExecutionCandidateKind::OriginalNativeRegion)
                        .then(runmat_accelerate_api::provider)
                        .flatten()
                        .filter(|selected| {
                            runmat_accelerate_api::same_provider_instance(*selected, provider)
                        }),
                ),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NativeIntegerFusionValueKind {
    Scalar,
    Tensor,
    ResidentTensor,
}

impl std::fmt::Display for NativeIntegerFusionValueKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Scalar => "integer scalar",
            Self::Tensor => "integer tensor",
            Self::ResidentTensor => "native integer gpuArray",
        })
    }
}

fn native_integer_fusion_value_kind(value: &Value) -> Option<NativeIntegerFusionValueKind> {
    match value {
        Value::Int(_) => Some(NativeIntegerFusionValueKind::Scalar),
        Value::Tensor(t) if t.integer_storage().is_some() => {
            Some(NativeIntegerFusionValueKind::Tensor)
        }
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_integer_type(handle).is_some() =>
        {
            Some(NativeIntegerFusionValueKind::ResidentTensor)
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
    let constants: Vec<&Value> = unbound_constant_values(request.plan).collect();
    reject_native_integer_fusion_values(&request.inputs, &constants)
}

fn scalar_from_value(value: &Value, observer: FusionPlacementObserver) -> ExecutionResult<f64> {
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
                )
                .into())
            }
        }
        Value::GpuTensor(_) => {
            let started = Instant::now();
            let gathered = gather_if_needed(value).map_err(ProviderExecutionError::from_source)?;
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
        )
        .into()),
    }
}

fn resolve_image_scalar_value(
    scalar: &ImageScalar,
    plan: &FusionGroupPlan,
    request: &FusionExecutionRequest<'_>,
    observer: FusionPlacementObserver,
) -> ExecutionResult<f64> {
    match scalar {
        ImageScalar::Constant(v) => Ok(*v),
        ImageScalar::Value(vid) => {
            if let Some(idx) = plan.inputs.iter().position(|id| *id == *vid) {
                let runtime_value = request
                    .inputs
                    .get(idx)
                    .ok_or_else(|| anyhow!("image normalize: runtime scalar missing"))?;
                return scalar_from_value(runtime_value, observer);
            }
            if let Some(value) = plan.const_values.get(vid) {
                return scalar_from_value(value, observer);
            }
            Err(anyhow!(
                "image normalize: scalar input {:?} not materialized in plan",
                vid
            )
            .into())
        }
    }
}

fn runtime_broadcast_shape(values: &[Value]) -> Option<Vec<usize>> {
    let mut shapes: Vec<Vec<usize>> = Vec::new();
    for value in values {
        match value {
            Value::GpuTensor(handle) => shapes.push(handle.shape().to_vec()),
            Value::Tensor(tensor) => shapes.push(tensor.shape().to_vec()),
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

/// Direct provider execution retaining the established value-returning API.
/// Executors that can run shared bytecode use the placement-aware entrypoint.
pub async fn execute_power_step_normalize(
    request: FusionExecutionRequest<'_>,
) -> ExecutionResult<Value> {
    match execute_power_step_normalize_placed(request).await? {
        FusionExecutionOutcome::Executed(value) => Ok(value),
        FusionExecutionOutcome::SharedRuntime(_) => {
            Err(anyhow!("direct fusion request selected shared runtime").into())
        }
    }
}

pub async fn execute_power_step_normalize_placed(
    request: FusionExecutionRequest<'_>,
) -> ExecutionResult<FusionExecutionOutcome<Value>> {
    let Some(bindings) = AdmittedBindings::new(&request) else {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    };
    crate::ensure_residency_hooks();
    reject_native_integer_fusion_inputs(&request)?;
    if request.plan.group.kind != FusionKind::PowerStepNormalize {
        return Err(anyhow!("unsupported fusion kind").into());
    }
    let provider = match admit_provider_availability(&request, provider())? {
        FusionExecutionOutcome::Executed(provider) => provider,
        FusionExecutionOutcome::SharedRuntime(observation) => {
            return Ok(FusionExecutionOutcome::SharedRuntime(observation));
        }
    };
    let selection = select_fusion_provider(&request, &bindings, provider, &[], None)?;
    let selection = match selection {
        FusionExecutionOutcome::Executed(selection) => selection,
        FusionExecutionOutcome::SharedRuntime(observation) => {
            return Ok(FusionExecutionOutcome::SharedRuntime(observation))
        }
    };
    let observer = selection.observer;
    let (lhs_vid, rhs_vid, epsilon) = match request.plan.pattern.as_ref() {
        Some(FusionPattern::PowerStepNormalize { lhs, rhs, epsilon }) => (*lhs, *rhs, *epsilon),
        _ => return Err(anyhow!("power-step normalization: missing pattern metadata").into()),
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

    let lhs_handle = ensure_gpu_tensor(provider, lhs_value, observer)?;
    let rhs_handle = ensure_gpu_tensor(provider, rhs_value, observer)?;

    let desc = PowerStepEpilogue { epsilon };
    observer.queue_unmeasured();
    let mut output = provider
        .matmul_power_step(&lhs_handle, &rhs_handle, &desc)
        .await?;
    observer.kernel_unmeasured();

    drop((lhs_handle, rhs_handle));

    mark_fusion_output(&mut output, &request.inputs)?;
    selection.submitted();
    Ok(FusionExecutionOutcome::Executed(Value::GpuTensor(output)))
}

/// Direct provider execution retaining the established value-returning API.
/// Executors that can run shared bytecode use the placement-aware entrypoint.
pub async fn execute_explained_variance(
    request: FusionExecutionRequest<'_>,
) -> runmat_accelerate_api::ProviderExecutionResult<Value> {
    match execute_explained_variance_placed(request).await? {
        FusionExecutionOutcome::Executed(value) => Ok(value),
        FusionExecutionOutcome::SharedRuntime(_) => {
            Err(anyhow!("direct fusion request selected shared runtime").into())
        }
    }
}

pub async fn execute_explained_variance_placed(
    request: FusionExecutionRequest<'_>,
) -> runmat_accelerate_api::ProviderExecutionResult<FusionExecutionOutcome<Value>> {
    let Some(bindings) = AdmittedBindings::new(&request) else {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    };
    crate::ensure_residency_hooks();
    reject_native_integer_fusion_inputs(&request)?;
    if request.plan.group.kind != FusionKind::ExplainedVariance {
        return Err(anyhow!("unsupported fusion kind").into());
    }
    let provider = match admit_provider_availability(&request, provider())? {
        FusionExecutionOutcome::Executed(provider) => provider,
        FusionExecutionOutcome::SharedRuntime(observation) => {
            return Ok(FusionExecutionOutcome::SharedRuntime(observation));
        }
    };
    let selection = select_fusion_provider(&request, &bindings, provider, &[], None)?;
    let selection = match selection {
        FusionExecutionOutcome::Executed(selection) => selection,
        FusionExecutionOutcome::SharedRuntime(observation) => {
            return Ok(FusionExecutionOutcome::SharedRuntime(observation))
        }
    };
    let observer = selection.observer;
    let (q_vid, g_vid) = match request.plan.pattern.as_ref() {
        Some(FusionPattern::ExplainedVariance { q, g }) => (*q, *g),
        _ => return Err(anyhow!("explained variance: missing pattern metadata").into()),
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

    let mut protected: Vec<_> = request
        .inputs
        .iter()
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle),
            _ => None,
        })
        .collect();
    let started = Instant::now();
    let q_handle = explained_input::prepare(provider, q_value, &protected)?;
    if matches!(q_value, Value::Tensor(_)) {
        observer.upload(started, Some(q_handle.physical().payload_bytes()));
    }
    protected.push(&q_handle);
    let started = Instant::now();
    let g_handle = explained_input::prepare(provider, g_value, &protected)?;
    if matches!(g_value, Value::Tensor(_)) {
        observer.upload(started, Some(g_handle.physical().payload_bytes()));
    }
    drop(protected);

    let debug_explained = std::env::var("RUNMAT_DEBUG_EXPLAINED").is_ok();
    if debug_explained {
        println!(
            "[explained] initial Q shape {:?}, G shape {:?}",
            q_handle.shape(),
            g_handle.shape()
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

    let q_shape = q_handle.shape().to_vec();
    if q_shape.len() != 2 {
        return Err(anyhow!("explained variance: Q must be 2-D").into());
    }
    let q_rows = q_shape[0];
    let q_cols = q_shape[1];
    if q_rows == 0 || q_cols == 0 {
        return Err(anyhow!("explained variance: zero-sized Q").into());
    }

    let g_shape = g_handle.shape().to_vec();
    if g_shape.len() != 2 {
        return Err(anyhow!("explained variance: G must be 2-D").into());
    }
    if g_shape[0] != q_rows || g_shape[1] != q_rows {
        return Err(anyhow!("explained variance: G shape mismatch").into());
    }

    let q_element = q_handle.physical().element_type();
    let product_element = if q_element == runmat_accelerate_api::NumericElementType::F32
        || g_handle.physical().element_type() == runmat_accelerate_api::NumericElementType::F32
    {
        runmat_accelerate_api::NumericElementType::F32
    } else {
        runmat_accelerate_api::NumericElementType::F64
    };
    // Transpose may lawfully retain Q's allocation. No operation may adopt G's
    // distinct allocation as Q's transpose.
    let transpose_protected: Vec<_> = (!q_handle.shares_allocation_with(&g_handle))
        .then_some(&g_handle)
        .into_iter()
        .collect();
    let q_transposed_view = explained_output::admit(
        provider.transpose(&q_handle)?,
        provider,
        &[q_cols, q_rows],
        q_element,
        &transpose_protected,
    )?;

    observer.queue_unmeasured();
    let tmp = explained_output::admit(
        provider.matmul(&q_transposed_view, &g_handle).await?,
        provider,
        &[q_cols, q_rows],
        product_element,
        &[&q_handle, &g_handle, &q_transposed_view],
    )?;
    observer.kernel_unmeasured();

    if debug_explained {
        println!(
            "[explained] after transpose(Q)*G tmp shape {:?}",
            tmp.shape()
        );
    }

    observer.queue_unmeasured();
    let product = explained_output::admit(
        provider.matmul(&tmp, &q_handle).await?,
        provider,
        &[q_cols, q_cols],
        product_element,
        &[&q_handle, &g_handle, &q_transposed_view, &tmp],
    )?;
    observer.kernel_unmeasured();

    if debug_explained {
        println!("[explained] product shape {:?}", product.shape());
    }

    let started = Instant::now();
    let diag = provider.diag_extract(&product, 0)?;
    observer.queue(started);
    observer.kernel_unmeasured();
    let diag = match diag.shape() {
        [len] if *len == q_cols => provider.reshape(&diag, &[*len, 1])?,
        [len, 1] if *len == q_cols => diag,
        _ => {
            return Err(outputs::reject_outputs(
                vec![diag],
                "explained variance: invalid diagonal shape",
            ))
        }
    };
    let mut diag = explained_output::admit(
        diag,
        provider,
        &[q_cols, 1],
        product_element,
        &[&q_handle, &g_handle, &q_transposed_view, &tmp, &product],
    )?;

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

    drop((tmp, product, q_handle, g_handle, q_transposed_view));

    mark_fusion_output(&mut diag, &request.inputs)?;
    selection.submitted();
    Ok(FusionExecutionOutcome::Executed(Value::GpuTensor(diag)))
}

/// Direct provider execution retaining the established value-returning API.
/// Executors that can run shared bytecode use the placement-aware entrypoint.
pub async fn execute_image_normalize(
    request: FusionExecutionRequest<'_>,
) -> ExecutionResult<Value> {
    match execute_image_normalize_placed(request).await? {
        FusionExecutionOutcome::Executed(value) => Ok(value),
        FusionExecutionOutcome::SharedRuntime(_) => {
            Err(anyhow!("direct fusion request selected shared runtime").into())
        }
    }
}

pub async fn execute_image_normalize_placed(
    request: FusionExecutionRequest<'_>,
) -> ExecutionResult<FusionExecutionOutcome<Value>> {
    execute_image_normalize_vm(request, None).await
}

/// The VM may decline an unselected image optimization before provider
/// effects. This route carries no host execution or FullCall completion right.
pub async fn execute_image_normalize_vm(
    request: FusionExecutionRequest<'_>,
    decline: Option<&runmat_runtime::context::VmFusionDeclineRight>,
) -> ExecutionResult<FusionExecutionOutcome<Value>> {
    let Some(bindings) = AdmittedBindings::new(&request) else {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    };
    crate::ensure_residency_hooks();
    reject_native_integer_fusion_inputs(&request)?;
    if request.plan.group.kind != FusionKind::ImageNormalize {
        return Err(anyhow!("unsupported fusion kind").into());
    }
    let pattern = match request.plan.pattern.as_ref() {
        Some(FusionPattern::ImageNormalize(pattern)) => pattern,
        _ => return Err(anyhow!("image normalize: missing pattern metadata").into()),
    };
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
    // Bound folded controls are never substituted from plan.const_values.
    // An absent or changed live request operand declines before provider work.
    if !crate::fusion::image_reduction::bound_image_controls_match(pattern, |id| {
        request
            .plan
            .inputs
            .iter()
            .position(|candidate| *candidate == id)
            .and_then(|position| request.inputs.get(position))
    }) {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    }
    let input_value = find_value(pattern.input)?;
    let input_precision = match input_value {
        Value::Tensor(tensor) => match tensor.numeric_dtype() {
            NumericDType::F32 => Some(ProviderPrecision::F32),
            NumericDType::F64 => Some(ProviderPrecision::F64),
            _ => None,
        },
        Value::GpuTensor(handle)
            if handle.physical().storage() == runmat_accelerate_api::GpuTensorStorage::Real
                && !runmat_accelerate_api::handle_is_logical(handle) =>
        {
            match handle.physical().element_type() {
                runmat_accelerate_api::NumericElementType::F32 => Some(ProviderPrecision::F32),
                runmat_accelerate_api::NumericElementType::F64 => Some(ProviderPrecision::F64),
                _ => None,
            }
        }
        _ => None,
    };
    let Some(input_precision) = input_precision else {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    };
    if pattern.requires_single_input && input_precision != ProviderPrecision::F32 {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    }
    let output_shape = match input_value {
        Value::Tensor(tensor) => tensor.shape(),
        Value::GpuTensor(handle) => handle.shape(),
        _ => unreachable!("image input precision admission excludes other values"),
    };
    // The matcher may see Unknown intermediate shapes and still recognize the
    // source pattern. Only an actual three-axis image can enter placement or
    // stage a provider input; all other ranks resume ordinary execution.
    if output_shape.len() != 3 {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    }
    let provider = match admit_provider_availability(&request, provider())? {
        FusionExecutionOutcome::Executed(provider) => provider,
        FusionExecutionOutcome::SharedRuntime(observation) => {
            return Ok(FusionExecutionOutcome::SharedRuntime(observation));
        }
    };
    // A resident double allocation cannot be reinterpreted by a provider's
    // single shader. Host doubles, however, retain the existing general
    // upload route, which may lower them to the provider's preferred single
    // storage before this operation.
    if matches!(input_value, Value::GpuTensor(_))
        && input_precision == ProviderPrecision::F64
        && provider.precision() == ProviderPrecision::F32
    {
        return Ok(FusionExecutionOutcome::SharedRuntime(
            FusionCpuObservation::new(None),
        ));
    }
    let output_precision = match input_value {
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F64 => {
            provider.precision()
        }
        _ => input_precision,
    };
    let selection = select_fusion_provider_with_implementation(
        &request,
        &bindings,
        provider,
        FusionProviderContract {
            elementwise_abi: None,
            input_indices: None,
            outputs: vec![dense_output_representation(output_precision, output_shape)],
            semantic_identity: None,
            scalar_work_units: None,
            host_scalar_work_units: None,
            matmul: None,
            preparation_program: None,
            shader_requirements: None,
        },
        decline.map(VmFusionSelectionRight::Decline),
        host_readback::CpuHostPlacement::Legacy,
        crate::placement::LocalProviderAdmission::Available,
    )?;
    let selection = match selection {
        FusionExecutionOutcome::Executed(selection) => selection,
        FusionExecutionOutcome::SharedRuntime(observation) => {
            return Ok(FusionExecutionOutcome::SharedRuntime(observation))
        }
    };
    let observer = selection.observer;
    if log::log_enabled!(log::Level::Debug) {
        log::debug!(
            "execute_image_normalize: plan inputs={:?} stack={:?}",
            request.plan.inputs,
            request.plan.stack_pattern
        );
    }

    let input_handle = ensure_image_normalize_input(provider, input_value, observer)?;
    let shape = input_handle.shape().to_vec();
    if shape.len() != 3 {
        return Err(anyhow!(
            "image normalize: expected 3-D input tensor, got shape {:?}",
            shape
        )
        .into());
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
    let output = provider.image_normalize(&input_handle, &desc).await?;
    observer.kernel_unmeasured();
    let mut output = image_output::admit(output, provider, &input_handle)?;

    drop(input_handle);

    mark_fusion_output(&mut output, &request.inputs)?;
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
        ) -> ExecutionResult<Value> = execute_reduction_with_shape;
        let _: for<'a> fn(FusionExecutionRequest<'a>, usize, usize, u32) -> ExecutionResult<Value> =
            execute_reduction;
        fn check_elementwise_signature(request: FusionExecutionRequest<'_>) {
            fn require_future(
                _: impl std::future::Future<
                    Output = runmat_accelerate_api::ProviderExecutionResult<
                        ElementwiseExecutionResult,
                    >,
                >,
            ) {
            }
            require_future(execute_elementwise(request));
        }
        let _: for<'a> fn(FusionExecutionRequest<'a>) = check_elementwise_signature;

        fn require_value_future(_: impl std::future::Future<Output = ExecutionResult<Value>>) {}
        fn check_async_signatures(request: FusionExecutionRequest<'_>, which: usize) {
            match which {
                0 => {
                    fn require_local_future(
                        _: impl std::future::Future<
                            Output = runmat_accelerate_api::ProviderExecutionResult<Value>,
                        >,
                    ) {
                    }
                    require_local_future(execute_explained_variance(request));
                }
                1 => require_value_future(execute_power_step_normalize(request)),
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
        let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
        let allocate = || {
            owner
                .allocate_zeroed(
                    vec![1, 1],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .unwrap()
        };
        let input = allocate();
        let automatic = allocate();
        let explicit = allocate();
        let explicit_input = input
            .handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .unwrap();
        let mut automatic_output = automatic.handle;
        let mut explicit_output = explicit.handle;

        mark_fusion_output(&mut automatic_output, &[Value::Num(1.0)]).unwrap();
        mark_fusion_output(
            &mut explicit_output,
            &[Value::GpuTensor(explicit_input.clone())],
        )
        .unwrap();

        assert_eq!(
            runmat_accelerate_api::handle_provenance(&automatic_output),
            Some(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        );
        assert!(runmat_accelerate_api::handle_is_explicit(&explicit_output));
        for handle in [&explicit_input, &automatic_output, &explicit_output] {
            handle.allocation_lease().validate_live().unwrap();
        }
        for probe in [&input.probe, &automatic.probe, &explicit.probe] {
            assert_eq!(probe.retirements(), 0);
        }
        drop((explicit_input, automatic_output, explicit_output));
        // with_provenance publishes an owned view; the original input view
        // remains live until its own reference is dropped.
        assert_eq!(input.probe.retirements(), 0);
        input.handle.allocation_lease().validate_live().unwrap();
        drop(input.handle);
        for probe in [&input.probe, &automatic.probe, &explicit.probe] {
            assert_eq!(probe.retirements(), 1);
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
