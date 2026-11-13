use anyhow::anyhow;
#[cfg(test)]
use once_cell::sync::Lazy;
use once_cell::sync::OnceCell;
use runmat_types::{standard, ClassIdentity};
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, Ordering};
#[cfg(feature = "wgpu")]
use std::sync::Arc;

mod cancellation;
mod numeric_download_stage;
mod numeric_readback;
mod platform;
pub use numeric_download_stage::{NumericDownloadStageLease, PreparedNumericDownloadStage};
pub use numeric_readback::{
    authorize_numeric_readback, numeric_readback_for_handle, NumericReadback, NumericReadbackFuture,
};
mod device_control;
mod provider_registry;
pub use device_control::{
    current_device_allocation_scope, device_control_provider_is_authorized,
    register_device_allocation_scope_resolver, validate_device_allocation_scope,
    DeviceAllocationAdmission, DeviceAllocationScope, ProviderDeviceControl,
    ProviderGpuDeviceFacts, StaleDeviceAllocationScope,
};
pub mod random;
pub mod stochastic_evolution;
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
pub use platform::AcceleratorThreadBound;
pub use provider_registry::same_provider_instance;
pub mod allocation;
pub use allocation::{
    GpuPhysicalDescriptor, GpuTensorHandle, GpuTensorLayout, GpuTensorView, GpuTensorWireDescriptor,
};
mod iir;
mod native_device;
pub use iir::{
    provider_iir_causal_query, provider_iir_state_shape, provider_iir_zero_phase_query,
    ProviderIirCoefficientSource, ProviderIirFilterOptions, ProviderIirFilterResult,
    ProviderIirIndexCancelled, ProviderIirZeroPhaseFault, ProviderIirZeroPhaseIndexBuilder,
    ProviderIirZeroPhaseIndexPlan, ProviderIirZeroPhaseRequest, ProviderPreparedIirCoefficients,
};
mod fusion_outputs;
mod logical_upload;
mod numeric_upload;
pub mod observation;
mod provider_execution;
mod residency;
mod view_plan;
pub use view_plan::{
    ProviderViewCapability, ProviderViewChain, ProviderViewOutcome, ProviderViewPlanError,
    ProviderViewSource, ProviderViewStep,
};
mod covariance;
pub use covariance::CovarianceWeightDomainError;
mod envelope;
pub use envelope::EnvelopeSignalDomainError;
mod deep_learning;
pub use deep_learning::{
    DeepLearningConv1DCapability, DeepLearningConv1DGradients, DeepLearningConv1DOperands,
    DeepLearningConv1DPaddingValue, DeepLearningConv1DRequest, DeepLearningFeedForwardCapability,
    DeepLearningFeedForwardStage, DeepLearningForwardBufferBound, DeepLearningForwardCapability,
    DeepLearningForwardGeometry, DeepLearningForwardOperation, DeepLearningForwardParameter,
    DeepLearningForwardParameters, DeepLearningForwardResources, DeepLearningForwardRoute,
    DeepLearningForwardRouteError, DeepLearningForwardSource, DeepLearningForwardStep,
    DeepLearningLayerNormCapability, DeepLearningLayerNormGradients, DeepLearningLayerNormRecorded,
    DeepLearningLayerNormRequest, DeepLearningPool1DCapability, DeepLearningPool1DRequest,
    DeepLearningPool1DResultAxis, DeepLearningPool1DRoute,
};
pub use provider_execution::{
    is_unsupported_provider_error, ProviderExecutionCompletionError, ProviderExecutionError,
    ProviderExecutionFuture, ProviderExecutionResult,
};
mod sequence;
pub use sequence::{ProviderSequenceKind, ProviderSequenceRequest};
mod placement;
mod sparse;

pub use cancellation::*;
mod power;
pub mod power_service;
mod resident_select;
pub use resident_select::ResidentSelectPlan;
/// Exact feasibility identity of the provider-owned resident selection hook.
pub const RESIDENT_SELECT_OPERATION_IDENTITY: &str = "provider.resident_select";
mod numerical_domain;
pub mod reduction;
pub use reduction::ProviderDoubleSumPreflight;
#[cfg(feature = "wgpu")]
pub mod native_wgpu_f64;
pub mod scalar_program;
#[cfg(feature = "wgpu")]
mod wgpu_access;
#[cfg(feature = "wgpu")]
mod wgpu_buffer_owner;
pub mod wgpu_ieee_words;
#[cfg(feature = "wgpu")]
pub use wgpu_access::{
    CanonicalWgpuEmissionAuthority, WgpuAccessError, WgpuCleanup, WgpuDeviceAccess, WgpuEmission,
    WgpuEmissionMode, WgpuNumericReadbackBuffer, WgpuNumericReadbackSource,
    WgpuNumericReadbackSubmission, WgpuNumericReadbackTimestampResources, WgpuScopeErrorKind,
    WgpuScopeErrors, WgpuScopeReceipt,
};
#[cfg(feature = "wgpu")]
pub use wgpu_buffer_owner::WgpuBufferOwner;

pub use fusion_outputs::{
    validate_fused_output_classes, ProviderFusedElementwiseAbi, ProviderFusedElementwiseRequest,
    ProviderFusedElementwiseWorkload,
};
pub use logical_upload::{
    upload_logical_protecting, upload_logical_protecting_admitted, HostLogicalTensorView,
    HostLogicalUploadScratch,
};
pub use native_device::*;
pub use numeric_upload::{
    class_preserving_upload_query, is_invalid_numeric_upload_result, upload_numeric_protecting,
    upload_numeric_protecting_admitted, upload_numeric_protecting_admitted_with_scratch,
    HostNumericUploadScratch, InvalidNumericUploadResult, NumericUploadCompletionError,
    NumericUploadError, NumericUploadResult,
};
mod resident_indexing;
pub use numerical_domain::{NumericalDomainEvidence, RoundDigitsMode};
pub use placement::*;
pub use power::{
    ExactSingleBasePower, PowerInputDescription, PowerResultDomain, PowerScalarControl,
    PowerScalarWorkload, RealPowerArithmeticMode, RealPowerOperand,
};
pub use residency::{clear_residency_checked, ResidencyCleanupError, ResidencyRestoreGuard};
pub use resident_indexing::ProviderScatterValueMapping;
pub use sparse::*;

type ResidencyMarkFn = fn(&GpuTensorHandle);
type ResidencyClearFn = fn(GpuHandleIdentity) -> Result<(), ResidencyCleanupError>;
type SequenceThresholdFn = fn() -> Option<usize>;
type AutomaticOffloadEnabledFn = fn() -> bool;
type WorkgroupSizeHintFn = fn() -> Option<u32>;
type ExecutionProviderAuthorizer =
    for<'a> fn(ExecutionProviderAdmission<'a>) -> ExecutionProviderAuthorization;

static RESIDENCY_MARK: OnceCell<ResidencyMarkFn> = OnceCell::new();
static RESIDENCY_CLEAR: OnceCell<ResidencyClearFn> = OnceCell::new();
static SEQUENCE_THRESHOLD_PROVIDER: OnceCell<SequenceThresholdFn> = OnceCell::new();
static AUTOMATIC_OFFLOAD_ENABLED_PROVIDER: OnceCell<AutomaticOffloadEnabledFn> = OnceCell::new();
static WORKGROUP_SIZE_HINT_PROVIDER: OnceCell<WorkgroupSizeHintFn> = OnceCell::new();
static EXECUTION_PROVIDER_AUTHORIZER: OnceCell<ExecutionProviderAuthorizer> = OnceCell::new();

/// Scheduler authorization state for one process-local acceleration provider.
/// Direct sessions are unrestricted. Scheduled invocations explicitly allow
/// or deny providers from their fenced device assignment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionProviderAuthorization {
    Unrestricted,
    Allowed,
    Denied,
}

/// The already-installed runtime policy observes both a selected provider and
/// effects that bypass its physical WGPU submission owner. An escape never
/// grants provider authorization.
pub enum ExecutionProviderAdmission<'a> {
    Provider(&'a dyn AccelProvider),
    /// Exact-source egress, not permission to create or compute device values.
    NumericReadback {
        provider: &'a dyn AccelProvider,
        handle: &'a GpuTensorHandle,
    },
    DeviceControl(&'a dyn AccelProvider),
    #[cfg(feature = "wgpu")]
    DeviceAccess(&'a WgpuDeviceAccess),
    DeviceEffect(ExecutionDeviceEffect),
    #[cfg(feature = "wgpu")]
    CanonicalWgpuSubmit(&'a WgpuDeviceAccess),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionDeviceEffect {
    RawWgpuExport,
    RawNativeExport,
    PlotDeviceEmission,
    ForeignExecution,
    /// `Queue::write_buffer` enqueues device transfer work outside measured
    /// encoder intervals, without introducing an uncounted kernel launch.
    ImplicitQueueTransfer,
}

/// Monotone negative observation at the effect's owning ingress. This uses
/// the same runtime callback as provider authorization, not a second registry.
pub fn observe_execution_device_effect(effect: ExecutionDeviceEffect) {
    #[cfg(feature = "wgpu")]
    observation::record_source_region_unattributed_device_effect();
    if let Some(authorizer) = EXECUTION_PROVIDER_AUTHORIZER.get() {
        let _ = authorizer(ExecutionProviderAdmission::DeviceEffect(effect));
    }
}

#[cfg(feature = "wgpu")]
fn observe_canonical_wgpu_submit(access: &WgpuDeviceAccess) {
    observation::record_source_region_wgpu_submission(access);
    if let Some(authorizer) = EXECUTION_PROVIDER_AUTHORIZER.get() {
        let _ = authorizer(ExecutionProviderAdmission::CanonicalWgpuSubmit(access));
    }
}

/// Installs the runtime-owned provider authorization policy. Provider
/// discovery remains in this crate; invocation and lease authority remain in
/// the runtime. The callback connects those domains without a dependency
/// cycle.
pub fn register_execution_provider_authorizer(authorizer: ExecutionProviderAuthorizer) {
    let _ = EXECUTION_PROVIDER_AUTHORIZER.set(authorizer);
}

pub type GpuHandleIdentity = allocation::TensorViewIdentity;

pub fn handle_identity(handle: &GpuTensorHandle) -> GpuHandleIdentity {
    handle.view_identity()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransposeInfo {
    pub base_rows: usize,
    pub base_cols: usize,
}

/// Register a callback used to mark residency tracking when GPU tensors are
/// created or returned by device-side execution paths.
pub fn register_residency_mark(handler: ResidencyMarkFn) {
    let _ = RESIDENCY_MARK.set(handler);
}

/// Mark residency metadata for the provided GPU tensor handle, if a backend
/// has registered a handler via [`register_residency_mark`].
pub fn mark_residency(handle: &GpuTensorHandle) {
    if let Some(handler) = RESIDENCY_MARK.get() {
        handler(handle);
    }
}

/// Register a callback used to clear residency tracking when GPU tensors are
/// gathered back to the host. Backends that maintain residency metadata should
/// install this hook during initialization.
pub fn register_residency_clear(handler: ResidencyClearFn) {
    let _ = RESIDENCY_CLEAR.set(handler);
}

/// Clear residency metadata for the provided GPU tensor handle, if a backend
/// has registered a handler via [`register_residency_clear`].
pub fn clear_residency(handle: &GpuTensorHandle) -> Result<(), ResidencyCleanupError> {
    clear_residency_checked(handle)
}

/// Register a callback that exposes the current sequence length threshold
/// derived from the auto-offload planner. Array constructors can use this hint
/// to decide when to prefer GPU residency automatically.
pub fn register_sequence_threshold_provider(provider: SequenceThresholdFn) {
    let _ = SEQUENCE_THRESHOLD_PROVIDER.set(provider);
}

/// Query the currently registered sequence threshold hint, if any.
pub fn sequence_threshold_hint() -> Option<usize> {
    SEQUENCE_THRESHOLD_PROVIDER
        .get()
        .and_then(|provider| provider())
}

/// Register the configured automatic-offload policy for implicit array
/// construction. Explicit resident inputs do not consult this policy.
pub fn register_automatic_offload_enabled_provider(provider: AutomaticOffloadEnabledFn) {
    let _ = AUTOMATIC_OFFLOAD_ENABLED_PROVIDER.set(provider);
}

/// Whether implicit array constructors may prefer a provider. An absent
/// policy hook preserves the legacy behavior for standalone providers.
pub fn automatic_offload_enabled() -> bool {
    AUTOMATIC_OFFLOAD_ENABLED_PROVIDER
        .get()
        .is_none_or(|provider| provider())
}

/// Register a callback that reports the calibrated workgroup size selected by
/// the active acceleration provider (if any). Plotting kernels can reuse this
/// hint to match backend tuning.
pub fn register_workgroup_size_hint_provider(provider: WorkgroupSizeHintFn) {
    let _ = WORKGROUP_SIZE_HINT_PROVIDER.set(provider);
}

/// Query the current workgroup size hint exposed by the provider.
pub fn workgroup_size_hint() -> Option<u32> {
    WORKGROUP_SIZE_HINT_PROVIDER
        .get()
        .and_then(|provider| provider())
}

/// Export a shared acceleration context (e.g., the active WGPU device) when the
/// current provider exposes one.
pub fn export_context(kind: AccelContextKind) -> Option<AccelContextHandle> {
    observe_execution_device_effect(ExecutionDeviceEffect::RawWgpuExport);
    provider().and_then(|p| p.export_context(kind))
}

/// Request a provider-owned WGPU buffer for zero-copy consumers. Returns `None`
/// when the exact owner does not expose this physical rendering form, before
/// allocation or dispatch. Operational failures retain their original cause.
#[cfg(feature = "wgpu")]
pub fn export_wgpu_buffer(
    handle: &GpuTensorHandle,
) -> ProviderExecutionResult<Option<WgpuBufferRef>> {
    observe_execution_device_effect(ExecutionDeviceEffect::RawWgpuExport);
    handle
        .allocation_lease()
        .validate_live()
        .map_err(ProviderExecutionError::from_source)?;
    let owner = provider_for_handle(handle)
        .ok_or_else(|| anyhow::anyhow!("no provider owns WGPU export input"))?;
    owner.export_wgpu_buffer(handle)
}

/// Floating precision is an immutable physical allocation fact.
pub fn handle_precision(handle: &GpuTensorHandle) -> Option<ProviderPrecision> {
    handle.physical().element_type().precision()
}

/// Underlying numeric class derives from physical storage and the admitted
/// logical interpretation. No mutable annotation can override either authority.
pub fn handle_class_identity(handle: &GpuTensorHandle) -> Option<ClassIdentity> {
    Some(if handle.view().is_logical() {
        standard::LOGICAL.owned()
    } else {
        handle.physical().element_type().class_identity()
    })
}

pub fn handle_is_logical(handle: &GpuTensorHandle) -> bool {
    handle.view().is_logical()
}

pub fn handle_transpose_info(handle: &GpuTensorHandle) -> Option<TransposeInfo> {
    match handle.view().layout() {
        GpuTensorLayout::Contiguous => None,
        GpuTensorLayout::Transpose(info) => Some(info),
    }
}

pub fn handle_is_transposed(handle: &GpuTensorHandle) -> bool {
    handle_transpose_info(handle).is_some()
}

/// Bytes in the native host payload returned by `download_numeric`. This is
/// logical view geometry, not the provider's packed device allocation size.
pub fn checked_numeric_download_host_payload_bytes(handle: &GpuTensorHandle) -> Option<u64> {
    let bytes = native_buffer_byte_length(
        handle.shape(),
        handle.physical().element_type(),
        handle.physical().storage(),
    )
    .ok()?;
    u64::try_from(bytes).ok()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuTensorStorage {
    Real,
    ComplexInterleaved,
}

/// Exact native integer element storage for host/device transfer contracts.
///
/// This intentionally lives in the acceleration API instead of depending on
/// `runmat-builtins`, so providers can move integer buffers without first
/// materializing RunMat's lossy floating compatibility view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntegerElementType {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl IntegerElementType {
    pub const fn element_size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 => 4,
            Self::I64 | Self::U64 => 8,
        }
    }
}

impl From<IntegerElementType> for runmat_types::IntegerClass {
    fn from(element_type: IntegerElementType) -> Self {
        match element_type {
            IntegerElementType::I8 => Self::Int8,
            IntegerElementType::I16 => Self::Int16,
            IntegerElementType::I32 => Self::Int32,
            IntegerElementType::I64 => Self::Int64,
            IntegerElementType::U8 => Self::UInt8,
            IntegerElementType::U16 => Self::UInt16,
            IntegerElementType::U32 => Self::UInt32,
            IntegerElementType::U64 => Self::UInt64,
        }
    }
}

impl From<runmat_types::IntegerClass> for IntegerElementType {
    fn from(class: runmat_types::IntegerClass) -> Self {
        use runmat_types::IntegerClass;

        match class {
            IntegerClass::Int8 => Self::I8,
            IntegerClass::Int16 => Self::I16,
            IntegerClass::Int32 => Self::I32,
            IntegerClass::Int64 => Self::I64,
            IntegerClass::UInt8 => Self::U8,
            IntegerClass::UInt16 => Self::U16,
            IntegerClass::UInt32 => Self::U32,
            IntegerClass::UInt64 => Self::U64,
        }
    }
}

/// Look up the exact native integer class stored by a GPU tensor handle.
pub fn handle_integer_type(handle: &GpuTensorHandle) -> Option<IntegerElementType> {
    handle.physical().element_type().integer_type()
}

/// Look up the source-language integer class stored by a GPU tensor handle.
pub fn handle_integer_class(handle: &GpuTensorHandle) -> Option<runmat_types::IntegerClass> {
    handle_integer_type(handle).map(Into::into)
}

impl Default for GpuTensorStorage {
    fn default() -> Self {
        Self::Real
    }
}

/// Durable physical description of a provider-owned numeric buffer.
///
/// Provider-created numeric handles populate element type and storage at
/// allocation time so cloning or serializing a handle cannot discard its
/// physical interpretation. A missing physical field is invalid at numeric
/// provider boundaries. Provenance is populated by ownership-aware runtime
/// boundaries.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuTensorDescriptor {
    pub element_type: Option<NumericElementType>,
    pub storage: Option<GpuTensorStorage>,
    pub provenance: Option<GpuHandleProvenance>,
}

impl GpuTensorDescriptor {
    pub const fn numeric(element_type: NumericElementType, storage: GpuTensorStorage) -> Self {
        Self {
            element_type: Some(element_type),
            storage: Some(storage),
            provenance: None,
        }
    }
}

/// Semantic origin of a resident handle.
///
/// Automatic residency is an implementation detail and may transparently return
/// to the host. Explicit residency represents user-visible `gpuArray` intent and
/// must be retained by operations that preserve gpuArray semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuHandleProvenance {
    Automatic,
    Explicit,
}

pub fn handle_provenance(handle: &GpuTensorHandle) -> Option<GpuHandleProvenance> {
    handle.view().provenance()
}

pub fn handle_is_explicit(handle: &GpuTensorHandle) -> bool {
    handle_provenance(handle) == Some(GpuHandleProvenance::Explicit)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderSpectralRange {
    Onesided,
    Twosided,
    Centered,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderSpectralFrameMode {
    Sliding {
        hop: usize,
    },
    ColumnSliding {
        hop: usize,
        input_rows: usize,
        frames_per_column: usize,
    },
    FoldedColumns {
        input_rows: usize,
    },
}

#[derive(Clone, Debug)]
pub struct ProviderSpectralRequest<'a> {
    pub input: &'a GpuTensorHandle,
    pub input_len: usize,
    pub input_complex: bool,
    pub window: &'a [f64],
    pub nfft: usize,
    pub frame_count: usize,
    pub frame_mode: ProviderSpectralFrameMode,
    pub range: ProviderSpectralRange,
    pub denominator: f64,
}

#[derive(Clone, Debug)]
pub struct ProviderSpectralResult {
    pub s: GpuTensorHandle,
    pub ps: GpuTensorHandle,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderEnvelopeMethod {
    Analytic,
    AnalyticFir { filter_len: usize },
    Rms { window_len: usize },
}

#[derive(Clone, Debug)]
pub struct ProviderEnvelopeRequest<'a> {
    pub input: &'a GpuTensorHandle,
    pub channel_len: usize,
    pub channel_count: usize,
    pub output_shape: &'a [usize],
    pub method: ProviderEnvelopeMethod,
}

#[derive(Clone, Debug)]
pub struct ProviderEnvelopeResult {
    pub upper: GpuTensorHandle,
    pub lower: GpuTensorHandle,
}

#[derive(Clone, Debug)]
pub struct ProviderHilbertRequest<'a> {
    pub input: &'a GpuTensorHandle,
    /// Optional FFT length along `dim`.
    pub length: Option<usize>,
    /// Zero-based transform dimension.
    pub dim: usize,
}

#[derive(Clone, Debug)]
pub struct ProviderModulationRequest<'a> {
    pub input: &'a GpuTensorHandle,
    /// `(real, imag)` pairs interleaved by symbol index.
    pub constellation: &'a [f64],
}

#[derive(Clone, Debug)]
pub struct ProviderBitModulationRequest<'a> {
    pub input: &'a GpuTensorHandle,
    /// Number of bit rows in the input before grouping.
    pub input_rows: usize,
    /// Number of input bits that form one output symbol.
    pub bits_per_symbol: usize,
    /// `(real, imag)` pairs interleaved by symbol index.
    pub constellation: &'a [f64],
}

pub async fn uniform_spectral_estimate(
    request: ProviderSpectralRequest<'_>,
) -> ProviderExecutionResult<ProviderSpectralResult> {
    validate_uniform_spectral_request(&request)?;
    request
        .input
        .allocation_lease()
        .validate_live()
        .map_err(ProviderExecutionError::from_source)?;

    let provider = provider_for_handle(request.input)
        .ok_or_else(|| anyhow!("uniform_spectral_estimate: GPU provider unavailable"))?;
    provider.uniform_spectral_estimate(&request).await
}

mod spectral_request;
pub use spectral_request::validate_uniform_spectral_request;

pub async fn signal_envelope(
    request: ProviderEnvelopeRequest<'_>,
) -> ProviderExecutionResult<ProviderEnvelopeResult> {
    let expected_len = request
        .channel_len
        .checked_mul(request.channel_count)
        .ok_or_else(|| anyhow!("signal_envelope: invalid request"))?;
    let output_len = request
        .output_shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| anyhow!("signal_envelope: invalid request"))?;
    let input_len = request
        .input
        .shape()
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| anyhow!("signal_envelope: invalid request"))?;
    if request.channel_len == 0
        || request.channel_count == 0
        || request.output_shape.is_empty()
        || output_len != expected_len
        || input_len != expected_len
        || !provider_envelope_input_shape_matches(
            request.input.shape(),
            request.channel_len,
            request.channel_count,
        )
    {
        return Err(anyhow!("signal_envelope: invalid request").into());
    }

    match request.method {
        ProviderEnvelopeMethod::AnalyticFir { filter_len }
        | ProviderEnvelopeMethod::Rms {
            window_len: filter_len,
        } if filter_len == 0 => return Err(anyhow!("signal_envelope: invalid request").into()),
        _ => {}
    }

    let provider = provider_for_handle(request.input)
        .ok_or_else(|| anyhow!("signal_envelope: GPU provider unavailable"))?;
    provider.signal_envelope(&request).await
}

fn provider_envelope_input_shape_matches(
    shape: &[usize],
    channel_len: usize,
    channel_count: usize,
) -> bool {
    if channel_count == 1 {
        return match shape {
            [len] => *len == channel_len,
            [rows, cols] => {
                (*rows == channel_len && *cols == 1) || (*rows == 1 && *cols == channel_len)
            }
            _ => false,
        };
    }

    matches!(shape, [rows, cols] if *rows == channel_len && *cols == channel_count)
}

pub async fn signal_hilbert(
    request: ProviderHilbertRequest<'_>,
) -> ProviderExecutionResult<GpuTensorHandle> {
    if request.length == Some(0) {
        return Err(anyhow!("signal_hilbert: invalid request").into());
    }
    if request.dim >= request.input.shape().len() {
        return Err(anyhow!("signal_hilbert: invalid request").into());
    }

    let provider = provider_for_handle(request.input)
        .ok_or_else(|| anyhow!("signal_hilbert: GPU provider unavailable"))?;
    provider.signal_hilbert(&request).await
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApiDeviceInfo {
    pub device_id: u32,
    pub name: String,
    pub vendor: String,
    pub memory_bytes: Option<u64>,
    pub backend: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReduceDimResult {
    pub values: GpuTensorHandle,
    pub indices: GpuTensorHandle,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProviderCumminResult {
    pub values: GpuTensorHandle,
    pub indices: GpuTensorHandle,
}

/// Result payload returned by provider-side `cummax` scans.
///
/// Alias of [`ProviderCumminResult`] because both operations return the same pair of tensors
/// (running values and MATLAB-compatible indices).
pub type ProviderCummaxResult = ProviderCumminResult;

/// Names a shared acceleration context that callers may request (e.g. plotting).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelContextKind {
    Plotting,
}

/// Handle returned by [`export_context`] that describes a shared GPU context.
#[derive(Clone)]
pub enum AccelContextHandle {
    #[cfg(feature = "wgpu")]
    Wgpu(WgpuContextHandle),
}

impl AccelContextHandle {
    /// Returns the underlying WGPU context when available.
    #[cfg(feature = "wgpu")]
    pub fn as_wgpu(&self) -> Option<&WgpuContextHandle> {
        match self {
            AccelContextHandle::Wgpu(ctx) => Some(ctx),
        }
    }
}

/// Shared WGPU device/queue pair exported by the acceleration provider.
#[cfg(feature = "wgpu")]
#[derive(Clone)]
pub struct WgpuContextHandle {
    pub instance: Arc<wgpu::Instance>,
    pub access: WgpuDeviceAccess,
    pub adapter: Arc<wgpu::Adapter>,
    pub adapter_info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
    pub features: wgpu::Features,
}

/// Owning WGPU export in the tensor's logical column-major contiguous order.
/// Contiguous inputs share their original Arc without copying. Providers must
/// materialize lazy views on-device before export; shape never describes a
/// transpose while the buffer silently retains its original physical order.
/// The carrier retains the authenticated allocation view for its backing;
/// materialized exports retain their own allocation, not the source lease.
///
/// Providers that expose this handle must ensure the buffer was created with
/// `wgpu::BufferUsages::COPY_SRC`, so zero-copy consumers can export or inspect
/// tensor data without depending on provider-specific readback APIs.
#[cfg(feature = "wgpu")]
#[derive(Clone)]
pub struct WgpuBufferRef {
    // Retains allocation accounting as well as the backend buffer. The backend
    // must associate this exact buffer with this handle before construction.
    owner: GpuTensorHandle,
    buffer: WgpuBufferOwner,
    access: WgpuDeviceAccess,
    pub len: usize,
    pub shape: Vec<usize>,
    pub element_size: usize,
    pub precision: ProviderPrecision,
}

#[cfg(feature = "wgpu")]
impl WgpuBufferRef {
    /// Publish an export after the backend has authenticated the buffer against
    /// its existing allocation record. Materialized exports must pass their own
    /// newly admitted allocation, never the source allocation's lease.
    /// `access` must be the existing authority which created this backing;
    /// raw buffer properties cannot authenticate its device association.
    pub fn from_owned_buffer(
        buffer: WgpuBufferOwner,
        handle: GpuTensorHandle,
        access: WgpuDeviceAccess,
    ) -> anyhow::Result<Self> {
        handle.allocation_lease().validate_live()?;
        anyhow::ensure!(
            matches!(
                handle.view().layout(),
                allocation::GpuTensorLayout::Contiguous
            ),
            "WGPU export requires a contiguous logical view"
        );
        anyhow::ensure!(
            handle.physical().storage() == GpuTensorStorage::Real,
            "WGPU floating export requires real storage"
        );
        let precision = handle_precision(&handle)
            .ok_or_else(|| anyhow::anyhow!("WGPU export requires floating storage"))?;
        let element_size = match precision {
            ProviderPrecision::F32 => 4,
            ProviderPrecision::F64 => 8,
        };
        let len = if handle.shape().contains(&0) {
            0
        } else {
            handle
                .shape()
                .iter()
                .try_fold(1usize, |n, d| n.checked_mul(*d))
                .ok_or_else(|| anyhow::anyhow!("WGPU export shape overflow"))?
        };
        anyhow::ensure!(
            buffer.size() >= handle.physical().requested_bytes(),
            "WGPU export buffer is smaller than its admitted allocation"
        );
        anyhow::ensure!(
            buffer.usage().contains(wgpu::BufferUsages::COPY_SRC),
            "WGPU export buffer lacks COPY_SRC usage"
        );
        Ok(Self {
            shape: handle.shape().to_vec(),
            owner: handle,
            buffer,
            access,
            len,
            element_size,
            precision,
        })
    }

    pub fn owning_handle(&self) -> &GpuTensorHandle {
        &self.owner
    }

    /// The backend must establish this association at construction. Reusing an
    /// export on another actual device is rejected before exposing its buffer.
    pub fn buffer<'a>(
        &'a self,
        permit: &'a WgpuEmission<'_>,
    ) -> Result<&'a wgpu::Buffer, WgpuAccessError> {
        permit.ensure_device(&self.access)?;
        self.owner
            .allocation_lease()
            .validate_live()
            .map_err(WgpuAccessError::Allocation)?;
        Ok(self.buffer.as_ref())
    }

    pub fn access(&self) -> &WgpuDeviceAccess {
        &self.access
    }
}

pub fn handle_storage(handle: &GpuTensorHandle) -> GpuTensorStorage {
    handle.physical().storage()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PagefunOp {
    Mtimes,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PagefunRequest {
    pub op: PagefunOp,
    pub inputs: Vec<GpuTensorHandle>,
    pub output_shape: Vec<usize>,
    pub page_dims: Vec<usize>,
    pub input_page_dims: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FindDirection {
    First,
    Last,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProviderFindResult {
    pub linear: GpuTensorHandle,
    pub rows: GpuTensorHandle,
    pub cols: GpuTensorHandle,
    pub values: Option<GpuTensorHandle>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderBandwidth {
    pub lower: u32,
    pub upper: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderSymmetryKind {
    Symmetric,
    Skew,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderHermitianKind {
    Hermitian,
    Skew,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProviderLuResult {
    pub combined: GpuTensorHandle,
    pub lower: GpuTensorHandle,
    pub upper: GpuTensorHandle,
    pub perm_matrix: GpuTensorHandle,
    pub perm_vector: GpuTensorHandle,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProviderCholResult {
    pub factor: GpuTensorHandle,
    /// MATLAB-compatible failure index (0 indicates success).
    pub info: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ProviderQrResult {
    R {
        r: GpuTensorHandle,
    },
    Qr {
        q: GpuTensorHandle,
        r: GpuTensorHandle,
    },
    Qrp {
        q: GpuTensorHandle,
        r: GpuTensorHandle,
        permutation: GpuTensorHandle,
    },
    /// Internal Rust factor envelope; not used by a script output-count request.
    FullEnvelope {
        q: GpuTensorHandle,
        r: GpuTensorHandle,
        perm_matrix: GpuTensorHandle,
        perm_vector: GpuTensorHandle,
    },
}

pub type ProviderQrPowerIterResult = ProviderQrResult;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct ProviderLinsolveOptions {
    pub lower: bool,
    pub upper: bool,
    pub rectangular: bool,
    pub transposed: bool,
    pub conjugate: bool,
    pub symmetric: bool,
    pub posdef: bool,
    pub need_rcond: bool,
    pub rcond: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProviderLinsolveResult {
    pub solution: GpuTensorHandle,
    pub reciprocal_condition: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct ProviderPinvOptions {
    pub tolerance: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProviderPolyvalMu {
    pub mean: f64,
    pub scale: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct ProviderPolyvalOptions {
    pub mu: Option<ProviderPolyvalMu>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderInvOptions {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderPolyfitResult {
    pub coefficients: Vec<f64>,
    /// Present only when the caller requested the QR statistics output.
    pub r_matrix: Option<Vec<f64>>,
    /// Residual norm accompanies the requested statistics, not coefficient-only fits.
    pub normr: Option<f64>,
    pub df: f64,
    pub mu: [f64; 2],
}

/// Coordinate system shared by the fitted coefficients and QR statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderPolyfitBasis {
    Original,
    CenteredScaled,
}

/// Requested polynomial outputs. The third output selects the centered basis;
/// the second requires representable QR statistics in that basis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderPolyfitOutputs {
    Coefficients,
    Statistics,
    StatisticsAndNormalization,
}

impl ProviderPolyfitOutputs {
    pub fn wants_statistics(self) -> bool {
        !matches!(self, Self::Coefficients)
    }

    pub fn basis(self) -> ProviderPolyfitBasis {
        match self {
            Self::StatisticsAndNormalization => ProviderPolyfitBasis::CenteredScaled,
            Self::Coefficients | Self::Statistics => ProviderPolyfitBasis::Original,
        }
    }
}

/// Numerator/denominator payload returned by provider-backed `polyder` quotient rule.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProviderPolyderQuotient {
    pub numerator: GpuTensorHandle,
    pub denominator: GpuTensorHandle,
}

/// Supported norm specifications for the `cond` builtin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderCondNorm {
    Two,
    One,
    Inf,
    Fro,
}

/// Supported norm orders for the `norm` builtin.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ProviderNormOrder {
    Two,
    One,
    Inf,
    NegInf,
    Zero,
    Fro,
    Nuc,
    P(f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderInterp1Method {
    Linear,
    Nearest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderInterp1Extrapolation {
    Nan,
    Extrapolate,
    Value,
}

#[derive(Debug, Clone)]
pub struct ProviderInterp1Request<'a> {
    /// Strictly increasing, finite sample coordinates validated by the runtime
    /// before dispatch. Providers may assume this monotonic domain invariant.
    pub x: &'a GpuTensorHandle,
    pub y: &'a GpuTensorHandle,
    pub xq: &'a GpuTensorHandle,
    pub sample_len: usize,
    pub series_count: usize,
    pub query_len: usize,
    pub output_shape: &'a [usize],
    pub method: ProviderInterp1Method,
    pub extrapolation: ProviderInterp1Extrapolation,
    pub extrapolation_value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProviderEigResult {
    pub eigenvalues: GpuTensorHandle,
    pub diagonal: GpuTensorHandle,
    pub right: GpuTensorHandle,
    pub left: Option<GpuTensorHandle>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderQrPivot {
    Matrix,
    Vector,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderQrOutputs {
    R,
    Qr,
    Qrp(ProviderQrPivot),
    FullEnvelope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ProviderQrColumnOrder {
    Unpivoted,
    #[default]
    Pivoted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderQrOptions {
    pub economy: bool,
    pub column_order: ProviderQrColumnOrder,
    pub outputs: ProviderQrOutputs,
}

impl Default for ProviderQrOptions {
    fn default() -> Self {
        Self {
            economy: false,
            column_order: ProviderQrColumnOrder::Pivoted,
            outputs: ProviderQrOutputs::FullEnvelope,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderPrecision {
    F32,
    F64,
}

/// Numerical derivative of an already executed resident activation.
/// Forward input and upstream gradient retain their native physical class.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProviderActivationAdjoint {
    Relu,
    Elu { alpha: f64 },
}

/// Operations required to reverse one admitted resident network route.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProviderResidentAdjointStage {
    Dense,
    Conv1D,
    Activation(ProviderActivationAdjoint),
    SoftmaxRows,
    LayerNorm,
    Pool1D,
}

/// Declares how provider-owned GPU handles may cross async spawn boundaries.
///
/// This is a runtime/provider policy surface (not a semantic type fact) used by
/// VM/runtime spawn handling to prevent unsynchronized device-handle races.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpawnHandleConcurrency {
    /// Provider supports immutable sharing of handle-backed values across spawned tasks.
    ImmutableShare,
    /// Provider supports copy-on-write semantics when spawned and parent tasks diverge.
    CopyOnWrite,
    /// Provider supports synchronized mutation for shared handles.
    SynchronizedMutation,
    /// Provider rejects spawned sharing of raw handles.
    Reject,
}

impl SpawnHandleConcurrency {
    pub fn as_str(self) -> &'static str {
        match self {
            SpawnHandleConcurrency::ImmutableShare => "immutable_share",
            SpawnHandleConcurrency::CopyOnWrite => "copy_on_write",
            SpawnHandleConcurrency::SynchronizedMutation => "synchronized_mutation",
            SpawnHandleConcurrency::Reject => "reject",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReductionTwoPassMode {
    Auto,
    ForceOn,
    ForceOff,
}

impl ReductionTwoPassMode {
    pub fn as_str(self) -> &'static str {
        match self {
            ReductionTwoPassMode::Auto => "auto",
            ReductionTwoPassMode::ForceOn => "force_on",
            ReductionTwoPassMode::ForceOff => "force_off",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ReductionFlavor {
    Sum,
    Mean,
    CustomScale(f64),
}

/// Physical shader requirements declared by the owning WGSL producer.
/// A provider checks these against its *enabled device*, not adapter offers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProviderShaderRequirements {
    Basic,
    F64IeeeWords,
}

impl ReductionFlavor {
    pub fn is_mean(self) -> bool {
        matches!(self, ReductionFlavor::Mean)
    }

    pub fn scale(self, reduce_len: usize) -> f64 {
        match self {
            ReductionFlavor::Sum => 1.0,
            ReductionFlavor::Mean => {
                if reduce_len == 0 {
                    1.0
                } else {
                    1.0 / reduce_len as f64
                }
            }
            ReductionFlavor::CustomScale(scale) => scale,
        }
    }
}

/// Normalisation mode for correlation coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorrcoefNormalization {
    Unbiased,
    Biased,
}

/// Row-selection strategy for correlation coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorrcoefRows {
    All,
    Complete,
    Pairwise,
}

/// Options controlling provider-backed correlation coefficient computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorrcoefOptions {
    pub normalization: CorrcoefNormalization,
    pub rows: CorrcoefRows,
}

impl Default for CorrcoefOptions {
    fn default() -> Self {
        Self {
            normalization: CorrcoefNormalization::Unbiased,
            rows: CorrcoefRows::All,
        }
    }
}

/// Normalisation mode used by covariance computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CovNormalization {
    Unbiased,
    Biased,
}

/// Row handling strategy for covariance computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CovRows {
    All,
    OmitRows,
    PartialRows,
}

/// Options controlling provider-backed covariance computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CovarianceOptions {
    pub normalization: CovNormalization,
    pub rows: CovRows,
    pub has_weight_vector: bool,
}

impl Default for CovarianceOptions {
    fn default() -> Self {
        Self {
            normalization: CovNormalization::Unbiased,
            rows: CovRows::All,
            has_weight_vector: false,
        }
    }
}

/// Normalization strategy used by provider-backed standard deviation reductions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderStdNormalization {
    Sample,
    Population,
}

/// NaN handling mode for provider-backed reductions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderNanMode {
    Include,
    Omit,
}

/// Moving-window reduction operation executed by acceleration providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderMovingWindowOp {
    Sum,
    Mean,
    Prod,
    Min,
    Max,
    Median,
    Std,
    Var,
}

/// Endpoint handling for provider-backed moving-window reductions.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ProviderMovingWindowEndpoints {
    Shrink,
    Discard,
    Fill(f64),
}

/// Request for provider-backed count-window moving reductions.
#[derive(Debug, Clone, Copy)]
pub struct ProviderMovingWindowRequest<'a> {
    pub input: &'a GpuTensorHandle,
    pub output_shape: &'a [usize],
    /// Zero-based dimension along which the moving window is applied.
    pub dim: usize,
    pub before: usize,
    pub after: usize,
    pub op: ProviderMovingWindowOp,
    pub endpoints: ProviderMovingWindowEndpoints,
    pub nan_mode: ProviderNanMode,
    pub normalization: ProviderStdNormalization,
}

/// Dimension selection for provider-backed `mode` reductions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderModeAxes {
    /// First non-singleton dimension selected by MATLAB default rules.
    Default,
    /// Zero-based dimension supplied by `mode(A, dim)`.
    Dim(usize),
    /// Collapse every element, as in `mode(A, "all")`.
    All,
}

/// Request for provider-backed modal reductions.
#[derive(Debug, Clone, Copy)]
pub struct ProviderModeRequest<'a> {
    pub input: &'a GpuTensorHandle,
    pub axes: ProviderModeAxes,
    pub want_frequency: bool,
    pub want_ties: bool,
}

/// Sorted tied modal values for each output slice.
///
/// `values` stores all tied values for all slices in ascending per-slice order.
/// `offsets` and `counts` are one entry per output slice and describe the
/// slice-local range inside `values`. Runtime callers convert this ragged
/// representation into MATLAB cell output `C`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderModeTiedSets {
    pub values: HostTensorOwned,
    pub offsets: Vec<usize>,
    pub counts: Vec<usize>,
}

/// Result of a provider-backed `mode` reduction.
///
/// `values` is the MATLAB `M` output and stays resident. `frequencies`, when
/// requested, is the MATLAB `F` output and stays resident. `ties`, when
/// requested, carries the MATLAB `C` tied-set data in a host ragged form because
/// the public runtime value is a cell array.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderModeResult {
    pub values: GpuTensorHandle,
    pub frequencies: Option<GpuTensorHandle>,
    pub ties: Option<ProviderModeTiedSets>,
}

/// Direction used when computing prefix sums on the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderScanDirection {
    Forward,
    Reverse,
}

/// Spacing input for provider-backed trapezoidal integration.
#[derive(Debug, Clone, Copy)]
pub enum ProviderTrapezoidSpacing<'a> {
    Unit,
    Scalar(f64),
    ScalarHandle(&'a GpuTensorHandle),
    Vector(&'a GpuTensorHandle),
    Tensor(&'a GpuTensorHandle),
}

/// Sort direction used by acceleration providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SortOrder {
    Ascend,
    Descend,
}

/// Comparison strategy applied during sorting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SortComparison {
    Auto,
    Real,
    Abs,
}

/// Host-resident outputs returned by provider-backed sort operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SortResult {
    pub values: HostTensorOwned,
    pub indices: HostTensorOwned,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SortRowsColumnSpec {
    pub index: usize,
    pub order: SortOrder,
}

/// Ordering applied by provider-backed `unique` operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UniqueOrder {
    Sorted,
    Stable,
}

/// Occurrence selection for provider-backed `unique` operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UniqueOccurrence {
    First,
    Last,
}

/// Options controlling provider-backed `unique` operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UniqueOptions {
    pub rows: bool,
    pub order: UniqueOrder,
    pub occurrence: UniqueOccurrence,
    pub treat_missing_as_distinct: bool,
    pub explicit_order: bool,
    pub explicit_occurrence: bool,
}

/// Host-resident outputs returned by provider-backed `unique` operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UniqueResult {
    pub values: HostTensorOwned,
    pub ia: HostTensorOwned,
    pub ic: HostTensorOwned,
}

/// Ordering applied by provider-backed `union` operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnionOrder {
    Sorted,
    Stable,
}

/// Options controlling provider-backed `union` operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnionOptions {
    pub rows: bool,
    pub order: UnionOrder,
}

/// Host-resident outputs returned by provider-backed `union` operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnionResult {
    pub values: HostTensorOwned,
    pub ia: HostTensorOwned,
    pub ib: HostTensorOwned,
}

/// Parameterisation of 2-D filters generated by `fspecial`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FspecialFilter {
    Average {
        rows: u32,
        cols: u32,
    },
    Disk {
        radius: f64,
        size: u32,
    },
    Gaussian {
        rows: u32,
        cols: u32,
        sigma: f64,
    },
    Laplacian {
        alpha: f64,
    },
    Log {
        rows: u32,
        cols: u32,
        sigma: f64,
    },
    Motion {
        length: u32,
        kernel_size: u32,
        angle_degrees: f64,
        oversample: u32,
    },
    Prewitt,
    Sobel,
    Unsharp {
        alpha: f64,
    },
}

/// Request dispatched to acceleration providers for `fspecial` kernels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FspecialRequest {
    pub filter: FspecialFilter,
}

/// Padding strategy used by `imfilter`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImfilterPadding {
    Constant,
    Replicate,
    Symmetric,
    Circular,
}

/// Output sizing mode used by `imfilter`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImfilterShape {
    Same,
    Full,
    Valid,
}

/// Correlation vs convolution behaviour for `imfilter`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImfilterMode {
    Correlation,
    Convolution,
}

/// Options supplied to acceleration providers for `imfilter`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImfilterOptions {
    pub padding: ImfilterPadding,
    pub constant_value: f64,
    pub shape: ImfilterShape,
    pub mode: ImfilterMode,
}

impl Default for ImfilterOptions {
    fn default() -> Self {
        Self {
            padding: ImfilterPadding::Constant,
            constant_value: 0.0,
            shape: ImfilterShape::Same,
            mode: ImfilterMode::Correlation,
        }
    }
}

/// Ordering applied by provider-backed `setdiff` operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SetdiffOrder {
    Sorted,
    Stable,
}

/// Options controlling provider-backed `setdiff` operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SetdiffOptions {
    pub rows: bool,
    pub order: SetdiffOrder,
}

/// Host-resident outputs returned by provider-backed `setdiff` operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SetdiffResult {
    pub values: HostTensorOwned,
    pub ia: HostTensorOwned,
}

/// Options controlling provider-backed `ismember` operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IsMemberOptions {
    pub rows: bool,
}

/// Host-resident logical output returned by providers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostLogicalOwned {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
}

/// Host-resident outputs returned by provider-backed `ismember` operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IsMemberResult {
    pub mask: HostLogicalOwned,
    pub loc: HostTensorOwned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderConvMode {
    Full,
    Same,
    Valid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderConvOrientation {
    Row,
    Column,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderConv1dOptions {
    pub mode: ProviderConvMode,
    pub orientation: ProviderConvOrientation,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProviderMoments2 {
    pub mean: GpuTensorHandle,
    pub ex2: GpuTensorHandle,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderDispatchStats {
    /// Number of GPU dispatches recorded for this category.
    pub count: u64,
    /// Accumulated wall-clock time of dispatches in nanoseconds (host measured).
    pub total_wall_time_ns: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderFallbackStat {
    pub reason: String,
    pub count: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProviderTelemetry {
    pub fused_elementwise: ProviderDispatchStats,
    pub fused_reduction: ProviderDispatchStats,
    pub matmul: ProviderDispatchStats,
    pub linsolve: ProviderDispatchStats,
    pub mldivide: ProviderDispatchStats,
    pub mrdivide: ProviderDispatchStats,
    pub upload_bytes: u64,
    pub download_bytes: u64,
    pub solve_fallbacks: Vec<ProviderFallbackStat>,
    pub fusion_cache_hits: u64,
    pub fusion_cache_misses: u64,
    pub bind_group_cache_hits: u64,
    pub bind_group_cache_misses: u64,
    /// Optional per-layout bind group cache counters (layout tags and their hit/miss counts)
    pub bind_group_cache_by_layout: Option<Vec<BindGroupLayoutTelemetry>>,
    /// Recent kernel launch metadata (bounded log; newest last)
    pub kernel_launches: Vec<KernelLaunchTelemetry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BindGroupLayoutTelemetry {
    pub tag: String,
    pub hits: u64,
    pub misses: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelAttrTelemetry {
    pub key: String,
    pub value: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelLaunchTelemetry {
    pub kernel: String,
    pub precision: Option<String>,
    pub shape: Vec<KernelAttrTelemetry>,
    pub tuning: Vec<KernelAttrTelemetry>,
}

pub type AccelProviderFuture<'a, T> = Pin<Box<dyn Future<Output = anyhow::Result<T>> + 'a>>;
pub type AccelDownloadFuture<'a> = AccelProviderFuture<'a, crate::HostTensorOwned>;
pub type AccelIntegerDownloadFuture<'a> = AccelProviderFuture<'a, crate::HostIntegerTensorOwned>;
pub type AccelNumericDownloadFuture<'a> = AccelProviderFuture<'a, crate::HostNumericTensorOwned>;

/// Provider-owned physical demand for one post-region queue-tail completion.
/// Child operations retain their own receipts; this grants no execution right.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProviderRegionTailResources {
    pub queue_slots: u32,
    pub provider_scratch_bytes: u64,
    /// Actual request-owned host completion backing. None is unknown, not0.
    pub completion_control_host_bytes: Option<u64>,
}

#[derive(Debug)]
pub struct UnsupportedProviderOperation {
    detail: String,
}

impl std::fmt::Display for UnsupportedProviderOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.detail)
    }
}

impl std::error::Error for UnsupportedProviderOperation {}

pub fn unsupported_provider_operation(detail: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(UnsupportedProviderOperation {
        detail: detail.into(),
    })
}

pub fn is_unsupported_provider_operation(error: &anyhow::Error) -> bool {
    is_unsupported_provider_error(error.as_ref())
}

fn unsupported_future<T>(message: &'static str) -> AccelProviderFuture<'static, T> {
    Box::pin(async move { Err(unsupported_provider_operation(message)) })
}

fn unsupported_execution_future<T>(message: &'static str) -> ProviderExecutionFuture<'static, T> {
    Box::pin(async move { Err(unsupported_provider_operation(message).into()) })
}

/// Device/provider interface that backends implement and register into the runtime layer
pub trait AccelProvider: AcceleratorThreadBound {
    /// Exact access identity for the canonical WGPU provider. This does not
    /// grant raw device or queue access to callers.
    #[cfg(feature = "wgpu")]
    fn wgpu_device_access(&self) -> Option<&WgpuDeviceAccess> {
        None
    }

    fn region_queue_tail_resources(&self) -> Option<ProviderRegionTailResources> {
        None
    }

    fn observe_region_queue_tail<'a>(
        &'a self,
        _device_id: u32,
        _completion_claim: Option<
            std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>,
        >,
    ) -> ProviderExecutionFuture<'a, ()> {
        unsupported_execution_future("VM region queue-tail observation is unsupported")
    }

    /// Effect-free ordinary real Sub/Div proposal. Source-level numerical
    /// admission and any selected-program binding remain separate.
    fn propose_real_binary(
        &self,
        _request: &ProviderRealBinaryRequest,
    ) -> ProviderExecutionResult<Option<ProviderRealBinaryPhysicalProposal>> {
        Ok(None)
    }

    /// Consume a privately minted recipe against both authentic live views.
    fn execute_real_binary_recipe<'a>(
        &'a self,
        _lhs: &'a GpuTensorHandle,
        _rhs: &'a GpuTensorHandle,
        _recipe: &'a ProviderRealBinaryRecipe,
        _admission: Option<&'a dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionFuture<'a, ProviderExecutedRealBinary> {
        Box::pin(async {
            Err(ProviderExecutionError::from(
                unsupported_provider_operation("real binary recipe execution is unsupported"),
            ))
        })
    }

    /// Provider-owned retained host metadata for one canonical dense output
    /// registration at the given shape rank. `None` is unknown, never zero:
    /// finite selected callers must decline before an effect. This does not
    /// claim arbitrary backend RSS or already-initialized provider caches.
    fn registered_output_host_metadata(
        &self,
        _rank: usize,
    ) -> Option<runmat_execution::resource::ProviderHostRegistrationMetadata> {
        None
    }

    /// Additional provider-owned table backing for the selected number of
    /// registrations at the current allocation-state snapshot. A changed
    /// snapshot is rechecked by `prepare_output_host_metadata` pre-effect.
    fn output_registration_table_upper_bytes(&self, _registrations: usize) -> Option<u64> {
        None
    }

    /// Request-local preparation records retained until selected stage
    /// settlement; distinct from registered outputs and table backing.
    fn output_registration_preparation_host_bytes(&self) -> Option<u64> {
        None
    }

    /// Control backing for a reusable concurrent registration reservation.
    /// Absent support is distinct from a zero-byte or cumulative reservation.
    fn reusable_output_registration_host_bytes(&self) -> Option<u64> {
        None
    }

    /// Reserve simultaneous physical records rather than total emissions.
    /// Returned slot ownership follows each record through exact retirement.
    fn prepare_reusable_output_host_metadata(
        &self,
        _admission: Box<dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
        _registrations: usize,
    ) -> ProviderExecutionResult<Box<dyn runmat_execution::resource::ProviderHostMetadataAdmission>>
    {
        Err(ProviderExecutionError::from(
            unsupported_provider_operation(
                "reusable provider registration capacity is unsupported",
            ),
        ))
    }

    /// Prepare provider-owned registry backing and the exact number of output
    /// slots before the selected operation can emit. `None` admission is only
    /// an unbounded route; finite callers pass an owned managed receipt.
    fn prepare_output_host_metadata(
        &self,
        _admission: Box<dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
        _registrations: usize,
    ) -> ProviderExecutionResult<Box<dyn runmat_execution::resource::ProviderHostMetadataAdmission>>
    {
        Err(ProviderExecutionError::from(
            unsupported_provider_operation("provider output metadata reservation is unsupported"),
        ))
    }

    /// Description-only physical proposal for one ordinary real-elementwise
    /// operation. It does not supply numerical or selected-source admission.
    fn propose_real_elementwise(
        &self,
        _request: &ProviderRealElementwiseRequest,
    ) -> ProviderExecutionResult<Option<ProviderRealElementwisePhysicalProposal>> {
        Ok(None)
    }

    fn propose_power_evaluation(
        &self,
        _base: &GpuTensorHandle,
        _exponent: &GpuTensorHandle,
        _domain: PowerResultDomain,
    ) -> ProviderExecutionResult<Option<power_service::PowerEvaluationPhysicalProposal>> {
        Ok(None)
    }

    fn execute_power_evaluation_recipe<'a>(
        &'a self,
        _recipe: &'a power_service::PowerEvaluationRecipe,
    ) -> ProviderExecutionFuture<'a, ProviderExecutedRealElementwise> {
        Box::pin(async {
            Err(ProviderExecutionError::from(
                unsupported_provider_operation(
                    "paired POWER evaluation recipe execution is unsupported",
                ),
            ))
        })
    }

    fn execute_power_evaluation_recipe_admitted<'a>(
        &'a self,
        recipe: &'a power_service::PowerEvaluationRecipe,
        admission: Option<&'a dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionFuture<'a, ProviderExecutedRealElementwise> {
        if admission.is_some() {
            Box::pin(async {
                Err(
                    unsupported_provider_operation("admitted POWER evaluation is unsupported")
                        .into(),
                )
            })
        } else {
            self.execute_power_evaluation_recipe(recipe)
        }
    }

    /// Handle-free physical projection for sources described by original
    /// live metadata or the provider's own dense allocator contract. No
    /// handle, numerical evidence or execution permission is minted here.
    fn propose_power_pair_geometry(
        &self,
        _inputs: &[ProviderRepresentation; 2],
        _descriptions: &[PowerInputDescription; 2],
        _frozen: Option<&power_service::PowerProbePhysicalProposal>,
    ) -> ProviderExecutionResult<Option<power_service::PowerPairPhysicalProjection>> {
        Ok(None)
    }

    /// Exact ordinary paired POWER probe. A frozen proposal carries physical
    /// snapshot/geometry, not numerical evidence or placement permission.
    fn propose_power_probe(
        &self,
        _base: &GpuTensorHandle,
        _exponent: &GpuTensorHandle,
        _frozen: Option<&power_service::PowerProbePhysicalProposal>,
    ) -> ProviderExecutionResult<Option<power_service::PowerProbePhysicalProposal>> {
        Ok(None)
    }

    fn execute_power_probe_recipe<'a>(
        &'a self,
        _recipe: &'a power_service::PowerProbeRecipe,
    ) -> ProviderExecutionFuture<'a, power_service::PowerProbeObserved> {
        Box::pin(async {
            Err(ProviderExecutionError::from(
                unsupported_provider_operation(
                    "paired POWER probe recipe execution is unsupported",
                ),
            ))
        })
    }

    /// Exact native global extrema tree, before allocation/submission. This
    /// supplies physical work only, never a numerical-domain classification.
    /// Complete global min/max tree for a native real source. This distinct
    /// physical contract propagates represented NaNs, including mixed inputs;
    /// an omit-NaN legacy reduction must decline rather than certify a probe.
    fn propose_global_extrema(
        &self,
        _request: &ProviderGlobalExtremaRequest,
    ) -> ProviderExecutionResult<Option<ProviderGlobalExtremaPhysicalProposal>> {
        Ok(None)
    }

    fn execute_global_extrema_recipe<'a>(
        &'a self,
        _source: &'a GpuTensorHandle,
        _recipe: &'a ProviderGlobalExtremaRecipe,
    ) -> ProviderExecutionFuture<'a, ProviderExecutedGlobalExtrema> {
        Box::pin(async {
            Err(ProviderExecutionError::from(
                unsupported_provider_operation("global extrema recipe execution is unsupported"),
            ))
        })
    }

    /// Consume the same privately minted recipe used for physical planning.
    /// Providers must authenticate the actual source and rederive the recipe
    /// before effects, then report observed allocation/submission resources.
    fn execute_real_elementwise_recipe<'a>(
        &'a self,
        _input: &'a GpuTensorHandle,
        _recipe: &'a ProviderRealElementwiseRecipe,
        _admission: Option<&'a dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionFuture<'a, ProviderExecutedRealElementwise> {
        Box::pin(async {
            Err(ProviderExecutionError::from(
                unsupported_provider_operation("real elementwise recipe execution is unsupported"),
            ))
        })
    }
    /// Project the ordinary resident F32/F64 dense-real output allocation key.
    /// Other encodings fail closed. This does not certify an operation,
    /// kernel, numerical domain, resource transcript, or placement.
    fn plan_dense_real_output_buffer_bytes(
        &self,
        _output: &ProviderRepresentation,
    ) -> ProviderExecutionResult<Option<u64>> {
        Ok(None)
    }

    /// Project the actual provider allocator's output extent for this exact
    /// symbolic POWER route. This is not a future handle or a placement ticket;
    /// the produced allocation must be checked against it after execution.
    fn plan_power_output_buffer_bytes(
        &self,
        _planned: &power_service::PowerSymbolicPlan,
        _domain: PowerResultDomain,
    ) -> ProviderExecutionResult<Option<u64>> {
        Ok(None)
    }

    /// Propose the same ordered real mean recipe for a future, handle-free
    /// intermediate. This is not allocation, source-owner or execution proof.
    fn plan_ordered_real_mean(
        &self,
        _source: &ProviderRepresentation,
        _source_buffer_bytes: u64,
        _axes_zero_based: &[usize],
        _output_precision: ProviderPrecision,
        _nan_mode: ProviderRealMeanNaNMode,
    ) -> anyhow::Result<Option<ProviderOrderedRealMeanRecipe>> {
        Ok(None)
    }

    /// Prepare one call-boundary, source-ordered real mean with Double
    /// accumulation. The default declines without touching the device.
    fn prepare_ordered_real_mean(
        &self,
        _source: &GpuTensorHandle,
        _axes_zero_based: &[usize],
        _output_precision: ProviderPrecision,
        _nan_mode: ProviderRealMeanNaNMode,
    ) -> anyhow::Result<Option<ProviderOrderedRealMean>> {
        Ok(None)
    }

    /// Consume a symbolic proposal only after authenticating the exact live
    /// view/lease and rederiving the same capabilities and physical recipe.
    fn prepare_ordered_real_mean_against_recipe(
        &self,
        _source: &GpuTensorHandle,
        _recipe: &ProviderOrderedRealMeanRecipe,
    ) -> anyhow::Result<Option<ProviderOrderedRealMean>> {
        Ok(None)
    }

    fn execute_ordered_real_mean(
        &self,
        _selected: ProviderOrderedRealMean,
        _admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<ProviderExecutedOrderedRealMean> {
        Err(ProviderExecutionError::from(
            unsupported_provider_operation("ordered real mean unavailable"),
        ))
    }

    /// Pin an exact live N-D mean cache hit for consumption by a selected
    /// staged route. Absence means cold or unsupported, never an advisory hit.
    /// Implementations must validate the source view, dimensions, and both
    /// returned allocation leases before returning a reservation.
    fn reserve_cached_mean_nd(
        &self,
        _source: &GpuTensorHandle,
        _dims_zero_based: &[usize],
    ) -> anyhow::Result<Option<ProviderPinnedNdMean>> {
        Ok(None)
    }
    /// Select the exact cold N-D moments route when no validated warm entry
    /// was pinned. The receipt owns the source lease and is consumed once by
    /// `execute_cold_mean_nd`; default providers cannot claim this route.
    fn prepare_cold_mean_nd(
        &self,
        _source: &GpuTensorHandle,
        _dims_zero_based: &[usize],
    ) -> anyhow::Result<Option<ProviderColdNdMean>> {
        Ok(None)
    }

    fn execute_cold_mean_nd(
        &self,
        _selected: ProviderColdNdMean,
        _admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> anyhow::Result<ProviderExecutedColdNdMean> {
        Err(unsupported_provider_operation(
            "selected cold ND mean unavailable",
        ))
    }
    /// Read-only whole-route derivative capability. `false` means the runtime
    /// must reject before executing the corresponding network forward pass.
    fn resident_network_adjoint_supported(
        &self,
        _stages: &[ProviderResidentAdjointStage],
        _precision: ProviderPrecision,
    ) -> bool {
        false
    }
    /// Upload a canonical host logical value through shared admission.
    fn upload_logical(
        &self,
        host: &HostLogicalTensorView<'_>,
    ) -> NumericUploadResult<GpuTensorHandle> {
        upload_logical_protecting(self, host, &[])
    }

    /// Producer-owned logical registration. Implementations choose their real
    /// U8/F32/F64 encoding and publish a logically classified owning view.
    /// Numeric upload followed by caller relabeling is not this contract.
    /// Only a certified pre-effect Unsupported result permits an automatic
    /// source to retain its host logical output. Providers must mark failures
    /// after possible allocation or submission as unproven or after-effect.
    fn upload_logical_raw(
        &self,
        _host: &HostLogicalTensorView<'_>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        Err(unsupported_provider_operation("provider does not implement logical upload").into())
    }
    /// A finite selected upload must consume the same request-local metadata
    /// permit as numeric publication. Existing unbounded providers retain the
    /// ordinary hook; a missing admitted implementation declines pre-effect.
    fn upload_logical_raw_admitted(
        &self,
        host: &HostLogicalTensorView<'_>,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
        scratch: Option<HostLogicalUploadScratch<'_>>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        if admission.is_some() || scratch.is_some() {
            return Err(unsupported_provider_operation(
                "provider does not implement admitted logical upload",
            )
            .into());
        }
        self.upload_logical_raw(host)
    }
    fn upload(&self, host: &crate::HostTensorView) -> anyhow::Result<GpuTensorHandle>;
    fn download<'a>(&'a self, h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a>;

    /// Upload one native numeric payload through the shared all-class transfer
    /// contract. Provider implementations override [`Self::upload_numeric_raw`],
    /// not this validated entrypoint. Callers with borrowed resident operands
    /// must use [`upload_numeric_protecting`] directly.
    fn upload_numeric(
        &self,
        host: &crate::HostNumericTensorView,
    ) -> NumericUploadResult<GpuTensorHandle> {
        upload_numeric_protecting(self, host, &[])
    }

    /// Provider allocation hook, called by the validated upload entrypoints.
    ///
    /// Implementors migrating native numeric support must rename their former
    /// `upload_numeric` override to this method. Return the produced handle
    /// without performing generic descriptor-rejection cleanup: the shared
    /// boundary knows which handles are borrowed. Errors must release only
    /// allocations actually owned by this invocation. The default preserves
    /// legacy real-double and real-integer transfers; native single and complex
    /// representations require an override. Legacy `upload`/`upload_integer`
    /// errors have no effect-phase receipt: even a nested Unsupported marker
    /// cannot authorize host replay. Providers that can certify an unsupported
    /// response before effects must override this typed raw hook.
    fn upload_numeric_raw(
        &self,
        host: &crate::HostNumericTensorView,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        host.validate()?;
        match (host.data, host.storage) {
            (crate::HostNumericDataView::F64(data), GpuTensorStorage::Real) => Ok(self
                .upload(&crate::HostTensorView {
                    data,
                    shape: host.shape,
                })
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?),
            (data, GpuTensorStorage::Real) if data.element_type().integer_type().is_some() => {
                let data = match data {
                    crate::HostNumericDataView::I8(data) => crate::HostIntegerDataView::I8(data),
                    crate::HostNumericDataView::I16(data) => crate::HostIntegerDataView::I16(data),
                    crate::HostNumericDataView::I32(data) => crate::HostIntegerDataView::I32(data),
                    crate::HostNumericDataView::I64(data) => crate::HostIntegerDataView::I64(data),
                    crate::HostNumericDataView::U8(data) => crate::HostIntegerDataView::U8(data),
                    crate::HostNumericDataView::U16(data) => crate::HostIntegerDataView::U16(data),
                    crate::HostNumericDataView::U32(data) => crate::HostIntegerDataView::U32(data),
                    crate::HostNumericDataView::U64(data) => crate::HostIntegerDataView::U64(data),
                    crate::HostNumericDataView::F64(_) | crate::HostNumericDataView::F32(_) => {
                        unreachable!("guarded integer numeric transfer")
                    }
                };
                Ok(self
                    .upload_integer(&crate::HostIntegerTensorView {
                        data,
                        shape: host.shape,
                    })
                    .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?)
            }
            (data, storage) => Err(anyhow!(
                "provider must implement native {:?} {:?} uploads through upload_numeric_raw",
                data.element_type(),
                storage
            )
            .into()),
        }
    }

    /// A selected finite request supplies its own retained host-metadata
    /// authority. Providers without a typed admission implementation decline
    /// before any upload effect; an ordinary unbounded request keeps the
    /// existing raw-upload contract.
    fn upload_numeric_raw_admitted(
        &self,
        host: &crate::HostNumericTensorView,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        if admission.is_some() {
            Err(unsupported_provider_operation(
                "provider does not implement admitted numeric upload",
            )
            .into())
        } else {
            self.upload_numeric_raw(host)
        }
    }

    /// Exact temporary u32 encoding extent for a native upload. None is an
    /// unknown bound, never zero. The count includes paired complex lanes.
    fn numeric_upload_encoding_words(
        &self,
        _element: NumericElementType,
        _lanes: usize,
    ) -> Option<usize> {
        None
    }

    /// Same raw publication contract with operation-owned encoding scratch.
    /// Unsupported scratch is declined before effects, without ordinary retry.
    fn upload_numeric_raw_admitted_with_scratch(
        &self,
        host: &crate::HostNumericTensorView,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
        scratch: Option<crate::HostNumericUploadScratch<'_>>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        if scratch.is_some() {
            return Err(unsupported_provider_operation(
                "provider does not implement admitted numeric encoding scratch",
            )
            .into());
        }
        self.upload_numeric_raw_admitted(host, admission)
    }

    /// Download one native numeric payload through the shared all-class
    /// transfer contract. The default adapter keeps exact real-integer and
    /// native-double downloads available while refusing to present widened
    /// single data as native `f32`.
    fn download_numeric<'a>(&'a self, h: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            let element_type = h.physical().element_type();
            let storage = h.physical().storage();
            if element_type.integer_type().is_some() {
                if storage != GpuTensorStorage::Real {
                    return Err(anyhow!(
                        "provider must implement native complex integer downloads through download_numeric"
                    ));
                }
                let downloaded = self.download_integer(h).await?;
                let numeric: crate::HostNumericTensorOwned = downloaded.into();
                numeric.validate()?;
                return Ok(numeric);
            }
            if element_type == NumericElementType::F32 {
                return Err(anyhow!(
                    "provider must implement native single downloads through download_numeric"
                ));
            }
            let downloaded = self.download(h).await?;
            let numeric = crate::HostNumericTensorOwned {
                data: crate::HostNumericDataOwned::F64(downloaded.data),
                shape: downloaded.shape,
                storage: downloaded.storage,
            };
            numeric.validate()?;
            Ok(numeric)
        })
    }

    /// Explicit exact-source egress. The capability authorizes only its own
    /// registered facade; providers with a distinct physical transfer gate
    /// override this rather than retrying ordinary compute on denial.
    fn download_numeric_readback<'a>(
        &'a self,
        readback: &'a NumericReadback<'_>,
    ) -> AccelNumericDownloadFuture<'a> {
        if let Err(error) = readback.authorize_provider(self) {
            return Box::pin(async move { Err(error) });
        }
        self.download_numeric(readback.source())
    }

    fn prepare_numeric_readback_stage(
        &self,
        readback: &NumericReadback<'_>,
        query: &ProviderFeasibilityQuery,
        completion_claim: std::sync::Arc<
            dyn runmat_execution::resource::RetainedHostAllocationClaim,
        >,
    ) -> anyhow::Result<PreparedNumericDownloadStage> {
        readback.authorize_provider(self)?;
        self.prepare_numeric_download_stage(readback.source(), query, completion_claim)
    }

    fn download_numeric_readback_admitted_stage<'a>(
        &'a self,
        readback: &'a NumericReadback<'_>,
        stage: PreparedNumericDownloadStage,
    ) -> AccelNumericDownloadFuture<'a> {
        if let Err(error) = readback.authorize_provider(self) {
            return Box::pin(async move { Err(error) });
        }
        self.download_numeric_admitted_stage(readback.source(), stage)
    }

    /// Selected finite readback transfers the caller's retained callback
    /// control claim into the provider's physical mapping owner. The default
    /// rejects it before any provider effect; ordinary unbounded downloads
    /// continue through `download_numeric`.
    fn download_numeric_admitted<'a>(
        &'a self,
        _h: &'a GpuTensorHandle,
        _completion_claim: std::sync::Arc<
            dyn runmat_execution::resource::RetainedHostAllocationClaim,
        >,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async {
            Err(unsupported_provider_operation(
                "provider does not implement admitted numeric download",
            ))
        })
    }

    /// Issue a provider-owned physical staging lease before the selected
    /// readback's first effect. The caller has already funded callback and
    /// stage-control metadata with `download_numeric_completion_host_bound`.
    /// The default cannot certify a physical staging lifetime.
    fn prepare_numeric_download_stage(
        &self,
        _source: &GpuTensorHandle,
        _query: &ProviderFeasibilityQuery,
        _completion_claim: std::sync::Arc<
            dyn runmat_execution::resource::RetainedHostAllocationClaim,
        >,
    ) -> anyhow::Result<PreparedNumericDownloadStage> {
        Err(unsupported_provider_operation(
            "provider does not implement admitted numeric download staging",
        ))
    }

    /// Consume the exact selected staging authority. An implementation must
    /// retain the stage through physical callback retirement on cancellation.
    /// On successful return, its callback and all staging leases must already
    /// be physically retired; a consumer may then reassign its settled host
    /// scratch reservation to the returned native payload's retained owner.
    fn download_numeric_admitted_stage<'a>(
        &'a self,
        _source: &'a GpuTensorHandle,
        _stage: PreparedNumericDownloadStage,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async {
            Err(unsupported_provider_operation(
                "provider does not implement admitted numeric download staging",
            ))
        })
    }

    /// Provider-owned upper bound for the callback state retained across an
    /// asynchronous numeric download. Native payload and transpose buffers
    /// belong to `download_numeric_host_workspace_bound` instead.
    fn download_numeric_completion_host_bound(
        &self,
        _handle: &GpuTensorHandle,
    ) -> runmat_execution::ResourceByteBound {
        runmat_execution::ResourceByteBound::Unavailable(
            runmat_execution::ResourceBoundUnavailable::NoScratchBound,
        )
    }

    /// Peak host workspace owned by one `download_numeric` call, including its
    /// returned native payload and shape plus any simultaneously live host
    /// temporaries. A consumer that moves that shape into a Tensor must not
    /// charge the same shape backing again.
    /// Provider-owned device/mapped staging allocations belong to provider
    /// scratch, not this host bound. A provider without a proven bound must
    /// leave it unavailable; callers must not substitute the payload length.
    fn download_numeric_host_workspace_bound(
        &self,
        _handle: &GpuTensorHandle,
    ) -> runmat_execution::ResourceByteBound {
        runmat_execution::ResourceByteBound::Unavailable(
            runmat_execution::ResourceBoundUnavailable::NoScratchBound,
        )
    }

    /// Peak provider-owned staging for one native numeric download. This is
    /// separate from the returned host payload and any host transpose clone.
    /// Unknown must remain unavailable under a finite provider scratch quota.
    fn download_numeric_provider_scratch_bound(
        &self,
        _handle: &GpuTensorHandle,
    ) -> runmat_execution::ResourceByteBound {
        runmat_execution::ResourceByteBound::Unavailable(
            runmat_execution::ResourceBoundUnavailable::NoScratchBound,
        )
    }

    /// Queue slots occupied by one native numeric download. The conservative
    /// default reserves one slot even when staging bytes are zero: zero
    /// scratch does not by itself prove that no command is submitted.
    fn download_numeric_queue_slots(&self, _handle: &GpuTensorHandle) -> u32 {
        1
    }

    /// Physical cost of downloading a result that this provider has not yet
    /// produced. The future representation is not an ownership receipt: the
    /// caller must still validate the actual handle and its exact download
    /// feasibility before readback. Unknown providers return no evidence.
    fn produced_numeric_readback_evidence(
        &self,
        _output: &ProviderRepresentation,
    ) -> Option<ProviderProducedNumericReadbackEvidence> {
        None
    }

    /// Upload exact native integer storage without converting through f64.
    /// Providers that do not expose native integer buffers must reject this
    /// operation rather than silently changing the value class or precision.
    fn upload_integer(
        &self,
        _host: &crate::HostIntegerTensorView,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "exact native integer buffers",
        ))
    }

    /// Download exact native integer storage without converting through f64.
    fn download_integer<'a>(&'a self, _h: &'a GpuTensorHandle) -> AccelIntegerDownloadFuture<'a> {
        Box::pin(async {
            Err(anyhow::anyhow!(
                "provider does not support exact native integer buffers"
            ))
        })
    }

    /// Explicit same-process acquisition only. Parsing a wire descriptor never
    /// consults ambient provider selection or creates a new allocation lease.
    fn acquire_live(
        &self,
        _wire: &GpuTensorWireDescriptor,
    ) -> Result<GpuTensorHandle, allocation::AllocationAcquisitionError> {
        Err(allocation::AllocationAcquisitionError::Unsupported)
    }
    fn device_info(&self) -> String;

    /// Native device ABI this provider can expose to extension code.
    ///
    /// Returning `None` is the conservative default. In particular, a WGPU
    /// buffer is not a CUDA allocation and must not be exported through this
    /// contract even when both ultimately use the same physical GPU.
    fn native_device_api(&self) -> Option<NativeDeviceApi> {
        None
    }

    /// Describe the provider-owned native device, context, and stream.
    fn native_device_context(&self) -> anyhow::Result<NativeDeviceContext> {
        Err(anyhow!(
            "provider '{}' does not expose a native device context",
            self.device_info()
        ))
    }

    /// Make the provider's native context current on this execution lane.
    fn enter_native_device_context(&self) -> anyhow::Result<NativeDeviceContextGuard> {
        Err(anyhow!(
            "provider '{}' does not support native device context entry",
            self.device_info()
        ))
    }

    /// Export one provider-owned buffer to compatible native extension code.
    fn export_native_device_buffer_raw(
        &self,
        _handle: &GpuTensorHandle,
        _access: NativeDeviceAccess,
    ) -> anyhow::Result<NativeDeviceBuffer> {
        Err(anyhow!(
            "provider '{}' does not expose native device buffers",
            self.device_info()
        ))
    }

    /// Allocate a provider-owned native numeric buffer.
    fn allocate_native_device_buffer(
        &self,
        _shape: &[usize],
        _class: NativeDeviceClass,
        _storage: GpuTensorStorage,
        _initialization: NativeDeviceInitialization,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow!(
            "provider '{}' does not allocate native extension buffers",
            self.device_info()
        ))
    }

    /// Make an independent provider-owned copy of a native device buffer.
    fn copy_native_device_buffer(
        &self,
        _source: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow!(
            "provider '{}' does not copy native extension buffers",
            self.device_info()
        ))
    }

    /// Copy one component from an interleaved complex native buffer into a
    /// new real buffer owned by this provider.
    fn copy_native_device_component(
        &self,
        _source: &GpuTensorHandle,
        _component: NativeDeviceComponent,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow!(
            "provider '{}' does not extract native complex components",
            self.device_info()
        ))
    }

    /// Combine matching real native buffers into a new interleaved complex
    /// buffer owned by this provider.
    fn combine_native_device_components(
        &self,
        _real: &GpuTensorHandle,
        _imaginary: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow!(
            "provider '{}' does not combine native complex components",
            self.device_info()
        ))
    }

    /// Wait for work submitted through the provider's native extension stream.
    fn synchronize_native_device(&self) -> anyhow::Result<()> {
        Err(anyhow!(
            "provider '{}' does not expose native device synchronization",
            self.device_info()
        ))
    }
    /// Returns the stable identifier used to route this provider's handles.
    /// Distinct concurrently registered provider instances must return distinct
    /// identifiers; reusing an identifier would make handle ownership ambiguous.
    fn device_id(&self) -> u32 {
        0
    }

    /// Declares provider policy for sharing `GpuTensorHandle` values across
    /// spawned async boundaries.
    ///
    /// Default is conservative rejection. Providers that can safely support
    /// cross-task sharing should override this.
    fn spawn_handle_concurrency(&self) -> SpawnHandleConcurrency {
        SpawnHandleConcurrency::Reject
    }

    /// Returns a versioned, side-effect-free description of the provider
    /// surfaces placement may consider. The default advertises only the
    /// mandatory floating-point transfer boundary.
    fn capability_snapshot(&self) -> ProviderCapabilitySnapshot {
        ProviderCapabilitySnapshot::conservative(self)
    }

    /// Borrows the same capability authority when its immutable snapshot is
    /// retained by the provider. Dynamic providers may return `None`; callers
    /// then use the owned snapshot and must account for that request-local copy.
    fn capability_snapshot_ref(&self) -> Option<&ProviderCapabilitySnapshot> {
        None
    }

    /// Returns a validated cluster-facing device description when this
    /// provider can execute remotely scheduled accelerator work. Providers
    /// opt in explicitly, so host-only compatibility providers are never
    /// advertised as hardware accelerators.
    fn execution_accelerator_device(
        &self,
        _inventory_epoch: u64,
    ) -> anyhow::Result<Option<runmat_execution::resource::AcceleratorDevice>> {
        Ok(None)
    }

    /// Complete observation coverage is separate from ordinary execution support.
    /// Implementations must audit the exact route, including transfers/adapters.
    fn operation_observation_capability(
        &self,
        _query: &ProviderFeasibilityQuery,
    ) -> observation::ProviderObservationCapability {
        observation::ProviderObservationCapability::Unsupported
    }

    /// Begin before staging. Unsupported providers must never return an empty
    /// receipt as a substitute for missing asynchronous completion evidence.
    fn begin_operation_observation(
        &self,
        _query: &ProviderFeasibilityQuery,
    ) -> observation::ProviderObservationSupport {
        observation::ProviderObservationSupport::Unsupported
    }

    fn image_stage_observation_capability(
        &self,
        _work: ProviderImageStagePhysicalWork,
    ) -> observation::ProviderObservationCapability {
        observation::ProviderObservationCapability::Unsupported
    }

    fn begin_image_stage_observation(
        &self,
        _work: ProviderImageStagePhysicalWork,
    ) -> observation::ProviderObservationSupport {
        observation::ProviderObservationSupport::Unsupported
    }

    fn real_unary_observation_capability(
        &self,
        _recipe: &ProviderRealUnaryCompound,
        _continuations: ProviderRealUnaryContinuations,
    ) -> observation::ProviderObservationCapability {
        observation::ProviderObservationCapability::Unsupported
    }

    fn power_pair_observation_capability(
        &self,
        _recipe: &power_service::PowerPairCompound,
    ) -> observation::ProviderObservationCapability {
        observation::ProviderObservationCapability::Unsupported
    }

    fn power_scalar_observation_capability(
        &self,
        _projection: &power_service::PowerTypedScalarProjection,
    ) -> observation::ProviderObservationCapability {
        observation::ProviderObservationCapability::Unsupported
    }

    fn begin_power_scalar_observation(
        &self,
        _projection: &power_service::PowerTypedScalarProjection,
    ) -> observation::ProviderObservationSupport {
        observation::ProviderObservationSupport::Unsupported
    }

    fn begin_power_pair_observation(
        &self,
        _recipe: &power_service::PowerPairCompound,
    ) -> observation::ProviderObservationSupport {
        observation::ProviderObservationSupport::Unsupported
    }

    /// Same provider-owned progress/budget/fence authority as ordinary
    /// operations, but scoped to the complete sealed compound rather than a
    /// real-only operation query whose original domain remains Unknown.
    fn begin_real_unary_observation(
        &self,
        _recipe: &ProviderRealUnaryCompound,
        _continuations: ProviderRealUnaryContinuations,
    ) -> observation::ProviderObservationSupport {
        observation::ProviderObservationSupport::Unsupported
    }

    /// Determines whether a representation-specific operation can execute
    /// without attempting allocation, compilation, transfer, or dispatch.
    fn query_feasibility(&self, query: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        query.conservative_transfer_feasibility(self.precision())
    }

    /// Determines whether a native sparse operation can execute without allocation, transfer,
    /// compilation, or dispatch. Providers must advertise the matching sparse capability before
    /// returning support. The default rejects without attempting a dense or host fallback.
    fn query_sparse_feasibility(&self, _query: &SparseFeasibilityQuery) -> SparseFeasibility {
        SparseFeasibility::rejected(
            ProviderRejectionCode::UnsupportedOperation,
            "provider.sparse.unsupported",
        )
    }

    /// Returns a side-effect-free sparse cost estimate after feasibility succeeds.
    fn estimate_sparse_cost(&self, query: &SparseCostQuery) -> Option<ProviderCostEstimate> {
        sparse::unsupported_sparse_cost(query)
    }

    /// Uploads one validated canonical CSR pattern into provider-owned preparation storage.
    fn upload_sparse_pattern(
        &self,
        _pattern: HostCsrPatternView<'_>,
    ) -> anyhow::Result<SparsePatternHandle> {
        Err(anyhow!("sparse pattern upload not supported by provider"))
    }

    /// Binds a floating value revision to an already uploaded pattern.
    fn bind_sparse_values(
        &self,
        _pattern: &SparsePatternHandle,
        _values: &HostNumericTensorView<'_>,
    ) -> anyhow::Result<SparseMatrixHandle> {
        Err(anyhow!("sparse value binding not supported by provider"))
    }

    fn free_sparse_pattern(&self, _pattern: &SparsePatternHandle) -> anyhow::Result<()> {
        Err(anyhow!("sparse pattern release not supported by provider"))
    }

    fn free_sparse_matrix(&self, _matrix: &SparseMatrixHandle) -> anyhow::Result<()> {
        Err(anyhow!("sparse matrix release not supported by provider"))
    }

    fn sparse_spmv<'a>(
        &'a self,
        _request: &'a SparseSpmvRequest<'a>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        sparse::unsupported_sparse_future()
    }

    fn prepare_sparse_preconditioner<'a>(
        &'a self,
        _request: &'a SparsePreconditionerRequest<'a>,
    ) -> AccelProviderFuture<'a, SparsePreconditionerHandle> {
        sparse::unsupported_sparse_future()
    }

    /// Uploads a multigrid hierarchy for device-resident preconditioning (see
    /// [`SparseMultigridRequest`]). The default is unsupported.
    fn prepare_sparse_multigrid(
        &self,
        _request: &SparseMultigridRequest<'_>,
    ) -> anyhow::Result<SparseMultigridHandle> {
        Err(anyhow!("sparse multigrid not supported by provider"))
    }

    /// Applies an uploaded hierarchy to one residual (one upload, one download).
    fn sparse_multigrid_solve<'a>(
        &'a self,
        _request: &'a SparseMultigridSolveRequest<'a>,
    ) -> AccelProviderFuture<'a, SparseMultigridSolveResult> {
        Box::pin(async { Err(anyhow!("sparse multigrid not supported by provider")) })
    }

    fn free_sparse_multigrid(&self, _hierarchy: &SparseMultigridHandle) -> anyhow::Result<()> {
        Err(anyhow!(
            "sparse multigrid release not supported by provider"
        ))
    }

    fn free_sparse_preconditioner(
        &self,
        _preconditioner: &SparsePreconditionerHandle,
    ) -> anyhow::Result<()> {
        Err(anyhow!(
            "sparse preconditioner release not supported by provider"
        ))
    }

    fn apply_sparse_preconditioner<'a>(
        &'a self,
        _request: &'a SparsePreconditionerApplyRequest<'a>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        sparse::unsupported_sparse_future()
    }

    /// Executes the admitted Krylov method as one provider-owned resident operation. Implementors
    /// return actual synchronization and dispatch evidence; callers must not reconstruct it.
    fn sparse_krylov_solve<'a>(
        &'a self,
        _request: &'a SparseKrylovRequest<'a>,
    ) -> AccelProviderFuture<'a, SparseKrylovResult> {
        sparse::unsupported_sparse_future()
    }

    /// Exact physical publication stage for a conditional host computation
    /// restoring one numeric result to this provider. Default unavailable:
    /// generic transfer feasibility alone does not prove staging or a fence.
    fn numeric_upload_physical_evidence(
        &self,
        _query: &ProviderFeasibilityQuery,
    ) -> Option<ProviderNumericUploadPhysicalEvidence> {
        None
    }

    /// Pure physical stage of the same typed floating-cast owner used by
    /// unary_double/unary_single. No handle or execution permission is minted.
    fn floating_cast_physical_evidence(
        &self,
        _input: &ProviderRepresentation,
        _target: ProviderPrecision,
    ) -> Option<ProviderFloatingCastPhysicalEvidence> {
        None
    }

    /// The same exact one-result upload boundary for class-preserving host
    /// publication. This includes native integer and logical routes; a
    /// generic byte estimate is not evidence of staging or completion.
    fn upload_physical_evidence(
        &self,
        _query: &ProviderFeasibilityQuery,
    ) -> Option<ProviderUploadPhysicalEvidence> {
        None
    }

    /// Exact native numeric readback admission for a live source handle.
    /// Generic dense feasibility cannot certify a lazy view's owner-specific
    /// transpose/readback path. The default keeps that route unsupported.
    fn query_numeric_download_feasibility(
        &self,
        source: &GpuTensorHandle,
        query: &ProviderFeasibilityQuery,
    ) -> ProviderFeasibility {
        if !query.matches_numeric_download(source) || source.device_id() != self.device_id() {
            return ProviderFeasibility::rejected(
                ProviderRejectionCode::InvalidShape,
                "provider.download.exact_contract",
            );
        }
        self.query_feasibility(query)
    }

    /// Returns a side-effect-free component cost estimate for an already
    /// feasible operation. Providers may return `None` until they have a
    /// trustworthy prior or observation. `None` means unavailable pricing,
    /// not unsupported execution. Placement preserves that absence; only an
    /// explicitly supplied physical prior or authorized complete-route
    /// observation may replace it. Feasibility remains a separate query.
    fn estimate_cost(&self, _query: &ProviderCostQuery) -> Option<ProviderCostEstimate> {
        None
    }

    fn elementwise_geometry(&self) -> Option<ProviderElementwiseGeometry> {
        None
    }

    /// Read-only exact program-cache lookup. No compilation, upload or warmup.
    /// Providers without an audited key inventory return None, never Warm.
    fn preparation_snapshot(
        &self,
        _query: &ProviderFeasibilityQuery,
        _program: ProviderPreparationProgram<'_>,
    ) -> Option<ProviderPreparationSnapshot> {
        None
    }

    /// Returns a side-effect-free resource snapshot for placement admission.
    /// Providers with allocator/queue telemetry should override this default;
    /// the conservative snapshot exposes declared capacity when available,
    /// leaves unknown queue state explicit, and never probes the device.
    fn placement_resources(&self) -> runmat_execution::ProviderResourceSnapshot {
        let info = self.device_info_struct();
        let capacity = info.memory_bytes;
        runmat_execution::ProviderResourceSnapshot {
            device_id: self.device_id(),
            capacity_bytes: capacity,
            live_bytes: 0,
            reclaimable_bytes: 0,
            scratch_available_bytes: capacity,
            queue_depth: None,
            queue_limit: None,
            lost: false,
            epoch: 0,
        }
    }

    /// Export a shared GPU context handle, allowing downstream systems (plotting, visualization)
    /// to reuse the same device/queue without copying tensor data back to the host.
    fn export_context(&self, _kind: AccelContextKind) -> Option<AccelContextHandle> {
        None
    }

    /// Export a provider-owned WGPU buffer for zero-copy integrations.
    /// None certifies no allocation, materialization or dispatch was attempted.
    /// Every operational failure is an error, never a capability miss.
    #[cfg(feature = "wgpu")]
    fn export_wgpu_buffer(
        &self,
        _handle: &GpuTensorHandle,
    ) -> ProviderExecutionResult<Option<WgpuBufferRef>> {
        let _ = _handle;
        Ok(None)
    }

    /// Gather logical elements from `source` at the provided zero-based linear `indices`,
    /// materialising a dense tensor with the specified `output_shape`.
    ///
    /// Indices address MATLAB logical elements, not raw provider storage lanes. Providers that
    /// store complex values as interleaved lanes must copy both lanes for each selected element
    /// and preserve the source storage kind on the output handle.
    fn gather_linear(
        &self,
        _source: &GpuTensorHandle,
        _indices: &[u32],
        _output_shape: &[usize],
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "gather_linear not supported by provider",
        ))
    }

    /// Temporary host metadata allocated by these existing native indexing
    /// owners. None/Unavailable is unknown, not zero. A RHS selects functional
    /// scatter; no RHS selects gather. Retained registration is quoted apart.
    fn indexing_host_workspace_bound(
        &self,
        _source: &GpuTensorHandle,
        _rhs: Option<&GpuTensorHandle>,
    ) -> runmat_execution::ResourceByteBound {
        runmat_execution::ResourceByteBound::Unavailable(
            runmat_execution::ResourceBoundUnavailable::NoScratchBound,
        )
    }

    /// Same gather kernel with request-local publication authority. Missing
    /// support declines before allocation/submission, never ordinary-then-claim.
    fn gather_linear_admitted(
        &self,
        _source: &GpuTensorHandle,
        _indices: &[u32],
        _output_shape: &[usize],
        _admission: &dyn runmat_execution::resource::ProviderHostMetadataAdmission,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        Err(unsupported_provider_operation("provider has no admitted indexed gather").into())
    }

    /// Native functional emission is synchronous; device execution remains
    /// asynchronous. This port adds no erased future allocation per operation.
    fn scatter_linear_functional_admitted(
        &self,
        _target: &GpuTensorHandle,
        _indices: &[u32],
        _values: &GpuTensorHandle,
        _mapping: ProviderScatterValueMapping,
        _admission: &dyn runmat_execution::resource::ProviderHostMetadataAdmission,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        Err(
            unsupported_provider_operation("provider has no admitted functional indexed scatter")
                .into(),
        )
    }

    /// Scatter logical elements from `values` into `target` at the provided zero-based linear
    /// `indices`.
    ///
    /// Indices address MATLAB logical elements, not raw provider storage lanes. Providers must
    /// ensure `values` has one logical element per index, copy all lanes for the handle storage
    /// kind, and update `target` in place without changing its shape or storage.
    fn scatter_linear(
        &self,
        _target: &GpuTensorHandle,
        _indices: &[u32],
        _values: &GpuTensorHandle,
    ) -> anyhow::Result<()> {
        Err(unsupported_provider_operation(
            "scatter_linear not supported by provider",
        ))
    }

    /// Functional linear scatter: publish a fresh allocation while leaving
    /// both `target` and `values` (including their retained aliases) unchanged.
    /// Indices address logical elements, with all storage lanes copied. An
    /// unsupported result must occur before mutation or publication; an
    /// operational failure after work has begun must retain its typed cause.
    fn scatter_linear_functional<'a>(
        &'a self,
        _target: &'a GpuTensorHandle,
        _indices: &'a [u32],
        _values: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        unsupported_execution_future("functional scatter_linear not supported by provider")
    }

    /// Optional genuine GPU object/control authority, never inferred from metadata.
    fn gpu_device_control(&self) -> Option<&dyn ProviderDeviceControl> {
        None
    }

    /// Structured device information (optional to override). Default adapts from `device_info()`.
    fn device_info_struct(&self) -> ApiDeviceInfo {
        ApiDeviceInfo {
            device_id: self.device_id(),
            name: self.device_info(),
            vendor: String::new(),
            memory_bytes: None,
            backend: None,
        }
    }

    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }

    /// Physical matrix/diagonal launch geometry, when this provider exposes
    /// that kernel recipe. Absence is unknown, never a global tuning fallback.
    fn matmul_geometry(&self) -> Option<ProviderMatmulGeometry> {
        None
    }

    /// Read-only limits for an audited ordinary dimensional-sum route.
    fn sum_geometry(&self) -> Option<ProviderSumGeometry> {
        None
    }

    /// Read-only limits for the two-output dimensional min/max hook. None
    /// leaves physical work unresolved for automatic or managed placement.
    fn extrema_geometry(&self) -> Option<ProviderExtremaGeometry> {
        None
    }

    /// Limits of this provider's actual one-dimensional convolution shader.
    /// None does not authorize a generic or estimated device recipe.
    fn convolution_geometry(&self) -> Option<ProviderConvolutionGeometry> {
        None
    }

    /// Read-only limits of the actual typed imfilter shader. The default
    /// asserts no direct route and never infers one from generic compute.
    fn imfilter_geometry(&self) -> Option<ProviderImfilterGeometry> {
        None
    }

    /// Exact provider-owned host and input-stage evidence for the same typed
    /// imfilter hook query returned Supported by query_feasibility. None is
    /// unproven, never a zero-workspace assumption.
    fn imfilter_physical_evidence(
        &self,
        _query: &ProviderFeasibilityQuery,
    ) -> Option<ProviderImfilterPhysicalEvidence> {
        None
    }

    /// Read a single scalar at linear index from a device tensor, returning it as f64.
    fn read_scalar(&self, _h: &GpuTensorHandle, _linear_index: usize) -> anyhow::Result<f64> {
        Err(unsupported_provider_operation(
            "read_scalar not supported by provider",
        ))
    }

    /// Allocate a zero-initialised tensor with the provided shape on the device.
    fn zeros(&self, _shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "zeros not supported by provider",
        ))
    }

    /// Allocate a zero-initialised tensor with the provided shape and storage layout.
    ///
    /// `shape` is the logical MATLAB shape. Providers must allocate enough raw lanes for the
    /// requested storage kind, e.g. two interleaved numeric lanes per logical element for complex
    /// tensors.
    fn zeros_with_storage(
        &self,
        shape: &[usize],
        storage: GpuTensorStorage,
    ) -> anyhow::Result<GpuTensorHandle> {
        if storage == GpuTensorStorage::Real {
            self.zeros(shape)
        } else {
            Err(anyhow::anyhow!(
                "zeros_with_storage not supported by provider for {storage:?}"
            ))
        }
    }

    /// Allocate a one-initialized tensor with the provided shape on the device.
    /// None certifies no attempt before any provider
    /// effects; every Err retains its cause and is terminal, never a miss.
    fn ones(&self, _shape: &[usize]) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        Ok(None)
    }

    /// Allocate zeros with the requested shape and the prototype's exact class/storage.
    /// The prototype must be a live allocation owned by this provider.
    fn zeros_like(
        &self,
        _prototype: &GpuTensorHandle,
        _shape: &[usize],
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        Ok(None)
    }

    /// Allocate a constant logical tensor through the provider's logical producer authority.
    fn logical_fill(
        &self,
        _shape: &[usize],
        _value: bool,
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        Ok(None)
    }

    /// Allocate an exact native-integer zero buffer using the element class of
    /// `prototype`. Implementations must reject non-native-integer prototypes
    /// rather than allocating a floating compatibility buffer.
    fn zeros_integer_like(
        &self,
        _prototype: &GpuTensorHandle,
        _shape: &[usize],
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!(
            "zeros_integer_like not supported by provider"
        ))
    }

    /// Optional constant constructor, with the same pure-None contract as ones.
    /// A provider owns its concrete algorithm; the default does not try raw
    /// zeros/scalar_add and cannot infer no attempt from their errors.
    fn fill(
        &self,
        _shape: &[usize],
        _value: f64,
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        Ok(None)
    }

    /// Allocate a tensor filled with a constant value, matching a prototype's residency.
    fn fill_like(
        &self,
        prototype: &GpuTensorHandle,
        value: f64,
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        // Positive zero has an exact typed constructor. Negative zero is not
        // a zero-buffer identity: leave its sign-preserving adaptation to an
        // actual implementation or the caller's certified-absence fallback.
        if value.to_bits() == 0.0f64.to_bits() {
            return self.zeros_like(prototype, prototype.shape());
        }
        Ok(None)
    }

    /// Allocate a one-initialised tensor matching the prototype tensor.
    fn ones_like(
        &self,
        prototype: &GpuTensorHandle,
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        self.fill_like(prototype, 1.0)
    }

    /// Allocate an identity tensor with ones along the leading diagonal of the first two axes.
    fn eye(&self, _shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "eye not supported by provider",
        ))
    }

    /// Allocate an identity tensor matching the prototype tensor's shape.
    fn eye_like(&self, prototype: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        self.eye(prototype.shape())
    }

    /// Construct MATLAB-style coordinate grids from axis vectors.
    fn meshgrid(&self, _axes: &[MeshgridAxisView<'_>]) -> anyhow::Result<ProviderMeshgridResult> {
        Err(anyhow::anyhow!("meshgrid not supported by provider"))
    }

    /// Construct MATLAB-style N-D coordinate grids from resident GPU axis vectors.
    fn ndgrid(&self, _request: &ProviderNdgridRequest<'_>) -> anyhow::Result<ProviderNdgridResult> {
        Err(unsupported_provider_operation(
            "ndgrid not supported by provider",
        ))
    }

    /// Compute vectorized Black-Scholes European call and put prices on resident GPU inputs.
    fn black_scholes_price(
        &self,
        _request: &ProviderBlackScholesPriceRequest<'_>,
    ) -> anyhow::Result<ProviderBlackScholesPriceResult> {
        Err(unsupported_provider_operation(
            "black_scholes_price not supported by provider",
        ))
    }

    /// Apply the Adam optimizer update to resident parameter and state tensors.
    fn adam_update(
        &self,
        _request: &ProviderAdamUpdateRequest<'_>,
    ) -> anyhow::Result<ProviderAdamUpdateResult> {
        Err(unsupported_provider_operation(
            "adam_update not supported by provider",
        ))
    }

    /// Compute per-element cross-entropy loss terms for resident prediction and target tensors.
    fn crossentropy_terms(
        &self,
        _request: &ProviderCrossentropyRequest<'_>,
    ) -> anyhow::Result<ProviderCrossentropyResult> {
        Err(unsupported_provider_operation(
            "crossentropy_terms not supported by provider",
        ))
    }

    /// Construct a diagonal matrix from a vector-like tensor. `offset` matches MATLAB semantics.
    fn diag_from_vector(
        &self,
        _vector: &GpuTensorHandle,
        _offset: isize,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "diag_from_vector not supported by provider",
        ))
    }

    /// Construct a diagonal matrix with an explicit shape from a vector-like tensor.
    /// `offset` matches MATLAB semantics; values that do not fit inside the requested
    /// rectangle are ignored.
    fn diag_from_vector_sized(
        &self,
        _vector: &GpuTensorHandle,
        _offset: isize,
        _rows: usize,
        _cols: usize,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "diag_from_vector_sized not supported by provider",
        ))
    }

    /// Extract a diagonal from a matrix-like tensor. The result is always a column vector.
    fn diag_extract(
        &self,
        _matrix: &GpuTensorHandle,
        _offset: isize,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "diag_extract not supported by provider",
        ))
    }

    /// Apply a lower-triangular mask to the first two dimensions of a tensor.
    fn tril<'a>(
        &'a self,
        _matrix: &'a GpuTensorHandle,
        _offset: isize,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        unsupported_execution_future("tril not supported by provider")
    }

    /// Apply an upper-triangular mask to the first two dimensions of a tensor.
    fn triu<'a>(
        &'a self,
        _matrix: &'a GpuTensorHandle,
        _offset: isize,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        unsupported_execution_future("triu not supported by provider")
    }

    /// Evaluate a polynomial expressed by `coefficients` at each element in `points`.
    fn polyval(
        &self,
        _coefficients: &GpuTensorHandle,
        _points: &GpuTensorHandle,
        _options: &ProviderPolyvalOptions,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "polyval not supported by provider",
        ))
    }

    /// Fit a polynomial of degree `degree` to `(x, y)` samples. Optional weights must match `x`.
    fn polyfit<'a>(
        &'a self,
        _x: &'a GpuTensorHandle,
        _y: &'a GpuTensorHandle,
        _degree: usize,
        _weights: Option<&'a GpuTensorHandle>,
        _outputs: ProviderPolyfitOutputs,
    ) -> AccelProviderFuture<'a, ProviderPolyfitResult> {
        Box::pin(async move {
            Err(unsupported_provider_operation(
                "polyfit not supported by provider",
            ))
        })
    }

    /// Differentiate a polynomial represented as a vector of coefficients.
    fn polyder_single<'a>(
        &'a self,
        _polynomial: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            Err(unsupported_provider_operation(
                "polyder_single not supported by provider",
            ))
        })
    }

    /// Apply the product rule to polynomials `p` and `q`.
    fn polyder_product<'a>(
        &'a self,
        _p: &'a GpuTensorHandle,
        _q: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            Err(unsupported_provider_operation(
                "polyder_product not supported by provider",
            ))
        })
    }

    /// Apply the quotient rule to polynomials `u` and `v`.
    fn polyder_quotient<'a>(
        &'a self,
        _u: &'a GpuTensorHandle,
        _v: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, ProviderPolyderQuotient> {
        Box::pin(async move {
            Err(unsupported_provider_operation(
                "polyder_quotient not supported by provider",
            ))
        })
    }

    /// Integrate a polynomial represented as a vector of coefficients and append a constant term.
    fn polyint(
        &self,
        _polynomial: &GpuTensorHandle,
        _constant: f64,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "polyint not supported by provider",
        ))
    }

    /// Allocate a tensor filled with random values drawn from U(0, 1).
    fn random_uniform(&self, _shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("random_uniform not supported by provider"))
    }

    /// Allocate a tensor filled with random values matching the prototype shape.
    fn random_uniform_like(&self, prototype: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        self.random_uniform(prototype.shape())
    }

    /// Allocate a tensor filled with standard normal (mean 0, stddev 1) random values.
    fn random_normal(&self, _shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("random_normal not supported by provider"))
    }

    /// Allocate a tensor of standard normal values matching a prototype's shape.
    fn random_normal_like(&self, prototype: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        self.random_normal(prototype.shape())
    }

    /// Execute one explicitly reserved GPU child-stream range. This hook is
    /// separate from legacy process-global RNG methods: they cannot prove the
    /// session counter, effect phase, or finite metadata admission.
    fn propose_gpu_random(
        &self,
        _request: &random::GpuRandomRequest<'_>,
    ) -> ProviderExecutionResult<Option<random::GpuRandomPhysicalProposal>> {
        Ok(None)
    }

    fn execute_gpu_random(
        &self,
        _request: &random::GpuRandomRequest<'_>,
        _metadata: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        Err(unsupported_provider_operation("request-bound GPU random is not supported").into())
    }

    fn propose_gpu_integer(
        &self,
        _request: &random::GpuIntegerRequest<'_>,
    ) -> ProviderExecutionResult<Option<random::GpuRandomPhysicalProposal>> {
        Ok(None)
    }

    fn execute_gpu_integer(
        &self,
        _request: &random::GpuIntegerRequest<'_>,
        _metadata: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        Err(unsupported_provider_operation("request-bound GPU integer is not supported").into())
    }

    /// One physically settled, bounded rejection-sampling batch. Providers
    /// that cannot retain an unpublished output/mask leave this route
    /// unsupported before effects; callers may choose a host-mediated route
    /// only before committing the child range or submitting a batch.
    fn execute_gpu_integer_batch<'a>(
        &'a self,
        _request: random::GpuIntegerBatchRequest<'a>,
        _metadata: Option<&'a dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
        _readback_control: Option<
            std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>,
        >,
    ) -> ProviderExecutionFuture<'a, random::GpuIntegerBatchProgress> {
        unsupported_execution_future("bounded GPU integer batches are not supported")
    }

    fn propose_gpu_integer_batch(
        &self,
        _request: random::GpuIntegerBatchRequest<'_>,
    ) -> ProviderExecutionResult<Option<random::GpuIntegerBatchPhysicalProposal>> {
        Ok(None)
    }

    /// Exponentially-distributed random values with mean `mu`.
    fn random_exponential(&self, _mu: f64, _shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "random_exponential not supported by provider",
        ))
    }

    /// Normal random values with mean `mu` and standard deviation `sigma`.
    fn random_normrnd(
        &self,
        _mu: f64,
        _sigma: f64,
        _shape: &[usize],
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "random_normrnd not supported by provider",
        ))
    }

    /// Uniform random values on the interval `[a, b)`.
    fn random_unifrnd(
        &self,
        _a: f64,
        _b: f64,
        _shape: &[usize],
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("random_unifrnd not supported by provider"))
    }

    /// Peak temporary host workspace required while selecting stochastic
    /// evolution geometry, excluding the request and metadata retained in the
    /// returned selection. Callers account those owned values separately.
    ///
    /// A provider must not be asked to plan under a finite host budget unless
    /// it supplies a proven bound. The conservative default therefore keeps
    /// planning unavailable rather than assuming that a synchronous method is
    /// allocation-free.
    fn stochastic_evolution_planning_host_workspace(
        &self,
        _rank: usize,
    ) -> runmat_execution::ResourceByteBound {
        runmat_execution::ResourceByteBound::Unavailable(
            runmat_execution::ResourceBoundUnavailable::NoScratchBound,
        )
    }

    /// Host control allocation required to acknowledge exact retirement of
    /// each selected stochastic source upload. `Known` is also the provider's
    /// contract that its upload implementation consumes the armed typed
    /// witness through canonical submitted-work completion. Runtime must
    /// decline the optimized source before effects for every other result.
    fn stochastic_evolution_source_transfer_retirement_host_workspace(
        &self,
    ) -> runmat_execution::ResourceByteBound {
        runmat_execution::ResourceByteBound::Unavailable(
            runmat_execution::ResourceBoundUnavailable::NoScratchBound,
        )
    }

    /// Select immutable provider-owned tile geometry and its exact physical
    /// proposal before Runtime draws or uploads any noise. `None` is a
    /// certified pre-effect decline; execution accepts only this selected
    /// plan/proposal pair.
    fn plan_stochastic_evolution(
        &self,
        _request: &stochastic_evolution::ProviderStochasticEvolutionWorkRequest,
    ) -> ProviderExecutionResult<Option<stochastic_evolution::ProviderStochasticEvolutionSelection>>
    {
        Ok(None)
    }

    /// Execute one bounded tile of one already-selected evolution step.
    /// Runtime owns the ordered RNG turn and supplies the exact noise tile.
    /// The default is certified unsupported before effects.
    fn execute_stochastic_evolution_tile<'a>(
        &'a self,
        _request: stochastic_evolution::ProviderStochasticEvolutionTileRequest<'a>,
        _metadata: Option<&'a dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
        _completion_control: Option<
            std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>,
        >,
    ) -> ProviderExecutionFuture<'a, stochastic_evolution::ProviderStochasticEvolutionProgress>
    {
        unsupported_execution_future("caller-supplied stochastic evolution is not supported")
    }

    /// Set the provider RNG state to align with the host RNG.
    fn set_rng_state(&self, _state: u64) -> anyhow::Result<()> {
        Err(anyhow::anyhow!("set_rng_state not supported by provider"))
    }

    /// Generate a 2-D correlation kernel matching MATLAB's `fspecial` builtin.
    fn fspecial(&self, _request: &FspecialRequest) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "fspecial not supported by provider",
        ))
    }

    /// Evaluate the `peaks` test surface on an n×n grid spanning [-3,3]×[-3,3].
    /// Returns the Z matrix (n×n) as a GPU tensor.
    fn peaks(&self, _n: usize) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("peaks not supported by provider"))
    }

    /// Evaluate the `peaks` formula element-wise on caller-supplied GPU coordinate tensors.
    /// X and Y must have the same shape. Returns a Z tensor of the same shape.
    fn peaks_xy(
        &self,
        _x: &GpuTensorHandle,
        _y: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("peaks_xy not supported by provider"))
    }

    fn hann_window(&self, _len: usize, _periodic: bool) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "hann_window not supported by provider",
        ))
    }

    fn hamming_window(&self, _len: usize, _periodic: bool) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "hamming_window not supported by provider",
        ))
    }

    fn blackman_window(&self, _len: usize, _periodic: bool) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "blackman_window not supported by provider",
        ))
    }

    /// Apply an N-D correlation/convolution with padding semantics matching MATLAB's `imfilter`.
    fn imfilter<'a>(
        &'a self,
        _image: &'a GpuTensorHandle,
        _kernel: &'a GpuTensorHandle,
        _options: &'a ImfilterOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("imfilter not supported by provider")
    }

    /// Allocate a Double random permutation of 1..=n, returning the first k
    /// elements. A provider may certify Unsupported only before random-state,
    /// allocation, or dispatch effects; all other failures are terminal.
    fn random_permutation(&self, _n: usize, _k: usize) -> ProviderExecutionResult<GpuTensorHandle> {
        Err(unsupported_provider_operation("random_permutation not supported by provider").into())
    }

    /// Allocate a Double permutation on the prototype's device. The output
    /// class is Double even when the numeric prototype has another class.
    fn random_permutation_like(
        &self,
        _prototype: &GpuTensorHandle,
        n: usize,
        k: usize,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.random_permutation(n, k)
    }

    /// Compute a covariance matrix across the columns of `matrix`.
    fn covariance<'a>(
        &'a self,
        _matrix: &'a GpuTensorHandle,
        _second: Option<&'a GpuTensorHandle>,
        _weights: Option<&'a GpuTensorHandle>,
        _options: &'a CovarianceOptions,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async {
            Err(unsupported_provider_operation("covariance not supported by provider").into())
        })
    }

    /// Compute a correlation coefficient matrix across the columns of `matrix`.
    fn corrcoef<'a>(
        &'a self,
        _matrix: &'a GpuTensorHandle,
        _options: &'a CorrcoefOptions,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async {
            Err(unsupported_provider_operation("corrcoef not supported by provider").into())
        })
    }

    /// Convert a resident covariance matrix into correlation and sigma outputs.
    fn covariance_to_correlation(
        &self,
        _matrix: &GpuTensorHandle,
    ) -> anyhow::Result<ProviderCovarianceToCorrelationResult> {
        Err(unsupported_provider_operation(
            "covariance_to_correlation not supported by provider",
        ))
    }

    // Optional operator hooks (default to unsupported)
    /// Generate a real row sequence in the requested physical class. `None`
    /// means no native route was attempted; an attempted failure is terminal.
    /// The first and final values are supplied separately so exponentiation
    /// and signed-zero endpoints do not drift through device arithmetic.
    fn sequence(
        &self,
        _request: &ProviderSequenceRequest,
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        Ok(None)
    }

    fn linspace(&self, _start: f64, _stop: f64, _count: usize) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("linspace not supported by provider"))
    }
    fn elem_add<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        unsupported_execution_future("elem_add not supported by provider")
    }
    fn elem_mul<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        unsupported_execution_future("elem_mul not supported by provider")
    }
    fn elem_max<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        unsupported_execution_future("elem_max not supported by provider")
    }
    /// Select between resident real tensors using a resident logical mask.
    /// Selection does not perform arithmetic blending, so an unselected NaN or
    /// infinity cannot contaminate the result. Providers must not download
    /// operands to the host to implement this hook.
    fn select<'a>(
        &'a self,
        _mask: &'a GpuTensorHandle,
        _then_value: &'a GpuTensorHandle,
        _else_value: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("select not supported by provider")
    }
    fn elem_min<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        unsupported_execution_future("elem_min not supported by provider")
    }
    fn elem_sub<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        unsupported_execution_future("elem_sub not supported by provider")
    }
    fn elem_div<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        unsupported_execution_future("elem_div not supported by provider")
    }
    /// Compute an exact MATLAB-style remainder for compatible native integer tensors.
    fn elem_rem<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        unsupported_execution_future("elem_rem not supported by provider")
    }
    /// Compute an exact MATLAB-style modulus for compatible native integer tensors.
    fn elem_mod<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        unsupported_execution_future("elem_mod not supported by provider")
    }
    fn elem_pow<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        unsupported_execution_future("elem_pow not supported by provider")
    }

    /// Classify the actual broadcasted pairs of real floating inputs. The
    /// result is scoped to these unchanged inputs and is not an extrema proof.
    fn classify_real_power<'a>(
        &'a self,
        _base: &'a GpuTensorHandle,
        _exponent: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, PowerResultDomain> {
        unsupported_future("classify_real_power not supported by provider")
    }

    /// Compute real-input power in its already-resolved result domain. Real
    /// NaNs propagate even for (1, NaN)/(NaN, 0); complex output is interleaved
    /// principal-branch storage. No residual threshold may change its domain.
    fn real_power_in_domain<'a>(
        &'a self,
        _base: &'a GpuTensorHandle,
        _exponent: &'a GpuTensorHandle,
        _domain: PowerResultDomain,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("real_power_in_domain not supported by provider")
    }

    /// Construct complex-interleaved GPU storage from a real-valued tensor,
    /// using zero for the imaginary lane.
    fn complex_from_real<'a>(
        &'a self,
        _real: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("complex_from_real not supported by provider")
    }

    /// Construct complex-interleaved GPU storage from real and imaginary tensors.
    ///
    /// Implementations should support equal shapes and scalar expansion for either
    /// operand, matching MATLAB's `complex(real, imag)` size rules.
    fn complex_from_real_imag<'a>(
        &'a self,
        _real: &'a GpuTensorHandle,
        _imag: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("complex_from_real_imag not supported by provider")
    }

    /// Map a resident real-valued symbol tensor through a complex constellation
    /// table and return complex-interleaved GPU storage.
    fn modulate_constellation<'a>(
        &'a self,
        _request: ProviderModulationRequest<'a>,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            Err(
                unsupported_provider_operation("modulate_constellation not supported by provider")
                    .into(),
            )
        })
    }

    /// Group a resident real/logical bit tensor into symbols, map through a complex
    /// constellation table, and return complex-interleaved GPU storage.
    fn modulate_bits_constellation<'a>(
        &'a self,
        _request: ProviderBitModulationRequest<'a>,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            Err(unsupported_provider_operation(
                "modulate_bits_constellation not supported by provider",
            )
            .into())
        })
    }

    fn elem_hypot<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("elem_hypot not supported by provider")
    }
    fn elem_ge<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async {
            Err(unsupported_provider_operation("elem_ge not supported by provider").into())
        })
    }
    fn elem_le<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async {
            Err(unsupported_provider_operation("elem_le not supported by provider").into())
        })
    }
    fn elem_lt<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async {
            Err(unsupported_provider_operation("elem_lt not supported by provider").into())
        })
    }
    fn elem_gt<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async {
            Err(unsupported_provider_operation("elem_gt not supported by provider").into())
        })
    }
    fn elem_eq<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async {
            Err(unsupported_provider_operation("elem_eq not supported by provider").into())
        })
    }
    fn elem_ne<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async {
            Err(unsupported_provider_operation("elem_ne not supported by provider").into())
        })
    }
    fn logical_and(
        &self,
        _a: &GpuTensorHandle,
        _b: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "logical_and not supported by provider",
        ))
    }
    fn logical_or(
        &self,
        _a: &GpuTensorHandle,
        _b: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "logical_or not supported by provider",
        ))
    }
    fn logical_xor(
        &self,
        _a: &GpuTensorHandle,
        _b: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "logical_xor not supported by provider",
        ))
    }
    fn logical_not(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "logical_not not supported by provider",
        ))
    }
    fn logical_islogical(&self, a: &GpuTensorHandle) -> anyhow::Result<bool> {
        Ok(handle_is_logical(a))
    }
    fn logical_isreal(&self, _a: &GpuTensorHandle) -> anyhow::Result<bool> {
        Err(anyhow::anyhow!("logical_isreal not supported by provider"))
    }
    fn logical_isfinite(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "logical_isfinite not supported by provider",
        ))
    }
    fn logical_isnan(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "logical_isnan not supported by provider",
        ))
    }
    fn logical_isinf(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "logical_isinf not supported by provider",
        ))
    }
    fn elem_atan2<'a>(
        &'a self,
        _y: &'a GpuTensorHandle,
        _x: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("elem_atan2 not supported by provider")
    }
    // Unary elementwise operations (optional)
    fn unary_sin<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_sin not supported by provider")
    }
    fn unary_sinc<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_sinc not supported by provider")
    }
    fn unary_gamma<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_gamma not supported by provider")
    }
    fn unary_gammaln<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_gammaln not supported by provider")
    }
    fn unary_erf<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_erf not supported by provider")
    }
    fn unary_erfcinv<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_erfcinv not supported by provider")
    }
    fn unary_factorial<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_factorial not supported by provider")
    }
    fn unary_asinh<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_asinh not supported by provider")
    }
    fn unary_sinh<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_sinh not supported by provider")
    }
    fn unary_cosh<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_cosh not supported by provider")
    }
    fn unary_asin<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_asin not supported by provider")
    }
    fn unary_acos<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_acos not supported by provider")
    }
    fn unary_acosh<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_acosh not supported by provider")
    }
    fn unary_tan<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_tan not supported by provider")
    }
    fn unary_tanh<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_tanh not supported by provider")
    }
    fn unary_atan<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_atan not supported by provider")
    }
    fn unary_atanh<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_atanh not supported by provider")
    }
    fn unary_ceil<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_ceil not supported by provider")
    }
    fn unary_floor<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_floor not supported by provider")
    }
    fn unary_round<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_round not supported by provider")
    }
    fn round_digits<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _digits: i32,
        _significant: bool,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async {
            Err(unsupported_provider_operation("round_digits not supported by provider").into())
        })
    }
    fn unary_fix<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_fix not supported by provider")
    }
    fn unary_cos<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_cos not supported by provider")
    }
    fn unary_angle<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_angle not supported by provider")
    }
    fn unary_imag<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_imag not supported by provider")
    }
    fn unary_real<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_real not supported by provider")
    }
    fn unary_conj<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        // None certifies no attempt; every Err remains an owning terminal failure.
        Box::pin(async { Ok(None) })
    }
    fn unary_abs<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_abs not supported by provider")
    }
    fn unary_sign<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_sign not supported by provider")
    }
    fn unary_heaviside<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_heaviside not supported by provider")
    }
    fn unary_exp<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_exp not supported by provider")
    }
    /// Apply ReLU while preserving NaN inputs, including their native precision.
    fn activation_relu<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("activation_relu not supported by provider")
    }
    /// Apply an exponential linear unit activation with the supplied alpha.
    fn activation_elu<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _alpha: f64,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("activation_elu not supported by provider")
    }
    /// Pure capability query used before any forward or backward provider
    /// effects. Backends must not report support for a precision they cannot
    /// execute without gathering or relabeling.
    fn activation_adjoint_supported(
        &self,
        _operation: ProviderActivationAdjoint,
        _precision: ProviderPrecision,
    ) -> bool {
        false
    }
    fn activation_adjoint<'a>(
        &'a self,
        _input: &'a GpuTensorHandle,
        _upstream: &'a GpuTensorHandle,
        _operation: ProviderActivationAdjoint,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("activation_adjoint not supported by provider")
    }
    /// Normalize each row of a real two-dimensional tensor with stable softmax.
    fn activation_softmax_rows<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async {
            Err(
                unsupported_provider_operation("activation_softmax_rows not supported by provider")
                    .into(),
            )
        })
    }
    fn unary_expm1<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_expm1 not supported by provider")
    }
    fn unary_log<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_log not supported by provider")
    }
    fn unary_log2<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_log2 not supported by provider")
    }
    fn unary_log10<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_log10 not supported by provider")
    }
    fn unary_log1p<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_log1p not supported by provider")
    }
    fn unary_sqrt<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_sqrt not supported by provider")
    }
    fn unary_double<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async {
            Err(unsupported_provider_operation("unary_double not supported by provider").into())
        })
    }
    fn unary_double_admitted<'a>(
        &'a self,
        value: &'a GpuTensorHandle,
        admission: Option<&'a dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        if admission.is_some() {
            Box::pin(async {
                Err(unsupported_provider_operation("admitted unary_double is unsupported").into())
            })
        } else {
            self.unary_double(value)
        }
    }
    fn unary_single<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async {
            Err(unsupported_provider_operation("unary_single not supported by provider").into())
        })
    }
    fn unary_single_admitted<'a>(
        &'a self,
        value: &'a GpuTensorHandle,
        admission: Option<&'a dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        if admission.is_some() {
            Box::pin(async {
                Err(unsupported_provider_operation("admitted unary_single is unsupported").into())
            })
        } else {
            self.unary_single(value)
        }
    }
    fn unary_pow2<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_pow2 not supported by provider")
    }
    fn unary_nextpow2<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_nextpow2 not supported by provider")
    }
    fn pow2_scale(
        &self,
        _mantissa: &GpuTensorHandle,
        _exponent: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("pow2_scale not supported by provider"))
    }
    // Left-scalar operations (broadcast with scalar on the left)
    fn scalar_rsub(&self, _a: &GpuTensorHandle, _scalar: f64) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("scalar_rsub not supported by provider"))
    }
    fn scalar_rdiv(&self, _a: &GpuTensorHandle, _scalar: f64) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("scalar_rdiv not supported by provider"))
    }
    // Scalar operations: apply op with scalar right-hand side (broadcast over a)
    fn scalar_add(&self, _a: &GpuTensorHandle, _scalar: f64) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "scalar_add not supported by provider",
        ))
    }
    fn scalar_sub(&self, _a: &GpuTensorHandle, _scalar: f64) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("scalar_sub not supported by provider"))
    }
    fn scalar_mul(&self, _a: &GpuTensorHandle, _scalar: f64) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "scalar_mul not supported by provider",
        ))
    }
    fn scalar_max(&self, _a: &GpuTensorHandle, _scalar: f64) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "scalar_max not supported by provider",
        ))
    }
    fn scalar_min(&self, _a: &GpuTensorHandle, _scalar: f64) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "scalar_min not supported by provider",
        ))
    }
    fn scalar_div(&self, _a: &GpuTensorHandle, _scalar: f64) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("scalar_div not supported by provider"))
    }
    /// None certifies no staging, allocation, submission, or execution attempt.
    /// A failure is terminal and never authorizes host replay.
    fn sort_dim<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dim: usize,
        _order: SortOrder,
        _comparison: SortComparison,
    ) -> ProviderExecutionFuture<'a, Option<SortResult>> {
        Box::pin(async { Ok(None) })
    }
    /// None certifies no staging, allocation, submission, or execution attempt.
    /// A failure is terminal and never authorizes host replay.
    fn sort_rows<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _columns: &'a [SortRowsColumnSpec],
        _comparison: SortComparison,
    ) -> ProviderExecutionFuture<'a, Option<SortResult>> {
        Box::pin(async { Ok(None) })
    }
    fn matmul<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("matmul not supported by provider")
    }

    fn syrk(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("syrk not supported by provider"))
    }
    fn pagefun(&self, _request: &PagefunRequest) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "pagefun not supported by provider",
        ))
    }

    /// Optional: matrix multiplication with an epilogue applied before store.
    ///
    /// The default implementation falls back to `matmul` when the epilogue is effectively a no-op
    /// (alpha=1, beta=0, no row/col scales), and otherwise returns `Err`.
    fn matmul_epilogue<'a>(
        &'a self,
        a: &'a GpuTensorHandle,
        b: &'a GpuTensorHandle,
        epilogue: &'a MatmulEpilogue,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if epilogue.is_noop() {
                return self
                    .matmul(a, b)
                    .await
                    .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects());
            }
            Err(unsupported_provider_operation("matmul_epilogue not supported by provider").into())
        })
    }
    fn image_normalize<'a>(
        &'a self,
        _input: &'a GpuTensorHandle,
        _desc: &'a ImageNormalizeDescriptor,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("image_normalize fusion not supported by provider")
    }
    fn matmul_power_step<'a>(
        &'a self,
        _lhs: &'a GpuTensorHandle,
        _rhs: &'a GpuTensorHandle,
        _epilogue: &'a PowerStepEpilogue,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("matmul_power_step normalization not supported by provider")
    }
    fn linsolve<'a>(
        &'a self,
        _lhs: &'a GpuTensorHandle,
        _rhs: &'a GpuTensorHandle,
        _options: &'a ProviderLinsolveOptions,
    ) -> ProviderExecutionFuture<'a, ProviderLinsolveResult> {
        Box::pin(async {
            Err(unsupported_provider_operation("linsolve not supported by provider").into())
        })
    }
    fn inv<'a>(
        &'a self,
        _matrix: &'a GpuTensorHandle,
        _options: ProviderInvOptions,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        unsupported_execution_future("inv not supported by provider")
    }
    fn pinv<'a>(
        &'a self,
        _matrix: &'a GpuTensorHandle,
        _options: ProviderPinvOptions,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        unsupported_execution_future("pinv not supported by provider")
    }
    fn cond<'a>(
        &'a self,
        _matrix: &'a GpuTensorHandle,
        _norm: ProviderCondNorm,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        unsupported_execution_future("cond not supported by provider")
    }
    fn norm<'a>(
        &'a self,
        _tensor: &'a GpuTensorHandle,
        _order: ProviderNormOrder,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        unsupported_execution_future("norm not supported by provider")
    }
    fn interp1<'a>(
        &'a self,
        _request: &'a ProviderInterp1Request<'a>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("interp1 not supported by provider")
    }
    fn rank<'a>(
        &'a self,
        _matrix: &'a GpuTensorHandle,
        _tolerance: Option<f64>,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        unsupported_execution_future("rank not supported by provider")
    }
    fn rcond<'a>(
        &'a self,
        _matrix: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        unsupported_execution_future("rcond not supported by provider")
    }
    fn mldivide<'a>(
        &'a self,
        _lhs: &'a GpuTensorHandle,
        _rhs: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            Err(unsupported_provider_operation("mldivide not supported by provider").into())
        })
    }
    fn mrdivide<'a>(
        &'a self,
        _lhs: &'a GpuTensorHandle,
        _rhs: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            Err(unsupported_provider_operation("mrdivide not supported by provider").into())
        })
    }
    fn eig<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _compute_left: bool,
    ) -> ProviderExecutionFuture<'a, ProviderEigResult> {
        Box::pin(async move {
            Err(unsupported_provider_operation("eig not supported by provider").into())
        })
    }
    fn lu<'a>(&'a self, _a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, ProviderLuResult> {
        unsupported_future("lu not supported by provider")
    }

    fn chol<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _lower: bool,
    ) -> ProviderExecutionFuture<'a, ProviderCholResult> {
        Box::pin(async {
            Err(unsupported_provider_operation("chol not supported by provider").into())
        })
    }
    fn qr<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _options: ProviderQrOptions,
    ) -> ProviderExecutionFuture<'a, ProviderQrResult> {
        Box::pin(async {
            Err(unsupported_provider_operation("qr not supported by provider").into())
        })
    }
    fn take_matmul_sources(
        &self,
        _product: &GpuTensorHandle,
    ) -> Option<(GpuTensorHandle, GpuTensorHandle)> {
        None
    }
    fn qr_power_iter<'a>(
        &'a self,
        product: &'a GpuTensorHandle,
        _product_lhs: Option<&'a GpuTensorHandle>,
        q_handle: &'a GpuTensorHandle,
        options: &'a ProviderQrOptions,
    ) -> ProviderExecutionFuture<'a, Option<ProviderQrPowerIterResult>> {
        let _ = (product, q_handle, options);
        Box::pin(async move { Ok(None) })
    }
    fn transpose(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "transpose not supported by provider",
        ))
    }
    fn conv1d(
        &self,
        _signal: &GpuTensorHandle,
        _kernel: &GpuTensorHandle,
        _options: ProviderConv1dOptions,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("conv1d not supported by provider"))
    }
    fn conv2d(
        &self,
        _signal: &GpuTensorHandle,
        _kernel: &GpuTensorHandle,
        _mode: ProviderConvMode,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("conv2d not supported by provider"))
    }
    fn iir_filter<'a>(
        &'a self,
        _coefficients: ProviderIirCoefficientSource<'a>,
        _x: &'a GpuTensorHandle,
        _options: ProviderIirFilterOptions,
    ) -> AccelProviderFuture<'a, ProviderIirFilterResult> {
        Box::pin(async move {
            Err(unsupported_provider_operation(
                "iir_filter not supported by provider",
            ))
        })
    }

    /// Submit a complete admitted zero-phase IIR route. Unsupported must be
    /// reported before any provider effect; an operational failure after staging
    /// must never be interpreted as permission to replay on the host.
    fn iir_zero_phase<'a>(
        &'a self,
        _request: &'a ProviderIirZeroPhaseRequest<'a>,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async {
            Err(ProviderExecutionError::from(
                unsupported_provider_operation("iir_zero_phase not supported by provider"),
            ))
        })
    }
    fn uniform_spectral_estimate<'a>(
        &'a self,
        _request: &'a ProviderSpectralRequest<'a>,
    ) -> ProviderExecutionFuture<'a, ProviderSpectralResult> {
        Box::pin(async {
            Err(unsupported_provider_operation(
                "uniform_spectral_estimate not supported by provider",
            )
            .into())
        })
    }
    fn signal_envelope<'a>(
        &'a self,
        _request: &'a ProviderEnvelopeRequest<'a>,
    ) -> ProviderExecutionFuture<'a, ProviderEnvelopeResult> {
        Box::pin(async {
            Err(unsupported_provider_operation("signal_envelope not supported by provider").into())
        })
    }
    /// Pure request-local admission for the optional envelope hook. `false`
    /// permits host selection before provider effects; every error after an
    /// admitted hook invocation is terminal, including Unsupported.
    fn supports_signal_envelope(&self, _request: &ProviderEnvelopeRequest<'_>) -> bool {
        false
    }
    /// Pure request-local admission for the optional analytic-signal hook.
    /// `false` certifies that no Hilbert hook was invoked, so an owning-runtime
    /// host implementation may be selected before provider effects. An
    /// implementation returning `true` must treat every later hook failure as
    /// terminal, including a reported Unsupported error after staging.
    fn supports_signal_hilbert(&self, _request: &ProviderHilbertRequest<'_>) -> bool {
        false
    }

    fn signal_hilbert<'a>(
        &'a self,
        _request: &'a ProviderHilbertRequest<'a>,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async {
            Err(unsupported_provider_operation("signal_hilbert not supported by provider").into())
        })
    }
    /// Reorder tensor dimensions according to zero-based `order`.
    /// None certifies no attempt or provider effects; an error never authorizes
    /// an implicit host replay, even when its cause reports Unsupported.
    fn permute(
        &self,
        _handle: &GpuTensorHandle,
        _order: &[usize],
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        Ok(None)
    }
    fn flip(
        &self,
        _handle: &GpuTensorHandle,
        _axes: &[usize],
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        // None certifies no staging, allocation or execution attempt.
        Ok(None)
    }
    /// Read-only admission for the legacy unphased `circshift` hook. `false`
    /// proves only that no provider call was made and permits host fallback;
    /// after `true`, any hook error is terminal because its effects are unknown.
    /// Providers must include actual class, shape, layout and shift limits.
    fn supports_circshift(&self, _handle: &GpuTensorHandle, _shifts: &[isize]) -> bool {
        false
    }

    fn circshift(
        &self,
        _handle: &GpuTensorHandle,
        _shifts: &[isize],
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("circshift not supported by provider"))
    }
    /// `None` certifies no staging, allocation, submission, or execution occurred.
    /// Errors are terminal and never authorize replay of this operation.
    fn diff_dim(
        &self,
        _handle: &GpuTensorHandle,
        _order: usize,
        _dim: usize,
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        Ok(None)
    }
    /// `None` certifies no staging, allocation, submission, or execution occurred.
    /// Errors are terminal and never authorize replay of this operation.
    fn gradient_dim(
        &self,
        _handle: &GpuTensorHandle,
        _dim: usize,
        _spacing: f64,
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        Ok(None)
    }
    fn gradient_dim_with_coordinates(
        &self,
        _handle: &GpuTensorHandle,
        _dim: usize,
        _coordinates: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "gradient_dim_with_coordinates not supported by provider",
        ))
    }
    /// Transform a zero-based dimension, optionally padding/truncating to `len`.
    /// The borrowed input and its aliases must remain unchanged. Return a fresh
    /// physical result for a nontrivial transform; an immutable view alias is
    /// permitted for an unchanged complex unit-length axis (no padding or
    /// truncation), with the same native class/storage and correct output shape.
    fn fft_dim<'a>(
        &'a self,
        _handle: &'a GpuTensorHandle,
        _len: Option<usize>,
        _dim: usize,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async {
            Err(unsupported_provider_operation("fft_dim not supported by provider").into())
        })
    }
    fn ifft_dim<'a>(
        &'a self,
        _handle: &'a GpuTensorHandle,
        _len: Option<usize>,
        _dim: usize,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async {
            Err(unsupported_provider_operation("ifft_dim not supported by provider").into())
        })
    }
    fn fft_extract_real<'a>(
        &'a self,
        _handle: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async {
            Err(unsupported_provider_operation("fft_extract_real not supported by provider").into())
        })
    }
    /// None certifies no staging, allocation, submission, or execution attempt.
    /// A failure is terminal and never authorizes host replay.
    fn unique<'a>(
        &'a self,
        _handle: &'a GpuTensorHandle,
        _options: &'a UniqueOptions,
    ) -> ProviderExecutionFuture<'a, Option<UniqueResult>> {
        Box::pin(async { Ok(None) })
    }
    /// None certifies no staging, allocation, submission, or execution attempt.
    /// A failure is terminal and never authorizes host replay.
    fn union<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
        _options: &'a UnionOptions,
    ) -> ProviderExecutionFuture<'a, Option<UnionResult>> {
        Box::pin(async { Ok(None) })
    }
    /// None certifies no staging, allocation, submission, or execution attempt.
    /// A failure is terminal and never authorizes host replay.
    fn setdiff<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
        _options: &'a SetdiffOptions,
    ) -> ProviderExecutionFuture<'a, Option<SetdiffResult>> {
        Box::pin(async { Ok(None) })
    }
    /// None certifies no staging, allocation, submission, or execution attempt.
    /// A failure is terminal and never authorizes host replay.
    fn ismember<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
        _options: &'a IsMemberOptions,
    ) -> ProviderExecutionFuture<'a, Option<IsMemberResult>> {
        Box::pin(async { Ok(None) })
    }
    fn reshape(
        &self,
        handle: &GpuTensorHandle,
        new_shape: &[usize],
    ) -> anyhow::Result<GpuTensorHandle> {
        if handle_is_transposed(handle) {
            return Err(unsupported_provider_operation(
                "reshape of a lazy transpose requires provider materialization",
            ));
        }
        let count = |shape: &[usize]| -> anyhow::Result<usize> {
            if shape.contains(&0) {
                return Ok(0);
            }
            shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or_else(|| anyhow!("reshape size overflow"))
        };
        if count(handle.shape())? != count(new_shape)? {
            return Err(anyhow!("reshape must preserve element count"));
        }
        Ok(handle.with_view(handle.view().clone().with_shape(new_shape.to_vec()))?)
    }
    /// Concatenate the provided tensors along the 1-based dimension `dim`.
    fn cat(&self, _dim: usize, _inputs: &[GpuTensorHandle]) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("cat not supported by provider"))
    }
    /// Concatenate resident tensors. Only a certified pre-effect Unsupported
    /// response permits the caller to retry through a host implementation.
    fn cat_optional<'a>(
        &'a self,
        _dim: usize,
        _inputs: &'a [GpuTensorHandle],
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        unsupported_execution_future("cat not supported by provider")
    }
    /// Tile storage without changing its numerical class. None certifies that
    /// no staging, allocation or submission was attempted for this request.
    /// Failures retain their cause and effect phase; they are not no-attempt.
    fn repmat(
        &self,
        _handle: &GpuTensorHandle,
        _reps: &[usize],
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        Ok(None)
    }
    /// Compute the Kronecker product of two tensors, matching MATLAB semantics.
    fn kron<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        unsupported_execution_future("kron not supported by provider")
    }
    /// Compute the cross product of 3-element vectors along a matching dimension.
    fn cross(
        &self,
        _lhs: &GpuTensorHandle,
        _rhs: &GpuTensorHandle,
        _dim: Option<usize>,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "cross not supported by provider",
        ))
    }
    /// Certify a known-unavailable double sum before a caller stages other
    /// provider work. The default cannot infer capability from the provider's
    /// preferred precision and leaves the existing hook attempt unchanged.
    fn preflight_double_sum(&self, _a: &GpuTensorHandle) -> ProviderDoubleSumPreflight {
        ProviderDoubleSumPreflight::Unknown
    }

    /// Ordinary sum accumulates and returns F64 physical values for real,
    /// logical, and integer inputs. Complex floating input retains complex
    /// storage with F64 components. Public `single`/`native` output adaptation
    /// happens after this numerical boundary; unsupported devices must return
    /// a typed unsupported-operation error before a lossy reduction.
    fn reduce_sum<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async {
            Err(unsupported_provider_operation("reduce_sum not supported by provider").into())
        })
    }
    /// Dimensional form of [`Self::reduce_sum`] with the reduced axis kept as
    /// a singleton. Its intermediate class has the same F64 contract.
    fn reduce_sum_dim<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dim: usize,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async {
            Err(unsupported_provider_operation("reduce_sum_dim not supported by provider").into())
        })
    }
    /// Pure admission for the real, signal-precision dimensional sum used by
    /// resident adjoints. This is not MATLAB's F64-accumulating public sum.
    fn real_sum_native_dim_supported(&self, _precision: ProviderPrecision) -> bool {
        false
    }
    /// Sum a real floating tensor along a zero-based dimension, retaining its
    /// F32 or F64 physical class and the reduced singleton axis. Providers
    /// must return a distinct live allocation when the shape changes.
    fn reduce_real_sum_native_dim<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dim: usize,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async {
            Err(unsupported_provider_operation("native real sum not supported by provider").into())
        })
    }
    /// Reduce a native integer gpuArray while preserving its exact element
    /// class. This is intentionally separate from `reduce_sum`, whose MATLAB
    /// default output semantics are floating point for integer inputs.
    fn reduce_integer_sum_native<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("reduce_integer_sum_native not supported by provider")
    }
    /// Dimension form of [`Self::reduce_integer_sum_native`].
    fn reduce_integer_sum_native_dim<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("reduce_integer_sum_native_dim not supported by provider")
    }
    fn dot<'a>(
        &'a self,
        _lhs: &'a GpuTensorHandle,
        _rhs: &'a GpuTensorHandle,
        _dim: Option<usize>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("dot not supported by provider")
    }
    /// `None` certifies no staging, allocation, submission, or execution occurred.
    /// Errors are terminal and never authorize replay of this operation.
    fn reduce_nnz<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async { Ok(None) })
    }
    /// `None` certifies no staging, allocation, submission, or execution occurred.
    /// Errors are terminal and never authorize replay of this operation.
    fn reduce_nnz_dim<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dim: usize,
    ) -> ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async { Ok(None) })
    }
    /// `None` certifies no staging, allocation, submission, or execution occurred.
    /// Errors are terminal and never authorize replay of this operation.
    fn reduce_prod<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async { Ok(None) })
    }
    /// `None` certifies no staging, allocation, submission, or execution occurred.
    /// Errors are terminal and never authorize replay of this operation.
    fn reduce_prod_dim<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dim: usize,
    ) -> ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async { Ok(None) })
    }
    /// Reduce a native integer gpuArray by product while preserving its exact
    /// element class, distinct from MATLAB's default floating-point output.
    /// `None` certifies no staging, allocation, submission, or execution occurred.
    /// Errors are terminal and never authorize replay of this operation.
    fn reduce_integer_prod_native<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async { Ok(None) })
    }
    /// Dimension form of [`Self::reduce_integer_prod_native`].
    /// `None` certifies no staging, allocation, submission, or execution occurred.
    /// Errors are terminal and never authorize replay of this operation.
    fn reduce_integer_prod_native_dim<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dim: usize,
    ) -> ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async { Ok(None) })
    }
    /// Compute MATLAB's explicit native-output mean for a native integer
    /// gpuArray. Providers must avoid floating-point accumulation so int64 and
    /// uint64 values remain exact before the final class-preserving rounding.
    fn reduce_integer_mean_native<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("reduce_integer_mean_native not supported by provider")
    }
    /// Dimension form of [`Self::reduce_integer_mean_native`].
    fn reduce_integer_mean_native_dim<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("reduce_integer_mean_native_dim not supported by provider")
    }
    /// Multi-dimension form of [`Self::reduce_integer_mean_native`]. All
    /// requested zero-based dimensions must be reduced in one logical pass so
    /// native integer output rounds only once.
    fn reduce_integer_mean_native_dims<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dims_zero_based: &'a [usize],
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("reduce_integer_mean_native_dims not supported by provider")
    }
    /// Convert a real gpuArray to exact native integer storage while preserving
    /// device residency. Floating-point inputs use MATLAB-compatible saturating
    /// rounding; native integer inputs use exact class-to-class clamping.
    fn cast_to_integer<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _target: IntegerElementType,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async {
            Err(unsupported_provider_operation("cast_to_integer not supported by provider").into())
        })
    }
    /// Optional global mean. None certifies no attempt, before provider effects;
    /// every error retains its cause and is terminal, not a capability miss.
    fn reduce_mean<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async { Ok(None) })
    }
    /// Reduce mean across multiple zero-based dimensions in one device pass.
    /// None certifies no attempt before effects; every error is terminal.
    fn reduce_mean_nd<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dims_zero_based: &'a [usize],
    ) -> ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async { Ok(None) })
    }
    fn deep_learning_global_average_pool_1d<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _request: DeepLearningPool1DRequest,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("deep-learning global-average-pool-1d not supported by provider")
    }
    /// Adjoint of a physical pooling reduction. `upstream` retains the
    /// reduced axis as a singleton; presentation is inverted by the caller.
    fn deep_learning_global_average_pool_1d_adjoint<'a>(
        &'a self,
        _input: &'a GpuTensorHandle,
        _upstream: &'a GpuTensorHandle,
        _request: DeepLearningPool1DRequest,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("deep-learning global-average-pool-1d adjoint not supported by provider")
    }
    /// Pure derivative admission, including authentic owner, class, input
    /// geometry, and the singleton-axis upstream descriptor.
    fn deep_learning_global_average_pool_1d_adjoint_capability(
        &self,
        _anchor: &GpuTensorHandle,
        _input_shape: &[usize],
        _upstream: Option<&GpuTensorHandle>,
        _request: DeepLearningPool1DRequest,
    ) -> anyhow::Result<DeepLearningPool1DCapability> {
        Ok(DeepLearningPool1DCapability::UnsupportedOperation)
    }
    /// Pure whole-route admission. An error denotes an invalid owner or
    /// malformed descriptor; unsupported valid work is a typed capability.
    fn deep_learning_global_average_pool_1d_capability(
        &self,
        _input: &GpuTensorHandle,
        _route: &DeepLearningPool1DRoute,
    ) -> anyhow::Result<DeepLearningPool1DCapability> {
        Ok(DeepLearningPool1DCapability::UnsupportedOperation)
    }
    /// Pure whole-route admission for a later pooling stage. `anchor`
    /// establishes the live provider and physical numeric class; the future
    /// input shape comes from the runtime's admitted per-layer axes.
    fn deep_learning_global_average_pool_1d_planned_capability(
        &self,
        _anchor: &GpuTensorHandle,
        _input_shape: &[usize],
        _route: &DeepLearningPool1DRoute,
    ) -> anyhow::Result<DeepLearningPool1DCapability> {
        Ok(DeepLearningPool1DCapability::UnsupportedOperation)
    }
    /// Stable, channel-affine layer normalization on resident real data.
    fn deep_learning_layer_norm<'a>(
        &'a self,
        _input: &'a GpuTensorHandle,
        _scale: &'a GpuTensorHandle,
        _offset: &'a GpuTensorHandle,
        _request: &'a DeepLearningLayerNormRequest,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("deep-learning layer normalization not supported by provider")
    }
    /// One forward execution whose exact scaled statistics remain available
    /// to reverse-mode differentiation. The ordinary forward hook need not
    /// retain this buffer.
    fn deep_learning_layer_norm_recorded<'a>(
        &'a self,
        _input: &'a GpuTensorHandle,
        _scale: &'a GpuTensorHandle,
        _offset: &'a GpuTensorHandle,
        _request: &'a DeepLearningLayerNormRequest,
    ) -> AccelProviderFuture<'a, DeepLearningLayerNormRecorded> {
        unsupported_future("recorded deep-learning layer normalization not supported by provider")
    }
    fn deep_learning_layer_norm_adjoint<'a>(
        &'a self,
        _input: &'a GpuTensorHandle,
        _scale: &'a GpuTensorHandle,
        _offset: &'a GpuTensorHandle,
        _statistics: &'a GpuTensorHandle,
        _upstream: &'a GpuTensorHandle,
        _request: &'a DeepLearningLayerNormRequest,
    ) -> AccelProviderFuture<'a, DeepLearningLayerNormGradients> {
        unsupported_future("deep-learning layer normalization adjoint not supported by provider")
    }
    /// Pure admission for the recorded forward/reverse pair. A later-stage
    /// query uses the real provider anchor plus admitted future geometry; a
    /// present upstream/statistics descriptor is authenticated as well.
    fn deep_learning_layer_norm_adjoint_capability(
        &self,
        _anchor: &GpuTensorHandle,
        _scale: Option<&GpuTensorHandle>,
        _offset: Option<&GpuTensorHandle>,
        _statistics: Option<&GpuTensorHandle>,
        _upstream: Option<&GpuTensorHandle>,
        _request: &DeepLearningLayerNormRequest,
    ) -> anyhow::Result<DeepLearningLayerNormCapability> {
        Ok(DeepLearningLayerNormCapability::UnsupportedOperation)
    }
    /// Pure admission for the entire normalization kernel and physical
    /// parameter geometry. No uploads, allocations, or dispatches occur.
    fn deep_learning_layer_norm_capability(
        &self,
        _input: &GpuTensorHandle,
        _scale: Option<&GpuTensorHandle>,
        _offset: Option<&GpuTensorHandle>,
        _request: &DeepLearningLayerNormRequest,
    ) -> anyhow::Result<DeepLearningLayerNormCapability> {
        Ok(DeepLearningLayerNormCapability::UnsupportedOperation)
    }
    /// Pure admission for a later normalization stage whose input does not
    /// exist yet. `anchor` establishes the device owner and physical element
    /// type; `request.shape` is the host-admitted future input geometry. A
    /// provider must validate the whole request without dispatch or upload.
    fn deep_learning_layer_norm_planned_capability(
        &self,
        _anchor: &GpuTensorHandle,
        _scale: Option<&GpuTensorHandle>,
        _offset: Option<&GpuTensorHandle>,
        _request: &DeepLearningLayerNormRequest,
    ) -> anyhow::Result<DeepLearningLayerNormCapability> {
        Ok(DeepLearningLayerNormCapability::UnsupportedOperation)
    }
    /// Learned multichannel 1-D cross-correlation, distinct from the signal
    /// `conv1d` operation and its reversal/alignment contract.
    fn deep_learning_conv1d<'a>(
        &'a self,
        _input: &'a GpuTensorHandle,
        _weights: &'a GpuTensorHandle,
        _bias: &'a GpuTensorHandle,
        _request: &'a DeepLearningConv1DRequest,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("deep-learning conv1d not supported by provider")
    }
    /// Reverse the exact learned cross-correlation using the forward input
    /// and prepared parameter leases, without gathering either operand.
    fn deep_learning_conv1d_adjoint<'a>(
        &'a self,
        _input: &'a GpuTensorHandle,
        _weights: &'a GpuTensorHandle,
        _bias: &'a GpuTensorHandle,
        _upstream: &'a GpuTensorHandle,
        _request: &'a DeepLearningConv1DRequest,
    ) -> AccelProviderFuture<'a, DeepLearningConv1DGradients> {
        unsupported_future("deep-learning conv1d adjoint not supported by provider")
    }
    /// A pure complete forward/adjoint route query. `anchor` proves ownership
    /// even when the future input has not been produced; a present `input`
    /// must match the physical request. Absent parameters or seed represent
    /// host values or not-yet-produced outputs, never permission to gather
    /// explicit residency.
    fn deep_learning_conv1d_capability(
        &self,
        _operands: DeepLearningConv1DOperands<'_>,
        _request: &DeepLearningConv1DRequest,
    ) -> anyhow::Result<DeepLearningConv1DCapability> {
        Ok(DeepLearningConv1DCapability::UnsupportedOperation)
    }
    /// Pure admission for a physical feed-forward stage in a planned
    /// sequence. `anchor` proves device ownership and numeric class. The
    /// proposed stage geometry can differ from the anchor's current shape.
    fn deep_learning_feed_forward_stage_capability(
        &self,
        _anchor: &GpuTensorHandle,
        _stage: DeepLearningFeedForwardStage,
    ) -> anyhow::Result<DeepLearningFeedForwardCapability> {
        Ok(DeepLearningFeedForwardCapability::UnsupportedOperation)
    }
    /// Effect-free admission of an already-lowered complete forward route.
    /// Unlike stage queries, this can inspect a host-origin candidate before
    /// upload. Resident sources remain authenticated by their owning provider.
    /// `None` means this optional whole-route query is absent; `Some` reports
    /// the provider's actual acceptance or rejection. Callers must never treat
    /// a present rejection as permission to execute a partial route.
    fn deep_learning_forward_route_capability(
        &self,
        _route: &DeepLearningForwardRoute<'_>,
    ) -> anyhow::Result<Option<DeepLearningForwardCapability>> {
        Ok(None)
    }
    /// Read-only preparation state for an admitted whole forward route.
    /// An absent snapshot means the provider cannot certify every pipeline
    /// that route may use; it must not be interpreted as a warm route.
    fn deep_learning_forward_preparation_snapshot(
        &self,
        _route: &DeepLearningForwardRoute<'_>,
    ) -> Option<ProviderPreparationSnapshot> {
        None
    }
    /// Pure admission of an exact physical view chain. The live anchor proves
    /// owner and element type; a planned source is geometry, not a lease.
    fn provider_view_chain_capability(
        &self,
        _anchor: &GpuTensorHandle,
        _chain: &ProviderViewChain,
    ) -> anyhow::Result<ProviderViewCapability> {
        Ok(ProviderViewCapability::UnsupportedOperation)
    }
    /// Reduce moments across multiple zero-based dimensions in one device pass.
    /// Returns mean (E[x]) and mean of squares (E[x^2]).
    fn reduce_moments_nd<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dims_zero_based: &'a [usize],
    ) -> AccelProviderFuture<'a, ProviderMoments2> {
        unsupported_future("reduce_moments_nd not supported by provider")
    }
    /// Optional dimensional mean with the same no-attempt contract as global mean.
    fn reduce_mean_dim<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dim: usize,
    ) -> ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async { Ok(None) })
    }
    fn reduce_std<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _normalization: ProviderStdNormalization,
        _nan_mode: ProviderNanMode,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        unsupported_execution_future("reduce_std not supported by provider")
    }
    fn reduce_std_dim<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dim: usize,
        _normalization: ProviderStdNormalization,
        _nan_mode: ProviderNanMode,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        unsupported_execution_future("reduce_std_dim not supported by provider")
    }
    fn reduce_any<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("reduce_any not supported by provider")
    }
    fn reduce_any_dim<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dim: usize,
        _omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("reduce_any_dim not supported by provider")
    }
    fn reduce_all<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("reduce_all not supported by provider")
    }
    fn reduce_all_dim<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dim: usize,
        _omit_nan: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("reduce_all_dim not supported by provider")
    }
    fn reduce_median<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("reduce_median not supported by provider")
    }
    fn reduce_median_dim<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("reduce_median_dim not supported by provider")
    }
    fn mode_values<'a>(
        &'a self,
        _request: &'a ProviderModeRequest<'a>,
    ) -> AccelProviderFuture<'a, ProviderModeResult> {
        unsupported_future("mode_values not supported by provider")
    }
    fn moving_window<'a>(
        &'a self,
        _request: &'a ProviderMovingWindowRequest<'a>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("moving_window not supported by provider")
    }
    fn reduce_min<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        unsupported_execution_future("reduce_min not supported by provider")
    }
    /// Reduce a zero-based dimension with fixed Include-NaN semantics.
    /// Return the first NaN (if any), otherwise the first minimum, with 1-based
    /// indices. Negative zero orders below positive zero; identical signed
    /// zeros retain the first index. Preserve numeric class and empty input shape.
    /// Unsupported representations must return typed Unsupported; omit-NaN
    /// requests use a separate domain path, not this hook.
    fn reduce_min_dim<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dim: usize,
    ) -> ProviderExecutionFuture<'a, ReduceDimResult> {
        unsupported_execution_future("reduce_min_dim not supported by provider")
    }
    fn reduce_max<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        unsupported_execution_future("reduce_max not supported by provider")
    }
    /// As `reduce_min_dim`, selecting the first maximum instead of minimum.
    /// NaN policy remains Include; positive zero orders above negative zero,
    /// and equal extrema (including identical signed zeros) retain the first index.
    fn reduce_max_dim<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dim: usize,
    ) -> ProviderExecutionFuture<'a, ReduceDimResult> {
        unsupported_execution_future("reduce_max_dim not supported by provider")
    }
    /// Returns `None` only before staging, allocation, submission, or execution.
    /// An error is terminal and never authorizes a host replay of this scan.
    fn cumsum_scan<'a>(
        &'a self,
        _input: &'a GpuTensorHandle,
        _dim: usize,
        _direction: ProviderScanDirection,
        _nan_mode: ProviderNanMode,
    ) -> ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async { Ok(None) })
    }
    /// Returns `None` only before staging, allocation, submission, or execution.
    /// An error is terminal and never authorizes a host replay of this scan.
    fn integer_cumsum_scan<'a>(
        &'a self,
        _input: &'a GpuTensorHandle,
        _dim: usize,
        _direction: ProviderScanDirection,
    ) -> ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async { Ok(None) })
    }
    /// `None` certifies no staging, allocation, submission, or execution occurred.
    /// Errors are terminal and never authorize replay of this operation.
    fn trapz_dim(
        &self,
        _input: &GpuTensorHandle,
        _dim: usize,
        _spacing: ProviderTrapezoidSpacing<'_>,
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        Ok(None)
    }
    /// `None` certifies no staging, allocation, submission, or execution occurred.
    /// Errors are terminal and never authorize replay of this operation.
    fn cumtrapz_dim(
        &self,
        _input: &GpuTensorHandle,
        _dim: usize,
        _spacing: ProviderTrapezoidSpacing<'_>,
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        Ok(None)
    }
    /// Returns `None` only before staging, allocation, submission, or execution.
    /// An error is terminal and never authorizes a host replay of this scan.
    fn cumprod_scan<'a>(
        &'a self,
        _input: &'a GpuTensorHandle,
        _dim: usize,
        _direction: ProviderScanDirection,
        _nan_mode: ProviderNanMode,
    ) -> ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async { Ok(None) })
    }
    /// Returns `None` only before staging, allocation, submission, or execution.
    /// An error is terminal and never authorizes a host replay of this scan.
    fn integer_cumprod_scan<'a>(
        &'a self,
        _input: &'a GpuTensorHandle,
        _dim: usize,
        _direction: ProviderScanDirection,
    ) -> ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async { Ok(None) })
    }
    /// Returns `None` only before staging, allocation, submission, or execution.
    /// An error is terminal and never authorizes a host replay of this scan.
    fn cummin_scan<'a>(
        &'a self,
        _input: &'a GpuTensorHandle,
        _dim: usize,
        _direction: ProviderScanDirection,
        _nan_mode: ProviderNanMode,
        _demand: ProviderExtremaDemand,
    ) -> ProviderExecutionFuture<'a, Option<ProviderCumminResult>> {
        Box::pin(async { Ok(None) })
    }
    /// Returns `None` only before staging, allocation, submission, or execution.
    /// An error is terminal and never authorizes a host replay of this scan.
    fn integer_cummin_scan<'a>(
        &'a self,
        _input: &'a GpuTensorHandle,
        _dim: usize,
        _direction: ProviderScanDirection,
        _demand: ProviderExtremaDemand,
    ) -> ProviderExecutionFuture<'a, Option<ProviderCumminResult>> {
        Box::pin(async { Ok(None) })
    }
    /// Returns `None` only before staging, allocation, submission, or execution.
    /// An error is terminal and never authorizes a host replay of this scan.
    fn cummax_scan<'a>(
        &'a self,
        _input: &'a GpuTensorHandle,
        _dim: usize,
        _direction: ProviderScanDirection,
        _nan_mode: ProviderNanMode,
        _demand: ProviderExtremaDemand,
    ) -> ProviderExecutionFuture<'a, Option<ProviderCummaxResult>> {
        Box::pin(async { Ok(None) })
    }
    /// Returns `None` only before staging, allocation, submission, or execution.
    /// An error is terminal and never authorizes a host replay of this scan.
    fn integer_cummax_scan<'a>(
        &'a self,
        _input: &'a GpuTensorHandle,
        _dim: usize,
        _direction: ProviderScanDirection,
        _demand: ProviderExtremaDemand,
    ) -> ProviderExecutionFuture<'a, Option<ProviderCummaxResult>> {
        Box::pin(async { Ok(None) })
    }

    /// Certifies that this provider admits the exact find request before any
    /// execution effect. A false result permits the runtime's owner-aware host
    /// path; an error after true is never replay authority.
    fn supports_find(
        &self,
        _a: &GpuTensorHandle,
        _limit: Option<usize>,
        _direction: FindDirection,
    ) -> bool {
        false
    }

    fn find(
        &self,
        _a: &GpuTensorHandle,
        _limit: Option<usize>,
        _direction: FindDirection,
    ) -> anyhow::Result<ProviderFindResult> {
        Err(anyhow::anyhow!("find not supported by provider"))
    }

    fn fused_elementwise<'a>(
        &'a self,
        _shader: &'a str,
        requirements: ProviderShaderRequirements,
        _inputs: &'a [GpuTensorHandle],
        _output_shape: &'a [usize],
        _len: usize,
        output_class: ProviderElementType,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if !self.supports_shader_requirements(requirements) {
                return Err(unsupported_provider_operation(
                    "selected fused elementwise shader requires unavailable features",
                )
                .into());
            }
            validate_fused_output_classes(self.precision(), &[output_class])?;
            Err(
                unsupported_provider_operation("fused_elementwise not supported by provider")
                    .into(),
            )
        })
    }

    /// Optional typed data ABI. Preferred provider precision is not proof that
    /// a provider can bind another physical floating class.
    fn supports_fused_elementwise_abi(&self, _abi: ProviderFusedElementwiseAbi) -> bool {
        false
    }

    /// Lazy typed-unsupported by default. Source semantics remain caller-owned;
    /// implementations validate actual inputs and freeze their physical recipe.
    fn fused_elementwise_typed<'a>(
        &'a self,
        _request: ProviderFusedElementwiseRequest<'a>,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async {
            Err(unsupported_provider_operation(
                "typed fused elementwise ABI is not supported by provider",
            )
            .into())
        })
    }
    fn fused_elementwise_multi_typed<'a>(
        &'a self,
        _request: ProviderFusedElementwiseRequest<'a>,
    ) -> ProviderExecutionFuture<'a, Vec<GpuTensorHandle>> {
        Box::pin(async {
            Err(unsupported_provider_operation(
                "typed fused multi-elementwise ABI is not supported by provider",
            )
            .into())
        })
    }

    /// Execute one fused kernel writing `output_classes.len()` output buffers.
    /// Output bindings follow the inputs, followed by one uniform `params`.
    /// Classes describe this physical hook, not a later output adaptation.
    /// Logical outputs use native floating encoding and producer classification.
    /// The default is typed Unsupported, never guessed shader rewriting.
    fn fused_elementwise_multi<'a>(
        &'a self,
        _shader: &'a str,
        requirements: ProviderShaderRequirements,
        _inputs: &'a [GpuTensorHandle],
        _output_shape: &'a [usize],
        _len: usize,
        output_classes: &'a [ProviderElementType],
    ) -> ProviderExecutionFuture<'a, Vec<GpuTensorHandle>> {
        Box::pin(async move {
            if !self.supports_shader_requirements(requirements) {
                return Err(unsupported_provider_operation(
                    "selected fused elementwise shader requires unavailable features",
                )
                .into());
            }
            validate_fused_output_classes(self.precision(), output_classes)?;
            Err(
                unsupported_provider_operation("fused_elementwise_multi not supported by provider")
                    .into(),
            )
        })
    }

    /// Build a numeric tensor where NaNs in `a` are replaced with 0.0 (device side).
    fn map_nan_to_zero(&self, _a: &GpuTensorHandle) -> ProviderExecutionResult<GpuTensorHandle> {
        Err(unsupported_provider_operation("map_nan_to_zero not supported by provider").into())
    }

    /// Build a numeric mask tensor with 1.0 where value is not NaN and 0.0 where value is NaN.
    fn not_nan_mask(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation(
            "not_nan_mask not supported by provider",
        ))
    }

    /// Generic fused reduction entrypoint.
    ///
    /// The shader is expected to implement a column-major reduction across `reduce_len` with
    /// `num_slices` independent slices (e.g., columns). Providers should create a uniform buffer
    /// compatible with the expected `Params/MParams` struct in the shader and dispatch
    /// `num_slices` workgroups with `workgroup_size` threads, or an equivalent strategy.
    #[allow(clippy::too_many_arguments)]
    fn fused_reduction(
        &self,
        _shader: &str,
        _requirements: ProviderShaderRequirements,
        _inputs: &[GpuTensorHandle],
        _output_shape: &[usize],
        _reduce_len: usize,
        _num_slices: usize,
        _workgroup_size: u32,
        _flavor: ReductionFlavor,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("fused_reduction not supported by provider"))
    }

    /// Enabled physical dimensions for one checked recipe, before effects.
    fn reduction_recipe_config(
        &self,
        _reduce_elements: u32,
        _slices: u32,
    ) -> ProviderExecutionResult<reduction::ReductionRecipeConfig> {
        Err(ProviderExecutionError::from(
            unsupported_provider_operation("typed reduction configuration unsupported"),
        ))
    }

    /// Pre-effect physical candidates for this exact checked recipe. This
    /// query never launches autotune trials or authenticates future leases.
    /// The caller chooses one through the canonical placement authority and
    /// passes that same selection to submit_reduction_recipe.
    fn reduction_recipe_candidates(
        &self,
        _recipe: &reduction::ProviderReductionRecipe,
        _host_metadata: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> ProviderExecutionResult<[Option<reduction::ReductionStrategy>; 2]> {
        Err(ProviderExecutionError::from(
            unsupported_provider_operation("typed reduction selection unsupported"),
        ))
    }

    /// Execute one immutable selected physical reduction recipe.
    fn submit_reduction_recipe(
        &self,
        _submission: &reduction::ReductionSubmission<'_>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        Err(ProviderExecutionError::from(
            unsupported_provider_operation("typed reduction recipe unsupported"),
        ))
    }

    /// Admission probe for a selected generated shader, before host uploads.
    /// The default cannot claim a non-basic device feature.
    fn supports_shader_requirements(&self, requirements: ProviderShaderRequirements) -> bool {
        requirements == ProviderShaderRequirements::Basic
    }

    /// Optionally pre-compile commonly used pipelines to amortize first-dispatch costs.
    fn warmup(&self) {}

    /// Returns (cache_hits, cache_misses) for fused pipeline cache, if supported.
    fn fused_cache_counters(&self) -> (u64, u64) {
        (0, 0)
    }

    /// Returns the duration of the last provider warmup in milliseconds, if known.
    fn last_warmup_millis(&self) -> Option<u64> {
        None
    }

    /// Returns a snapshot of provider telemetry counters if supported.
    fn telemetry_snapshot(&self) -> ProviderTelemetry {
        let (hits, misses) = self.fused_cache_counters();
        ProviderTelemetry {
            fused_elementwise: ProviderDispatchStats::default(),
            fused_reduction: ProviderDispatchStats::default(),
            matmul: ProviderDispatchStats::default(),
            linsolve: ProviderDispatchStats::default(),
            mldivide: ProviderDispatchStats::default(),
            mrdivide: ProviderDispatchStats::default(),
            upload_bytes: 0,
            download_bytes: 0,
            solve_fallbacks: Vec::new(),
            fusion_cache_hits: hits,
            fusion_cache_misses: misses,
            bind_group_cache_hits: 0,
            bind_group_cache_misses: 0,
            bind_group_cache_by_layout: None,
            kernel_launches: Vec::new(),
        }
    }

    /// Reset all telemetry counters maintained by the provider, if supported.
    fn reset_telemetry(&self) {}

    /// Default reduction workgroup size the provider prefers.
    fn default_reduction_workgroup_size(&self) -> u32 {
        256
    }

    /// Threshold above which provider will prefer two-pass reduction.
    fn two_pass_threshold(&self) -> usize {
        1024
    }

    /// Current two-pass mode preference (auto/forced on/off).
    fn reduction_two_pass_mode(&self) -> ReductionTwoPassMode {
        ReductionTwoPassMode::Auto
    }

    /// Fast-path: write a GPU column in a matrix from a GPU vector, returning a new handle.
    /// Expected: `values.shape == [rows, 1]` (or `[rows]`) and `col_index < cols`.
    fn scatter_column<'a>(
        &'a self,
        _matrix: &'a GpuTensorHandle,
        _col_index: usize,
        _values: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            Err(unsupported_provider_operation("scatter_column not supported by provider").into())
        })
    }

    /// Fast-path: write a GPU row in a matrix from a GPU vector, returning a new handle.
    /// Expected: `values.shape == [1, cols]` (or `[cols]`) and `row_index < rows`.
    fn scatter_row<'a>(
        &'a self,
        _matrix: &'a GpuTensorHandle,
        _row_index: usize,
        _values: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            Err(unsupported_provider_operation("scatter_row not supported by provider").into())
        })
    }

    fn sub2ind<'a>(
        &'a self,
        _dims: &'a [usize],
        _strides: &'a [usize],
        _inputs: &'a [&'a GpuTensorHandle],
        _scalar_mask: &'a [bool],
        _len: usize,
        _output_shape: &'a [usize],
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            Err(unsupported_provider_operation("sub2ind not supported by provider").into())
        })
    }

    /// Returns true if the provider offers a device-side `ind2sub` implementation.
    fn supports_ind2sub(&self) -> bool {
        false
    }

    /// Convert linear indices into per-dimension subscripts on the device.
    fn ind2sub<'a>(
        &'a self,
        _dims: &'a [usize],
        _strides: &'a [usize],
        _indices: &'a GpuTensorHandle,
        _total: usize,
        _len: usize,
        _output_shape: &'a [usize],
    ) -> ProviderExecutionFuture<'a, Vec<GpuTensorHandle>> {
        Box::pin(async move { Err(anyhow::anyhow!("ind2sub not supported by provider").into()) })
    }

    /// Determine if a matrix is symmetric (or skew-symmetric) without gathering it to the host.
    fn issymmetric(
        &self,
        _matrix: &GpuTensorHandle,
        _kind: ProviderSymmetryKind,
        _tolerance: f64,
    ) -> anyhow::Result<bool> {
        Err(unsupported_provider_operation(
            "issymmetric predicate not supported by provider",
        ))
    }

    /// Determine if a matrix is Hermitian (or skew-Hermitian) without gathering it to the host.
    fn ishermitian<'a>(
        &'a self,
        _matrix: &'a GpuTensorHandle,
        _kind: ProviderHermitianKind,
        _tolerance: f64,
    ) -> AccelProviderFuture<'a, bool> {
        Box::pin(async move {
            Err(unsupported_provider_operation(
                "ishermitian predicate not supported by provider",
            ))
        })
    }

    /// Inspect the bandwidth of a matrix without gathering it back to the host.
    fn bandwidth(&self, _matrix: &GpuTensorHandle) -> anyhow::Result<ProviderBandwidth> {
        Err(unsupported_provider_operation(
            "bandwidth not supported by provider",
        ))
    }

    /// Compute the symmetric reverse Cuthill-McKee permutation for the matrix.
    ///
    /// Implementations may execute on the device or gather to the host. The permutation should be
    /// returned as zero-based indices.
    fn sym_rcm<'a>(&'a self, _matrix: &'a GpuTensorHandle) -> AccelProviderFuture<'a, Vec<usize>> {
        Box::pin(async move {
            Err(unsupported_provider_operation(
                "sym_rcm not supported by provider",
            ))
        })
    }
}

static DEVICE_ID_COUNTER: AtomicU32 = AtomicU32::new(1);

thread_local! {
    static THREAD_PROVIDER: Cell<Option<&'static dyn AccelProvider>> = Cell::new(None);
}
fn replace_thread_provider(
    provider: Option<&'static dyn AccelProvider>,
) -> Option<&'static dyn AccelProvider> {
    THREAD_PROVIDER.with(|cell| cell.replace(provider))
}
fn current_thread_provider() -> Option<&'static dyn AccelProvider> {
    THREAD_PROVIDER.with(Cell::get)
}

/// Register a global acceleration provider.
///
/// # Safety
/// - The caller must guarantee that `p` is valid for the entire program lifetime
///   (e.g., a `'static` singleton), as the runtime stores a raw reference globally.
/// - Concurrent callers must ensure registration happens once or is properly
///   synchronized; this function does not enforce thread-safety for re-registration.
/// - Distinct live provider instances that can own handles must not share a
///   `device_id`; the handle registry uses that identifier as its ownership key.
pub unsafe fn register_provider(p: &'static dyn AccelProvider) {
    replace_thread_provider(Some(p));
    provider_registry::set_default(Some(p));
    register_provider_for_device(p.device_id(), p);
}

unsafe fn register_provider_for_device(device_id: u32, provider: &'static dyn AccelProvider) {
    provider_registry::register(device_id, provider);
}

/// Register a provider as an addressable device owner without selecting it as
/// the ambient placement provider.
///
/// This supports specialized native-device providers that coexist with WGPU
/// or another general compute provider. Handles produced by the provider route
/// through its distinct device id.
///
/// # Safety
///
/// The provider must remain valid for the process lifetime, and its device id
/// must not be reused by a different live provider.
pub unsafe fn register_device_provider(provider: &'static dyn AccelProvider) {
    // SAFETY: the caller supplies the same lifetime and identity guarantees as
    // `register_provider`; this variant changes only ambient selection.
    unsafe { register_provider_for_device(provider.device_id(), provider) };
}

fn primary_provider_without_registry_scan() -> Option<&'static dyn AccelProvider> {
    current_thread_provider()
        .filter(|provider| provider_is_authorized(*provider))
        .or_else(|| {
            provider_registry::default().filter(|provider| provider_is_authorized(*provider))
        })
}

/// Select the same current/default/registered owner as `provider` without
/// constructing the registry's fallback candidate vector before a finite
/// request's host-preparation receipt. Authorization runs outside the registry
/// lock, and the fallback retains device-id order.
pub fn provider_without_registry_allocation() -> Option<&'static dyn AccelProvider> {
    let selected = primary_provider_without_registry_scan().or_else(|| {
        let mut prior_device = None;
        while let Some((device, provider)) = provider_registry::next_after(prior_device) {
            if matches!(
                provider_authorization(provider),
                ExecutionProviderAuthorization::Allowed
            ) {
                return Some(provider);
            }
            prior_device = Some(device);
        }
        None
    });
    audit_vm_region_provider_selection(selected)
}

pub fn provider() -> Option<&'static dyn AccelProvider> {
    let selected = if let Some(provider) = primary_provider_without_registry_scan() {
        Some(provider)
    } else {
        // Copy references before callbacks; no registry lock or RefCell borrow
        // crosses provider code.
        registered_providers().into_iter().find(|provider| {
            matches!(
                provider_authorization(*provider),
                ExecutionProviderAuthorization::Allowed
            )
        })
    };
    audit_vm_region_provider_selection(selected)
}

fn audit_vm_region_provider_selection(
    selected: Option<&'static dyn AccelProvider>,
) -> Option<&'static dyn AccelProvider> {
    #[cfg(feature = "wgpu")]
    if let Some(provider) = selected {
        observation::record_source_region_provider_selection(provider);
    }
    selected
}

/// Clear the active provider selection without invalidating providers that own live handles.
///
/// Registered device owners are retained because a [`GpuTensorHandle`] may outlive its provider's
/// tenure as the global default. Removing its owner here would make that otherwise-valid handle
/// impossible to gather or operate on. Tests that need a different default can safely call this
/// function and register another provider; device ids keep the handle namespaces distinct.
pub fn clear_provider() {
    replace_thread_provider(None);
    provider_registry::set_default(None);
}

pub fn provider_for_device(device_id: u32) -> Option<&'static dyn AccelProvider> {
    let selected = if let Some(registered) = provider_registry::get(device_id) {
        provider_is_authorized(registered).then_some(registered)
    } else if let Some(provider) =
        current_thread_provider().filter(|provider| provider.device_id() == device_id)
    {
        provider_is_authorized(provider).then_some(provider)
    } else {
        // Retained execution-selection compatibility only. Physical retirement
        // never consults this registry or its ambient fallback.
        provider_registry::default().filter(|provider| {
            matches!(
                provider_authorization(*provider),
                ExecutionProviderAuthorization::Unrestricted
            )
        })
    };
    audit_vm_region_provider_selection(selected)
}

/// Returns every addressable provider in deterministic process-local order.
/// Provider registration guarantees that these references remain valid for
/// the process lifetime.
pub fn registered_providers() -> Vec<&'static dyn AccelProvider> {
    provider_registry::all()
}

/// Inspect retained owners without a candidate-vector allocation. The registry
/// lock is released before a caller invokes provider code. This does not grant
/// execution/control authority or change ambient selection.
pub fn next_registered_provider(after: Option<u32>) -> Option<(u32, &'static dyn AccelProvider)> {
    provider_registry::next_after(after)
}

pub fn provider_for_handle(handle: &GpuTensorHandle) -> Option<&'static dyn AccelProvider> {
    provider_for_device(handle.device_id())
}

/// Select a registered provider that explicitly exposes `api`.
///
/// The thread-selected provider wins when compatible. Otherwise selection is
/// deterministic by provider device id, allowing a native extension backend
/// to coexist with a different ambient placement provider.
pub fn provider_for_native_device(api: NativeDeviceApi) -> Option<&'static dyn AccelProvider> {
    let selected = if let Some(selected) = current_thread_provider().filter(|provider| {
        provider.native_device_api() == Some(api) && provider_is_authorized(*provider)
    }) {
        Some(selected)
    } else {
        registered_providers().into_iter().find(|provider| {
            provider.native_device_api() == Some(api) && provider_is_authorized(*provider)
        })
    };
    audit_vm_region_provider_selection(selected)
}

/// Returns whether a process-local provider advertises the exact device
/// contract named by a scheduler lease.
pub fn provider_matches_execution_lease(
    provider: &(impl AccelProvider + ?Sized),
    lease: &runmat_execution::resource::AcceleratorDeviceLease,
) -> bool {
    provider
        .execution_accelerator_device(lease.inventory_epoch)
        .ok()
        .flatten()
        .is_some_and(|device| lease.matches_device(&device))
}

fn provider_authorization(provider: &dyn AccelProvider) -> ExecutionProviderAuthorization {
    EXECUTION_PROVIDER_AUTHORIZER
        .get()
        .map(|authorizer| authorizer(ExecutionProviderAdmission::Provider(provider)))
        .unwrap_or(ExecutionProviderAuthorization::Unrestricted)
}

fn provider_is_authorized(provider: &dyn AccelProvider) -> bool {
    !matches!(
        provider_authorization(provider),
        ExecutionProviderAuthorization::Denied
    )
}

pub fn spawn_handle_concurrency_for(handle: &GpuTensorHandle) -> Option<SpawnHandleConcurrency> {
    provider_for_handle(handle).map(AccelProvider::spawn_handle_concurrency)
}

pub fn next_device_id() -> u32 {
    DEVICE_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

pub struct ThreadProviderGuard {
    prev: Option<&'static dyn AccelProvider>,
}

impl ThreadProviderGuard {
    pub fn set(provider: Option<&'static dyn AccelProvider>) -> Self {
        let prev = replace_thread_provider(provider);
        ThreadProviderGuard { prev }
    }
}

impl Drop for ThreadProviderGuard {
    fn drop(&mut self) {
        let prev = self.prev.take();
        replace_thread_provider(prev);
    }
}

pub fn set_thread_provider(provider: Option<&'static dyn AccelProvider>) {
    replace_thread_provider(provider);
}

/// Convenience: perform elementwise add via provider if possible; otherwise return None
pub async fn try_elem_add(a: &GpuTensorHandle, b: &GpuTensorHandle) -> Option<GpuTensorHandle> {
    if a.device_id() == b.device_id() {
        let p = provider_for_handle(a)?;
        if let Ok(h) = p.elem_add(a, b).await {
            return Some(h);
        }
    }
    None
}

/// Convenience: perform elementwise hypot via provider if possible; otherwise return None
pub async fn try_elem_hypot(a: &GpuTensorHandle, b: &GpuTensorHandle) -> Option<GpuTensorHandle> {
    if a.device_id() == b.device_id() {
        let p = provider_for_handle(a)?;
        if let Ok(h) = p.elem_hypot(a, b).await {
            return Some(h);
        }
    }
    None
}

/// Convenience: perform elementwise max via provider if possible; otherwise return None
pub async fn try_elem_max(a: &GpuTensorHandle, b: &GpuTensorHandle) -> Option<GpuTensorHandle> {
    if a.device_id() == b.device_id() {
        let p = provider_for_handle(a)?;
        if let Ok(h) = p.elem_max(a, b).await {
            return Some(h);
        }
    }
    None
}

/// Convenience: perform elementwise min via provider if possible; otherwise return None
pub async fn try_elem_min(a: &GpuTensorHandle, b: &GpuTensorHandle) -> Option<GpuTensorHandle> {
    if a.device_id() == b.device_id() {
        let p = provider_for_handle(a)?;
        if let Ok(h) = p.elem_min(a, b).await {
            return Some(h);
        }
    }
    None
}

/// Convenience: perform elementwise atan2 via provider if possible; otherwise return None
pub async fn try_elem_atan2(y: &GpuTensorHandle, x: &GpuTensorHandle) -> Option<GpuTensorHandle> {
    if y.device_id() == x.device_id() {
        let p = provider_for_handle(y)?;
        if let Ok(h) = p.elem_atan2(y, x).await {
            return Some(h);
        }
    }
    None
}

/// Physical element type carried by a numeric provider transfer.
///
/// This type is exhaustive over RunMat's real numeric classes and lives in the
/// acceleration API so providers do not depend on runtime or builtin storage
/// types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NumericElementType {
    F64,
    F32,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl NumericElementType {
    pub const fn element_size(self) -> usize {
        match self {
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::I16 | Self::U16 => 2,
            Self::I8 | Self::U8 => 1,
        }
    }

    pub const fn class_name(self) -> &'static str {
        match self {
            Self::F64 => "double",
            Self::F32 => "single",
            Self::I8 => "int8",
            Self::I16 => "int16",
            Self::I32 => "int32",
            Self::I64 => "int64",
            Self::U8 => "uint8",
            Self::U16 => "uint16",
            Self::U32 => "uint32",
            Self::U64 => "uint64",
        }
    }

    pub fn class_identity(self) -> ClassIdentity {
        match self {
            Self::F64 => standard::DOUBLE.owned(),
            Self::F32 => standard::SINGLE.owned(),
            Self::I8 => standard::INT8.owned(),
            Self::I16 => standard::INT16.owned(),
            Self::I32 => standard::INT32.owned(),
            Self::I64 => standard::INT64.owned(),
            Self::U8 => standard::UINT8.owned(),
            Self::U16 => standard::UINT16.owned(),
            Self::U32 => standard::UINT32.owned(),
            Self::U64 => standard::UINT64.owned(),
        }
    }

    pub const fn precision(self) -> Option<ProviderPrecision> {
        match self {
            Self::F64 => Some(ProviderPrecision::F64),
            Self::F32 => Some(ProviderPrecision::F32),
            Self::I8
            | Self::I16
            | Self::I32
            | Self::I64
            | Self::U8
            | Self::U16
            | Self::U32
            | Self::U64 => None,
        }
    }

    pub const fn integer_type(self) -> Option<IntegerElementType> {
        match self {
            Self::F64 | Self::F32 => None,
            Self::I8 => Some(IntegerElementType::I8),
            Self::I16 => Some(IntegerElementType::I16),
            Self::I32 => Some(IntegerElementType::I32),
            Self::I64 => Some(IntegerElementType::I64),
            Self::U8 => Some(IntegerElementType::U8),
            Self::U16 => Some(IntegerElementType::U16),
            Self::U32 => Some(IntegerElementType::U32),
            Self::U64 => Some(IntegerElementType::U64),
        }
    }
}

impl From<IntegerElementType> for NumericElementType {
    fn from(value: IntegerElementType) -> Self {
        match value {
            IntegerElementType::I8 => Self::I8,
            IntegerElementType::I16 => Self::I16,
            IntegerElementType::I32 => Self::I32,
            IntegerElementType::I64 => Self::I64,
            IntegerElementType::U8 => Self::U8,
            IntegerElementType::U16 => Self::U16,
            IntegerElementType::U32 => Self::U32,
            IntegerElementType::U64 => Self::U64,
        }
    }
}

/// Borrowed native numeric buffer used for provider transfers.
#[derive(Debug, Clone, Copy)]
pub enum HostNumericDataView<'a> {
    F64(&'a [f64]),
    F32(&'a [f32]),
    I8(&'a [i8]),
    I16(&'a [i16]),
    I32(&'a [i32]),
    I64(&'a [i64]),
    U8(&'a [u8]),
    U16(&'a [u16]),
    U32(&'a [u32]),
    U64(&'a [u64]),
}

impl HostNumericDataView<'_> {
    pub const fn element_type(self) -> NumericElementType {
        match self {
            Self::F64(_) => NumericElementType::F64,
            Self::F32(_) => NumericElementType::F32,
            Self::I8(_) => NumericElementType::I8,
            Self::I16(_) => NumericElementType::I16,
            Self::I32(_) => NumericElementType::I32,
            Self::I64(_) => NumericElementType::I64,
            Self::U8(_) => NumericElementType::U8,
            Self::U16(_) => NumericElementType::U16,
            Self::U32(_) => NumericElementType::U32,
            Self::U64(_) => NumericElementType::U64,
        }
    }

    pub const fn len(self) -> usize {
        match self {
            Self::F64(data) => data.len(),
            Self::F32(data) => data.len(),
            Self::I8(data) => data.len(),
            Self::I16(data) => data.len(),
            Self::I32(data) => data.len(),
            Self::I64(data) => data.len(),
            Self::U8(data) => data.len(),
            Self::U16(data) => data.len(),
            Self::U32(data) => data.len(),
            Self::U64(data) => data.len(),
        }
    }

    pub const fn is_empty(self) -> bool {
        self.len() == 0
    }
}

/// Borrowed tensor transfer with one native numeric payload and explicit
/// real/complex layout.
#[derive(Debug, Clone, Copy)]
pub struct HostNumericTensorView<'a> {
    pub data: HostNumericDataView<'a>,
    pub shape: &'a [usize],
    pub storage: GpuTensorStorage,
}

impl HostNumericTensorView<'_> {
    pub fn validate(self) -> anyhow::Result<()> {
        let logical_len = if self.shape.contains(&0) {
            0
        } else {
            self.shape
                .iter()
                .try_fold(1usize, |len, &dim| len.checked_mul(dim))
                .ok_or_else(|| anyhow!("numeric transfer shape overflows usize"))?
        };
        let expected = match self.storage {
            GpuTensorStorage::Real => logical_len,
            GpuTensorStorage::ComplexInterleaved => logical_len
                .checked_mul(2)
                .ok_or_else(|| anyhow!("complex numeric transfer length overflows usize"))?,
        };
        if self.data.len() != expected {
            return Err(anyhow!(
                "{} transfer has {} elements for shape {:?} and {:?} storage; expected {}",
                self.data.element_type().class_name(),
                self.data.len(),
                self.shape,
                self.storage,
                expected
            ));
        }
        Ok(())
    }

    pub const fn element_type(self) -> NumericElementType {
        self.data.element_type()
    }
}

/// Owned native numeric buffer returned by a provider download.
#[derive(Debug, Clone, PartialEq)]
pub enum HostNumericDataOwned {
    F64(Vec<f64>),
    F32(Vec<f32>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

impl HostNumericDataOwned {
    pub const fn element_type(&self) -> NumericElementType {
        match self {
            Self::F64(_) => NumericElementType::F64,
            Self::F32(_) => NumericElementType::F32,
            Self::I8(_) => NumericElementType::I8,
            Self::I16(_) => NumericElementType::I16,
            Self::I32(_) => NumericElementType::I32,
            Self::I64(_) => NumericElementType::I64,
            Self::U8(_) => NumericElementType::U8,
            Self::U16(_) => NumericElementType::U16,
            Self::U32(_) => NumericElementType::U32,
            Self::U64(_) => NumericElementType::U64,
        }
    }

    pub fn as_view(&self) -> HostNumericDataView<'_> {
        match self {
            Self::F64(data) => HostNumericDataView::F64(data),
            Self::F32(data) => HostNumericDataView::F32(data),
            Self::I8(data) => HostNumericDataView::I8(data),
            Self::I16(data) => HostNumericDataView::I16(data),
            Self::I32(data) => HostNumericDataView::I32(data),
            Self::I64(data) => HostNumericDataView::I64(data),
            Self::U8(data) => HostNumericDataView::U8(data),
            Self::U16(data) => HostNumericDataView::U16(data),
            Self::U32(data) => HostNumericDataView::U32(data),
            Self::U64(data) => HostNumericDataView::U64(data),
        }
    }

    pub fn len(&self) -> usize {
        self.as_view().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Owned tensor transfer with one native numeric payload and explicit
/// real/complex layout.
#[derive(Debug, Clone, PartialEq)]
pub struct HostNumericTensorOwned {
    pub data: HostNumericDataOwned,
    pub shape: Vec<usize>,
    pub storage: GpuTensorStorage,
}

impl HostNumericTensorOwned {
    pub fn as_view(&self) -> HostNumericTensorView<'_> {
        HostNumericTensorView {
            data: self.data.as_view(),
            shape: &self.shape,
            storage: self.storage,
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        self.as_view().validate()
    }
}

// Minimal host tensor views to avoid depending on runmat-builtins and cycles
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostTensorOwned {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub storage: GpuTensorStorage,
}

#[derive(Debug)]
pub struct HostTensorView<'a> {
    pub data: &'a [f64],
    pub shape: &'a [usize],
}

/// Borrowed exact native-integer buffer used for provider transfers.
#[derive(Debug, Clone, Copy)]
pub enum HostIntegerDataView<'a> {
    I8(&'a [i8]),
    I16(&'a [i16]),
    I32(&'a [i32]),
    I64(&'a [i64]),
    U8(&'a [u8]),
    U16(&'a [u16]),
    U32(&'a [u32]),
    U64(&'a [u64]),
}

impl HostIntegerDataView<'_> {
    pub fn element_type(self) -> IntegerElementType {
        match self {
            Self::I8(_) => IntegerElementType::I8,
            Self::I16(_) => IntegerElementType::I16,
            Self::I32(_) => IntegerElementType::I32,
            Self::I64(_) => IntegerElementType::I64,
            Self::U8(_) => IntegerElementType::U8,
            Self::U16(_) => IntegerElementType::U16,
            Self::U32(_) => IntegerElementType::U32,
            Self::U64(_) => IntegerElementType::U64,
        }
    }

    pub fn len(self) -> usize {
        match self {
            Self::I8(data) => data.len(),
            Self::I16(data) => data.len(),
            Self::I32(data) => data.len(),
            Self::I64(data) => data.len(),
            Self::U8(data) => data.len(),
            Self::U16(data) => data.len(),
            Self::U32(data) => data.len(),
            Self::U64(data) => data.len(),
        }
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
}

/// Borrowed exact integer tensor transfer payload.
#[derive(Debug, Clone, Copy)]
pub struct HostIntegerTensorView<'a> {
    pub data: HostIntegerDataView<'a>,
    pub shape: &'a [usize],
}

/// Owned exact native-integer buffer returned by a provider download.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostIntegerDataOwned {
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

impl HostIntegerDataOwned {
    pub fn element_type(&self) -> IntegerElementType {
        match self {
            Self::I8(_) => IntegerElementType::I8,
            Self::I16(_) => IntegerElementType::I16,
            Self::I32(_) => IntegerElementType::I32,
            Self::I64(_) => IntegerElementType::I64,
            Self::U8(_) => IntegerElementType::U8,
            Self::U16(_) => IntegerElementType::U16,
            Self::U32(_) => IntegerElementType::U32,
            Self::U64(_) => IntegerElementType::U64,
        }
    }
}

/// Exact integer tensor returned by a provider download.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostIntegerTensorOwned {
    pub data: HostIntegerDataOwned,
    pub shape: Vec<usize>,
}

impl<'a> From<HostTensorView<'a>> for HostNumericTensorView<'a> {
    fn from(value: HostTensorView<'a>) -> Self {
        Self {
            data: HostNumericDataView::F64(value.data),
            shape: value.shape,
            storage: GpuTensorStorage::Real,
        }
    }
}

impl<'a> From<&'a HostTensorOwned> for HostNumericTensorView<'a> {
    fn from(value: &'a HostTensorOwned) -> Self {
        Self {
            data: HostNumericDataView::F64(&value.data),
            shape: &value.shape,
            storage: value.storage,
        }
    }
}

impl<'a> From<HostIntegerDataView<'a>> for HostNumericDataView<'a> {
    fn from(value: HostIntegerDataView<'a>) -> Self {
        match value {
            HostIntegerDataView::I8(data) => Self::I8(data),
            HostIntegerDataView::I16(data) => Self::I16(data),
            HostIntegerDataView::I32(data) => Self::I32(data),
            HostIntegerDataView::I64(data) => Self::I64(data),
            HostIntegerDataView::U8(data) => Self::U8(data),
            HostIntegerDataView::U16(data) => Self::U16(data),
            HostIntegerDataView::U32(data) => Self::U32(data),
            HostIntegerDataView::U64(data) => Self::U64(data),
        }
    }
}

impl<'a> From<HostIntegerTensorView<'a>> for HostNumericTensorView<'a> {
    fn from(value: HostIntegerTensorView<'a>) -> Self {
        Self {
            data: value.data.into(),
            shape: value.shape,
            storage: GpuTensorStorage::Real,
        }
    }
}

impl From<HostIntegerDataOwned> for HostNumericDataOwned {
    fn from(value: HostIntegerDataOwned) -> Self {
        match value {
            HostIntegerDataOwned::I8(data) => Self::I8(data),
            HostIntegerDataOwned::I16(data) => Self::I16(data),
            HostIntegerDataOwned::I32(data) => Self::I32(data),
            HostIntegerDataOwned::I64(data) => Self::I64(data),
            HostIntegerDataOwned::U8(data) => Self::U8(data),
            HostIntegerDataOwned::U16(data) => Self::U16(data),
            HostIntegerDataOwned::U32(data) => Self::U32(data),
            HostIntegerDataOwned::U64(data) => Self::U64(data),
        }
    }
}

impl From<HostIntegerTensorOwned> for HostNumericTensorOwned {
    fn from(value: HostIntegerTensorOwned) -> Self {
        Self {
            data: value.data.into(),
            shape: value.shape,
            storage: GpuTensorStorage::Real,
        }
    }
}

/// Lightweight 1-D axis view used by provider meshgrid hooks.
#[derive(Debug)]
pub struct MeshgridAxisView<'a> {
    pub data: &'a [f64],
}

/// Provider-side meshgrid result containing coordinate tensor handles.
#[derive(Debug, Clone)]
pub struct ProviderMeshgridResult {
    pub outputs: Vec<GpuTensorHandle>,
}

/// Single resident GPU axis supplied to provider-side `ndgrid`.
#[derive(Debug, Clone, Copy)]
pub struct ProviderNdgridAxis<'a> {
    pub handle: &'a GpuTensorHandle,
}

/// Provider-side `ndgrid` request. `output_shape` is already MATLAB-normalized
/// by the runtime, including the single-axis `[n, 1]` case.
#[derive(Debug, Clone, Copy)]
pub struct ProviderNdgridRequest<'a> {
    pub axes: &'a [ProviderNdgridAxis<'a>],
    pub output_shape: &'a [usize],
    pub output_count: usize,
}

/// Provider-side ndgrid result containing coordinate tensor handles.
#[derive(Debug, Clone)]
pub struct ProviderNdgridResult {
    pub outputs: Vec<GpuTensorHandle>,
}

/// Single broadcastable resident GPU input supplied to provider-side Black-Scholes pricing.
#[derive(Debug, Clone, Copy)]
pub struct ProviderBlackScholesPriceInput<'a> {
    pub handle: &'a GpuTensorHandle,
    /// Input shape aligned to `output_shape` rank using MATLAB implicit-expansion rules.
    pub shape: &'a [usize],
    /// Column-major strides for `shape`.
    pub strides: &'a [usize],
}

/// Provider-side Black-Scholes price request. Inputs are ordered as
/// Price, Strike, Rate, Time, Volatility, Yield.
#[derive(Debug, Clone, Copy)]
pub struct ProviderBlackScholesPriceRequest<'a> {
    pub inputs: &'a [ProviderBlackScholesPriceInput<'a>],
    pub output_shape: &'a [usize],
    pub len: usize,
}

/// Provider-side Black-Scholes price result containing call and put handles.
#[derive(Debug, Clone)]
pub struct ProviderBlackScholesPriceResult {
    pub call: GpuTensorHandle,
    pub put: GpuTensorHandle,
}

/// Provider-side covariance-to-correlation result containing the correlation
/// matrix and column-vector standard deviations.
#[derive(Debug, Clone)]
pub struct ProviderCovarianceToCorrelationResult {
    pub correlation: GpuTensorHandle,
    pub sigma: GpuTensorHandle,
}

/// Provider-side Adam optimizer update request.
#[derive(Debug, Clone, Copy)]
pub struct ProviderAdamUpdateRequest<'a> {
    pub parameters: &'a GpuTensorHandle,
    pub gradient: &'a GpuTensorHandle,
    pub average_grad: Option<&'a GpuTensorHandle>,
    pub average_sq_grad: Option<&'a GpuTensorHandle>,
    pub iteration: usize,
    pub learn_rate: f64,
    pub gradient_decay_factor: f64,
    pub squared_gradient_decay_factor: f64,
    pub epsilon: f64,
}

/// Provider-side Adam optimizer update result.
#[derive(Debug, Clone)]
pub struct ProviderAdamUpdateResult {
    pub parameters: GpuTensorHandle,
    pub average_grad: GpuTensorHandle,
    pub average_sq_grad: GpuTensorHandle,
}

/// Provider-side cross-entropy mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderCrossentropyMode {
    SingleLabel,
    MultiLabel,
}

/// Provider-side cross-entropy request.
#[derive(Debug, Clone, Copy)]
pub struct ProviderCrossentropyRequest<'a> {
    pub predictions: &'a GpuTensorHandle,
    pub targets: &'a GpuTensorHandle,
    pub weights: Option<&'a GpuTensorHandle>,
    pub mask: Option<&'a GpuTensorHandle>,
    pub mode: ProviderCrossentropyMode,
}

/// Provider-side cross-entropy result containing per-element loss terms.
#[derive(Debug, Clone)]
pub struct ProviderCrossentropyResult {
    pub losses: GpuTensorHandle,
}

/// Descriptor for GEMM epilogues applied to `C = A * B` before storing to `C`.
///
/// Supported operations:
/// - Apply the explicitly ordered scalar affine operations.
/// - Multiply output by per-row and/or per-column scale vectors (broadcasted).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaleOp {
    Multiply,
    Divide,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MatmulEpilogue {
    /// Exact ordered arithmetic. Missing phases perform no arithmetic.
    #[serde(default)]
    pub affine: MatmulAffine,
    /// Optional per-row scale (length m). When present, output[row, col] *= row_scale[row].
    pub row_scale: Option<GpuTensorHandle>,
    /// Optional per-column scale (length n). When present, output[row, col] *= col_scale[col].
    pub col_scale: Option<GpuTensorHandle>,
    /// Row scale operation (multiply or divide). Ignored when `row_scale` is None.
    pub row_op: ScaleOp,
    /// Column scale operation (multiply or divide). Ignored when `col_scale` is None.
    pub col_op: ScaleOp,
    /// Optional lower clamp bound applied after scale/bias.
    #[serde(default)]
    pub clamp_min: Option<f64>,
    /// Optional upper clamp bound applied after scale/bias.
    #[serde(default)]
    pub clamp_max: Option<f64>,
    /// Optional power exponent applied after clamp (final operation in the epilogue).
    #[serde(default)]
    pub pow_exponent: Option<f64>,
    /// Optional output buffer for the diagonal of the result (length min(m, n)).
    #[serde(default)]
    pub diag_output: Option<GpuTensorHandle>,
}

impl MatmulEpilogue {
    pub fn noop() -> Self {
        Self {
            affine: MatmulAffine::default(),
            row_scale: None,
            col_scale: None,
            row_op: ScaleOp::Multiply,
            col_op: ScaleOp::Multiply,
            clamp_min: None,
            clamp_max: None,
            pow_exponent: None,
            diag_output: None,
        }
    }
    pub fn is_noop(&self) -> bool {
        self.affine.scale.is_none()
            && self.affine.offset.is_none()
            && self.row_scale.is_none()
            && self.col_scale.is_none()
            && self.clamp_min.is_none()
            && self.clamp_max.is_none()
            && self.pow_exponent.is_none()
            && self.diag_output.is_none()
    }
}

mod matmul_affine;
pub use matmul_affine::{MatmulAffine, MatmulOffset};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PowerStepEpilogue {
    pub epsilon: f64,
}

impl Default for PowerStepEpilogue {
    fn default() -> Self {
        Self { epsilon: 0.0 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageNormalizeDescriptor {
    pub batch: usize,
    pub height: usize,
    pub width: usize,
    pub epsilon: f64,
    #[serde(default)]
    pub gain: Option<f64>,
    #[serde(default)]
    pub bias: Option<f64>,
    #[serde(default)]
    pub gamma: Option<f64>,
    #[serde(default = "default_image_normalize_clamp_zero")]
    pub clamp_zero: bool,
}

fn default_image_normalize_clamp_zero() -> bool {
    true
}

#[cfg(test)]
mod optional_provider_defaults_tests;

#[cfg(test)]
mod tests {
    use super::*;

    struct TestProvider {
        device_id: u32,
        name: &'static str,
        spawn_concurrency: SpawnHandleConcurrency,
    }

    struct DefaultNumericAdapterProvider;

    #[test]
    fn default_logical_operations_are_typed_unsupported() {
        let provider = DefaultNumericAdapterProvider;
        let input = crate::test_support::handle(
            404,
            vec![1, 1],
            NumericElementType::F64,
            GpuTensorStorage::Real,
        );
        for result in [
            provider.logical_and(&input, &input),
            provider.logical_or(&input, &input),
            provider.logical_xor(&input, &input),
            provider.logical_not(&input),
        ] {
            assert!(is_unsupported_provider_operation(&result.unwrap_err()));
        }
    }

    #[test]
    fn default_adam_update_is_typed_unsupported() {
        let provider = DefaultNumericAdapterProvider;
        let parameter = crate::test_support::handle(
            404,
            vec![1, 2],
            NumericElementType::F64,
            GpuTensorStorage::Real,
        );
        let request = ProviderAdamUpdateRequest {
            parameters: &parameter,
            gradient: &parameter,
            average_grad: None,
            average_sq_grad: None,
            iteration: 1,
            learn_rate: 0.001,
            gradient_decay_factor: 0.9,
            squared_gradient_decay_factor: 0.999,
            epsilon: 1.0e-8,
        };
        let error = provider.adam_update(&request).unwrap_err();
        assert!(is_unsupported_provider_operation(&error));
        let operational = anyhow::anyhow!("execution failed: unsupported device allocation");
        assert!(!is_unsupported_provider_operation(&operational));
    }

    #[test]
    fn generic_fill_preserves_operational_failure_without_upload_retry() {
        struct FailedFill {
            fail_allocation: bool,
            unsupported_after_allocation: bool,
        }
        impl AccelProvider for FailedFill {
            fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
                panic!("operational failure must not retry upload")
            }
            fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
                Box::pin(async { unreachable!() })
            }
            fn device_info(&self) -> String {
                "failed-fill".into()
            }
            fn zeros(&self, shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
                if self.fail_allocation {
                    return Err(std::io::Error::other("allocation failed").into());
                }
                Ok(crate::test_support::handle(
                    404,
                    shape.to_vec(),
                    NumericElementType::F64,
                    GpuTensorStorage::Real,
                ))
            }
            fn scalar_add(&self, _: &GpuTensorHandle, _: f64) -> anyhow::Result<GpuTensorHandle> {
                if self.unsupported_after_allocation {
                    return Err(unsupported_provider_operation(
                        "legacy add refused after allocation",
                    ));
                }
                Err(std::io::Error::other("execution failed").into())
            }
            fn fill(
                &self,
                shape: &[usize],
                value: f64,
            ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
                let base = self.zeros(shape).map_err(|source| {
                    ProviderExecutionError::from(source).with_unproven_effects()
                })?;
                self.scalar_add(&base, value)
                    .map(Some)
                    .map_err(|source| ProviderExecutionError::from(source).after_effect())
            }
        }
        for (fail_allocation, unsupported_after_allocation) in
            [(true, false), (false, false), (false, true)]
        {
            let error = FailedFill {
                fail_allocation,
                unsupported_after_allocation,
            }
            .fill(&[1, 2], 3.0)
            .unwrap_err();
            assert!(!error.is_unsupported());
            assert_eq!(error.is_after_effect(), !fail_allocation);
            let mut source: &(dyn std::error::Error + 'static) = &error;
            while !(source.is::<std::io::Error>() || source.is::<UnsupportedProviderOperation>()) {
                source = source
                    .source()
                    .expect("exact legacy constructor cause retained");
            }
            assert_eq!(
                source.is::<UnsupportedProviderOperation>(),
                unsupported_after_allocation
            );
        }
    }

    #[test]
    fn optional_constant_constructors_default_to_no_attempt() {
        let provider = DefaultNumericAdapterProvider;
        let prototype = crate::test_support::handle(
            404,
            vec![1, 2],
            NumericElementType::F32,
            GpuTensorStorage::Real,
        );
        for result in [
            provider.zeros_like(&prototype, &[2, 3]),
            provider.logical_fill(&[2, 3], true),
            provider.fill_like(&prototype, 2.0),
            provider.ones_like(&prototype),
            provider.ones(&[2, 3]),
            provider.fill(&[2, 3], f64::NAN),
        ] {
            assert!(result.unwrap().is_none());
        }
        struct ZeroOnly {
            owner: crate::test_support::TestAllocationOwner,
            calls: std::sync::atomic::AtomicUsize,
        }
        impl AccelProvider for ZeroOnly {
            fn device_id(&self) -> u32 {
                self.owner.device_id()
            }
            fn device_info(&self) -> String {
                "typed zero-only constructor".into()
            }
            fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
                panic!("typed zero delegation must not upload")
            }
            fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
                panic!("typed zero delegation must not download")
            }
            fn zeros_like(
                &self,
                _: &GpuTensorHandle,
                shape: &[usize],
            ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
                self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                self.owner
                    .allocate_zeroed(
                        shape.to_vec(),
                        NumericElementType::F32,
                        GpuTensorStorage::Real,
                    )
                    .map(|output| Some(output.handle))
                    .map_err(|source| ProviderExecutionError::from(source).with_unproven_effects())
            }
        }
        let provider = ZeroOnly {
            owner: crate::test_support::TestAllocationOwner::new(),
            calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let input = provider
            .owner
            .allocate_zeroed(vec![2, 3], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap();
        let output = provider.fill_like(&input.handle, 0.0).unwrap().unwrap();
        assert_eq!(output.shape(), input.handle.shape());
        assert_eq!(output.physical().element_type(), NumericElementType::F32);
        assert!(!output.shares_allocation_with(&input.handle));
        assert!(provider.fill_like(&input.handle, -0.0).unwrap().is_none());
        assert_eq!(provider.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(input.probe.retirements(), 0);
        // The defaults above still establish pure absence for every hook.
        let provider = DefaultNumericAdapterProvider;
        assert!(provider.flip(&prototype, &[0]).unwrap().is_none());
        assert!(resolve_ready(provider.reduce_mean(&prototype))
            .unwrap()
            .is_none());
        assert!(resolve_ready(provider.reduce_mean_dim(&prototype, 0))
            .unwrap()
            .is_none());
        assert!(resolve_ready(provider.reduce_mean_nd(&prototype, &[0]))
            .unwrap()
            .is_none());
        assert!(is_unsupported_provider_operation(
            &provider.eye(&[2, 3]).unwrap_err()
        ));
        assert!(is_unsupported_provider_operation(
            &provider
                .ndgrid(&ProviderNdgridRequest {
                    axes: &[],
                    output_shape: &[2, 3],
                    output_count: 1,
                })
                .unwrap_err()
        ));
    }

    impl AccelProvider for DefaultNumericAdapterProvider {
        fn device_id(&self) -> u32 {
            404
        }
        fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            assert_eq!(host.data, &[1.0, 2.0]);
            Ok(crate::test_support::handle(
                404,
                host.shape.to_vec(),
                NumericElementType::F64,
                GpuTensorStorage::Real,
            ))
        }

        fn download<'a>(&'a self, h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async move {
                Ok(HostTensorOwned {
                    data: vec![1.0, 2.0],
                    shape: h.shape().to_vec(),
                    storage: GpuTensorStorage::Real,
                })
            })
        }

        fn upload_integer(&self, host: &HostIntegerTensorView) -> anyhow::Result<GpuTensorHandle> {
            assert!(matches!(host.data, HostIntegerDataView::U64([1, u64::MAX])));
            Ok(crate::test_support::handle(
                404,
                host.shape.to_vec(),
                NumericElementType::U64,
                GpuTensorStorage::Real,
            ))
        }

        fn download_integer<'a>(
            &'a self,
            h: &'a GpuTensorHandle,
        ) -> AccelIntegerDownloadFuture<'a> {
            Box::pin(async move {
                Ok(HostIntegerTensorOwned {
                    data: HostIntegerDataOwned::U64(vec![1, u64::MAX]),
                    shape: h.shape().to_vec(),
                })
            })
        }

        fn device_info(&self) -> String {
            "legacy-numeric-provider".to_string()
        }
    }

    impl AccelProvider for TestProvider {
        fn upload(&self, _host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Err(anyhow!("test provider upload should not be called"))
        }

        fn download<'a>(&'a self, _h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            unsupported_future("test provider download should not be called")
        }

        fn device_info(&self) -> String {
            self.name.to_string()
        }

        fn device_id(&self) -> u32 {
            self.device_id
        }

        fn spawn_handle_concurrency(&self) -> SpawnHandleConcurrency {
            self.spawn_concurrency
        }
    }

    #[test]
    fn default_device_info_uses_the_provider_owner_id() {
        let info = PROVIDER_A.device_info_struct();
        assert_eq!(info.device_id, PROVIDER_A.device_id());
        assert_eq!(info.device_id, 101);
        assert_eq!(info.name, "provider-a");
    }

    static PROVIDER_TEST_LOCK: Lazy<std::sync::Mutex<()>> = Lazy::new(|| std::sync::Mutex::new(()));
    static PROVIDER_A: TestProvider = TestProvider {
        device_id: 101,
        name: "provider-a",
        spawn_concurrency: SpawnHandleConcurrency::ImmutableShare,
    };
    static PROVIDER_B: TestProvider = TestProvider {
        device_id: 202,
        name: "provider-b",
        spawn_concurrency: SpawnHandleConcurrency::Reject,
    };
    static PROVIDER_C: TestProvider = TestProvider {
        device_id: 303,
        name: "provider-c",
        spawn_concurrency: SpawnHandleConcurrency::CopyOnWrite,
    };

    fn resolve_ready<F: Future>(future: F) -> F::Output {
        struct NoopWake;
        impl std::task::Wake for NoopWake {
            fn wake(self: std::sync::Arc<Self>) {}
        }
        let waker = std::task::Waker::from(std::sync::Arc::new(NoopWake));
        let mut context = std::task::Context::from_waker(&waker);
        let mut future = std::pin::pin!(future);
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(output) => output,
            std::task::Poll::Pending => panic!("test adapter future unexpectedly pending"),
        }
    }

    fn register_test_providers() {
        clear_provider();
        unsafe {
            register_provider(&PROVIDER_A);
            register_provider(&PROVIDER_B);
        }
    }

    fn test_handle(device_id: u32) -> GpuTensorHandle {
        crate::test_support::handle(
            device_id,
            vec![1],
            NumericElementType::F64,
            GpuTensorStorage::Real,
        )
    }

    #[test]
    fn numeric_transfer_contract_is_exhaustive_and_native() {
        let f64_values = [1.0_f64, 2.0];
        let f32_values = [1.0_f32, 2.0];
        let i8_values = [1_i8, 2];
        let i16_values = [1_i16, 2];
        let i32_values = [1_i32, 2];
        let i64_values = [1_i64, 9_007_199_254_740_993];
        let u8_values = [1_u8, 2];
        let u16_values = [1_u16, 2];
        let u32_values = [1_u32, 2];
        let u64_values = [1_u64, 9_007_199_254_740_993];
        let cases = [
            (
                HostNumericDataView::F64(&f64_values),
                NumericElementType::F64,
            ),
            (
                HostNumericDataView::F32(&f32_values),
                NumericElementType::F32,
            ),
            (HostNumericDataView::I8(&i8_values), NumericElementType::I8),
            (
                HostNumericDataView::I16(&i16_values),
                NumericElementType::I16,
            ),
            (
                HostNumericDataView::I32(&i32_values),
                NumericElementType::I32,
            ),
            (
                HostNumericDataView::I64(&i64_values),
                NumericElementType::I64,
            ),
            (HostNumericDataView::U8(&u8_values), NumericElementType::U8),
            (
                HostNumericDataView::U16(&u16_values),
                NumericElementType::U16,
            ),
            (
                HostNumericDataView::U32(&u32_values),
                NumericElementType::U32,
            ),
            (
                HostNumericDataView::U64(&u64_values),
                NumericElementType::U64,
            ),
        ];

        for (data, element_type) in cases {
            let transfer = HostNumericTensorView {
                data,
                shape: &[1, 2],
                storage: GpuTensorStorage::Real,
            };
            transfer.validate().expect("matching native transfer");
            assert_eq!(transfer.element_type(), element_type);
            assert_eq!(element_type.class_name(), data.element_type().class_name());
            assert_eq!(
                element_type.element_size(),
                data.element_type().element_size()
            );
        }

        assert_eq!(
            NumericElementType::F64.precision(),
            Some(ProviderPrecision::F64)
        );
        assert_eq!(
            NumericElementType::F32.precision(),
            Some(ProviderPrecision::F32)
        );
        assert_eq!(
            NumericElementType::U64.integer_type(),
            Some(IntegerElementType::U64)
        );
        assert_eq!(
            NumericElementType::I64.integer_type(),
            Some(IntegerElementType::I64)
        );
        assert_eq!(NumericElementType::F64.integer_type(), None);
    }

    #[test]
    fn numeric_transfer_contract_validates_real_and_complex_lengths() {
        let real = HostNumericTensorOwned {
            data: HostNumericDataOwned::F32(vec![1.0, 2.0]),
            shape: vec![2, 1],
            storage: GpuTensorStorage::Real,
        };
        real.validate().expect("native single real transfer");

        let complex = HostNumericTensorOwned {
            data: HostNumericDataOwned::U64(vec![1, u64::MAX, 2, u64::MAX - 1]),
            shape: vec![1, 2],
            storage: GpuTensorStorage::ComplexInterleaved,
        };
        complex.validate().expect("native complex integer transfer");

        let mismatch = HostNumericTensorView {
            data: HostNumericDataView::I16(&[1, 2, 3]),
            shape: &[2, 2],
            storage: GpuTensorStorage::Real,
        };
        let error = mismatch.validate().expect_err("shape mismatch must reject");
        assert!(error.to_string().contains("expected 4"));

        let overflow = HostNumericTensorView {
            data: HostNumericDataView::F64(&[]),
            shape: &[usize::MAX, 2],
            storage: GpuTensorStorage::Real,
        };
        assert!(overflow.validate().is_err());

        for shape in [
            [0, usize::MAX, usize::MAX],
            [usize::MAX, 0, usize::MAX],
            [usize::MAX, usize::MAX, 0],
        ] {
            for storage in [GpuTensorStorage::Real, GpuTensorStorage::ComplexInterleaved] {
                HostNumericTensorView {
                    data: HostNumericDataView::U64(&[]),
                    shape: &shape,
                    storage,
                }
                .validate()
                .expect("zero extent wins independently of dimension order");
            }
        }
    }

    #[test]
    fn gpu_handle_descriptor_survives_clone_serialization_and_semantic_metadata_cleanup() {
        let handle = crate::test_support::handle(
            17,
            vec![2, 3],
            NumericElementType::U64,
            GpuTensorStorage::ComplexInterleaved,
        )
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
        let cloned = handle.clone();
        assert!(cloned.shares_allocation_with(&handle));
        assert_eq!(handle_integer_type(&cloned), Some(IntegerElementType::U64));
        assert_eq!(
            handle_storage(&cloned),
            GpuTensorStorage::ComplexInterleaved
        );
        assert_eq!(
            handle_class_identity(&cloned),
            Some(standard::UINT64.owned())
        );
        let wire: GpuTensorWireDescriptor =
            serde_json::from_str(&serde_json::to_string(&cloned).unwrap()).unwrap();
        assert_eq!(wire, handle.wire_descriptor());
        let acquired = handle
            .allocation_lease()
            .downgrade()
            .acquire_tensor(&wire)
            .unwrap();
        assert!(acquired.shares_allocation_with(&handle));
        assert!(serde_json::from_str::<GpuTensorWireDescriptor>(
            r#"{"shape":[1,1],"device_id":5,"buffer_id":8}"#
        )
        .is_err());
    }

    #[test]
    fn durable_physical_descriptor_is_the_only_numeric_authority() {
        let floating = crate::test_support::handle(
            23,
            vec![1, 2],
            NumericElementType::F32,
            GpuTensorStorage::ComplexInterleaved,
        );

        assert_eq!(handle_precision(&floating), Some(ProviderPrecision::F32));
        assert_eq!(handle_integer_type(&floating), None);
        assert_eq!(
            handle_storage(&floating),
            GpuTensorStorage::ComplexInterleaved
        );
        assert_eq!(
            handle_class_identity(&floating),
            Some(standard::SINGLE.owned())
        );

        let integer = crate::test_support::handle(
            37,
            vec![1, 2],
            NumericElementType::U64,
            GpuTensorStorage::Real,
        );

        assert_eq!(handle_precision(&integer), None);
        assert_eq!(handle_integer_type(&integer), Some(IntegerElementType::U64));
        assert_eq!(
            handle_class_identity(&integer),
            Some(standard::UINT64.owned())
        );
    }

    #[test]
    fn removed_physical_side_metadata_surface_stays_absent() {
        let source = include_str!("lib.rs");
        let forbidden = [
            ["set_handle_", "class_identity"].concat(),
            ["set_handle_", "logical"].concat(),
            ["record_handle_", "transpose"].concat(),
            ["set_handle_", "precision"].concat(),
            ["set_handle_", "integer_type"].concat(),
            ["set_handle_", "storage"].concat(),
            ["clear_handle_", "precision"].concat(),
            ["clear_handle_", "integer_type"].concat(),
            ["clear_handle_", "storage"].concat(),
            ["HANDLE_", "PRECISIONS"].concat(),
            ["HANDLE_", "INTEGER_TYPES"].concat(),
            ["HANDLE_", "STORAGES"].concat(),
        ];
        for symbol in forbidden {
            assert!(
                !source.contains(&symbol),
                "removed symbol returned: {symbol}"
            );
        }
    }

    #[test]
    fn numeric_transfer_default_adapter_preserves_double_and_integer_storage() {
        let provider = DefaultNumericAdapterProvider;
        let double = HostNumericTensorView {
            data: HostNumericDataView::F64(&[1.0, 2.0]),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        };
        let double_handle = provider
            .upload_numeric(&double)
            .expect("double upload adapter");
        let double_download = resolve_ready(provider.download_numeric(&double_handle))
            .expect("double download adapter");
        assert_eq!(
            double_download.data,
            HostNumericDataOwned::F64(vec![1.0, 2.0])
        );

        let integer = HostNumericTensorView {
            data: HostNumericDataView::U64(&[1, u64::MAX]),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        };
        let integer_handle = provider
            .upload_numeric(&integer)
            .expect("integer upload adapter");
        let integer_download = resolve_ready(provider.download_numeric(&integer_handle))
            .expect("integer download adapter");
        assert_eq!(
            integer_download.data,
            HostNumericDataOwned::U64(vec![1, u64::MAX])
        );
    }

    #[test]
    fn numeric_transfer_default_adapter_rejects_widened_single_and_complex_integer() {
        let provider = DefaultNumericAdapterProvider;
        let single = HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, 2.0]),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        };
        assert!(provider.upload_numeric(&single).is_err());

        let complex_integer = HostNumericTensorView {
            data: HostNumericDataView::I16(&[1, 2, 3, 4]),
            shape: &[1, 2],
            storage: GpuTensorStorage::ComplexInterleaved,
        };
        assert!(provider.upload_numeric(&complex_integer).is_err());

        let single_handle = crate::test_support::handle(
            404,
            vec![1, 2],
            NumericElementType::F32,
            GpuTensorStorage::Real,
        );
        let error = resolve_ready(provider.download_numeric(&single_handle))
            .expect_err("widened single download must reject");
        assert!(error.to_string().contains("native single"));

        let complex_integer_handle = crate::test_support::handle(
            404,
            vec![1, 2],
            NumericElementType::I16,
            GpuTensorStorage::ComplexInterleaved,
        );
        let error = resolve_ready(provider.download_numeric(&complex_integer_handle))
            .expect_err("complex integer default download must reject");
        assert!(error.to_string().contains("complex integer"));

        // Descriptor-less data remains wire data; it cannot mint a live handle.
        assert!(serde_json::from_str::<GpuTensorWireDescriptor>(r#"{}"#).is_err());
    }

    #[test]
    fn conservative_feasibility_rejects_operations_without_execution() {
        let query = ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new("legacy.elementwise"),
            family: ProviderOperationFamily::Elementwise,
            inputs: vec![ProviderRepresentation {
                element_type: ProviderElementType::F64,
                storage: ProviderStorage::DenseReal,
                layout: ProviderLayout::ColumnMajorContiguous,
                shape: vec![4, 4],
                residency: ProviderResidency::Host,
            }],
            outputs: Vec::new(),
            workload: ProviderWorkload {
                elements: Some(16),
                ..ProviderWorkload::default()
            },
        };

        assert!(matches!(
            PROVIDER_A.query_feasibility(&query),
            ProviderFeasibility::Rejected {
                rejection: ProviderRejection {
                    code: ProviderRejectionCode::UnsupportedOperation,
                    ..
                }
            }
        ));
        assert!(!PROVIDER_A
            .capability_snapshot()
            .supports(ProviderOperationFamily::Elementwise));

        let snapshot = PROVIDER_A.capability_snapshot();
        let encoded = serde_json::to_string(&snapshot).expect("serialize capability snapshot");
        let decoded: ProviderCapabilitySnapshot =
            serde_json::from_str(&encoded).expect("deserialize capability snapshot");
        assert_eq!(decoded, snapshot);
    }

    #[test]
    fn semantic_handle_metadata_is_namespaced_by_device_and_buffer() {
        let (logical, _) = crate::test_support::owned(
            PROVIDER_A.device_id(),
            vec![2, 3],
            NumericElementType::F32,
            GpuTensorStorage::Real,
            true,
        );
        let first = logical
            .with_view(
                logical
                    .view()
                    .clone()
                    .with_shape(vec![3, 2])
                    .with_layout(GpuTensorLayout::Transpose(TransposeInfo {
                        base_rows: 2,
                        base_cols: 3,
                    }))
                    .with_provenance(GpuHandleProvenance::Automatic),
            )
            .unwrap();
        let second = crate::test_support::handle(
            PROVIDER_B.device_id(),
            vec![2, 3],
            NumericElementType::U64,
            GpuTensorStorage::ComplexInterleaved,
        )
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
        assert_eq!(
            handle_class_identity(&first),
            Some(standard::LOGICAL.owned())
        );
        assert_eq!(
            handle_class_identity(&second),
            Some(standard::UINT64.owned())
        );
        assert_eq!(handle_precision(&first), Some(ProviderPrecision::F32));
        assert_eq!(handle_integer_type(&second), Some(IntegerElementType::U64));
        assert!(handle_is_transposed(&first));
        assert!(!handle_is_transposed(&logical));
        assert!(first.shares_allocation_with(&logical));
        assert_ne!(handle_identity(&first), handle_identity(&logical));
        assert!(handle_is_explicit(&second));
        let numeric = crate::test_support::handle(
            17,
            vec![2, 3],
            NumericElementType::F32,
            GpuTensorStorage::Real,
        );
        assert!(numeric
            .with_view(numeric.view().clone().with_logical(true))
            .is_err());
        assert_eq!(
            handle_class_identity(&numeric),
            Some(standard::SINGLE.owned())
        );
    }

    fn spectral_input() -> GpuTensorHandle {
        crate::test_support::handle(
            PROVIDER_A.device_id(),
            vec![16, 1],
            NumericElementType::F64,
            GpuTensorStorage::Real,
        )
    }

    fn spectral_request<'a>(
        input: &'a GpuTensorHandle,
        frame_mode: ProviderSpectralFrameMode,
    ) -> ProviderSpectralRequest<'a> {
        static WINDOW: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
        ProviderSpectralRequest {
            input,
            input_len: 16,
            input_complex: false,
            window: &WINDOW,
            nfft: 8,
            frame_count: 3,
            frame_mode,
            range: ProviderSpectralRange::Onesided,
            denominator: 1.0,
        }
    }

    #[test]
    fn provider_envelope_shape_guard_rejects_equal_len_layout_spoofing() {
        assert!(provider_envelope_input_shape_matches(&[2, 3], 2, 3));
        assert!(provider_envelope_input_shape_matches(&[6, 1], 6, 1));
        assert!(provider_envelope_input_shape_matches(&[1, 6], 6, 1));
        assert!(provider_envelope_input_shape_matches(&[6], 6, 1));

        assert!(!provider_envelope_input_shape_matches(&[3, 2], 2, 3));
        assert!(!provider_envelope_input_shape_matches(&[6], 2, 3));
        assert!(!provider_envelope_input_shape_matches(&[2, 1, 3], 2, 3));
    }

    #[test]
    fn provider_for_device_prefers_registered_device_over_thread_provider() {
        let _lock = PROVIDER_TEST_LOCK
            .lock()
            .expect("provider test lock poisoned");
        register_test_providers();
        let _thread_provider = ThreadProviderGuard::set(Some(&PROVIDER_B));

        let provider = provider_for_device(PROVIDER_A.device_id()).expect("provider for device");

        assert_eq!(provider.device_info(), PROVIDER_A.name);
        clear_provider();
    }

    #[test]
    fn provider_for_handle_uses_handle_device_owner() {
        let _lock = PROVIDER_TEST_LOCK
            .lock()
            .expect("provider test lock poisoned");
        register_test_providers();
        let _thread_provider = ThreadProviderGuard::set(Some(&PROVIDER_B));

        let provider =
            provider_for_handle(&test_handle(PROVIDER_A.device_id())).expect("provider for handle");

        assert_eq!(provider.device_info(), PROVIDER_A.name);
        clear_provider();
    }

    #[test]
    fn clearing_active_provider_preserves_live_handle_owner() {
        let _lock = PROVIDER_TEST_LOCK
            .lock()
            .expect("provider test lock poisoned");
        register_test_providers();
        let handle = test_handle(PROVIDER_A.device_id());

        clear_provider();

        assert!(provider().is_none());
        let owner = provider_for_handle(&handle).expect("registered handle owner survives clear");
        assert_eq!(owner.device_info(), PROVIDER_A.name);
    }

    #[test]
    fn concurrent_registration_keeps_each_threads_active_provider() {
        let _lock = PROVIDER_TEST_LOCK
            .lock()
            .expect("provider test lock poisoned");
        clear_provider();

        let first = std::thread::spawn(|| {
            unsafe { register_provider(&PROVIDER_A) };
            std::thread::yield_now();
            provider().map(AccelProvider::device_info)
        });
        let second = std::thread::spawn(|| {
            unsafe { register_provider(&PROVIDER_B) };
            std::thread::yield_now();
            provider().map(AccelProvider::device_info)
        });

        assert_eq!(
            first.join().expect("first registration thread"),
            Some(PROVIDER_A.name.to_string())
        );
        assert_eq!(
            second.join().expect("second registration thread"),
            Some(PROVIDER_B.name.to_string())
        );
        clear_provider();
    }

    #[test]
    fn spawn_handle_concurrency_for_uses_registered_owner() {
        let _lock = PROVIDER_TEST_LOCK
            .lock()
            .expect("provider test lock poisoned");
        register_test_providers();
        let _thread_provider = ThreadProviderGuard::set(Some(&PROVIDER_B));

        let concurrency = spawn_handle_concurrency_for(&test_handle(PROVIDER_A.device_id()))
            .expect("spawn concurrency");

        assert_eq!(concurrency, PROVIDER_A.spawn_concurrency);
        clear_provider();
    }

    #[test]
    fn provider_keeps_thread_local_active_provider_semantics() {
        let _lock = PROVIDER_TEST_LOCK
            .lock()
            .expect("provider test lock poisoned");
        register_test_providers();
        let _thread_provider = ThreadProviderGuard::set(Some(&PROVIDER_A));

        let active = provider().expect("active provider");

        assert_eq!(active.device_info(), PROVIDER_A.name);
        clear_provider();
    }

    #[test]
    fn unregistered_thread_provider_only_matches_own_device_before_global_fallback() {
        let _lock = PROVIDER_TEST_LOCK
            .lock()
            .expect("provider test lock poisoned");
        clear_provider();
        unsafe {
            register_provider(&PROVIDER_A);
        }
        let _thread_provider = ThreadProviderGuard::set(Some(&PROVIDER_C));

        let own_device = provider_for_device(PROVIDER_C.device_id()).expect("own provider");
        let fallback = provider_for_device(404).expect("global fallback provider");

        assert_eq!(own_device.device_info(), PROVIDER_C.name);
        assert_eq!(fallback.device_info(), PROVIDER_A.name);
        clear_provider();
    }

    #[test]
    fn uniform_spectral_request_validates_sliding_input_coverage() {
        let input = spectral_input();
        let mut request = spectral_request(&input, ProviderSpectralFrameMode::Sliding { hop: 6 });
        assert!(validate_uniform_spectral_request(&request).is_ok());

        request.input_len = 15;
        assert!(validate_uniform_spectral_request(&request).is_err());
    }

    #[test]
    fn uniform_spectral_request_rejects_sliding_coverage_overflow() {
        let input = spectral_input();
        let mut request = spectral_request(&input, ProviderSpectralFrameMode::Sliding { hop: 2 });
        request.frame_count = usize::MAX;

        assert!(validate_uniform_spectral_request(&request).is_err());
    }

    #[test]
    fn uniform_spectral_request_validates_folded_input_coverage() {
        let input = spectral_input();
        let mut request = spectral_request(
            &input,
            ProviderSpectralFrameMode::FoldedColumns { input_rows: 5 },
        );
        assert!(validate_uniform_spectral_request(&request).is_ok());

        request.frame_mode = ProviderSpectralFrameMode::FoldedColumns { input_rows: 0 };
        assert!(validate_uniform_spectral_request(&request).is_err());

        request.frame_mode = ProviderSpectralFrameMode::FoldedColumns { input_rows: 6 };
        assert!(validate_uniform_spectral_request(&request).is_err());
    }

    #[test]
    fn uniform_spectral_request_rejects_folded_coverage_overflow() {
        let input = spectral_input();
        let request = spectral_request(
            &input,
            ProviderSpectralFrameMode::FoldedColumns {
                input_rows: usize::MAX,
            },
        );

        assert!(validate_uniform_spectral_request(&request).is_err());
    }

    #[test]
    fn image_normalize_descriptor_omitted_clamp_zero_defaults_true() {
        let payload = r#"{
            "batch": 2,
            "height": 4,
            "width": 5,
            "epsilon": 0.000001
        }"#;

        let desc: ImageNormalizeDescriptor =
            serde_json::from_str(payload).expect("deserialize descriptor");

        assert!(
            desc.clamp_zero,
            "legacy serialized descriptors should default to clamped image normalize"
        );
    }

    #[test]
    fn image_normalize_descriptor_explicit_false_preserves_unclamped() {
        let payload = r#"{
            "batch": 2,
            "height": 4,
            "width": 5,
            "epsilon": 0.000001,
            "clamp_zero": false
        }"#;

        let desc: ImageNormalizeDescriptor =
            serde_json::from_str(payload).expect("deserialize descriptor");

        assert!(
            !desc.clamp_zero,
            "explicit clamp_zero=false should preserve unclamped semantics"
        );
    }

    #[test]
    fn unsupported_provider_operations_have_typed_identity() {
        let unsupported = unsupported_provider_operation("test operation is unavailable");
        let dynamic = unsupported_provider_operation(format!("unsupported precision {}", 64));
        let ordinary = anyhow::anyhow!("test operation is unavailable");

        assert!(is_unsupported_provider_operation(&unsupported));
        assert!(is_unsupported_provider_operation(&dynamic));
        assert!(!is_unsupported_provider_operation(&ordinary));
        assert_eq!(unsupported.to_string(), "test operation is unavailable");
        assert_eq!(dynamic.to_string(), "unsupported precision 64");
    }

    #[test]
    fn provider_integer_elements_round_trip_source_language_classes() {
        for element_type in [
            IntegerElementType::I8,
            IntegerElementType::I16,
            IntegerElementType::I32,
            IntegerElementType::I64,
            IntegerElementType::U8,
            IntegerElementType::U16,
            IntegerElementType::U32,
            IntegerElementType::U64,
        ] {
            let class = runmat_types::IntegerClass::from(element_type);
            assert_eq!(IntegerElementType::from(class), element_type);
        }
    }
}
