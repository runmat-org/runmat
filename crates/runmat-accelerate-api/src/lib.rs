use anyhow::anyhow;
use once_cell::sync::{Lazy, OnceCell};
use serde::{Deserialize, Serialize};
#[cfg(not(target_arch = "wasm32"))]
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, Ordering};
#[cfg(feature = "wgpu")]
use std::sync::Arc;
#[cfg(target_arch = "wasm32")]
use std::sync::Mutex;
use std::sync::RwLock;

type ResidencyClearFn = fn(&GpuTensorHandle);
type SequenceThresholdFn = fn() -> Option<usize>;
type WorkgroupSizeHintFn = fn() -> Option<u32>;

static RESIDENCY_CLEAR: OnceCell<ResidencyClearFn> = OnceCell::new();
static SEQUENCE_THRESHOLD_PROVIDER: OnceCell<SequenceThresholdFn> = OnceCell::new();
static WORKGROUP_SIZE_HINT_PROVIDER: OnceCell<WorkgroupSizeHintFn> = OnceCell::new();

static LOGICAL_HANDLES: Lazy<RwLock<HashSet<u64>>> = Lazy::new(|| RwLock::new(HashSet::new()));
static LOGICAL_HANDLE_HITS: Lazy<RwLock<HashMap<u64, u64>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
static TRANSPOSED_HANDLES: Lazy<RwLock<HashMap<u64, TransposeInfo>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

static HANDLE_PRECISIONS: Lazy<RwLock<HashMap<u64, ProviderPrecision>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransposeInfo {
    pub base_rows: usize,
    pub base_cols: usize,
}

/// Register a callback used to clear residency tracking when GPU tensors are
/// gathered back to the host. Backends that maintain residency metadata should
/// install this hook during initialization.
pub fn register_residency_clear(handler: ResidencyClearFn) {
    let _ = RESIDENCY_CLEAR.set(handler);
}

/// Clear residency metadata for the provided GPU tensor handle, if a backend
/// has registered a handler via [`register_residency_clear`].
pub fn clear_residency(handle: &GpuTensorHandle) {
    if let Some(handler) = RESIDENCY_CLEAR.get() {
        handler(handle);
    }
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
    provider().and_then(|p| p.export_context(kind))
}

/// Request a provider-owned WGPU buffer for zero-copy consumers. Returns `None`
/// when the active provider does not expose buffers or does not support the
/// supplied handle.
#[cfg(feature = "wgpu")]
pub fn export_wgpu_buffer(handle: &GpuTensorHandle) -> Option<WgpuBufferRef> {
    provider().and_then(|p| p.export_wgpu_buffer(handle))
}

/// Record the precision associated with a GPU tensor handle so host operations can
/// reconstruct the original dtype when gathering back to the CPU.
pub fn set_handle_precision(handle: &GpuTensorHandle, precision: ProviderPrecision) {
    if let Ok(mut guard) = HANDLE_PRECISIONS.write() {
        guard.insert(handle.buffer_id, precision);
    }
}

/// Look up the recorded precision for a GPU tensor handle, if any.
pub fn handle_precision(handle: &GpuTensorHandle) -> Option<ProviderPrecision> {
    HANDLE_PRECISIONS
        .read()
        .ok()
        .and_then(|guard| guard.get(&handle.buffer_id).copied())
}

/// Clear any recorded precision metadata for a GPU tensor handle.
pub fn clear_handle_precision(handle: &GpuTensorHandle) {
    if let Ok(mut guard) = HANDLE_PRECISIONS.write() {
        guard.remove(&handle.buffer_id);
    }
}

/// Annotate a GPU tensor handle as logically-typed (`logical` in MATLAB terms)
/// or clear the logical flag when `logical` is `false`.
pub fn set_handle_logical(handle: &GpuTensorHandle, logical: bool) {
    if let Ok(mut guard) = LOGICAL_HANDLES.write() {
        if logical {
            guard.insert(handle.buffer_id);
            if let Ok(mut hits) = LOGICAL_HANDLE_HITS.write() {
                *hits.entry(handle.buffer_id).or_insert(0) += 1;
            }
        } else {
            guard.remove(&handle.buffer_id);
            if let Ok(mut hits) = LOGICAL_HANDLE_HITS.write() {
                hits.remove(&handle.buffer_id);
            }
        }
    }
}

/// Convenience helper for clearing logical annotations explicitly.
pub fn clear_handle_logical(handle: &GpuTensorHandle) {
    set_handle_logical(handle, false);
}

/// Returns true when the supplied handle has been marked as logical.
pub fn handle_is_logical(handle: &GpuTensorHandle) -> bool {
    LOGICAL_HANDLES
        .read()
        .map(|guard| guard.contains(&handle.buffer_id))
        .unwrap_or(false)
}

pub fn handle_logical_hits(buffer_id: u64) -> Option<u64> {
    LOGICAL_HANDLE_HITS
        .read()
        .ok()
        .and_then(|guard| guard.get(&buffer_id).copied())
}

pub fn record_handle_transpose(handle: &GpuTensorHandle, base_rows: usize, base_cols: usize) {
    if let Ok(mut guard) = TRANSPOSED_HANDLES.write() {
        guard.insert(
            handle.buffer_id,
            TransposeInfo {
                base_rows,
                base_cols,
            },
        );
    }
}

pub fn clear_handle_transpose(handle: &GpuTensorHandle) {
    if let Ok(mut guard) = TRANSPOSED_HANDLES.write() {
        guard.remove(&handle.buffer_id);
    }
}

pub fn handle_transpose_info(handle: &GpuTensorHandle) -> Option<TransposeInfo> {
    TRANSPOSED_HANDLES
        .read()
        .ok()
        .and_then(|guard| guard.get(&handle.buffer_id).copied())
}

pub fn handle_is_transposed(handle: &GpuTensorHandle) -> bool {
    handle_transpose_info(handle).is_some()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuTensorHandle {
    pub shape: Vec<usize>,
    pub device_id: u32,
    pub buffer_id: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApiDeviceInfo {
    pub device_id: u32,
    pub name: String,
    pub vendor: String,
    pub memory_bytes: Option<u64>,
    pub backend: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReduceDimResult {
    pub values: GpuTensorHandle,
    pub indices: GpuTensorHandle,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter: Arc<wgpu::Adapter>,
    pub adapter_info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
    pub features: wgpu::Features,
}

/// Borrowed reference to a provider-owned WGPU buffer corresponding to a `GpuTensorHandle`.
#[cfg(feature = "wgpu")]
#[derive(Clone)]
pub struct WgpuBufferRef {
    pub buffer: Arc<wgpu::Buffer>,
    pub len: usize,
    pub shape: Vec<usize>,
    pub element_size: usize,
    pub precision: ProviderPrecision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PagefunOp {
    Mtimes,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderLuResult {
    pub combined: GpuTensorHandle,
    pub lower: GpuTensorHandle,
    pub upper: GpuTensorHandle,
    pub perm_matrix: GpuTensorHandle,
    pub perm_vector: GpuTensorHandle,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderCholResult {
    pub factor: GpuTensorHandle,
    /// MATLAB-compatible failure index (0 indicates success).
    pub info: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderQrResult {
    pub q: GpuTensorHandle,
    pub r: GpuTensorHandle,
    pub perm_matrix: GpuTensorHandle,
    pub perm_vector: GpuTensorHandle,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderQrPowerIterResult {
    pub q: GpuTensorHandle,
    pub r: GpuTensorHandle,
    pub perm_matrix: GpuTensorHandle,
    pub perm_vector: GpuTensorHandle,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct ProviderLinsolveOptions {
    pub lower: bool,
    pub upper: bool,
    pub rectangular: bool,
    pub transposed: bool,
    pub conjugate: bool,
    pub symmetric: bool,
    pub posdef: bool,
    pub rcond: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    pub r_matrix: Vec<f64>,
    pub normr: f64,
    pub df: f64,
    pub mu: [f64; 2],
}

/// Numerator/denominator payload returned by provider-backed `polyder` quotient rule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
pub struct ProviderQrOptions {
    pub economy: bool,
    pub pivot: ProviderQrPivot,
}

impl Default for ProviderQrOptions {
    fn default() -> Self {
        Self {
            economy: false,
            pivot: ProviderQrPivot::Matrix,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderPrecision {
    F32,
    F64,
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

/// Direction used when computing prefix sums on the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderScanDirection {
    Forward,
    Reverse,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderIirFilterOptions {
    /// Zero-based dimension along which filtering should be applied.
    pub dim: usize,
    /// Optional initial conditions (state vector) residing on the device.
    pub zi: Option<GpuTensorHandle>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderIirFilterResult {
    /// Filtered output tensor, matching the input signal shape.
    pub output: GpuTensorHandle,
    /// Final conditions for the filter state (same shape as the requested `zi` layout).
    pub final_state: Option<GpuTensorHandle>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProviderTelemetry {
    pub fused_elementwise: ProviderDispatchStats,
    pub fused_reduction: ProviderDispatchStats,
    pub matmul: ProviderDispatchStats,
    pub upload_bytes: u64,
    pub download_bytes: u64,
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

fn unsupported_future<T>(message: &'static str) -> AccelProviderFuture<'static, T> {
    Box::pin(async move { Err(anyhow::anyhow!(message)) })
}

/// Device/provider interface that backends implement and register into the runtime layer
pub trait AccelProvider: Send + Sync {
    fn upload(&self, host: &crate::HostTensorView) -> anyhow::Result<GpuTensorHandle>;
    fn download<'a>(&'a self, h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a>;
    fn free(&self, h: &GpuTensorHandle) -> anyhow::Result<()>;
    fn device_info(&self) -> String;
    fn device_id(&self) -> u32 {
        0
    }

    /// Export a shared GPU context handle, allowing downstream systems (plotting, visualization)
    /// to reuse the same device/queue without copying tensor data back to the host.
    fn export_context(&self, _kind: AccelContextKind) -> Option<AccelContextHandle> {
        None
    }

    /// Export a provider-owned WGPU buffer for zero-copy integrations.
    #[cfg(feature = "wgpu")]
    fn export_wgpu_buffer(&self, _handle: &GpuTensorHandle) -> Option<WgpuBufferRef> {
        let _ = _handle;
        None
    }

    /// Gather elements from `source` at the provided zero-based linear `indices`, materialising
    /// a dense tensor with the specified `output_shape`.
    fn gather_linear(
        &self,
        _source: &GpuTensorHandle,
        _indices: &[u32],
        _output_shape: &[usize],
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("gather_linear not supported by provider"))
    }

    /// Scatter the contents of `values` into `target` at the provided zero-based linear `indices`.
    ///
    /// The provider must ensure `values.len() == indices.len()` and update `target` in place.
    fn scatter_linear(
        &self,
        _target: &GpuTensorHandle,
        _indices: &[u32],
        _values: &GpuTensorHandle,
    ) -> anyhow::Result<()> {
        Err(anyhow::anyhow!("scatter_linear not supported by provider"))
    }

    /// Structured device information (optional to override). Default adapts from `device_info()`.
    fn device_info_struct(&self) -> ApiDeviceInfo {
        ApiDeviceInfo {
            device_id: 0,
            name: self.device_info(),
            vendor: String::new(),
            memory_bytes: None,
            backend: None,
        }
    }

    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }

    /// Read a single scalar at linear index from a device tensor, returning it as f64.
    fn read_scalar(&self, _h: &GpuTensorHandle, _linear_index: usize) -> anyhow::Result<f64> {
        Err(anyhow::anyhow!("read_scalar not supported by provider"))
    }

    /// Allocate a zero-initialised tensor with the provided shape on the device.
    fn zeros(&self, _shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("zeros not supported by provider"))
    }

    /// Allocate a one-initialised tensor with the provided shape on the device.
    fn ones(&self, _shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("ones not supported by provider"))
    }

    /// Allocate a zero-initialised tensor matching the prototype tensor.
    fn zeros_like(&self, prototype: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        self.zeros(&prototype.shape)
    }

    /// Allocate a tensor filled with a constant value on the device.
    fn fill(&self, shape: &[usize], value: f64) -> anyhow::Result<GpuTensorHandle> {
        if value == 0.0 {
            return self.zeros(shape);
        }
        if let Ok(base) = self.zeros(shape) {
            match self.scalar_add(&base, value) {
                Ok(out) => {
                    let _ = self.free(&base);
                    return Ok(out);
                }
                Err(_) => {
                    let _ = self.free(&base);
                }
            }
        }
        let len: usize = shape.iter().copied().product();
        let data = vec![value; len];
        let view = HostTensorView { data: &data, shape };
        self.upload(&view)
    }

    /// Allocate a tensor filled with a constant value, matching a prototype's residency.
    fn fill_like(
        &self,
        prototype: &GpuTensorHandle,
        value: f64,
    ) -> anyhow::Result<GpuTensorHandle> {
        if value == 0.0 {
            return self.zeros_like(prototype);
        }
        if let Ok(base) = self.zeros_like(prototype) {
            match self.scalar_add(&base, value) {
                Ok(out) => {
                    let _ = self.free(&base);
                    return Ok(out);
                }
                Err(_) => {
                    let _ = self.free(&base);
                }
            }
        }
        self.fill(&prototype.shape, value)
    }

    /// Allocate a one-initialised tensor matching the prototype tensor.
    fn ones_like(&self, prototype: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        self.ones(&prototype.shape)
    }

    /// Allocate an identity tensor with ones along the leading diagonal of the first two axes.
    fn eye(&self, _shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("eye not supported by provider"))
    }

    /// Allocate an identity tensor matching the prototype tensor's shape.
    fn eye_like(&self, prototype: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        self.eye(&prototype.shape)
    }

    /// Construct MATLAB-style coordinate grids from axis vectors.
    fn meshgrid(&self, _axes: &[MeshgridAxisView<'_>]) -> anyhow::Result<ProviderMeshgridResult> {
        Err(anyhow::anyhow!("meshgrid not supported by provider"))
    }

    /// Construct a diagonal matrix from a vector-like tensor. `offset` matches MATLAB semantics.
    fn diag_from_vector(
        &self,
        _vector: &GpuTensorHandle,
        _offset: isize,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!(
            "diag_from_vector not supported by provider"
        ))
    }

    /// Extract a diagonal from a matrix-like tensor. The result is always a column vector.
    fn diag_extract(
        &self,
        _matrix: &GpuTensorHandle,
        _offset: isize,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("diag_extract not supported by provider"))
    }

    /// Apply a lower-triangular mask to the first two dimensions of a tensor.
    fn tril<'a>(
        &'a self,
        _matrix: &'a GpuTensorHandle,
        _offset: isize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { Err(anyhow!("tril not supported by provider")) })
    }

    /// Apply an upper-triangular mask to the first two dimensions of a tensor.
    fn triu<'a>(
        &'a self,
        _matrix: &'a GpuTensorHandle,
        _offset: isize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { Err(anyhow!("triu not supported by provider")) })
    }

    /// Evaluate a polynomial expressed by `coefficients` at each element in `points`.
    fn polyval(
        &self,
        _coefficients: &GpuTensorHandle,
        _points: &GpuTensorHandle,
        _options: &ProviderPolyvalOptions,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("polyval not supported by provider"))
    }

    /// Fit a polynomial of degree `degree` to `(x, y)` samples. Optional weights must match `x`.
    fn polyfit<'a>(
        &'a self,
        _x: &'a GpuTensorHandle,
        _y: &'a GpuTensorHandle,
        _degree: usize,
        _weights: Option<&'a GpuTensorHandle>,
    ) -> AccelProviderFuture<'a, ProviderPolyfitResult> {
        Box::pin(async move { Err(anyhow::anyhow!("polyfit not supported by provider")) })
    }

    /// Differentiate a polynomial represented as a vector of coefficients.
    fn polyder_single<'a>(
        &'a self,
        _polynomial: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { Err(anyhow::anyhow!("polyder_single not supported by provider")) })
    }

    /// Apply the product rule to polynomials `p` and `q`.
    fn polyder_product<'a>(
        &'a self,
        _p: &'a GpuTensorHandle,
        _q: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { Err(anyhow::anyhow!("polyder_product not supported by provider")) })
    }

    /// Apply the quotient rule to polynomials `u` and `v`.
    fn polyder_quotient<'a>(
        &'a self,
        _u: &'a GpuTensorHandle,
        _v: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, ProviderPolyderQuotient> {
        Box::pin(async move {
            Err(anyhow::anyhow!(
                "polyder_quotient not supported by provider"
            ))
        })
    }

    /// Integrate a polynomial represented as a vector of coefficients and append a constant term.
    fn polyint(
        &self,
        _polynomial: &GpuTensorHandle,
        _constant: f64,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("polyint not supported by provider"))
    }

    /// Allocate a tensor filled with random values drawn from U(0, 1).
    fn random_uniform(&self, _shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("random_uniform not supported by provider"))
    }

    /// Allocate a tensor filled with random values matching the prototype shape.
    fn random_uniform_like(&self, prototype: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        self.random_uniform(&prototype.shape)
    }

    /// Allocate a tensor filled with standard normal (mean 0, stddev 1) random values.
    fn random_normal(&self, _shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("random_normal not supported by provider"))
    }

    /// Allocate a tensor of standard normal values matching a prototype's shape.
    fn random_normal_like(&self, prototype: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        self.random_normal(&prototype.shape)
    }

    fn stochastic_evolution(
        &self,
        _state: &GpuTensorHandle,
        _drift: f64,
        _scale: f64,
        _steps: u32,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!(
            "stochastic_evolution not supported by provider"
        ))
    }

    /// Set the provider RNG state to align with the host RNG.
    fn set_rng_state(&self, _state: u64) -> anyhow::Result<()> {
        Err(anyhow::anyhow!("set_rng_state not supported by provider"))
    }

    /// Generate a 2-D correlation kernel matching MATLAB's `fspecial` builtin.
    fn fspecial(&self, _request: &FspecialRequest) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("fspecial not supported by provider"))
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

    /// Allocate a tensor filled with random integers over an inclusive range.
    fn random_integer_range(
        &self,
        _lower: i64,
        _upper: i64,
        _shape: &[usize],
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!(
            "random_integer_range not supported by provider"
        ))
    }

    /// Allocate a random integer tensor matching the prototype shape.
    fn random_integer_like(
        &self,
        prototype: &GpuTensorHandle,
        lower: i64,
        upper: i64,
    ) -> anyhow::Result<GpuTensorHandle> {
        self.random_integer_range(lower, upper, &prototype.shape)
    }

    /// Allocate a random permutation of 1..=n, returning the first k elements.
    fn random_permutation(&self, _n: usize, _k: usize) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow!("random_permutation not supported by provider"))
    }

    /// Allocate a random permutation matching the prototype residency.
    fn random_permutation_like(
        &self,
        _prototype: &GpuTensorHandle,
        n: usize,
        k: usize,
    ) -> anyhow::Result<GpuTensorHandle> {
        self.random_permutation(n, k)
    }

    /// Compute a covariance matrix across the columns of `matrix`.
    fn covariance<'a>(
        &'a self,
        _matrix: &'a GpuTensorHandle,
        _second: Option<&'a GpuTensorHandle>,
        _weights: Option<&'a GpuTensorHandle>,
        _options: &'a CovarianceOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("covariance not supported by provider")
    }

    /// Compute a correlation coefficient matrix across the columns of `matrix`.
    fn corrcoef<'a>(
        &'a self,
        _matrix: &'a GpuTensorHandle,
        _options: &'a CorrcoefOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("corrcoef not supported by provider")
    }

    // Optional operator hooks (default to unsupported)
    fn linspace(&self, _start: f64, _stop: f64, _count: usize) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("linspace not supported by provider"))
    }
    fn elem_add<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("elem_add not supported by provider")
    }
    fn elem_mul<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("elem_mul not supported by provider")
    }
    fn elem_max<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("elem_max not supported by provider")
    }
    fn elem_min<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("elem_min not supported by provider")
    }
    fn elem_sub<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("elem_sub not supported by provider")
    }
    fn elem_div<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("elem_div not supported by provider")
    }
    fn elem_pow<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("elem_pow not supported by provider")
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
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("elem_ge not supported by provider")
    }
    fn elem_le<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("elem_le not supported by provider")
    }
    fn elem_lt<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("elem_lt not supported by provider")
    }
    fn elem_gt<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("elem_gt not supported by provider")
    }
    fn elem_eq<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("elem_eq not supported by provider")
    }
    fn elem_ne<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("elem_ne not supported by provider")
    }
    fn logical_and(
        &self,
        _a: &GpuTensorHandle,
        _b: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("logical_and not supported by provider"))
    }
    fn logical_or(
        &self,
        _a: &GpuTensorHandle,
        _b: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("logical_or not supported by provider"))
    }
    fn logical_xor(
        &self,
        _a: &GpuTensorHandle,
        _b: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("logical_xor not supported by provider"))
    }
    fn logical_not(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("logical_not not supported by provider"))
    }
    fn logical_islogical(&self, a: &GpuTensorHandle) -> anyhow::Result<bool> {
        Ok(handle_is_logical(a))
    }
    fn logical_isreal(&self, _a: &GpuTensorHandle) -> anyhow::Result<bool> {
        Err(anyhow::anyhow!("logical_isreal not supported by provider"))
    }
    fn logical_isfinite(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!(
            "logical_isfinite not supported by provider"
        ))
    }
    fn logical_isnan(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("logical_isnan not supported by provider"))
    }
    fn logical_isinf(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("logical_isinf not supported by provider"))
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
    fn unary_gamma<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_gamma not supported by provider")
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
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_conj not supported by provider")
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
    fn unary_exp<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_exp not supported by provider")
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
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_double not supported by provider")
    }
    fn unary_single<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_single not supported by provider")
    }
    fn unary_pow2<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("unary_pow2 not supported by provider")
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
        Err(anyhow::anyhow!("scalar_add not supported by provider"))
    }
    fn scalar_sub(&self, _a: &GpuTensorHandle, _scalar: f64) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("scalar_sub not supported by provider"))
    }
    fn scalar_mul(&self, _a: &GpuTensorHandle, _scalar: f64) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("scalar_mul not supported by provider"))
    }
    fn scalar_max(&self, _a: &GpuTensorHandle, _scalar: f64) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("scalar_max not supported by provider"))
    }
    fn scalar_min(&self, _a: &GpuTensorHandle, _scalar: f64) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("scalar_min not supported by provider"))
    }
    fn scalar_div(&self, _a: &GpuTensorHandle, _scalar: f64) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("scalar_div not supported by provider"))
    }
    fn sort_dim<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dim: usize,
        _order: SortOrder,
        _comparison: SortComparison,
    ) -> AccelProviderFuture<'a, SortResult> {
        unsupported_future("sort_dim not supported by provider")
    }
    fn sort_rows<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _columns: &'a [SortRowsColumnSpec],
        _comparison: SortComparison,
    ) -> AccelProviderFuture<'a, SortResult> {
        unsupported_future("sort_rows not supported by provider")
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
        Err(anyhow::anyhow!("pagefun not supported by provider"))
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
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if epilogue.is_noop() {
                return self.matmul(a, b).await;
            }
            Err(anyhow::anyhow!("matmul_epilogue not supported by provider"))
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
    ) -> AccelProviderFuture<'a, ProviderLinsolveResult> {
        unsupported_future("linsolve not supported by provider")
    }
    fn inv<'a>(
        &'a self,
        _matrix: &'a GpuTensorHandle,
        _options: ProviderInvOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("inv not supported by provider")
    }
    fn pinv<'a>(
        &'a self,
        _matrix: &'a GpuTensorHandle,
        _options: ProviderPinvOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("pinv not supported by provider")
    }
    fn cond<'a>(
        &'a self,
        _matrix: &'a GpuTensorHandle,
        _norm: ProviderCondNorm,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { Err(anyhow::anyhow!("cond not supported by provider")) })
    }
    fn norm<'a>(
        &'a self,
        _tensor: &'a GpuTensorHandle,
        _order: ProviderNormOrder,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { Err(anyhow::anyhow!("norm not supported by provider")) })
    }
    fn rank<'a>(
        &'a self,
        _matrix: &'a GpuTensorHandle,
        _tolerance: Option<f64>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { Err(anyhow::anyhow!("rank not supported by provider")) })
    }
    fn rcond<'a>(
        &'a self,
        _matrix: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { Err(anyhow::anyhow!("rcond not supported by provider")) })
    }
    fn mldivide<'a>(
        &'a self,
        _lhs: &'a GpuTensorHandle,
        _rhs: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { Err(anyhow::anyhow!("mldivide not supported by provider")) })
    }
    fn mrdivide<'a>(
        &'a self,
        _lhs: &'a GpuTensorHandle,
        _rhs: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { Err(anyhow::anyhow!("mrdivide not supported by provider")) })
    }
    fn eig<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _compute_left: bool,
    ) -> AccelProviderFuture<'a, ProviderEigResult> {
        Box::pin(async move { Err(anyhow::anyhow!("eig not supported by provider")) })
    }
    fn lu<'a>(&'a self, _a: &'a GpuTensorHandle) -> AccelProviderFuture<'a, ProviderLuResult> {
        Box::pin(async move { Err(anyhow::anyhow!("lu not supported by provider")) })
    }

    fn chol<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _lower: bool,
    ) -> AccelProviderFuture<'a, ProviderCholResult> {
        Box::pin(async move { Err(anyhow::anyhow!("chol not supported by provider")) })
    }
    fn qr<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _options: ProviderQrOptions,
    ) -> AccelProviderFuture<'a, ProviderQrResult> {
        Box::pin(async move { Err(anyhow::anyhow!("qr not supported by provider")) })
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
    ) -> AccelProviderFuture<'a, Option<ProviderQrPowerIterResult>> {
        let _ = (product, q_handle, options);
        Box::pin(async move { Ok(None) })
    }
    fn transpose(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("transpose not supported by provider"))
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
        _b: &'a GpuTensorHandle,
        _a: &'a GpuTensorHandle,
        _x: &'a GpuTensorHandle,
        _options: ProviderIirFilterOptions,
    ) -> AccelProviderFuture<'a, ProviderIirFilterResult> {
        Box::pin(async move { Err(anyhow::anyhow!("iir_filter not supported by provider")) })
    }
    /// Reorder tensor dimensions according to `order`, expressed as zero-based indices.
    fn permute(
        &self,
        _handle: &GpuTensorHandle,
        _order: &[usize],
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("permute not supported by provider"))
    }
    fn flip(&self, _handle: &GpuTensorHandle, _axes: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("flip not supported by provider"))
    }
    fn circshift(
        &self,
        _handle: &GpuTensorHandle,
        _shifts: &[isize],
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("circshift not supported by provider"))
    }
    fn diff_dim(
        &self,
        _handle: &GpuTensorHandle,
        _order: usize,
        _dim: usize,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("diff_dim not supported by provider"))
    }
    /// Perform an in-place FFT along a zero-based dimension, optionally padding/truncating to `len`.
    fn fft_dim<'a>(
        &'a self,
        _handle: &'a GpuTensorHandle,
        _len: Option<usize>,
        _dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("fft_dim not supported by provider")
    }
    fn ifft_dim<'a>(
        &'a self,
        _handle: &'a GpuTensorHandle,
        _len: Option<usize>,
        _dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("ifft_dim not supported by provider")
    }
    fn unique<'a>(
        &'a self,
        _handle: &'a GpuTensorHandle,
        _options: &'a UniqueOptions,
    ) -> AccelProviderFuture<'a, UniqueResult> {
        Box::pin(async move { Err(anyhow::anyhow!("unique not supported by provider")) })
    }
    fn union<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
        _options: &'a UnionOptions,
    ) -> AccelProviderFuture<'a, UnionResult> {
        Box::pin(async move { Err(anyhow::anyhow!("union not supported by provider")) })
    }
    fn setdiff<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
        _options: &'a SetdiffOptions,
    ) -> AccelProviderFuture<'a, SetdiffResult> {
        Box::pin(async move { Err(anyhow::anyhow!("setdiff not supported by provider")) })
    }
    fn ismember<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _b: &'a GpuTensorHandle,
        _options: &'a IsMemberOptions,
    ) -> AccelProviderFuture<'a, IsMemberResult> {
        Box::pin(async move { Err(anyhow::anyhow!("ismember not supported by provider")) })
    }
    fn reshape(
        &self,
        handle: &GpuTensorHandle,
        new_shape: &[usize],
    ) -> anyhow::Result<GpuTensorHandle> {
        let mut updated = handle.clone();
        updated.shape = new_shape.to_vec();
        Ok(updated)
    }
    /// Concatenate the provided tensors along the 1-based dimension `dim`.
    fn cat(&self, _dim: usize, _inputs: &[GpuTensorHandle]) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("cat not supported by provider"))
    }
    fn repmat(
        &self,
        _handle: &GpuTensorHandle,
        _reps: &[usize],
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("repmat not supported by provider"))
    }
    /// Compute the Kronecker product of two tensors, matching MATLAB semantics.
    fn kron(&self, _a: &GpuTensorHandle, _b: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("kron not supported by provider"))
    }
    fn reduce_sum<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("reduce_sum not supported by provider")
    }
    fn reduce_sum_dim<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("reduce_sum_dim not supported by provider")
    }
    fn dot<'a>(
        &'a self,
        _lhs: &'a GpuTensorHandle,
        _rhs: &'a GpuTensorHandle,
        _dim: Option<usize>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("dot not supported by provider")
    }
    fn reduce_nnz<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("reduce_nnz not supported by provider")
    }
    fn reduce_nnz_dim<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("reduce_nnz_dim not supported by provider")
    }
    fn reduce_prod<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("reduce_prod not supported by provider")
    }
    fn reduce_prod_dim<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("reduce_prod_dim not supported by provider")
    }
    fn reduce_mean<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("reduce_mean not supported by provider")
    }
    /// Reduce mean across multiple zero-based dimensions in one device pass.
    fn reduce_mean_nd<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dims_zero_based: &'a [usize],
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("reduce_mean_nd not supported by provider")
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
    fn reduce_mean_dim<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dim: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("reduce_mean_dim not supported by provider")
    }
    fn reduce_std<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _normalization: ProviderStdNormalization,
        _nan_mode: ProviderNanMode,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("reduce_std not supported by provider")
    }
    fn reduce_std_dim<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dim: usize,
        _normalization: ProviderStdNormalization,
        _nan_mode: ProviderNanMode,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("reduce_std_dim not supported by provider")
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
    fn reduce_min<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("reduce_min not supported by provider")
    }
    fn reduce_min_dim<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dim: usize,
    ) -> AccelProviderFuture<'a, ReduceDimResult> {
        unsupported_future("reduce_min_dim not supported by provider")
    }
    fn reduce_max<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        unsupported_future("reduce_max not supported by provider")
    }
    fn reduce_max_dim<'a>(
        &'a self,
        _a: &'a GpuTensorHandle,
        _dim: usize,
    ) -> AccelProviderFuture<'a, ReduceDimResult> {
        unsupported_future("reduce_max_dim not supported by provider")
    }
    fn cumsum_scan(
        &self,
        _input: &GpuTensorHandle,
        _dim: usize,
        _direction: ProviderScanDirection,
        _nan_mode: ProviderNanMode,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("cumsum_scan not supported by provider"))
    }
    fn cumprod_scan(
        &self,
        _input: &GpuTensorHandle,
        _dim: usize,
        _direction: ProviderScanDirection,
        _nan_mode: ProviderNanMode,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("cumprod_scan not supported by provider"))
    }
    fn cummin_scan(
        &self,
        _input: &GpuTensorHandle,
        _dim: usize,
        _direction: ProviderScanDirection,
        _nan_mode: ProviderNanMode,
    ) -> anyhow::Result<ProviderCumminResult> {
        Err(anyhow::anyhow!("cummin_scan not supported by provider"))
    }
    fn cummax_scan(
        &self,
        _input: &GpuTensorHandle,
        _dim: usize,
        _direction: ProviderScanDirection,
        _nan_mode: ProviderNanMode,
    ) -> anyhow::Result<ProviderCummaxResult> {
        Err(anyhow::anyhow!("cummax_scan not supported by provider"))
    }

    fn find(
        &self,
        _a: &GpuTensorHandle,
        _limit: Option<usize>,
        _direction: FindDirection,
    ) -> anyhow::Result<ProviderFindResult> {
        Err(anyhow::anyhow!("find not supported by provider"))
    }

    fn fused_elementwise(
        &self,
        _shader: &str,
        _inputs: &[GpuTensorHandle],
        _output_shape: &[usize],
        _len: usize,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!(
            "fused_elementwise not supported by provider"
        ))
    }

    /// Build a numeric tensor where NaNs in `a` are replaced with 0.0 (device side).
    fn map_nan_to_zero(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("map_nan_to_zero not supported by provider"))
    }

    /// Build a numeric mask tensor with 1.0 where value is not NaN and 0.0 where value is NaN.
    fn not_nan_mask(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("not_nan_mask not supported by provider"))
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
        _inputs: &[GpuTensorHandle],
        _output_shape: &[usize],
        _reduce_len: usize,
        _num_slices: usize,
        _workgroup_size: u32,
        _flavor: ReductionFlavor,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("fused_reduction not supported by provider"))
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
            upload_bytes: 0,
            download_bytes: 0,
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
    fn scatter_column(
        &self,
        _matrix: &GpuTensorHandle,
        _col_index: usize,
        _values: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("scatter_column not supported by provider"))
    }

    /// Fast-path: write a GPU row in a matrix from a GPU vector, returning a new handle.
    /// Expected: `values.shape == [1, cols]` (or `[cols]`) and `row_index < rows`.
    fn scatter_row(
        &self,
        _matrix: &GpuTensorHandle,
        _row_index: usize,
        _values: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("scatter_row not supported by provider"))
    }

    fn sub2ind(
        &self,
        _dims: &[usize],
        _strides: &[usize],
        _inputs: &[&GpuTensorHandle],
        _scalar_mask: &[bool],
        _len: usize,
        _output_shape: &[usize],
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("sub2ind not supported by provider"))
    }

    /// Returns true if the provider offers a device-side `ind2sub` implementation.
    fn supports_ind2sub(&self) -> bool {
        false
    }

    /// Convert linear indices into per-dimension subscripts on the device.
    fn ind2sub(
        &self,
        _dims: &[usize],
        _strides: &[usize],
        _indices: &GpuTensorHandle,
        _total: usize,
        _len: usize,
        _output_shape: &[usize],
    ) -> anyhow::Result<Vec<GpuTensorHandle>> {
        Err(anyhow::anyhow!("ind2sub not supported by provider"))
    }

    /// Determine if a matrix is symmetric (or skew-symmetric) without gathering it to the host.
    fn issymmetric(
        &self,
        _matrix: &GpuTensorHandle,
        _kind: ProviderSymmetryKind,
        _tolerance: f64,
    ) -> anyhow::Result<bool> {
        Err(anyhow::anyhow!(
            "issymmetric predicate not supported by provider"
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
            Err(anyhow::anyhow!(
                "ishermitian predicate not supported by provider"
            ))
        })
    }

    /// Inspect the bandwidth of a matrix without gathering it back to the host.
    fn bandwidth(&self, _matrix: &GpuTensorHandle) -> anyhow::Result<ProviderBandwidth> {
        Err(anyhow::anyhow!("bandwidth not supported by provider"))
    }

    /// Compute the symmetric reverse Cuthill-McKee permutation for the matrix.
    ///
    /// Implementations may execute on the device or gather to the host. The permutation should be
    /// returned as zero-based indices.
    fn sym_rcm<'a>(&'a self, _matrix: &'a GpuTensorHandle) -> AccelProviderFuture<'a, Vec<usize>> {
        Box::pin(async move { Err(anyhow::anyhow!("sym_rcm not supported by provider")) })
    }
}

static GLOBAL_PROVIDER: Lazy<RwLock<Option<&'static dyn AccelProvider>>> =
    Lazy::new(|| RwLock::new(None));
static PROVIDER_REGISTRY: Lazy<RwLock<HashMap<u32, &'static dyn AccelProvider>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
static DEVICE_ID_COUNTER: AtomicU32 = AtomicU32::new(1);

#[cfg(not(target_arch = "wasm32"))]
thread_local! {
    static THREAD_PROVIDER: Cell<Option<&'static dyn AccelProvider>> = Cell::new(None);
}

#[cfg(target_arch = "wasm32")]
static WASM_THREAD_PROVIDER: Lazy<Mutex<Option<&'static dyn AccelProvider>>> =
    Lazy::new(|| Mutex::new(None));

#[cfg(not(target_arch = "wasm32"))]
fn replace_thread_provider(
    provider: Option<&'static dyn AccelProvider>,
) -> Option<&'static dyn AccelProvider> {
    THREAD_PROVIDER.with(|cell| {
        let prev = cell.get();
        cell.set(provider);
        prev
    })
}

#[cfg(target_arch = "wasm32")]
fn replace_thread_provider(
    provider: Option<&'static dyn AccelProvider>,
) -> Option<&'static dyn AccelProvider> {
    let mut slot = WASM_THREAD_PROVIDER
        .lock()
        .expect("wasm provider mutex poisoned");
    let prev = *slot;
    *slot = provider;
    prev
}

#[cfg(not(target_arch = "wasm32"))]
fn current_thread_provider() -> Option<&'static dyn AccelProvider> {
    THREAD_PROVIDER.with(|cell| cell.get())
}

#[cfg(target_arch = "wasm32")]
fn current_thread_provider() -> Option<&'static dyn AccelProvider> {
    WASM_THREAD_PROVIDER
        .lock()
        .expect("wasm provider mutex poisoned")
        .as_ref()
        .copied()
}

/// Register a global acceleration provider.
///
/// # Safety
/// - The caller must guarantee that `p` is valid for the entire program lifetime
///   (e.g., a `'static` singleton), as the runtime stores a raw reference globally.
/// - Concurrent callers must ensure registration happens once or is properly
///   synchronized; this function does not enforce thread-safety for re-registration.
pub unsafe fn register_provider(p: &'static dyn AccelProvider) {
    if let Ok(mut guard) = GLOBAL_PROVIDER.write() {
        *guard = Some(p);
    }
    register_provider_for_device(p.device_id(), p);
}

unsafe fn register_provider_for_device(device_id: u32, provider: &'static dyn AccelProvider) {
    if let Ok(mut guard) = PROVIDER_REGISTRY.write() {
        guard.insert(device_id, provider);
    }
}

pub fn provider() -> Option<&'static dyn AccelProvider> {
    if let Some(p) = current_thread_provider() {
        return Some(p);
    }
    GLOBAL_PROVIDER
        .read()
        .ok()
        .and_then(|guard| guard.as_ref().copied())
}

/// Clear the globally registered provider. Intended for tests to ensure deterministic behaviour.
pub fn clear_provider() {
    if let Ok(mut guard) = GLOBAL_PROVIDER.write() {
        *guard = None;
    }
    if let Ok(mut map) = PROVIDER_REGISTRY.write() {
        map.clear();
    }
}

pub fn provider_for_device(device_id: u32) -> Option<&'static dyn AccelProvider> {
    PROVIDER_REGISTRY
        .read()
        .ok()
        .and_then(|guard| guard.get(&device_id).copied())
        .or_else(|| provider())
}

pub fn provider_for_handle(handle: &GpuTensorHandle) -> Option<&'static dyn AccelProvider> {
    provider_for_device(handle.device_id)
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
    if let Some(p) = provider() {
        if let Ok(h) = p.elem_add(a, b).await {
            return Some(h);
        }
    }
    None
}

/// Convenience: perform elementwise hypot via provider if possible; otherwise return None
pub async fn try_elem_hypot(a: &GpuTensorHandle, b: &GpuTensorHandle) -> Option<GpuTensorHandle> {
    if let Some(p) = provider() {
        if let Ok(h) = p.elem_hypot(a, b).await {
            return Some(h);
        }
    }
    None
}

/// Convenience: perform elementwise max via provider if possible; otherwise return None
pub async fn try_elem_max(a: &GpuTensorHandle, b: &GpuTensorHandle) -> Option<GpuTensorHandle> {
    if let Some(p) = provider() {
        if let Ok(h) = p.elem_max(a, b).await {
            return Some(h);
        }
    }
    None
}

/// Convenience: perform elementwise min via provider if possible; otherwise return None
pub async fn try_elem_min(a: &GpuTensorHandle, b: &GpuTensorHandle) -> Option<GpuTensorHandle> {
    if let Some(p) = provider() {
        if let Ok(h) = p.elem_min(a, b).await {
            return Some(h);
        }
    }
    None
}

/// Convenience: perform elementwise atan2 via provider if possible; otherwise return None
pub async fn try_elem_atan2(y: &GpuTensorHandle, x: &GpuTensorHandle) -> Option<GpuTensorHandle> {
    if let Some(p) = provider() {
        if let Ok(h) = p.elem_atan2(y, x).await {
            return Some(h);
        }
    }
    None
}

// Minimal host tensor views to avoid depending on runmat-builtins and cycles
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostTensorOwned {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

#[derive(Debug)]
pub struct HostTensorView<'a> {
    pub data: &'a [f64],
    pub shape: &'a [usize],
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

/// Descriptor for GEMM epilogues applied to `C = A * B` before storing to `C`.
///
/// Supported operations:
/// - Scale by `alpha` and add scalar `beta`.
/// - Multiply output by per-row and/or per-column scale vectors (broadcasted).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaleOp {
    Multiply,
    Divide,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatmulEpilogue {
    /// Scalar multiply applied to each output element.
    pub alpha: f64,
    /// Scalar add applied to each output element after scaling.
    pub beta: f64,
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
            alpha: 1.0,
            beta: 0.0,
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
        self.alpha == 1.0
            && self.beta == 0.0
            && self.row_scale.is_none()
            && self.col_scale.is_none()
            && self.clamp_min.is_none()
            && self.clamp_max.is_none()
            && self.pow_exponent.is_none()
            && self.diag_output.is_none()
    }
}

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
}
