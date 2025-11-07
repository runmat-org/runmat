use anyhow::anyhow;
use once_cell::sync::{Lazy, OnceCell};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

type ResidencyClearFn = fn(&GpuTensorHandle);

static RESIDENCY_CLEAR: OnceCell<ResidencyClearFn> = OnceCell::new();

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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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

impl Default for ProviderLinsolveOptions {
    fn default() -> Self {
        Self {
            lower: false,
            upper: false,
            rectangular: false,
            transposed: false,
            conjugate: false,
            symmetric: false,
            posdef: false,
            rcond: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderLinsolveResult {
    pub solution: GpuTensorHandle,
    pub reciprocal_condition: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProviderPinvOptions {
    pub tolerance: Option<f64>,
}

impl Default for ProviderPinvOptions {
    fn default() -> Self {
        Self { tolerance: None }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProviderPolyvalMu {
    pub mean: f64,
    pub scale: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProviderPolyvalOptions {
    pub mu: Option<ProviderPolyvalMu>,
}

impl Default for ProviderPolyvalOptions {
    fn default() -> Self {
        Self { mu: None }
    }
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
}

/// Device/provider interface that backends implement and register into the runtime layer
pub trait AccelProvider: Send + Sync {
    fn upload(&self, host: &crate::HostTensorView) -> anyhow::Result<GpuTensorHandle>;
    fn download(&self, h: &GpuTensorHandle) -> anyhow::Result<crate::HostTensorOwned>;
    fn free(&self, h: &GpuTensorHandle) -> anyhow::Result<()>;
    fn device_info(&self) -> String;

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
    fn tril(&self, _matrix: &GpuTensorHandle, _offset: isize) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow!("tril not supported by provider"))
    }

    /// Apply an upper-triangular mask to the first two dimensions of a tensor.
    fn triu(&self, _matrix: &GpuTensorHandle, _offset: isize) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow!("triu not supported by provider"))
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
    fn polyfit(
        &self,
        _x: &GpuTensorHandle,
        _y: &GpuTensorHandle,
        _degree: usize,
        _weights: Option<&GpuTensorHandle>,
    ) -> anyhow::Result<ProviderPolyfitResult> {
        Err(anyhow::anyhow!("polyfit not supported by provider"))
    }

    /// Differentiate a polynomial represented as a vector of coefficients.
    fn polyder_single(&self, _polynomial: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("polyder_single not supported by provider"))
    }

    /// Apply the product rule to polynomials `p` and `q`.
    fn polyder_product(
        &self,
        _p: &GpuTensorHandle,
        _q: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("polyder_product not supported by provider"))
    }

    /// Apply the quotient rule to polynomials `u` and `v`.
    fn polyder_quotient(
        &self,
        _u: &GpuTensorHandle,
        _v: &GpuTensorHandle,
    ) -> anyhow::Result<ProviderPolyderQuotient> {
        Err(anyhow::anyhow!(
            "polyder_quotient not supported by provider"
        ))
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

    /// Set the provider RNG state to align with the host RNG.
    fn set_rng_state(&self, _state: u64) -> anyhow::Result<()> {
        Err(anyhow::anyhow!("set_rng_state not supported by provider"))
    }

    /// Generate a 2-D correlation kernel matching MATLAB's `fspecial` builtin.
    fn fspecial(&self, _request: &FspecialRequest) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("fspecial not supported by provider"))
    }

    /// Apply an N-D correlation/convolution with padding semantics matching MATLAB's `imfilter`.
    fn imfilter(
        &self,
        _image: &GpuTensorHandle,
        _kernel: &GpuTensorHandle,
        _options: &ImfilterOptions,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("imfilter not supported by provider"))
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
    fn covariance(
        &self,
        _matrix: &GpuTensorHandle,
        _second: Option<&GpuTensorHandle>,
        _weights: Option<&GpuTensorHandle>,
        _options: &CovarianceOptions,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("covariance not supported by provider"))
    }

    /// Compute a correlation coefficient matrix across the columns of `matrix`.
    fn corrcoef(
        &self,
        _matrix: &GpuTensorHandle,
        _options: &CorrcoefOptions,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("corrcoef not supported by provider"))
    }

    // Optional operator hooks (default to unsupported)
    fn linspace(&self, _start: f64, _stop: f64, _count: usize) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("linspace not supported by provider"))
    }
    fn elem_add(
        &self,
        _a: &GpuTensorHandle,
        _b: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("elem_add not supported by provider"))
    }
    fn elem_mul(
        &self,
        _a: &GpuTensorHandle,
        _b: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("elem_mul not supported by provider"))
    }
    fn elem_max(
        &self,
        _a: &GpuTensorHandle,
        _b: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("elem_max not supported by provider"))
    }
    fn elem_min(
        &self,
        _a: &GpuTensorHandle,
        _b: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("elem_min not supported by provider"))
    }
    fn elem_sub(
        &self,
        _a: &GpuTensorHandle,
        _b: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("elem_sub not supported by provider"))
    }
    fn elem_div(
        &self,
        _a: &GpuTensorHandle,
        _b: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("elem_div not supported by provider"))
    }
    fn elem_pow(
        &self,
        _a: &GpuTensorHandle,
        _b: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("elem_pow not supported by provider"))
    }

    fn elem_hypot(
        &self,
        _a: &GpuTensorHandle,
        _b: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("elem_hypot not supported by provider"))
    }
    fn elem_ge(
        &self,
        _a: &GpuTensorHandle,
        _b: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("elem_ge not supported by provider"))
    }
    fn elem_le(
        &self,
        _a: &GpuTensorHandle,
        _b: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("elem_le not supported by provider"))
    }
    fn elem_lt(
        &self,
        _a: &GpuTensorHandle,
        _b: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("elem_lt not supported by provider"))
    }
    fn elem_gt(
        &self,
        _a: &GpuTensorHandle,
        _b: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("elem_gt not supported by provider"))
    }
    fn elem_eq(
        &self,
        _a: &GpuTensorHandle,
        _b: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("elem_eq not supported by provider"))
    }
    fn elem_ne(
        &self,
        _a: &GpuTensorHandle,
        _b: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("elem_ne not supported by provider"))
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
    fn elem_atan2(
        &self,
        _y: &GpuTensorHandle,
        _x: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("elem_atan2 not supported by provider"))
    }
    // Unary elementwise operations (optional)
    fn unary_sin(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_sin not supported by provider"))
    }
    fn unary_gamma(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_gamma not supported by provider"))
    }
    fn unary_factorial(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_factorial not supported by provider"))
    }
    fn unary_asinh(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_asinh not supported by provider"))
    }
    fn unary_sinh(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_sinh not supported by provider"))
    }
    fn unary_cosh(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_cosh not supported by provider"))
    }
    fn unary_asin(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_asin not supported by provider"))
    }
    fn unary_acos(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_acos not supported by provider"))
    }
    fn unary_acosh(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_acosh not supported by provider"))
    }
    fn unary_tan(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_tan not supported by provider"))
    }
    fn unary_tanh(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_tanh not supported by provider"))
    }
    fn unary_atan(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_atan not supported by provider"))
    }
    fn unary_atanh(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_atanh not supported by provider"))
    }
    fn unary_ceil(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_ceil not supported by provider"))
    }
    fn unary_floor(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_floor not supported by provider"))
    }
    fn unary_round(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_round not supported by provider"))
    }
    fn unary_fix(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_fix not supported by provider"))
    }
    fn unary_cos(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_cos not supported by provider"))
    }
    fn unary_angle(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_angle not supported by provider"))
    }
    fn unary_imag(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_imag not supported by provider"))
    }
    fn unary_real(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_real not supported by provider"))
    }
    fn unary_conj(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_conj not supported by provider"))
    }
    fn unary_abs(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_abs not supported by provider"))
    }
    fn unary_sign(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_sign not supported by provider"))
    }
    fn unary_exp(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_exp not supported by provider"))
    }
    fn unary_expm1(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_expm1 not supported by provider"))
    }
    fn unary_log(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_log not supported by provider"))
    }
    fn unary_log2(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_log2 not supported by provider"))
    }
    fn unary_log10(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_log10 not supported by provider"))
    }
    fn unary_log1p(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_log1p not supported by provider"))
    }
    fn unary_sqrt(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_sqrt not supported by provider"))
    }
    fn unary_double(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_double not supported by provider"))
    }
    fn unary_single(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_single not supported by provider"))
    }
    fn unary_pow2(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_pow2 not supported by provider"))
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
    fn sort_dim(
        &self,
        _a: &GpuTensorHandle,
        _dim: usize,
        _order: SortOrder,
        _comparison: SortComparison,
    ) -> anyhow::Result<SortResult> {
        Err(anyhow::anyhow!("sort_dim not supported by provider"))
    }
    fn sort_rows(
        &self,
        _a: &GpuTensorHandle,
        _columns: &[SortRowsColumnSpec],
        _comparison: SortComparison,
    ) -> anyhow::Result<SortResult> {
        Err(anyhow::anyhow!("sort_rows not supported by provider"))
    }
    fn matmul(
        &self,
        _a: &GpuTensorHandle,
        _b: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("matmul not supported by provider"))
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
    fn matmul_epilogue(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        epilogue: &MatmulEpilogue,
    ) -> anyhow::Result<GpuTensorHandle> {
        if epilogue.is_noop() {
            return self.matmul(a, b);
        }
        Err(anyhow::anyhow!("matmul_epilogue not supported by provider"))
    }
    fn image_normalize(
        &self,
        _input: &GpuTensorHandle,
        _desc: &ImageNormalizeDescriptor,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!(
            "image_normalize fusion not supported by provider"
        ))
    }
    fn matmul_power_step(
        &self,
        _lhs: &GpuTensorHandle,
        _rhs: &GpuTensorHandle,
        _epilogue: &PowerStepEpilogue,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!(
            "matmul_power_step normalization not supported by provider"
        ))
    }
    fn linsolve(
        &self,
        _lhs: &GpuTensorHandle,
        _rhs: &GpuTensorHandle,
        _options: &ProviderLinsolveOptions,
    ) -> anyhow::Result<ProviderLinsolveResult> {
        Err(anyhow::anyhow!("linsolve not supported by provider"))
    }
    fn inv(
        &self,
        _matrix: &GpuTensorHandle,
        _options: ProviderInvOptions,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("inv not supported by provider"))
    }
    fn pinv(
        &self,
        _matrix: &GpuTensorHandle,
        _options: ProviderPinvOptions,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("pinv not supported by provider"))
    }
    fn cond(
        &self,
        _matrix: &GpuTensorHandle,
        _norm: ProviderCondNorm,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("cond not supported by provider"))
    }
    fn norm(
        &self,
        _tensor: &GpuTensorHandle,
        _order: ProviderNormOrder,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("norm not supported by provider"))
    }
    fn rank(
        &self,
        _matrix: &GpuTensorHandle,
        _tolerance: Option<f64>,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("rank not supported by provider"))
    }
    fn rcond(&self, _matrix: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("rcond not supported by provider"))
    }
    fn mldivide(
        &self,
        _lhs: &GpuTensorHandle,
        _rhs: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("mldivide not supported by provider"))
    }
    fn mrdivide(
        &self,
        _lhs: &GpuTensorHandle,
        _rhs: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("mrdivide not supported by provider"))
    }
    fn eig(&self, _a: &GpuTensorHandle, _compute_left: bool) -> anyhow::Result<ProviderEigResult> {
        Err(anyhow::anyhow!("eig not supported by provider"))
    }
    fn lu(&self, _a: &GpuTensorHandle) -> anyhow::Result<ProviderLuResult> {
        Err(anyhow::anyhow!("lu not supported by provider"))
    }

    fn chol(&self, _a: &GpuTensorHandle, _lower: bool) -> anyhow::Result<ProviderCholResult> {
        Err(anyhow::anyhow!("chol not supported by provider"))
    }
    fn qr(
        &self,
        _a: &GpuTensorHandle,
        _options: ProviderQrOptions,
    ) -> anyhow::Result<ProviderQrResult> {
        Err(anyhow::anyhow!("qr not supported by provider"))
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
    fn iir_filter(
        &self,
        _b: &GpuTensorHandle,
        _a: &GpuTensorHandle,
        _x: &GpuTensorHandle,
        _options: ProviderIirFilterOptions,
    ) -> anyhow::Result<ProviderIirFilterResult> {
        Err(anyhow::anyhow!("iir_filter not supported by provider"))
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
    fn fft_dim(
        &self,
        _handle: &GpuTensorHandle,
        _len: Option<usize>,
        _dim: usize,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("fft_dim not supported by provider"))
    }
    fn ifft_dim(
        &self,
        _handle: &GpuTensorHandle,
        _len: Option<usize>,
        _dim: usize,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("ifft_dim not supported by provider"))
    }
    fn unique(
        &self,
        _handle: &GpuTensorHandle,
        _options: &UniqueOptions,
    ) -> anyhow::Result<UniqueResult> {
        Err(anyhow::anyhow!("unique not supported by provider"))
    }
    fn union(
        &self,
        _a: &GpuTensorHandle,
        _b: &GpuTensorHandle,
        _options: &UnionOptions,
    ) -> anyhow::Result<UnionResult> {
        Err(anyhow::anyhow!("union not supported by provider"))
    }
    fn setdiff(
        &self,
        _a: &GpuTensorHandle,
        _b: &GpuTensorHandle,
        _options: &SetdiffOptions,
    ) -> anyhow::Result<SetdiffResult> {
        Err(anyhow::anyhow!("setdiff not supported by provider"))
    }
    fn ismember(
        &self,
        _a: &GpuTensorHandle,
        _b: &GpuTensorHandle,
        _options: &IsMemberOptions,
    ) -> anyhow::Result<IsMemberResult> {
        Err(anyhow::anyhow!("ismember not supported by provider"))
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
    fn reduce_sum(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("reduce_sum not supported by provider"))
    }
    fn reduce_sum_dim(&self, _a: &GpuTensorHandle, _dim: usize) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("reduce_sum_dim not supported by provider"))
    }
    fn dot(
        &self,
        _lhs: &GpuTensorHandle,
        _rhs: &GpuTensorHandle,
        _dim: Option<usize>,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("dot not supported by provider"))
    }
    fn reduce_nnz(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("reduce_nnz not supported by provider"))
    }
    fn reduce_nnz_dim(&self, _a: &GpuTensorHandle, _dim: usize) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("reduce_nnz_dim not supported by provider"))
    }
    fn reduce_prod(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("reduce_prod not supported by provider"))
    }
    fn reduce_prod_dim(
        &self,
        _a: &GpuTensorHandle,
        _dim: usize,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("reduce_prod_dim not supported by provider"))
    }
    fn reduce_mean(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("reduce_mean not supported by provider"))
    }
    /// Reduce mean across multiple zero-based dimensions in one device pass.
    fn reduce_mean_nd(
        &self,
        _a: &GpuTensorHandle,
        _dims_zero_based: &[usize],
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("reduce_mean_nd not supported by provider"))
    }
    /// Reduce moments across multiple zero-based dimensions in one device pass.
    /// Returns mean (E[x]) and mean of squares (E[x^2]).
    fn reduce_moments_nd(
        &self,
        _a: &GpuTensorHandle,
        _dims_zero_based: &[usize],
    ) -> anyhow::Result<ProviderMoments2> {
        Err(anyhow::anyhow!(
            "reduce_moments_nd not supported by provider"
        ))
    }
    fn reduce_mean_dim(
        &self,
        _a: &GpuTensorHandle,
        _dim: usize,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("reduce_mean_dim not supported by provider"))
    }
    fn reduce_std(
        &self,
        _a: &GpuTensorHandle,
        _normalization: ProviderStdNormalization,
        _nan_mode: ProviderNanMode,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("reduce_std not supported by provider"))
    }
    fn reduce_std_dim(
        &self,
        _a: &GpuTensorHandle,
        _dim: usize,
        _normalization: ProviderStdNormalization,
        _nan_mode: ProviderNanMode,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("reduce_std_dim not supported by provider"))
    }
    fn reduce_any(&self, _a: &GpuTensorHandle, _omit_nan: bool) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("reduce_any not supported by provider"))
    }
    fn reduce_any_dim(
        &self,
        _a: &GpuTensorHandle,
        _dim: usize,
        _omit_nan: bool,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("reduce_any_dim not supported by provider"))
    }
    fn reduce_all(&self, _a: &GpuTensorHandle, _omit_nan: bool) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("reduce_all not supported by provider"))
    }
    fn reduce_all_dim(
        &self,
        _a: &GpuTensorHandle,
        _dim: usize,
        _omit_nan: bool,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("reduce_all_dim not supported by provider"))
    }
    fn reduce_median(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("reduce_median not supported by provider"))
    }
    fn reduce_median_dim(
        &self,
        _a: &GpuTensorHandle,
        _dim: usize,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!(
            "reduce_median_dim not supported by provider"
        ))
    }
    fn reduce_min(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("reduce_min not supported by provider"))
    }
    fn reduce_min_dim(&self, _a: &GpuTensorHandle, _dim: usize) -> anyhow::Result<ReduceDimResult> {
        Err(anyhow::anyhow!("reduce_min_dim not supported by provider"))
    }
    fn reduce_max(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("reduce_max not supported by provider"))
    }
    fn reduce_max_dim(&self, _a: &GpuTensorHandle, _dim: usize) -> anyhow::Result<ReduceDimResult> {
        Err(anyhow::anyhow!("reduce_max_dim not supported by provider"))
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
    fn fused_reduction(
        &self,
        _shader: &str,
        _inputs: &[GpuTensorHandle],
        _output_shape: &[usize],
        _reduce_len: usize,
        _num_slices: usize,
        _workgroup_size: u32,
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
    fn ishermitian(
        &self,
        _matrix: &GpuTensorHandle,
        _kind: ProviderHermitianKind,
        _tolerance: f64,
    ) -> anyhow::Result<bool> {
        Err(anyhow::anyhow!(
            "ishermitian predicate not supported by provider"
        ))
    }

    /// Inspect the bandwidth of a matrix without gathering it back to the host.
    fn bandwidth(&self, _matrix: &GpuTensorHandle) -> anyhow::Result<ProviderBandwidth> {
        Err(anyhow::anyhow!("bandwidth not supported by provider"))
    }

    /// Compute the symmetric reverse Cuthill-McKee permutation for the matrix.
    ///
    /// Implementations may execute on the device or gather to the host. The permutation should be
    /// returned as zero-based indices.
    fn sym_rcm(&self, _matrix: &GpuTensorHandle) -> anyhow::Result<Vec<usize>> {
        Err(anyhow::anyhow!("sym_rcm not supported by provider"))
    }
}

static GLOBAL_PROVIDER: Lazy<RwLock<Option<&'static dyn AccelProvider>>> =
    Lazy::new(|| RwLock::new(None));

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
}

pub fn provider() -> Option<&'static dyn AccelProvider> {
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
}

/// Convenience: perform elementwise add via provider if possible; otherwise return None
pub fn try_elem_add(a: &GpuTensorHandle, b: &GpuTensorHandle) -> Option<GpuTensorHandle> {
    if let Some(p) = provider() {
        if let Ok(h) = p.elem_add(a, b) {
            return Some(h);
        }
    }
    None
}

/// Convenience: perform elementwise hypot via provider if possible; otherwise return None
pub fn try_elem_hypot(a: &GpuTensorHandle, b: &GpuTensorHandle) -> Option<GpuTensorHandle> {
    if let Some(p) = provider() {
        if let Ok(h) = p.elem_hypot(a, b) {
            return Some(h);
        }
    }
    None
}

/// Convenience: perform elementwise max via provider if possible; otherwise return None
pub fn try_elem_max(a: &GpuTensorHandle, b: &GpuTensorHandle) -> Option<GpuTensorHandle> {
    if let Some(p) = provider() {
        if let Ok(h) = p.elem_max(a, b) {
            return Some(h);
        }
    }
    None
}

/// Convenience: perform elementwise min via provider if possible; otherwise return None
pub fn try_elem_min(a: &GpuTensorHandle, b: &GpuTensorHandle) -> Option<GpuTensorHandle> {
    if let Some(p) = provider() {
        if let Ok(h) = p.elem_min(a, b) {
            return Some(h);
        }
    }
    None
}

/// Convenience: perform elementwise atan2 via provider if possible; otherwise return None
pub fn try_elem_atan2(y: &GpuTensorHandle, x: &GpuTensorHandle) -> Option<GpuTensorHandle> {
    if let Some(p) = provider() {
        if let Ok(h) = p.elem_atan2(y, x) {
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
