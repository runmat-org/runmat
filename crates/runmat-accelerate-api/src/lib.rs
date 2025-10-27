use anyhow::anyhow;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};

type ResidencyClearFn = fn(&GpuTensorHandle);

static RESIDENCY_CLEAR: OnceCell<ResidencyClearFn> = OnceCell::new();

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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderInvOptions {}

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
    fn elem_hypot(
        &self,
        _a: &GpuTensorHandle,
        _b: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("elem_hypot not supported by provider"))
    }
    // Unary elementwise operations (optional)
    fn unary_sin(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_sin not supported by provider"))
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

static mut GLOBAL_PROVIDER: Option<&'static dyn AccelProvider> = None;

/// Register a global acceleration provider.
///
/// # Safety
/// - The caller must guarantee that `p` is valid for the entire program lifetime
///   (e.g., a `'static` singleton), as the runtime stores a raw reference globally.
/// - Concurrent callers must ensure registration happens once or is properly
///   synchronized; this function does not enforce thread-safety for re-registration.
pub unsafe fn register_provider(p: &'static dyn AccelProvider) {
    GLOBAL_PROVIDER = Some(p);
}

pub fn provider() -> Option<&'static dyn AccelProvider> {
    unsafe { GLOBAL_PROVIDER }
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
