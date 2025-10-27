use anyhow::anyhow;
use once_cell::sync::{Lazy, OnceCell};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::RwLock;

type ResidencyClearFn = fn(&GpuTensorHandle);

static RESIDENCY_CLEAR: OnceCell<ResidencyClearFn> = OnceCell::new();

static LOGICAL_HANDLES: Lazy<RwLock<HashSet<u64>>> = Lazy::new(|| RwLock::new(HashSet::new()));

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

/// Annotate a GPU tensor handle as logically-typed (`logical` in MATLAB terms)
/// or clear the logical flag when `logical` is `false`.
pub fn set_handle_logical(handle: &GpuTensorHandle, logical: bool) {
    if let Ok(mut guard) = LOGICAL_HANDLES.write() {
        if logical {
            guard.insert(handle.buffer_id);
        } else {
            guard.remove(&handle.buffer_id);
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
    // Unary elementwise operations (optional)
    fn unary_sin(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_sin not supported by provider"))
    }
    fn unary_cos(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_cos not supported by provider"))
    }
    fn unary_abs(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_abs not supported by provider"))
    }
    fn unary_exp(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_exp not supported by provider"))
    }
    fn unary_log(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_log not supported by provider"))
    }
    fn unary_sqrt(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("unary_sqrt not supported by provider"))
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
    fn transpose(&self, _a: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("transpose not supported by provider"))
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
