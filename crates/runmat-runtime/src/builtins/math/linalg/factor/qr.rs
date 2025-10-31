//! MATLAB-compatible `qr` builtin with pivoted and economy forms.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::register_builtin_fusion_spec;
use crate::register_builtin_gpu_spec;
use num_complex::Complex64;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use super::lu::PivotMode;

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "qr"
category: "math/linalg/factor"
keywords: ["qr", "factorization", "decomposition", "householder", "pivoting"]
summary: "QR factorization with optional column pivoting and economy-size outputs."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f64"]
  broadcasting: "none"
  notes: "Bundled WGPU provider reuses the host QR and re-uploads outputs; other providers fall back to the same host path automatically."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::linalg::factor::qr::tests"
  integration: "builtins::math::linalg::factor::qr::tests::qr_three_outputs_reconstructs_input"
---

# What does the `qr` function do in MATLAB / RunMat?
`qr(A)` computes an orthogonal (or unitary)–upper-triangular factorization of a real or complex matrix `A`. It matches MATLAB’s dense semantics, supporting full-size, economy-size, and column-pivoted variants.

## How does the `qr` function behave in MATLAB / RunMat?
- Single output: `R = qr(A)` returns the upper-triangular (or upper-trapezoidal) factor. Column pivoting is applied implicitly for numerical stability.
- Two outputs: `[Q, R] = qr(A)` yields the orthonormal/unitary factor `Q` and the upper-triangular `R` such that `Q * R = A * E`, where `E` is the column permutation matrix selected during factorization. When you omit `E`, the columns of `R` already appear in that permuted order.
- Three outputs: `[Q, R, E] = qr(A)` additionally returns the column permutation so that `A * E = Q * R`. Use the `'vector'` option to receive a pivot vector instead of a permutation matrix.
- Economy size: `qr(A, 0)` or `qr(A, 'econ')` returns the reduced-size factors, mirroring MATLAB’s behaviour: when `m ≥ n`, `Q` is `m × n` and `R` is `n × n`. For wide matrices (`m < n`) the economy form equals the full form.
- Logical inputs are promoted to double precision. Scalars and vectors are treated as 1-D or 2-D dense arrays.

## GPU execution in RunMat
- When the active acceleration provider exposes the dedicated QR hook (the bundled WGPU backend does), RunMat stages the input through the provider, performs the factorisation via the runtime implementation, and re-uploads all outputs so they remain resident as `gpuTensor` handles.
- If no provider hook is present, RunMat gathers the input to the host, performs the CPU factorisation, and leaves the outputs on the host until an explicit `gpuArray` request.

## Examples of using the `qr` function in MATLAB / RunMat

### Computing the full QR factorization of a square matrix
```matlab
A = [12 -51 4; 6 167 -68; -4 24 -41];
[Q, R] = qr(A);
```
`Q` is orthogonal and `R` is upper-triangular. The product `Q * R` reconstructs `A`.

### Obtaining orthonormal Q and upper-triangular R for a tall matrix
```matlab
A = [1 2; 3 4; 5 6];
[Q, R] = qr(A);
```
`Q` is `3×3`, `R` is `3×2`, and `Q * R` equals `A`.

### Working with column pivoting when requesting three outputs
```matlab
A = [1 1 0; 1 0 1; 0 1 1];
[Q, R, E] = qr(A);
```
`E` is a permutation matrix that reorders the columns of `A` to maximise numerical stability.

### Using economy-size QR to reduce memory footprint
```matlab
A = randn(1000, 20);
[Q, R] = qr(A, 0);
```
Here `Q` is `1000×20` and `R` is `20×20`, reducing memory usage compared with the full factors.

### Receiving the permutation vector instead of a matrix
```matlab
A = magic(4);
[Q, R, p] = qr(A, 'vector');
```
`p` is a column vector of 1-based indices satisfying `A(:,p) = Q*R`.

### Running QR on a gpuArray input (with automatic fallback)
```matlab
G = gpuArray(rand(256, 64));
[Q, R] = qr(G, 'econ');
class(Q)
```
The default WGPU provider keeps the operands on the device by staging through its upload path, while other providers fall back to the CPU implementation and re-upload results automatically when possible.

## FAQ

### Does `qr(A)` always perform column pivoting?
Yes. MATLAB switched to pivoted QR for the single-output form many releases ago; RunMat mirrors that behaviour for improved numerical stability.

### How do I request the reduced (economy) factors?
Pass `0`, `'0'`, or `'econ'` as the second argument. For example, `[Q,R] = qr(A,0)` or `[Q,R] = qr(A,'econ')`.

### How can I obtain the permutation vector instead of a matrix?
Add the option `'vector'`, e.g., `[Q,R,p] = qr(A,'vector')`. When you also request economy size, use `[Q,R,p] = qr(A,'econ','vector')`.

### Are complex matrices supported?
Yes. Inputs of type complex double produce complex orthonormal factors using complex Householder reflectors.

### What precision do the returned factors use?
The builtin always produces double-precision (or complex double) outputs, matching MATLAB’s dense QR behaviour.

### Can I call `qr` on logical arrays?
Yes. Logical inputs are converted to double precision before factorisation.

### What happens if I pass more than two option arguments?
The builtin raises a MATLAB-compatible error (`qr: too many option arguments`). Only one size option and one permutation option are accepted.

### Do gpuArray outputs stay on the device?
Yes when an acceleration provider exposes the QR hook; the bundled WGPU backend does so by staging through the host implementation and re-uploading the factors. Otherwise RunMat gathers the data to the host, factors it there, and uploads the results back automatically.

### How can I verify the factorisation?
Check that `Q'*Q` is (approximately) the identity matrix and that `Q*R` equals `A*E`. For vector permutations, `A(:,p)` reproduces the pivoted columns.

## See Also
[lu](./lu), [chol](./chol), [svd](../structure/svd), [det](../../det), [gpuArray](../../../acceleration/gpu/gpuArray), [gather](../../../acceleration/gpu/gather)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/math/linalg/factor/qr.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/factor/qr.rs)
- Issues & feedback: [RunMat issue tracker](https://github.com/runmat-org/runmat/issues/new/choose)
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "qr",
    op_kind: GpuOpKind::Custom("qr-factor"),
    supported_precisions: &[ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("qr")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may download to host and re-upload results; the bundled WGPU backend currently uses the runtime QR implementation.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "qr",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "QR factorisation executes eagerly and does not participate in fusion.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("qr", DOC_MD);

#[runtime_builtin(
    name = "qr",
    category = "math/linalg/factor",
    summary = "QR factorization with optional column pivoting and economy-size outputs.",
    keywords = "qr,factorization,decomposition,householder",
    accel = "sink",
    sink = true
)]
fn qr_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let eval = evaluate(value, &rest)?;
    Ok(eval.r())
}

/// Output envelope for QR factorisation, used by the builtin and the VM multi-output path.
#[derive(Clone)]
pub struct QrEval {
    q: Value,
    r: Value,
    perm_matrix: Value,
    perm_vector: Value,
    mode: QrMode,
    pivot_mode: PivotMode,
}

impl QrEval {
    /// Orthogonal/unitary factor `Q`.
    pub fn q(&self) -> Value {
        self.q.clone()
    }

    /// Upper-triangular (or trapezoidal) factor `R`.
    pub fn r(&self) -> Value {
        self.r.clone()
    }

    /// Permutation output respecting the requested pivot mode.
    pub fn permutation(&self) -> Value {
        match self.pivot_mode {
            PivotMode::Matrix => self.perm_matrix.clone(),
            PivotMode::Vector => self.perm_vector.clone(),
        }
    }

    /// Always-available permutation matrix.
    pub fn permutation_matrix(&self) -> Value {
        self.perm_matrix.clone()
    }

    /// Always-available permutation vector (1-based indices).
    pub fn permutation_vector(&self) -> Value {
        self.perm_vector.clone()
    }

    /// Selected pivot mode.
    pub fn pivot_mode(&self) -> PivotMode {
        self.pivot_mode
    }

    /// Selected size mode (full or economy).
    pub fn mode(&self) -> QrMode {
        self.mode
    }
}

/// Size mode for QR outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QrMode {
    Full,
    Economy,
}

impl Default for QrMode {
    fn default() -> Self {
        QrMode::Full
    }
}

#[derive(Clone, Copy, Debug)]
struct QrOptions {
    mode: QrMode,
    pivot: PivotMode,
}

impl Default for QrOptions {
    fn default() -> Self {
        Self {
            mode: QrMode::Full,
            pivot: PivotMode::Matrix,
        }
    }
}

/// Evaluate the builtin with full access to multiple outputs.
pub fn evaluate(value: Value, args: &[Value]) -> Result<QrEval, String> {
    let options = parse_options(args)?;
    match value {
        Value::GpuTensor(handle) => {
            if let Some(eval) = evaluate_gpu(&handle, &options)? {
                return Ok(eval);
            }
            let tensor = gpu_helpers::gather_tensor(&handle)?;
            let prefer_gpu = runmat_accelerate_api::provider().is_some();
            evaluate_host_value(Value::Tensor(tensor), options, prefer_gpu)
        }
        other => evaluate_host_value(other, options, false),
    }
}

fn evaluate_gpu(handle: &GpuTensorHandle, options: &QrOptions) -> Result<Option<QrEval>, String> {
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(None),
    };
    let provider_options = runmat_accelerate_api::ProviderQrOptions {
        economy: matches!(options.mode, QrMode::Economy),
        pivot: match options.pivot {
            PivotMode::Matrix => runmat_accelerate_api::ProviderQrPivot::Matrix,
            PivotMode::Vector => runmat_accelerate_api::ProviderQrPivot::Vector,
        },
    };
    match provider.qr(handle, provider_options) {
        Ok(result) => Ok(Some(QrEval {
            q: Value::GpuTensor(result.q),
            r: Value::GpuTensor(result.r),
            perm_matrix: Value::GpuTensor(result.perm_matrix),
            perm_vector: Value::GpuTensor(result.perm_vector),
            mode: options.mode,
            pivot_mode: options.pivot,
        })),
        Err(_) => Ok(None),
    }
}

fn evaluate_host_value(
    value: Value,
    options: QrOptions,
    prefer_gpu: bool,
) -> Result<QrEval, String> {
    let matrix = extract_matrix(value)?;
    let components = qr_factor(matrix)?;
    assemble_eval(components, options, prefer_gpu)
}

fn parse_options(args: &[Value]) -> Result<QrOptions, String> {
    if args.len() > 2 {
        return Err("qr: too many option arguments".to_string());
    }
    let mut opts = QrOptions::default();
    for arg in args {
        if is_zero_scalar(arg) {
            opts.mode = QrMode::Economy;
            continue;
        }
        if let Some(text) = tensor::value_to_string(arg) {
            let normalized = text.trim().to_ascii_lowercase();
            if normalized == "0" || normalized == "econ" || normalized == "economy" {
                opts.mode = QrMode::Economy;
                continue;
            }
            if normalized == "vector" {
                opts.pivot = PivotMode::Vector;
                continue;
            }
            if normalized == "matrix" {
                opts.pivot = PivotMode::Matrix;
                continue;
            }
            return Err(format!("qr: unknown option '{text}'"));
        }
        return Err(
            "qr: option must be numeric zero or a string ('econ', 'matrix', 'vector')".to_string(),
        );
    }
    Ok(opts)
}

fn is_zero_scalar(value: &Value) -> bool {
    match value {
        Value::Num(n) => n.abs() <= EPS_SCALAR,
        Value::Int(i) => i.to_i64() == 0,
        Value::Bool(b) => !b,
        _ => false,
    }
}

fn extract_matrix(value: Value) -> Result<ColMajorMatrix, String> {
    match value {
        Value::Tensor(t) => ColMajorMatrix::from_tensor(&t).map_err(|e| format!("qr: {e}")),
        Value::ComplexTensor(ct) => {
            ColMajorMatrix::from_complex_tensor(&ct).map_err(|e| format!("qr: {e}"))
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            ColMajorMatrix::from_tensor(&tensor).map_err(|e| format!("qr: {e}"))
        }
        Value::Num(n) => Ok(ColMajorMatrix::from_scalar(Complex64::new(n, 0.0))),
        Value::Int(i) => Ok(ColMajorMatrix::from_scalar(Complex64::new(i.to_f64(), 0.0))),
        Value::Bool(b) => Ok(ColMajorMatrix::from_scalar(Complex64::new(
            if b { 1.0 } else { 0.0 },
            0.0,
        ))),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor(&handle)?;
            ColMajorMatrix::from_tensor(&tensor).map_err(|e| format!("qr: {e}"))
        }
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) => {
            Err("qr: expected a numeric matrix".to_string())
        }
        other => Err(format!("qr: unsupported input type {other:?}")),
    }
}

fn assemble_eval(
    components: QrComponents,
    options: QrOptions,
    prefer_gpu: bool,
) -> Result<QrEval, String> {
    let rows = components.reflectors.rows;
    let cols = components.reflectors.cols;
    let mut q_full = build_q(&components.reflectors, &components.taus);
    let mut r_full = build_full_r(&components.reflectors);
    q_full.clean(EPS_CLEAN);
    r_full.clean(EPS_CLEAN);
    let perm_matrix = perm_matrix(cols, &components.permutation);
    let perm_vector = components.permutation.clone();

    let (q_value, r_value) = match options.mode {
        QrMode::Full => (
            matrix_to_value(&q_full, "qr", prefer_gpu)?,
            matrix_to_value(&r_full, "qr", prefer_gpu)?,
        ),
        QrMode::Economy => {
            let mut q_econ = if rows >= cols {
                q_full.take_columns(cols)
            } else {
                q_full
            };
            let mut r_econ = if rows >= cols {
                r_full.take_rows(cols)
            } else {
                r_full
            };
            q_econ.clean(EPS_CLEAN);
            r_econ.clean(EPS_CLEAN);
            (
                matrix_to_value(&q_econ, "qr", prefer_gpu)?,
                matrix_to_value(&r_econ, "qr", prefer_gpu)?,
            )
        }
    };

    let perm_matrix_value = matrix_to_value(&perm_matrix, "qr", prefer_gpu)?;
    let perm_vector_value = maybe_upload_value(pivot_vector_to_value(&perm_vector)?, prefer_gpu);

    Ok(QrEval {
        q: q_value,
        r: r_value,
        perm_matrix: perm_matrix_value,
        perm_vector: perm_vector_value,
        mode: options.mode,
        pivot_mode: options.pivot,
    })
}

fn pivot_vector_to_value(pivot: &[usize]) -> Result<Value, String> {
    let mut data = Vec::with_capacity(pivot.len());
    for &idx in pivot {
        data.push((idx + 1) as f64);
    }
    let tensor = Tensor::new(data, vec![pivot.len(), 1]).map_err(|e| format!("qr: {e}"))?;
    Ok(Value::Tensor(tensor))
}

fn perm_matrix(size: usize, perm: &[usize]) -> ColMajorMatrix {
    let mut matrix = ColMajorMatrix::zeros(size, size);
    for (col, &row_idx) in perm.iter().enumerate() {
        if row_idx < size {
            matrix.set(row_idx, col, Complex64::new(1.0, 0.0));
        }
    }
    matrix
}

fn matrix_to_value(
    matrix: &ColMajorMatrix,
    label: &str,
    prefer_gpu: bool,
) -> Result<Value, String> {
    let value = matrix.to_value(label)?;
    Ok(maybe_upload_value(value, prefer_gpu))
}

fn maybe_upload_value(value: Value, prefer_gpu: bool) -> Value {
    if !prefer_gpu {
        return value;
    }
    match value {
        Value::Tensor(tensor) => {
            if let Some(provider) = runmat_accelerate_api::provider() {
                let view = runmat_accelerate_api::HostTensorView {
                    data: &tensor.data,
                    shape: &tensor.shape,
                };
                if let Ok(handle) = provider.upload(&view) {
                    return Value::GpuTensor(handle);
                }
            }
            Value::Tensor(tensor)
        }
        other => other,
    }
}

struct QrComponents {
    reflectors: ColMajorMatrix,
    taus: Vec<Complex64>,
    permutation: Vec<usize>,
}

fn qr_factor(mut matrix: ColMajorMatrix) -> Result<QrComponents, String> {
    let rows = matrix.rows;
    let cols = matrix.cols;
    let min_dim = rows.min(cols);
    let mut taus = vec![Complex64::new(0.0, 0.0); min_dim];
    let mut permutation: Vec<usize> = (0..cols).collect();
    let mut col_norms: Vec<f64> = (0..cols)
        .map(|j| column_norm_sq_from(&matrix, j, 0))
        .collect();

    for k in 0..min_dim {
        let pivot = select_pivot(&col_norms, k);
        if pivot != k {
            matrix.swap_columns(k, pivot);
            col_norms.swap(k, pivot);
            permutation.swap(k, pivot);
        }

        let tau = {
            let column_slice = matrix.column_segment_mut(k, k);
            householder(column_slice)
        };
        taus[k] = tau;
        if tau.norm() != 0.0 {
            let v = householder_vector(&matrix, k);
            for j in (k + 1)..cols {
                apply_householder(&mut matrix, &v, tau, k, j);
            }
        }

        for j in (k + 1)..cols {
            col_norms[j] = column_norm_sq_from(&matrix, j, k + 1);
        }
    }

    Ok(QrComponents {
        reflectors: matrix,
        taus,
        permutation,
    })
}

fn select_pivot(col_norms: &[f64], start: usize) -> usize {
    let mut pivot = start;
    let mut max_norm = 0.0;
    for (offset, &norm) in col_norms.iter().enumerate().skip(start) {
        if norm > max_norm {
            max_norm = norm;
            pivot = offset;
        }
    }
    pivot
}

fn column_norm_sq_from(matrix: &ColMajorMatrix, col: usize, start_row: usize) -> f64 {
    let mut sum = 0.0;
    for row in start_row..matrix.rows {
        sum += matrix.get(row, col).norm_sqr();
    }
    sum
}

fn householder(column: &mut [Complex64]) -> Complex64 {
    if column.is_empty() {
        return Complex64::new(0.0, 0.0);
    }
    let alpha = column[0];
    let mut tail_norm_sq = 0.0;
    for z in column.iter().skip(1) {
        tail_norm_sq += z.norm_sqr();
    }

    if tail_norm_sq <= EPS_CLEAN && alpha.norm() <= EPS_CLEAN {
        column[0] = Complex64::new(0.0, 0.0);
        for z in column.iter_mut().skip(1) {
            *z = Complex64::new(0.0, 0.0);
        }
        return Complex64::new(0.0, 0.0);
    }

    if tail_norm_sq <= EPS_CLEAN && alpha.im.abs() <= EPS_CLEAN && alpha.re >= 0.0 {
        for z in column.iter_mut().skip(1) {
            *z = Complex64::new(0.0, 0.0);
        }
        return Complex64::new(0.0, 0.0);
    }

    let alpha_abs = alpha.norm();
    let total_norm = (alpha_abs * alpha_abs + tail_norm_sq).sqrt();
    let sign = if alpha_abs <= EPS_CLEAN {
        Complex64::new(1.0, 0.0)
    } else {
        alpha / Complex64::new(alpha_abs, 0.0)
    };
    let beta = -sign * Complex64::new(total_norm, 0.0);
    let tau = if beta.norm() <= EPS_CLEAN {
        Complex64::new(0.0, 0.0)
    } else {
        (beta - alpha) / beta
    };
    let denom = alpha - beta;
    if denom.norm() <= EPS_CLEAN {
        for z in column.iter_mut().skip(1) {
            *z = Complex64::new(0.0, 0.0);
        }
    } else {
        for z in column.iter_mut().skip(1) {
            *z /= denom;
        }
    }
    column[0] = beta;
    tau
}

fn householder_vector(matrix: &ColMajorMatrix, k: usize) -> Vec<Complex64> {
    let mut v = Vec::with_capacity(matrix.rows - k);
    v.push(Complex64::new(1.0, 0.0));
    for row in (k + 1)..matrix.rows {
        v.push(matrix.get(row, k));
    }
    v
}

fn apply_householder(
    matrix: &mut ColMajorMatrix,
    v: &[Complex64],
    tau: Complex64,
    k: usize,
    target_col: usize,
) {
    let mut dot = Complex64::new(0.0, 0.0);
    for (offset, vi) in v.iter().enumerate() {
        let row = k + offset;
        dot += vi.conj() * matrix.get(row, target_col);
    }
    dot *= tau;
    for (offset, vi) in v.iter().enumerate() {
        let row = k + offset;
        let updated = matrix.get(row, target_col) - vi * dot;
        matrix.set(row, target_col, updated);
    }
}

fn build_q(reflectors: &ColMajorMatrix, taus: &[Complex64]) -> ColMajorMatrix {
    let rows = reflectors.rows;
    let mut q = ColMajorMatrix::identity(rows);
    let min_dim = taus.len();
    for k in (0..min_dim).rev() {
        let tau = taus[k];
        if tau.norm() == 0.0 {
            continue;
        }
        let v = householder_vector(reflectors, k);
        for col in 0..rows {
            let mut dot = Complex64::new(0.0, 0.0);
            for (offset, vi) in v.iter().enumerate() {
                let row = k + offset;
                dot += vi.conj() * q.get(row, col);
            }
            dot *= tau;
            for (offset, vi) in v.iter().enumerate() {
                let row = k + offset;
                let updated = q.get(row, col) - vi * dot;
                q.set(row, col, updated);
            }
        }
    }
    q
}

fn build_full_r(reflectors: &ColMajorMatrix) -> ColMajorMatrix {
    let rows = reflectors.rows;
    let cols = reflectors.cols;
    let mut r = ColMajorMatrix::zeros(rows, cols);
    for col in 0..cols {
        for row in 0..rows {
            if row <= col {
                let mut val = reflectors.get(row, col);
                if val.norm() <= EPS_CLEAN {
                    val = Complex64::new(0.0, 0.0);
                } else if val.im.abs() <= EPS_CLEAN {
                    val = Complex64::new(val.re, 0.0);
                }
                r.set(row, col, val);
            }
        }
    }
    r
}

struct ColMajorMatrix {
    rows: usize,
    cols: usize,
    data: Vec<Complex64>,
}

impl ColMajorMatrix {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![Complex64::new(0.0, 0.0); rows.saturating_mul(cols)],
        }
    }

    fn identity(size: usize) -> Self {
        let mut matrix = Self::zeros(size, size);
        for i in 0..size {
            matrix.set(i, i, Complex64::new(1.0, 0.0));
        }
        matrix
    }

    fn from_scalar(value: Complex64) -> Self {
        Self {
            rows: 1,
            cols: 1,
            data: vec![value],
        }
    }

    fn from_tensor(tensor: &Tensor) -> Result<Self, String> {
        if tensor.shape.len() > 2 {
            return Err("input must be 2-D".to_string());
        }
        let rows = tensor.rows();
        let cols = tensor.cols();
        let mut data = vec![Complex64::new(0.0, 0.0); rows.saturating_mul(cols)];
        for col in 0..cols {
            for row in 0..rows {
                let idx = row + col * rows;
                data[idx] = Complex64::new(tensor.data[idx], 0.0);
            }
        }
        Ok(Self { rows, cols, data })
    }

    fn from_complex_tensor(tensor: &ComplexTensor) -> Result<Self, String> {
        if tensor.shape.len() > 2 {
            return Err("input must be 2-D".to_string());
        }
        let rows = tensor.rows;
        let cols = tensor.cols;
        let mut data = vec![Complex64::new(0.0, 0.0); rows.saturating_mul(cols)];
        for col in 0..cols {
            for row in 0..rows {
                let idx = row + col * rows;
                let (re, im) = tensor.data[idx];
                data[idx] = Complex64::new(re, im);
            }
        }
        Ok(Self { rows, cols, data })
    }

    fn get(&self, row: usize, col: usize) -> Complex64 {
        self.data[row + col * self.rows]
    }

    fn set(&mut self, row: usize, col: usize, value: Complex64) {
        let idx = row + col * self.rows;
        self.data[idx] = value;
    }

    fn swap_columns(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        for row in 0..self.rows {
            let idx_a = row + a * self.rows;
            let idx_b = row + b * self.rows;
            self.data.swap(idx_a, idx_b);
        }
    }

    fn column_segment_mut(&mut self, col: usize, start_row: usize) -> &mut [Complex64] {
        let offset = start_row + col * self.rows;
        let len = self.rows.saturating_sub(start_row);
        &mut self.data[offset..offset.saturating_add(len)]
    }

    fn to_value(&self, label: &str) -> Result<Value, String> {
        if self.data.iter().all(|z| z.im.abs() <= EPS_CLEAN) {
            let mut real_data = Vec::with_capacity(self.rows * self.cols);
            for col in 0..self.cols {
                for row in 0..self.rows {
                    real_data.push(self.get(row, col).re);
                }
            }
            let tensor = Tensor::new(real_data, vec![self.rows, self.cols])
                .map_err(|e| format!("{label}: {e}"))?;
            Ok(Value::Tensor(tensor))
        } else {
            let mut complex_data = Vec::with_capacity(self.rows * self.cols);
            for col in 0..self.cols {
                for row in 0..self.rows {
                    let val = self.get(row, col);
                    complex_data.push((val.re, val.im));
                }
            }
            let tensor = ComplexTensor::new(complex_data, vec![self.rows, self.cols])
                .map_err(|e| format!("{label}: {e}"))?;
            Ok(Value::ComplexTensor(tensor))
        }
    }

    fn take_columns(&self, count: usize) -> ColMajorMatrix {
        let cols = count.min(self.cols);
        let mut data = vec![Complex64::new(0.0, 0.0); self.rows * cols];
        for col in 0..cols {
            for row in 0..self.rows {
                data[row + col * self.rows] = self.get(row, col);
            }
        }
        ColMajorMatrix {
            rows: self.rows,
            cols,
            data,
        }
    }

    fn take_rows(&self, count: usize) -> ColMajorMatrix {
        let rows = count.min(self.rows);
        let mut data = vec![Complex64::new(0.0, 0.0); rows * self.cols];
        for col in 0..self.cols {
            for row in 0..rows {
                data[row + col * rows] = self.get(row, col);
            }
        }
        ColMajorMatrix {
            rows,
            cols: self.cols,
            data,
        }
    }

    fn clean(&mut self, tol: f64) {
        for val in &mut self.data {
            if val.norm() <= tol {
                *val = Complex64::new(0.0, 0.0);
            } else if val.im.abs() <= tol {
                *val = Complex64::new(val.re, 0.0);
            }
        }
    }
}

const EPS_SCALAR: f64 = 1.0e-12;
const EPS_CLEAN: f64 = 1.0e-12;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::Tensor as Matrix;

    fn tensor_from_value(value: Value) -> Matrix {
        match value {
            Value::Tensor(t) => t,
            other => panic!("expected dense tensor, got {other:?}"),
        }
    }

    fn tensor_close(a: &Matrix, b: &Matrix, tol: f64) {
        assert_eq!(a.shape, b.shape);
        for (lhs, rhs) in a.data.iter().zip(&b.data) {
            if (lhs - rhs).abs() > tol {
                panic!("tensor mismatch: lhs={lhs}, rhs={rhs}, tol={tol}");
            }
        }
    }

    #[test]
    fn qr_single_output_returns_upper_triangular() {
        let data = vec![1.0, 4.0, 2.0, 5.0];
        let a = Matrix::new(data.clone(), vec![2, 2]).unwrap();
        let r_value = qr_builtin(Value::Tensor(a.clone()), Vec::new()).expect("qr");
        let eval = evaluate(Value::Tensor(a), &[]).expect("evaluate");
        let r_eval = tensor_from_value(eval.r());
        let r_builtin = tensor_from_value(r_value);
        tensor_close(&r_eval, &r_builtin, 1e-10);
    }

    #[test]
    fn qr_three_outputs_reconstructs_input() {
        let data = vec![1.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        let a = Matrix::new(data.clone(), vec![3, 2]).unwrap();
        let eval = evaluate(Value::Tensor(a.clone()), &[]).expect("evaluate");
        let q = tensor_from_value(eval.q());
        let r = tensor_from_value(eval.r());
        let e = tensor_from_value(eval.permutation_matrix());
        let mut qtq_data = vec![0.0; q.cols() * q.cols()];
        for i in 0..q.cols() {
            for j in 0..q.cols() {
                let mut sum = 0.0;
                for k in 0..q.rows() {
                    sum += q.data[k + i * q.rows()] * q.data[k + j * q.rows()];
                }
                qtq_data[i + j * q.cols()] = sum;
            }
        }
        let qtq = Matrix::new(qtq_data, vec![q.cols(), q.cols()]).unwrap();
        let identity = Matrix::new(
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            vec![3, 3],
        )
        .unwrap();
        tensor_close(&qtq, &identity, 1e-10);

        let qr = crate::matrix::matrix_mul(&q, &r).expect("Q*R");
        let ae = crate::matrix::matrix_mul(&a, &e).expect("A*E");
        tensor_close(&qr, &ae, 1e-10);
    }

    #[test]
    fn qr_vector_option_returns_pivot_vector() {
        let data = vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let a = Matrix::new(data, vec![3, 2]).unwrap();
        let eval = evaluate(Value::Tensor(a), &[Value::from("vector")]).expect("evaluate");
        assert_eq!(eval.pivot_mode(), PivotMode::Vector);
        let vector = tensor_from_value(eval.permutation());
        assert_eq!(vector.shape, vec![2, 1]);
        assert!(vector.data.iter().all(|v| *v == 1.0 || *v == 2.0));
    }

    #[test]
    fn qr_economy_shapes_for_tall_matrix() {
        let data: Vec<f64> = (0..12).map(|i| (i + 1) as f64).collect();
        let a = Matrix::new(data, vec![4, 3]).unwrap();
        let eval = evaluate(Value::Tensor(a), &[Value::from(0.0)]).expect("evaluate econ");
        assert_eq!(eval.mode(), QrMode::Economy);
        let q = tensor_from_value(eval.q());
        let r = tensor_from_value(eval.r());
        assert_eq!(q.shape, vec![4, 3]);
        assert_eq!(r.shape, vec![3, 3]);
    }

    #[test]
    fn qr_economy_wide_matrix_matches_full() {
        let data: Vec<f64> = (0..12).map(|i| (i + 1) as f64).collect();
        let a = Matrix::new(data, vec![3, 4]).unwrap();
        let full = evaluate(Value::Tensor(a.clone()), &[]).expect("full");
        let econ = evaluate(Value::Tensor(a), &[Value::from("econ")]).expect("econ");
        let q_full = tensor_from_value(full.q());
        let r_full = tensor_from_value(full.r());
        let q_econ = tensor_from_value(econ.q());
        let r_econ = tensor_from_value(econ.r());
        tensor_close(&q_full, &q_econ, 1e-10);
        tensor_close(&r_full, &r_econ, 1e-10);
    }

    #[test]
    fn qr_gpu_provider_returns_gpu_results() {
        test_support::with_test_provider(|provider| {
            let data: Vec<f64> = vec![1.0, 4.0, 2.0, 5.0];
            let a = Matrix::new(data.clone(), vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &a.data,
                shape: &a.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu_eval = evaluate(Value::GpuTensor(handle), &[]).expect("evaluate gpu");
            assert!(matches!(gpu_eval.q(), Value::GpuTensor(_)));
            assert!(matches!(gpu_eval.r(), Value::GpuTensor(_)));
            assert!(matches!(gpu_eval.permutation_matrix(), Value::GpuTensor(_)));
            assert!(matches!(gpu_eval.permutation_vector(), Value::GpuTensor(_)));

            let gathered_q = test_support::gather(gpu_eval.q()).expect("gather Q");
            let gathered_r = test_support::gather(gpu_eval.r()).expect("gather R");
            let gathered_p = test_support::gather(gpu_eval.permutation_matrix())
                .expect("gather permutation matrix");
            let gathered_vec = test_support::gather(gpu_eval.permutation_vector())
                .expect("gather permutation vector");

            let host_eval = evaluate(Value::Tensor(a), &[]).expect("host eval");
            let host_q = tensor_from_value(host_eval.q());
            let host_r = tensor_from_value(host_eval.r());
            let host_p = tensor_from_value(host_eval.permutation_matrix());
            let host_vec = tensor_from_value(host_eval.permutation_vector());

            tensor_close(&gathered_q, &host_q, 1e-10);
            tensor_close(&gathered_r, &host_r, 1e-10);
            tensor_close(&gathered_p, &host_p, 1e-10);
            tensor_close(&gathered_vec, &host_vec, 1e-10);
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn qr_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("register wgpu provider");

        let tol = match runmat_accelerate_api::provider()
            .expect("provider")
            .precision()
        {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };

        let tensor = Matrix::new(vec![3.0, 0.0, 4.0, 4.0, 0.0, 5.0], vec![3, 2]).unwrap();
        let host_eval = evaluate(Value::Tensor(tensor.clone()), &[]).expect("host eval");
        let host_q = tensor_from_value(host_eval.q());
        let host_r = tensor_from_value(host_eval.r());
        let host_p = tensor_from_value(host_eval.permutation_matrix());

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("provider");
        let handle = provider.upload(&view).expect("upload");
        let gpu_eval = evaluate(Value::GpuTensor(handle), &[]).expect("gpu eval");

        let gpu_q = test_support::gather(gpu_eval.q()).expect("gather Q");
        let gpu_r = test_support::gather(gpu_eval.r()).expect("gather R");
        let gpu_p = test_support::gather(gpu_eval.permutation_matrix()).expect("gather P");
        let gpu_vec =
            test_support::gather(gpu_eval.permutation_vector()).expect("gather pivot vector");

        tensor_close(&gpu_q, &host_q, tol);
        tensor_close(&gpu_r, &host_r, tol);
        tensor_close(&gpu_p, &host_p, tol);
        let host_vec = tensor_from_value(host_eval.permutation_vector());
        tensor_close(&gpu_vec, &host_vec, tol);
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
