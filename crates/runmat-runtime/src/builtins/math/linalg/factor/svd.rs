//! MATLAB-compatible `svd` builtin with economy and vector output options.
//!
//! The implementation mirrors MATLAB semantics for dense matrices, handling the
//! single-output singular value vector as well as the `[U,S,V]` multi-output
//! form. Economy-size reductions and the `"vector"` selector are supported, and
//! complex inputs produce unitary factors using conjugate transposes where
//! appropriate. GPU inputs are gathered to the host unless a specialised
//! provider hook is implemented; documentation below details the fallback plan.

use std::cmp::Ordering;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "svd",
        wasm_path = "crate::builtins::math::linalg::factor::svd"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "svd"
category: "math/linalg/factor"
keywords: ["svd", "singular value decomposition", "linalg", "economy", "vector", "gpu"]
summary: "Singular value decomposition with full, economy, and vectorised forms."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f64"]
  broadcasting: "none"
  notes: "No provider hook today; gpuArray inputs gather to host, execute the CPU SVD, and return host tensors."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::linalg::factor::svd::tests"
  integration: "builtins::math::linalg::factor::svd::tests::svd_three_outputs_reconstruct"
---

# What does the `svd` function do in MATLAB / RunMat?
`svd(A)` computes the singular value decomposition of a real or complex matrix `A`. It factors
`A` into the product `U * S * V'`, where `U` and `V` are orthogonal/unitary matrices and `S` is
diagonal (or rectangular diagonal) with non-negative, descending singular values.

## How does the `svd` function behave in MATLAB / RunMat?
- Single output `s = svd(A)` returns the singular values as a column vector sorted in descending order.
- Three outputs `[U,S,V] = svd(A)` return the full-sized factors with `U` square `m×m`, `S` shaped
  `m×n`, and `V` square `n×n` (`m = size(A,1)`, `n = size(A,2)`).
- Economy form `[U,S,V] = svd(A,'econ')` (or `svd(A,0)`) reduces the shapes to the rank-defining
  dimension so that `U` and `V` drop the redundant orthogonal columns.
- Vector form `[U,s,V] = svd(A,'vector')` supplies the singular values as a vector instead of a
  diagonal matrix. You can combine `'vector'` with `'econ'`.
- Logical and integer inputs are promoted to double precision before factorisation.
- Complex inputs yield unitary `U` and `V` (conjugate-transpose preserves orthogonality) with real,
  non-negative singular values.
- Empty matrices, row/column vectors, and scalars are all supported and follow MATLAB’s shape conventions.

## `svd` Function GPU Execution Behaviour
- RunMat reserves a dedicated `svd` provider hook; once a backend implements it, the factors can stay
  on the device as `gpuTensor` handles without round-tripping through host memory.
- Today no provider ships that hook, so `gpuArray` inputs are gathered to the host, the CPU SVD
  executes, and the factors are returned as host tensors. You can re-establish residency with
  `gpuArray(s)` if you need to continue on the GPU.
- Because SVD is a residency sink, the fusion planner treats it as a barrier—preceding GPU tensors
  are gathered and subsequent ops run on the host unless you manually promote them again.

## Examples of using the `svd` function in MATLAB / RunMat

### Getting the singular values of a matrix
```matlab
A = [1 2 3; 4 5 6; 7 8 9];
s = svd(A);
```
`s` is a column vector containing the three singular values of `A`.

### Full SVD and reconstruction of a square matrix
```matlab
A = [3 1; 0 2];
[U,S,V] = svd(A);
A_recon = U * S * V';
```
`A_recon` matches `A` (up to numerical precision).

### Economy-size SVD for a tall matrix
```matlab
A = randn(6, 3);
[U,S,V] = svd(A, 'econ');
size(U) %  6 x 3
size(S) %  3 x 3
size(V) %  3 x 3
```
Only the required orthonormal columns are retained.

### Economy-size SVD for a wide matrix
```matlab
A = randn(3, 6);
[U,S,V] = svd(A, 'econ');
size(U) %  3 x 3
size(S) %  3 x 6
size(V) %  6 x 6
```
`S` keeps the original column dimension so that `U*S*V'` still reconstructs `A`.

### Requesting vector form of the singular values
```matlab
A = [10 0; 0 1];
[U,s,V] = svd(A, 'vector');
```
`s` is a 2×1 column vector `[10; 1]` that you can use without converting from a diagonal matrix.

### Computing the SVD of a complex matrix
```matlab
A = [1+2i, 2-1i; 0, 3i];
[U,S,V] = svd(A);
```
Both `U` and `V` are unitary complex matrices and `S` holds the real singular values on its diagonal.

### Running `svd` on a `gpuArray` (automatic host fallback today)
```matlab
G = gpuArray(randn(128, 64));
s = svd(G);           % Values are gathered to host transparently
```
When a future provider implements `svd`, these tensors will remain on the GPU without code changes.

## FAQ

### How are the singular values ordered?
They are returned in non-increasing order. MATLAB’s sign conventions are followed: values are
non-negative and appear on the diagonal of `S` (or inside the vector form).

### What is the difference between full and economy forms?
Full SVD returns square `U` and `V` (`m×m` and `n×n`). Economy SVD trims them to
`m×min(m,n)` and `n×min(m,n)`. The `S` factor keeps the same column dimension as the input; it is
`min(m,n)×min(m,n)` when `m ≥ n` and `min(m,n)×n` when `m < n`. Use economy when you do not
need the redundant orthogonal columns.

### What does the `"vector"` option change?
It affects the second output. With `"vector"`, `S` is returned as a column vector of singular
values, matching `svd(A)` in the single-output form. Without it, `S` is a diagonal matrix.

### Can I mix `'econ'` and `'vector'`?
Yes. Any order of the options is accepted (`svd(A,'vector','econ')` and `svd(A,'econ','vector')`
both work), and the returned dimensions mirror MATLAB’s behaviour.

### What happens with scalars or empty matrices?
`svd` of a scalar returns its absolute value. Empty matrices return empty factors with consistent
dimensions so that downstream code can continue to operate without special cases.

### Does RunMat require BLAS/LAPACK for `svd`?
No. The builtin is always available. When BLAS/LAPACK is enabled, the host implementation leverages
those libraries through `nalgebra` for performance; otherwise a pure-Rust algorithm is used under the hood.

### Will the results stay on the GPU?
Not yet. Presently the builtin gathers GPU operands to the host, runs the CPU factorisation, and
returns host tensors. The GPU spec already reserves a hook so providers can keep everything
device-resident once GPU kernels land.

## See Also
[eig](./eig), [qr](./qr), [lu](./lu), [chol](./chol), [gpuArray](../../../acceleration/gpu/gpuArray), [gather](../../../acceleration/gpu/gather)

## Source & Feedback
- Implementation: `crates/runmat-runtime/src/builtins/math/linalg/factor/svd.rs`
- Feedback: [RunMat issue tracker](https://github.com/runmat-org/runmat/issues/new/choose)
"#;

#[runmat_macros::register_gpu_spec(wasm_path = "crate::builtins::math::linalg::factor::svd")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "svd",
    op_kind: GpuOpKind::Custom("svd-factor"),
    supported_precisions: &[ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("svd")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "GPU inputs are gathered to the host until a provider implements the reserved `svd` hook.",
};

#[runmat_macros::register_fusion_spec(wasm_path = "crate::builtins::math::linalg::factor::svd")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "svd",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "SVD executes eagerly and does not participate in fusion planning.",
};

#[runtime_builtin(
    name = "svd",
    category = "math/linalg/factor",
    summary = "Singular value decomposition with MATLAB-compatible economy and vector options.",
    keywords = "svd,singular value decomposition,economy,vector",
    accel = "sink",
    sink = true,
    wasm_path = "crate::builtins::math::linalg::factor::svd"
)]
fn svd_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let eval = evaluate(value, &rest)?;
    Ok(eval.singular_values())
}

#[derive(Clone, Debug)]
pub struct SvdEval {
    singular_vector: Value,
    sigma_matrix: Value,
    sigma_value: Value,
    u: Value,
    v: Value,
}

impl SvdEval {
    pub fn singular_values(&self) -> Value {
        self.singular_vector.clone()
    }

    pub fn sigma(&self) -> Value {
        self.sigma_value.clone()
    }

    pub fn sigma_matrix(&self) -> Value {
        self.sigma_matrix.clone()
    }

    pub fn u(&self) -> Value {
        self.u.clone()
    }

    pub fn v(&self) -> Value {
        self.v.clone()
    }
}

#[derive(Clone, Copy)]
struct SvdOptions {
    econ: bool,
    sigma_format: SigmaFormat,
}

impl Default for SvdOptions {
    fn default() -> Self {
        Self {
            econ: false,
            sigma_format: SigmaFormat::Matrix,
        }
    }
}

#[derive(Clone, Copy)]
enum SigmaFormat {
    Matrix,
    Vector,
}

pub fn evaluate(value: Value, args: &[Value]) -> Result<SvdEval, String> {
    let options = parse_options(args)?;
    evaluate_value(value, options)
}

fn evaluate_value(value: Value, options: SvdOptions) -> Result<SvdEval, String> {
    match value {
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor(&handle)?;
            evaluate_tensor(Value::Tensor(tensor), options)
        }
        other => evaluate_tensor(other, options),
    }
}

fn evaluate_tensor(value: Value, options: SvdOptions) -> Result<SvdEval, String> {
    match value_to_numeric_matrix(value)? {
        NumericMatrix::Real(matrix) => compute_svd_real(matrix, &options),
        NumericMatrix::Complex(matrix) => compute_svd_complex(matrix, &options),
    }
}

enum NumericMatrix {
    Real(DMatrix<f64>),
    Complex(DMatrix<Complex64>),
}

fn value_to_numeric_matrix(value: Value) -> Result<NumericMatrix, String> {
    match value {
        Value::Tensor(tensor) => tensor_to_matrix(&tensor).map(NumericMatrix::Real),
        Value::ComplexTensor(ct) => complex_tensor_to_matrix(&ct).map(NumericMatrix::Complex),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            tensor_to_matrix(&tensor).map(NumericMatrix::Real)
        }
        Value::Num(n) => Ok(NumericMatrix::Real(DMatrix::from_element(1, 1, n))),
        Value::Int(i) => Ok(NumericMatrix::Real(DMatrix::from_element(1, 1, i.to_f64()))),
        Value::Bool(b) => Ok(NumericMatrix::Real(DMatrix::from_element(
            1,
            1,
            if b { 1.0 } else { 0.0 },
        ))),
        Value::Complex(re, im) => Ok(NumericMatrix::Complex(DMatrix::from_element(
            1,
            1,
            Complex64::new(re, im),
        ))),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor(&handle)?;
            tensor_to_matrix(&tensor).map(NumericMatrix::Real)
        }
        other => Err(format!(
            "svd: unsupported input type {other:?}; expected numeric or logical values"
        )),
    }
}

fn tensor_to_matrix(tensor: &Tensor) -> Result<DMatrix<f64>, String> {
    if tensor.shape.len() > 2 {
        return Err("svd: input must be 2-D".to_string());
    }
    let rows = tensor.rows();
    let cols = tensor.cols();
    Ok(DMatrix::from_column_slice(rows, cols, &tensor.data))
}

fn complex_tensor_to_matrix(tensor: &ComplexTensor) -> Result<DMatrix<Complex64>, String> {
    if tensor.shape.len() > 2 {
        return Err("svd: input must be 2-D".to_string());
    }
    let rows = tensor.rows;
    let cols = tensor.cols;
    let mut data = Vec::with_capacity(tensor.data.len());
    for &(re, im) in &tensor.data {
        data.push(Complex64::new(re, im));
    }
    Ok(DMatrix::from_column_slice(rows, cols, &data))
}

fn parse_options(args: &[Value]) -> Result<SvdOptions, String> {
    let mut opts = SvdOptions::default();
    if args.len() > 2 {
        return Err("svd: too many option arguments".to_string());
    }
    for arg in args {
        match arg {
            Value::Int(i) => {
                if i.to_i64() == 0 {
                    opts.econ = true;
                } else {
                    return Err("svd: numeric option must be 0 to request economy size".to_string());
                }
                continue;
            }
            Value::Num(n) => {
                if *n == 0.0 {
                    opts.econ = true;
                } else {
                    return Err("svd: numeric option must be 0 to request economy size".to_string());
                }
                continue;
            }
            _ => {}
        }
        if let Some(text) = tensor::value_to_string(arg) {
            match text.trim().to_ascii_lowercase().as_str() {
                "econ" | "economy" | "0" => {
                    opts.econ = true;
                }
                "full" => {
                    opts.econ = false;
                }
                "vector" => opts.sigma_format = SigmaFormat::Vector,
                "matrix" => opts.sigma_format = SigmaFormat::Matrix,
                other => {
                    return Err(format!("svd: unknown option '{other}'"));
                }
            }
            continue;
        }
        return Err(format!(
            "svd: expected option strings ('econ','vector','matrix') or the numeric value 0, got {arg:?}"
        ));
    }
    Ok(opts)
}

fn compute_svd_real(matrix: DMatrix<f64>, options: &SvdOptions) -> Result<SvdEval, String> {
    let (mut u, mut singular_values, mut v) = factorize(matrix)?;
    ensure_descending(&mut u, &mut singular_values, &mut v);
    let (u, v) = shape_factors_real(u, v, singular_values.len(), options.econ);
    assemble_eval_real(&u, &v, &singular_values, options)
}

fn compute_svd_complex(
    matrix: DMatrix<Complex64>,
    options: &SvdOptions,
) -> Result<SvdEval, String> {
    let (mut u, mut singular_values, mut v) = factorize(matrix)?;
    ensure_descending(&mut u, &mut singular_values, &mut v);
    let (u, v) = shape_factors_complex(u, v, singular_values.len(), options.econ);
    assemble_eval_complex(&u, &v, &singular_values, options)
}

type FactorizationResult<T> = (
    DMatrix<T>,
    DVector<<T as nalgebra::ComplexField>::RealField>,
    DMatrix<T>,
);

fn factorize<T: nalgebra::ComplexField>(
    matrix: DMatrix<T>,
) -> Result<FactorizationResult<T>, String> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    if rows == 0 || cols == 0 {
        let diag_len = rows.min(cols);
        let u = DMatrix::zeros(rows, diag_len);
        let singular_values = DVector::zeros(diag_len);
        let v = DMatrix::zeros(cols, diag_len);
        return Ok((u, singular_values, v));
    }
    let svd = nalgebra::linalg::SVD::new(matrix, true, true);
    let u = svd
        .u
        .ok_or_else(|| "svd: failed to compute left singular vectors".to_string())?;
    let v_t = svd
        .v_t
        .ok_or_else(|| "svd: failed to compute right singular vectors".to_string())?;
    let singular_values = svd.singular_values;
    let u = u.resize(rows, singular_values.len(), T::zero());
    let v = v_t.adjoint().resize(cols, singular_values.len(), T::zero());
    Ok((u, singular_values, v))
}

fn ensure_descending<T: nalgebra::ComplexField>(
    u: &mut DMatrix<T>,
    singular_values: &mut DVector<T::RealField>,
    v: &mut DMatrix<T>,
) {
    let len = singular_values.len();
    if len <= 1 {
        return;
    }
    let mut order: Vec<usize> = (0..len).collect();
    order.sort_by(|&a, &b| {
        let va = singular_values[a].clone();
        let vb = singular_values[b].clone();
        vb.partial_cmp(&va).unwrap_or(Ordering::Equal)
    });
    if order.iter().enumerate().all(|(idx, orig)| idx == *orig) {
        return;
    }

    let mut sorted = Vec::with_capacity(len);
    for &idx in &order {
        sorted.push(singular_values[idx].clone());
    }
    *singular_values = DVector::from_vec(sorted);

    let mut new_u = DMatrix::zeros(u.nrows(), u.ncols());
    for (col, &orig) in order.iter().enumerate() {
        let column = u.column(orig).into_owned();
        new_u.set_column(col, &column);
    }
    *u = new_u;

    let mut new_v = DMatrix::zeros(v.nrows(), v.ncols());
    for (col, &orig) in order.iter().enumerate() {
        let column = v.column(orig).into_owned();
        new_v.set_column(col, &column);
    }
    *v = new_v;
}

fn shape_factors_real(
    mut u: DMatrix<f64>,
    mut v: DMatrix<f64>,
    diag_len: usize,
    econ: bool,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let rows = u.nrows();
    let cols = v.nrows();
    if cols > diag_len {
        v = extend_orthonormal_real(&v, cols);
    }
    if econ {
        return (u, v);
    }
    if rows > diag_len {
        u = extend_orthonormal_real(&u, rows);
    }
    (u, v)
}

fn shape_factors_complex(
    mut u: DMatrix<Complex64>,
    mut v: DMatrix<Complex64>,
    diag_len: usize,
    econ: bool,
) -> (DMatrix<Complex64>, DMatrix<Complex64>) {
    let rows = u.nrows();
    let cols = v.nrows();
    if cols > diag_len {
        v = extend_orthonormal_complex(&v, cols);
    }
    if econ {
        return (u, v);
    }
    if rows > diag_len {
        u = extend_orthonormal_complex(&u, rows);
    }
    (u, v)
}

fn assemble_eval_real(
    u: &DMatrix<f64>,
    v: &DMatrix<f64>,
    singular_values: &DVector<f64>,
    options: &SvdOptions,
) -> Result<SvdEval, String> {
    let diag = singular_values.as_slice().to_vec();
    let (s_rows, s_cols) = sigma_shape(u.nrows(), v.nrows(), diag.len(), options.econ);
    let sigma_matrix = diag_matrix_value(&diag, s_rows, s_cols)?;
    let singular_vector = singular_vector_value(&diag)?;
    let sigma_value = match options.sigma_format {
        SigmaFormat::Matrix => sigma_matrix.clone(),
        SigmaFormat::Vector => singular_vector.clone(),
    };
    let u_value = matrix_real_to_value(u)?;
    let v_value = matrix_real_to_value(v)?;
    Ok(SvdEval {
        singular_vector,
        sigma_matrix,
        sigma_value,
        u: u_value,
        v: v_value,
    })
}

fn assemble_eval_complex(
    u: &DMatrix<Complex64>,
    v: &DMatrix<Complex64>,
    singular_values: &DVector<f64>,
    options: &SvdOptions,
) -> Result<SvdEval, String> {
    let diag = singular_values.as_slice().to_vec();
    let (s_rows, s_cols) = sigma_shape(u.nrows(), v.nrows(), diag.len(), options.econ);
    let sigma_matrix = diag_matrix_value(&diag, s_rows, s_cols)?;
    let singular_vector = singular_vector_value(&diag)?;
    let sigma_value = match options.sigma_format {
        SigmaFormat::Matrix => sigma_matrix.clone(),
        SigmaFormat::Vector => singular_vector.clone(),
    };
    let u_value = matrix_complex_to_value(u)?;
    let v_value = matrix_complex_to_value(v)?;
    Ok(SvdEval {
        singular_vector,
        sigma_matrix,
        sigma_value,
        u: u_value,
        v: v_value,
    })
}

fn sigma_shape(m: usize, n: usize, diag_len: usize, econ: bool) -> (usize, usize) {
    if econ {
        if m >= n {
            (diag_len, diag_len)
        } else {
            (diag_len, n)
        }
    } else {
        (m, n)
    }
}

fn singular_vector_value(values: &[f64]) -> Result<Value, String> {
    if values.is_empty() {
        let tensor = Tensor::new(Vec::new(), vec![0, 0]).map_err(|e| format!("svd: {e}"))?;
        return Ok(Value::Tensor(tensor));
    }
    let rows = values.len();
    let tensor = Tensor::new(values.to_vec(), vec![rows, 1]).map_err(|e| format!("svd: {e}"))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn diag_matrix_value(values: &[f64], rows: usize, cols: usize) -> Result<Value, String> {
    if rows == 0 || cols == 0 {
        let tensor = Tensor::new(Vec::new(), vec![rows, cols]).map_err(|e| format!("svd: {e}"))?;
        return Ok(Value::Tensor(tensor));
    }
    let mut data = vec![0.0; rows * cols];
    let diag_len = values.len();
    for i in 0..diag_len.min(rows.min(cols)) {
        data[i + i * rows] = values[i];
    }
    let tensor = Tensor::new(data, vec![rows, cols]).map_err(|e| format!("svd: {e}"))?;
    Ok(Value::Tensor(tensor))
}

fn matrix_real_to_value(matrix: &DMatrix<f64>) -> Result<Value, String> {
    let tensor = Tensor::new(
        matrix.as_slice().to_vec(),
        vec![matrix.nrows(), matrix.ncols()],
    )
    .map_err(|e| format!("svd: {e}"))?;
    Ok(Value::Tensor(tensor))
}

fn matrix_complex_to_value(matrix: &DMatrix<Complex64>) -> Result<Value, String> {
    let mut data = Vec::with_capacity(matrix.nrows() * matrix.ncols());
    for value in matrix.iter() {
        data.push((value.re, value.im));
    }
    let tensor = ComplexTensor::new(data, vec![matrix.nrows(), matrix.ncols()])
        .map_err(|e| format!("svd: {e}"))?;
    Ok(Value::ComplexTensor(tensor))
}

fn extend_orthonormal_real(matrix: &DMatrix<f64>, target_cols: usize) -> DMatrix<f64> {
    let rows = matrix.nrows();
    let existing = matrix.ncols();
    if existing >= target_cols {
        return matrix.clone();
    }
    if rows == 0 {
        return DMatrix::zeros(0, target_cols);
    }
    let mut full = DMatrix::<f64>::zeros(rows, target_cols);
    full.view_mut((0, 0), (rows, existing)).copy_from(matrix);
    let mut col = existing;
    let mut basis_index = 0usize;
    let mut attempts = 0usize;
    let max_attempts = rows * 6;
    let eps = 1e-10;
    while col < target_cols && attempts <= max_attempts {
        let mut vec = DVector::<f64>::zeros(rows);
        if basis_index < rows {
            vec[basis_index] = 1.0;
            basis_index += 1;
        } else {
            for r in 0..rows {
                let phase = ((r + attempts) % rows) as f64;
                vec[r] = (phase * 1.337).sin();
            }
        }
        for j in 0..col {
            let proj = full.column(j).dot(&vec);
            vec -= full.column(j) * proj;
        }
        let norm = vec.norm();
        if norm > eps {
            vec /= norm;
            full.set_column(col, &vec);
            col += 1;
        }
        attempts += 1;
    }
    if col < target_cols {
        for remaining in col..target_cols {
            let mut vec = DVector::<f64>::zeros(rows);
            vec[remaining % rows] = 1.0;
            for j in 0..remaining {
                let proj = full.column(j).dot(&vec);
                vec -= full.column(j) * proj;
            }
            let norm = vec.norm();
            if norm > eps {
                vec /= norm;
                full.set_column(remaining, &vec);
            }
        }
    }
    full
}

fn extend_orthonormal_complex(
    matrix: &DMatrix<Complex64>,
    target_cols: usize,
) -> DMatrix<Complex64> {
    let rows = matrix.nrows();
    let existing = matrix.ncols();
    if existing >= target_cols {
        return matrix.clone();
    }
    if rows == 0 {
        return DMatrix::zeros(0, target_cols);
    }
    let mut full = DMatrix::<Complex64>::zeros(rows, target_cols);
    full.view_mut((0, 0), (rows, existing)).copy_from(matrix);
    let mut col = existing;
    let mut basis_index = 0usize;
    let mut attempts = 0usize;
    let max_attempts = rows * 6;
    let eps = 1e-10;
    while col < target_cols && attempts <= max_attempts {
        let mut vec = DVector::<Complex64>::zeros(rows);
        if basis_index < rows {
            vec[basis_index] = Complex64::new(1.0, 0.0);
            basis_index += 1;
        } else {
            for r in 0..rows {
                let phase = ((r + attempts) % rows) as f64;
                vec[r] = Complex64::new((phase * 1.137).cos(), (phase * 0.911).sin());
            }
        }
        for j in 0..col {
            let proj = full.column(j).dot(&vec);
            vec -= full.column(j) * proj;
        }
        let norm = vec.norm();
        if norm > eps {
            vec /= norm.into();
            full.set_column(col, &vec);
            col += 1;
        }
        attempts += 1;
    }
    if col < target_cols {
        for remaining in col..target_cols {
            let mut vec = DVector::<Complex64>::zeros(rows);
            vec[remaining % rows] = Complex64::new(1.0, 0.0);
            for j in 0..remaining {
                let proj = full.column(j).dot(&vec);
                vec -= full.column(j) * proj;
            }
            let norm = vec.norm();
            if norm > eps {
                vec /= norm.into();
                full.set_column(remaining, &vec);
            }
        }
    }
    full
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use nalgebra::DMatrix;
    use runmat_builtins::{LogicalArray, Tensor};

    fn tensor_from_value(value: Value) -> Tensor {
        match value {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).expect("tensor"),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    fn dmatrix_from_value(value: Value) -> DMatrix<f64> {
        match value {
            Value::Tensor(t) => DMatrix::from_column_slice(t.rows(), t.cols(), &t.data),
            Value::Num(n) => DMatrix::from_element(1, 1, n),
            other => panic!("expected real tensor, got {other:?}"),
        }
    }

    fn matrix_close(lhs: &DMatrix<f64>, rhs: &DMatrix<f64>, tol: f64) {
        assert_eq!(lhs.shape(), rhs.shape(), "shape mismatch");
        for (a, b) in lhs.iter().zip(rhs.iter()) {
            assert!((a - b).abs() <= tol, "{a} vs {b} (tol {tol})");
        }
    }

    fn complex_matrix_from_value(value: Value) -> DMatrix<Complex64> {
        match value {
            Value::ComplexTensor(ct) => {
                let data: Vec<Complex64> = ct
                    .data
                    .iter()
                    .map(|(re, im)| Complex64::new(*re, *im))
                    .collect();
                DMatrix::from_column_slice(ct.rows, ct.cols, &data)
            }
            Value::Tensor(t) => {
                let data: Vec<Complex64> = t
                    .data
                    .iter()
                    .copied()
                    .map(|v| Complex64::new(v, 0.0))
                    .collect();
                DMatrix::from_column_slice(t.rows(), t.cols(), &data)
            }
            Value::Num(n) => DMatrix::from_element(1, 1, Complex64::new(n, 0.0)),
            Value::Complex(re, im) => DMatrix::from_element(1, 1, Complex64::new(re, im)),
            other => panic!("expected complex-compatible value, got {other:?}"),
        }
    }

    fn complex_matrix_close(lhs: &DMatrix<Complex64>, rhs: &DMatrix<Complex64>, tol: f64) {
        assert_eq!(lhs.shape(), rhs.shape(), "shape mismatch");
        for (a, b) in lhs.iter().zip(rhs.iter()) {
            let diff = (*a - *b).norm();
            assert!(diff <= tol, "{a} vs {b} (tol {tol})");
        }
    }

    #[test]
    fn svd_scalar_returns_absolute_value() {
        let result = svd_builtin(Value::Num(-3.0), Vec::new()).expect("svd");
        match result {
            Value::Num(n) => assert!((n - 3.0).abs() < 1e-12),
            other => panic!("expected scalar singular value, got {other:?}"),
        }
    }

    #[test]
    fn svd_three_outputs_reconstruct() {
        let matrix = Tensor::new(vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0], vec![3, 2]).expect("tensor");
        let eval = evaluate(Value::Tensor(matrix.clone()), &[]).expect("svd evaluate");

        let u = dmatrix_from_value(eval.u());
        let s = dmatrix_from_value(eval.sigma_matrix());
        let v = dmatrix_from_value(eval.v());

        let recon = &u * &s * v.transpose();
        let original = DMatrix::from_column_slice(matrix.rows(), matrix.cols(), &matrix.data);
        matrix_close(&recon, &original, 1e-10);
    }

    #[test]
    fn svd_empty_matrix_returns_empty_outputs() {
        let tensor = Tensor::new(Vec::new(), vec![0, 0]).expect("tensor");
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("svd empty");
        match eval.singular_values() {
            Value::Tensor(t) => assert_eq!(t.shape, vec![0, 0]),
            other => panic!("expected empty singular values, got {other:?}"),
        }
        match eval.sigma_matrix() {
            Value::Tensor(t) => assert_eq!(t.shape, vec![0, 0]),
            other => panic!("expected empty sigma matrix, got {other:?}"),
        }
        match eval.u() {
            Value::Tensor(t) => assert_eq!(t.shape, vec![0, 0]),
            Value::ComplexTensor(ct) => assert_eq!(ct.shape, vec![0, 0]),
            other => panic!("expected empty U, got {other:?}"),
        }
        match eval.v() {
            Value::Tensor(t) => assert_eq!(t.shape, vec![0, 0]),
            Value::ComplexTensor(ct) => assert_eq!(ct.shape, vec![0, 0]),
            other => panic!("expected empty V, got {other:?}"),
        }
    }

    #[test]
    fn svd_complex_matrix_reconstructs() {
        let complex = ComplexTensor::new(
            vec![(1.0, 2.0), (2.0, -1.0), (0.0, 0.0), (3.0, 1.0)],
            vec![2, 2],
        )
        .expect("complex tensor");
        let eval = evaluate(Value::ComplexTensor(complex.clone()), &[]).expect("svd complex");

        let u = complex_matrix_from_value(eval.u());
        let s_real = dmatrix_from_value(eval.sigma_matrix());
        let s_complex = s_real.map(|v| Complex64::new(v, 0.0));
        let v = complex_matrix_from_value(eval.v());

        let recon = &u * s_complex * v.adjoint();
        let original_data: Vec<Complex64> = complex
            .data
            .iter()
            .map(|(re, im)| Complex64::new(*re, *im))
            .collect();
        let original = DMatrix::from_column_slice(complex.rows, complex.cols, &original_data);
        complex_matrix_close(&recon, &original, 1e-10);
    }

    #[test]
    fn svd_numeric_zero_and_string_zero_request_economy() {
        let tall = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).expect("tensor");

        let eval_numeric =
            evaluate(Value::Tensor(tall.clone()), &[Value::Num(0.0)]).expect("numeric econ");
        let eval_string =
            evaluate(Value::Tensor(tall.clone()), &[Value::from("0")]).expect("string econ");
        let eval_word =
            evaluate(Value::Tensor(tall.clone()), &[Value::from("economy")]).expect("word econ");

        for eval in &[eval_numeric, eval_string, eval_word] {
            let u = tensor_from_value(eval.u());
            let s = tensor_from_value(eval.sigma());
            assert_eq!(u.shape, vec![3, 2], "unexpected U shape {:?}", u.shape);
            assert_eq!(s.shape, vec![2, 2], "unexpected S shape {:?}", s.shape);
        }
    }

    #[test]
    fn svd_matrix_option_overrides_vector() {
        let matrix = Tensor::new(vec![3.0, 0.0, 0.0, 1.0], vec![2, 2]).expect("tensor");
        let eval = evaluate(
            Value::Tensor(matrix),
            &[Value::from("vector"), Value::from("matrix")],
        )
        .expect("svd");
        match eval.sigma() {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert!(t.data[0] >= t.data[1]);
            }
            other => panic!("expected sigma matrix, got {other:?}"),
        }
    }

    #[test]
    fn svd_invalid_option_errors() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("tensor");
        let err = evaluate(Value::Tensor(matrix), &[Value::from("bogus")]).unwrap_err();
        assert!(err.contains("unknown option"));
    }

    #[test]
    fn svd_too_many_option_arguments_errors() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("tensor");
        let err = evaluate(
            Value::Tensor(matrix),
            &[
                Value::from("econ"),
                Value::from("vector"),
                Value::from("matrix"),
            ],
        )
        .unwrap_err();
        assert!(err.contains("too many option arguments"));
    }

    #[test]
    fn svd_numeric_nonzero_option_errors() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("tensor");
        let err = evaluate(Value::Tensor(matrix), &[Value::Num(1.0)]).unwrap_err();
        assert!(
            err.contains("numeric option must be 0"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn svd_logical_input_matches_numeric() {
        let logical = LogicalArray::new(vec![1, 0, 1, 1], vec![2, 2]).expect("logical");
        let logical_eval =
            evaluate(Value::LogicalArray(logical.clone()), &[]).expect("logical svd");
        let numeric = Tensor::new(vec![1.0, 0.0, 1.0, 1.0], vec![2, 2]).expect("numeric tensor");
        let numeric_eval = evaluate(Value::Tensor(numeric), &[]).expect("numeric svd");

        let logical_s = tensor_from_value(logical_eval.singular_values());
        let numeric_s = tensor_from_value(numeric_eval.singular_values());
        assert_eq!(logical_s.shape, numeric_s.shape);
        for (a, b) in logical_s.data.iter().zip(numeric_s.data.iter()) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    #[test]
    fn svd_three_dimensional_input_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2, 1]).expect("tensor");
        let err = evaluate(Value::Tensor(tensor), &[]).unwrap_err();
        assert!(err.contains("must be 2-D"));
    }

    #[test]
    fn svd_econ_shapes_match_matlab() {
        let tall = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).expect("tensor");
        let eval = evaluate(Value::Tensor(tall), &[Value::from("econ")]).expect("svd");
        let u = tensor_from_value(eval.u());
        let s = tensor_from_value(eval.sigma());
        let v = tensor_from_value(eval.v());
        assert_eq!(u.shape, vec![3, 2]);
        assert_eq!(s.shape, vec![2, 2]);
        assert_eq!(v.shape, vec![2, 2]);

        let wide =
            Tensor::new(vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0], vec![2, 4]).expect("tensor");
        let eval = evaluate(
            Value::Tensor(wide),
            &[Value::from("econ"), Value::from("vector")],
        )
        .expect("svd");
        let u = tensor_from_value(eval.u());
        let s_val = eval.sigma();
        match s_val {
            Value::Tensor(t) => assert_eq!(t.shape, vec![2, 1]),
            Value::Num(_) => {}
            other => panic!("unexpected sigma value {other:?}"),
        }
        assert_eq!(u.shape, vec![2, 2]);
        let s_matrix = tensor_from_value(eval.sigma_matrix());
        assert_eq!(s_matrix.shape, vec![2, 4]);
        match eval.v() {
            Value::Tensor(t) => assert_eq!(t.shape, vec![4, 4]),
            Value::ComplexTensor(ct) => assert_eq!(ct.shape, vec![4, 4]),
            other => panic!("unexpected V shape from {other:?}"),
        }
    }

    #[test]
    fn svd_vector_option_returns_column_vector() {
        let matrix = Tensor::new(vec![3.0, 0.0, 0.0, 1.0], vec![2, 2]).expect("tensor");
        let eval = evaluate(Value::Tensor(matrix), &[Value::from("vector")]).expect("svd");
        match eval.sigma() {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert!(t.data[0] >= t.data[1]);
            }
            Value::Num(n) => assert!(n >= 0.0),
            other => panic!("expected vector singular values, got {other:?}"),
        }
    }

    #[test]
    fn svd_gpu_input_gathers_to_host() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).expect("tensor");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = svd_builtin(Value::GpuTensor(handle), Vec::new()).expect("svd");
            match result {
                Value::Tensor(t) => assert_eq!(t.shape, vec![2, 1]),
                Value::Num(n) => assert!(n >= 0.0),
                other => panic!("expected host tensor, got {other:?}"),
            }
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn svd_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("register wgpu provider");

        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).expect("tensor");
        let host_eval = evaluate(Value::Tensor(tensor.clone()), &[]).expect("host eval");
        let host_u = dmatrix_from_value(host_eval.u());
        let host_s = dmatrix_from_value(host_eval.sigma_matrix());
        let host_v = dmatrix_from_value(host_eval.v());

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("provider");
        let handle = provider.upload(&view).expect("upload");
        let gpu_eval = evaluate(Value::GpuTensor(handle), &[]).expect("gpu eval");
        let gpu_u = dmatrix_from_value(gpu_eval.u());
        let gpu_s = dmatrix_from_value(gpu_eval.sigma_matrix());
        let gpu_v = dmatrix_from_value(gpu_eval.v());

        matrix_close(&gpu_u, &host_u, 1e-10);
        matrix_close(&gpu_s, &host_s, 1e-10);
        matrix_close(&gpu_v, &host_v, 1e-10);
    }

    #[test]
    fn svd_vector_matches_host_norm() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).expect("tensor");
        let s = svd_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("svd");
        let expected = tensor.data.iter().map(|v| v * v).sum::<f64>().sqrt();
        match s {
            Value::Num(n) => assert!((n - expected).abs() < 1e-10),
            Value::Tensor(t) => assert!((t.data[0] - expected).abs() < 1e-10),
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
