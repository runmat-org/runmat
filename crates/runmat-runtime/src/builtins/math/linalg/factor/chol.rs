//! MATLAB-compatible `chol` builtin with upper/lower forms and failure flag.
//!
//! This implementation matches MATLAB semantics for dense matrices, including
//! the two-output form that reports the leading minor index when a matrix fails
//! the positive-definiteness test. GPU execution is delegated to acceleration
//! providers when available, with automatic host fallbacks.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, random_args, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, ProviderCholResult};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

const BUILTIN_NAME: &str = "chol";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "chol",
        builtin_path = "crate::builtins::math::linalg::factor::chol"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "chol"
category: "math/linalg/factor"
keywords: ["chol", "cholesky", "factorization", "positive definite", "lower", "upper"]
summary: "Cholesky factorization with MATLAB-compatible upper and lower forms."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f64"]
  broadcasting: "none"
  notes: "Uses provider chol hook when available, otherwise gathers to host and re-uploads the factor."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::linalg::factor::chol::tests"
  integration: "builtins::math::linalg::factor::chol::tests::chol_gpu_provider_roundtrip"
---

# What does the `chol` function do in MATLAB / RunMat?
`chol(A)` computes a Cholesky factorization of a Hermitian positive-definite matrix `A`.
By default, it returns the upper-triangular factor `R` satisfying `R' * R = A`. The two-output
form `[R, p] = chol(A)` reports whether the factorization succeeded: `p = 0` for success, and
otherwise `p` equals the index of the first leading principal minor that is not positive definite.

## How does the `chol` function behave in MATLAB / RunMat?
- Single output (`chol(A)`): Returns the upper-triangular factor. If `A` is not positive definite,
  a MATLAB-compatible error (`"Matrix must be positive definite."`) is raised.
- Two outputs (`[R, p] = chol(A)`): Returns the factor `R` and a flag `p`. When `p > 0`, `R`
  contains the partial factorization up to the `(p-1)`-th column.
- Lower form (`chol(A, 'lower')`): Returns a lower-triangular factor `L` such that `L * L' = A`.
  The two-output variant `[L, p] = chol(A, 'lower')` follows the same failure semantics.
- The option `'upper'` is accepted for completeness and matches the default behaviour.
- Real logical inputs are promoted to double precision. Complex inputs must be Hermitian positive
  definite (HPD); the factor preserves complex values.
- Scalar and `0×0` inputs are supported. `chol([])` returns an empty matrix with `p = 0`.

## `chol` Function GPU Execution Behaviour
When RunMat Accelerate is active, the planner keeps `gpuArray` inputs on device and asks the
registered provider for a `chol` factorization. Providers that implement the hook (the WGPU and
in-process backends included) return the MATLAB-compatible flag `p` while leaving the triangular
factor resident on the GPU.
If the provider lacks that hook or the matrix uses an unsupported precision/type, RunMat gathers the
data, executes the CPU implementation, and re-uploads the factor when a provider is present. The
flag output is always materialised as a host scalar.

## Examples of using the `chol` function in MATLAB / RunMat

### Upper-triangular Cholesky factor of a symmetric positive-definite matrix
```matlab
A = [4 12 -16; 12 37 -43; -16 -43 98];
R = chol(A);
```
Expected output:
```matlab
R =
     2     6    -8
     0     1    -5
     0     0     3
```

### Lower-triangular factor using the `'lower'` option
```matlab
A = [25 15 -5; 15 18  0; -5 0 11];
L = chol(A, 'lower');
```
Expected output:
```matlab
L =
     5     0     0
     3     3     0
    -1     1     3
```

### Detecting a non-positive-definite matrix with the two-output form
```matlab
A = [1 2; 2 1];
[R, p] = chol(A);
```
Expected output:
```matlab
p =
     2

R =
     1     2
     0     0
```

### Using the Cholesky factor to solve linear systems
```matlab
A = [10 2 3; 2 9 1; 3 1 7];
b = [1; 2; 3];
[R, p] = chol(A);
if p == 0
    y = R' \ b;
    x = R \ y;
end
```
`x` is the solution to `A * x = b`.

### Cholesky factor of a complex Hermitian positive-definite matrix
```matlab
A = [5 1-2i; 1+2i 4];
[R, p] = chol(A);
```
Expected output:
```matlab
p =
     0

R =
    2.2361   0.2236 -0.8944i
         0   1.9849 +0.5046i
```

### Running `chol` on a `gpuArray`
```matlab
G = gpuArray([6 2; 2 5]);
[R, p] = chol(G);
class(R)
```
Expected output:
```matlab
p =
     0

ans =
    'gpuArray'
```
When no GPU provider is registered, RunMat automatically gathers the input, performs the host
factorization, and leaves the results on the host.

## FAQ

### What does the second output `p` mean?
`p` is zero when the factorization succeeds. Otherwise, it is the index of the first leading
principal minor that is not positive definite. The returned factor contains the partial result up to
that point.

### How do I request the lower-triangular factor?
Pass `'lower'` (case-insensitive) as the second argument: `[L, p] = chol(A, 'lower')`. The default
behaviour (`chol(A)`) returns the upper-triangular factor.

### Why do I get an error when using a single output on an indefinite matrix?
The single-output form matches MATLAB and throws an error when the matrix is not positive definite.
Use the two-output form `[R, p]` to obtain the partial factor and inspect `p` without raising an
error.

### Does `chol` accept sparse matrices?
Not yet. RunMat currently implements the dense MATLAB semantics. Sparse support is planned in a
future release.

### Can I pass logical or integer arrays?
Yes. They are promoted to double precision before factorization, matching MATLAB behaviour.

### Do complex inputs need to be Hermitian?
Yes. `chol` operates on Hermitian positive-definite matrices. If the matrix is not Hermitian, `p`
will be non-zero in the two-output form, and the single-output form raises an error.

### How should I interpret the result when `p > 0`?
Only the leading `(p-1)` columns/rows of the factor are valid. The remaining portions are zeros,
mirroring MATLAB’s contract.

### Does the GPU path return the same flag `p`?
Yes. Providers report the MATLAB-compatible failure index, and RunMat converts it into the scalar
second output.

### Is the factor unique?
The standard Cholesky factorization returns an upper-triangular matrix with positive diagonal
entries. The lower-triangular form is the conjugate transpose of the upper form and is likewise
unique under the same assumptions.

### How should I choose between `chol` and `lu`/`qr`?
Use `chol` when the matrix is Hermitian positive definite—its triangular factors are cheaper to
compute and exploit symmetry. Use `lu` or `qr` for more general matrices.

## See Also
[lu](./lu), [qr](./qr), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/math/linalg/factor/chol.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/factor/chol.rs)
- Issues & feedback: [RunMat issue tracker](https://github.com/runmat-org/runmat/issues/new/choose)
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::factor::chol")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "chol",
    op_kind: GpuOpKind::Custom("chol-factor"),
    supported_precisions: &[ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("chol")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Uses the provider 'chol' hook when present; otherwise gathers to the host implementation.",
};

fn chol_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn with_chol_context(mut error: RuntimeError) -> RuntimeError {
    if error.message() == "interaction pending..." {
        return build_runtime_error("interaction pending...")
            .with_builtin(BUILTIN_NAME)
            .build();
    }
    if error.context.builtin.is_none() {
        error.context = error.context.with_builtin(BUILTIN_NAME);
    }
    error
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::factor::chol")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "chol",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Factorisation executes eagerly and does not participate in expression fusion.",
};

#[runtime_builtin(
    name = "chol",
    category = "math/linalg/factor",
    summary = "Cholesky factorization with MATLAB-compatible upper and lower forms.",
    keywords = "chol,cholesky,factorization,positive-definite",
    accel = "sink",
    sink = true,
    builtin_path = "crate::builtins::math::linalg::factor::chol"
)]
async fn chol_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(value, &rest).await?;
    if !eval.is_positive_definite() {
        return Err(chol_error("Matrix must be positive definite."));
    }
    Ok(eval.factor())
}

/// Evaluate `chol` while keeping both the factor and the failure index available.
#[derive(Clone)]
pub struct CholEval {
    factor: Value,
    flag: usize,
    triangle: CholTriangle,
}

impl CholEval {
    /// The factor (`R` or `L`) requested by the caller.
    pub fn factor(&self) -> Value {
        self.factor.clone()
    }

    /// MATLAB-compatible failure index (0 indicates success).
    pub fn flag(&self) -> Value {
        Value::Num(self.flag as f64)
    }

    /// Zero-based flag value (0 indicates success).
    pub fn flag_index(&self) -> usize {
        self.flag
    }

    /// The triangle variant that was requested.
    pub fn triangle(&self) -> CholTriangle {
        self.triangle
    }

    /// Returns true when the input matrix was positive definite.
    pub fn is_positive_definite(&self) -> bool {
        self.flag == 0
    }

    fn from_components(components: CholComponents, triangle: CholTriangle) -> BuiltinResult<Self> {
        let factor_matrix = match triangle {
            CholTriangle::Upper => components.upper.clone(),
            CholTriangle::Lower => components.upper.conjugate_transpose(),
        };
        let factor = matrix_to_value("chol", &factor_matrix)?;
        Ok(Self {
            factor,
            flag: components.info,
            triangle,
        })
    }

    fn from_provider(result: ProviderCholResult, triangle: CholTriangle) -> Self {
        Self {
            factor: Value::GpuTensor(result.factor),
            flag: result.info as usize,
            triangle,
        }
    }
}

/// Triangle variant for the Cholesky factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CholTriangle {
    Upper,
    Lower,
}

/// Compute the Cholesky factorization for the given value and option list.
pub async fn evaluate(value: Value, args: &[Value]) -> BuiltinResult<CholEval> {
    let triangle = parse_triangle(args)?;
    match value {
        Value::GpuTensor(handle) => {
            if let Some(eval) = evaluate_gpu(&handle, triangle)? {
                return Ok(eval);
            }
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(with_chol_context)?;
            evaluate_host_value(Value::Tensor(tensor), triangle).await
        }
        other => evaluate_host_value(other, triangle).await,
    }
}

async fn evaluate_host_value(value: Value, triangle: CholTriangle) -> BuiltinResult<CholEval> {
    let matrix = extract_matrix(value).await?;
    if matrix.rows != matrix.cols {
        return Err(chol_error("chol: input matrix must be square"));
    }
    let components = chol_factor(matrix)?;
    CholEval::from_components(components, triangle)
}

fn evaluate_gpu(
    handle: &GpuTensorHandle,
    triangle: CholTriangle,
) -> BuiltinResult<Option<CholEval>> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let lower = matches!(triangle, CholTriangle::Lower);
        if let Ok(result) = provider.chol(handle, lower) {
            return Ok(Some(CholEval::from_provider(result, triangle)));
        }
    }
    Ok(None)
}

fn parse_triangle(args: &[Value]) -> BuiltinResult<CholTriangle> {
    if args.is_empty() {
        return Ok(CholTriangle::Upper);
    }
    if args.len() > 1 {
        return Err(chol_error("chol: too many option arguments"));
    }
    let Some(option) = tensor::value_to_string(&args[0]) else {
        return Err(chol_error(
            "chol: option must be a string or character vector",
        ));
    };
    match option.trim().to_ascii_lowercase().as_str() {
        "upper" => Ok(CholTriangle::Upper),
        "lower" => Ok(CholTriangle::Lower),
        other => Err(chol_error(format!("chol: unknown option '{other}'"))),
    }
}

const EPS: f64 = 1.0e-12;

#[inline]
fn hermitian_pair_matches(a: Complex64, b: Complex64) -> bool {
    let diff = a - b.conj();
    let scale = a.norm().max(b.norm()).max(1.0);
    diff.norm() <= EPS * scale
}

fn chol_factor(matrix: RowMajorMatrix) -> BuiltinResult<CholComponents> {
    let n = matrix.rows;
    if n == 0 {
        return Ok(CholComponents {
            upper: RowMajorMatrix::zeros(0, 0),
            info: 0,
        });
    }
    let mut upper = RowMajorMatrix::zeros(n, n);
    let mut info = 0usize;

    'outer: for j in 0..n {
        for i in 0..j {
            if !hermitian_pair_matches(matrix.get(i, j), matrix.get(j, i)) {
                info = j + 1;
                break 'outer;
            }
        }

        for i in 0..=j {
            let mut sum = matrix.get(i, j);
            for k in 0..i {
                let rik = upper.get(k, i).conj();
                let rkj = upper.get(k, j);
                sum -= rik * rkj;
            }
            if i == j {
                let imag_tol = EPS * sum.re.abs().max(1.0);
                if !sum.re.is_finite()
                    || !sum.im.is_finite()
                    || sum.re <= 0.0
                    || sum.im.abs() > imag_tol
                {
                    info = j + 1;
                    break 'outer;
                }
                let diag = sum.re.sqrt();
                upper.set(i, i, Complex64::new(diag, 0.0));
            } else {
                let denom = upper.get(i, i);
                if denom.norm() <= EPS {
                    info = i + 1;
                    break 'outer;
                }
                upper.set(i, j, sum / denom);
            }
        }
    }

    if info != 0 {
        let start = info.saturating_sub(1).min(n);
        for row in start..n {
            for col in row..n {
                upper.set(row, col, Complex64::new(0.0, 0.0));
            }
        }
    }

    Ok(CholComponents { upper, info })
}

async fn extract_matrix(value: Value) -> BuiltinResult<RowMajorMatrix> {
    match value {
        Value::Tensor(tensor) => RowMajorMatrix::from_tensor(&tensor, "chol"),
        Value::ComplexTensor(ct) => RowMajorMatrix::from_complex_tensor(&ct, "chol"),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|err| chol_error(format!("chol: {err}")))?;
            RowMajorMatrix::from_tensor(&tensor, "chol")
        }
        Value::Num(n) => Ok(RowMajorMatrix::from_scalar(Complex64::new(n, 0.0))),
        Value::Int(i) => Ok(RowMajorMatrix::from_scalar(Complex64::new(i.to_f64(), 0.0))),
        Value::Bool(b) => Ok(RowMajorMatrix::from_scalar(Complex64::new(
            if b { 1.0 } else { 0.0 },
            0.0,
        ))),
        Value::Complex(re, im) => Ok(RowMajorMatrix::from_scalar(Complex64::new(re, im))),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(with_chol_context)?;
            RowMajorMatrix::from_tensor(&tensor, "chol")
        }
        other => Err(chol_error(format!(
            "chol: unsupported input type {:?}; expected numeric or logical values",
            other
        ))),
    }
}

fn matrix_to_value(label: &str, matrix: &RowMajorMatrix) -> BuiltinResult<Value> {
    let mut has_imag = false;
    for val in &matrix.data {
        if val.im.abs() > EPS {
            has_imag = true;
            break;
        }
    }
    if has_imag {
        let mut data = Vec::with_capacity(matrix.rows * matrix.cols);
        for col in 0..matrix.cols {
            for row in 0..matrix.rows {
                let idx = row * matrix.cols + col;
                let v = matrix.data[idx];
                data.push((v.re, v.im));
            }
        }
        let tensor = ComplexTensor::new(data, vec![matrix.rows, matrix.cols])
            .map_err(|e| chol_error(format!("{label}: {e}")))?;
        Ok(random_args::complex_tensor_into_value(tensor))
    } else {
        let mut data = Vec::with_capacity(matrix.rows * matrix.cols);
        for col in 0..matrix.cols {
            for row in 0..matrix.rows {
                let idx = row * matrix.cols + col;
                data.push(matrix.data[idx].re);
            }
        }
        let tensor = Tensor::new(data, vec![matrix.rows, matrix.cols])
            .map_err(|e| chol_error(format!("{label}: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

struct CholComponents {
    upper: RowMajorMatrix,
    info: usize,
}

#[derive(Clone)]
struct RowMajorMatrix {
    rows: usize,
    cols: usize,
    data: Vec<Complex64>,
}

impl RowMajorMatrix {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![Complex64::new(0.0, 0.0); rows.saturating_mul(cols)],
        }
    }

    fn from_scalar(value: Complex64) -> Self {
        Self {
            rows: 1,
            cols: 1,
            data: vec![value],
        }
    }

    fn from_tensor(tensor: &Tensor, label: &str) -> BuiltinResult<Self> {
        if tensor.shape.len() > 2 {
            return Err(chol_error(format!("{label}: input must be 2-D")));
        }
        let rows = tensor.rows();
        let cols = tensor.cols();
        let mut data = vec![Complex64::new(0.0, 0.0); rows.saturating_mul(cols)];
        for col in 0..cols {
            for row in 0..rows {
                let idx_col_major = row + col * rows;
                let idx_row_major = row * cols + col;
                data[idx_row_major] = Complex64::new(tensor.data[idx_col_major], 0.0);
            }
        }
        Ok(Self { rows, cols, data })
    }

    fn from_complex_tensor(tensor: &ComplexTensor, label: &str) -> BuiltinResult<Self> {
        if tensor.shape.len() > 2 {
            return Err(chol_error(format!("{label}: input must be 2-D")));
        }
        let rows = tensor.rows;
        let cols = tensor.cols;
        let mut data = vec![Complex64::new(0.0, 0.0); rows.saturating_mul(cols)];
        for col in 0..cols {
            for row in 0..rows {
                let idx_col_major = row + col * rows;
                let idx_row_major = row * cols + col;
                let (re, im) = tensor.data[idx_col_major];
                data[idx_row_major] = Complex64::new(re, im);
            }
        }
        Ok(Self { rows, cols, data })
    }

    fn get(&self, row: usize, col: usize) -> Complex64 {
        self.data[row * self.cols + col]
    }

    fn set(&mut self, row: usize, col: usize, value: Complex64) {
        self.data[row * self.cols + col] = value;
    }

    fn conjugate_transpose(&self) -> Self {
        let mut out = RowMajorMatrix::zeros(self.cols, self.rows);
        for row in 0..self.rows {
            for col in row..self.cols {
                let value = self.get(row, col);
                out.set(col, row, value.conj());
            }
        }
        out
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{LogicalArray, Tensor as Matrix};

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    fn tensor_from_value(value: Value) -> Matrix {
        match value {
            Value::Tensor(t) => t,
            Value::Num(n) => Matrix::new(vec![n], vec![1, 1]).expect("tensor"),
            other => panic!("expected tensor value, got {other:?}"),
        }
    }

    fn reconstruct_from_upper(matrix: &Matrix) -> Matrix {
        let rows = matrix.rows();
        let cols = matrix.cols();
        assert_eq!(rows, cols, "expected square matrix");
        let mut data = vec![0.0; rows * cols];
        // Compute R' * R for validation (column-major input)
        for i in 0..rows {
            for j in 0..rows {
                let mut sum = 0.0;
                for k in 0..rows {
                    let rik = if k <= i {
                        matrix.data[k + i * rows]
                    } else {
                        0.0
                    };
                    let rjk = if k <= j {
                        matrix.data[k + j * rows]
                    } else {
                        0.0
                    };
                    sum += rik * rjk;
                }
                data[i + j * rows] = sum;
            }
        }
        Matrix::new(data, vec![rows, rows]).expect("matrix")
    }

    fn reconstruct_from_lower(matrix: &Matrix) -> Matrix {
        let rows = matrix.rows();
        let cols = matrix.cols();
        assert_eq!(rows, cols, "expected square matrix");
        let mut data = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..rows {
                let mut sum = 0.0;
                for k in 0..rows {
                    let lik = if i >= k {
                        matrix.data[i + k * rows]
                    } else {
                        0.0
                    };
                    let ljk = if j >= k {
                        matrix.data[j + k * rows]
                    } else {
                        0.0
                    };
                    sum += lik * ljk;
                }
                data[i + j * rows] = sum;
            }
        }
        Matrix::new(data, vec![rows, rows]).expect("matrix")
    }

    fn tensor_close(lhs: &Matrix, rhs: &Matrix, tol: f64) {
        assert_eq!(lhs.shape, rhs.shape, "shape mismatch");
        for (a, b) in lhs.data.iter().zip(rhs.data.iter()) {
            assert!(
                (a - b).abs() <= tol,
                "tensors differ: {a} vs {b} (tol {tol})"
            );
        }
    }

    fn complex_tensor_from_value(value: Value) -> ComplexTensor {
        match value {
            Value::ComplexTensor(ct) => ct,
            Value::Complex(re, im) => {
                ComplexTensor::new(vec![(re, im)], vec![1, 1]).expect("complex tensor")
            }
            Value::Tensor(t) => {
                let data: Vec<(f64, f64)> = t.data.iter().map(|&v| (v, 0.0)).collect();
                ComplexTensor::new(data, t.shape.clone()).expect("complex tensor")
            }
            Value::Num(n) => {
                ComplexTensor::new(vec![(n, 0.0)], vec![1, 1]).expect("complex tensor")
            }
            other => panic!("expected complex-capable value, got {other:?}"),
        }
    }

    fn reconstruct_complex_upper(matrix: &ComplexTensor) -> ComplexTensor {
        let rows = matrix.rows;
        let cols = matrix.cols;
        assert_eq!(rows, cols, "expected square matrix");
        let mut data = vec![(0.0, 0.0); rows * rows];
        for i in 0..rows {
            for j in 0..rows {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..rows {
                    let rik = if k <= i {
                        let (re, im) = matrix.data[k + i * rows];
                        Complex64::new(re, im)
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    let rjk = if k <= j {
                        let (re, im) = matrix.data[k + j * rows];
                        Complex64::new(re, im)
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    sum += rik.conj() * rjk;
                }
                data[i + j * rows] = (sum.re, sum.im);
            }
        }
        ComplexTensor::new(data, vec![rows, rows]).expect("complex tensor")
    }

    fn reconstruct_complex_lower(matrix: &ComplexTensor) -> ComplexTensor {
        let rows = matrix.rows;
        let cols = matrix.cols;
        assert_eq!(rows, cols, "expected square matrix");
        let mut data = vec![(0.0, 0.0); rows * rows];
        for i in 0..rows {
            for j in 0..rows {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..rows {
                    let lik = if i >= k {
                        let (re, im) = matrix.data[i + k * rows];
                        Complex64::new(re, im)
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    let ljk = if j >= k {
                        let (re, im) = matrix.data[j + k * rows];
                        Complex64::new(re, im)
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    sum += lik * ljk.conj();
                }
                data[i + j * rows] = (sum.re, sum.im);
            }
        }
        ComplexTensor::new(data, vec![rows, rows]).expect("complex tensor")
    }

    fn complex_tensor_close(lhs: &ComplexTensor, rhs: &ComplexTensor, tol: f64) {
        assert_eq!(lhs.shape, rhs.shape, "shape mismatch");
        for ((ar, ai), (br, bi)) in lhs.data.iter().zip(rhs.data.iter()) {
            let a = Complex64::new(*ar, *ai);
            let b = Complex64::new(*br, *bi);
            assert!(
                (a - b).norm() <= tol,
                "tensors differ: {a:?} vs {b:?} (tol {tol})"
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_upper_factor_matches_reference() {
        let a = Matrix::new(
            vec![
                4.0, 12.0, -16.0, //
                12.0, 37.0, -43.0, //
                -16.0, -43.0, 98.0,
            ],
            vec![3, 3],
        )
        .unwrap();
        let r = chol_builtin(Value::Tensor(a.clone()), Vec::new()).expect("chol");
        let r_tensor = tensor_from_value(r);
        assert_eq!(r_tensor.shape, vec![3, 3]);
        for diag in 0..3 {
            let value = r_tensor.data[diag + diag * 3];
            assert!(value > 0.0, "Cholesky diagonal must be positive");
        }
        let recon = reconstruct_from_upper(&r_tensor);
        tensor_close(&recon, &a, 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_upper_option_matches_default() {
        let a = Matrix::new(
            vec![
                7.0, 2.0, 1.0, //
                2.0, 5.0, 2.0, //
                1.0, 2.0, 3.0,
            ],
            vec![3, 3],
        )
        .unwrap();
        let default = chol_builtin(Value::Tensor(a.clone()), Vec::new()).expect("chol");
        let explicit =
            chol_builtin(Value::Tensor(a.clone()), vec![Value::from("upper")]).expect("chol upper");
        let default_tensor = tensor_from_value(default);
        let explicit_tensor = tensor_from_value(explicit);
        tensor_close(&default_tensor, &explicit_tensor, 1e-12);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_lower_option_returns_lower_factor() {
        let a = Matrix::new(
            vec![
                25.0, 15.0, -5.0, //
                15.0, 18.0, 0.0, //
                -5.0, 0.0, 11.0,
            ],
            vec![3, 3],
        )
        .unwrap();
        let result =
            chol_builtin(Value::Tensor(a.clone()), vec![Value::from("lower")]).expect("chol");
        let l = tensor_from_value(result);
        assert_eq!(l.shape, vec![3, 3]);
        for diag in 0..3 {
            let value = l.data[diag + diag * 3];
            assert!(value > 0.0, "Cholesky diagonal must be positive");
        }
        let recon = reconstruct_from_lower(&l);
        tensor_close(&recon, &a, 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_two_output_lower_variant() {
        let a = Matrix::new(
            vec![
                9.0, 3.0, 3.0, //
                3.0, 5.0, 1.0, //
                3.0, 1.0, 7.0,
            ],
            vec![3, 3],
        )
        .unwrap();
        let eval = evaluate(Value::Tensor(a.clone()), &[Value::from("lower")]).expect("chol eval");
        assert_eq!(eval.flag_index(), 0);
        assert_eq!(eval.triangle(), CholTriangle::Lower);
        let factor = tensor_from_value(eval.factor());
        let recon = reconstruct_from_lower(&factor);
        tensor_close(&recon, &a, 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_two_output_reports_failure() {
        let a = Matrix::new(vec![1.0, 2.0, 2.0, 1.0], vec![2, 2]).expect("matrix");
        let eval = evaluate(Value::Tensor(a), &[]).expect("chol eval");
        assert_eq!(eval.flag_index(), 2);
        let factor = tensor_from_value(eval.factor());
        assert_eq!(factor.shape, vec![2, 2]);
        assert!((factor.data[0] - 1.0).abs() < 1e-12);
        assert!((factor.data[1] - 0.0).abs() < 1e-12);
        assert!((factor.data[2] - 2.0).abs() < 1e-12);
        assert!((factor.data[3] - 0.0).abs() < 1e-12);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_single_output_errors_on_failure() {
        let a = Matrix::new(vec![1.0, 2.0, 2.0, 1.0], vec![2, 2]).expect("matrix");
        let err = error_message(chol_builtin(Value::Tensor(a), Vec::new()).unwrap_err());
        assert!(err.contains("positive definite"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_invalid_option_errors() {
        let a = Matrix::new(vec![4.0, 1.0, 1.0, 3.0], vec![2, 2]).unwrap();
        let err = error_message(
            chol_builtin(Value::Tensor(a), vec![Value::from("diagonal")]).unwrap_err(),
        );
        assert!(err.to_ascii_lowercase().contains("unknown option"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_non_square_errors() {
        let a = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let err = error_message(chol_builtin(Value::Tensor(a), Vec::new()).unwrap_err());
        assert!(err.to_ascii_lowercase().contains("square"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_empty_matrix_returns_empty() {
        let empty = Matrix::new(Vec::<f64>::new(), vec![0, 0]).unwrap();
        let eval = evaluate(Value::Tensor(empty.clone()), &[]).expect("chol eval");
        assert_eq!(eval.flag_index(), 0);
        let factor = tensor_from_value(eval.factor());
        assert_eq!(factor.shape, vec![0, 0]);
        assert!(factor.data.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_non_hermitian_reports_failure() {
        let a = Matrix::new(vec![2.0, 1.0, 0.0, 2.0], vec![2, 2]).expect("matrix");
        let eval = evaluate(Value::Tensor(a), &[]).expect("chol eval");
        assert_eq!(eval.flag_index(), 2);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_logical_input_factorizes() {
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).expect("logical array");
        let result = chol_builtin(Value::LogicalArray(logical), Vec::new()).expect("chol");
        let factor = tensor_from_value(result);
        let recon = reconstruct_from_upper(&factor);
        let identity = Matrix::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        tensor_close(&recon, &identity, 1e-12);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_complex_positive_definite() {
        let complex = ComplexTensor::new(
            vec![(5.0, 0.0), (1.0, 2.0), (1.0, -2.0), (4.0, 0.0)],
            vec![2, 2],
        )
        .unwrap();
        let eval = evaluate(Value::ComplexTensor(complex.clone()), &[]).expect("chol eval");
        assert_eq!(eval.flag_index(), 0);
        let factor = complex_tensor_from_value(eval.factor());
        let recon = reconstruct_complex_upper(&factor);
        complex_tensor_close(&recon, &complex, 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_complex_lower_variant() {
        let complex = ComplexTensor::new(
            vec![(5.0, 0.0), (1.0, 2.0), (1.0, -2.0), (4.0, 0.0)],
            vec![2, 2],
        )
        .unwrap();
        let eval = evaluate(
            Value::ComplexTensor(complex.clone()),
            &[Value::from("lower")],
        )
        .expect("chol eval");
        assert_eq!(eval.flag_index(), 0);
        let factor = complex_tensor_from_value(eval.factor());
        let recon = reconstruct_complex_lower(&factor);
        complex_tensor_close(&recon, &complex, 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let a = Matrix::new(vec![6.0, 2.0, 2.0, 5.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &a.data,
                shape: &a.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = chol_builtin(Value::GpuTensor(handle), Vec::new()).expect("chol");
            let gathered = test_support::gather(result).expect("gather");
            let recon = reconstruct_from_upper(&gathered);
            tensor_close(&recon, &a, 1e-10);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_gpu_failure_flag() {
        test_support::with_test_provider(|provider| {
            let a = Matrix::new(vec![1.0, 2.0, 2.0, 1.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &a.data,
                shape: &a.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let eval = evaluate(Value::GpuTensor(handle), &[]).expect("chol eval");
            assert_eq!(eval.flag_index(), 2);
            let factor = eval.factor();
            assert!(matches!(factor, Value::GpuTensor(_)));
            let gathered = test_support::gather(factor).expect("gather factor");
            assert!((gathered.data[0] - 1.0).abs() < 1e-12);
            assert!((gathered.data[1] - 0.0).abs() < 1e-12);
            assert!((gathered.data[2] - 2.0).abs() < 1e-12);
            assert!((gathered.data[3] - 0.0).abs() < 1e-12);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn chol_wgpu_matches_cpu() {
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

        let tensor = Matrix::new(
            vec![
                10.0, 2.0, 3.0, //
                2.0, 9.0, 1.0, //
                3.0, 1.0, 7.0,
            ],
            vec![3, 3],
        )
        .unwrap();

        let host_eval = evaluate(Value::Tensor(tensor.clone()), &[]).expect("host eval");
        let host_factor = tensor_from_value(host_eval.factor());

        let provider = runmat_accelerate_api::provider().expect("provider");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");

        let gpu_eval = evaluate(Value::GpuTensor(handle), &[]).expect("gpu eval");
        assert_eq!(gpu_eval.flag_index(), 0, "gpu chol should succeed");
        let gpu_factor = test_support::gather(gpu_eval.factor()).expect("gather factor");

        tensor_close(&gpu_factor, &host_factor, tol);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chol_accepts_scalar() {
        let result = chol_builtin(Value::Num(9.0), Vec::new()).expect("chol");
        match result {
            Value::Num(n) => assert!((n - 3.0).abs() < 1e-12),
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                assert!((t.data[0] - 3.0).abs() < 1e-12);
            }
            other => panic!("expected scalar-like, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    fn chol_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::chol_builtin(value, rest))
    }

    fn evaluate(value: Value, args: &[Value]) -> BuiltinResult<CholEval> {
        block_on(super::evaluate(value, args))
    }
}
