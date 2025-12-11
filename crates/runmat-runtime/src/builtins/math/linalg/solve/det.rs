//! MATLAB-compatible `det` builtin with GPU-aware semantics for RunMat.

use nalgebra::{DMatrix, LU};
use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

const NAME: &str = "det";

#[cfg_attr(feature = "doc_export", runmat_macros::register_doc_text(name = NAME))]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "det"
category: "math/linalg/solve"
keywords: ["det", "determinant", "linear algebra", "matrix", "gpu"]
summary: "Compute the determinant of a square matrix with MATLAB-compatible pivoting and GPU fallbacks."
references: ["https://www.mathworks.com/help/matlab/ref/det.html"]
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f64"]
  broadcasting: "none"
  notes: "Uses provider LU factorization when available; real matrices re-upload a scalar result, while complex inputs gather to the host."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::linalg::solve::det::tests"
  gpu: "builtins::math::linalg::solve::det::tests::det_gpu_roundtrip_matches_cpu"
  wgpu: "builtins::math::linalg::solve::det::tests::det_wgpu_matches_cpu"
  doc: "builtins::math::linalg::solve::det::tests::doc_examples_present"
---

# What does the `det` function do in MATLAB / RunMat?
`det(A)` returns the determinant of a real or complex square matrix `A`. Scalars behave like
`det(1x1) = A(1,1)`, and the empty matrix (`[]`) has determinant `1`, matching MATLAB's convention.

## How does the `det` function behave in MATLAB / RunMat?
- Inputs must be 2-D square matrices. RunMat rejects non-square or higher-rank inputs with the MATLAB error
  `"det: input must be a square matrix."`
- Determinants are computed from an LU factorization with partial pivoting, ensuring numerical behavior that
  mirrors MATLAB. Singular matrices return `0` (within floating-point round-off).
- Logical and integer inputs are promoted to double precision before evaluation.
- Complex inputs are handled with full complex arithmetic. The result has complex type whenever the input is complex.
- The determinant of the empty matrix (`[]`) is `1`.

## `det` Function GPU Execution Behaviour
When a GPU acceleration provider is active, RunMat first asks the provider for an LU factorization via the `lu` hook.
Providers that implement it keep the factors on device, multiply the diagonal of `U`, and re-upload the scalar
determinant for real inputs so downstream kernels keep running on the GPU. Complex matrices currently gather to the
host for the final scalar. Providers that do not expose `lu` (including the current fallback backends) automatically
route to the shared CPU implementation with no user intervention required.

## Examples of using the `det` function in MATLAB / RunMat

### Determinant Of A 2x2 Matrix
```matlab
A = [4 -2; 1 3];
d = det(A);
```
Expected output:
```matlab
d = 14
```

### Checking Zero Determinant For Singular Matrix
```matlab
B = [1 2; 2 4];
d = det(B);
```
Expected output:
```matlab
d = 0
```

### Determinant Of A Complex Matrix
```matlab
C = [1+2i 0; 3i 4-1i];
d = det(C);
```
Expected output (values rounded):
```matlab
d = 6 + 7i
```

### Determinant Equals Product Of Diagonal For Triangular Matrix
```matlab
U = [3 2 1; 0 5 -1; 0 0 2];
d = det(U);
```
Expected output:
```matlab
d = 30
```

### Determinant Of An Empty Matrix
```matlab
d = det([]);
```
Expected output:
```matlab
d = 1
```

### Using `det` With `gpuArray` Data
```matlab
G = gpuArray([2 0 0; 0 3 0; 1 0 4]);
d_gpu = det(G);     % stays on the GPU when a provider is active
d = gather(d_gpu);
```
Expected output:
```matlab
d = 24
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do NOT need to call `gpuArray` yourself in RunMat (unlike MATLAB). The fusion planner keeps producers on the
GPU, `det` invokes the provider's `lu` hook when available, and real determinants are re-uploaded as device scalars so
subsequent kernels keep their residency. When the provider lacks `lu`, RunMat transparently gathers the matrix, computes
the determinant with the host fallback, and re-uploads the scalar for real inputs. Complex determinants currently return
host scalars until device complex scalars land. `gpuArray` and `gather` remain available for explicit residency control
and MATLAB compatibility.

## FAQ

### Why does `det` require square matrices?
The determinant is only defined for square matrices. RunMat mirrors MATLAB by rejecting non-square inputs with
`"det: input must be a square matrix."`

### What is `det([])`?
The determinant of the empty matrix (`[]`) is `1`, matching MATLAB's convention for the product of an empty diagonal.

### Does `det` warn for nearly singular matrices?
No. RunMat returns the floating-point result of the LU-based determinant. Very small magnitudes indicate near singularity,
just as in MATLAB.

### How does `det` handle complex matrices?
Complex matrices are factorized in complex arithmetic. The result is complex; if the imaginary part happens to be zero,
it is still reported through the complex number interface.

### Will the result stay on the GPU?
Yes, when a GPU provider is active RunMat re-uploads the scalar determinant so that subsequent GPU code continues to
operate on device-resident data. Providers without LU support fall back to the CPU path and return a host scalar.

### Can `det` overflow or underflow?
Large determinants can overflow or underflow double precision just as in MATLAB. RunMat does not rescale the matrix; if
you require scaling, consider `log(det(A))` via `lu` or `chol`.

### Do logical inputs work?
Yes. Logical arrays are promoted to doubles before computing the determinant.

### Is the determinant computed exactly?
No. The LU factorization works in floating-point, so the result is subject to round-off. The behavior matches MATLAB's
`det`.

## See Also
[inv](./inv), [pinv](./pinv), [lu](../factor/lu), [trace](../ops/trace), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `det` function is available at: [`crates/runmat-runtime/src/builtins/math/linalg/solve/det.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/solve/det.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[derive(Debug, Clone, Copy)]
enum Determinant {
    Real(f64),
    Complex(f64, f64),
}

impl Determinant {
    fn apply_sign(self, sign: f64) -> Self {
        match self {
            Self::Real(value) => Self::Real(value * sign),
            Self::Complex(re, im) => Self::Complex(re * sign, im * sign),
        }
    }

    fn into_value(self) -> Value {
        match self {
            Self::Real(value) => Value::Num(value),
            Self::Complex(re, im) => Value::Complex(re, im),
        }
    }
}

#[runmat_macros::register_gpu_spec(wasm_path = "crate::builtins::math::linalg::solve::det")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("det"),
    supported_precisions: &[ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("lu")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Real inputs re-upload their determinant to preserve residency; complex inputs currently return host scalars when LU hooks are available.",
};

#[runmat_macros::register_fusion_spec(wasm_path = "crate::builtins::math::linalg::solve::det")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Determinant evaluation is a terminal scalar operation and does not participate in fusion plans.",
};

#[runtime_builtin(
    name = "det",
    category = "math/linalg/solve",
    summary = "Compute the determinant of a square matrix.",
    keywords = "det,determinant,linear algebra,matrix,gpu",
    accel = "det",
    wasm_path = "crate::builtins::math::linalg::solve::det"
)]
fn det_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => det_gpu(handle),
        Value::ComplexTensor(tensor) => det_complex_value(tensor),
        Value::Complex(re, im) => Ok(Value::Complex(re, im)),
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other)?;
            det_real_value(tensor)
        }
    }
}

fn det_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        match det_gpu_via_provider(provider, &handle) {
            Ok(Some(value)) => return Ok(value),
            Ok(None) => {}
            Err(err) => return Err(err),
        }
    }

    let gathered_value = {
        let proxy = Value::GpuTensor(handle.clone());
        gpu_helpers::gather_value(&proxy).map_err(|e| format!("{NAME}: {e}"))?
    };

    let det_result = determinant_from_value(gathered_value)?;
    match det_result {
        Determinant::Real(det) => {
            if let Some(provider) = runmat_accelerate_api::provider() {
                if let Ok(uploaded) = upload_scalar(provider, det) {
                    return Ok(Value::GpuTensor(uploaded));
                }
            }
            Ok(Value::Num(det))
        }
        Determinant::Complex(re, im) => Ok(Value::Complex(re, im)),
    }
}

fn det_real_value(tensor: Tensor) -> Result<Value, String> {
    Ok(Determinant::Real(det_real_tensor(&tensor)?).into_value())
}

fn det_complex_value(tensor: ComplexTensor) -> Result<Value, String> {
    let (re, im) = det_complex_tensor(&tensor)?;
    Ok(Determinant::Complex(re, im).into_value())
}

fn determinant_from_value(value: Value) -> Result<Determinant, String> {
    match value {
        Value::Num(n) => Ok(Determinant::Real(n)),
        Value::Tensor(tensor) => det_real_tensor(&tensor).map(Determinant::Real),
        Value::ComplexTensor(tensor) => {
            det_complex_tensor(&tensor).map(|(re, im)| Determinant::Complex(re, im))
        }
        Value::Complex(re, im) => Ok(Determinant::Complex(re, im)),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            det_real_tensor(&tensor).map(Determinant::Real)
        }
        Value::Int(int_value) => Ok(Determinant::Real(int_value.to_f64())),
        Value::Bool(flag) => Ok(Determinant::Real(if flag { 1.0 } else { 0.0 })),
        other => Err(format!(
            "{NAME}: unsupported input type {:?}; expected numeric or logical values",
            other
        )),
    }
}

fn det_real_tensor(matrix: &Tensor) -> Result<f64, String> {
    let (rows, cols) = matrix_dimensions(matrix.shape.as_slice())?;
    if rows != cols {
        return Err(format!("{NAME}: input must be a square matrix."));
    }
    if rows == 0 && cols == 0 {
        return Ok(1.0);
    }
    if matrix.data.len() == 1 {
        return Ok(matrix.data[0]);
    }
    let lu = LU::new(DMatrix::from_column_slice(rows, cols, &matrix.data));
    Ok(lu.determinant())
}

fn det_complex_tensor(matrix: &ComplexTensor) -> Result<(f64, f64), String> {
    let (rows, cols) = matrix_dimensions(matrix.shape.as_slice())?;
    if rows != cols {
        return Err(format!("{NAME}: input must be a square matrix."));
    }
    if rows == 0 && cols == 0 {
        return Ok((1.0, 0.0));
    }
    if matrix.data.len() == 1 {
        return Ok(matrix.data[0]);
    }
    let data: Vec<Complex64> = matrix
        .data
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let lu = LU::new(DMatrix::from_column_slice(rows, cols, &data));
    let det = lu.determinant();
    Ok((det.re, det.im))
}

fn matrix_dimensions(shape: &[usize]) -> Result<(usize, usize), String> {
    match shape {
        [] => Ok((1, 1)),
        [rows] => {
            if *rows == 1 {
                Ok((1, 1))
            } else {
                Err(format!("{NAME}: input must be a square matrix."))
            }
        }
        [rows, cols] => Ok((*rows, *cols)),
        _ => Err(format!("{NAME}: input must be a square matrix.")),
    }
}

fn upload_scalar(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    value: f64,
) -> Result<GpuTensorHandle, String> {
    let data = [value];
    let shape = [1usize, 1usize];
    let view = HostTensorView {
        data: &data,
        shape: &shape,
    };
    provider.upload(&view).map_err(|e| format!("{NAME}: {e}"))
}

fn det_gpu_via_provider(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    handle: &GpuTensorHandle,
) -> Result<Option<Value>, String> {
    let (rows, cols) = matrix_dimensions(handle.shape.as_slice())?;
    if rows != cols {
        return Err(format!("{NAME}: input must be a square matrix."));
    }
    if rows == 0 {
        let uploaded = upload_scalar(provider, 1.0)?;
        return Ok(Some(Value::GpuTensor(uploaded)));
    }

    let lu_result = match provider.lu(handle) {
        Ok(result) => result,
        Err(_) => return Ok(None),
    };

    let handles_to_free = [
        lu_result.combined.clone(),
        lu_result.lower.clone(),
        lu_result.upper.clone(),
        lu_result.perm_matrix.clone(),
        lu_result.perm_vector.clone(),
    ];

    let outcome = (|| -> Result<Option<Value>, String> {
        enum UpperFactor {
            Real(Tensor),
            Complex(ComplexTensor),
        }

        let upper_factor = match gpu_helpers::gather_tensor(&lu_result.upper) {
            Ok(tensor) => UpperFactor::Real(tensor),
            Err(_) => {
                let value = Value::GpuTensor(lu_result.upper.clone());
                match gpu_helpers::gather_value(&value) {
                    Ok(Value::Tensor(tensor)) => UpperFactor::Real(tensor),
                    Ok(Value::ComplexTensor(tensor)) => UpperFactor::Complex(tensor),
                    Ok(Value::Num(n)) => {
                        let tensor =
                            Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("{NAME}: {e}"))?;
                        UpperFactor::Real(tensor)
                    }
                    _ => return Ok(None),
                }
            }
        };

        let pivot_tensor = match gpu_helpers::gather_tensor(&lu_result.perm_vector) {
            Ok(tensor) => tensor,
            Err(_) => return Ok(None),
        };

        let determinant = match upper_factor {
            UpperFactor::Real(tensor) => match diagonal_product_real(&tensor, rows) {
                Ok(value) => Determinant::Real(value),
                Err(_) => return Ok(None),
            },
            UpperFactor::Complex(tensor) => match diagonal_product_complex(&tensor, rows) {
                Ok((re, im)) => Determinant::Complex(re, im),
                Err(_) => return Ok(None),
            },
        };

        let permutation_sign = match permutation_sign_from_tensor(&pivot_tensor, rows) {
            Ok(value) => value,
            Err(_) => return Ok(None),
        };

        let determinant = determinant.apply_sign(permutation_sign);

        match determinant {
            Determinant::Real(value) => {
                let uploaded = match upload_scalar(provider, value) {
                    Ok(handle) => handle,
                    Err(_) => return Ok(None),
                };
                Ok(Some(Value::GpuTensor(uploaded)))
            }
            Determinant::Complex(re, im) => Ok(Some(Value::Complex(re, im))),
        }
    })();

    for handle_to_free in &handles_to_free {
        let _ = provider.free(handle_to_free);
    }

    outcome
}

fn diagonal_product_real(upper: &Tensor, dimension: usize) -> Result<f64, String> {
    if dimension == 0 {
        return Ok(1.0);
    }
    let rows = upper.rows();
    let cols = upper.cols();
    if rows < dimension || cols < dimension {
        return Err(format!("{NAME}: upper factor shape mismatch"));
    }
    let mut product = 1.0f64;
    for i in 0..dimension {
        let idx = i + i * rows;
        let value = *upper
            .data
            .get(idx)
            .ok_or_else(|| format!("{NAME}: upper factor diagonal out of range"))?;
        product *= value;
    }
    Ok(product)
}

fn diagonal_product_complex(upper: &ComplexTensor, dimension: usize) -> Result<(f64, f64), String> {
    if dimension == 0 {
        return Ok((1.0, 0.0));
    }
    let rows = upper.rows;
    let cols = upper.cols;
    if rows < dimension || cols < dimension {
        return Err(format!("{NAME}: upper factor shape mismatch"));
    }
    let mut product = Complex64::new(1.0, 0.0);
    for i in 0..dimension {
        let idx = i + i * rows;
        let (re, im) = *upper
            .data
            .get(idx)
            .ok_or_else(|| format!("{NAME}: upper factor diagonal out of range"))?;
        product *= Complex64::new(re, im);
    }
    Ok((product.re, product.im))
}

fn permutation_sign_from_tensor(pivots: &Tensor, expected_len: usize) -> Result<f64, String> {
    if expected_len == 0 {
        return Ok(1.0);
    }
    if pivots.data.len() != expected_len {
        return Err(format!("{NAME}: pivot vector length mismatch"));
    }
    let len = pivots.data.len();
    let mut permutation = Vec::with_capacity(len);
    let mut seen = vec![false; len];
    for &raw in &pivots.data {
        if !raw.is_finite() {
            return Err(format!("{NAME}: pivot vector contains non-finite entries"));
        }
        let rounded = raw.round();
        if (rounded - raw).abs() > 1.0e-6 {
            return Err(format!("{NAME}: pivot vector must contain integer values"));
        }
        if rounded < 1.0 {
            return Err(format!("{NAME}: pivot vector index out of range"));
        }
        let idx = (rounded as isize - 1) as usize;
        if idx >= len {
            return Err(format!("{NAME}: pivot vector index out of range"));
        }
        if seen[idx] {
            return Err(format!("{NAME}: pivot vector must describe a permutation"));
        }
        seen[idx] = true;
        permutation.push(idx);
    }
    Ok(permutation_sign(&permutation))
}

fn permutation_sign(permutation: &[usize]) -> f64 {
    let mut visited = vec![false; permutation.len()];
    let mut sign = 1.0f64;
    for start in 0..permutation.len() {
        if visited[start] {
            continue;
        }
        let mut length = 0usize;
        let mut current = start;
        while current < permutation.len() && !visited[current] {
            visited[current] = true;
            current = permutation[current];
            length += 1;
        }
        if length > 0 && length.is_multiple_of(2) {
            sign = -sign;
        }
    }
    sign
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::LogicalArray;

    #[test]
    fn det_basic_2x2() {
        let tensor = Tensor::new(vec![4.0, 1.0, -2.0, 3.0], vec![2, 2]).unwrap();
        let result = det_real_value(tensor).expect("det");
        match result {
            Value::Num(v) => assert!((v - 14.0).abs() < 1e-12),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn det_non_square_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let err = det_real_value(tensor).unwrap_err();
        assert!(err.contains("det: input must be a square matrix."));
    }

    #[test]
    fn det_complex_matrix() {
        let data = vec![(1.0, 2.0), (0.0, 0.0), (0.0, 3.0), (4.0, -1.0)];
        let tensor = ComplexTensor::new(data, vec![2, 2]).unwrap();
        let value = det_complex_value(tensor).expect("det");
        match value {
            Value::Complex(re, im) => {
                assert!((re - 6.0).abs() < 1e-12);
                assert!((im - 7.0).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn det_complex_scalar_input() {
        let value = det_builtin(Value::Complex(3.0, -2.0)).expect("det");
        match value {
            Value::Complex(re, im) => {
                assert_eq!(re, 3.0);
                assert_eq!(im, -2.0);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[test]
    fn det_logical_matrix_promotes() {
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let result = det_builtin(Value::LogicalArray(logical)).expect("det");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn det_singular_returns_zero() {
        let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result = det_real_value(tensor).expect("det");
        match result {
            Value::Num(v) => assert!(v.abs() < 1e-12),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn det_empty_matrix_returns_one() {
        let tensor = Tensor::new(vec![], vec![0, 0]).unwrap();
        let result = det_real_value(tensor).expect("det");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn det_gpu_roundtrip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(
                vec![2.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0],
                vec![3, 3],
            )
            .unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu_value = det_builtin(Value::GpuTensor(handle)).expect("det");
            let gathered = test_support::gather(gpu_value).expect("gather");
            assert!(
                (gathered.data[0] - 24.0).abs() < 1e-12,
                "got {}",
                gathered.data[0]
            );
        });
    }

    #[test]
    fn det_gpu_permutation_sign() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu_value = det_builtin(Value::GpuTensor(handle)).expect("det");
            let gathered = test_support::gather(gpu_value).expect("gather");
            assert!(
                (gathered.data[0] + 1.0).abs() < 1e-12,
                "expected -1, got {}",
                gathered.data[0]
            );
        });
    }

    #[test]
    fn det_rejects_higher_rank_arrays() {
        let tensor = Tensor::new(vec![1.0; 8], vec![2, 2, 2]).unwrap();
        let err = det_real_value(tensor).unwrap_err();
        assert!(err.contains("det: input must be a square matrix."));
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn det_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(
            vec![3.0, 1.0, 0.0, 0.0, 5.0, 2.0, 4.0, 0.0, 6.0],
            vec![3, 3],
        )
        .unwrap();
        let cpu_det = det_real_tensor(&tensor).expect("cpu det");
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = det_builtin(Value::GpuTensor(handle)).expect("det");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![1, 1]);
        let det_gpu = gathered.data[0];
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1.0e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1.0e-5,
        };
        assert!(
            (det_gpu - cpu_det).abs() < tol,
            "gpu det {det_gpu} differs from cpu det {cpu_det}"
        );
    }
}
