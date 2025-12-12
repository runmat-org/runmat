//! MATLAB-compatible `inv` builtin with GPU-aware fallbacks.

use nalgebra::DMatrix;
use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, ProviderInvOptions};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

const NAME: &str = "inv";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = NAME,
        builtin_path = "crate::builtins::math::linalg::solve::inv"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "inv"
category: "math/linalg/solve"
keywords: ["inv", "matrix inverse", "linear algebra", "solve", "gpu"]
summary: "Compute the inverse of a square matrix with MATLAB-compatible pivoting and GPU fallbacks."
references: ["https://www.mathworks.com/help/matlab/ref/inv.html"]
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Uses the acceleration provider's `inv` hook when available; the default WGPU backend gathers to the host, computes the inverse, and re-uploads the result to preserve residency."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "uniform"
requires_feature: null
tested:
  unit: "builtins::math::linalg::solve::inv::tests"
  gpu: "builtins::math::linalg::solve::inv::tests::inv_gpu_round_trip_matches_cpu"
  wgpu: "builtins::math::linalg::solve::inv::tests::inv_wgpu_matches_cpu"
  doc: "builtins::math::linalg::solve::inv::tests::doc_examples_present"
---

# What does the `inv` function do in MATLAB / RunMat?
`X = inv(A)` returns the matrix inverse of a square, full-rank matrix `A`. The result satisfies
`A * X = eye(size(A))` within round-off. Scalars behave like `1 ./ A`, matching MATLAB semantics.

## How does the `inv` function behave in MATLAB / RunMat?
- Inputs must be 2-D matrices (trailing singleton dimensions are accepted). Non-square matrices raise
  the MATLAB error `"inv: input must be a square matrix."`
- Singular or rank-deficient matrices raise `"inv: matrix is singular to working precision."`
- Logical and integer inputs are promoted to double precision before inversion.
- Complex inputs are handled in full complex arithmetic.
- Empty matrices return an empty matrix with the same dimensions (e.g., `inv([])` yields `[]`).

## `inv` GPU execution behaviour
When a GPU acceleration provider is active, RunMat forwards the operation to its `inv` hook. If the
provider does not implement a native kernel, RunMat gathers the data to the host, uses the shared CPU
implementation, and attempts to re-upload the result so downstream GPU work keeps its residency. The
shipping WGPU backend currently follows this gather/compute/upload pattern.

## Examples of using the `inv` function in MATLAB / RunMat

### Inverting a 2x2 matrix for solving linear systems
```matlab
A = [4 -2; 1 3];
X = inv(A);
```
Expected output:
```matlab
X =
    0.3    0.2
   -0.1    0.4
```

### Checking that `inv(A)` produces the identity matrix
```matlab
A = [2 1 0; 0 1 -1; 0 0 3];
X = inv(A);
product = A * X;
```
Expected output:
```matlab
product =
    1.0000         0         0
         0    1.0000         0
         0         0    1.0000
```

### Inverting a diagonal matrix with symbolic structure
```matlab
D = diag([2, 5, 10]);
X = inv(D);
```
Expected output:
```matlab
X =
    0.5000         0         0
         0    0.2000         0
         0         0    0.1000
```

### Computing the inverse of a complex matrix
```matlab
A = [1+2i  0; 3i  4-1i];
X = inv(A);
```
Expected output:
```matlab
X =
   0.2105 - 0.1053i  -0.0158 - 0.1579i
  -0.1579 - 0.1184i   0.0526 + 0.2632i
```

### Using `inv` on a GPU-resident matrix
```matlab
G = gpuArray([3 1; 0 2]);
invG = inv(G);       % stays on the GPU when the provider implements inv
result = gather(invG);
```
Expected output:
```matlab
result =
    0.3333   -0.1667
         0    0.5000
```

### Handling singular matrices gracefully
```matlab
A = [1 2; 2 4];
X = inv(A);
```
Expected output:
```
Error using inv
inv: matrix is singular to working precision.
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You typically do not need to move data manually. If `A` already resides on the GPU and the provider
implements `inv`, the computation stays on the device. Providers without a native kernel (including
the current WGPU backend) download `A`, compute the inverse on the host, and re-upload the result, so
subsequent GPU code continues to operate on device-resident data. `gpuArray` remains available for
compatibility and for explicitly seeding GPU residency.

## FAQ

### Do I need to use `inv` to solve linear systems?
Prefer `mldivide` (`A \\ b`) or `linsolve` for numerical stability and performance. Use `inv` only
when you explicitly need the inverse matrix.

### What error do I get for singular matrices?
RunMat mirrors MATLAB and raises `"inv: matrix is singular to working precision."` when LU
factorisation detects a zero pivot.

### Can I invert non-square matrices?
No. `inv` requires square matrices. Use `pinv` for pseudoinverses of rectangular matrices.

### Does `inv` support complex numbers?
Yes. Complex matrices are inverted using full complex arithmetic.

### What happens with empty matrices?
`inv([])` returns `[]` (an empty matrix) without error.

### Does `inv` preserve GPU residency?
If the acceleration provider exposes an `inv` hook, the operation stays on the GPU. Otherwise, RunMat
gathers, computes on the host, and re-uploads so the caller still receives a GPU tensor.

## See Also
[pinv](./pinv), [linsolve](./linsolve), [mldivide](../ops/mldivide), [det](../ops/det), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::solve::inv")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("inv"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("inv")],
    constant_strategy: ConstantStrategy::UniformBuffer,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may implement a native inverse; the reference WGPU backend gathers to the host implementation and re-uploads the result.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::solve::inv")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::UniformBuffer,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Matrix inversion is a terminal operation and does not participate in fusion pipelines.",
};

#[runtime_builtin(
    name = "inv",
    category = "math/linalg/solve",
    summary = "Compute the inverse of a square matrix.",
    keywords = "inv,matrix inverse,linear solve,gpu",
    accel = "inv",
    builtin_path = "crate::builtins::math::linalg::solve::inv"
)]
fn inv_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => inv_gpu(handle),
        Value::ComplexTensor(tensor) => inv_complex_value(tensor),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| format!("{NAME}: {e}"))?;
            inv_complex_value(tensor)
        }
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other)?;
            inv_real_value(tensor)
        }
    }
}

fn inv_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let options = ProviderInvOptions::default();
        match provider.inv(&handle, options) {
            Ok(result) => return Ok(Value::GpuTensor(result)),
            Err(_) => {
                // Fall back to host implementation and attempt to re-upload.
            }
        }
        let gathered = gpu_helpers::gather_tensor(&handle)?;
        let inv = inv_real_tensor(&gathered)?;
        if let Ok(uploaded) = provider.upload(&runmat_accelerate_api::HostTensorView {
            data: &inv.data,
            shape: &inv.shape,
        }) {
            return Ok(Value::GpuTensor(uploaded));
        }
        return Ok(tensor::tensor_into_value(inv));
    }

    let gathered = gpu_helpers::gather_tensor(&handle)?;
    let inv = inv_real_tensor(&gathered)?;
    Ok(tensor::tensor_into_value(inv))
}

fn inv_real_value(tensor: Tensor) -> Result<Value, String> {
    let inv = inv_real_tensor(&tensor)?;
    Ok(tensor::tensor_into_value(inv))
}

fn inv_complex_value(tensor: ComplexTensor) -> Result<Value, String> {
    let inv = inv_complex_tensor(&tensor)?;
    if inv.data.len() == 1 {
        let (re, im) = inv.data[0];
        Ok(Value::Complex(re, im))
    } else {
        Ok(Value::ComplexTensor(inv))
    }
}

fn inv_real_tensor(matrix: &Tensor) -> Result<Tensor, String> {
    let (rows, cols) = matrix_dimensions(matrix.shape.as_slice())?;
    if rows == 0 && cols == 0 {
        return Tensor::new(Vec::new(), matrix.shape.clone()).map_err(|e| format!("{NAME}: {e}"));
    }
    if rows != cols {
        return Err(format!("{NAME}: input must be a square matrix."));
    }
    if rows == 0 || cols == 0 {
        return Tensor::new(Vec::new(), matrix.shape.clone()).map_err(|e| format!("{NAME}: {e}"));
    }
    let dm = DMatrix::from_column_slice(rows, cols, &matrix.data);
    let inverse = dm
        .try_inverse()
        .ok_or_else(|| format!("{NAME}: matrix is singular to working precision."))?;
    matrix_to_tensor(NAME, inverse, &matrix.shape)
}

fn inv_complex_tensor(matrix: &ComplexTensor) -> Result<ComplexTensor, String> {
    let (rows, cols) = matrix_dimensions(matrix.shape.as_slice())?;
    if rows == 0 && cols == 0 {
        return ComplexTensor::new(Vec::new(), matrix.shape.clone())
            .map_err(|e| format!("{NAME}: {e}"));
    }
    if rows != cols {
        return Err(format!("{NAME}: input must be a square matrix."));
    }
    if rows == 0 || cols == 0 {
        return ComplexTensor::new(Vec::new(), matrix.shape.clone())
            .map_err(|e| format!("{NAME}: {e}"));
    }
    let data: Vec<Complex64> = matrix
        .data
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let dm = DMatrix::from_column_slice(rows, cols, &data);
    let inverse = dm
        .try_inverse()
        .ok_or_else(|| format!("{NAME}: matrix is singular to working precision."))?;
    matrix_to_complex_tensor(NAME, inverse, &matrix.shape)
}

fn matrix_dimensions(shape: &[usize]) -> Result<(usize, usize), String> {
    match shape.len() {
        0 => Ok((1, 1)),
        1 => {
            if shape[0] == 1 {
                Ok((1, 1))
            } else {
                Err(format!("{NAME}: input must be a square matrix."))
            }
        }
        _ => {
            if shape.len() > 2 && shape.iter().skip(2).any(|&dim| dim != 1) {
                Err(format!("{NAME}: inputs must be 2-D matrices."))
            } else {
                Ok((shape[0], shape[1]))
            }
        }
    }
}

fn matrix_to_tensor(label: &str, matrix: DMatrix<f64>, shape: &[usize]) -> Result<Tensor, String> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    debug_assert_eq!(rows * cols, matrix.len());
    Tensor::new(matrix.as_slice().to_vec(), shape.to_vec()).map_err(|e| format!("{label}: {e}"))
}

fn matrix_to_complex_tensor(
    label: &str,
    matrix: DMatrix<Complex64>,
    shape: &[usize],
) -> Result<ComplexTensor, String> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    let data: Vec<(f64, f64)> = matrix.as_slice().iter().map(|c| (c.re, c.im)).collect();
    debug_assert_eq!(rows * cols, matrix.len());
    ComplexTensor::new(data, shape.to_vec()).map_err(|e| format!("{label}: {e}"))
}

/// Host helper used by acceleration providers that delegate `inv` back to the CPU path.
pub fn inv_host_real_for_provider(matrix: &Tensor) -> Result<Tensor, String> {
    inv_real_tensor(matrix)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use nalgebra::DMatrix;
    use num_complex::Complex64;
    use runmat_builtins::{IntValue, Tensor, Value};

    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider::{self, WgpuProviderOptions};

    #[test]
    fn inv_scalar_num() {
        let result = inv_builtin(Value::Num(4.0)).expect("inv");
        match result {
            Value::Num(v) => assert!((v - 0.25).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn inv_square_matrix() {
        let data = vec![4.0, 1.0, -2.0, 3.0];
        let tensor = Tensor::new(data.clone(), vec![2, 2]).unwrap();
        let result = inv_builtin(Value::Tensor(tensor)).expect("inv");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                let a = DMatrix::from_column_slice(2, 2, &data);
                let inv_m = DMatrix::from_column_slice(2, 2, &out.data);
                let identity = &a * &inv_m;
                for r in 0..2 {
                    for c in 0..2 {
                        let expected = if r == c { 1.0 } else { 0.0 };
                        assert!((identity[(r, c)] - expected).abs() < 1e-12);
                    }
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn inv_empty_matrix_returns_empty() {
        let tensor = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let result = inv_builtin(Value::Tensor(tensor.clone())).expect("inv");
        match result {
            Value::Tensor(out) => {
                assert!(out.data.is_empty());
                assert_eq!(out.shape, vec![0, 0]);
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[test]
    fn inv_trailing_singleton_dimension_preserved() {
        let tensor =
            Tensor::new(vec![4.0, 0.0, 0.0, 2.0], vec![2, 2, 1]).expect("tensor construction");
        let result = inv_builtin(Value::Tensor(tensor)).expect("inv");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2, 1]);
                let expected = vec![0.25, 0.0, 0.0, 0.5];
                assert_eq!(out.data, expected);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn inv_complex_scalar() {
        let result = inv_builtin(Value::Complex(2.0, -1.0)).expect("inv");
        match result {
            Value::Complex(re, im) => {
                let expected = Complex64::new(1.0, 0.0) / Complex64::new(2.0, -1.0);
                assert!((re - expected.re).abs() < 1e-12);
                assert!((im - expected.im).abs() < 1e-12);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[test]
    fn inv_complex_matrix() {
        let raw = vec![(1.0, 2.0), (0.0, 3.0), (0.0, 0.0), (4.0, -1.0)];
        let tensor = ComplexTensor::new(raw.clone(), vec![2, 2]).unwrap();
        let result = inv_builtin(Value::ComplexTensor(tensor)).expect("inv");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                let input: Vec<Complex64> =
                    raw.iter().map(|&(re, im)| Complex64::new(re, im)).collect();
                let inv_vec: Vec<Complex64> = out
                    .data
                    .iter()
                    .map(|&(re, im)| Complex64::new(re, im))
                    .collect();
                let a = DMatrix::from_column_slice(2, 2, &input);
                let inv_m = DMatrix::from_column_slice(2, 2, &inv_vec);
                let identity = &a * &inv_m;
                for r in 0..2 {
                    for c in 0..2 {
                        let expected = if r == c {
                            Complex64::new(1.0, 0.0)
                        } else {
                            Complex64::new(0.0, 0.0)
                        };
                        let delta = identity[(r, c)] - expected;
                        assert!(delta.norm() < 1e-10, "identity mismatch at ({r},{c})");
                    }
                }
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[test]
    fn inv_rejects_higher_rank_tensor() {
        let tensor = Tensor::new(vec![1.0; 8], vec![2, 2, 2]).unwrap();
        let err = inv_builtin(Value::Tensor(tensor)).unwrap_err();
        assert!(err.contains("2-D"), "{err}");
    }

    #[test]
    fn inv_non_square_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let err = inv_builtin(Value::Tensor(tensor)).unwrap_err();
        assert!(err.contains("square matrix"), "{err}");
    }

    #[test]
    fn inv_singular_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let err = inv_builtin(Value::Tensor(tensor)).unwrap_err();
        assert!(err.contains("singular"), "{err}");
    }

    #[test]
    fn inv_gpu_round_trip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0, 0.0, 1.0, 2.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu_value = inv_builtin(Value::GpuTensor(handle)).expect("gpu inv");
            let gathered = test_support::gather(gpu_value).expect("gather");
            let cpu = inv_real_tensor(&tensor).expect("cpu");
            assert_eq!(gathered.shape, cpu.shape);
            for (a, b) in gathered.data.iter().zip(cpu.data.iter()) {
                assert!((a - b).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn inv_scalar_int_promotes() {
        let result = inv_builtin(Value::Int(IntValue::I32(2))).expect("inv");
        match result {
            Value::Num(v) => assert!((v - 0.5).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn inv_wgpu_matches_cpu() {
        let _ = provider::register_wgpu_provider(WgpuProviderOptions::default());

        let tensor = Tensor::new(vec![4.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let cpu = inv_real_tensor(&tensor).expect("cpu");

        let provider = runmat_accelerate_api::provider().expect("provider");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");

        let gpu_value = inv_builtin(Value::GpuTensor(handle)).expect("gpu inv");
        let gathered = test_support::gather(gpu_value).expect("gather");

        assert_eq!(gathered.shape, cpu.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (a, b) in gathered.data.iter().zip(cpu.data.iter()) {
            assert!((*a - *b).abs() < tol, "expected {b}, got {a}");
        }
    }
}
