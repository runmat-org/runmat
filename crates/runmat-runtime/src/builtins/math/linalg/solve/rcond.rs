//! MATLAB-compatible `rcond` builtin with GPU-aware fallbacks.

use nalgebra::{linalg::SVD, DMatrix};
use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::linalg::{matrix_dimensions_for, singular_value_rcond};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "rcond";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = NAME,
        builtin_path = "crate::builtins::math::linalg::solve::rcond"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "rcond"
category: "math/linalg/solve"
keywords: ["rcond", "reciprocal condition", "condition number", "linear algebra", "gpu"]
summary: "Estimate the reciprocal condition number of a square matrix (1 / cond(A)) with MATLAB-compatible behaviour."
references: ["https://www.mathworks.com/help/matlab/ref/rcond.html"]
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "When a provider exposes a dense solver hook it may reuse that path; the shipping backends gather to the host, run the shared SVD-based estimator, and optionally re-upload the scalar."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::linalg::solve::rcond::tests"
  gpu: "builtins::math::linalg::solve::rcond::tests::rcond_gpu_round_trip_matches_cpu"
  wgpu: "builtins::math::linalg::solve::rcond::tests::rcond_wgpu_matches_cpu"
  doc: "builtins::math::linalg::solve::rcond::tests::doc_examples_present"
---

# What does the `rcond` function do in MATLAB / RunMat?
`rc = rcond(A)` returns an estimate of the reciprocal of the condition number of a square matrix `A`
in the 1-norm. Well-conditioned matrices yield values close to `1`, whereas singular or numerically
singular matrices yield values near zero. The estimate mirrors MATLAB: it is based on the ratio of
the smallest to largest singular values.

## How does the `rcond` function behave in MATLAB / RunMat?
- Inputs must be square matrices (scalars count as `1 × 1`). Non-square inputs trigger the MATLAB
  error `"rcond: input must be a square matrix."`
- For the empty matrix (`0 × 0`) the reciprocal condition number is `Inf`.
- `rcond(0)` returns `0`. Non-zero scalars return `1`.
- Logical and integer arrays are promoted to double precision before analysis.
- Complex matrices are supported; the singular values are computed in complex arithmetic.
- The implementation runs an SVD and computes `min(s) / max(s)`, honouring MATLAB's tolerance logic.

## `rcond` function GPU execution behaviour
When the input resides on a GPU, RunMat first invokes the active provider's `rcond` hook that this
builtin registers. When that hook is unavailable the runtime attempts to reuse the provider's
`linsolve` factorisations to recover the reciprocal condition number. If neither path is implemented
the matrix is gathered to the host, the shared SVD-based estimator is executed, and the scalar
result is uploaded back to the device so subsequent GPU work keeps residency.

## Examples of using the `rcond` function in MATLAB / RunMat

### Reciprocal condition of the identity matrix
```matlab
A = eye(3);
rc = rcond(A);
```
Expected output:
```matlab
rc = 1
```

### Detecting a nearly singular matrix
```matlab
A = diag([1, 1e-8]);
rc = rcond(A);
```
Expected output:
```matlab
rc = 1e-8
```

### Reciprocal condition of a singular matrix
```matlab
A = [1 2; 2 4];
rc = rcond(A);
```
Expected output:
```matlab
rc = 0
```

### Reciprocal condition for a complex matrix
```matlab
A = [1+2i 0; 3i 2-1i];
rc = rcond(A);
```
Expected output (values rounded):
```matlab
rc = 0.1887
```

### Handling the empty matrix
```matlab
rc = rcond([]);
```
Expected output:
```matlab
rc = Inf
```

### Estimating conditioning on GPU data
```matlab
G = gpuArray([2 0; 0 0.5]);
rc_gpu = rcond(G);
rc = gather(rc_gpu);
```
Expected output:
```matlab
rc = 0.25
```

## GPU residency in RunMat (Do I need `gpuArray`?)
Manually calling `gpuArray` is rarely necessary. When matrices already live on the device, RunMat
invokes the provider hook if one exists, or automatically gathers, evaluates, and re-uploads the
scalar result. This preserves MATLAB compatibility while keeping the runtime efficient and ergonomic.

## FAQ

### What range of values should I expect from `rcond`?
`rcond` returns a value in `[0, 1]`. Values near `1` indicate a well-conditioned matrix. Values near
`0` indicate singular or ill-conditioned matrices.

### Why does `rcond` return `Inf` for the empty matrix?
The empty matrix is perfectly conditioned by convention, so the reciprocal condition number is `Inf`
(`1 / 0`).

### Does `rcond` warn me about singular systems automatically?
`rcond` itself does not issue warnings. Use the value to decide whether the matrix is singular to
working precision, similar to MATLAB workflows (`if rcond(A) < eps, ...`).

### Is the SVD tolerance the same as MATLAB?
Yes. The estimator matches MATLAB's definition by deriving the result from the singular values. The
same tolerance logic is shared with `pinv` and `rank`.

### Will calling `rcond` move my data off the GPU?
Only if the active provider lacks a dedicated implementation. In that case RunMat transparently
downloads the matrix, computes the estimate on the host, and re-uploads the scalar.

## See also
[inv](./inv), [pinv](./pinv), [rank](./rank), [det](./det), [linsolve](./linsolve), [gpuArray](./gpuarray), [gather](./gather)

## Source & feedback
- The implementation lives at `crates/runmat-runtime/src/builtins/math/linalg/solve/rcond.rs`.
- Found a behavioural difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose)
  with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::solve::rcond")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("rcond"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("rcond")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may reuse dense solver factorizations to expose rcond; current backends gather to the host and re-upload a scalar value when possible.",
};

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(NAME).build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    if err.message() == "interaction pending..." {
        return build_runtime_error("interaction pending...")
            .with_builtin(NAME)
            .build();
    }
    let mut builder = build_runtime_error(err.message()).with_builtin(NAME);
    if let Some(identifier) = err.identifier() {
        builder = builder.with_identifier(identifier.to_string());
    }
    if let Some(task_id) = err.context.task_id.clone() {
        builder = builder.with_task_id(task_id);
    }
    if !err.context.call_stack.is_empty() {
        builder = builder.with_call_stack(err.context.call_stack.clone());
    }
    if let Some(phase) = err.context.phase.clone() {
        builder = builder.with_phase(phase);
    }
    builder.with_source(err).build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::solve::rcond")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not fusible; rcond consumes an entire matrix and returns a scalar estimate.",
};

#[runtime_builtin(
    name = "rcond",
    category = "math/linalg/solve",
    summary = "Estimate the reciprocal condition number of a square matrix.",
    keywords = "rcond,condition number,reciprocal,gpu",
    accel = "rcond",
    builtin_path = "crate::builtins::math::linalg::solve::rcond"
)]
fn rcond_builtin(value: Value) -> BuiltinResult<Value> {
    let estimate = match value {
        Value::GpuTensor(handle) => return rcond_gpu(handle),
        Value::ComplexTensor(matrix) => rcond_complex_tensor(&matrix)?,
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(builtin_error)?;
            rcond_complex_tensor(&tensor)?
        }
        other => {
            let matrix = tensor::value_into_tensor_for(NAME, other).map_err(builtin_error)?;
            rcond_real_tensor(&matrix)?
        }
    };
    Ok(Value::Num(estimate))
}

fn rcond_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let (rows, cols) = matrix_dimensions_for(NAME, &handle.shape).map_err(builtin_error)?;
    if rows != cols {
        return Err(builtin_error(format!("{NAME}: input must be a square matrix.")));
    }

    if rows == 0 {
        if let Some(provider) = runmat_accelerate_api::provider() {
            match upload_scalar(provider, f64::INFINITY) {
                Ok(uploaded) => return Ok(Value::GpuTensor(uploaded)),
                Err(err) => {
                    if err.message() == "interaction pending..." {
                        return Err(build_runtime_error("interaction pending...")
                            .with_builtin(NAME)
                            .build());
                    }
                }
            }
        }
        return Ok(Value::Num(f64::INFINITY));
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        match provider.rcond(&handle) {
            Ok(result) => return Ok(Value::GpuTensor(result)),
            Err(_) => match rcond_gpu_via_linsolve(provider, &handle, rows) {
                Ok(Some(value)) => return Ok(value),
                Ok(None) => {}
                Err(err) => {
                    if err.message() == "interaction pending..." {
                        return Err(build_runtime_error("interaction pending...")
                            .with_builtin(NAME)
                            .build());
                    }
                    return Err(err);
                }
            },
        }
    }

    let gathered =
        gpu_helpers::gather_value(&Value::GpuTensor(handle.clone())).map_err(map_control_flow)?;
    let estimate = match gathered {
        Value::Tensor(tensor) => rcond_real_tensor(&tensor)?,
        Value::ComplexTensor(tensor) => rcond_complex_tensor(&tensor)?,
        Value::Num(n) => {
            if n == 0.0 {
                0.0
            } else {
                1.0
            }
        }
        Value::Complex(re, im) => {
            if re == 0.0 && im == 0.0 {
                0.0
            } else {
                1.0
            }
        }
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other).map_err(builtin_error)?;
            rcond_real_tensor(&tensor)?
        }
    };

    if let Some(provider) = runmat_accelerate_api::provider() {
        match upload_scalar(provider, estimate) {
            Ok(uploaded) => return Ok(Value::GpuTensor(uploaded)),
            Err(err) => {
                if err.message() == "interaction pending..." {
                    return Err(build_runtime_error("interaction pending...")
                        .with_builtin(NAME)
                        .build());
                }
            }
        }
    }

    Ok(Value::Num(estimate))
}

fn rcond_gpu_via_linsolve(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    handle: &GpuTensorHandle,
    order: usize,
) -> BuiltinResult<Option<Value>> {
    if order == 0 {
        return upload_scalar(provider, f64::INFINITY).map(|gpu| Some(Value::GpuTensor(gpu)));
    }

    // Attempt to reuse provider linsolve to retrieve an rcond estimate.
    let identity = match upload_identity(provider, order) {
        Ok(id) => id,
        Err(err) => {
            if err.message() == "interaction pending..." {
                return Err(build_runtime_error("interaction pending...")
                    .with_builtin(NAME)
                    .build());
            }
            return Ok(None);
        }
    };

    let options = runmat_accelerate_api::ProviderLinsolveOptions {
        rcond: None,
        ..Default::default()
    };
    let outcome = provider.linsolve(handle, &identity, &options);

    let _ = provider.free(&identity);

    let result = match outcome {
        Ok(res) => res,
        Err(_) => return Ok(None),
    };

    let rcond_value = result.reciprocal_condition;
    let _ = provider.free(&result.solution);

    match upload_scalar(provider, rcond_value) {
        Ok(uploaded) => return Ok(Some(Value::GpuTensor(uploaded))),
        Err(err) => {
            if err.message() == "interaction pending..." {
                return Err(build_runtime_error("interaction pending...")
                    .with_builtin(NAME)
                    .build());
            }
        }
    }

    Ok(Some(Value::Num(rcond_value)))
}

fn rcond_real_tensor(matrix: &Tensor) -> BuiltinResult<f64> {
    rcond_real_tensor_impl(matrix)
}

fn rcond_complex_tensor(matrix: &ComplexTensor) -> BuiltinResult<f64> {
    rcond_complex_tensor_impl(matrix)
}

fn rcond_real_tensor_impl(matrix: &Tensor) -> BuiltinResult<f64> {
    let (rows, cols) = matrix_dimensions_for(NAME, &matrix.shape).map_err(builtin_error)?;
    if rows != cols {
        return Err(builtin_error(format!(
            "{NAME}: input must be a square matrix."
        )));
    }
    if rows == 0 {
        return Ok(f64::INFINITY);
    }
    if matrix.data.len() == 1 {
        return Ok(if matrix.data[0] == 0.0 { 0.0 } else { 1.0 });
    }
    let a = DMatrix::from_column_slice(rows, cols, &matrix.data);
    let svd = SVD::new(a, false, false);
    Ok(singular_value_rcond(svd.singular_values.as_slice()))
}

fn rcond_complex_tensor_impl(matrix: &ComplexTensor) -> BuiltinResult<f64> {
    let (rows, cols) = matrix_dimensions_for(NAME, &matrix.shape).map_err(builtin_error)?;
    if rows != cols {
        return Err(builtin_error(format!(
            "{NAME}: input must be a square matrix."
        )));
    }
    if rows == 0 {
        return Ok(f64::INFINITY);
    }
    if matrix.data.len() == 1 {
        let (re, im) = matrix.data[0];
        let magnitude = re.hypot(im);
        return Ok(if magnitude == 0.0 { 0.0 } else { 1.0 });
    }
    let data: Vec<Complex64> = matrix
        .data
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let a = DMatrix::from_column_slice(rows, cols, &data);
    let svd = SVD::new(a, false, false);
    Ok(singular_value_rcond(svd.singular_values.as_slice()))
}

fn upload_scalar(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    value: f64,
) -> BuiltinResult<GpuTensorHandle> {
    let data = [value];
    let shape = [1usize, 1usize];
    let view = HostTensorView {
        data: &data,
        shape: &shape,
    };
    provider
        .upload(&view)
        .map_err(|e| builtin_error(format!("{NAME}: {e}")))
}

fn upload_identity(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    n: usize,
) -> BuiltinResult<GpuTensorHandle> {
    if n == 0 {
        return upload_scalar(provider, 0.0);
    }
    let mut data = vec![0.0_f64; n * n];
    for i in 0..n {
        data[i + i * n] = 1.0;
    }
    let shape = vec![n, n];
    let view = HostTensorView {
        data: &data,
        shape: &shape,
    };
    provider
        .upload(&view)
        .map_err(|e| builtin_error(format!("{NAME}: {e}")))
}

/// Host-accessible helper for acceleration providers that gather matrices.
pub fn rcond_host_real_for_provider(matrix: &Tensor) -> BuiltinResult<f64> {
    rcond_real_tensor_impl(matrix)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
        err
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rcond_identity_is_one() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let result = rcond_builtin(Value::Tensor(tensor)).expect("rcond");
        match result {
            Value::Num(value) => assert!((value - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rcond_zero_is_zero() {
        let tensor = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let result = rcond_builtin(Value::Tensor(tensor)).expect("rcond");
        match result {
            Value::Num(value) => assert_eq!(value, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rcond_nearly_singular() {
        let tensor =
            Tensor::new(vec![1.0, 0.0, 0.0, 1.0e-8], vec![2, 2]).expect("tensor construction");
        let result = rcond_builtin(Value::Tensor(tensor)).expect("rcond");
        match result {
            Value::Num(value) => assert!((value - 1.0e-8).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rcond_singular_matrix_zero() {
        let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result = rcond_builtin(Value::Tensor(tensor)).expect("rcond");
        match result {
            Value::Num(value) => assert_eq!(value, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rcond_complex_matrix_supported() {
        let data = vec![(1.0, 2.0), (0.0, 0.0), (0.0, 3.0), (2.0, -1.0)];
        let tensor = ComplexTensor::new(data, vec![2, 2]).unwrap();
        let result = rcond_builtin(Value::ComplexTensor(tensor)).expect("rcond");
        match result {
            Value::Num(value) => {
                assert!(value > 0.0 && value <= 1.0, "unexpected rcond {value}")
            }
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rcond_rejects_non_square() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let err = unwrap_error(rcond_builtin(Value::Tensor(tensor)).unwrap_err());
        assert_eq!(err.message(), "rcond: input must be a square matrix.");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rcond_handles_empty_matrix() {
        let tensor = Tensor::new(vec![], vec![0, 0]).unwrap();
        let result = rcond_builtin(Value::Tensor(tensor)).expect("rcond");
        match result {
            Value::Num(value) => assert!(value.is_infinite() && value.is_sign_positive()),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rcond_accepts_scalar_int() {
        let int_value = Value::Int(IntValue::I32(5));
        let result = rcond_builtin(int_value).expect("rcond");
        match result {
            Value::Num(value) => assert!((value - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rcond_gpu_round_trip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![2.0, 0.0, 0.0, 0.5], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu_value = rcond_builtin(Value::GpuTensor(handle)).expect("rcond");
            let gathered = test_support::gather(gpu_value).expect("gather");
            assert_eq!(gathered.shape, vec![1, 1]);
            assert!((gathered.data[0] - 0.25).abs() < 1e-12);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn rcond_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tol = match runmat_accelerate_api::provider()
            .expect("wgpu provider")
            .precision()
        {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        let tensor = Tensor::new(vec![3.0, 0.0, 0.0, 0.25], vec![2, 2]).unwrap();
        let cpu_value = rcond_builtin(Value::Tensor(tensor.clone())).expect("cpu rcond");
        let cpu_scalar = match cpu_value {
            Value::Num(v) => v,
            other => panic!("expected scalar CPU value, got {other:?}"),
        };

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");

        let gpu_value = rcond_builtin(Value::GpuTensor(handle)).expect("gpu rcond");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape, vec![1, 1]);
        assert!((gathered.data[0] - cpu_scalar).abs() < tol);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let snippets = test_support::doc_examples(DOC_MD);
        assert!(!snippets.is_empty());
    }
}
