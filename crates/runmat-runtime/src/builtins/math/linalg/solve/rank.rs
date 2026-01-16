//! MATLAB-compatible `rank` builtin that counts singular values above a tolerance.

use nalgebra::{linalg::SVD, DMatrix};
use num_complex::Complex64;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::linalg::{
    matrix_dimensions_for, parse_tolerance_arg, svd_default_tolerance,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeControlFlow};

const NAME: &str = "rank";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = NAME,
        builtin_path = "crate::builtins::math::linalg::solve::rank"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "rank"
category: "math/linalg/solve"
keywords: ["rank", "singular value decomposition", "tolerance", "matrix rank", "gpu"]
summary: "Compute the numerical rank of a matrix using SVD with MATLAB-compatible tolerance handling."
references: ["https://www.mathworks.com/help/matlab/ref/rank.html"]
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Providers may implement a dedicated rank hook; current backends gather to the host and reuse the shared SVD path."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::linalg::solve::rank::tests"
  gpu: "builtins::math::linalg::solve::rank::tests::rank_gpu_round_trip"
  doc: "builtins::math::linalg::solve::rank::tests::doc_examples_present"
  wgpu: "builtins::math::linalg::solve::rank::tests::rank_wgpu_matches_cpu"
---

# What does the `rank` function do in MATLAB / RunMat?
`r = rank(A)` returns the numerical rank of a real or complex matrix `A`. The rank equals the
number of singular values greater than a tolerance derived from the matrix size and the largest
singular value. RunMat mirrors MATLAB’s logic exactly so that results agree bit-for-bit with the
reference implementation.

## How does the `rank` function behave in MATLAB / RunMat?
- Inputs must behave like 2-D matrices. Trailing singleton dimensions are accepted; other higher
  ranks result in `"rank: inputs must be 2-D matrices or vectors"`.
- The default tolerance is `tol = max(size(A)) * eps(max(s))`, where `s` are the singular values
  from an SVD of `A`. You can override this by supplying a second argument: `rank(A, tol)`.
- When you provide an explicit tolerance it must be a finite, non-negative scalar. Non-scalars,
  `NaN`, `Inf`, or negative values raise MATLAB-compatible errors.
- Logical and integer inputs are promoted to double precision before taking the SVD.
- `rank([])` returns `0`. Rank is always reported as a double scalar (e.g., `2.0`).
- Complex inputs use a complex SVD so that conjugate transposes and magnitudes follow MATLAB’s
  conventions.

## `rank` Function GPU Execution Behaviour
When a GPU acceleration provider is active, RunMat first offers the computation through the
reserved `rank` provider hook. Backends that implement it can stay fully on-device and return a
`gpuTensor` scalar. Providers without that hook—including today’s WGPU backend—gather the matrix to
host memory, reuse the shared SVD logic, and then re-upload the scalar rank so downstream kernels
continue on the GPU without user intervention. Auto-offload treats the builtin as an eager sink, so
any fused producers flush before `rank` executes and residency bookkeeping remains consistent.

## Examples of using the `rank` function in MATLAB / RunMat

### Determining the rank of a full matrix
```matlab
A = [1 2; 3 4];
rk = rank(A);
```
Expected output:
```matlab
rk = 2
```

### Detecting rank deficiency in a singular matrix
```matlab
B = [1 2; 2 4];
rk = rank(B);
```
Expected output:
```matlab
rk = 1
```

### Applying a custom tolerance to suppress tiny singular values
```matlab
C = diag([1, 1e-12]);
rk_default = rank(C);          % counts both singular values (rank 2)
rk_custom  = rank(C, 1e-9);    % treats the small value as zero (rank 1)
```

### Computing the rank of a tall matrix
```matlab
A = [1 0; 0 0; 0 1];
rk = rank(A);
```
Expected output:
```matlab
rk = 2
```

### Evaluating the rank of a complex matrix
```matlab
Z = [1+1i 0; 0 2-3i];
rk = rank(Z);
```
Expected output:
```matlab
rk = 2
```

### Checking the rank of an empty matrix
```matlab
E = [];
rk = rank(E);
```
Expected output:
```matlab
rk = 0
```

### Using `rank` with `gpuArray` data
```matlab
G = gpuArray([1 2 3; 3 6 9; 0 1 0]);
rk = rank(G);      % Computation stays on the GPU when the provider supports it
rk_host = gather(rk);
```
Expected output:
```matlab
rk_host = 2
```

## GPU residency in RunMat (Do I need `gpuArray`?)
RunMat’s planner automatically keeps matrices on the GPU when a provider implements the `rank`
hook. If the hook is missing, the builtin transparently gathers the matrix, computes the SVD on
the CPU, and uploads the scalar result so later GPU work remains resident. You can still seed
residency manually with `gpuArray` for MATLAB compatibility, but it is rarely required.

## FAQ

### How is the default tolerance chosen?
RunMat computes the default tolerance exactly as MATLAB: `max(size(A)) * eps(max(s))`, where `s`
are the singular values of `A`. This scales the cutoff with matrix size and magnitude.

### What does `rank([])` return?
The rank of the empty matrix is `0`. This matches MATLAB’s convention that an empty product has
neutral value.

### Does `rank` return an integer or a double?
`rank` returns a double-precision scalar, mirroring MATLAB’s numeric tower. The value is always an
integer-valued double.

### How does `rank` behave for vectors or scalars?
Scalars are treated as `1×1` matrices. `rank([0])` returns `0`, while `rank([5])` returns `1`. Row
or column vectors behave as matrices with one dimension equal to 1.

### Can `rank` detect symbolic rank or exact arithmetic?
No. Like MATLAB, RunMat’s `rank` relies on floating-point SVD and is subject to the chosen
tolerance. For symbolic or exact arithmetic you would use a computer algebra system.

### Will `rank` participate in fusion or auto-offload?
No. `rank` is a residency sink that eagerly computes an SVD. Fusion groups terminate before the
call, and the planner treats the builtin as a scalar reduction.

### Is the tolerance argument optional?
Yes. `rank(A)` uses the default tolerance and mirrors MATLAB. Supplying `rank(A, tol)` overrides
the cutoff. Non-scalar or negative tolerances raise MATLAB-compatible errors.

### What happens if the matrix contains NaNs or Infs?
Singular values involving `NaN` propagate and typically produce a rank of `0`. Infinite values
yield infinite singular values and therefore produce a rank equal to the number of infinite entries
above tolerance, matching MATLAB’s behaviour.

### Does `rank` allocate large temporary buffers?
Only enough memory for the SVD factors. For host execution this is handled by `nalgebra` (and
LAPACK when enabled). GPU providers are free to reuse buffers or stream the computation.

## See Also
[pinv](./pinv), [svd](./svd), [inv](./inv), [det](./det), [null](./null), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- Implementation: `crates/runmat-runtime/src/builtins/math/linalg/solve/rank.rs`
- Found a behavioural difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::solve::rank")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("matrix-rank"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("rank")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may keep the computation on-device via the `rank` hook; the reference backend gathers to the host and re-uploads a scalar.",
};

fn builtin_error(message: impl Into<String>) -> RuntimeControlFlow {
    build_runtime_error(message).with_builtin(NAME).build().into()
}

fn map_control_flow(flow: RuntimeControlFlow) -> RuntimeControlFlow {
    match flow {
        RuntimeControlFlow::Suspend(pending) => RuntimeControlFlow::Suspend(pending),
        RuntimeControlFlow::Error(err) => {
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
            builder.with_source(err).build().into()
        }
    }
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::solve::rank")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "`rank` terminates fusion plans and executes eagerly via an SVD.",
};

#[runtime_builtin(
    name = "rank",
    category = "math/linalg/solve",
    summary = "Compute the numerical rank of a matrix using SVD with MATLAB-compatible tolerance handling.",
    keywords = "rank,svd,tolerance,matrix,gpu",
    accel = "rank",
    builtin_path = "crate::builtins::math::linalg::solve::rank"
)]
fn rank_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let tol = parse_tolerance_arg(NAME, &rest).map_err(builtin_error)?;
    match value {
        Value::GpuTensor(handle) => rank_gpu(handle, tol),
        Value::ComplexTensor(tensor) => rank_complex_tensor_value(tensor, tol),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(builtin_error)?;
            rank_complex_tensor_value(tensor, tol)
        }
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other).map_err(builtin_error)?;
            rank_real_tensor_value(tensor, tol)
        }
    }
}

fn rank_gpu(handle: GpuTensorHandle, tol: Option<f64>) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        match provider.rank(&handle, tol) {
            Ok(device_scalar) => return Ok(Value::GpuTensor(device_scalar)),
            Err(_) => {
                // Fall through to host-based fallback.
            }
        }
    }

    let gathered =
        gpu_helpers::gather_value(&Value::GpuTensor(handle.clone())).map_err(map_control_flow)?;
    let rank = rank_scalar_from_value(gathered, tol)?;

    if let Some(provider) = runmat_accelerate_api::provider() {
        match upload_rank_scalar(provider, rank) {
            Ok(uploaded) => return Ok(Value::GpuTensor(uploaded)),
            Err(RuntimeControlFlow::Suspend(pending)) => {
                return Err(RuntimeControlFlow::Suspend(pending))
            }
            Err(RuntimeControlFlow::Error(_)) => {}
        }
    }

    Ok(Value::Num(rank))
}

fn upload_rank_scalar(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    rank: f64,
) -> BuiltinResult<GpuTensorHandle> {
    let data = [rank];
    let shape = [1usize, 1usize];
    let view = HostTensorView {
        data: &data,
        shape: &shape,
    };
    provider
        .upload(&view)
        .map_err(|e| builtin_error(format!("{NAME}: {e}")))
}

fn rank_real_tensor_value(tensor: Tensor, tol: Option<f64>) -> BuiltinResult<Value> {
    let rank = rank_real_tensor(&tensor, tol)?;
    Ok(Value::Num(rank as f64))
}

fn rank_complex_tensor_value(tensor: ComplexTensor, tol: Option<f64>) -> BuiltinResult<Value> {
    let rank = rank_complex_tensor(&tensor, tol)?;
    Ok(Value::Num(rank as f64))
}

fn rank_scalar_from_value(value: Value, tol: Option<f64>) -> BuiltinResult<f64> {
    match value {
        Value::Tensor(t) => rank_real_tensor(&t, tol).map(|r| r as f64),
        Value::ComplexTensor(t) => rank_complex_tensor(&t, tol).map(|r| r as f64),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(builtin_error)?;
            rank_complex_tensor(&tensor, tol).map(|r| r as f64)
        }
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other).map_err(builtin_error)?;
            rank_real_tensor(&tensor, tol).map(|r| r as f64)
        }
    }
}

fn rank_real_tensor(matrix: &Tensor, tol: Option<f64>) -> BuiltinResult<usize> {
    rank_real_tensor_impl(matrix, tol)
}

fn rank_complex_tensor(matrix: &ComplexTensor, tol: Option<f64>) -> BuiltinResult<usize> {
    rank_complex_tensor_impl(matrix, tol)
}

fn rank_real_tensor_impl(matrix: &Tensor, tol: Option<f64>) -> BuiltinResult<usize> {
    let (rows, cols) =
        matrix_dimensions_for(NAME, matrix.shape.as_slice()).map_err(builtin_error)?;
    if rows == 0 || cols == 0 {
        return Ok(0);
    }
    let dm = DMatrix::from_column_slice(rows, cols, &matrix.data);
    let svd = SVD::new(dm, false, false);
    let cutoff =
        tol.unwrap_or_else(|| svd_default_tolerance(svd.singular_values.as_slice(), rows, cols));
    Ok(svd
        .singular_values
        .iter()
        .filter(|&&value| value.is_infinite() || value > cutoff)
        .count())
}

fn rank_complex_tensor_impl(matrix: &ComplexTensor, tol: Option<f64>) -> BuiltinResult<usize> {
    let (rows, cols) =
        matrix_dimensions_for(NAME, matrix.shape.as_slice()).map_err(builtin_error)?;
    if rows == 0 || cols == 0 {
        return Ok(0);
    }
    let data: Vec<Complex64> = matrix
        .data
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let dm = DMatrix::from_column_slice(rows, cols, &data);
    let svd = SVD::new(dm, false, false);
    let cutoff =
        tol.unwrap_or_else(|| svd_default_tolerance(svd.singular_values.as_slice(), rows, cols));
    Ok(svd
        .singular_values
        .iter()
        .filter(|&&value| value.is_infinite() || value > cutoff)
        .count())
}

/// Host helper used by acceleration providers that defer rank to the shared implementation.
pub fn rank_host_real_for_provider(matrix: &Tensor, tol: Option<f64>) -> BuiltinResult<usize> {
    rank_real_tensor_impl(matrix, tol)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::RuntimeControlFlow;
    use runmat_builtins::{IntValue, Value};

    fn unwrap_error(flow: crate::RuntimeControlFlow) -> crate::RuntimeError {
        match flow {
            RuntimeControlFlow::Error(err) => err,
            RuntimeControlFlow::Suspend(_) => panic!("unexpected suspend"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rank_full_matrix() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result = rank_real_tensor_value(tensor, None).expect("rank");
        match result {
            Value::Num(r) => assert_eq!(r, 2.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rank_singular_matrix() {
        let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result = rank_real_tensor_value(tensor, None).expect("rank");
        match result {
            Value::Num(r) => assert_eq!(r, 1.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rank_default_tolerance_reduces_rank() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1e-16], vec![2, 2]).unwrap();
        let rank = rank_real_tensor(&tensor, None).expect("rank");
        assert_eq!(rank, 1);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rank_custom_tolerance_behavior() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1e-4], vec![2, 2]).unwrap();
        let default_rank = rank_real_tensor(&tensor, None).expect("rank");
        let custom_rank = rank_real_tensor(&tensor, Some(1e-3)).expect("rank");
        assert_eq!(default_rank, 2);
        assert_eq!(custom_rank, 1);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rank_empty_matrix_returns_zero() {
        let tensor = Tensor::new(Vec::<f64>::new(), vec![0, 0]).unwrap();
        let result = rank_real_tensor_value(tensor, None).expect("rank");
        match result {
            Value::Num(r) => assert_eq!(r, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rank_vector_input() {
        let tensor = Tensor::new(vec![1.0, 0.0, 2.0], vec![3, 1]).unwrap();
        let rank = rank_real_tensor(&tensor, None).expect("rank");
        assert_eq!(rank, 1);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rank_zero_vector_is_zero() {
        let tensor = Tensor::new(vec![0.0, 0.0, 0.0], vec![3, 1]).unwrap();
        let rank = rank_real_tensor(&tensor, None).expect("rank");
        assert_eq!(rank, 0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rank_invalid_shape_errors() {
        let tensor = Tensor::new(vec![0.0; 8], vec![2, 2, 2]).unwrap();
        let err = unwrap_error(rank_builtin(Value::Tensor(tensor), Vec::new()).unwrap_err());
        assert!(
            err.message().contains("2-D matrices or vectors"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rank_negative_tolerance_errors() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let err =
            unwrap_error(rank_builtin(Value::Tensor(tensor), vec![Value::Num(-1.0)]).unwrap_err());
        assert!(
            err.message().contains("tolerance must be >= 0"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rank_non_scalar_tolerance_errors() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let tol = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = unwrap_error(
            rank_builtin(Value::Tensor(tensor), vec![Value::Tensor(tol)]).unwrap_err(),
        );
        assert!(
            err.message().contains("tolerance must be a real scalar"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rank_complex_matrix() {
        let tensor = ComplexTensor::new(
            vec![(1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (2.0, -1.0)],
            vec![2, 2],
        )
        .unwrap();
        let result = rank_complex_tensor_value(tensor, None).expect("rank");
        match result {
            Value::Num(r) => assert_eq!(r, 2.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rank_scalar_bool_and_int() {
        let bool_rank = rank_builtin(Value::Bool(false), Vec::new()).expect("rank");
        let int_rank = rank_builtin(Value::Int(IntValue::I32(5)), Vec::new()).expect("rank");
        match bool_rank {
            Value::Num(r) => assert_eq!(r, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
        match int_rank {
            Value::Num(r) => assert_eq!(r, 1.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rank_gpu_round_trip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = rank_builtin(Value::GpuTensor(handle), Vec::new()).expect("rank");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.data[0], 1.0);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn rank_wgpu_matches_cpu() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };

        let _ = register_wgpu_provider(WgpuProviderOptions::default());

        let tensor = Tensor::new(
            vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0],
            vec![3, 3],
        )
        .unwrap();
        let cpu_rank = rank_real_tensor(&tensor, None).expect("cpu rank") as f64;

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("provider");
        let handle = provider.upload(&view).expect("upload");

        let gpu_value = rank_builtin(Value::GpuTensor(handle), Vec::new()).expect("rank");
        let gathered = test_support::gather(gpu_value).expect("gather");

        assert_eq!(gathered.data, vec![cpu_rank]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
