//! MATLAB-compatible `factorial` builtin with GPU-aware semantics for RunMat.
//!
//! Implements element-wise factorial for numerical inputs, mirroring MATLAB’s
//! restrictions to non-negative integers while providing documented fallbacks
//! when GPU providers lack a dedicated kernel.

use once_cell::sync::Lazy;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const MAX_FACTORIAL_N: usize = 170;

static FACT_TABLE: Lazy<[f64; MAX_FACTORIAL_N + 1]> = Lazy::new(|| {
    let mut table = [1.0f64; MAX_FACTORIAL_N + 1];
    let mut acc = 1.0;
    for (n, slot) in table.iter_mut().enumerate().skip(1) {
        acc *= n as f64;
        *slot = acc;
    }
    table
});

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "factorial",
        builtin_path = "crate::builtins::math::elementwise::factorial"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "factorial"
category: "math/elementwise"
keywords: ["factorial", "combinatorics", "n!", "permutations", "gpu", "like"]
summary: "Element-wise factorial for non-negative integers with MATLAB-compatible NaN/Inf behaviour."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Calls unary_factorial when the provider implements it; otherwise gathers to the host and re-uploads only when a 'like' prototype is provided."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::elementwise::factorial::tests"
  integration: "builtins::math::elementwise::factorial::tests::factorial_gpu_provider_roundtrip"
---

# What does the `factorial` function do in MATLAB / RunMat?
`factorial(X)` computes `X!` (the product of the integers from `1` to `X`) for every element of `X`.
Inputs must be non-negative integers; MATLAB semantics dictate that non-integers return `NaN`,
and integers larger than `170` overflow to `Inf` in double precision.

## How does the `factorial` function behave in MATLAB / RunMat?
- Scalars, vectors, matrices, and N-D tensors are processed element-by-element with MATLAB’s implicit expansion rules.
- Logical inputs promote to double precision (`true → 1`, `false → 0`) before evaluation; integer classes are cast to double.
- Non-integer or negative inputs yield `NaN`; large integers (`n ≥ 171`) overflow to `Inf`, matching MATLAB’s overflow handling in double precision.
- `factorial(0)` and `factorial(-0.0)` both return `1`, in accordance with the definition `0! = 1`.
- Results are real doubles. Passing `'like', prototype` lets you retain host or GPU residency to integrate with existing pipelines.
- GPU tensors remain on device when the active provider implements `unary_factorial`; otherwise RunMat gathers to the host, computes the result, and re-uploads only when you explicitly request GPU residency via `'like'`.

## `factorial` Function GPU Execution Behaviour
- RunMat Accelerate first tries the provider’s `unary_factorial` hook. Simple in-process providers can satisfy this by mirroring the CPU calculation.
- When the hook is unavailable (currently the WGPU backend), RunMat transparently gathers the tensor, evaluates `factorial` on the CPU, and returns the host result.
- Provide `'like', gpuArray(...)` to force the fallback path to re-upload the result so downstream GPU code keeps working.
- Fusion currently bypasses factorial because the operation is not polynomial; element-wise kernels fall back to the scalar implementation.

## Examples of using the `factorial` function in MATLAB / RunMat

### Factorial of a single integer value

```matlab
y = factorial(5)
```

Expected output:

```matlab
y = 120
```

### Factorial of zero returns one

```matlab
factorial(0)
```

Expected output:

```matlab
ans = 1
```

### Factorial across a vector of non-negative integers

```matlab
vals = factorial([0 1 3 5]);
```

Expected output:

```matlab
vals = [1 1 6 120]
```

### Detecting invalid non-integer inputs

```matlab
result = factorial([2.5 -1 4]);
```

Expected output:

```matlab
result = [NaN NaN 24]
```

### Handling large inputs that overflow to infinity

```matlab
big = factorial(171);
```

Expected output:

```matlab
big = Inf
```

### Using factorial with `gpuArray` inputs

```matlab
G = gpuArray(uint16([3 4 5]));
R = factorial(G);
host = gather(R);
```

Expected output:

```matlab
host = [6 24 120]
```

### Keeping results on the GPU with `'like'`

```matlab
proto = gpuArray.zeros(1, 1, 'single');
deviceResult = factorial([3 4], 'like', proto);
gathered = gather(deviceResult);
```

Expected output:

```matlab
gathered =
  1x2 single
     6    24
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You do **not** need to call `gpuArray` manually to get correct results. When the provider supports
`unary_factorial`, tensors stay on the GPU automatically. If the provider does not, RunMat gathers to
the host and computes the answer there. Provide `'like', gpuArray(...)` if you want the fallback path
to re-upload the result automatically.

## FAQ

### What inputs are valid for `factorial`?
Any non-negative integer (including zero). Logical and integer arrays are accepted; doubles must be
exact integers within floating-point tolerance. Non-integer or negative values return `NaN`.

### Why do large integers return `Inf`?
Double precision overflows at `171!`. MATLAB returns `Inf` for those values, and RunMat mirrors that
behaviour.

### Does `factorial` support complex numbers?
No. Use `gamma(z + 1)` if you need the analytic continuation for complex arguments.

### How does `factorial` behave with `NaN` or `Inf` inputs?
`factorial(NaN)` returns `NaN`. `factorial(Inf)` returns `Inf`. Negative infinity propagates to `NaN`.

### Can I keep the output on the GPU?
Yes, either when the provider implements `unary_factorial` or by passing `'like', gpuArray(...)`,
which uploads the host-computed result back to the GPU after the fallback path.

### Why does `factorial` return double precision even for integer inputs?
MATLAB defines `factorial` to return doubles so the result matches downstream functions that expect
floating-point inputs, and RunMat follows the same convention.

### How does `factorial` interact with fusion?
Factorial currently bypasses the fusion planner because it is not built from primitive arithmetic
ops. Surrounding element-wise expressions still fuse; factorial runs as an isolated scalar op inside
those kernels.

### What error message should I expect for unsupported types?
Passing strings, structs, or complex numbers raises `factorial: unsupported input type ...` so you
can correct the call site quickly.

## See Also
[gamma](./gamma), [power](./power), [prod](./prod), [permute](./permute)

## Source & Feedback
- The full source code for the implementation of the `factorial` function is available at: [`crates/runmat-runtime/src/builtins/math/elementwise/factorial.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/factorial.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::factorial")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "factorial",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary {
        name: "unary_factorial",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may implement unary_factorial; otherwise the runtime gathers to host and mirrors MATLAB overflow/NaN behaviour.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::elementwise::factorial"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "factorial",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Factorial is evaluated as a scalar helper; fusion currently bypasses it and executes the standalone host or provider kernel.",
};

const BUILTIN_NAME: &str = "factorial";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "factorial",
    category = "math/elementwise",
    summary = "Element-wise factorial for non-negative integers.",
    keywords = "factorial,n!,permutation,gpu",
    accel = "unary",
    builtin_path = "crate::builtins::math::elementwise::factorial"
)]
async fn factorial_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let output = parse_output_template(&rest)?;
    let base = match value {
        Value::GpuTensor(handle) => factorial_gpu(handle).await?,
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            return Err(builtin_error(
                "factorial: complex inputs are not supported; use gamma(z + 1) instead",
            ))
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            return Err(builtin_error(
                "factorial: expected numeric or logical input",
            ))
        }
        other => {
            let tensor = tensor::value_into_tensor_for("factorial", other)
                .map_err(|e| builtin_error(format!("factorial: {e}")))?;
            factorial_tensor(tensor).map(tensor::tensor_into_value)?
        }
    };
    apply_output_template(base, &output).await
}

#[derive(Clone)]
enum OutputTemplate {
    Default,
    Like(Value),
}

fn parse_output_template(args: &[Value]) -> BuiltinResult<OutputTemplate> {
    match args.len() {
        0 => Ok(OutputTemplate::Default),
        1 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Err(builtin_error("factorial: expected prototype after 'like'"))
            } else {
                Err(builtin_error(
                    "factorial: unrecognised option; only 'like' is supported",
                ))
            }
        }
        2 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Ok(OutputTemplate::Like(args[1].clone()))
            } else {
                Err(builtin_error(
                    "factorial: unrecognised option; only 'like' is supported",
                ))
            }
        }
        _ => Err(builtin_error("factorial: too many input arguments")),
    }
}

async fn factorial_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_factorial(&handle) {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    Ok(tensor::tensor_into_value(factorial_tensor(tensor)?))
}

fn factorial_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let mut data = Vec::with_capacity(tensor.data.len());
    for &value in &tensor.data {
        data.push(factorial_scalar(value));
    }
    Tensor::new(data, tensor.shape.clone()).map_err(|e| builtin_error(format!("factorial: {e}")))
}

fn factorial_scalar(value: f64) -> f64 {
    if value.is_nan() {
        return f64::NAN;
    }
    if value == 0.0 {
        return 1.0;
    }
    if value.is_infinite() {
        return if value.is_sign_positive() {
            f64::INFINITY
        } else {
            f64::NAN
        };
    }
    if value < 0.0 {
        return f64::NAN;
    }
    let Some(n) = classify_nonnegative_integer(value) else {
        return f64::NAN;
    };
    if n > MAX_FACTORIAL_N {
        return f64::INFINITY;
    }
    FACT_TABLE[n]
}

fn classify_nonnegative_integer(value: f64) -> Option<usize> {
    if !value.is_finite() {
        return None;
    }
    if value < 0.0 {
        return None;
    }
    let rounded = value.round();
    let tol = f64::EPSILON * value.abs().max(1.0);
    if (value - rounded).abs() > tol {
        return None;
    }
    if rounded < 0.0 {
        return None;
    }
    Some(rounded as usize)
}

async fn apply_output_template(value: Value, template: &OutputTemplate) -> BuiltinResult<Value> {
    match template {
        OutputTemplate::Default => Ok(value),
        OutputTemplate::Like(proto) => {
            let analysis = analyse_like_prototype(proto).await?;
            match analysis.device {
                DevicePreference::Host => convert_to_host_like(value).await,
                DevicePreference::Gpu => convert_to_gpu_like(value),
            }
        }
    }
}

#[derive(Clone, Copy)]
enum DevicePreference {
    Host,
    Gpu,
}

struct LikeAnalysis {
    device: DevicePreference,
}

#[async_recursion::async_recursion(?Send)]
async fn analyse_like_prototype(proto: &Value) -> BuiltinResult<LikeAnalysis> {
    match proto {
        Value::GpuTensor(_) => Ok(LikeAnalysis {
            device: DevicePreference::Gpu,
        }),
        Value::Tensor(_) | Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok(LikeAnalysis {
            device: DevicePreference::Host,
        }),
        Value::LogicalArray(_) => Ok(LikeAnalysis {
            device: DevicePreference::Host,
        }),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(builtin_error(
            "factorial: complex prototypes for 'like' are not supported; results are always real",
        )),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => Err(builtin_error(
            "factorial: prototype must be numeric or a gpuArray",
        )),
        other => {
            let gathered = gpu_helpers::gather_value_async(other)
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            analyse_like_prototype(&gathered).await
        }
    }
}

async fn convert_to_host_like(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
            .await
            .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME)),
        other => Ok(other),
    }
}

fn convert_to_gpu_like(value: Value) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        builtin_error(
            "factorial: GPU output requested via 'like' but no acceleration provider is active",
        )
    })?;
    match value {
        Value::GpuTensor(handle) => Ok(Value::GpuTensor(handle)),
        Value::Tensor(tensor) => upload_tensor(provider, tensor),
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| builtin_error(format!("factorial: {e}")))?;
            upload_tensor(provider, tensor)
        }
        Value::Int(i) => convert_to_gpu_like(Value::Num(i.to_f64())),
        Value::Bool(b) => convert_to_gpu_like(Value::Num(if b { 1.0 } else { 0.0 })),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| builtin_error(format!("factorial: {e}")))?;
            upload_tensor(provider, tensor)
        }
        other => Err(builtin_error(format!(
            "factorial: cannot place value {other:?} on the GPU via 'like'"
        ))),
    }
}

fn upload_tensor(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    tensor: Tensor,
) -> BuiltinResult<Value> {
    let view = HostTensorView {
        data: &tensor.data,
        shape: &tensor.shape,
    };
    let handle = provider
        .upload(&view)
        .map_err(|e| builtin_error(format!("factorial: failed to upload GPU result: {e}")))?;
    Ok(Value::GpuTensor(handle))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, LogicalArray, Tensor};

    fn factorial_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::factorial_builtin(value, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_scalar_positive() {
        let result = factorial_builtin(Value::Num(5.0), Vec::new()).expect("factorial");
        assert_eq!(result, Value::Num(120.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_zero_is_one() {
        let result = factorial_builtin(Value::Num(0.0), Vec::new()).expect("factorial");
        assert_eq!(result, Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_vector_inputs() {
        let tensor = Tensor::new(vec![0.0, 1.0, 3.0, 5.0], vec![4, 1]).unwrap();
        let result = factorial_builtin(Value::Tensor(tensor), Vec::new()).expect("factorial");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![4, 1]);
                assert_eq!(out.data, vec![1.0, 1.0, 6.0, 120.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_non_integer_produces_nan() {
        let result = factorial_builtin(Value::Num(2.5), Vec::new()).expect("factorial");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected scalar NaN, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_negative_produces_nan() {
        let tensor = Tensor::new(vec![-1.0, 3.0], vec![2, 1]).unwrap();
        let result = factorial_builtin(Value::Tensor(tensor), Vec::new()).expect("factorial");
        match result {
            Value::Tensor(out) => {
                assert!(out.data[0].is_nan());
                assert_eq!(out.data[1], 6.0);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_small_positive_non_integer_nan() {
        let result = factorial_builtin(Value::Num(1e-12), Vec::new()).expect("factorial");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected scalar NaN, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_overflow_returns_inf() {
        let result = factorial_builtin(Value::Num(171.0), Vec::new()).expect("factorial");
        match result {
            Value::Num(v) => assert!(v.is_infinite()),
            other => panic!("expected scalar Inf, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_like_missing_prototype_errors() {
        let err = factorial_builtin(Value::Num(3.0), vec![Value::from("like")])
            .expect_err("expected error");
        assert!(err.message().contains("prototype"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_like_gpu_prototype_uploads() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");
            let result = factorial_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            )
            .expect("factorial");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape, vec![2, 1]);
                    assert_eq!(gathered.data, vec![6.0, 24.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 3.0, 5.0], vec![4, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = factorial_builtin(Value::GpuTensor(handle), Vec::new()).expect("fact");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.data, vec![1.0, 1.0, 6.0, 120.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_like_host_with_gpu_input_gathers() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = factorial_builtin(
                Value::GpuTensor(handle),
                vec![Value::from("like"), Value::Num(0.0)],
            )
            .expect("factorial");
            match result {
                Value::Tensor(t) => {
                    assert_eq!(t.data, vec![6.0, 24.0]);
                }
                other => panic!("expected host tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_logical_input_promotes() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![3, 1]).unwrap();
        let result = factorial_builtin(Value::LogicalArray(logical), Vec::new()).expect("fact");
        match result {
            Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 1.0, 1.0]),
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_int_input_promotes_to_double() {
        let value = Value::Int(IntValue::U16(5));
        let result = factorial_builtin(value, Vec::new()).expect("factorial");
        assert_eq!(result, Value::Num(120.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_nan_propagates() {
        let result = factorial_builtin(Value::Num(f64::NAN), Vec::new()).expect("factorial");
        match result {
            Value::Num(v) => assert!(v.is_nan()),
            other => panic!("expected scalar NaN, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_complex_input_errors() {
        let err = factorial_builtin(Value::Complex(1.0, 0.5), Vec::new())
            .expect_err("expected complex rejection");
        assert!(err.message().contains("complex"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_string_input_errors() {
        let err = factorial_builtin(Value::from("hello"), Vec::new())
            .expect_err("expected string rejection");
        assert!(err.message().contains("numeric"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn factorial_like_complex_prototype_rejected() {
        let err = factorial_builtin(
            Value::Num(3.0),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        )
        .expect_err("expected complex prototype rejection");
        assert!(err.message().contains("complex"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn factorial_wgpu_matches_cpu_after_gather() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0, 1.0, 4.0], vec![3, 1]).unwrap();
        let cpu = factorial_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("cpu");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = factorial_gpu(handle).expect("gpu");
        let gathered = test_support::gather(gpu).expect("gather");
        let cpu_tensor = match cpu {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("unexpected cpu result {other:?}"),
        };
        assert_eq!(gathered.shape, cpu_tensor.shape);
        assert_eq!(gathered.data, cpu_tensor.data);
    }
}
