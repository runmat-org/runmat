//! MATLAB-compatible `isfinite` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, BuiltinResult, RuntimeControlFlow};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "isfinite",
        builtin_path = "crate::builtins::logical::tests::isfinite"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "isfinite"
category: "logical/tests"
keywords: ["isfinite", "finite mask", "numeric predicate", "gpuArray isfinite", "MATLAB isfinite"]
summary: "Return a logical mask indicating which elements of the input are finite."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Runs on the GPU when the provider exposes `logical_isfinite`; otherwise the runtime gathers to the host transparently."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::logical::tests::isfinite::tests"
  integration: "builtins::logical::tests::isfinite::tests::isfinite_gpu_roundtrip"
  gpu: "builtins::logical::tests::isfinite::tests::isfinite_wgpu_matches_host_path"
---

# What does the `isfinite` function do in MATLAB / RunMat?
`mask = isfinite(x)` returns a logical scalar or array indicating which elements of `x` are finite (not `NaN`, `+Inf`, or `-Inf`). The output matches MATLAB's semantics for scalars, matrices, higher-dimensional tensors, and `gpuArray` values.

## How does the `isfinite` function behave in MATLAB / RunMat?
- Numeric scalars return a logical scalar (`true`/`false`).
- Numeric arrays return a logical array of the same size, with `true` wherever the corresponding element is finite.
- Complex inputs report `true` when both the real and imaginary components are finite.
- Logical inputs return `true` because logical values (0 or 1) are finite by construction.
- Character arrays return logical arrays of ones (characters map to finite Unicode code points).
- String scalars return `false`; string arrays return logical arrays of zeros, mirroring MATLAB behavior.
- When the input is a `gpuArray`, RunMat keeps the computation on the device if the active acceleration provider implements the `logical_isfinite` hook; otherwise the runtime gathers the data back to the host automatically.

## Examples of using the `isfinite` function in MATLAB / RunMat

### Check if a scalar is finite

```matlab
result = isfinite(42);
```

Expected output:

```matlab
result =
     1
```

### Create a finite mask for a numeric matrix

```matlab
A = [1 NaN; Inf 4];
mask = isfinite(A);
```

Expected output:

```matlab
mask =
  2×2 logical array
     1     0
     0     1
```

### Identify finite components in a complex array

```matlab
Z = [1+2i Inf+0i 3+NaNi];
mask = isfinite(Z);
```

Expected output:

```matlab
mask =
  1×3 logical array
     1     0     0
```

### Apply `isfinite` to character data

```matlab
chars = ['R' 'u' 'n'];
mask = isfinite(chars);
```

Expected output:

```matlab
mask =
  1×3 logical array
     1     1     1
```

### Run `isfinite` directly on the GPU

```matlab
G = gpuArray([1 -Inf 5]);
mask_gpu = isfinite(G);
mask = gather(mask_gpu);
```

Expected output:

```matlab
mask =
  1×3 logical array
     1     0     1
```

## `isfinite` Function GPU Execution Behaviour
When RunMat Accelerate is active, `isfinite` looks for the provider hook `logical_isfinite`. Providers that implement the hook execute the finite test entirely on the GPU, producing a logical `gpuArray` result without any host transfers. If the hook is absent, RunMat gathers the input tensor back to the CPU, computes the mask on the host, and returns a regular logical array so the builtin always succeeds.

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** need to call `gpuArray` explicitly. RunMat's auto-offload planner keeps tensors on the GPU across fused expressions when that improves performance. You can still seed residency manually with `gpuArray` for compatibility with MATLAB scripts or when you want fine-grained control over data movement.

## FAQ

### Does `isfinite` treat `NaN` or `Inf` values as finite?
No. `isfinite` only returns `true` for values that are neither `NaN` nor infinite. Use `isnan` or `isinf` if you need to distinguish between those cases.

### What does `isfinite` return for logical inputs?
Logical inputs always produce `true` because logical values are limited to 0 or 1, which are finite.

### How does `isfinite` handle complex numbers?
It returns `true` only when both the real and imaginary components of the element are finite, matching MATLAB semantics.

### What happens with string or character inputs?
String scalars return `false`, and string arrays return logical zeros. Character arrays return logical ones because their Unicode code points are finite.

### Can I fuse `isfinite` with other elementwise operations?
Yes. The fusion planner treats `isfinite` as an elementwise operation, so expressions like `isfinite(A ./ B)` remain eligible for GPU fusion when the provider advertises support.

### Is there a performance difference between `isfinite`, `isnan`, and `isinf`?
Each predicate performs a single elementwise test. Performance is dominated by memory bandwidth, so they have comparable cost on both CPU and GPU.

## See Also
[isinf](./isinf), [isnan](./isnan), [isreal](./isreal), [gpuArray](./gpuarray), [gather](./gather)
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::tests::isfinite")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isfinite",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary {
        name: "logical_isfinite",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Dispatches to the provider `logical_isfinite` hook when available; otherwise the runtime gathers to host and computes the mask on the CPU.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::tests::isfinite")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isfinite",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            let (zero, one) = match ctx.scalar_ty {
                ScalarType::F32 => ("0.0", "1.0"),
                ScalarType::F64 => ("f64(0.0)", "f64(1.0)"),
                other => return Err(FusionError::UnsupportedPrecision(other)),
            };
            Ok(format!("select({zero}, {one}, isFinite({input}))"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fused kernels emit 0/1 masks; providers can override with native logical-isfinite implementations.",
};

const BUILTIN_NAME: &str = "isfinite";
const IDENTIFIER_INVALID_INPUT: &str = "MATLAB:isfinite:InvalidInput";
const IDENTIFIER_INTERNAL: &str = "RunMat:isfinite:InternalError";

#[runtime_builtin(
    name = "isfinite",
    category = "logical/tests",
    summary = "Return a logical mask indicating which elements of the input are finite.",
    keywords = "isfinite,finite,logical,gpu",
    accel = "elementwise",
    builtin_path = "crate::builtins::logical::tests::isfinite"
)]
fn isfinite_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => {
            if let Some(provider) = runmat_accelerate_api::provider() {
                if let Ok(mask) = provider.logical_isfinite(&handle) {
                    return Ok(gpu_helpers::logical_gpu_value(mask));
                }
            }
            let tensor = gpu_helpers::gather_tensor(&handle)
                .map_err(|err| internal_error(BUILTIN_NAME, format!("{BUILTIN_NAME}: {err}")))?;
            isfinite_tensor(BUILTIN_NAME, tensor)
        }
        other => isfinite_host(other),
    }
}

fn isfinite_host(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Num(x) => Ok(Value::Bool(x.is_finite())),
        Value::Int(_) | Value::Bool(_) => Ok(Value::Bool(true)),
        Value::Complex(re, im) => Ok(Value::Bool(re.is_finite() && im.is_finite())),
        Value::Tensor(tensor) => isfinite_tensor(BUILTIN_NAME, tensor),
        Value::ComplexTensor(tensor) => isfinite_complex_tensor(BUILTIN_NAME, tensor),
        Value::LogicalArray(array) => logical_full(BUILTIN_NAME, array.shape, true),
        Value::CharArray(array) => logical_full(BUILTIN_NAME, vec![array.rows, array.cols], true),
        Value::String(_) => Ok(Value::Bool(false)),
        Value::StringArray(array) => logical_full(BUILTIN_NAME, array.shape, false),
        _ => Err(build_runtime_error(format!(
            "{BUILTIN_NAME}: expected numeric, logical, char, or string input"
        ))
        .with_identifier(IDENTIFIER_INVALID_INPUT)
        .with_builtin(BUILTIN_NAME)
        .build()
        .into()),
    }
}

fn isfinite_tensor(name: &str, tensor: Tensor) -> BuiltinResult<Value> {
    let data = tensor
        .data
        .iter()
        .map(|&x| if x.is_finite() { 1u8 } else { 0u8 })
        .collect::<Vec<_>>();
    logical_result(name, data, tensor.shape)
}

fn isfinite_complex_tensor(name: &str, tensor: ComplexTensor) -> BuiltinResult<Value> {
    let data = tensor
        .data
        .iter()
        .map(|&(re, im)| {
            if re.is_finite() && im.is_finite() {
                1u8
            } else {
                0u8
            }
        })
        .collect::<Vec<_>>();
    logical_result(name, data, tensor.shape)
}

fn logical_full(name: &str, shape: Vec<usize>, value: bool) -> BuiltinResult<Value> {
    let total = tensor::element_count(&shape);
    if total == 0 {
        return LogicalArray::new(Vec::new(), shape)
            .map(Value::LogicalArray)
            .map_err(|e| logical_array_error(name, e));
    }
    let fill = if value { 1u8 } else { 0u8 };
    let bits = vec![fill; total];
    logical_result(name, bits, shape)
}

fn logical_result(name: &str, bits: Vec<u8>, shape: Vec<usize>) -> BuiltinResult<Value> {
    let total = tensor::element_count(&shape);
    if total != bits.len() {
        return Err(internal_error(
            name,
            format!(
                "{name}: internal error, mask length {} does not match shape {:?}",
                bits.len(),
                shape
            ),
        ));
    }
    if total == 1 {
        Ok(Value::Bool(bits[0] != 0))
    } else {
        LogicalArray::new(bits, shape)
            .map(Value::LogicalArray)
            .map_err(|e| logical_array_error(name, e))
    }
}

fn logical_array_error(name: &str, err: impl std::fmt::Display) -> RuntimeControlFlow {
    internal_error(name, format!("{name}: {err}"))
}

fn internal_error(name: &str, message: impl Into<String>) -> RuntimeControlFlow {
    build_runtime_error(message)
        .with_identifier(IDENTIFIER_INTERNAL)
        .with_builtin(name)
        .build()
        .into()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::{RuntimeControlFlow, RuntimeError};
    use runmat_builtins::{CharArray, IntValue, StringArray};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_scalar_true() {
        let result = isfinite_builtin(Value::Num(42.0)).expect("isfinite");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_scalar_false_for_nan() {
        let result = isfinite_builtin(Value::Num(f64::NAN)).expect("isfinite");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_scalar_false_for_inf() {
        let result = isfinite_builtin(Value::Num(f64::INFINITY)).expect("isfinite");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_int_and_bool_true() {
        let int_val = isfinite_builtin(Value::Int(IntValue::I32(7))).expect("isfinite");
        let bool_val = isfinite_builtin(Value::Bool(false)).expect("isfinite");
        assert_eq!(int_val, Value::Bool(true));
        assert_eq!(bool_val, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_complex_requires_both_components_finite() {
        let finite = isfinite_builtin(Value::Complex(1.0, -2.0)).expect("isfinite");
        assert_eq!(finite, Value::Bool(true));

        let inf_real = isfinite_builtin(Value::Complex(f64::INFINITY, 0.0)).expect("isfinite");
        assert_eq!(inf_real, Value::Bool(false));

        let nan_imag = isfinite_builtin(Value::Complex(0.0, f64::NAN)).expect("isfinite");
        assert_eq!(nan_imag, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_tensor_mask() {
        let tensor =
            Tensor::new(vec![1.0, f64::NAN, f64::INFINITY, -5.0], vec![2, 2]).expect("tensor");
        let result = isfinite_builtin(Value::Tensor(tensor)).expect("isfinite");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![2, 2]);
                assert_eq!(mask.data, vec![1, 0, 0, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_complex_tensor_mask() {
        let tensor = ComplexTensor::new(
            vec![(0.0, 0.0), (f64::NAN, 0.0), (1.0, f64::INFINITY)],
            vec![3, 1],
        )
        .unwrap();
        let result = isfinite_builtin(Value::ComplexTensor(tensor)).expect("isfinite");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![3, 1]);
                assert_eq!(mask.data, vec![1, 0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_logical_array_returns_ones() {
        let logical = LogicalArray::new(vec![0, 1, 0], vec![3, 1]).unwrap();
        let result = isfinite_builtin(Value::LogicalArray(logical)).expect("isfinite");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![3, 1]);
                assert!(mask.data.iter().all(|&bit| bit == 1));
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_char_array_returns_ones() {
        let array = CharArray::new("Run".chars().collect(), 1, 3).unwrap();
        let result = isfinite_builtin(Value::CharArray(array)).expect("isfinite");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![1, 3]);
                assert_eq!(mask.data, vec![1, 1, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_string_scalar_false() {
        let result = isfinite_builtin(Value::String("42".to_string())).expect("isfinite");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_string_array_returns_zeros() {
        let strings = StringArray::new(vec!["foo".into(), "bar".into()], vec![1, 2]).unwrap();
        let result = isfinite_builtin(Value::StringArray(strings)).expect("isfinite");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![1, 2]);
                assert_eq!(mask.data, vec![0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_empty_tensor_preserves_shape() {
        let tensor = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let result = isfinite_builtin(Value::Tensor(tensor)).expect("isfinite");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![0, 3]);
                assert!(mask.data.is_empty());
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_rejects_unsupported_types() {
        let err = unwrap_error(
            isfinite_builtin(Value::FunctionHandle("foo".to_string()))
                .expect_err("isfinite should reject function handles"),
        );
        assert!(
            err.message()
                .contains("expected numeric, logical, char, or string input"),
            "unexpected error message: {err:?}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfinite_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, f64::INFINITY, 3.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = isfinite_builtin(Value::GpuTensor(handle)).expect("isfinite");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![3, 1]);
            assert_eq!(gathered.data, vec![1.0, 0.0, 1.0]);
        });
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
    fn isfinite_wgpu_matches_host_path() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor =
            Tensor::new(vec![1.0, f64::NAN, f64::INFINITY, 5.0], vec![4, 1]).expect("tensor");
        let cpu = isfinite_tensor("isfinite", tensor.clone()).expect("cpu path");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("upload");
        let gpu = isfinite_builtin(Value::GpuTensor(handle)).expect("gpu path");
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::LogicalArray(expected), Tensor { data, shape, .. }) => {
                assert_eq!(shape, expected.shape);
                let expected_f64: Vec<f64> = expected
                    .data
                    .iter()
                    .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                    .collect();
                assert_eq!(data, expected_f64);
            }
            (Value::Bool(flag), Tensor { data, .. }) => {
                assert_eq!(data, vec![if flag { 1.0 } else { 0.0 }]);
            }
            other => panic!("unexpected results {other:?}"),
        }
    }

    fn unwrap_error(flow: RuntimeControlFlow) -> RuntimeError {
        match flow {
            RuntimeControlFlow::Error(err) => err,
            RuntimeControlFlow::Suspend(_) => panic!("unexpected suspend in isfinite tests"),
        }
    }
}
