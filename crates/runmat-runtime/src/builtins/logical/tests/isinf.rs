//! MATLAB-compatible `isinf` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "isinf",
        builtin_path = "crate::builtins::logical::tests::isinf"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "isinf"
category: "logical/tests"
keywords: ["isinf", "infinite mask", "numeric predicate", "gpuArray isinf", "MATLAB isinf"]
summary: "Return a logical mask indicating which elements of the input are ±Inf."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Runs on the GPU when the provider exposes `logical_isinf`; otherwise the runtime gathers to the host transparently."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::logical::tests::isinf::tests"
  integration: "builtins::logical::tests::isinf::tests::isinf_gpu_roundtrip"
  gpu: "builtins::logical::tests::isinf::tests::isinf_wgpu_matches_host_path"
---

# What does the `isinf` function do in MATLAB / RunMat?
`mask = isinf(x)` returns a logical scalar or array indicating which elements of `x` are positive or negative infinity. The output matches MATLAB's semantics for scalars, matrices, higher-dimensional tensors, strings, logical arrays, and gpuArray values.

## How does the `isinf` function behave in MATLAB / RunMat?
- Numeric scalars return a logical scalar (`true`/`false`).
- Numeric arrays return a logical array of the same size, with `true` wherever the corresponding element is `+Inf` or `-Inf`.
- Complex inputs report `true` when either the real or the imaginary component is infinite.
- Logical inputs return `false` because logical values (0 or 1) are finite by construction.
- Character arrays return logical arrays of zeros (characters map to finite code points).
- String arrays return logical arrays of zeros, mirroring MATLAB behavior.
- `string` scalars (string objects) return a logical scalar `false`.
- When the input is a `gpuArray`, RunMat keeps the computation on the device if the active acceleration provider implements the `logical_isinf` hook; otherwise the runtime gathers the data back to the host automatically.

## Examples of using the `isinf` function in MATLAB / RunMat

### Checking if a scalar is infinite

```matlab
result = isinf(1/0);
```

Expected output:

```matlab
result =
     1
```

### Detecting infinities after division by zero

```matlab
A = [1 0; 2 0];
quot = 1 ./ A;
mask = isinf(quot);
```

Expected output:

```matlab
mask =
  2×2 logical array
     0     1
     0     1
```

### Flagging infinite components of a complex array

```matlab
Z = [Inf+0i 1-Inf*1i 2+3i];
mask = isinf(Z);
```

Expected output:

```matlab
mask =
  1×3 logical array
     1     1     0
```

### Applying `isinf` to character data

```matlab
chars = ['R' 'u' 'n'];
mask = isinf(chars);
```

Expected output:

```matlab
mask =
  1×3 logical array
     0     0     0
```

### Running `isinf` directly on the GPU

```matlab
G = gpuArray([1 -Inf Inf]);
mask_gpu = isinf(G);
mask = gather(mask_gpu);
```

Expected output:

```matlab
mask =
  1×3 logical array
     0     1     1
```

## `isinf` Function GPU Execution Behaviour
When RunMat Accelerate is active, `isinf` looks for the provider hook `logical_isinf`. Providers that implement the hook execute the infinity test entirely on the GPU, producing a logical gpuArray result without any host transfers. If the hook is absent, RunMat gathers the input tensor back to the CPU, computes the mask on the host, and returns a regular logical array so the builtin always succeeds.

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** need to call `gpuArray` explicitly. RunMat's auto-offload planner keeps tensors on the GPU across fused expressions when that improves performance. You can still seed residency manually with `gpuArray` for compatibility with MATLAB scripts or when you want fine-grained control over data movement.

## FAQ

### Does `isinf` treat `NaN` values as infinite?
No. `isinf` only reports `true` for IEEE positive or negative infinity. `NaN` values return `false`, so you can combine `isinf` with `isnan` when you need to distinguish between the two.

### What does `isinf` return for logical inputs?
Logical inputs always produce `false` because logical values are limited to 0 or 1, which are finite.

### How does `isinf` handle complex numbers?
It returns `true` when either the real or the imaginary component is infinite, matching MATLAB semantics.

### Does `isinf` move data between the CPU and GPU?
Only when necessary. If the selected provider implements `logical_isinf`, all work stays on the GPU. Otherwise, RunMat gathers the tensor to the host, computes the result, and delivers a logical array.

### What happens with string or character inputs?
String scalars return `false`. Character arrays and string arrays return logical zeros with the same shape as the input.

### Is there a performance difference between `isinf`, `isnan`, and `isfinite`?
Each predicate performs a single elementwise test. Performance is dominated by memory bandwidth, so they have comparable cost on both CPU and GPU.

## See Also
[isfinite](./isfinite), [isnan](./isnan), [isreal](./isreal), [gpuArray](./gpuarray), [gather](./gather)
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::tests::isinf")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isinf",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary {
        name: "logical_isinf",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Dispatches to the provider `logical_isinf` hook when available; otherwise the runtime gathers to host and builds the logical mask on the CPU.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::tests::isinf")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isinf",
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
            Ok(format!("select({zero}, {one}, isInf({input}))"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fused kernels emit 0/1 masks; providers can override with native logical-isinf implementations.",
};

const BUILTIN_NAME: &str = "isinf";
const IDENTIFIER_INVALID_INPUT: &str = "MATLAB:isinf:InvalidInput";
const IDENTIFIER_INTERNAL: &str = "RunMat:isinf:InternalError";

#[runtime_builtin(
    name = "isinf",
    category = "logical/tests",
    summary = "Return a logical mask indicating which elements of the input are ±Inf.",
    keywords = "isinf,infinity,logical,gpu",
    accel = "elementwise",
    builtin_path = "crate::builtins::logical::tests::isinf"
)]
fn isinf_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => {
            if let Some(provider) = runmat_accelerate_api::provider() {
                if let Ok(mask) = provider.logical_isinf(&handle) {
                    return Ok(gpu_helpers::logical_gpu_value(mask));
                }
            }
            let tensor = gpu_helpers::gather_tensor(&handle)
                .map_err(|err| internal_error(BUILTIN_NAME, format!("{BUILTIN_NAME}: {err}")))?;
            isinf_tensor(BUILTIN_NAME, tensor)
        }
        other => isinf_host(other),
    }
}

fn isinf_host(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Num(x) => Ok(Value::Bool(x.is_infinite())),
        Value::Int(_) | Value::Bool(_) => Ok(Value::Bool(false)),
        Value::Complex(re, im) => Ok(Value::Bool(re.is_infinite() || im.is_infinite())),
        Value::Tensor(tensor) => isinf_tensor(BUILTIN_NAME, tensor),
        Value::ComplexTensor(tensor) => isinf_complex_tensor(BUILTIN_NAME, tensor),
        Value::LogicalArray(array) => {
            let LogicalArray { shape, .. } = array;
            logical_zeros(BUILTIN_NAME, shape)
        }
        Value::CharArray(array) => {
            let CharArray { rows, cols, .. } = array;
            logical_zeros(BUILTIN_NAME, vec![rows, cols])
        }
        Value::String(_) => Ok(Value::Bool(false)),
        Value::StringArray(array) => {
            let StringArray { shape, .. } = array;
            logical_zeros(BUILTIN_NAME, shape)
        }
        _ => Err(build_runtime_error(format!(
            "{BUILTIN_NAME}: expected numeric, logical, char, or string input"
        ))
        .with_identifier(IDENTIFIER_INVALID_INPUT)
        .with_builtin(BUILTIN_NAME)
        .build()),
    }
}

fn isinf_tensor(name: &str, tensor: Tensor) -> BuiltinResult<Value> {
    let data = tensor
        .data
        .iter()
        .map(|&x| if x.is_infinite() { 1u8 } else { 0u8 })
        .collect::<Vec<_>>();
    logical_result(name, data, tensor.shape)
}

fn isinf_complex_tensor(name: &str, tensor: ComplexTensor) -> BuiltinResult<Value> {
    let data = tensor
        .data
        .iter()
        .map(|&(re, im)| {
            if re.is_infinite() || im.is_infinite() {
                1u8
            } else {
                0u8
            }
        })
        .collect::<Vec<_>>();
    logical_result(name, data, tensor.shape)
}

fn logical_zeros(name: &str, shape: Vec<usize>) -> BuiltinResult<Value> {
    let total = tensor::element_count(&shape);
    if total == 0 {
        return LogicalArray::new(Vec::new(), shape)
            .map(Value::LogicalArray)
            .map_err(|e| logical_array_error(name, e));
    }
    let data = vec![0u8; total];
    logical_result(name, data, shape)
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

fn logical_array_error(name: &str, err: impl std::fmt::Display) -> RuntimeError {
    internal_error(name, format!("{name}: {err}"))
}

fn internal_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier(IDENTIFIER_INTERNAL)
        .with_builtin(name)
        .build()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::IntValue;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_scalar_positive() {
        let result = isinf_builtin(Value::Num(f64::INFINITY)).expect("isinf");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_scalar_negative() {
        let result = isinf_builtin(Value::Num(f64::NEG_INFINITY)).expect("isinf");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_scalar_finite() {
        let result = isinf_builtin(Value::Num(42.0)).expect("isinf");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_scalar_nan_false() {
        let result = isinf_builtin(Value::Num(f64::NAN)).expect("isinf");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_scalar_bool_false() {
        let result = isinf_builtin(Value::Bool(true)).expect("isinf");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_scalar_int_false() {
        let result = isinf_builtin(Value::Int(IntValue::I32(7))).expect("isinf");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_complex_scalar_detects_infinite_components() {
        let finite = isinf_builtin(Value::Complex(1.0, 2.0)).expect("isinf");
        assert_eq!(finite, Value::Bool(false));

        let inf_real = isinf_builtin(Value::Complex(f64::INFINITY, 0.0)).expect("isinf");
        assert_eq!(inf_real, Value::Bool(true));

        let inf_imag = isinf_builtin(Value::Complex(0.0, f64::NEG_INFINITY)).expect("isinf");
        assert_eq!(inf_imag, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_tensor_mask() {
        let tensor =
            Tensor::new(vec![1.0, f64::INFINITY, -f64::INFINITY, 0.0], vec![2, 2]).unwrap();
        let result = isinf_builtin(Value::Tensor(tensor)).expect("isinf");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![2, 2]);
                assert_eq!(mask.data, vec![0, 1, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_logical_array_returns_zeros() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![3, 1]).unwrap();
        let result = isinf_builtin(Value::LogicalArray(logical)).expect("isinf");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![3, 1]);
                assert!(mask.data.iter().all(|&bit| bit == 0));
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_complex_tensor_mask() {
        let tensor = ComplexTensor::new(
            vec![(0.0, 0.0), (f64::INFINITY, 1.0), (2.0, f64::NEG_INFINITY)],
            vec![3, 1],
        )
        .unwrap();
        let result = isinf_builtin(Value::ComplexTensor(tensor)).expect("isinf");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![3, 1]);
                assert_eq!(mask.data, vec![0, 1, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_string_scalar_false() {
        let result = isinf_builtin(Value::String("Inf".to_string())).expect("isinf");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_string_array_returns_all_false() {
        let strings = StringArray::new(vec!["foo".into(), "bar".into()], vec![1, 2]).unwrap();
        let result = isinf_builtin(Value::StringArray(strings)).expect("isinf");
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
    fn isinf_empty_tensor_preserves_shape() {
        let tensor = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
        let result = isinf_builtin(Value::Tensor(tensor)).expect("isinf");
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
    fn isinf_singleton_tensor_returns_scalar_bool() {
        let tensor = Tensor::new(vec![f64::INFINITY], vec![1, 1]).unwrap();
        let result = isinf_builtin(Value::Tensor(tensor)).expect("isinf");
        assert_eq!(result, Value::Bool(true));

        let finite = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let result = isinf_builtin(Value::Tensor(finite)).expect("isinf");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_rejects_unsupported_types() {
        let err = isinf_builtin(Value::FunctionHandle("foo".to_string()))
            .expect_err("isinf should reject function handles");
        assert!(
            err.message()
                .contains("expected numeric, logical, char, or string input"),
            "unexpected error message: {err:?}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_char_array_returns_zeros() {
        let array = CharArray::new("Inf".chars().collect(), 1, 3).unwrap();
        let result = isinf_builtin(Value::CharArray(array)).expect("isinf");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![1, 3]);
                assert_eq!(mask.data, vec![0, 0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isinf_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, f64::INFINITY, -f64::INFINITY], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = isinf_builtin(Value::GpuTensor(handle)).expect("isinf");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![3, 1]);
            assert_eq!(gathered.data, vec![0.0, 1.0, 1.0]);
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
    fn isinf_wgpu_matches_host_path() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor =
            Tensor::new(vec![1.0, f64::INFINITY, -f64::INFINITY, 0.0], vec![2, 2]).unwrap();
        let cpu = isinf_tensor("isinf", tensor.clone()).expect("cpu path");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("upload");
        let gpu = isinf_builtin(Value::GpuTensor(handle)).expect("gpu path");
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
}
