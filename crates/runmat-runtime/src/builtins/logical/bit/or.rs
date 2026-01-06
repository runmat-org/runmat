//! MATLAB-compatible logical `or` builtin with GPU support.

use runmat_builtins::{CharArray, ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "or",
        builtin_path = "crate::builtins::logical::bit::or"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "or"
category: "logical/bit"
keywords: ["logical or", "elementwise or", "boolean or", "MATLAB or", "gpuArray or"]
summary: "Element-wise logical OR for scalars, arrays, and gpuArray values."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Runs entirely on the GPU when the active provider implements `logical_or`; otherwise inputs gather back to the host automatically."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::logical::bit::or::tests"
  integration: "builtins::logical::bit::or::tests::or_gpu_roundtrip"
  gpu: "builtins::logical::bit::or::tests::or_wgpu_matches_host_path"
---

# What does the `or` function do in MATLAB / RunMat?
`or(A, B)` returns the element-wise logical OR of its inputs. Any non-zero value (including `NaN`) evaluates to `true`. The result is a logical scalar when the broadcasted shape has exactly one element, or a logical array otherwise.

## How does the `or` function behave in MATLAB / RunMat?
- Accepts logical, numeric, complex, and character arrays; character code points of zero evaluate to `false`.
- Supports MATLAB-style implicit expansion so scalars and singleton dimensions broadcast automatically.
- Propagates empty dimensions: if a broadcasted axis has length `0`, the output is an empty logical array with the same shape.
- Treats `NaN` values as `true`, matching MATLAB's element-wise logical semantics.
- Keeps `gpuArray` inputs on device when the active provider exposes the `logical_or` hook; otherwise the runtime gathers to host transparently.

## Examples of using the `or` function in MATLAB / RunMat

### Check if either logical scalar is true

```matlab
result = or(true, false)
```

Expected output:

```matlab
result =
     1
```

### Combine numeric arrays element-wise with logical OR

```matlab
A = [1 0 2 0];
B = [3 4 0 0];
C = or(A, B)
```

Expected output:

```matlab
C =
  1×4 logical array
     1     1     1     0
```

### Apply logical OR with automatic scalar expansion

```matlab
mask = [1; 0; 3; 0];
flag = or(mask, 0)
```

Expected output:

```matlab
flag =
  4×1 logical array
     1
     0
     1
     0
```

### Use logical OR with character arrays

```matlab
lhs = ['R' 'u' 0];
rhs = ['R' 0 'n'];
match = or(lhs, rhs)
```

Expected output:

```matlab
match =
  1×3 logical array
     1     1     1
```

### Run logical OR directly on the GPU

```matlab
G1 = gpuArray([0 2 0 4]);
G2 = gpuArray([1 0 3 0]);
deviceResult = or(G1, G2)
hostResult = gather(deviceResult)
```

Expected output:

```matlab
deviceResult =
  1×4 gpuArray logical array
     1     1     1     1
hostResult =
  1×4 logical array
     1     1     1     1
```

## `or` Function GPU Execution Behaviour
When both operands reside on the GPU and the active provider implements the `logical_or` hook, RunMat lowers the call into a device kernel that writes `0` or `1` for each element. The fusion planner treats `or` as an element-wise operation, so fused expressions (for example, `or(A > 0, B)`) stay on device without intermediate gathers. If the provider lacks the hook, the runtime gathers the inputs to host memory automatically and executes the CPU implementation instead of failing.

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** need to call `gpuArray` explicitly. RunMat's native auto-offload logic moves data to the GPU when a fused expression benefits from device execution, and results stay resident until you gather them or the planner needs host access. Call `gpuArray` only when you want to seed residency manually, or when you are porting MATLAB code that already uses explicit device transfers.

## FAQ

### Does `or` return logical values?
Yes. The result is a logical scalar (`true`/`false`) when the broadcasted shape contains exactly one element; otherwise the function returns a logical array. On the GPU the kernel writes `0.0`/`1.0` elements, and the runtime converts them back to logical values when you gather.

### How are `NaN` values handled?
`NaN` counts as `true`. For example, `or(NaN, 0)` returns `true` because at least one operand evaluates to non-zero.

### Is implicit expansion supported?
Yes. The inputs follow MATLAB-style implicit expansion rules: dimensions of length `1` broadcast across the other input. Fully incompatible shapes raise a size-mismatch error.

### Can I use `or` with complex numbers?
Yes. Real or complex inputs return `true` when either the real or imaginary component is non-zero. For example, `or(0 + 0i, 0 + 2i)` returns `true`.

### How does `or` differ from the `|` operator?
They share the same element-wise semantics. The functional form is convenient for higher-order code (for example, `arrayfun(@or, ...)`) and for aligning with MATLAB documentation. Use `&&` or `||` for short-circuit scalar logic.

### What happens when only one input is a `gpuArray`?
RunMat promotes the other input to the GPU before dispatch when the auto-offload planner decides it is profitable. If the provider lacks a device implementation, both operands gather to host automatically and the logical result executes on the CPU.

## See Also
[gpuArray](./gpuarray), [gather](./gather)
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::bit::or")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "or",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Binary {
        name: "logical_or",
        commutative: true,
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Falls back to host execution when the provider does not implement logical_or; non-zero (including NaN) inputs map to true.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::bit::or")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "or",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let lhs = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            let rhs = ctx.inputs.get(1).ok_or(FusionError::MissingInput(1))?;
            let zero = match ctx.scalar_ty {
                ScalarType::F32 => "0.0".to_string(),
                ScalarType::F64 => "f64(0.0)".to_string(),
                _ => return Err(FusionError::UnsupportedPrecision(ctx.scalar_ty)),
            };
            let one = match ctx.scalar_ty {
                ScalarType::F32 => "1.0".to_string(),
                ScalarType::F64 => "f64(1.0)".to_string(),
                _ => return Err(FusionError::UnsupportedPrecision(ctx.scalar_ty)),
            };
            let cond = format!("(({lhs} != {zero}) || ({rhs} != {zero}))");
            Ok(format!("select({zero}, {one}, {cond})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes:
        "Fusion generates WGSL kernels that treat non-zero inputs as true and write 0/1 outputs.",
};

#[runtime_builtin(
    name = "or",
    category = "logical/bit",
    summary = "Element-wise logical OR for scalars, arrays, and gpuArray values.",
    keywords = "logical,or,elementwise,boolean,gpu",
    accel = "elementwise",
    builtin_path = "crate::builtins::logical::bit::or"
)]
fn or_builtin(lhs: Value, rhs: Value) -> Result<Value, String> {
    if let (Value::GpuTensor(ref a), Value::GpuTensor(ref b)) = (&lhs, &rhs) {
        if let Some(provider) = runmat_accelerate_api::provider() {
            if let Ok(handle) = provider.logical_or(a, b) {
                return Ok(gpu_helpers::logical_gpu_value(handle));
            }
        }
    }
    or_host(lhs, rhs)
}

fn or_host(lhs: Value, rhs: Value) -> Result<Value, String> {
    let left = logical_buffer_from("or", lhs)?;
    let right = logical_buffer_from("or", rhs)?;
    let shape = broadcast_shapes("or", &left.shape, &right.shape)?;
    let total = tensor::element_count(&shape);
    if total == 0 {
        return logical_value("or", Vec::new(), shape);
    }

    let strides_left = compute_strides(&left.shape);
    let strides_right = compute_strides(&right.shape);

    let mut data = Vec::with_capacity(total);
    for linear in 0..total {
        let lhs_bit = if left.data.is_empty() {
            0
        } else {
            let idx = broadcast_index(linear, &shape, &left.shape, &strides_left);
            *left.data.get(idx).unwrap_or(&0)
        };
        let rhs_bit = if right.data.is_empty() {
            0
        } else {
            let idx = broadcast_index(linear, &shape, &right.shape, &strides_right);
            *right.data.get(idx).unwrap_or(&0)
        };
        data.push(if lhs_bit != 0 || rhs_bit != 0 { 1 } else { 0 });
    }

    logical_value("or", data, shape)
}

fn logical_value(fn_name: &str, data: Vec<u8>, shape: Vec<usize>) -> Result<Value, String> {
    if data.len() == 1 && tensor::element_count(&shape) == 1 {
        Ok(Value::Bool(data[0] != 0))
    } else {
        LogicalArray::new(data, shape)
            .map(Value::LogicalArray)
            .map_err(|e| format!("{fn_name}: {e}"))
    }
}

struct LogicalBuffer {
    data: Vec<u8>,
    shape: Vec<usize>,
}

fn logical_buffer_from(name: &str, value: Value) -> Result<LogicalBuffer, String> {
    match value {
        Value::LogicalArray(array) => {
            let LogicalArray { data, shape } = array;
            Ok(LogicalBuffer { data, shape })
        }
        Value::Bool(flag) => Ok(LogicalBuffer {
            data: vec![if flag { 1 } else { 0 }],
            shape: vec![1, 1],
        }),
        Value::Num(n) => Ok(LogicalBuffer {
            data: vec![logical_from_f64(n)],
            shape: vec![1, 1],
        }),
        Value::Int(i) => Ok(LogicalBuffer {
            data: vec![if i.to_i64() != 0 { 1 } else { 0 }],
            shape: vec![1, 1],
        }),
        Value::Complex(re, im) => Ok(LogicalBuffer {
            data: vec![logical_from_complex(re, im)],
            shape: vec![1, 1],
        }),
        Value::Tensor(tensor) => tensor_to_logical_buffer(tensor),
        Value::ComplexTensor(tensor) => complex_tensor_to_logical_buffer(tensor),
        Value::CharArray(array) => char_array_to_logical_buffer(array),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor(&handle)?;
            tensor_to_logical_buffer(tensor)
        }
        other => Err(format!(
            "{name}: unsupported input type {:?}; expected logical, numeric, complex, or character data",
            other
        )),
    }
}

fn tensor_to_logical_buffer(tensor: Tensor) -> Result<LogicalBuffer, String> {
    let Tensor { data, shape, .. } = tensor;
    let mapped = data.into_iter().map(logical_from_f64).collect();
    Ok(LogicalBuffer {
        data: mapped,
        shape,
    })
}

fn complex_tensor_to_logical_buffer(tensor: ComplexTensor) -> Result<LogicalBuffer, String> {
    let ComplexTensor { data, shape, .. } = tensor;
    let mapped = data
        .into_iter()
        .map(|(re, im)| logical_from_complex(re, im))
        .collect();
    Ok(LogicalBuffer {
        data: mapped,
        shape,
    })
}

fn char_array_to_logical_buffer(array: CharArray) -> Result<LogicalBuffer, String> {
    let CharArray { data, rows, cols } = array;
    let mapped = data
        .into_iter()
        .map(|ch| if ch == '\0' { 0 } else { 1 })
        .collect();
    Ok(LogicalBuffer {
        data: mapped,
        shape: vec![rows, cols],
    })
}

#[inline]
fn logical_from_f64(value: f64) -> u8 {
    if value != 0.0 {
        1
    } else {
        0
    }
}

#[inline]
fn logical_from_complex(re: f64, im: f64) -> u8 {
    if re != 0.0 || im != 0.0 {
        1
    } else {
        0
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::ProviderPrecision;
    use runmat_builtins::IntValue;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn or_of_booleans() {
        assert_eq!(
            or_builtin(Value::Bool(true), Value::Bool(false)).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            or_builtin(Value::Bool(false), Value::Bool(false)).unwrap(),
            Value::Bool(false)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn or_numeric_arrays() {
        let a = Tensor::new(vec![1.0, 0.0, 2.0, 0.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![3.0, 4.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let result = or_builtin(Value::Tensor(a), Value::Tensor(b)).unwrap();
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![2, 2]);
                assert_eq!(array.data, vec![1, 1, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn or_scalar_broadcasts() {
        let tensor = Tensor::new(vec![1.0, 0.0, 3.0, 0.0], vec![4, 1]).unwrap();
        let result = or_builtin(Value::Tensor(tensor), Value::Int(IntValue::I32(0))).unwrap();
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![4, 1]);
                assert_eq!(array.data, vec![1, 0, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn or_char_arrays() {
        let lhs = CharArray::new(vec!['R', 'u', '\0'], 1, 3).unwrap();
        let rhs = CharArray::new(vec!['R', '\0', 'n'], 1, 3).unwrap();
        let result =
            or_builtin(Value::CharArray(lhs), Value::CharArray(rhs)).expect("or char arrays");
        match result {
            Value::LogicalArray(arr) => {
                assert_eq!(arr.shape, vec![1, 3]);
                assert_eq!(arr.data, vec![1, 1, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn or_treats_nan_as_true() {
        let result = or_builtin(Value::Num(f64::NAN), Value::Num(0.0)).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn or_complex_inputs() {
        let result = or_builtin(Value::Complex(0.0, 0.0), Value::Complex(0.0, 0.0)).unwrap();
        assert_eq!(result, Value::Bool(false));

        let result = or_builtin(Value::Complex(0.0, 0.0), Value::Complex(0.0, 2.0)).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn or_size_mismatch_errors() {
        let lhs = Tensor::new(vec![1.0, 0.0, 2.0, 0.0], vec![2, 2]).unwrap();
        let rhs = Tensor::new(vec![1.0, 0.0, 3.0], vec![3, 1]).unwrap();
        let err = or_builtin(Value::Tensor(lhs), Value::Tensor(rhs)).unwrap_err();
        assert!(err.contains("size mismatch"), "unexpected error: {err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn or_rejects_unsupported_types() {
        let err = or_builtin(Value::String("runmat".into()), Value::Bool(true)).unwrap_err();
        assert!(
            err.contains("unsupported input type"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn or_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![0.0, 2.0, 0.0, 4.0], vec![2, 2]).unwrap();
            let rhs = Tensor::new(vec![1.0, 0.0, 3.0, 0.0], vec![2, 2]).unwrap();
            let lhs_view = HostTensorView {
                data: &lhs.data,
                shape: &lhs.shape,
            };
            let rhs_view = HostTensorView {
                data: &rhs.data,
                shape: &rhs.shape,
            };
            let a = provider.upload(&lhs_view).unwrap();
            let b = provider.upload(&rhs_view).unwrap();
            let result = or_builtin(Value::GpuTensor(a), Value::GpuTensor(b)).unwrap();
            let gathered = test_support::gather(result).unwrap();
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![1.0, 1.0, 1.0, 1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn or_gpu_supports_broadcast() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![0.0, 2.0, 0.0, 4.0], vec![4, 1]).unwrap();
            let rhs = Tensor::new(vec![0.0], vec![1, 1]).unwrap();

            let lhs_view = HostTensorView {
                data: &lhs.data,
                shape: &lhs.shape,
            };
            let rhs_view = HostTensorView {
                data: &rhs.data,
                shape: &rhs.shape,
            };

            let gpu_lhs = provider.upload(&lhs_view).expect("upload lhs");
            let gpu_rhs = provider.upload(&rhs_view).expect("upload rhs");

            let result =
                or_builtin(Value::GpuTensor(gpu_lhs), Value::GpuTensor(gpu_rhs)).expect("or");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.data, vec![0.0, 1.0, 0.0, 1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn or_wgpu_matches_host_path() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider registered");

        let lhs = Tensor::new(vec![0.0, 1.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let rhs = Tensor::new(vec![1.0, 0.0, 3.0, 4.0], vec![2, 2]).unwrap();

        let cpu_value =
            or_host(Value::Tensor(lhs.clone()), Value::Tensor(rhs.clone())).expect("host or");
        let (expected_data, expected_shape) = match cpu_value {
            Value::LogicalArray(arr) => (arr.data.clone(), arr.shape.clone()),
            other => panic!("expected logical array, got {other:?}"),
        };

        let view_lhs = HostTensorView {
            data: &lhs.data,
            shape: &lhs.shape,
        };
        let view_rhs = HostTensorView {
            data: &rhs.data,
            shape: &rhs.shape,
        };
        let gpu_lhs = provider.upload(&view_lhs).expect("upload lhs");
        let gpu_rhs = provider.upload(&view_rhs).expect("upload rhs");

        let gpu_value =
            or_builtin(Value::GpuTensor(gpu_lhs), Value::GpuTensor(gpu_rhs)).expect("gpu or");
        let gathered = test_support::gather(gpu_value).expect("gather gpu result");

        assert_eq!(gathered.shape, expected_shape);
        let tol = match provider.precision() {
            ProviderPrecision::F64 => 1e-12,
            ProviderPrecision::F32 => 1e-5,
        };
        for (idx, (actual, expected)) in gathered.data.iter().zip(expected_data.iter()).enumerate()
        {
            let expected_f = if *expected != 0 { 1.0 } else { 0.0 };
            assert!(
                (actual - expected_f).abs() <= tol,
                "mismatch at index {idx}: got {actual}, expected {expected_f}"
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
