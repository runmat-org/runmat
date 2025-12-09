//! MATLAB-compatible logical `not` builtin with GPU support.

use runmat_builtins::{CharArray, ComplexTensor, LogicalArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg(feature = "doc_export")]
#[runmat_macros::register_doc_text(name = "not")]
pub const DOC_MD: &str = r#"---
title: "not"
category: "logical/bit"
keywords: ["logical not", "boolean negation", "gpuArray not", "MATLAB not", "logical invert"]
summary: "Element-wise logical negation for scalars, arrays, and gpuArray values."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Executes on the GPU when the provider implements `logical_not`; otherwise inputs gather to the host transparently."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::logical::bit::not::tests"
  integration: "builtins::logical::bit::not::tests::not_gpu_roundtrip"
  gpu: "builtins::logical::bit::not::tests::not_wgpu_matches_host_path"
---

# What does the `not` function do in MATLAB / RunMat?
`not(X)` inverts the logical interpretation of every element in `X`. Values that evaluate to `true` become `false`, and vice versa. MATLAB treats any non-zero (or non-empty) numeric or complex value as `true`, including `NaN`.

## How does the `not` function behave in MATLAB / RunMat?
- Works on scalars, vectors, matrices, and N-D tensors with MATLAB broadcasting semantics.
- Accepts logical, numeric, complex, and character arrays. Character code points equal to zero become `true`, all others become `false`.
- Returns a logical scalar when the input has exactly one element; otherwise the result is a logical array matching the input shape.
- Honors gpuArray residency. If the active acceleration provider exposes `logical_not`, the entire operation runs on the GPU; otherwise RunMat falls back to the CPU path automatically.
- `NaN` evaluates to `true`, so `not(NaN)` produces `false`, consistent with MATLAB.

## `not` Function GPU Execution Behaviour
When RunMat Accelerate is active, `not` dispatches to the provider hook `logical_not`. Providers write `0` or `1` into a device buffer, keeping the result resident on the GPU. If the provider does not implement the hook, RunMat gathers the input to host memory, executes the CPU implementation, and (if the caller passed a `gpuArray`) returns a logical array on the host so the call never fails.

## Examples of using the `not` function in MATLAB / RunMat

### Checking if a scalar value is zero

```matlab
result = not(5)
```

Expected output:

```matlab
result =
     0
```

### Negating a logical mask to find the complement

```matlab
mask = [true false true];
inverseMask = not(mask)
```

Expected output:

```matlab
inverseMask =
  1×3 logical array
     0     1     0
```

### Turning nonzero numeric entries into false values

```matlab
A = [0 1 2 0];
B = not(A)
```

Expected output:

```matlab
B =
  1×4 logical array
     1     0     0     1
```

### Flipping zero and nonzero character codes

```matlab
chars = ['A' 0 'C'];
flags = not(chars)
```

Expected output:

```matlab
flags =
  1×3 logical array
     0     1     0
```

### Performing logical NOT directly on the GPU

```matlab
G = gpuArray([0 4 0 9]);
deviceResult = not(G);
hostResult = gather(deviceResult)
```

Expected output:

```matlab
deviceResult =
  1×4 gpuArray logical array
     1     0     1     0
hostResult =
  1×4 logical array
     1     0     1     0
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** have to call `gpuArray` manually. RunMat's auto-offload planner moves data to the GPU when a fused expression benefits from device execution. The `not` builtin preserves existing residency; results remain on the GPU until you gather them or another operation requires host access. Use `gpuArray` when porting MATLAB code that already does so explicitly or when you want to pin tensors to the GPU ahead of time.

## FAQ

### Does `not` return logical values?
Yes. Scalar inputs yield logical scalars (`true`/`false`). Array inputs produce logical arrays where each element is either `0` or `1`.

### How does `not` treat `NaN` or complex numbers?
`NaN` and complex numbers with any non-zero component evaluate as `true`, so `not(NaN)` and `not(1+2i)` return `false`.

### Can I pass a `gpuArray` to `not`?
Absolutely. If the provider implements `logical_not`, the negation runs entirely on the GPU. Otherwise the runtime gathers to the host, performs the operation, and returns a logical array.

### What happens with empty arrays?
Empty inputs produce empty logical outputs with matching shape, preserving MATLAB's empty propagation semantics.

### Is there a difference between `not(X)` and `~X`?
No. They share the same element-wise semantics. The functional form is convenient for higher-order APIs or when passing the operator as a handle.

### Does `not` modify the input in place?
No. It returns a new logical value. When operating on gpuArrays, the provider writes into a fresh buffer so the original data remains unchanged.

## See Also
[and](./and), [or](./or), [xor](./xor), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "not",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary {
        name: "logical_not",
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Dispatches to the provider `logical_not` hook when available; otherwise the runtime gathers to host and performs the negation on the CPU.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "not",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            let (zero, one) = match ctx.scalar_ty {
                ScalarType::F32 => ("0.0".to_string(), "1.0".to_string()),
                ScalarType::F64 => ("f64(0.0)".to_string(), "f64(1.0)".to_string()),
                _ => return Err(FusionError::UnsupportedPrecision(ctx.scalar_ty)),
            };
            let cond = format!("({input} != {zero})");
            Ok(format!("select({one}, {zero}, {cond})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion kernels treat any non-zero input as true and write 0/1 outputs, matching MATLAB logical semantics.",
};

#[runtime_builtin(
    name = "not",
    category = "logical/bit",
    summary = "Element-wise logical negation for scalars, arrays, and gpuArray values.",
    keywords = "logical,not,boolean,gpu",
    accel = "elementwise"
)]
fn not_builtin(value: Value) -> Result<Value, String> {
    if let Value::GpuTensor(ref handle) = value {
        if let Some(provider) = runmat_accelerate_api::provider() {
            if let Ok(device_out) = provider.logical_not(handle) {
                return Ok(gpu_helpers::logical_gpu_value(device_out));
            }
        }
    }
    not_host(value)
}

fn not_host(value: Value) -> Result<Value, String> {
    let buffer = logical_buffer_from("not", value)?;
    let LogicalBuffer { data, shape } = buffer;
    let total = tensor::element_count(&shape);
    if total == 0 {
        return logical_value("not", Vec::new(), shape);
    }
    let mapped = data
        .into_iter()
        .map(|bit| if bit == 0 { 1 } else { 0 })
        .collect::<Vec<_>>();
    logical_value("not", mapped, shape)
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
    let mapped = data
        .into_iter()
        .map(|v| if v != 0.0 { 1 } else { 0 })
        .collect();
    Ok(LogicalBuffer {
        data: mapped,
        shape,
    })
}

fn complex_tensor_to_logical_buffer(tensor: ComplexTensor) -> Result<LogicalBuffer, String> {
    let ComplexTensor { data, shape, .. } = tensor;
    let mapped = data
        .into_iter()
        .map(|(re, im)| if re != 0.0 || im != 0.0 { 1 } else { 0 })
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
mod tests {
    use super::*;
    #[cfg(feature = "wgpu")]
    use crate::builtins::common::tensor;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::ProviderPrecision;
    use runmat_builtins::{CharArray, ComplexTensor, IntValue, LogicalArray, Tensor};

    #[test]
    fn not_of_booleans() {
        assert_eq!(not_builtin(Value::Bool(true)).unwrap(), Value::Bool(false));
        assert_eq!(not_builtin(Value::Bool(false)).unwrap(), Value::Bool(true));
    }

    #[test]
    fn not_numeric_array() {
        let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 0.0], vec![2, 2]).unwrap();
        let result = not_builtin(Value::Tensor(tensor)).unwrap();
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![2, 2]);
                assert_eq!(array.data, vec![1, 0, 0, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn not_complex_scalar() {
        let result =
            not_builtin(Value::Complex(0.0, 0.0)).expect("not complex zero should succeed");
        assert_eq!(result, Value::Bool(true));

        let result =
            not_builtin(Value::Complex(1.0, 0.0)).expect("not complex nonzero should succeed");
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn not_nan_yields_false() {
        let result = not_builtin(Value::Num(f64::NAN)).unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn not_char_array() {
        let chars = CharArray::new_row("A\0C");
        let result = not_builtin(Value::CharArray(chars)).unwrap();
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![1, 3]);
                assert_eq!(array.data, vec![0, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn not_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 0.0, 2.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = not_builtin(Value::GpuTensor(handle)).expect("not on gpu");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![1.0, 0.0, 1.0, 0.0]);
        });
    }

    #[test]
    fn not_accepts_int_inputs() {
        let value = Value::Int(IntValue::I32(0));
        assert_eq!(not_builtin(value).unwrap(), Value::Bool(true));
    }

    #[test]
    fn not_tensor_scalar_returns_bool() {
        let tensor = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        assert_eq!(
            not_builtin(Value::Tensor(tensor)).unwrap(),
            Value::Bool(false)
        );

        let tensor = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        assert_eq!(
            not_builtin(Value::Tensor(tensor)).unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn not_empty_tensor_preserves_shape() {
        let tensor = Tensor::new(Vec::<f64>::new(), vec![0, 3]).unwrap();
        let result = not_builtin(Value::Tensor(tensor)).unwrap();
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![0, 3]);
                assert!(array.data.is_empty());
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn not_complex_tensor() {
        let tensor =
            ComplexTensor::new(vec![(0.0, 0.0), (1.0, 0.0), (0.0, -2.0)], vec![3, 1]).unwrap();
        let result = not_builtin(Value::ComplexTensor(tensor)).unwrap();
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![3, 1]);
                assert_eq!(array.data, vec![1, 0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn not_logical_array_flips_bits() {
        let array = LogicalArray::new(vec![1, 0, 1, 1], vec![2, 2]).unwrap();
        let result = not_builtin(Value::LogicalArray(array)).unwrap();
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data, vec![0, 1, 0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn not_rejects_string_input() {
        let err = not_builtin(Value::String("abc".into())).unwrap_err();
        assert!(
            err.contains("unsupported input type"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn not_wgpu_matches_host_path() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0, 3.0, 0.0, -1.0], vec![2, 2]).unwrap();
        let cpu = not_host(Value::Tensor(tensor.clone())).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = not_builtin(Value::GpuTensor(handle)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        let cpu_tensor = tensor::value_to_tensor(&cpu).expect("cpu tensor");
        assert_eq!(gathered.shape, cpu_tensor.shape);
        let tol = match runmat_accelerate_api::provider().unwrap().precision() {
            ProviderPrecision::F64 => 1e-12,
            ProviderPrecision::F32 => 1e-5,
        };
        for (expected, actual) in cpu_tensor.data.iter().zip(gathered.data.iter()) {
            assert!((*expected - *actual).abs() < tol, "{expected} vs {actual}");
        }
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
