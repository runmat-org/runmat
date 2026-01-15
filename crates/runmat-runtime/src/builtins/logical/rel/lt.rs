//! MATLAB-compatible `lt` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{CharArray, LogicalArray, StringArray, Tensor, Value};
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
        name = "lt",
        builtin_path = "crate::builtins::logical::rel::lt"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "lt"
category: "logical/rel"
keywords: ["lt", "<", "less than", "logical comparison", "gpuArray less-than"]
summary: "Element-wise less-than comparison for scalars, arrays, strings, and gpuArray inputs."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Uses provider elem_lt kernels when available; otherwise inputs gather back to host memory transparently."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::logical::rel::lt::tests"
  integration: "builtins::logical::rel::lt::tests::lt_gpu_provider_roundtrip"
  gpu: "builtins::logical::rel::lt::tests::lt_wgpu_matches_host"
---

# What does the `lt` function do in MATLAB / RunMat?
`lt(A, B)` (or the infix `A < B`) performs an element-wise less-than comparison. The result is a logical scalar when the broadcasted shape contains one element, or a logical array otherwise.

## How does the `lt` function behave in MATLAB / RunMat?
- Numeric, logical, and character inputs compare element-wise using MATLAB's implicit expansion rules.
- Character arrays compare by Unicode code point; mixing them with numeric arrays behaves like comparing numeric codes (`'A' < 66`).
- String scalars and string arrays compare lexically; implicit expansion works across string dimensions.
- Complex inputs are not supported, matching MATLAB's behaviour.
- Mixed numeric/string inputs raise MATLAB-compatible type errors.

## `lt` Function GPU Execution Behaviour
When both operands are `gpuArray` values and the active acceleration provider implements the `elem_lt` hook, RunMat executes the comparison entirely on the device and returns a `gpuArray` logical result. If the provider does not expose this hook, the runtime gathers the inputs to host memory automatically and performs the CPU comparison instead of failing.

## Examples of using the `lt` function in MATLAB / RunMat

### Determine Whether One Scalar Is Less Than Another

```matlab
flag = lt(17, 42);
```

Expected output:

```matlab
flag =
     1
```

### Filter Matrix Elements Below A Threshold

```matlab
M = [1 2 3; 4 5 6];
mask = lt(M, 3);
```

Expected output:

```matlab
mask =
  2×3 logical array
     1     1     0
     0     0     0
```

### Apply Less-Than With Implicit Expansion

```matlab
v = [1 3 5 7];
mask = lt(v, [2 6]);
```

Expected output:

```matlab
mask =
  1×4 logical array
     1     0     0     0
```

### Compare Character Data To Numeric Codes

```matlab
letters = ['A' 'B' 'C'];
isBeforeC = lt(letters, 67);
```

Expected output:

```matlab
isBeforeC =
  1×3 logical array
     1     1     0
```

### Compare String Arrays Lexicographically

```matlab
names = ["alice" "charlie" "bob"];
earlier = lt(names, "bob");
```

Expected output:

```matlab
earlier =
  1×3 logical array
     1     0     0
```

### Run `lt` Directly On `gpuArray` Inputs

```matlab
G1 = gpuArray([1 4 7]);
G2 = gpuArray([2 4 8]);
deviceResult = lt(G1, G2);
hostResult = gather(deviceResult);
```

Expected output:

```matlab
deviceResult =
  1×3 gpuArray logical array
     1     0     1
hostResult =
  1×3 logical array
     1     0     1
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You usually do **not** need to call `gpuArray` explicitly. RunMat's native auto-offload planner keeps intermediate results on the GPU when fused expressions benefit from device execution. Explicit `gpuArray` and `gather` calls remain available for compatibility with MATLAB code that manages residency manually.

## FAQ

### Does `lt` return logical values?
Yes. Scalars return `true` or `false`. Arrays return logical arrays, and `gpuArray` inputs return `gpuArray` logical outputs.

### How are NaN values treated?
Any comparison involving `NaN` returns `false`, matching MATLAB behaviour.

### Can I compare complex numbers with `lt`?
No. MATLAB does not define relational ordering for complex numbers, so RunMat raises a MATLAB-compatible error when complex inputs are supplied.

### How are strings compared?
String scalars and arrays compare lexicographically using Unicode code points, with full support for implicit expansion against scalar strings.

### Are character vectors treated as numbers or text?
Character arrays participate as numeric code points when compared to numeric inputs, and they are converted to strings when compared against string scalars or arrays.

### Do I need to gather results manually after a GPU comparison?
No. When both inputs are `gpuArray` values and the provider supports `elem_lt`, the result stays on the GPU. Otherwise, RunMat gathers inputs transparently and returns a host logical array.

### Does implicit expansion apply to string arrays?
Yes. String arrays support MATLAB-style implicit expansion, so you can compare against scalar strings without manual replication.

### Can I fuse `lt` inside GPU expressions?
Yes. The builtin registers element-wise fusion metadata so the planner can fuse comparisons with surrounding GPU-friendly operations.

## See Also
[gt](./gt), [ge](./ge), [eq](./eq), [ne](./ne)

## Source & Feedback
- The full source code for the implementation of the `lt` function is available at: [`crates/runmat-runtime/src/builtins/logical/rel/lt.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/rel/lt.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::rel::lt")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "lt",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Binary {
        name: "elem_lt",
        commutative: false,
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Prefers provider elem_lt kernels when available; otherwise inputs gather to host tensors automatically.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::rel::lt")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "lt",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let lhs = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            let rhs = ctx.inputs.get(1).ok_or(FusionError::MissingInput(1))?;
            let (zero, one) = match ctx.scalar_ty {
                ScalarType::F32 => ("0.0", "1.0"),
                ScalarType::F64 => ("f64(0.0)", "f64(1.0)"),
                _ => return Err(FusionError::UnsupportedPrecision(ctx.scalar_ty)),
            };
            Ok(format!("select({zero}, {one}, ({lhs} < {rhs}))"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes:
        "Fusion emits comparison kernels that write 1 when the left operand is less than the right.",
};

#[runtime_builtin(
    name = "lt",
    category = "logical/rel",
    summary = "Element-wise less-than comparison for scalars, arrays, and gpuArray inputs.",
    keywords = "lt,less than,comparison,logical,gpu",
    accel = "elementwise",
    builtin_path = "crate::builtins::logical::rel::lt"
)]
fn lt_builtin(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    if let (Value::GpuTensor(ref a), Value::GpuTensor(ref b)) = (&lhs, &rhs) {
        if let Some(result) = try_lt_gpu(a, b) {
            return (result).map_err(Into::into);
        }
    }
    lt_host(lhs, rhs).map_err(Into::into)
}

fn try_lt_gpu(a: &GpuTensorHandle, b: &GpuTensorHandle) -> Option<Result<Value, String>> {
    let provider = runmat_accelerate_api::provider()?;
    match provider.elem_lt(a, b) {
        Ok(handle) => Some(Ok(gpu_helpers::logical_gpu_value(handle))),
        Err(err) => {
            drop(err);
            None
        }
    }
}

fn lt_host(lhs: Value, rhs: Value) -> Result<Value, String> {
    let (lhs, rhs) = normalize_char_string(lhs, rhs);

    let left = LtOperand::from_value(lhs)?;
    let right = LtOperand::from_value(rhs)?;

    match (left, right) {
        (LtOperand::Numeric(a), LtOperand::Numeric(b)) => {
            let (data, shape) = numeric_lt(&a, &b)?;
            logical_result(data, shape)
        }
        (LtOperand::String(a), LtOperand::String(b)) => {
            let (data, shape) = string_lt(&a, &b)?;
            logical_result(data, shape)
        }
        (LtOperand::Numeric(_), LtOperand::String(_))
        | (LtOperand::String(_), LtOperand::Numeric(_)) => {
            Err("lt: mixing numeric and string inputs is not supported".to_string())
        }
    }
}

fn normalize_char_string(lhs: Value, rhs: Value) -> (Value, Value) {
    match (lhs, rhs) {
        (Value::CharArray(ca), Value::String(s)) => {
            let text: String = ca.data.into_iter().collect();
            (Value::String(text), Value::String(s))
        }
        (Value::String(s), Value::CharArray(ca)) => {
            let text: String = ca.data.into_iter().collect();
            (Value::String(s), Value::String(text))
        }
        (Value::CharArray(ca), Value::StringArray(sa)) => {
            let text: String = ca.data.into_iter().collect();
            (Value::String(text), Value::StringArray(sa))
        }
        (Value::StringArray(sa), Value::CharArray(ca)) => {
            let text: String = ca.data.into_iter().collect();
            (Value::StringArray(sa), Value::String(text))
        }
        (lhs, rhs) => (lhs, rhs),
    }
}

fn logical_result(data: Vec<u8>, shape: Vec<usize>) -> Result<Value, String> {
    if tensor::element_count(&shape) <= 1 && data.len() == 1 {
        Ok(Value::Bool(data[0] != 0))
    } else {
        LogicalArray::new(data, shape)
            .map(Value::LogicalArray)
            .map_err(|e| format!("lt: {e}"))
    }
}

enum LtOperand {
    Numeric(NumericBuffer),
    String(StringBuffer),
}

impl LtOperand {
    fn from_value(value: Value) -> Result<Self, String> {
        match value {
            Value::Num(n) => Ok(LtOperand::Numeric(NumericBuffer::scalar(n))),
            Value::Bool(flag) => Ok(LtOperand::Numeric(NumericBuffer::scalar(if flag {
                1.0
            } else {
                0.0
            }))),
            Value::Int(i) => Ok(LtOperand::Numeric(NumericBuffer::scalar(i.to_f64()))),
            Value::Tensor(tensor) => Ok(LtOperand::Numeric(NumericBuffer::from_tensor(tensor))),
            Value::LogicalArray(array) => {
                Ok(LtOperand::Numeric(NumericBuffer::from_logical(array)))
            }
            Value::CharArray(array) => {
                Ok(LtOperand::Numeric(NumericBuffer::from_char_array(array)))
            }
            Value::String(s) => Ok(LtOperand::String(StringBuffer::scalar(s))),
            Value::StringArray(sa) => Ok(LtOperand::String(StringBuffer::from_array(sa))),
            Value::GpuTensor(handle) => {
                let tensor = gpu_helpers::gather_tensor(&handle)?;
                Ok(LtOperand::Numeric(NumericBuffer::from_tensor(tensor)))
            }
            Value::Complex(_, _) | Value::ComplexTensor(_) => {
                Err("lt: complex inputs are not supported".to_string())
            }
            unsupported => Err(format!("lt: unsupported input type {unsupported:?}")),
        }
    }
}

fn numeric_lt(lhs: &NumericBuffer, rhs: &NumericBuffer) -> Result<(Vec<u8>, Vec<usize>), String> {
    let shape = broadcast_shapes("lt", &lhs.shape, &rhs.shape)?;
    let total = tensor::element_count(&shape);
    if total == 0 {
        return Ok((Vec::new(), shape));
    }
    let strides_l = compute_strides(&lhs.shape);
    let strides_r = compute_strides(&rhs.shape);
    let mut out = Vec::with_capacity(total);
    for idx in 0..total {
        let lhs_val = if lhs.data.is_empty() {
            0.0
        } else {
            let offset = broadcast_index(idx, &shape, &lhs.shape, &strides_l);
            lhs.data[offset]
        };
        let rhs_val = if rhs.data.is_empty() {
            0.0
        } else {
            let offset = broadcast_index(idx, &shape, &rhs.shape, &strides_r);
            rhs.data[offset]
        };
        out.push(if lhs_val < rhs_val { 1 } else { 0 });
    }
    Ok((out, shape))
}

fn string_lt(lhs: &StringBuffer, rhs: &StringBuffer) -> Result<(Vec<u8>, Vec<usize>), String> {
    let shape = broadcast_shapes("lt", &lhs.shape, &rhs.shape)?;
    let total = tensor::element_count(&shape);
    if total == 0 {
        return Ok((Vec::new(), shape));
    }
    let strides_l = compute_strides(&lhs.shape);
    let strides_r = compute_strides(&rhs.shape);
    let mut out = Vec::with_capacity(total);
    for idx in 0..total {
        let lhs_val = if lhs.data.is_empty() {
            ""
        } else {
            let offset = broadcast_index(idx, &shape, &lhs.shape, &strides_l);
            lhs.data[offset].as_str()
        };
        let rhs_val = if rhs.data.is_empty() {
            ""
        } else {
            let offset = broadcast_index(idx, &shape, &rhs.shape, &strides_r);
            rhs.data[offset].as_str()
        };
        out.push(if lhs_val < rhs_val { 1 } else { 0 });
    }
    Ok((out, shape))
}

#[derive(Debug)]
struct NumericBuffer {
    data: Vec<f64>,
    shape: Vec<usize>,
}

impl NumericBuffer {
    fn scalar(value: f64) -> Self {
        Self {
            data: vec![value],
            shape: vec![1, 1],
        }
    }

    fn from_tensor(tensor: Tensor) -> Self {
        Self {
            data: tensor.data,
            shape: tensor.shape,
        }
    }

    fn from_logical(array: LogicalArray) -> Self {
        let shape = array.shape.clone();
        let data = array
            .data
            .into_iter()
            .map(|b| if b != 0 { 1.0 } else { 0.0 })
            .collect();
        Self { data, shape }
    }

    fn from_char_array(array: CharArray) -> Self {
        let rows = array.rows;
        let cols = array.cols;
        if rows == 0 || cols == 0 {
            return Self {
                data: Vec::new(),
                shape: vec![rows, cols],
            };
        }
        let mut data = Vec::with_capacity(rows * cols);
        for c in 0..cols {
            for r in 0..rows {
                let idx = r * cols + c;
                let ch = array.data[idx];
                data.push(ch as u32 as f64);
            }
        }
        Self {
            data,
            shape: vec![rows, cols],
        }
    }
}

#[derive(Debug)]
struct StringBuffer {
    data: Vec<String>,
    shape: Vec<usize>,
}

impl StringBuffer {
    fn scalar(value: String) -> Self {
        Self {
            data: vec![value],
            shape: vec![1, 1],
        }
    }

    fn from_array(array: StringArray) -> Self {
        Self {
            data: array.data,
            shape: array.shape,
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lt_scalar_true() {
        let result = lt_builtin(Value::Num(3.0), Value::Num(4.0)).expect("lt");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lt_scalar_false() {
        let result = lt_builtin(Value::Num(4.0), Value::Num(3.0)).expect("lt");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lt_vector_broadcast() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![1, 4]).unwrap();
        let result = lt_builtin(Value::Tensor(tensor), Value::Num(3.0)).expect("lt");
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![1, 4]);
                assert_eq!(array.data, vec![1, 0, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lt_char_array_against_numeric() {
        let chars = CharArray::new(vec!['A', 'B', 'C'], 1, 3).unwrap();
        let tensor = Tensor::new(vec![66.0, 66.0, 66.0], vec![1, 3]).unwrap();
        let result = lt_builtin(Value::CharArray(chars), Value::Tensor(tensor)).expect("lt");
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![1, 3]);
                assert_eq!(array.data, vec![1, 0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lt_string_array_against_scalar() {
        let array = StringArray::new(vec!["apple".into(), "carrot".into()], vec![1, 2]).unwrap();
        let result =
            lt_builtin(Value::StringArray(array), Value::String("banana".into())).expect("lt");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![1, 2]);
                assert_eq!(mask.data, vec![1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lt_string_numeric_error() {
        let err =
            lt_builtin(Value::String("apple".into()), Value::Num(3.0)).expect_err("expected error");
        assert!(
            err.contains("mixing numeric and string"),
            "unexpected message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lt_complex_error() {
        let err = lt_builtin(Value::Complex(1.0, 1.0), Value::Num(0.0)).expect_err("lt");
        assert!(
            err.contains("complex"),
            "expected complex error message, got {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lt_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![1.0, 4.0, 7.0], vec![1, 3]).unwrap();
            let rhs = Tensor::new(vec![2.0, 4.0, 8.0], vec![1, 3]).unwrap();
            let view_l = HostTensorView {
                data: &lhs.data,
                shape: &lhs.shape,
            };
            let view_r = HostTensorView {
                data: &rhs.data,
                shape: &rhs.shape,
            };
            let handle_l = provider.upload(&view_l).expect("upload lhs");
            let handle_r = provider.upload(&view_r).expect("upload rhs");
            let result =
                lt_builtin(Value::GpuTensor(handle_l), Value::GpuTensor(handle_r)).expect("lt");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 3]);
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
    fn lt_wgpu_matches_host() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let lhs = Tensor::new(vec![0.0, 2.0, 5.0, 7.0], vec![4, 1]).unwrap();
        let rhs = Tensor::new(vec![1.0, 2.5, 4.0, 8.0], vec![4, 1]).unwrap();
        let cpu = lt_host(Value::Tensor(lhs.clone()), Value::Tensor(rhs.clone())).unwrap();

        let view_l = HostTensorView {
            data: &lhs.data,
            shape: &lhs.shape,
        };
        let view_r = HostTensorView {
            data: &rhs.data,
            shape: &rhs.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("provider");
        let handle_l = provider.upload(&view_l).expect("upload lhs");
        let handle_r = provider.upload(&view_r).expect("upload rhs");
        let gpu = lt_builtin(Value::GpuTensor(handle_l), Value::GpuTensor(handle_r)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");

        match (cpu, gathered) {
            (Value::LogicalArray(host), tensor) => {
                assert_eq!(tensor.shape, host.shape);
                let expected: Vec<f64> = host
                    .data
                    .iter()
                    .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                    .collect();
                assert_eq!(tensor.data, expected);
            }
            (Value::Bool(host_flag), tensor) => {
                assert_eq!(tensor.shape, vec![1, 1]);
                let expected = if host_flag { 1.0 } else { 0.0 };
                assert_eq!(tensor.data, vec![expected]);
            }
            other => panic!("unexpected output combination: {other:?}"),
        }
    }
}
