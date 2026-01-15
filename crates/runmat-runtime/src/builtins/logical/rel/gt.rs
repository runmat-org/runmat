//! MATLAB-compatible `gt` builtin with GPU-aware semantics for RunMat.

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
        name = "gt",
        builtin_path = "crate::builtins::logical::rel::gt"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "gt"
category: "logical/rel"
keywords: ["gt", ">", "greater than", "logical comparison", "gpuArray greater-than"]
summary: "Element-wise greater-than comparison for scalars, arrays, strings, and gpuArray inputs."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Uses provider elem_gt kernels when available; otherwise inputs gather back to host memory transparently."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::logical::rel::gt::tests"
  integration: "builtins::logical::rel::gt::tests::gt_gpu_provider_roundtrip"
  gpu: "builtins::logical::rel::gt::tests::gt_wgpu_matches_host"
---

# What does the `gt` function do in MATLAB / RunMat?
`gt(A, B)` (or the infix `A > B`) performs an element-wise greater-than comparison. The result is a logical scalar when the broadcasted shape contains one element, or a logical array otherwise.

## How does the `gt` function behave in MATLAB / RunMat?
- Numeric, logical, and character inputs compare element-wise using MATLAB's implicit expansion rules.
- Character arrays compare by Unicode code point; mixing them with numeric arrays behaves like comparing numeric codes (`'A' > 60`).
- String scalars and string arrays compare lexically; implicit expansion works across string dimensions.
- Complex inputs are not supported, matching MATLAB's behaviour.
- Mixed numeric/string inputs raise MATLAB-compatible type errors.

## `gt` Function GPU Execution Behaviour
When both operands are `gpuArray` values and the active acceleration provider implements the `elem_gt` hook, RunMat executes the comparison entirely on the device and returns a `gpuArray` logical result. If the provider does not expose this hook, the runtime gathers the inputs to host memory automatically and performs the CPU comparison instead of failing.

## Examples of using the `gt` function in MATLAB / RunMat

### Determine Whether One Scalar Exceeds Another

```matlab
flag = gt(42, 17);
```

Expected output:

```matlab
flag =
     1
```

### Filter Matrix Elements Greater Than a Threshold

```matlab
M = [1 2 3; 4 5 6];
mask = gt(M, 3);
```

Expected output:

```matlab
mask =
  2×3 logical array
     0     0     0
     1     1     1
```

### Apply Greater-Than With Implicit Expansion

```matlab
v = [1 3 5 7];
mask = gt(v, [2 6]);
```

Expected output:

```matlab
mask =
  1×4 logical array
     0     0     1     1
```

### Compare Character Data to Numeric Codes

```matlab
letters = ['A' 'B' 'C'];
isAfterB = gt(letters, 66);
```

Expected output:

```matlab
isAfterB =
  1×3 logical array
     0     0     1
```

### Compare String Arrays Lexicographically

```matlab
names = ["alice" "charlie" "bob"];
later = gt(names, "bob");
```

Expected output:

```matlab
later =
  1×3 logical array
     0     1     0
```

### Run `gt` Directly On `gpuArray` Inputs

```matlab
G1 = gpuArray([1 4 7]);
G2 = gpuArray([0 5 6]);
deviceResult = gt(G1, G2);
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

### Does `gt` return logical values?
Yes. Scalars return `true` or `false`. Arrays return logical arrays, and `gpuArray` inputs return `gpuArray` logical outputs.

### How are NaN values treated?
Any comparison involving `NaN` returns `false`, matching MATLAB behaviour.

### Can I compare complex numbers with `gt`?
No. MATLAB does not define relational ordering for complex numbers, so RunMat raises a MATLAB-compatible error when complex inputs are supplied.

### How are strings compared?
String scalars and arrays compare lexicographically using Unicode code points, with full support for implicit expansion against scalar strings.

### Are character vectors treated as numbers or text?
Character arrays participate as numeric code points when compared to numeric inputs, and they are converted to strings when compared against string scalars or arrays.

### Do I need to gather results manually after a GPU comparison?
No. When both inputs are `gpuArray` values and the provider supports `elem_gt`, the result stays on the GPU. Otherwise, RunMat gathers inputs transparently and returns a host logical array.

### Does implicit expansion apply to string arrays?
Yes. String arrays support MATLAB-style implicit expansion, so you can compare against scalar strings without manual replication.

### Can I fuse `gt` inside GPU expressions?
Yes. The builtin registers element-wise fusion metadata so the planner can fuse comparisons with surrounding GPU-friendly operations.

## See Also
[eq](./eq), [ne](./ne)

## Source & Feedback
- The full source code for the implementation of the `gt` function is available at: [`crates/runmat-runtime/src/builtins/logical/rel/gt.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/rel/gt.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::rel::gt")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "gt",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Binary {
        name: "elem_gt",
        commutative: false,
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Prefers provider elem_gt kernels when available; otherwise inputs gather to host tensors automatically.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::rel::gt")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "gt",
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
            Ok(format!("select({zero}, {one}, ({lhs} > {rhs}))"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits comparison kernels that write 1 when the left operand is greater than the right.",
};

#[runtime_builtin(
    name = "gt",
    category = "logical/rel",
    summary = "Element-wise greater-than comparison for scalars, arrays, and gpuArray inputs.",
    keywords = "gt,greater than,comparison,logical,gpu",
    accel = "elementwise",
    builtin_path = "crate::builtins::logical::rel::gt"
)]
fn gt_builtin(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    if let (Value::GpuTensor(ref a), Value::GpuTensor(ref b)) = (&lhs, &rhs) {
        if let Some(result) = try_gt_gpu(a, b) {
            return (result).map_err(Into::into);
        }
    }
    gt_host(lhs, rhs).map_err(Into::into)
}

fn try_gt_gpu(a: &GpuTensorHandle, b: &GpuTensorHandle) -> Option<Result<Value, String>> {
    let provider = runmat_accelerate_api::provider()?;
    match provider.elem_gt(a, b) {
        Ok(handle) => Some(Ok(gpu_helpers::logical_gpu_value(handle))),
        Err(err) => {
            drop(err);
            None
        }
    }
}

fn gt_host(lhs: Value, rhs: Value) -> Result<Value, String> {
    let (lhs, rhs) = normalize_char_string(lhs, rhs);

    let left = GtOperand::from_value(lhs)?;
    let right = GtOperand::from_value(rhs)?;

    match (left, right) {
        (GtOperand::Numeric(a), GtOperand::Numeric(b)) => {
            let (data, shape) = numeric_gt(&a, &b)?;
            logical_result(data, shape)
        }
        (GtOperand::String(a), GtOperand::String(b)) => {
            let (data, shape) = string_gt(&a, &b)?;
            logical_result(data, shape)
        }
        (GtOperand::Numeric(_), GtOperand::String(_))
        | (GtOperand::String(_), GtOperand::Numeric(_)) => {
            Err("gt: mixing numeric and string inputs is not supported".to_string())
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
            .map_err(|e| format!("gt: {e}"))
    }
}

enum GtOperand {
    Numeric(NumericBuffer),
    String(StringBuffer),
}

impl GtOperand {
    fn from_value(value: Value) -> Result<Self, String> {
        match value {
            Value::Num(n) => Ok(GtOperand::Numeric(NumericBuffer::scalar(n))),
            Value::Bool(flag) => Ok(GtOperand::Numeric(NumericBuffer::scalar(if flag {
                1.0
            } else {
                0.0
            }))),
            Value::Int(i) => Ok(GtOperand::Numeric(NumericBuffer::scalar(i.to_f64()))),
            Value::Tensor(tensor) => Ok(GtOperand::Numeric(NumericBuffer::from_tensor(tensor))),
            Value::LogicalArray(array) => {
                Ok(GtOperand::Numeric(NumericBuffer::from_logical(array)))
            }
            Value::CharArray(array) => {
                Ok(GtOperand::Numeric(NumericBuffer::from_char_array(array)))
            }
            Value::String(s) => Ok(GtOperand::String(StringBuffer::scalar(s))),
            Value::StringArray(sa) => Ok(GtOperand::String(StringBuffer::from_array(sa))),
            Value::GpuTensor(handle) => {
                let tensor = gpu_helpers::gather_tensor(&handle)?;
                Ok(GtOperand::Numeric(NumericBuffer::from_tensor(tensor)))
            }
            Value::Complex(_, _) | Value::ComplexTensor(_) => {
                Err("gt: complex inputs are not supported".to_string())
            }
            unsupported => Err(format!("gt: unsupported input type {unsupported:?}")),
        }
    }
}

fn numeric_gt(lhs: &NumericBuffer, rhs: &NumericBuffer) -> Result<(Vec<u8>, Vec<usize>), String> {
    let shape = broadcast_shapes("gt", &lhs.shape, &rhs.shape)?;
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
        out.push(if lhs_val > rhs_val { 1 } else { 0 });
    }
    Ok((out, shape))
}

fn string_gt(lhs: &StringBuffer, rhs: &StringBuffer) -> Result<(Vec<u8>, Vec<usize>), String> {
    let shape = broadcast_shapes("gt", &lhs.shape, &rhs.shape)?;
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
        out.push(if lhs_val > rhs_val { 1 } else { 0 });
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
    fn gt_scalar_true() {
        let result = gt_builtin(Value::Num(5.0), Value::Num(4.0)).expect("gt");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gt_scalar_false() {
        let result = gt_builtin(Value::Num(2.0), Value::Num(3.0)).expect("gt");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gt_vector_broadcast() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![1, 4]).unwrap();
        let result = gt_builtin(Value::Tensor(tensor), Value::Num(2.0)).expect("gt");
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![1, 4]);
                assert_eq!(array.data, vec![0, 1, 0, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gt_char_array_against_numeric() {
        let chars = CharArray::new(vec!['A', 'B', 'C'], 1, 3).unwrap();
        let tensor = Tensor::new(vec![65.0, 65.0, 65.0], vec![1, 3]).unwrap();
        let result = gt_builtin(Value::CharArray(chars), Value::Tensor(tensor)).expect("gt");
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![1, 3]);
                assert_eq!(array.data, vec![0, 1, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gt_string_array_against_scalar() {
        let array = StringArray::new(vec!["apple".into(), "carrot".into()], vec![1, 2]).unwrap();
        let result =
            gt_builtin(Value::StringArray(array), Value::String("banana".into())).expect("gt");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![1, 2]);
                assert_eq!(mask.data, vec![0, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gt_string_numeric_error() {
        let err =
            gt_builtin(Value::String("apple".into()), Value::Num(3.0)).expect_err("expected error");
        assert!(
            err.contains("mixing numeric and string"),
            "unexpected message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gt_complex_error() {
        let err = gt_builtin(Value::Complex(1.0, 1.0), Value::Num(0.0)).expect_err("gt");
        assert!(
            err.contains("complex"),
            "expected complex error message, got {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gt_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![1.0, 4.0, 7.0], vec![1, 3]).unwrap();
            let rhs = Tensor::new(vec![0.0, 5.0, 6.0], vec![1, 3]).unwrap();
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
                gt_builtin(Value::GpuTensor(handle_l), Value::GpuTensor(handle_r)).expect("gt");
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
    fn gt_wgpu_matches_host() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let lhs = Tensor::new(vec![0.0, 2.0, 5.0, 8.0], vec![4, 1]).unwrap();
        let rhs = Tensor::new(vec![1.0, 1.5, 4.0, 8.0], vec![4, 1]).unwrap();
        let cpu = gt_host(Value::Tensor(lhs.clone()), Value::Tensor(rhs.clone())).unwrap();

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
        let gpu = gt_builtin(Value::GpuTensor(handle_l), Value::GpuTensor(handle_r)).unwrap();
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
