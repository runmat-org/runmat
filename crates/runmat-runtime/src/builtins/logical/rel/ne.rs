//! MATLAB-compatible `ne` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, RuntimeControlFlow};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "ne",
        builtin_path = "crate::builtins::logical::rel::ne"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "ne"
category: "logical/rel"
keywords: ["ne", "~=", "not equal", "logical comparison", "gpuArray inequality"]
summary: "Element-wise inequality comparison for scalars, arrays, strings, and gpuArray inputs."
references: []
gpu_support:
  elementwise: true
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Uses provider elem_ne kernels when available; otherwise inputs gather back to host memory transparently."
fusion:
  elementwise: true
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: "wgpu"
tested:
  unit: "builtins::logical::rel::ne::tests"
  integration: "builtins::logical::rel::ne::tests::ne_gpu_provider_roundtrip"
  gpu: "builtins::logical::rel::ne::tests::ne_wgpu_matches_host"
---

# What does the `ne` function do in MATLAB / RunMat?
`ne(A, B)` (or the infix `A ~= B`) performs an element-wise inequality comparison. The result is a logical scalar when the broadcasted shape contains one element, or a logical array otherwise.

## How does the `ne` function behave in MATLAB / RunMat?
- Numeric, logical, and character inputs are compared element-wise using MATLAB's implicit expansion rules.
- Complex numbers are considered different when either their real **or** imaginary part differs.
- Character arrays compare by Unicode code point; you can mix them with numeric arrays (`'A' ~= 65`) or strings.
- String scalars and string arrays compare lexically; implicit expansion works across the string dimensions.
- Handle objects compare by identity rather than by structural equality.
- Mixed numeric/string inputs raise MATLAB-compatible type errors.

## `ne` Function GPU Execution Behaviour
When both operands are `gpuArray` values and the active acceleration provider implements the `elem_ne` hook, RunMat executes the comparison entirely on the device and returns a `gpuArray` logical result. If the provider does not expose this hook, the runtime gathers the inputs to host memory automatically and performs the CPU comparison instead of failing.

## Examples of using the `ne` function in MATLAB / RunMat

### Checking if two scalars differ

```matlab
flag = ne(42, 7);
```

Expected output:

```matlab
flag =
     1
```

### Finding mismatched elements between vectors

```matlab
A = [1 2 3 4];
B = [1 0 3 5];
mask = ne(A, B);
```

Expected output:

```matlab
mask =
  1×4 logical array
     0     1     0     1
```

### Using implicit expansion for inequality

```matlab
M = [1 2 3; 4 5 6];
sel = ne(M, 2);
```

Expected output:

```matlab
sel =
  2×3 logical array
     1     0     1
     1     1     1
```

### Comparing text values to numeric codes

```matlab
letters = ['A' 'B' 'C'];
notA = ne(letters, 65);
```

Expected output:

```matlab
notA =
  1×3 logical array
     0     1     1
```

### Running `~=` directly on gpuArray inputs

```matlab
G1 = gpuArray([1 2 3]);
G2 = gpuArray([0 2 4]);
deviceResult = ne(G1, G2);
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

### Does `ne` return logical values?
Yes. Scalars return `true` or `false`. Arrays return logical arrays, and `gpuArray` inputs return `gpuArray` logical outputs.

### How are NaN values treated?
`NaN ~= NaN` evaluates to `true`, matching MATLAB's behaviour because equality comparisons return `false`.

### Can I compare complex numbers with `ne`?
Yes. Results are `true` when either the real or imaginary component differs.

### Are character vectors treated as numbers or text?
Both: they compare numerically (character code) against numeric inputs, and textually when compared to strings or other character arrays.

### What happens when I mix numeric and string inputs?
RunMat raises a MATLAB-compatible error describing the unsupported type combination.

### Do handle objects compare by value?
No. Handles compare by identity: two handles are different unless they reference the same underlying object.

### Does implicit expansion apply to string arrays?
Yes. String arrays support MATLAB-style implicit expansion, so you can compare against scalar strings without manual replication.

### Can I chain `ne` inside fused GPU expressions?
Yes. The builtin registers element-wise fusion metadata so the planner can fuse inequality checks with surrounding GPU-friendly operations.

### Is there a shorthand for calling `ne`?
Yes. You can use the operator form `A ~= B`, which maps directly to this builtin.

## See Also
[`eq`](./eq), [`lt`](./lt), [`gt`](./gt), [`gpuArray`](./gpuarray), [`gather`](./gather)
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::rel::ne")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ne",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Binary {
        name: "elem_ne",
        commutative: true,
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Prefers provider elem_ne kernels when available; otherwise inputs gather to host tensors automatically.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::rel::ne")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ne",
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
            Ok(format!("select({zero}, {one}, ({lhs} != {rhs}))"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits comparison kernels that write 1 when operands differ; providers may override with specialised shaders.",
};

const BUILTIN_NAME: &str = "ne";
const IDENT_INVALID_INPUT: &str = "MATLAB:ne:InvalidInput";
const IDENT_SIZE_MISMATCH: &str = "MATLAB:ne:SizeMismatch";

fn ne_error(message: impl Into<String>, identifier: &'static str) -> RuntimeControlFlow {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_identifier(identifier)
        .build()
        .into()
}

#[runtime_builtin(
    name = "ne",
    category = "logical/rel",
    summary = "Element-wise inequality comparison for scalars, arrays, and gpuArray inputs.",
    keywords = "ne,not equal,comparison,logical,gpu",
    accel = "elementwise",
    builtin_path = "crate::builtins::logical::rel::ne"
)]
fn ne_builtin(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    if let (Value::GpuTensor(ref a), Value::GpuTensor(ref b)) = (&lhs, &rhs) {
        if let Some(result) = try_ne_gpu(a, b) {
            return result;
        }
    }
    ne_host(lhs, rhs)
}

fn try_ne_gpu(a: &GpuTensorHandle, b: &GpuTensorHandle) -> Option<crate::BuiltinResult<Value>> {
    let provider = runmat_accelerate_api::provider()?;
    match provider.elem_ne(a, b) {
        Ok(handle) => Some(Ok(gpu_helpers::logical_gpu_value(handle))),
        Err(err) => {
            drop(err);
            None
        }
    }
}

fn ne_host(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    if let Some(value) = ne_identity(&lhs, &rhs) {
        return Ok(value);
    }

    let (lhs, rhs) = normalize_char_string(lhs, rhs);

    let left = NeOperand::from_value(lhs)?;
    let right = NeOperand::from_value(rhs)?;

    match (left, right) {
        (NeOperand::Numeric(a), NeOperand::Numeric(b)) => {
            let (data, shape) = numeric_ne(&a, &b)?;
            logical_result(data, shape)
        }
        (NeOperand::Numeric(a), NeOperand::Complex(b)) => {
            let promoted = promote_numeric_to_complex(&a);
            let (data, shape) = complex_ne(&promoted, &b)?;
            logical_result(data, shape)
        }
        (NeOperand::Complex(a), NeOperand::Numeric(b)) => {
            let promoted = promote_numeric_to_complex(&b);
            let (data, shape) = complex_ne(&a, &promoted)?;
            logical_result(data, shape)
        }
        (NeOperand::Complex(a), NeOperand::Complex(b)) => {
            let (data, shape) = complex_ne(&a, &b)?;
            logical_result(data, shape)
        }
        (NeOperand::String(a), NeOperand::String(b)) => {
            let (data, shape) = string_ne(&a, &b)?;
            logical_result(data, shape)
        }
        (NeOperand::Numeric(_), NeOperand::String(_))
        | (NeOperand::Complex(_), NeOperand::String(_))
        | (NeOperand::String(_), NeOperand::Numeric(_))
        | (NeOperand::String(_), NeOperand::Complex(_)) => Err(ne_error(
            "ne: mixing numeric and string inputs is not supported",
            IDENT_INVALID_INPUT,
        )),
    }
}

fn ne_identity(lhs: &Value, rhs: &Value) -> Option<Value> {
    match (handle_ptr(lhs), handle_ptr(rhs)) {
        (Some(a), Some(b)) => Some(Value::Bool(a != b)),
        (Some(_), None) | (None, Some(_)) => Some(Value::Bool(true)),
        (None, None) => None,
    }
}

fn handle_ptr(value: &Value) -> Option<usize> {
    match value {
        Value::HandleObject(handle) => Some(unsafe { handle.target.as_raw() } as usize),
        Value::Listener(listener) => Some(unsafe { listener.target.as_raw() } as usize),
        _ => None,
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

fn logical_result(data: Vec<u8>, shape: Vec<usize>) -> crate::BuiltinResult<Value> {
    if tensor::element_count(&shape) <= 1 && data.len() == 1 {
        Ok(Value::Bool(data[0] != 0))
    } else {
        LogicalArray::new(data, shape)
            .map(Value::LogicalArray)
            .map_err(|e| ne_error(format!("ne: {e}"), IDENT_INVALID_INPUT))
    }
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
struct ComplexBuffer {
    data: Vec<(f64, f64)>,
    shape: Vec<usize>,
}

impl ComplexBuffer {
    fn scalar(re: f64, im: f64) -> Self {
        Self {
            data: vec![(re, im)],
            shape: vec![1, 1],
        }
    }

    fn from_tensor(tensor: ComplexTensor) -> Self {
        Self {
            data: tensor.data,
            shape: tensor.shape,
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
        let StringArray { data, shape, .. } = array;
        Self { data, shape }
    }
}

#[derive(Debug)]
enum NeOperand {
    Numeric(NumericBuffer),
    Complex(ComplexBuffer),
    String(StringBuffer),
}

impl NeOperand {
    fn from_value(value: Value) -> crate::BuiltinResult<Self> {
        match value {
            Value::Num(n) => Ok(NeOperand::Numeric(NumericBuffer::scalar(n))),
            Value::Bool(flag) => Ok(NeOperand::Numeric(NumericBuffer::scalar(if flag {
                1.0
            } else {
                0.0
            }))),
            Value::Int(i) => Ok(NeOperand::Numeric(NumericBuffer::scalar(i.to_f64()))),
            Value::Tensor(tensor) => Ok(NeOperand::Numeric(NumericBuffer::from_tensor(tensor))),
            Value::LogicalArray(array) => {
                Ok(NeOperand::Numeric(NumericBuffer::from_logical(array)))
            }
            Value::Complex(re, im) => Ok(NeOperand::Complex(ComplexBuffer::scalar(re, im))),
            Value::ComplexTensor(tensor) => {
                Ok(NeOperand::Complex(ComplexBuffer::from_tensor(tensor)))
            }
            Value::String(s) => Ok(NeOperand::String(StringBuffer::scalar(s))),
            Value::StringArray(sa) => Ok(NeOperand::String(StringBuffer::from_array(sa))),
            Value::CharArray(array) => {
                Ok(NeOperand::Numeric(NumericBuffer::from_char_array(array)))
            }
            Value::GpuTensor(handle) => {
                let tensor = gpu_helpers::gather_tensor(&handle)
                    .map_err(|err| ne_error(format!("{BUILTIN_NAME}: {err}"), IDENT_INVALID_INPUT))?;
                Ok(NeOperand::Numeric(NumericBuffer::from_tensor(tensor)))
            }
            unsupported => Err(ne_error(
                format!("ne: unsupported input type {unsupported:?}"),
                IDENT_INVALID_INPUT,
            )),
        }
    }
}

fn numeric_ne(
    lhs: &NumericBuffer,
    rhs: &NumericBuffer,
) -> crate::BuiltinResult<(Vec<u8>, Vec<usize>)> {
    let shape = broadcast_shapes(BUILTIN_NAME, &lhs.shape, &rhs.shape)
        .map_err(|err| ne_error(err, IDENT_SIZE_MISMATCH))?;
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
        out.push(if lhs_val != rhs_val { 1 } else { 0 });
    }
    Ok((out, shape))
}

fn complex_ne(
    lhs: &ComplexBuffer,
    rhs: &ComplexBuffer,
) -> crate::BuiltinResult<(Vec<u8>, Vec<usize>)> {
    let shape = broadcast_shapes(BUILTIN_NAME, &lhs.shape, &rhs.shape)
        .map_err(|err| ne_error(err, IDENT_SIZE_MISMATCH))?;
    let total = tensor::element_count(&shape);
    if total == 0 {
        return Ok((Vec::new(), shape));
    }
    let strides_l = compute_strides(&lhs.shape);
    let strides_r = compute_strides(&rhs.shape);
    let mut out = Vec::with_capacity(total);
    for idx in 0..total {
        let lhs_val = if lhs.data.is_empty() {
            (0.0, 0.0)
        } else {
            let offset = broadcast_index(idx, &shape, &lhs.shape, &strides_l);
            lhs.data[offset]
        };
        let rhs_val = if rhs.data.is_empty() {
            (0.0, 0.0)
        } else {
            let offset = broadcast_index(idx, &shape, &rhs.shape, &strides_r);
            rhs.data[offset]
        };
        out.push(if lhs_val.0 != rhs_val.0 || lhs_val.1 != rhs_val.1 {
            1
        } else {
            0
        });
    }
    Ok((out, shape))
}

fn string_ne(
    lhs: &StringBuffer,
    rhs: &StringBuffer,
) -> crate::BuiltinResult<(Vec<u8>, Vec<usize>)> {
    let shape = broadcast_shapes(BUILTIN_NAME, &lhs.shape, &rhs.shape)
        .map_err(|err| ne_error(err, IDENT_SIZE_MISMATCH))?;
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
        out.push(if lhs_val != rhs_val { 1 } else { 0 });
    }
    Ok((out, shape))
}

fn promote_numeric_to_complex(buffer: &NumericBuffer) -> ComplexBuffer {
    ComplexBuffer {
        data: buffer.data.iter().map(|&v| (v, 0.0)).collect(),
        shape: buffer.shape.clone(),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::RuntimeControlFlow;
    use runmat_accelerate_api::HostTensorView;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::ProviderPrecision;
    use runmat_builtins::HandleRef;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ne_scalar_true() {
        let result = ne_builtin(Value::Num(5.0), Value::Num(4.0)).expect("ne");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ne_scalar_false() {
        let result = ne_builtin(Value::Num(5.0), Value::Num(5.0)).expect("ne");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ne_vector_broadcast() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 2.0], vec![1, 4]).unwrap();
        let result = ne_builtin(Value::Tensor(tensor), Value::Num(2.0)).expect("ne");
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
    fn ne_char_array_against_numeric() {
        let char_array = CharArray::new(vec!['A', 'B', 'A'], 1, 3).unwrap();
        let tensor = Tensor::new(vec![65.0, 66.0, 65.0], vec![1, 3]).unwrap();
        let result = ne_builtin(Value::CharArray(char_array), Value::Tensor(tensor)).expect("ne");
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![1, 3]);
                assert_eq!(array.data, vec![0, 0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ne_string_array_broadcast() {
        let sa = StringArray::new(vec!["red".into(), "blue".into()], vec![1, 2]).unwrap();
        let result = ne_builtin(Value::StringArray(sa), Value::String("red".into())).expect("ne");
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![1, 2]);
                assert_eq!(array.data, vec![0, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ne_handle_identity() {
        unsafe {
            let raw = Box::into_raw(Box::new(Value::Num(1.0)));
            let gc = runmat_gc_api::GcPtr::from_raw(raw);
            let handle = HandleRef {
                class_name: "TestHandle".into(),
                target: gc,
                valid: true,
            };
            let lhs = Value::HandleObject(handle.clone());
            let rhs = Value::HandleObject(handle);
            let result = ne_builtin(lhs, rhs).expect("ne");
            assert_eq!(result, Value::Bool(false));
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ne_handle_difference() {
        unsafe {
            let raw_a = Box::into_raw(Box::new(Value::Num(1.0)));
            let raw_b = Box::into_raw(Box::new(Value::Num(2.0)));
            let handle_a = HandleRef {
                class_name: "TestHandle".into(),
                target: runmat_gc_api::GcPtr::from_raw(raw_a),
                valid: true,
            };
            let handle_b = HandleRef {
                class_name: "TestHandle".into(),
                target: runmat_gc_api::GcPtr::from_raw(raw_b),
                valid: true,
            };
            let result = ne_builtin(Value::HandleObject(handle_a), Value::HandleObject(handle_b))
                .expect("ne");
            assert_eq!(result, Value::Bool(true));
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ne_mixed_numeric_string_error() {
        let err = ne_builtin(Value::Num(1.0), Value::String("a".into())).unwrap_err();
        match err {
            RuntimeControlFlow::Error(err) => {
                assert!(err.message().contains("mixing numeric and string inputs"));
                assert_eq!(err.identifier(), Some(IDENT_INVALID_INPUT));
            }
            RuntimeControlFlow::Suspend(_) => panic!("unexpected suspension"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ne_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
            let tensor_b = Tensor::new(vec![1.0, 0.0, 3.0, 5.0], vec![2, 2]).unwrap();
            let view_a = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let view_b = HostTensorView {
                data: &tensor_b.data,
                shape: &tensor_b.shape,
            };
            let h_a = provider.upload(&view_a).expect("upload a");
            let h_b = provider.upload(&view_b).expect("upload b");
            let result = ne_builtin(Value::GpuTensor(h_a), Value::GpuTensor(h_b)).expect("ne");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![0.0, 1.0, 1.0, 0.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ne_gpu_falls_back_to_host_when_only_one_tensor() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = ne_builtin(Value::GpuTensor(handle), Value::Num(2.0)).expect("ne");
            match result {
                Value::LogicalArray(array) => {
                    assert_eq!(array.data, vec![1, 0, 1]);
                }
                other => panic!("expected logical array, got {other:?}"),
            }
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
    fn ne_wgpu_matches_host() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 0.0, 3.0, 5.0], vec![2, 2]).unwrap();
        let cpu = ne_host(Value::Tensor(a.clone()), Value::Tensor(b.clone())).unwrap();
        let view_a = HostTensorView {
            data: &a.data,
            shape: &a.shape,
        };
        let view_b = HostTensorView {
            data: &b.data,
            shape: &b.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("provider");
        let h_a = provider.upload(&view_a).unwrap();
        let h_b = provider.upload(&view_b).unwrap();
        let gpu = ne_builtin(Value::GpuTensor(h_a), Value::GpuTensor(h_b)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::LogicalArray(cp), gt) => {
                assert_eq!(gt.shape, cp.shape);
                let tol = match provider.precision() {
                    ProviderPrecision::F64 => 1e-12,
                    ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gt.data.iter().zip(cp.data.iter()) {
                    let diff = *a - f64::from(*b);
                    assert!(
                        diff.abs() < tol,
                        "mismatch between GPU and CPU logical results"
                    );
                }
            }
            _ => panic!("unexpected result variants"),
        }
    }
}
