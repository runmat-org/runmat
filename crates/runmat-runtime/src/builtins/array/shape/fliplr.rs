//! MATLAB-compatible `fliplr` builtin with GPU-aware semantics for RunMat.

use super::flip::{
    complex_tensor_into_value, flip_char_array, flip_complex_tensor, flip_gpu, flip_logical_array,
    flip_string_array, flip_tensor,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use runmat_builtins::{ComplexTensor, Value};
use runmat_macros::runtime_builtin;

const LR_DIM: [usize; 1] = [2];

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "fliplr",
        builtin_path = "crate::builtins::array::shape::fliplr"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "fliplr"
category: "array/shape"
keywords: ["fliplr", "flip", "horizontal", "matrix", "gpu"]
summary: "Flip an array left-to-right along the second dimension."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64", "i32", "bool"]
  broadcasting: "none"
  notes: "Uses the generic flip provider hook with axis=1; falls back to gather→flip→upload when unavailable."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::shape::fliplr::tests"
  integration: [
    "builtins::array::shape::fliplr::tests::fliplr_gpu_roundtrip",
    "builtins::array::shape::fliplr::tests::fliplr_wgpu_matches_cpu"
  ]
---

# What does the `fliplr` function do in MATLAB / RunMat?
`fliplr(A)` mirrors `A` across its vertical axis, reversing the column order (dimension 2).
It accepts scalars, vectors, matrices, N-D tensors, logical arrays, character arrays,
string arrays, complex data, and gpuArray handles, matching MATLAB semantics.

## How does the `fliplr` function behave in MATLAB / RunMat?
- Always reverses dimension 2 (columns) and leaves all other dimensions untouched, even for rank > 2 data.
- Inputs with fewer than two columns (column vectors, scalars) are returned unchanged because the second dimension is singleton.
- Logical, numeric, complex, character, and string arrays all preserve their MATLAB types and storage layout.
- gpuArray inputs execute on the device via the generic `flip` provider hook (axis = 1); when that hook is missing,
  RunMat gathers once, mirrors the data on the host, and uploads the result so the returned value is still a gpuArray.
- Dimensions larger than `ndims(A)` are treated as singleton axes, so `fliplr` never errors when `A` has rank < 2.

## `fliplr` Function GPU Execution Behaviour
RunMat first tries to execute `fliplr` on the GPU by delegating to the provider’s generic `flip`
implementation with axis `1` (zero-based). If the provider does not implement this hook, RunMat
transparently gathers the tensor, performs the horizontal flip on the host, and uploads the result
back to the device so residency is preserved.

## Examples of using the `fliplr` function in MATLAB / RunMat

### Reverse Columns of a Matrix
```matlab
A = [1 2 3; 4 5 6];
B = fliplr(A);
```
Expected output:

```matlab
B =
     3     2     1
     6     5     4
```

### Mirror an Image Matrix Horizontally
```matlab
img = reshape(1:16, 4, 4);
mirrored = fliplr(img);
```
Expected output:
```matlab
mirrored =
    4     3     2     1
    8     7     6     5
   12    11    10     9
   16    15    14    13
```

### Flip the Order of Polynomial Coefficients
```matlab
c = [1 0 -2 5];
rev = fliplr(c);
```
Expected output:
```matlab
rev = [5  -2  0  1];
```

### Reverse Each Slice in a 3-D Array Along Columns
```matlab
T = reshape(1:24, [3 4 2]);
F = fliplr(T);
```
Expected output:
```matlab
F(:,:,1) =
     4     3     2     1
     8     7     6     5
    12    11    10     9

F(:,:,2) =
    16    15    14    13
    20    19    18    17
    24    23    22    21
```

### Preserve Column Vector Orientation
```matlab
col = (1:4)';
same = fliplr(col);
```
Expected output:
```matlab
same =
     1
     2
     3
     4
```

### Flip Characters in a Two-Row Char Array
```matlab
C = ['r','u','n'; 'm','a','t'];
Ct = fliplr(C);
```
Expected output:
```matlab
Ct =
    'nur'
    'tam'
```

### Keep gpuArray Results on the Device While Flipping Columns
```matlab
G = gpuArray(rand(8, 8));
H = fliplr(G);
```
Expected workflow:
```matlab
isa(H, 'gpuArray')
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You typically do not need to call `gpuArray` directly. RunMat’s auto-offload planner keeps tensors on
the GPU when profitable and only gathers when a provider lacks the flip hook. Even in that fallback,
`fliplr` uploads the flipped result back to the device so subsequent operations can stay gpu-resident.

## FAQ
### Does `fliplr` change column vectors?
No. A column vector has a singleton second dimension, so reversing that axis leaves the data unchanged.

### Is `fliplr` the same as calling `flip(A, 2)`?
Yes. `fliplr` is a convenience wrapper around `flip` that always targets dimension 2 (columns).

### Can I apply `fliplr` to N-D tensors?
Absolutely. Only dimension 2 is reversed; all other axes keep their original order regardless of rank.

### Does `fliplr` support string and character arrays?
Yes. String arrays reorder their elements, and character arrays mirror each row while preserving UTF-8 data.

### What happens on the GPU if there is no flip kernel?
RunMat gathers the tensor once, mirrors it on the CPU, and uploads the result so you still receive a gpuArray.

### Does `fliplr` allocate new GPU buffers?
Providers may reuse storage, but the builtin always returns a fresh handle. The simple provider uploads a new buffer.

### Is `fliplr` numerically stable?
Yes. The function only reorders elements; values are never modified, so it is numerically stable.

## See Also
- [`flip`](./flip)
- [`flipud`](./flipud)
- [`permute`](./permute)
- [`reshape`](./reshape)
- [`gpuArray`](../../acceleration/gpu/gpuArray)
- [`gather`](../../acceleration/gpu/gather)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/array/shape/fliplr.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/shape/fliplr.rs)
- Found a behavioural difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::fliplr")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fliplr",
    op_kind: GpuOpKind::Custom("flip"),
    supported_precisions: &[
        ScalarType::F32,
        ScalarType::F64,
        ScalarType::I32,
        ScalarType::Bool,
    ],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("flip")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Delegates to the generic flip hook with axis=1; falls back to host mirror when the hook is missing.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::fliplr")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fliplr",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Acts as a data-reordering barrier; fusion planner preserves residency but does not fuse through fliplr.",
};

#[runtime_builtin(
    name = "fliplr",
    category = "array/shape",
    summary = "Flip an array left-to-right along the second dimension.",
    keywords = "fliplr,flip,horizontal,matrix,gpu",
    accel = "custom",
    builtin_path = "crate::builtins::array::shape::fliplr"
)]
fn fliplr_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::Tensor(tensor) => flip_tensor(tensor, &LR_DIM)
            .map_err(|e| e.replace("flip", "fliplr"))
            .map(tensor::tensor_into_value),
        Value::LogicalArray(array) => flip_logical_array(array, &LR_DIM)
            .map_err(|e| e.replace("flip", "fliplr"))
            .map(Value::LogicalArray),
        Value::ComplexTensor(ct) => flip_complex_tensor(ct, &LR_DIM)
            .map_err(|e| e.replace("flip", "fliplr"))
            .map(Value::ComplexTensor),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| format!("fliplr: {e}"))?;
            flip_complex_tensor(tensor, &LR_DIM)
                .map_err(|e| e.replace("flip", "fliplr"))
                .map(complex_tensor_into_value)
        }
        Value::StringArray(strings) => flip_string_array(strings, &LR_DIM)
            .map_err(|e| e.replace("flip", "fliplr"))
            .map(Value::StringArray),
        Value::CharArray(chars) => flip_char_array(chars, &LR_DIM)
            .map_err(|e| e.replace("flip", "fliplr"))
            .map(Value::CharArray),
        Value::String(scalar) => Ok(Value::String(scalar)),
        Value::Num(n) => {
            let tensor = tensor::value_into_tensor_for("fliplr", Value::Num(n))?;
            flip_tensor(tensor, &LR_DIM)
                .map_err(|e| e.replace("flip", "fliplr"))
                .map(tensor::tensor_into_value)
        }
        Value::Int(i) => {
            let tensor = tensor::value_into_tensor_for("fliplr", Value::Int(i))?;
            flip_tensor(tensor, &LR_DIM)
                .map_err(|e| e.replace("flip", "fliplr"))
                .map(tensor::tensor_into_value)
        }
        Value::Bool(flag) => {
            let tensor = tensor::value_into_tensor_for("fliplr", Value::Bool(flag))?;
            flip_tensor(tensor, &LR_DIM)
                .map_err(|e| e.replace("flip", "fliplr"))
                .map(tensor::tensor_into_value)
        }
        Value::GpuTensor(handle) => {
            flip_gpu(handle, &LR_DIM).map_err(|e| e.replace("flip", "fliplr"))
        }
        Value::Cell(_) => Err("fliplr: cell arrays are not yet supported".to_string()),
        Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::ClassRef(_)
        | Value::MException(_) => Err("fliplr: unsupported input type".to_string()),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{CharArray, LogicalArray, StringArray, StructValue, Tensor, Value};

    #[test]
    fn fliplr_matrix_reverses_columns() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![3, 2]).expect("tensor");
        let expected = flip_tensor(tensor.clone(), &LR_DIM).expect("expected");
        let result = fliplr_builtin(Value::Tensor(tensor)).expect("fliplr");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, expected.shape);
                assert_eq!(out.data, expected.data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn fliplr_row_vector_reverses_order() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let expected = flip_tensor(tensor.clone(), &LR_DIM).expect("expected");
        let result = fliplr_builtin(Value::Tensor(tensor)).expect("fliplr");
        match result {
            Value::Tensor(out) => assert_eq!(out.data, expected.data),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn fliplr_column_vector_noop() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let expected = tensor.clone();
        let result = fliplr_builtin(Value::Tensor(tensor)).expect("fliplr");
        match result {
            Value::Tensor(out) => assert_eq!(out.data, expected.data),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn fliplr_nd_tensor_flips_second_dim_only() {
        let tensor = Tensor::new((1..=24).map(|v| v as f64).collect(), vec![3, 4, 2]).unwrap();
        let expected = flip_tensor(tensor.clone(), &LR_DIM).expect("expected");
        let result = fliplr_builtin(Value::Tensor(tensor)).expect("fliplr");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, expected.shape);
                assert_eq!(out.data, expected.data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn fliplr_char_array() {
        let chars = CharArray::new("runmat".chars().collect(), 2, 3).unwrap();
        let result = fliplr_builtin(Value::CharArray(chars)).expect("fliplr");
        match result {
            Value::CharArray(out) => {
                let collected: String = out.data.iter().collect();
                assert_eq!(collected, "nurtam");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn fliplr_string_array() {
        let strings =
            StringArray::new(vec!["left".into(), "right".into()], vec![1, 2]).expect("strings");
        let result = fliplr_builtin(Value::StringArray(strings)).expect("fliplr");
        match result {
            Value::StringArray(out) => assert_eq!(out.data, vec!["right", "left"]),
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn fliplr_logical_array_preserves_bits() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
        let expected = flip_logical_array(logical.clone(), &LR_DIM).expect("expected");
        let result = fliplr_builtin(Value::LogicalArray(logical)).expect("fliplr");
        match result {
            Value::LogicalArray(out) => assert_eq!(out.data, expected.data),
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn fliplr_scalar_numeric_noop() {
        let result = fliplr_builtin(Value::Num(42.0)).expect("fliplr");
        match result {
            Value::Num(v) => assert_eq!(v, 42.0),
            other => panic!("expected numeric scalar, got {other:?}"),
        }
    }

    #[test]
    fn fliplr_string_scalar_noop() {
        let result = fliplr_builtin(Value::String("runmat".into())).expect("fliplr");
        match result {
            Value::String(s) => assert_eq!(s, "runmat"),
            other => panic!("expected string scalar, got {other:?}"),
        }
    }

    #[test]
    fn fliplr_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new((1..=12).map(|v| v as f64).collect(), vec![3, 4]).expect("tensor");
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = fliplr_builtin(Value::GpuTensor(handle)).expect("fliplr gpu");
            let gathered = test_support::gather(result).expect("gather");
            let expected = flip_tensor(tensor, &LR_DIM).expect("expected");
            assert_eq!(gathered.shape, expected.shape);
            assert_eq!(gathered.data, expected.data);
        });
    }

    #[test]
    fn fliplr_rejects_unsupported_type() {
        let value = Value::Struct(StructValue::new());
        let err = fliplr_builtin(value).expect_err("structs are unsupported");
        assert!(
            err.contains("unsupported input type"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn fliplr_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new((1..=16).map(|v| v as f64).collect(), vec![4, 4]).unwrap();
        let expected = flip_tensor(tensor.clone(), &LR_DIM).expect("expected");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let handle = provider.upload(&view).expect("upload");
        let value = fliplr_builtin(Value::GpuTensor(handle)).expect("fliplr gpu");
        let gathered = test_support::gather(value).expect("gather");
        assert_eq!(gathered.shape, expected.shape);
        assert_eq!(gathered.data, expected.data);
    }

    #[test]
    fn doc_examples_present() {
        let blocks = crate::builtins::common::test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
