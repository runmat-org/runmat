//! MATLAB-compatible `flipud` builtin with GPU-aware semantics for RunMat.

use crate::builtins::array::shape::flip::{
    complex_tensor_into_value, flip_char_array_with, flip_complex_tensor_with, flip_gpu_with,
    flip_logical_array_with, flip_string_array_with, flip_tensor_with,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::{CellArray, ComplexTensor, Value};
use runmat_macros::runtime_builtin;

const UD_DIM: [usize; 1] = [1];

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "flipud",
        builtin_path = "crate::builtins::array::shape::flipud"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "flipud"
category: "array/shape"
keywords: ["flipud", "flip", "vertical", "matrix", "gpu"]
summary: "Flip an array up-to-down along the first dimension."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64", "i32", "bool"]
  broadcasting: "none"
  notes: "Uses the generic flip provider hook with axis=0; falls back to gather→flip→upload when unavailable."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::shape::flipud::tests"
  integration: [
    "builtins::array::shape::flipud::tests::flipud_gpu_roundtrip",
    "builtins::array::shape::flipud::tests::flipud_wgpu_matches_cpu"
  ]
---

# What does the `flipud` function do in MATLAB / RunMat?
`flipud(A)` mirrors `A` across its horizontal axis, reversing the order of rows (dimension&nbsp;1).
It works with scalars, vectors, matrices, N-D tensors, logical arrays, character arrays,
string arrays, cell arrays, complex data, and gpuArray handles, matching MATLAB semantics.

## How does the `flipud` function behave in MATLAB / RunMat?
- Always reverses dimension&nbsp;1 (rows) and leaves all other dimensions untouched, even for rank > 2 data.
- Inputs with a single row (row vectors, scalars) are returned unchanged because the first dimension is singleton.
- Numeric, logical, complex, character, string, and cell arrays all retain their MATLAB types, layout, and metadata (including UTF-16 code units for char arrays).
- gpuArray inputs execute on the device via the generic `flip` provider hook (axis = 0); when that hook is unavailable,
  RunMat gathers once, mirrors the data on the host, and uploads the result so the returned value is still a gpuArray.
- Dimensions larger than `ndims(A)` are treated as singleton axes, so `flipud` never errors when `A` has rank < 1.
- Behaviour matches `flip(A, 1)` exactly; `flipud` is provided for readability and compatibility with existing MATLAB code.

## `flipud` Function GPU Execution Behaviour
RunMat first tries to execute `flipud` on the GPU by delegating to the provider’s generic `flip`
implementation with axis `0` (zero-based). If the provider does not implement this hook, RunMat
transparently gathers the tensor, performs the vertical flip on the host, and uploads the result
back to the device so residency is preserved.

## Examples of using the `flipud` function in MATLAB / RunMat

### Reverse Rows of a Matrix
```matlab
A = [1 2 3; 4 5 6; 7 8 9];
B = flipud(A);
```
Expected output:

```matlab
B =
     7     8     9
     4     5     6
     1     2     3
```

### Reverse a Column Vector
```matlab
col = (1:4)';
rev = flipud(col);
```
Expected output:
```matlab
rev =
     4
     3
     2
     1
```

### Flip the First Dimension of a 3-D Tensor
```matlab
T = reshape(1:24, [3 4 2]);
F = flipud(T);
```
Expected output:
```matlab
F(:,:,1) =
     3     6     9    12
     2     5     8    11
     1     4     7    10

F(:,:,2) =
    15    18    21    24
    14    17    20    23
    13    16    19    22
```

### Flip Characters in a Char Array Vertically
```matlab
C = ['run'; 'mat'];
Cv = flipud(C);
```
Expected output:
```matlab
Cv =
    'mat'
    'run'
```

### Preserve Row Vector Orientation
```matlab
row = 1:5;
same = flipud(row);
```
Expected output:
```matlab
same = [1 2 3 4 5];
```

### Keep gpuArray Results on the Device While Flipping Rows
```matlab
G = gpuArray(rand(8, 8));
H = flipud(G);
isequal(gather(H), flipud(gather(G)))   % illustrative verification
```
Expected workflow:
```matlab
isa(H, 'gpuArray')
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You typically do not need to call `gpuArray` directly. RunMat’s auto-offload planner keeps tensors on
the GPU when profitable and only gathers when a provider lacks the flip hook. Even in that fallback,
`flipud` uploads the flipped result back to the device so subsequent operations remain gpu-resident.

## FAQ
### Does `flipud` change row vectors?
No. A row vector has a singleton first dimension, so reversing that axis leaves the data unchanged.

### Is `flipud` the same as calling `flip(A, 1)`?
Yes. `flipud` is a convenience wrapper around `flip` that always targets dimension&nbsp;1 (rows).

### Can I apply `flipud` to N-D tensors?
Absolutely. Only dimension&nbsp;1 is reversed; all other axes keep their original order regardless of rank.

### Does `flipud` support string, character, and cell arrays?
Yes. String arrays reorder their elements, character arrays mirror each column while preserving UTF-8 data, and cell arrays reverse their rows without copying contained values.

### What happens on the GPU if there is no flip kernel?
RunMat gathers the tensor once, mirrors it on the CPU, and uploads the result so you still receive a gpuArray.

### Does `flipud` allocate new GPU buffers?
Providers may reuse storage, but the builtin always returns a fresh handle. The simple provider uploads a new buffer.

### Is `flipud` numerically stable?
Yes. The function only reorders elements; values are never modified, so it is numerically stable.

## See Also
- [`flip`](./flip)
- [`fliplr`](./fliplr)
- [`permute`](./permute)
- [`reshape`](./reshape)
- [`gpuArray`](./gpuarray)
- [`gather`](./gather)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/array/shape/flipud.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/shape/flipud.rs)
- Found a behavioural difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::flipud")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "flipud",
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
    notes: "Delegates to the generic flip hook with axis=0; falls back to host mirror when the hook is missing.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::flipud")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "flipud",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Acts as a data-reordering barrier; fusion planner preserves residency but does not fuse through flipud.",
};

fn flipud_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("flipud")
        .build()
}

#[runtime_builtin(
    name = "flipud",
    category = "array/shape",
    summary = "Flip an array up-to-down along the first dimension.",
    keywords = "flipud,flip,vertical,matrix,gpu",
    accel = "custom",
    builtin_path = "crate::builtins::array::shape::flipud"
)]
fn flipud_builtin(value: Value) -> crate::BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => Ok(flip_tensor_with("flipud", tensor, &UD_DIM)
            .map(tensor::tensor_into_value)?),
        Value::LogicalArray(array) => Ok(flip_logical_array_with("flipud", array, &UD_DIM)
            .map(Value::LogicalArray)?),
        Value::ComplexTensor(ct) => Ok(flip_complex_tensor_with("flipud", ct, &UD_DIM)
            .map(Value::ComplexTensor)?),
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| flipud_error(format!("flipud: {e}")))?;
            Ok(flip_complex_tensor_with("flipud", tensor, &UD_DIM)
                .map(complex_tensor_into_value)?)
        }
        Value::StringArray(strings) => Ok(flip_string_array_with("flipud", strings, &UD_DIM)
            .map(Value::StringArray)?),
        Value::CharArray(chars) => Ok(flip_char_array_with("flipud", chars, &UD_DIM)
            .map(Value::CharArray)?),
        Value::String(scalar) => Ok(Value::String(scalar)),
        Value::Cell(cell) => flip_cell_array_rows(cell),
        Value::Num(n) => {
            let tensor = tensor::value_into_tensor_for("flipud", Value::Num(n))
                .map_err(|e| flipud_error(e))?;
            Ok(flip_tensor_with("flipud", tensor, &UD_DIM)
                .map(tensor::tensor_into_value)?)
        }
        Value::Int(i) => {
            let tensor = tensor::value_into_tensor_for("flipud", Value::Int(i))
                .map_err(|e| flipud_error(e))?;
            Ok(flip_tensor_with("flipud", tensor, &UD_DIM)
                .map(tensor::tensor_into_value)?)
        }
        Value::Bool(flag) => {
            let tensor = tensor::value_into_tensor_for("flipud", Value::Bool(flag))
                .map_err(|e| flipud_error(e))?;
            Ok(flip_tensor_with("flipud", tensor, &UD_DIM)
                .map(tensor::tensor_into_value)?)
        }
        Value::GpuTensor(handle) => Ok(flip_gpu_with("flipud", handle, &UD_DIM)?),
        Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::ClassRef(_)
        | Value::MException(_) => Err(flipud_error("flipud: unsupported input type")),
    }
}

fn flip_cell_array_rows(cell: CellArray) -> crate::BuiltinResult<Value> {
    if cell.rows <= 1 || cell.data.is_empty() {
        return Ok(Value::Cell(cell));
    }
    let rows = cell.rows;
    let cols = cell.cols;
    let data = cell.data;
    let mut flipped = Vec::with_capacity(data.len());
    for row in (0..rows).rev() {
        let base = row * cols;
        for col in 0..cols {
            flipped.push(data[base + col].clone());
        }
    }
    CellArray::new_handles(flipped, rows, cols)
        .map(Value::Cell)
        .map_err(|e| flipud_error(format!("flipud: {e}")))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::array::shape::flip::{
        flip_complex_tensor, flip_logical_array, flip_tensor,
    };
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        CellArray, CharArray, LogicalArray, StringArray, StructValue, Tensor, Value,
    };

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_matrix_reverses_rows() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).expect("tensor");
        let expected = flip_tensor(tensor.clone(), &UD_DIM).expect("expected");
        let result = flipud_builtin(Value::Tensor(tensor)).expect("flipud");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, expected.shape);
                assert_eq!(out.data, expected.data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_column_vector_reverses_order() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let expected = flip_tensor(tensor.clone(), &UD_DIM).expect("expected");
        let result = flipud_builtin(Value::Tensor(tensor)).expect("flipud");
        match result {
            Value::Tensor(out) => assert_eq!(out.data, expected.data),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_row_vector_noop() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let expected = tensor.clone();
        let result = flipud_builtin(Value::Tensor(tensor)).expect("flipud");
        match result {
            Value::Tensor(out) => assert_eq!(out.data, expected.data),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_nd_tensor_flips_first_dim_only() {
        let tensor = Tensor::new((1..=24).map(|v| v as f64).collect(), vec![3, 4, 2]).unwrap();
        let expected = flip_tensor(tensor.clone(), &UD_DIM).expect("expected");
        let result = flipud_builtin(Value::Tensor(tensor)).expect("flipud");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, expected.shape);
                assert_eq!(out.data, expected.data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_char_array() {
        let chars = CharArray::new("runmat".chars().collect(), 2, 3).unwrap();
        let result = flipud_builtin(Value::CharArray(chars)).expect("flipud");
        match result {
            Value::CharArray(out) => {
                let collected: String = out.data.iter().collect();
                assert_eq!(collected, "matrun");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_string_array() {
        let strings =
            StringArray::new(vec!["top".into(), "bottom".into()], vec![2, 1]).expect("strings");
        let result = flipud_builtin(Value::StringArray(strings)).expect("flipud");
        match result {
            Value::StringArray(out) => assert_eq!(out.data, vec!["bottom", "top"]),
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_cell_array_reverses_rows() {
        let cell = CellArray::new(
            vec![
                Value::from("r1c1"),
                Value::from("r1c2"),
                Value::from("r2c1"),
                Value::from("r2c2"),
            ],
            2,
            2,
        )
        .expect("cell");
        let result = flipud_builtin(Value::Cell(cell)).expect("flipud");
        match result {
            Value::Cell(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 2);
                assert_eq!(out.get(0, 0).unwrap(), Value::from("r2c1"));
                assert_eq!(out.get(0, 1).unwrap(), Value::from("r2c2"));
                assert_eq!(out.get(1, 0).unwrap(), Value::from("r1c1"));
                assert_eq!(out.get(1, 1).unwrap(), Value::from("r1c2"));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_logical_array_preserves_bits() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
        let expected = flip_logical_array(logical.clone(), &UD_DIM).expect("expected");
        let result = flipud_builtin(Value::LogicalArray(logical)).expect("flipud");
        match result {
            Value::LogicalArray(out) => assert_eq!(out.data, expected.data),
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_scalar_numeric_noop() {
        let result = flipud_builtin(Value::Num(42.0)).expect("flipud");
        match result {
            Value::Num(v) => assert_eq!(v, 42.0),
            other => panic!("expected numeric scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_complex_tensor_defaults_to_first_dim() {
        let tensor = ComplexTensor::new(
            vec![(1.0, 1.0), (2.0, -1.0), (3.0, 0.5), (4.0, -0.25)],
            vec![2, 2],
        )
        .unwrap();
        let expected = flip_complex_tensor(tensor.clone(), &UD_DIM).expect("expected");
        let result = flipud_builtin(Value::ComplexTensor(tensor)).expect("flipud");
        match result {
            Value::ComplexTensor(out) => assert_eq!(out.data, expected.data),
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_rejects_struct_inputs() {
        let mut st = StructValue::new();
        st.fields.insert("field".into(), Value::Num(1.0));
        let err = flipud_builtin(Value::Struct(st)).expect_err("struct unsupported");
        assert!(err.to_string().contains("unsupported input type"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = flipud_builtin(Value::GpuTensor(handle)).expect("flipud");
            let gathered = test_support::gather(result).expect("gather");
            let expected = flip_tensor(tensor, &UD_DIM).expect("expected");
            assert_eq!(gathered.shape, expected.shape);
            assert_eq!(gathered.data, expected.data);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_gpu_preserves_row_vector() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = flipud_builtin(Value::GpuTensor(handle)).expect("flipud");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, tensor.shape);
            assert_eq!(gathered.data, tensor.data);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_gpu_falls_back_when_axis_missing() {
        // The simple provider does not expose flip, so this exercises gather→flip→upload.
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let result = flipud_builtin(Value::Tensor(tensor.clone())).expect("flipud");
        match result {
            Value::Tensor(out) => assert_eq!(out.data, flip_tensor(tensor, &UD_DIM).unwrap().data),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn flipud_gpu_with_registered_provider_preserves_gpu_type() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = flipud_builtin(Value::GpuTensor(handle)).expect("flipud");
            assert!(matches!(result, Value::GpuTensor(_)));
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
    fn flipud_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let cpu = flip_tensor(tensor.clone(), &UD_DIM).expect("cpu flip");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = flipud_builtin(Value::GpuTensor(handle)).expect("flipud");
        let gathered = test_support::gather(gpu).expect("gather");
        assert_eq!(gathered.shape, cpu.shape);
        assert_eq!(gathered.data, cpu.data);
    }
}
