//! MATLAB-compatible `vertcat` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use runmat_builtins::{IntValue, Tensor, Value};
use runmat_macros::runtime_builtin;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "vertcat",
        builtin_path = "crate::builtins::array::shape::vertcat"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "vertcat"
category: "array/shape"
keywords: ["vertcat", "vertical concatenation", "stack rows", "square brackets", "gpu"]
summary: "Stack inputs top-to-bottom (dimension 1) exactly like MATLAB's semicolon syntax."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Delegates to cat with dim=1; providers that expose cat run entirely on the GPU, otherwise RunMat gathers to the host and re-uploads the result."
fusion:
  elementwise: false
  reduction: false
  max_inputs: null
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::array::shape::vertcat::tests"
  integration:
    - "builtins::array::shape::vertcat::tests::vertcat_gpu_roundtrip"
    - "builtins::array::shape::vertcat::tests::vertcat_like_gpu_from_host_inputs"
---

# What does the `vertcat` function do in MATLAB / RunMat?
`vertcat(A1, A2, …)` vertically concatenates its inputs, matching the behavior of MATLAB's
semicolon array construction `[A1; A2; …]`. It is the standard way to stack matrices,
vectors, or higher-dimensional slices on top of each other.

## How does the `vertcat` function behave in MATLAB / RunMat?
- Hard-codes `dim = 1`. All inputs must agree on every dimension except the first.
- Accepts numeric, logical, complex, character, string, and cell arrays with MATLAB-compatible
  type checking. Mixing classes is an error.
- Scalars are treated as `1×1`, enabling concise row-building such as `vertcat(1, 2, 3)`.
- Empty inputs participate naturally. If any shared dimension is zero, the result is empty with
  the expected shape.
- The optional trailing `'like', prototype` pair forces the output to match the prototype's data
  residency (CPU vs GPU) and numeric flavour.
- Mixing `gpuArray` inputs with host inputs is disallowed. Convert explicitly via `gpuArray`
  or `gather` to control residency.

## `vertcat` function GPU execution behaviour
`vertcat` delegates to `cat(dim = 1, …)`. When the active acceleration provider implements the
`cat` hook, concatenation happens entirely on the GPU, keeping `gpuArray` inputs resident and
avoiding costly round-trips. Providers that lack this hook trigger a transparent fallback:
RunMat gathers the operands to the host, concatenates them with MATLAB semantics, and uploads
the result back to the originating device so downstream code still sees a `gpuArray`. This mirrors
MATLAB's explicit GPU workflows while keeping RunMat's auto-offload planner informed.

## Examples of using the `vertcat` function in MATLAB / RunMat

### Stacking matrices by adding rows
```matlab
A = [1 2; 3 4];
B = [5 6; 7 8];
C = vertcat(A, B);
```
Expected output:
```matlab
C =
     1     2
     3     4
     5     6
     7     8
```

### Building a column vector from scalars
```matlab
col = vertcat(1, 2, 3, 4);
```
Expected output:
```matlab
col =
     1
     2
     3
     4
```

### Combining character arrays into taller text blocks
```matlab
top = ['RunMat'];
bottom = ['Rocks!'];
banner = vertcat(top, bottom);
```
Expected behavior: `banner` is a 2×6 character array containing both rows.

### Joining string arrays into multi-row tables
```matlab
header = ["Name" "Score"];
rows = ["Alice" "98"; "Bob" "92"];
table = vertcat(header, rows);
```
Expected behavior: `size(table)` returns `[3 2]` with the header on top.

### Preserving logical masks when stacking
```matlab
mask1 = logical([1 0 1]);
mask2 = logical([0 1 0]);
stacked = vertcat(mask1, mask2);
```
Expected behavior: `stacked` is 2×3 logical with rows from both masks.

### Extending cell arrays downwards
```matlab
row1 = {1, "low"};
row2 = {2, "high"};
grid = vertcat(row1, row2);
```
Expected behavior: `grid` is a 2×2 cell array with each original row preserved.

### Keeping gpuArray inputs resident on the device
```matlab
G1 = gpuArray(rand(128, 256));
G2 = gpuArray(rand(64, 256));
stacked = vertcat(G1, G2);
```
Expected behavior: `stacked` remains a `gpuArray` with size `[192 256]`.

### Requesting GPU output with the `'like'` prototype
```matlab
proto = gpuArray.zeros(1, 3);
result = vertcat([1 2 3], [4 5 6], "like", proto);
```
Expected behavior: even though the inputs are on the CPU, `result` is uploaded to the GPU.

### Working with empty rows without surprises
```matlab
empty = zeros(0, 3);
combo = vertcat(empty, empty);
```
Expected behavior: `size(combo)` returns `[0 3]`.

### Stacking complex matrices preserves imaginary parts
```matlab
z1 = complex([1 2], [3 4]);
z2 = complex([5 6], [7 8]);
joined = vertcat(z1, z2);
```
Expected behavior: `joined` is 4×1 complex with both sets of values.

## FAQ

**Does `vertcat` require at least one input?**  
No. Calling it with no inputs returns the canonical `0×0` double, just like `[]`.

**Can I mix numeric and logical types?**  
Numeric inputs (real or complex) and logicals must all share the same class. Mixing with strings,
chars, or cells is not allowed.

**How strict are dimension checks?**  
All dimensions except the first must match exactly. RunMat reports the mismatch using MATLAB-style
error messages sourced from `cat`.

**How do I concatenate along other dimensions?**  
Use `cat(dim, …)` directly. `vertcat` is simply `cat(1, …)` with additional ergonomics.

**What happens if I pass a `gpuArray` and a CPU array?**  
RunMat raises an error mirroring MATLAB. Convert everything to the same residency before calling.

**Does `vertcat` participate in fusion pipelines?**  
No. Concatenation materialises results immediately and is treated as a fusion sink.

## See Also
- [`cat`](./cat)
- [`horzcat`](./horzcat)
- [`reshape`](./reshape)
- [`gpuArray`](../../acceleration/gpu/gpuArray)
- [`gather`](../../acceleration/gpu/gather)

## Source & Feedback
- Implementation: `crates/runmat-runtime/src/builtins/array/shape/vertcat.rs`
- Found an issue or behavioral difference? [Open a RunMat issue](https://github.com/runmat-org/runmat/issues/new/choose).
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::vertcat")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "vertcat",
    op_kind: GpuOpKind::Custom("cat"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("cat")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Delegates to cat(dim=1); providers without cat fall back to host gather + upload.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::vertcat")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "vertcat",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Concatenation materialises outputs immediately, terminating fusion pipelines.",
};

#[runtime_builtin(
    name = "vertcat",
    category = "array/shape",
    summary = "Concatenate inputs vertically (dimension 1) just like MATLAB semicolons.",
    keywords = "vertcat,vertical concatenation,array,gpu",
    accel = "array_construct",
    builtin_path = "crate::builtins::array::shape::vertcat"
)]
fn vertcat_builtin(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return empty_double();
    }
    if args.len() == 1 {
        return Ok(args.into_iter().next().unwrap());
    }

    let mut forwarded = Vec::with_capacity(args.len() + 1);
    forwarded.push(Value::Int(IntValue::I32(1)));
    forwarded.extend(args);
    crate::call_builtin("cat", &forwarded).map_err(adapt_cat_error)
}

fn empty_double() -> Result<Value, String> {
    Tensor::new(Vec::new(), vec![0, 0])
        .map(Value::Tensor)
        .map_err(|e| format!("vertcat: {e}"))
}

fn adapt_cat_error(message: String) -> String {
    if let Some(rest) = message.strip_prefix("cat:") {
        format!("vertcat:{rest}")
    } else if let Some(idx) = message.find("cat:") {
        let rest = &message[idx + 4..];
        format!("vertcat:{rest}")
    } else if message.starts_with("vertcat:") {
        message
    } else {
        format!("vertcat: {message}")
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{CellArray, CharArray, ComplexTensor, LogicalArray, StringArray, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn empty_invocation_returns_zero_by_zero() {
        let result = vertcat_builtin(Vec::new()).expect("vertcat");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn single_argument_round_trips() {
        let value = Value::Num(42.0);
        let result = vertcat_builtin(vec![value.clone()]).expect("vertcat");
        assert_eq!(result, value);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_vertical_concat() {
        let top = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let bottom = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();
        let result =
            vertcat_builtin(vec![Value::Tensor(top), Value::Tensor(bottom)]).expect("vertcat");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![4, 2]);
                assert_eq!(t.data, vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_arrays_concatenate_rows() {
        let top = CharArray::new("RunMat".chars().collect(), 1, 6).unwrap();
        let bottom = CharArray::new("Rocks!".chars().collect(), 1, 6).unwrap();
        let result = vertcat_builtin(vec![Value::CharArray(top), Value::CharArray(bottom)])
            .expect("vertcat");
        match result {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 2);
                assert_eq!(arr.cols, 6);
                let first: String = arr.data[..6].iter().collect();
                let second: String = arr.data[6..].iter().collect();
                assert_eq!(first, "RunMat");
                assert_eq!(second, "Rocks!");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_arrays_concatenate_rows() {
        let header = StringArray::new(vec!["Name".into(), "Score".into()], vec![1, 2]).unwrap();
        let rows = StringArray::new(vec!["Alice".into(), "98".into()], vec![1, 2]).unwrap();
        let result = vertcat_builtin(vec![Value::StringArray(header), Value::StringArray(rows)])
            .expect("vertcat");
        match result {
            Value::StringArray(arr) => {
                assert_eq!(arr.shape, vec![2, 2]);
                assert_eq!(arr.data, vec!["Name", "Alice", "Score", "98"]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mismatched_columns_error_mentions_vertcat() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let b = Tensor::new(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]).unwrap();
        let err = vertcat_builtin(vec![Value::Tensor(a), Value::Tensor(b)]).unwrap_err();
        assert!(err.starts_with("vertcat:"));
        assert!(err.contains("dimension 2 mismatch"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vertcat_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let top = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
            let bottom = Tensor::new(vec![5.0, 7.0], vec![2, 1]).unwrap();
            let view_top = runmat_accelerate_api::HostTensorView {
                data: &top.data,
                shape: &top.shape,
            };
            let view_bottom = runmat_accelerate_api::HostTensorView {
                data: &bottom.data,
                shape: &bottom.shape,
            };
            let h_top = provider.upload(&view_top).expect("upload top");
            let h_bottom = provider.upload(&view_bottom).expect("upload bottom");
            let result = vertcat_builtin(vec![Value::GpuTensor(h_top), Value::GpuTensor(h_bottom)])
                .expect("vertcat");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.data, vec![1.0, 3.0, 5.0, 7.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_arrays_concatenate_rows() {
        let first = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap();
        let second = LogicalArray::new(vec![0, 1, 0], vec![1, 3]).unwrap();
        let result = vertcat_builtin(vec![
            Value::LogicalArray(first),
            Value::LogicalArray(second),
        ])
        .expect("vertcat logical");
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![2, 3]);
                assert_eq!(array.data, vec![1, 0, 0, 1, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_arrays_concatenate_rows() {
        let first = ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![2, 1]).unwrap();
        let second = ComplexTensor::new(vec![(5.0, 6.0), (7.0, 8.0)], vec![2, 1]).unwrap();
        let result = vertcat_builtin(vec![
            Value::ComplexTensor(first),
            Value::ComplexTensor(second),
        ])
        .expect("vertcat complex");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![4, 1]);
                assert_eq!(
                    ct.data,
                    vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
                );
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_arrays_concatenate_rows() {
        let first = CellArray::new(vec![Value::Num(1.0), Value::from("low")], 1, 2).unwrap();
        let second = CellArray::new(vec![Value::Num(2.0), Value::from("high")], 1, 2).unwrap();
        let result =
            vertcat_builtin(vec![Value::Cell(first), Value::Cell(second)]).expect("vertcat cell");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 2);
                assert_eq!(cell.cols, 2);
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vertcat_like_gpu_from_host_inputs() {
        test_support::with_test_provider(|provider| {
            let prototype = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let proto_view = runmat_accelerate_api::HostTensorView {
                data: &prototype.data,
                shape: &prototype.shape,
            };
            let proto_handle = provider.upload(&proto_view).expect("upload proto");

            let top = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
            let bottom = Tensor::new(vec![5.0, 7.0], vec![2, 1]).unwrap();
            let result = vertcat_builtin(vec![
                Value::Tensor(top),
                Value::Tensor(bottom),
                Value::from("like"),
                Value::GpuTensor(proto_handle),
            ])
            .expect("vertcat like");
            let handle = match result {
                Value::GpuTensor(h) => h,
                other => panic!("expected gpu tensor, got {other:?}"),
            };
            let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.data, vec![1.0, 3.0, 5.0, 7.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn vertcat_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let top = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
        let bottom = Tensor::new(vec![5.0, 7.0], vec![2, 1]).unwrap();

        let cpu_value = vertcat_builtin(vec![
            Value::Tensor(top.clone()),
            Value::Tensor(bottom.clone()),
        ])
        .expect("cpu vertcat");
        let expected = match cpu_value {
            Value::Tensor(t) => t,
            other => panic!("expected tensor output, got {other:?}"),
        };

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view_top = runmat_accelerate_api::HostTensorView {
            data: &top.data,
            shape: &top.shape,
        };
        let view_bottom = runmat_accelerate_api::HostTensorView {
            data: &bottom.data,
            shape: &bottom.shape,
        };
        let ht = provider.upload(&view_top).expect("upload top");
        let hb = provider.upload(&view_bottom).expect("upload bottom");
        let gpu_value =
            vertcat_builtin(vec![Value::GpuTensor(ht), Value::GpuTensor(hb)]).expect("gpu vertcat");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape, expected.shape);
        assert_eq!(gathered.data, expected.data);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
