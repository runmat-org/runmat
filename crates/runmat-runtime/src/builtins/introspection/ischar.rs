//! MATLAB-compatible `ischar` builtin with GPU-aware semantics for RunMat.
//!
//! Detects whether a value is a MATLAB character array while preserving host/GPU residency.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "ischar",
        wasm_path = "crate::builtins::introspection::ischar"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "ischar"
category: "introspection"
keywords: ["ischar", "character arrays", "type checking", "char", "logical predicate", "gpuArray"]
summary: "Test whether a value is a MATLAB character array (char vector or char matrix)."
references:
  - "https://www.mathworks.com/help/matlab/ref/ischar.html"
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Returns a host logical scalar; GPU inputs are inspected without launching kernels."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::introspection::ischar::tests"
  integration: "builtins::introspection::ischar::tests::ischar_gpu_inputs_return_false"
---

# What does the `ischar` function do in MATLAB / RunMat?
`tf = ischar(x)` returns logical `true` when `x` is a MATLAB character array (a char row vector or
matrix) and logical `false` otherwise. Use it to distinguish traditional char arrays from the newer
string type, numeric tensors, cells, or gpuArray values.

## How does the `ischar` function behave in MATLAB / RunMat?
- Character vectors such as `'RunMat'` and char matrices created with `['A'; 'B']` return `true`.
- Empty char arrays (including `''` and `char.empty(...)`) still count as character arrays.
- String scalars, string arrays, cell arrays, numeric values, and logical masks all return `false`.
- gpuArray inputs return `false` unless they are explicitly stored as char arrays (which RunMat does
  not currently support), matching MATLAB’s behaviour where gpuArray does not host char data.
- The result is always a host logical scalar (`logical(0)` or `logical(1)`), so it can be used in
  branching, masking, or validation code without additional conversions.

## `ischar` Function GPU Execution Behaviour
`ischar` never launches GPU kernels. When you pass a gpuArray value, RunMat inspects the residency
metadata and simply reports `false` because device-resident buffers are numeric or logical. This
matches MATLAB’s semantics and avoids unnecessary host↔device transfers.

## GPU residency in RunMat (Do I need `gpuArray`?)

You usually do NOT need to call `gpuArray` yourself in RunMat (unlike MATLAB). 

In RunMat, the fusion planner keeps residency on GPU in branches of fused expressions. As such,
calling `ischar` on a gpuArray result preserves the device buffer, and `ischar` answers the query
using metadata only.

To preserve backwards compatibility with MathWorks MATLAB, and for when you want to explicitly
bootstrap GPU residency, you can call `gpuArray` explicitly to move data to the GPU if you want to
be explicit about the residency.

Since MathWorks MATLAB does not have a fusion planner, and they kept their parallel execution
toolbox separate from the core language, as their toolbox is a separate commercial product,
MathWorks MATLAB users need to call `gpuArray` to move data to the GPU manually whereas RunMat users
can rely on the fusion planner to keep data on the GPU automatically.

## Examples of using the `ischar` function in MATLAB / RunMat

### Checking if a character vector is a char array
```matlab
tf = ischar('RunMat');
```
Expected output:
```matlab
tf = logical(1)
```

### Detecting multi-row char matrices
```matlab
letters = ['ab'; 'cd'];
tf = ischar(letters);
```
Expected output:
```matlab
tf = logical(1)
```

### Distinguishing between char arrays and string scalars
```matlab
tf_char = ischar('hello');
tf_string = ischar("hello");
```
Expected output:
```matlab
tf_char = logical(1)
tf_string = logical(0)
```

### Recognising that numeric and logical data are not char arrays
```matlab
numbers = [1 2 3];
mask = true(1, 3);
tf_numbers = ischar(numbers);
tf_mask = ischar(mask);
```
Expected output:
```matlab
tf_numbers = logical(0)
tf_mask = logical(0)
```

### Testing gpuArray inputs
```matlab
G = gpuArray(ones(2, 2));
tf_gpu = ischar(G);
```
Expected output:
```matlab
tf_gpu = logical(0)
```

## FAQ

### Does `ischar` treat empty character arrays as char?
Yes. Both `''` and arrays created with `char.empty` return `logical(1)` because they are still
character arrays, even when they have zero elements.

### Why does `ischar` return `false` for string scalars?
Strings in MATLAB are a distinct type introduced in R2016b. They are not interchangeable with char
arrays, so `ischar("text")` correctly returns `logical(0)`.

### What about cell arrays of characters?
Cell arrays such as `{'a', 'b'}` return `logical(0)` because they are cell containers rather than a
single char array. Use `iscellstr` if you need to validate cell-of-char collections.

### Can gpuArray hold char data in RunMat?
Not currently. gpuArray values in RunMat contain numeric or logical tensors, so `ischar` reports
`false` for all gpuArray inputs, mirroring MATLAB’s behaviour.

### How do I convert a string to a char array if `ischar` returns false?
Use `char(stringScalar)` or the `convertStringsToChars` utility when you need to operate on char
arrays for legacy APIs.

### Is there any performance cost when checking large arrays?
No. `ischar` only inspects the value’s metadata; it does not copy or iterate over the array
elements, so the cost is constant regardless of array size.

## See Also
[isa](./isa), [char](../strings/core/char), [string](../strings/core/string), [strcmp](../strings/core/strcmp), [gpuArray](../acceleration/gpu/gpuArray)

## Source & Feedback
- The full source code for the implementation of the `ischar` function is available at: [`crates/runmat-runtime/src/builtins/introspection/ischar.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/ischar.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(wasm_path = "crate::builtins::introspection::ischar")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ischar",
    op_kind: GpuOpKind::Custom("metadata"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Runs entirely on the host and inspects value metadata; gpuArray inputs return logical false.",
};

#[runmat_macros::register_fusion_spec(wasm_path = "crate::builtins::introspection::ischar")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ischar",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Metadata-only predicate that does not participate in fusion planning.",
};

#[runtime_builtin(
    name = "ischar",
    category = "introspection",
    summary = "Return true when a value is a MATLAB character array.",
    keywords = "ischar,char array,type checking,introspection",
    accel = "metadata",
    wasm_path = "crate::builtins::introspection::ischar"
)]
fn ischar_builtin(value: Value) -> Result<Value, String> {
    Ok(Value::Bool(matches!(value, Value::CharArray(_))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{CellArray, CharArray, LogicalArray, StringArray, StructValue, Tensor};

    #[test]
    fn character_vector_reports_true() {
        let chars = CharArray::new_row("RunMat");
        let result = ischar_builtin(Value::CharArray(chars)).expect("ischar");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn character_matrix_reports_true() {
        let chars = CharArray::new(vec!['a', 'b', 'c', 'd', 'e', 'f'], 2, 3).expect("char array");
        let result = ischar_builtin(Value::CharArray(chars)).expect("ischar");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn string_values_report_false() {
        let scalar = Value::String("RunMat".to_string());
        let array = Value::StringArray(
            StringArray::new(vec!["a".to_string(), "b".to_string()], vec![1, 2]).expect("strings"),
        );
        assert_eq!(ischar_builtin(scalar).expect("ischar"), Value::Bool(false));
        assert_eq!(ischar_builtin(array).expect("ischar"), Value::Bool(false));
    }

    #[test]
    fn numeric_and_logical_values_report_false() {
        let numeric = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).expect("tensor"));
        let logical = Value::Bool(true);
        let logical_array =
            Value::LogicalArray(LogicalArray::new(vec![1u8], vec![1, 1]).expect("logical array"));
        assert_eq!(ischar_builtin(numeric).expect("ischar"), Value::Bool(false));
        assert_eq!(ischar_builtin(logical).expect("ischar"), Value::Bool(false));
        assert_eq!(
            ischar_builtin(logical_array).expect("ischar"),
            Value::Bool(false)
        );
    }

    #[test]
    fn cell_array_reports_false() {
        let cell = CellArray::new(vec![Value::Num(1.0), Value::from("text")], 1, 2).expect("cell");
        let result = ischar_builtin(Value::Cell(cell)).expect("ischar");
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn struct_value_reports_false() {
        let mut st = StructValue::new();
        st.insert("field", Value::Num(1.0));
        let result = ischar_builtin(Value::Struct(st)).expect("ischar");
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn function_handle_reports_false() {
        let fh = Value::FunctionHandle("sin".to_string());
        let result = ischar_builtin(fh).expect("ischar");
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn ischar_gpu_inputs_return_false() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).expect("tensor");
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let value = Value::GpuTensor(handle);
            let result = ischar_builtin(value).expect("ischar");
            assert_eq!(result, Value::Bool(false));
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn ischar_wgpu_numeric_returns_false() {
        let _ = register_wgpu_provider(WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let data = vec![0.0, 1.0];
        let shape = vec![2, 1];
        let view = HostTensorView {
            data: &data,
            shape: &shape,
        };
        let handle = provider.upload(&view).expect("upload to GPU");
        let result = ischar_builtin(Value::GpuTensor(handle)).expect("ischar");
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn empty_character_array_reports_true() {
        let chars = CharArray::new(Vec::new(), 0, 0).expect("empty char array");
        let result = ischar_builtin(Value::CharArray(chars)).expect("ischar");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(
            !blocks.is_empty(),
            "expected DOC_MD to include at least one executable example"
        );
    }
}
