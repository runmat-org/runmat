//! MATLAB-compatible `isstring` builtin with GPU-aware semantics for RunMat.
//!
//! Determines whether a value is a MATLAB string array (including string scalars) without moving
//! data between host and device memory.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "isstring",
        builtin_path = "crate::builtins::introspection::isstring"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "isstring"
category: "introspection"
keywords: ["isstring", "string array", "type checking", "string scalar", "logical predicate", "gpuArray"]
summary: "Return true when a value is a MATLAB string array (including string scalars)."
references:
  - "https://www.mathworks.com/help/matlab/ref/isstring.html"
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Inspects metadata on the host; gpuArray inputs are classified without kernel launches."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::introspection::isstring::tests"
  integration: "builtins::introspection::isstring::tests::isstring_gpu_inputs_return_false"
---

# What does the `isstring` function do in MATLAB / RunMat?
`tf = isstring(x)` returns a logical scalar `true` when `x` is a MATLAB string array (including
string scalars and empty string arrays) and logical `false` otherwise. Use it when you need to
validate arguments, branch on string support, or distinguish strings from other text
representations.

## How does the `isstring` function behave in MATLAB / RunMat?
- String scalars such as `"RunMat"` and string arrays like `["alpha", "beta"]` both return
  `logical(1)`.
- Empty string arrays produced by `strings(0)` or `string.empty(...)` are still recognised as the
  string type, so they return `logical(1)`.
- Character arrays created with single quotes (`'text'`), cell arrays of character vectors, numeric
  tensors, logical masks, structs, tables, and other non-string containers return `logical(0)`.
- gpuArray inputs currently store numeric or logical tensors in RunMat, so `isstring(gpuArray(...))`
  returns `logical(0)`, matching MATLAB’s behaviour.
- The result is always a host logical scalar, making it easy to use for argument validation,
  control-flow guards, or input parsing.

## `isstring` Function GPU Execution Behaviour
`isstring` runs entirely from metadata. When you pass a gpuArray, RunMat inspects the handle and
returns `logical(0)` because device buffers do not yet store string data. No kernels are launched,
and no host↔device transfers occur, so the device allocation remains resident for subsequent
operations.

## GPU residency in RunMat (Do I need `gpuArray`?)

You usually do NOT need to call `gpuArray` yourself in RunMat (unlike MATLAB). 

In RunMat, the fusion planner keeps residency on GPU in branches of fused expressions. As such,
calling `isstring` on a gpuArray result leaves the device allocation untouched while still answering
the question using metadata only.

To preserve backwards compatibility with MathWorks MATLAB, and for when you want to explicitly
bootstrap GPU residency, you can call `gpuArray` explicitly to move data to the GPU if you want to
be explicit about the residency.

Since MathWorks MATLAB does not have a fusion planner, and they kept their parallel execution
toolbox separate from the core language, as their toolbox is a separate commercial product,
MathWorks MATLAB users need to call `gpuArray` to move data to the GPU manually whereas RunMat users
can rely on the fusion planner to keep data on the GPU automatically.

## Examples of using the `isstring` function in MATLAB / RunMat

### Checking if a string scalar is recognised
```matlab
tf = isstring("RunMat");
```
Expected output:
```matlab
tf = logical(1)
```

### Detecting string arrays with multiple elements
```matlab
values = ["alpha", "beta", "gamma"];
tf = isstring(values);
```
Expected output:
```matlab
tf = logical(1)
```

### Distinguishing strings from character vectors
```matlab
tf_string = isstring("hello");
tf_char   = isstring('hello');
```
Expected output:
```matlab
tf_string = logical(1)
tf_char   = logical(0)
```

### Confirming that empty string arrays are still strings
```matlab
emptyStrs = strings(0, 2);
tf = isstring(emptyStrs);
```
Expected output:
```matlab
tf = logical(1)
```

### Recognising that numeric and logical arrays are not strings
```matlab
numbers = [1 2 3];
mask = true(1, 3);
tf_numbers = isstring(numbers);
tf_mask = isstring(mask);
```
Expected output:
```matlab
tf_numbers = logical(0)
tf_mask = logical(0)
```

### Validating cell arrays of character vectors
```matlab
cells = {'a', 'b', 'c'};
tf = isstring(cells);
```
Expected output:
```matlab
tf = logical(0)
```

### Inspecting gpuArray inputs without transfers
```matlab
G = gpuArray(ones(2, 2));
tf = isstring(G);
```
Expected output:
```matlab
tf = logical(0)
```

### Using `isstring` in argument validation
```matlab
function greet(name)
    if ~isstring(name)
        error("Name must be provided as a string.");
    end
    disp("Hello " + name + "!");
end

greet("RunMat");
% greet(42); % throws "Name must be provided as a string."
```
Expected output:
```matlab
Hello RunMat!
```

## FAQ

### Does `isstring` return true for empty string arrays?
Yes. Empty string arrays are still of type string, so `isstring(strings(0))` returns `logical(1)`.

### How is `isstring` different from `ischar`?
`isstring` recognises MATLAB’s modern string type introduced in R2016b. `ischar` recognises legacy
character arrays created with single quotes. Only `isstring` returns `true` for double-quoted
values.

### What about string arrays stored inside cell arrays?
Cell arrays are containers, so `isstring({'a', 'b'})` returns `logical(0)`. Use `iscellstr` or
`all(isstring(cells))` to validate the contents instead.

### Can gpuArray hold string data in RunMat?
Not currently. gpuArray values hold numeric or logical tensors, so `isstring` always reports
`logical(0)` for device-resident data.

### Does `isstring` inspect array contents element by element?
No. It only checks the value’s type metadata, so the cost is constant regardless of array size.

## See Also
[ischar](./ischar), [isa](./isa), [string](../strings/core/string), [convertStringsToChars](../strings/manipulation/convertStringsToChars), [gpuArray](../acceleration/gpu/gpuArray)

## Source & Feedback
- The full source code for the implementation of the `isstring` function is available at: [`crates/runmat-runtime/src/builtins/introspection/isstring.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/isstring.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::introspection::isstring")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isstring",
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
    notes: "Metadata-only predicate; gpuArray inputs stay on device while the result is returned on the host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::introspection::isstring")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isstring",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Type-check predicate that does not participate in fusion planning.",
};

#[runtime_builtin(
    name = "isstring",
    category = "introspection",
    summary = "Return true when a value is a MATLAB string array.",
    keywords = "isstring,string array,string scalar,type checking,introspection",
    accel = "metadata",
    builtin_path = "crate::builtins::introspection::isstring"
)]
fn isstring_builtin(value: Value) -> Result<Value, String> {
    Ok(Value::Bool(matches!(
        value,
        Value::String(_) | Value::StringArray(_)
    )))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        CellArray, CharArray, Closure, ComplexTensor, LogicalArray, MException, ObjectInstance,
        StringArray, StructValue, Tensor,
    };

    #[test]
    fn string_scalar_reports_true() {
        let value = Value::String("RunMat".to_string());
        let result = isstring_builtin(value).expect("isstring");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn complex_object_and_handle_variants_report_false() {
        let complex_scalar = Value::Complex(1.25, -3.5);
        let complex_tensor = Value::ComplexTensor(
            ComplexTensor::new(vec![(0.0, 1.0), (2.0, -2.5)], vec![2, 1]).expect("complex tensor"),
        );
        let object = Value::Object(ObjectInstance::new("ExampleClass".to_string()));
        let closure = Value::Closure(Closure {
            function_name: "some_func".to_string(),
            captures: Vec::new(),
        });
        let class_ref = Value::ClassRef("pkg.Type".to_string());
        let exception = Value::MException(MException::new(
            "RunMat:Test".to_string(),
            "example".to_string(),
        ));
        let function_handle = Value::FunctionHandle("sin".to_string());

        for candidate in vec![
            complex_scalar,
            complex_tensor,
            object,
            closure,
            class_ref,
            exception,
            function_handle,
        ] {
            assert_eq!(
                isstring_builtin(candidate).expect("isstring"),
                Value::Bool(false)
            );
        }
    }

    #[test]
    fn string_array_reports_true() {
        let array = StringArray::new(vec!["one".to_string(), "two".to_string()], vec![1, 2])
            .expect("string array");
        let result = isstring_builtin(Value::StringArray(array)).expect("isstring");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn empty_string_array_reports_true() {
        let array = StringArray::new(vec![], vec![0, 0]).expect("empty string array");
        let result = isstring_builtin(Value::StringArray(array)).expect("isstring");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn character_arrays_report_false() {
        let chars = CharArray::new_row("RunMat");
        let result = isstring_builtin(Value::CharArray(chars)).expect("isstring");
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn numeric_and_logical_values_report_false() {
        let tensor = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).expect("tensor"));
        let logical = Value::Bool(true);
        let logical_array = Value::LogicalArray(
            LogicalArray::new(vec![1u8, 0u8], vec![2, 1]).expect("logical array"),
        );

        assert_eq!(
            isstring_builtin(tensor).expect("isstring"),
            Value::Bool(false)
        );
        assert_eq!(
            isstring_builtin(logical).expect("isstring"),
            Value::Bool(false)
        );
        assert_eq!(
            isstring_builtin(logical_array).expect("isstring"),
            Value::Bool(false)
        );
    }

    #[test]
    fn cell_and_struct_values_report_false() {
        let empty_cell = CellArray::new(Vec::<Value>::new(), 0, 0).expect("cell");
        let empty_struct = StructValue::new();
        assert_eq!(
            isstring_builtin(Value::Cell(empty_cell)).expect("isstring"),
            Value::Bool(false)
        );
        assert_eq!(
            isstring_builtin(Value::Struct(empty_struct)).expect("isstring"),
            Value::Bool(false)
        );
    }

    #[test]
    fn isstring_gpu_inputs_return_false() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).expect("tensor");
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = isstring_builtin(Value::GpuTensor(handle)).expect("isstring");
            assert_eq!(result, Value::Bool(false));
        });
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn isstring_wgpu_numeric_returns_false() {
        register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu provider");
        let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).expect("tensor");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("upload");
        let result = isstring_builtin(Value::GpuTensor(handle)).expect("isstring");
        assert_eq!(result, Value::Bool(false));
    }
}
