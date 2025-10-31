//! MATLAB-compatible `class` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};
use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "class"
category: "introspection"
keywords: ["class", "type inspection", "gpuArray class", "MATLAB class name", "isa"]
summary: "Return the MATLAB class name for scalars, arrays, handles, and objects."
references:
  - "https://www.mathworks.com/help/matlab/ref/class.html"
  - "https://www.mathworks.com/help/matlab/ref/isa.html"
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "No GPU kernels run. RunMat inspects metadata for gpuArray inputs and returns the class string without gathering data."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::introspection::class::tests"
  integration: null
---

# What does the `class` function do in MATLAB / RunMat?
`class(x)` returns the name of the MATLAB class that `x` belongs to. The result is a string scalar that you can use for diagnostics, branching, or display logic.

## How does the `class` function behave in MATLAB / RunMat?
- Real or complex double-precision inputs (including empty numeric arrays) report `"double"`. Integer scalars or tensors report their narrow class such as `"int16"` or `"uint8"`.
- Logical scalars and arrays report `"logical"`. Character arrays return `"char"`, while string scalars or arrays return `"string"`.
- Cell arrays, structs, and user-defined `classdef` objects report `"cell"`, `"struct"`, or the class name that defined the object.
- Function handles and anonymous closures both report `"function_handle"`. Listeners return `"event.listener"`.
- `gpuArray` values report `"gpuArray"` without gathering data. Use `classUnderlying` if you need the element type of the device array.
- Handle objects, including deleted handles, return the class name that introduced the handle. Metadata-only values such as `classref("Point")` report `"meta.class"`.

## GPU Execution Behaviour
`class` is an introspection-only builtin. When the input resides on the GPU, RunMat reads the residency metadata, returns the class name immediately, and leaves the data in place. No kernels are launched and no buffers are copied back to the CPU, so providers do not need to expose any hooks.

## Examples of using the `class` function in MATLAB / RunMat

### Check the class of a numeric scalar
```matlab
c = class(42);
```
Expected output:
```matlab
c = "double"
```

### Inspect the class of a string array
```matlab
names = ["Ada", "Grace", "Edsger"];
class_name = class(names);
```
Expected output:
```matlab
class_name = "string"
```

### Determine whether a cell array stays typed as a cell
```matlab
cells = {1, "two", 3};
cell_class = class(cells);
```
Expected output:
```matlab
cell_class = "cell"
```

### Detect the class of gpuArray data without gathering
```matlab
G = gpuArray(rand(4));
g_class = class(G);
```
Expected output:
```matlab
g_class = "gpuArray"
```

### Report the class name of a custom object
```matlab
P = classref("Point").origin();
point_class = class(P);
```
Expected output:
```matlab
point_class = "Point"
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You do not need to call `gpuArray` purely to query a value’s class. The auto-offload planner keeps tensors on the GPU whenever profitable, and `class` simply returns the residency-aware class name without altering where the data lives. Explicit `gpuArray` and `gather` calls remain available for compatibility with MATLAB code that manages residency manually.

## FAQ

### Does `class` return a string or a character array?
`class` returns a string scalar, matching modern MATLAB behaviour. Convert the result with `char(class(...))` if you need a legacy character row vector.

### How do I check whether a value is numeric?
Use `isa(x, "numeric")` to handle floats, complex numbers, and integers. `class` reports the concrete class (`"double"`, `"int32"`, and so on).

### Will `class` gather gpuArray inputs?
No. RunMat inspects metadata and returns `"gpuArray"` immediately; buffers stay on the device.

### What does `class` return for handle objects?
`class` returns the object’s defining class name (for example `"MyHandleClass"`). Use `isa(h, "handle")` to test for handle semantics.

### What happens if I pass a struct or cell array?
Structs return `"struct"` and cell arrays return `"cell"`, matching MATLAB.

### What is the class of function handles?
Both named function handles and anonymous closures report `"function_handle"`.

### Can I distinguish string and char arrays?
Yes. `class("hello")` yields `"string"` while `class('hello')` yields `"char"`.

### What about metadata objects such as `classref`?
Calling `class(classref("Point"))` returns `"meta.class"`. Use the returned meta-class object to inspect properties or superclasses.

## See Also
`isa`, `isnumeric`, `classref`, `gpuArray`, `gather`

## Source & Feedback
- The full source code for `class` lives at `crates/runmat-runtime/src/builtins/introspection/class.rs`.
- Found a behavioural difference? Please open an issue with a minimal reproduction.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "class",
    op_kind: GpuOpKind::Custom("introspection"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Introspection-only builtin; providers do not need to implement hooks. RunMat reads residency metadata and returns a host string.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "class",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion; class executes on the host and returns a string scalar.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("class", DOC_MD);

#[runtime_builtin(
    name = "class",
    category = "introspection",
    summary = "Return the MATLAB class name for scalars, arrays, and objects.",
    keywords = "class,type inspection,type name,gpuArray class"
)]
fn class_builtin(value: Value) -> Result<String, String> {
    Ok(class_name_for_value(&value))
}

/// Return the canonical MATLAB class name for a runtime value.
pub(crate) fn class_name_for_value(value: &Value) -> String {
    match value {
        Value::Num(_) | Value::Tensor(_) | Value::ComplexTensor(_) | Value::Complex(_, _) => {
            "double".to_string()
        }
        Value::Int(iv) => iv.class_name().to_string(),
        Value::Bool(_) | Value::LogicalArray(_) => "logical".to_string(),
        Value::String(_) | Value::StringArray(_) => "string".to_string(),
        Value::CharArray(_) => "char".to_string(),
        Value::Cell(_) => "cell".to_string(),
        Value::Struct(_) => "struct".to_string(),
        Value::GpuTensor(_) => "gpuArray".to_string(),
        Value::FunctionHandle(_) | Value::Closure(_) => "function_handle".to_string(),
        Value::HandleObject(handle) => {
            if handle.class_name.is_empty() {
                "handle".to_string()
            } else {
                handle.class_name.clone()
            }
        }
        Value::Listener(_) => "event.listener".to_string(),
        Value::Object(obj) => obj.class_name.clone(),
        Value::ClassRef(_) => "meta.class".to_string(),
        Value::MException(_) => "MException".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        CellArray, CharArray, Closure, ComplexTensor, HandleRef, IntValue, Listener, LogicalArray,
        MException, ObjectInstance, StringArray, StructValue, Tensor,
    };
    use runmat_gc_api::GcPtr;

    #[test]
    fn class_reports_double_for_numeric_scalars() {
        let name = class_builtin(Value::Num(3.14)).expect("class");
        assert_eq!(name, "double");
    }

    #[test]
    fn class_reports_integer_type_names() {
        let name = class_builtin(Value::Int(IntValue::I32(12))).expect("class");
        assert_eq!(name, "int32");
    }

    #[test]
    fn class_reports_expected_names_for_core_types() {
        let logical_scalar = Value::Bool(true);
        assert_eq!(class_name_for_value(&logical_scalar), "logical");

        let logical_array = Value::LogicalArray(
            LogicalArray::new(vec![1u8, 0u8, 1u8, 1u8], vec![2, 2]).expect("logical array"),
        );
        assert_eq!(class_name_for_value(&logical_array), "logical");

        let string_scalar = Value::String("hello".to_string());
        assert_eq!(class_name_for_value(&string_scalar), "string");

        let string_array = Value::StringArray(
            StringArray::new(vec!["Ada".into(), "Grace".into()], vec![1, 2]).expect("string array"),
        );
        assert_eq!(class_name_for_value(&string_array), "string");

        let char_array = Value::CharArray(CharArray::new_row("abc"));
        assert_eq!(class_name_for_value(&char_array), "char");

        let real_tensor = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
        assert_eq!(class_name_for_value(&real_tensor), "double");

        let complex_scalar = Value::Complex(1.0, -1.0);
        assert_eq!(class_name_for_value(&complex_scalar), "double");

        let complex_tensor = Value::ComplexTensor(
            ComplexTensor::new(vec![(1.0, 1.0), (2.0, -3.0)], vec![2, 1]).expect("complex tensor"),
        );
        assert_eq!(class_name_for_value(&complex_tensor), "double");

        let cell =
            Value::Cell(CellArray::new(vec![Value::Num(1.0), Value::Bool(false)], 1, 2).unwrap());
        assert_eq!(class_name_for_value(&cell), "cell");

        let mut st = StructValue::new();
        st.fields.insert("field".into(), Value::Num(42.0));
        let struct_value = Value::Struct(st);
        assert_eq!(class_name_for_value(&struct_value), "struct");

        let func_handle = Value::FunctionHandle("sin".into());
        assert_eq!(class_name_for_value(&func_handle), "function_handle");

        let closure = Value::Closure(Closure {
            function_name: "anon".into(),
            captures: vec![],
        });
        assert_eq!(class_name_for_value(&closure), "function_handle");

        let class_ref = Value::ClassRef("pkg.Point".into());
        assert_eq!(class_name_for_value(&class_ref), "meta.class");

        let exception = Value::MException(MException::new("id:err".into(), "fail".into()));
        assert_eq!(class_name_for_value(&exception), "MException");
    }

    #[test]
    fn class_reports_gpuarray_without_gather() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let name = class_builtin(Value::GpuTensor(handle)).expect("class");
            assert_eq!(name, "gpuArray");
        });
    }

    #[test]
    fn class_reports_handle_class_names() {
        let fallback = HandleRef {
            class_name: String::new(),
            target: GcPtr::null(),
            valid: false,
        };
        let fallback_name = class_builtin(Value::HandleObject(fallback)).expect("class");
        assert_eq!(fallback_name, "handle");

        let handle = HandleRef {
            class_name: "MockHandle".into(),
            target: GcPtr::null(),
            valid: true,
        };
        let name = class_builtin(Value::HandleObject(handle)).expect("class");
        assert_eq!(name, "MockHandle");
    }

    #[test]
    fn class_reports_object_and_listener_classes() {
        let object = ObjectInstance::new("pkg.Point".into());
        let obj_name = class_builtin(Value::Object(object)).expect("class object");
        assert_eq!(obj_name, "pkg.Point");

        let listener = Listener {
            id: 1,
            target: GcPtr::null(),
            event_name: "changed".into(),
            callback: GcPtr::null(),
            enabled: true,
            valid: true,
        };
        let listener_name = class_builtin(Value::Listener(listener)).expect("class listener");
        assert_eq!(listener_name, "event.listener");
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn class_reports_gpuarray_with_wgpu_provider() {
        use runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider;
        use runmat_accelerate_api::AccelProvider;

        // Attempt to register a WGPU provider; skip if the environment lacks a compatible adapter.
        let provider = match ensure_wgpu_provider() {
            Ok(Some(p)) => p,
            _ => return,
        };

        let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("wgpu upload");
        let name = class_builtin(Value::GpuTensor(handle)).expect("class");
        assert_eq!(name, "gpuArray");
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
