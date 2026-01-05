//! MATLAB-compatible `isa` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::introspection::class::class_name_for_value;
use runmat_accelerate_api::handle_is_logical;
use runmat_builtins::{get_class, Value};
use runmat_macros::runtime_builtin;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "isa",
        builtin_path = "crate::builtins::introspection::isa"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "isa"
category: "introspection"
keywords: ["isa", "type checking", "class comparison", "numeric category", "gpuArray"]
summary: "Test whether a value belongs to a specified MATLAB class or abstract category."
references:
  - "https://www.mathworks.com/help/matlab/ref/isa.html"
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Returns host logical scalars and inspects gpuArray metadata without launching kernels."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::introspection::isa::tests"
  integration: "builtins::introspection::isa::tests::isa_gpu_arrays_treat_metadata_correctly"
---

# What does the `isa` function do in MATLAB / RunMat?
`isa(x, T)` returns logical `true` when the value `x` is a member of the MATLAB class or abstract
category identified by `T`. Use it to branch on numeric vs integer data, handle objects, GPU
arrays, function handles, and user-defined classes.

## How does the `isa` function behave in MATLAB / RunMat?
- Exact class names (such as `"double"`, `"cell"`, `"struct"`, `"event.listener"`) match the value’s
  dynamic class. Comparisons are case-insensitive.
- Abstract categories (`"numeric"`, `"float"`, `"integer"`, `"handle"`, `"function_handle"`,
  `"gpuArray"`, `"listener"`) mirror MATLAB’s grouping rules.
- Logical scalars, logical arrays, and logical gpuArray masks satisfy `"logical"` and do not
  register as `"numeric"`. RunMat consults handle metadata rather than downloading device buffers.
- Handle inheritance is respected: if a class derives from `handle`, `isa(obj, "handle")` returns
  `true` even when `class(obj)` reports a subclass name.
- `isa` accepts either a string scalar or a character row vector as the class designator.
- gpuArray inputs never need to be gathered; RunMat checks residency metadata to answer the query.

## `isa` Function GPU Execution Behaviour
`isa` never launches GPU kernels. For gpuArray inputs RunMat inspects the `GpuTensorHandle`
metadata to identify the device array and satisfy category checks such as `"numeric"` and
`"gpuArray"`. Because the builtin only returns a host logical scalar it does not participate in
fusion, and providers do not need to expose additional hooks.

## GPU residency in RunMat (Do I need `gpuArray`?)

You usually do NOT need to call `gpuArray` yourself in RunMat (unlike MATLAB). 

In RunMat, the fusion planner keeps residency on GPU in branches of fused expressions. As such, when
you call `isa` on a gpuArray result, the planner preserves the device buffer and `isa` answers the
query using metadata only.

To preserve backwards compatibility with MathWorks MATLAB, and for when you want to explicitly
bootstrap GPU residency, you can call `gpuArray` explicitly to move data to the GPU if you want to
be explicit about the residency.

Since MathWorks MATLAB does not have a fusion planner, and they kept their parallel execution
toolbox separate from the core language, as their toolbox is a separate commercial product,
MathWorks MATLAB users need to call `gpuArray` to move data to the GPU manually whereas RunMat users
can rely on the fusion planner to keep data on the GPU automatically.

## Examples of using the `isa` function in MATLAB / RunMat

### Checking whether a scalar is double precision
```matlab
tf = isa(42, "double");
```
Expected output:
```matlab
tf = logical(1)
```

### Testing numeric arrays against the numeric category
```matlab
A = rand(3, 4);
is_numeric = isa(A, "numeric");
is_integer = isa(A, "integer");
```
Expected output:
```matlab
is_numeric = logical(1)
is_integer = logical(0)
```

### Confirming that gpuArray data counts as numeric
```matlab
G = gpuArray(rand(1024, 1024));
tf_numeric = isa(G, "numeric");
tf_double = isa(G, "double");   % false — the class reports gpuArray
```
Expected output:
```matlab
tf_numeric = logical(1)
tf_double = logical(0)
```

### Determining whether a GPU mask is logical
```matlab
G = gpuArray(rand(5) > 0.5);
is_mask = isa(G, "logical");
is_numeric = isa(G, "numeric");
```
Expected output:
```matlab
is_mask = logical(1)
is_numeric = logical(0)
```

### Validating handle subclasses
```matlab
pt = pkg.TestHandle();          % derives from handle
tf_handle = isa(pt, "handle");
tf_exact = isa(pt, "pkg.TestHandle");
```
Expected output:
```matlab
tf_handle = logical(1)
tf_exact = logical(1)
```

### Comparing string scalars and character vectors
```matlab
s = "RunMat";
c = 'RunMat';
tf_string = isa(s, "string");
tf_char = isa(c, "char");
```
Expected output:
```matlab
tf_string = logical(1)
tf_char = logical(1)
```

### Detecting function handles and listeners
```matlab
fh = @sin;
tf_handle = isa(fh, "function_handle");
lst = addlistener(TestSource, "Changed", @disp);
tf_listener = isa(lst, "listener");
```
Expected output:
```matlab
tf_handle = logical(1)
tf_listener = logical(1)
```

## FAQ

### Does `isa` treat gpuArray values as numeric?
Yes for numeric device arrays. `isa(gpuArray(rand(4)), "numeric")` and `isa(..., "float")` return
`true`, even though the concrete class is `"gpuArray"`. Logical gpuArray masks skip the numeric
categories and only satisfy `"logical"`. Use `isa(..., "double")` when you specifically need host
doubles.

### How do I check for integer types?
Use the `"integer"` category. It matches MATLAB’s native integer classes (`int8`, `uint16`, and so
on). For example `isa(int32(5), "integer")` returns `true` while `isa(5, "integer")` returns `false`.

### Does `isa` understand handle inheritance?
Yes. If a class derives from `handle`, then the object passes the `"handle"` category check. The
builtin walks parent classes registered with RunMat to mirror MATLAB semantics.

### What kinds of strings can I pass as the type name?
Provide either a string scalar (double quotes) or a character row vector (single quotes). Multi-row
character arrays and string arrays with more than one element throw a descriptive error.

### How do I detect function handles or anonymous functions?
Both named function handles and closures pass `isa(value, "function_handle")`.

### Will `isa` gather gpuArray inputs or launch kernels?
No. The builtin is metadata-only and returns a host logical scalar without copying device buffers.

### How are listeners treated?
Event listeners pass both `isa(listener, "event.listener")` and the compatibility alias
`isa(listener, "listener")`. They also satisfy the `"handle"` category.

### Does `isa` differentiate between string and char arrays?
Yes. String scalars pass `"string"` while character arrays pass `"char"`. They are not interchangeable.

### Can I use `isa` to test meta-class values?
Yes. Meta-class references created with `classref` return `true` for `"meta.class"`.

## See Also
[`class`](./class), [`isnumeric`](./isnumeric), [`islogical`](./islogical), `isaUnderlying`, [`gpuArray`](./gpuarray), [`gather`](./gather)
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::introspection::isa")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isa",
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
    notes: "Metadata predicate that returns host logical scalars; no GPU kernels or gathers are required.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::introspection::isa")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isa",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Not eligible for fusion planning; isa executes on the host and produces a logical scalar.",
};

#[runtime_builtin(
    name = "isa",
    category = "introspection",
    summary = "Test whether a value belongs to a specified MATLAB class or abstract category.",
    keywords = "isa,type checking,class comparison,numeric category,gpuArray",
    accel = "metadata",
    builtin_path = "crate::builtins::introspection::isa"
)]
fn isa_builtin(value: Value, class_designator: Value) -> Result<Value, String> {
    let type_name = parse_type_name(&class_designator)?;
    let result = value_is_a(&value, &type_name);
    Ok(Value::Bool(result))
}

fn parse_type_name(value: &Value) -> Result<String, String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::StringArray(sa) => {
            if sa.rows == 1 && sa.cols == 1 && !sa.data.is_empty() {
                Ok(sa.data[0].clone())
            } else {
                Err("isa: TYPE must be a string scalar or character vector".to_string())
            }
        }
        Value::CharArray(ca) => {
            if ca.rows == 1 {
                Ok(ca.data.iter().collect())
            } else {
                Err("isa: TYPE must be a string scalar or character vector".to_string())
            }
        }
        _ => Err("isa: TYPE must be a string scalar or character vector".to_string()),
    }
}

fn value_is_a(value: &Value, requested: &str) -> bool {
    let trimmed = requested.trim();
    if trimmed.is_empty() {
        return false;
    }
    let requested_lower = trimmed.to_ascii_lowercase();
    match requested_lower.as_str() {
        "numeric" => is_numeric(value),
        "float" => is_float(value),
        "integer" => is_integer(value),
        "logical" => is_logical(value),
        "char" => matches!(value, Value::CharArray(_)),
        "string" => matches!(value, Value::String(_) | Value::StringArray(_)),
        "cell" => matches!(value, Value::Cell(_)),
        "struct" => matches!(value, Value::Struct(_)),
        "function_handle" => matches!(value, Value::FunctionHandle(_) | Value::Closure(_)),
        "gpuarray" => matches!(value, Value::GpuTensor(_)),
        "listener" | "event.listener" => matches!(value, Value::Listener(_)),
        "meta.class" => matches!(value, Value::ClassRef(_)),
        "mexception" => matches!(value, Value::MException(_)),
        "handle" => is_handle_like(value),
        _ => {
            let actual = class_name_for_value(value);
            if actual.eq_ignore_ascii_case(trimmed) {
                return true;
            }
            match value {
                Value::Object(obj) => class_inherits(&obj.class_name, &requested_lower),
                Value::HandleObject(handle) => {
                    !handle.class_name.is_empty()
                        && class_inherits(&handle.class_name, &requested_lower)
                }
                _ => false,
            }
        }
    }
}

fn is_numeric(value: &Value) -> bool {
    match value {
        Value::Num(_)
        | Value::Tensor(_)
        | Value::ComplexTensor(_)
        | Value::Complex(_, _)
        | Value::Int(_) => true,
        Value::GpuTensor(handle) => !handle_is_logical(handle),
        _ => false,
    }
}

fn is_float(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Tensor(_) | Value::ComplexTensor(_) | Value::Complex(_, _) => true,
        Value::GpuTensor(handle) => !handle_is_logical(handle),
        _ => false,
    }
}

fn is_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
}

fn is_logical(value: &Value) -> bool {
    match value {
        Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::GpuTensor(handle) => handle_is_logical(handle),
        _ => false,
    }
}

fn is_handle_like(value: &Value) -> bool {
    match value {
        Value::HandleObject(_) | Value::Listener(_) => true,
        Value::Object(obj) => class_inherits(&obj.class_name, "handle"),
        _ => false,
    }
}

fn class_inherits(class_name: &str, requested_lower: &str) -> bool {
    if class_name.eq_ignore_ascii_case(requested_lower) {
        return true;
    }
    let mut cursor = Some(class_name.to_string());
    while let Some(name) = cursor {
        if name.eq_ignore_ascii_case(requested_lower) {
            return true;
        }
        if let Some(def) = get_class(&name) {
            cursor = def.parent.clone();
        } else {
            break;
        }
    }
    false
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::{gpu_helpers, test_support};
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        CellArray, CharArray, ClassDef, HandleRef, IntValue, Listener, LogicalArray,
        ObjectInstance, StringArray, StructValue, Tensor,
    };
    use runmat_gc_api::GcPtr;
    use std::collections::HashMap;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isa_reports_expected_results_for_doubles() {
        let double_result = isa_builtin(Value::Num(42.0), Value::from("double")).expect("isa");
        assert_eq!(double_result, Value::Bool(true));

        let numeric_result = isa_builtin(Value::Num(42.0), Value::from("numeric")).expect("isa");
        assert_eq!(numeric_result, Value::Bool(true));

        let integer_result = isa_builtin(Value::Num(42.0), Value::from("integer")).expect("isa");
        assert_eq!(integer_result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isa_integer_category_matches_int_values() {
        let value = Value::Int(IntValue::I16(12));
        let int_result = isa_builtin(value.clone(), Value::from("integer")).expect("isa");
        assert_eq!(int_result, Value::Bool(true));

        let float_result = isa_builtin(value, Value::from("float")).expect("isa");
        assert_eq!(float_result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isa_handles_logical_and_char_types() {
        let logical = Value::LogicalArray(LogicalArray::new(vec![1], vec![1]).unwrap());
        assert_eq!(
            isa_builtin(logical, Value::from("logical")).expect("isa"),
            Value::Bool(true)
        );

        let char_array = Value::CharArray(CharArray::new_row("RunMat"));
        assert_eq!(
            isa_builtin(char_array, Value::from("char")).expect("isa"),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isa_string_and_struct_detection() {
        let string_scalar = Value::String("runmat".into());
        assert_eq!(
            isa_builtin(string_scalar, Value::from("string")).expect("isa"),
            Value::Bool(true)
        );

        let mut st = StructValue::new();
        st.fields.insert("field".into(), Value::Num(1.0));
        assert_eq!(
            isa_builtin(Value::Struct(st), Value::from("struct")).expect("isa"),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isa_cell_and_function_handle() {
        let cell = Value::Cell(CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap());
        assert_eq!(
            isa_builtin(cell, Value::from("cell")).expect("isa"),
            Value::Bool(true)
        );

        let func = Value::FunctionHandle("sin".into());
        assert_eq!(
            isa_builtin(func, Value::from("function_handle")).expect("isa"),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isa_gpu_arrays_treat_metadata_correctly() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu_value = Value::GpuTensor(handle);

            let numeric = isa_builtin(gpu_value.clone(), Value::from("numeric")).expect("isa");
            assert_eq!(numeric, Value::Bool(true));

            let double = isa_builtin(gpu_value, Value::from("double")).expect("isa");
            assert_eq!(double, Value::Bool(false));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isa_gpu_logical_handles_match_categories() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 0.0, 1.0, 0.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let logical_value = gpu_helpers::logical_gpu_value(handle.clone());

            let logical = isa_builtin(logical_value.clone(), Value::from("logical")).expect("isa");
            assert_eq!(logical, Value::Bool(true));

            let numeric =
                isa_builtin(logical_value, Value::from("numeric")).expect("isa numeric false");
            assert_eq!(numeric, Value::Bool(false));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isa_handle_aliases_and_inheritance() {
        let handle = HandleRef {
            class_name: "TestHandle".into(),
            target: GcPtr::null(),
            valid: true,
        };
        assert_eq!(
            isa_builtin(Value::HandleObject(handle), Value::from("handle")).expect("isa"),
            Value::Bool(true)
        );

        // Register a class that derives from handle and ensure inheritance is respected.
        let class_name = "pkg.TestHandle";
        let def = ClassDef {
            name: class_name.into(),
            parent: Some("handle".into()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        };
        runmat_builtins::register_class(def);
        let obj = Value::Object(ObjectInstance::new(class_name.into()));
        let handle_result = isa_builtin(obj.clone(), Value::from("handle")).expect("isa");
        assert_eq!(handle_result, Value::Bool(true));
        let exact = isa_builtin(obj, Value::from(class_name)).expect("isa");
        assert_eq!(exact, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isa_listener_alias_matches() {
        let listener = Listener {
            id: 1,
            target: GcPtr::null(),
            event_name: "Changed".into(),
            callback: GcPtr::null(),
            enabled: true,
            valid: true,
        };
        let value = Value::Listener(listener);
        assert_eq!(
            isa_builtin(value.clone(), Value::from("listener")).expect("isa"),
            Value::Bool(true)
        );
        assert_eq!(
            isa_builtin(value.clone(), Value::from("event.listener")).expect("isa"),
            Value::Bool(true)
        );
        assert_eq!(
            isa_builtin(value, Value::from("handle")).expect("isa"),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isa_meta_class_detection() {
        let meta = Value::ClassRef("Point".into());
        assert_eq!(
            isa_builtin(meta, Value::from("meta.class")).expect("isa"),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isa_errors_on_invalid_type_designator() {
        let type_array = Value::StringArray(
            StringArray::new(vec!["double".into(), "single".into()], vec![1, 2]).unwrap(),
        );
        let err = isa_builtin(Value::Num(1.0), type_array).unwrap_err();
        assert_eq!(err, "isa: TYPE must be a string scalar or character vector");
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
    fn isa_gpuarray_with_wgpu_provider_matches_numeric_category() {
        use runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider;
        use runmat_accelerate_api::AccelProvider;

        let provider = match ensure_wgpu_provider() {
            Ok(Some(p)) => p,
            _ => return,
        };

        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("wgpu upload");
        let value = Value::GpuTensor(handle);

        let numeric = isa_builtin(value.clone(), Value::from("numeric")).expect("isa numeric");
        assert_eq!(numeric, Value::Bool(true));

        let dbl = isa_builtin(value, Value::from("double")).expect("isa double");
        assert_eq!(dbl, Value::Bool(false));
    }
}
