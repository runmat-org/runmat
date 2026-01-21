//! MATLAB-compatible `fieldnames` builtin.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use runmat_builtins::{
    CellArray, CharArray, HandleRef, Listener, ObjectInstance, StructValue, Value,
};
use runmat_macros::runtime_builtin;
use std::collections::BTreeSet;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "fieldnames",
        builtin_path = "crate::builtins::structs::core::fieldnames"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "fieldnames"
category: "structs/core"
keywords: ["fieldnames", "struct", "introspection", "struct array", "object", "handle", "properties"]
summary: "List the field names of structs, struct arrays, or objects."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs entirely on the host. GPU-resident values or handles remain on the device; no kernels are dispatched."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::structs::core::fieldnames::tests"
  integration: "builtins::structs::core::fieldnames::tests::fieldnames_struct_array_collects_union"
---

# What does the `fieldnames` function do in MATLAB / RunMat?
`fieldnames(S)` returns a column cell array of character vectors containing the field names defined
on `S`. RunMat mirrors MATLAB by supporting scalar structs, struct arrays, and MATLAB-style objects
(including handle objects). The output is always deterministic and case-sensitive.

## How does the `fieldnames` function behave in MATLAB / RunMat?
- Works with scalar structs as well as struct arrays created by `struct`, `load`, or other builtins.
- Returns a column cell array (`n × 1`) whose elements are character vectors (1 × N char arrays).
- Objects (value or handle) report the union of their public, non-static class properties together
  with any dynamic properties stored on the instance.
- For struct arrays, the union of field names across all elements is reported.
- Field names are sorted alphabetically and remain case-sensitive (`name` and `Name` are distinct).
- Empty structs yield an empty `0 × 1` cell array. GPU-resident field values stay on the device—
  `fieldnames` only inspects metadata.

## `fieldnames` Function GPU Execution Behaviour
`fieldnames` is metadata-only. It does not launch kernels or gather data. GPU tensors or handles that
live inside structs or objects remain device-resident, and the builtin simply walks the host-side
descriptors that remember their names.

## GPU residency in RunMat (Do I need `gpuArray`?)
You do not need to call `gpuArray` when using `fieldnames`. RunMat leaves each value—CPU or GPU—where
it already resides. When structs or objects contain GPU handles, those handles remain valid and stay
resident on the device throughout the introspection call.

## Examples of using the `fieldnames` function in MATLAB / RunMat

### Listing the fields of a scalar structure
```matlab
s = struct("name", "Ada", "score", 42);
fields = fieldnames(s);
```

Expected output:
```matlab
fields =
    {'name'}
    {'score'}
```

### Inspecting field names before accessing values
```matlab
stats = struct("min", 1.2, "max", 9.8);
if any(strcmp(fieldnames(stats), "median"))
    disp(stats.median);
else
    disp("median not available");
end
```

Expected output:
```matlab
median not available
```

### Gathering field names from a struct array
```matlab
people = struct("name", {"Ada", "Grace"}, "id", {101, 102});
fields = fieldnames(people);
```

Expected output:
```matlab
fields =
    {'id'}
    {'name'}
```

### Listing the properties of an object
```matlab
% Save this class in Counter.m before running the example:
% classdef Counter
%     properties
%         Value = 0
%         Step  = 1
%     end
% end
c = Counter;
props = fieldnames(c);
```

Expected output:
```matlab
props =
    {'Step'}
    {'Value'}
```

### Discovering fields inside nested structs
```matlab
config.database = struct("host", "db.local", "port", 5432);
names = fieldnames(config.database);
```

Expected output:
```matlab
names =
    {'host'}
    {'port'}
```

### Handling empty structs safely
```matlab
emptyScalar = struct();
emptyArray = struct("id", {});
fs = fieldnames(emptyScalar);
fa = fieldnames(emptyArray);
```

Expected output:
```matlab
fs =
  0x1 empty cell array

fa =
  0x1 empty cell array
```

## FAQ

### What does `fieldnames` return?
It returns a column cell array of character vectors for every field or property defined on the input
struct, struct array, or object instance.

### Are the field names sorted?
Yes. RunMat stores fields and properties in hash maps, so `fieldnames` sorts them alphabetically to
produce a stable, deterministic result.

### Does `fieldnames` gather GPU data back to the CPU?
No. The builtin only inspects host-side metadata; GPU-resident tensors or handles remain untouched.

### How do struct arrays affect the result?
All struct elements are examined and the union of their field names is returned. Duplicate names are
collapsed automatically.

### Does `fieldnames` work with objects and handle objects?
Yes. The builtin merges public, non-static class properties with any dynamic properties stored on the
instance. Handle objects reuse the same logic even when their payloads are shared across references.

### What happens with empty structs or empty struct arrays?
The result is an empty column cell array (`0 × 1`). This matches MATLAB behaviour for empty scalars
and for struct arrays that contain no elements.

### Can I use `fieldnames` on unsupported inputs?
No. Passing anything other than a struct, struct array, or object raises
`fieldnames: expected struct, struct array, or object`.

## See Also
[struct](./struct), [isfield](./isfield), [getfield](./getfield), [setfield](./setfield), [class](./class)

## Source & Feedback
- Implementation: `crates/runmat-runtime/src/builtins/structs/core/fieldnames.rs`
- Found a bug or behaviour mismatch? Open an issue at `https://github.com/runmat-org/runmat/issues/new/choose`.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::structs::core::fieldnames")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fieldnames",
    op_kind: GpuOpKind::Custom("fieldnames"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host-only introspection; providers do not participate.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::structs::core::fieldnames")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fieldnames",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner treats fieldnames as a host inspector; it terminates any pending fusion group.",
};

fn fieldnames_flow(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("fieldnames")
        .build()
}

#[runtime_builtin(
    name = "fieldnames",
    category = "structs/core",
    summary = "List the field names of scalar structs or struct arrays.",
    keywords = "fieldnames,struct,introspection,fields",
    builtin_path = "crate::builtins::structs::core::fieldnames"
)]
async fn fieldnames_builtin(value: Value) -> BuiltinResult<Value> {
    let names = match &value {
        Value::Struct(st) => collect_struct_fieldnames(st),
        Value::Cell(cell) => collect_struct_array_fieldnames(cell)?,
        Value::Object(obj) => collect_object_fieldnames(obj),
        Value::HandleObject(handle) => collect_handle_fieldnames(handle)?,
        Value::Listener(listener) => collect_listener_fieldnames(listener),
        other => {
            return Err(fieldnames_flow(format!(
                "fieldnames: expected struct, struct array, or object (got {other:?})"
            )))
        }
    };

    let rows = names.len();
    let cells: Vec<Value> = names
        .into_iter()
        .map(|name| Value::CharArray(CharArray::new_row(&name)))
        .collect();
    crate::make_cell(cells, rows, 1).map_err(|e| fieldnames_flow(format!("fieldnames: {e}")))
}

fn collect_struct_fieldnames(st: &StructValue) -> Vec<String> {
    let mut names: Vec<String> = st.fields.keys().cloned().collect();
    names.sort();
    names
}

fn collect_struct_array_fieldnames(array: &CellArray) -> BuiltinResult<Vec<String>> {
    let mut names = BTreeSet::new();
    for handle in array.data.iter() {
        let value = unsafe { &*handle.as_raw() };
        let Value::Struct(st) = value else {
            return Err(fieldnames_flow(
                "fieldnames: expected struct array contents to be structs",
            ));
        };
        names.extend(st.fields.keys().cloned());
    }
    Ok(names.into_iter().collect())
}

fn collect_object_fieldnames(obj: &ObjectInstance) -> Vec<String> {
    let mut names = class_instance_property_names(&obj.class_name);
    names.extend(obj.properties.keys().cloned());
    names.into_iter().collect()
}

fn collect_handle_fieldnames(handle: &HandleRef) -> BuiltinResult<Vec<String>> {
    let mut names = class_instance_property_names(&handle.class_name);

    if handle.valid {
        let target = unsafe { &*handle.target.as_raw() };
        match target {
            Value::Struct(st) => {
                names.extend(collect_struct_fieldnames(st));
            }
            Value::Cell(array) => {
                names.extend(collect_struct_array_fieldnames(array)?);
            }
            Value::Object(obj) => {
                names.extend(collect_object_fieldnames(obj));
            }
            Value::Listener(listener) => {
                names.extend(collect_listener_fieldnames(listener));
            }
            Value::HandleObject(other) => {
                names.extend(class_instance_property_names(&other.class_name));
            }
            _ => {}
        }
    }

    Ok(names.into_iter().collect())
}

fn collect_listener_fieldnames(_listener: &Listener) -> Vec<String> {
    let mut names = vec![
        "callback".to_string(),
        "enabled".to_string(),
        "event_name".to_string(),
        "id".to_string(),
        "target".to_string(),
        "valid".to_string(),
    ];
    names.sort();
    names
}

fn class_instance_property_names(class_name: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    if let Some(class_def) = runmat_builtins::get_class(class_name) {
        for (name, prop) in &class_def.properties {
            if !prop.is_static {
                names.insert(name.clone());
            }
        }
    }
    names
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_builtins::{
        Access, CellArray, ClassDef, HandleRef, ObjectInstance, PropertyDef, StructValue, Value,
    };
    use std::collections::HashMap;

    use crate::builtins::common::test_support;

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    fn run_fieldnames(value: Value) -> BuiltinResult<Value> {
        futures::executor::block_on(fieldnames_builtin(value))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fieldnames_returns_sorted_names_for_scalar_struct() {
        let mut fields = StructValue::new();
        fields.fields.insert("beta".to_string(), Value::Num(1.0));
        fields.fields.insert("alpha".to_string(), Value::Num(2.0));
        let result = run_fieldnames(Value::Struct(fields)).expect("fieldnames");
        let Value::Cell(cell) = result else {
            panic!("expected cell array result");
        };
        assert_eq!(cell.cols, 1);
        assert_eq!(cell.rows, 2);
        let collected = cell_strings(&cell);
        assert_eq!(collected, vec!["alpha".to_string(), "beta".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fieldnames_struct_array_collects_union() {
        let mut first = StructValue::new();
        first
            .fields
            .insert("name".to_string(), Value::from("Ada".to_string()));
        first.fields.insert("id".to_string(), Value::Num(101.0));

        let mut second = StructValue::new();
        second
            .fields
            .insert("name".to_string(), Value::from("Grace".to_string()));
        second
            .fields
            .insert("department".to_string(), Value::from("Research"));

        let cell = CellArray::new_with_shape(
            vec![Value::Struct(first), Value::Struct(second)],
            vec![1, 2],
        )
        .expect("struct array");

        let result = run_fieldnames(Value::Cell(cell)).expect("fieldnames");
        let Value::Cell(names) = result else {
            panic!("expected cell array result");
        };
        assert_eq!(names.cols, 1);
        assert_eq!(names.rows, 3);
        let collected = cell_strings(&names);
        assert_eq!(
            collected,
            vec![
                "department".to_string(),
                "id".to_string(),
                "name".to_string()
            ]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fieldnames_errors_for_non_struct_inputs() {
        let err = error_message(run_fieldnames(Value::Num(1.0)).unwrap_err());
        assert!(
            err.contains("expected struct, struct array, or object"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fieldnames_handles_empty_struct_array() {
        let empty_array = CellArray::new(Vec::new(), 0, 0).expect("empty struct array backing");
        let result = run_fieldnames(Value::Cell(empty_array)).expect("fieldnames");
        let Value::Cell(cell) = result else {
            panic!("expected cell array");
        };
        assert_eq!(cell.rows, 0);
        assert_eq!(cell.cols, 1);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fieldnames_cell_without_struct_errors() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).expect("cell");
        let err = error_message(run_fieldnames(Value::Cell(cell)).unwrap_err());
        assert!(
            err.contains("expected struct array contents to be structs"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fieldnames_preserves_case_distinctions() {
        let mut fields = StructValue::new();
        fields.fields.insert("name".to_string(), Value::Num(1.0));
        fields.fields.insert("Name".to_string(), Value::Num(2.0));
        let Value::Cell(cell) = run_fieldnames(Value::Struct(fields)).expect("fieldnames") else {
            panic!("expected cell array result");
        };
        let collected = cell_strings(&cell);
        assert_eq!(collected, vec!["Name".to_string(), "name".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fieldnames_object_includes_class_and_dynamic_properties() {
        let class_name = "runmat.unittest.FieldnamesObject";
        let mut def = ClassDef {
            name: class_name.to_string(),
            parent: None,
            properties: HashMap::new(),
            methods: HashMap::new(),
        };
        def.properties.insert(
            "Value".to_string(),
            PropertyDef {
                name: "Value".to_string(),
                is_static: false,
                is_dependent: false,
                get_access: Access::Public,
                set_access: Access::Public,
                default_value: None,
            },
        );
        def.properties.insert(
            "Version".to_string(),
            PropertyDef {
                name: "Version".to_string(),
                is_static: true,
                is_dependent: false,
                get_access: Access::Public,
                set_access: Access::Public,
                default_value: None,
            },
        );
        runmat_builtins::register_class(def);

        let mut obj = ObjectInstance::new(class_name.to_string());
        obj.properties.insert("Step".to_string(), Value::Num(2.0));

        let Value::Cell(cell) = run_fieldnames(Value::Object(obj)).expect("fieldnames object")
        else {
            panic!("expected cell array");
        };
        let collected = cell_strings(&cell);
        assert_eq!(collected, vec!["Step".to_string(), "Value".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fieldnames_handle_object_merges_class_and_target() {
        let class_name = "runmat.unittest.FieldnamesHandle";
        let mut def = ClassDef {
            name: class_name.to_string(),
            parent: None,
            properties: HashMap::new(),
            methods: HashMap::new(),
        };
        def.properties.insert(
            "Enabled".to_string(),
            PropertyDef {
                name: "Enabled".to_string(),
                is_static: false,
                is_dependent: false,
                get_access: Access::Public,
                set_access: Access::Public,
                default_value: None,
            },
        );
        runmat_builtins::register_class(def);

        let mut payload = StructValue::new();
        payload
            .fields
            .insert("Status".to_string(), Value::from("ready"));
        let target = unsafe {
            runmat_gc_api::GcPtr::from_raw(Box::into_raw(Box::new(Value::Struct(payload))))
        };

        let handle = HandleRef {
            class_name: class_name.to_string(),
            target,
            valid: true,
        };

        let Value::Cell(cell) =
            run_fieldnames(Value::HandleObject(handle)).expect("fieldnames handle")
        else {
            panic!("expected cell array");
        };
        let collected = cell_strings(&cell);
        assert_eq!(collected, vec!["Enabled".to_string(), "Status".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    fn cell_strings(cell: &CellArray) -> Vec<String> {
        cell.data
            .iter()
            .map(|ptr| match unsafe { &*ptr.as_raw() } {
                Value::CharArray(ca) => ca.data.iter().collect(),
                other => panic!("expected character array cell element, got {other:?}"),
            })
            .collect()
    }
}
