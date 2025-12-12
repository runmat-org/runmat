//! MATLAB-compatible `struct` builtin.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use runmat_builtins::{CellArray, CharArray, StructValue, Value};
use runmat_macros::runtime_builtin;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "struct",
        builtin_path = "crate::builtins::structs::core::r#struct"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "struct"
category: "structs/core"
keywords: ["struct", "structure", "name-value", "record", "struct array"]
summary: "Create scalar structs or struct arrays from name/value pairs."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Struct construction runs on the host. GPU tensors stay as handles inside the resulting struct or struct array."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::structs::core::r#struct::tests"
  integration: "builtins::structs::core::r#struct::tests::struct_preserves_gpu_handles_with_registered_provider"
---

# What does the `struct` function do in MATLAB / RunMat?
`S = struct(...)` creates scalar structs or struct arrays by pairing field names with values. The
inputs can be simple name/value pairs, existing structs, or cell arrays whose elements are expanded
into struct array entries.

## How does the `struct` function behave in MATLAB / RunMat?
- Field names must satisfy the MATLAB `isvarname` rules: they start with a letter or underscore and
  contain only letters, digits, or underscores.
- The last occurrence of a repeated field name wins and overwrites earlier values.
- String scalars, character vectors, and single-element string arrays are accepted as field names.
- `struct()` returns a scalar struct with no fields, while `struct([])` yields a `0×0` struct array.
- When any value input is a cell array, every cell array input must share the same size. Non-cell
  inputs are replicated across every element of the resulting struct array.
- Passing an existing struct or struct array (`struct(S)`) creates a deep copy; the original data is
  untouched.

## `struct` Function GPU Execution Behaviour
`struct` performs all bookkeeping on the host. GPU-resident values—such as tensors created with
`gpuArray`—are stored as-is inside the resulting struct or struct array. No kernels are launched and
no data is implicitly gathered back to the CPU.

## GPU residency in RunMat (Do I need `gpuArray`?)
Usually not. RunMat's planner keeps GPU values resident as long as downstream operations can profit
from them. You can still seed GPU residency explicitly with `gpuArray` for MATLAB compatibility; the
handles remain untouched inside the struct until another builtin decides to gather or operate on
them.

## Examples

### Creating a simple structure for named fields
```matlab
s = struct("name", "Ada", "score", 42);
disp(s.name);
disp(s.score);
```

Expected output:
```matlab
Ada
    42
```

### Building a struct array from paired cell inputs
```matlab
names = {"Ada", "Grace"};
ages = {36, 45};
people = struct("name", names, "age", ages);
{people.name}
```

Expected output:
```matlab
    {'Ada'}    {'Grace'}
```

### Broadcasting scalars across a struct array
```matlab
ids = struct("id", {101, 102, 103}, "department", "Research");
{ids.department}
```

Expected output:
```matlab
    {'Research'}    {'Research'}    {'Research'}
```

### Copying an existing structure
```matlab
a = struct("id", 7, "label", "demo");
b = struct(a);
b.id = 8;
disp([a.id b.id]);
```

Expected output:
```matlab
     7     8
```

### Building an empty struct array
```matlab
s = struct([]);
disp(size(s));
```

Expected output:
```matlab
     0     0
```

## FAQ

### Do field names have to be valid identifiers?
Yes. RunMat mirrors MATLAB and requires names to satisfy `isvarname`. Names must begin with a letter
or underscore and may contain letters, digits, and underscores.

### How do I create a struct array?
Provide one or more value arguments as cell arrays with identical sizes. Each cell contributes the
value for the corresponding struct element. Non-cell values are replicated across all elements.

### What happens when the same field name appears more than once?
The last value wins; earlier values for the same field are overwritten.

### Does `struct` gather GPU data back to the CPU?
No. GPU tensors remain device-resident handles inside the resulting struct or struct array.

### Can I pass non-string objects as field names?
No. Field names must be provided as string scalars, character vectors, or single-element string
arrays. Passing other types raises an error.

## See Also
[load](../../io/mat/load), [whos](../../introspection/whos), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::structs::core::r#struct")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "struct",
    op_kind: GpuOpKind::Custom("struct"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host-only construction; GPU values are preserved as handles without gathering.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::structs::core::r#struct")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "struct",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Struct creation breaks fusion planning but retains GPU residency for field values.",
};

struct FieldEntry {
    name: String,
    value: FieldValue,
}

enum FieldValue {
    Single(Value),
    Cell(CellArray),
}

#[runtime_builtin(
    name = "struct",
    category = "structs/core",
    summary = "Create scalar structs or struct arrays from name/value pairs.",
    keywords = "struct,structure,name-value,record",
    builtin_path = "crate::builtins::structs::core::r#struct"
)]
fn struct_builtin(rest: Vec<Value>) -> Result<Value, String> {
    match rest.len() {
        0 => Ok(Value::Struct(StructValue::new())),
        1 => match rest.into_iter().next().unwrap() {
            Value::Struct(existing) => Ok(Value::Struct(existing.clone())),
            Value::Cell(cell) => clone_struct_array(&cell),
            Value::Tensor(tensor) if tensor.data.is_empty() => empty_struct_array(),
            Value::LogicalArray(logical) if logical.data.is_empty() => empty_struct_array(),
            other => Err(format!(
                "struct: expected name/value pairs, an existing struct or struct array, or [] to create an empty struct array (got {other:?})"
            )),
        },
        len if len % 2 == 0 => build_from_pairs(rest),
        _ => Err("struct: expected name/value pairs".to_string()),
    }
}

fn build_from_pairs(args: Vec<Value>) -> Result<Value, String> {
    let mut entries: Vec<FieldEntry> = Vec::new();
    let mut target_shape: Option<Vec<usize>> = None;

    let mut iter = args.into_iter();
    while let (Some(name_value), Some(field_value)) = (iter.next(), iter.next()) {
        let field_name = parse_field_name(&name_value)?;
        match field_value {
            Value::Cell(cell) => {
                let shape = cell.shape.clone();
                if let Some(existing) = &target_shape {
                    if *existing != shape {
                        return Err("struct: cell inputs must have matching sizes".to_string());
                    }
                } else {
                    target_shape = Some(shape);
                }
                entries.push(FieldEntry {
                    name: field_name,
                    value: FieldValue::Cell(cell),
                });
            }
            other => entries.push(FieldEntry {
                name: field_name,
                value: FieldValue::Single(other),
            }),
        }
    }

    if let Some(shape) = target_shape {
        build_struct_array(entries, shape)
    } else {
        build_scalar_struct(entries)
    }
}

fn build_scalar_struct(entries: Vec<FieldEntry>) -> Result<Value, String> {
    let mut fields = StructValue::new();
    for entry in entries {
        match entry.value {
            FieldValue::Single(value) => {
                fields.fields.insert(entry.name, value);
            }
            FieldValue::Cell(cell) => {
                let shape = cell.shape.clone();
                return build_struct_array(
                    vec![FieldEntry {
                        name: entry.name,
                        value: FieldValue::Cell(cell),
                    }],
                    shape,
                );
            }
        }
    }
    Ok(Value::Struct(fields))
}

fn build_struct_array(entries: Vec<FieldEntry>, shape: Vec<usize>) -> Result<Value, String> {
    let total_len = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| "struct: struct array size exceeds platform limits".to_string())?;

    for entry in &entries {
        if let FieldValue::Cell(cell) = &entry.value {
            if cell.data.len() != total_len {
                return Err("struct: cell inputs must have matching sizes".to_string());
            }
        }
    }

    let mut structs: Vec<Value> = Vec::with_capacity(total_len);
    for idx in 0..total_len {
        let mut fields = StructValue::new();
        for entry in &entries {
            let value = match &entry.value {
                FieldValue::Single(val) => val.clone(),
                FieldValue::Cell(cell) => clone_cell_element(cell, idx)?,
            };
            fields.fields.insert(entry.name.clone(), value);
        }
        structs.push(Value::Struct(fields));
    }

    CellArray::new_with_shape(structs, shape)
        .map(Value::Cell)
        .map_err(|e| format!("struct: failed to assemble struct array: {e}"))
}

fn clone_cell_element(cell: &CellArray, index: usize) -> Result<Value, String> {
    cell.data
        .get(index)
        .map(|ptr| unsafe { &*ptr.as_raw() }.clone())
        .ok_or_else(|| "struct: cell inputs must have matching sizes".to_string())
}

fn empty_struct_array() -> Result<Value, String> {
    CellArray::new(Vec::new(), 0, 0)
        .map(Value::Cell)
        .map_err(|e| format!("struct: failed to create empty struct array: {e}"))
}

fn clone_struct_array(array: &CellArray) -> Result<Value, String> {
    let mut values: Vec<Value> = Vec::with_capacity(array.data.len());
    for (index, handle) in array.data.iter().enumerate() {
        let value = unsafe { &*handle.as_raw() }.clone();
        if !matches!(value, Value::Struct(_)) {
            return Err(format!(
                "struct: single argument cell input must contain structs (element {} is not a struct)",
                index + 1
            ));
        }
        values.push(value);
    }
    CellArray::new_with_shape(values, array.shape.clone())
        .map(Value::Cell)
        .map_err(|e| format!("struct: failed to copy struct array: {e}"))
}

fn parse_field_name(value: &Value) -> Result<String, String> {
    let text = match value {
        Value::String(s) => s.clone(),
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                sa.data[0].clone()
            } else {
                return Err(
                    "struct: field names must be scalar string arrays or character vectors"
                        .to_string(),
                );
            }
        }
        Value::CharArray(ca) => char_array_to_string(ca)?,
        _ => return Err("struct: field names must be strings or character vectors".to_string()),
    };

    validate_field_name(&text)?;
    Ok(text)
}

fn char_array_to_string(ca: &CharArray) -> Result<String, String> {
    if ca.rows > 1 {
        return Err("struct: field names must be 1-by-N character vectors".to_string());
    }
    let mut out = String::with_capacity(ca.data.len());
    for ch in &ca.data {
        out.push(*ch);
    }
    Ok(out)
}

fn validate_field_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("struct: field names must be nonempty".to_string());
    }
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return Err("struct: field names must be nonempty".to_string());
    };
    if !is_first_char_valid(first) {
        return Err(format!(
            "struct: field names must begin with a letter or underscore (got '{name}')"
        ));
    }
    if let Some(bad) = chars.find(|c| !is_subsequent_char_valid(*c)) {
        return Err(format!(
            "struct: invalid character '{bad}' in field name '{name}'"
        ));
    }
    Ok(())
}

fn is_first_char_valid(c: char) -> bool {
    c == '_' || c.is_ascii_alphabetic()
}

fn is_subsequent_char_valid(c: char) -> bool {
    c == '_' || c.is_ascii_alphanumeric()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_accelerate_api::GpuTensorHandle;
    use runmat_builtins::{CellArray, IntValue, StringArray, StructValue, Tensor};

    use crate::builtins::common::test_support;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::HostTensorView;

    #[test]
    fn struct_empty() {
        let Value::Struct(s) = struct_builtin(Vec::new()).expect("struct") else {
            panic!("expected struct value");
        };
        assert!(s.fields.is_empty());
    }

    #[test]
    fn struct_empty_from_empty_matrix() {
        let tensor = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let value = struct_builtin(vec![Value::Tensor(tensor)]).expect("struct([])");
        match value {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 0);
                assert_eq!(cell.cols, 0);
                assert!(cell.data.is_empty());
            }
            other => panic!("expected empty struct array, got {other:?}"),
        }
    }

    #[test]
    fn struct_name_value_pairs() {
        let args = vec![
            Value::from("name"),
            Value::from("Ada"),
            Value::from("score"),
            Value::Int(IntValue::I32(42)),
        ];
        let Value::Struct(s) = struct_builtin(args).expect("struct") else {
            panic!("expected struct value");
        };
        assert_eq!(s.fields.len(), 2);
        assert!(matches!(s.fields.get("name"), Some(Value::String(v)) if v == "Ada"));
        assert!(matches!(
            s.fields.get("score"),
            Some(Value::Int(IntValue::I32(42)))
        ));
    }

    #[test]
    fn struct_struct_array_from_cells() {
        let names = CellArray::new(vec![Value::from("Ada"), Value::from("Grace")], 1, 2).unwrap();
        let ages = CellArray::new(
            vec![Value::Int(IntValue::I32(36)), Value::Int(IntValue::I32(45))],
            1,
            2,
        )
        .unwrap();
        let result = struct_builtin(vec![
            Value::from("name"),
            Value::Cell(names),
            Value::from("age"),
            Value::Cell(ages),
        ])
        .expect("struct array");
        let structs = expect_struct_array(result);
        assert_eq!(structs.len(), 2);
        assert!(matches!(
            structs[0].fields.get("name"),
            Some(Value::String(v)) if v == "Ada"
        ));
        assert!(matches!(
            structs[1].fields.get("age"),
            Some(Value::Int(IntValue::I32(45)))
        ));
    }

    #[test]
    fn struct_struct_array_replicates_scalars() {
        let names = CellArray::new(vec![Value::from("Ada"), Value::from("Grace")], 1, 2).unwrap();
        let result = struct_builtin(vec![
            Value::from("name"),
            Value::Cell(names),
            Value::from("department"),
            Value::from("Research"),
        ])
        .expect("struct array");
        let structs = expect_struct_array(result);
        assert_eq!(structs.len(), 2);
        for entry in structs {
            assert!(matches!(
                entry.fields.get("department"),
                Some(Value::String(v)) if v == "Research"
            ));
        }
    }

    #[test]
    fn struct_struct_array_cell_size_mismatch_errors() {
        let names = CellArray::new(vec![Value::from("Ada"), Value::from("Grace")], 1, 2).unwrap();
        let scores = CellArray::new(vec![Value::Int(IntValue::I32(1))], 1, 1).unwrap();
        let err = struct_builtin(vec![
            Value::from("name"),
            Value::Cell(names),
            Value::from("score"),
            Value::Cell(scores),
        ])
        .unwrap_err();
        assert!(err.contains("matching sizes"));
    }

    #[test]
    fn struct_overwrites_duplicates() {
        let args = vec![
            Value::from("version"),
            Value::Int(IntValue::I32(1)),
            Value::from("version"),
            Value::Int(IntValue::I32(2)),
        ];
        let Value::Struct(s) = struct_builtin(args).expect("struct") else {
            panic!("expected struct value");
        };
        assert_eq!(s.fields.len(), 1);
        assert!(matches!(
            s.fields.get("version"),
            Some(Value::Int(IntValue::I32(2)))
        ));
    }

    #[test]
    fn struct_rejects_odd_arguments() {
        let err = struct_builtin(vec![Value::from("name")]).unwrap_err();
        assert!(err.contains("name/value pairs"));
    }

    #[test]
    fn struct_rejects_invalid_field_name() {
        let err =
            struct_builtin(vec![Value::from("1bad"), Value::Int(IntValue::I32(1))]).unwrap_err();
        assert!(err.contains("begin with a letter or underscore"));
    }

    #[test]
    fn struct_rejects_non_text_field_name() {
        let err = struct_builtin(vec![Value::Num(1.0), Value::Int(IntValue::I32(1))]).unwrap_err();
        assert!(err.contains("strings or character vectors"));
    }

    #[test]
    fn struct_accepts_char_vector_name() {
        let chars = CharArray::new("field".chars().collect(), 1, 5).unwrap();
        let args = vec![Value::CharArray(chars), Value::Num(1.0)];
        let Value::Struct(s) = struct_builtin(args).expect("struct") else {
            panic!("expected struct value");
        };
        assert!(s.fields.contains_key("field"));
    }

    #[test]
    fn struct_accepts_string_scalar_name() {
        let sa = StringArray::new(vec!["field".to_string()], vec![1]).unwrap();
        let args = vec![Value::StringArray(sa), Value::Num(1.0)];
        let Value::Struct(s) = struct_builtin(args).expect("struct") else {
            panic!("expected struct value");
        };
        assert!(s.fields.contains_key("field"));
    }

    #[test]
    fn struct_allows_existing_struct_copy() {
        let mut base = StructValue::new();
        base.fields
            .insert("id".to_string(), Value::Int(IntValue::I32(7)));
        let copy = struct_builtin(vec![Value::Struct(base.clone())]).expect("struct");
        assert_eq!(copy, Value::Struct(base));
    }

    #[test]
    fn struct_copies_struct_array_argument() {
        let mut proto = StructValue::new();
        proto
            .fields
            .insert("id".into(), Value::Int(IntValue::I32(7)));
        let struct_array = CellArray::new(
            vec![
                Value::Struct(proto.clone()),
                Value::Struct(proto.clone()),
                Value::Struct(proto.clone()),
            ],
            1,
            3,
        )
        .unwrap();
        let original = struct_array.clone();
        let result = struct_builtin(vec![Value::Cell(struct_array)]).expect("struct array clone");
        let cloned = expect_struct_array(result);
        let baseline = expect_struct_array(Value::Cell(original));
        assert_eq!(cloned, baseline);
    }

    #[test]
    fn struct_rejects_cell_argument_without_structs() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let err = struct_builtin(vec![Value::Cell(cell)]).unwrap_err();
        assert!(err.contains("must contain structs"));
    }

    #[test]
    fn struct_preserves_gpu_tensor_handles() {
        let handle = GpuTensorHandle {
            shape: vec![2, 2],
            device_id: 1,
            buffer_id: 99,
        };
        let args = vec![Value::from("data"), Value::GpuTensor(handle.clone())];
        let Value::Struct(s) = struct_builtin(args).expect("struct") else {
            panic!("expected struct value");
        };
        assert!(matches!(s.fields.get("data"), Some(Value::GpuTensor(h)) if h == &handle));
    }

    #[test]
    fn struct_struct_array_preserves_gpu_handles() {
        let first = GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 2,
            buffer_id: 11,
        };
        let second = GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 2,
            buffer_id: 12,
        };
        let cell = CellArray::new(
            vec![
                Value::GpuTensor(first.clone()),
                Value::GpuTensor(second.clone()),
            ],
            1,
            2,
        )
        .unwrap();
        let result = struct_builtin(vec![Value::from("payload"), Value::Cell(cell)])
            .expect("struct array gpu handles");
        let structs = expect_struct_array(result);
        assert!(matches!(
            structs[0].fields.get("payload"),
            Some(Value::GpuTensor(h)) if h == &first
        ));
        assert!(matches!(
            structs[1].fields.get("payload"),
            Some(Value::GpuTensor(h)) if h == &second
        ));
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn struct_preserves_gpu_handles_with_registered_provider() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let host = HostTensorView {
            data: &[1.0, 2.0],
            shape: &[2, 1],
        };
        let handle = provider.upload(&host).expect("upload");
        let args = vec![Value::from("gpu"), Value::GpuTensor(handle.clone())];
        let Value::Struct(s) = struct_builtin(args).expect("struct") else {
            panic!("expected struct value");
        };
        assert!(matches!(s.fields.get("gpu"), Some(Value::GpuTensor(h)) if h == &handle));
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    fn expect_struct_array(value: Value) -> Vec<StructValue> {
        match value {
            Value::Cell(cell) => cell
                .data
                .iter()
                .map(|ptr| unsafe { &*ptr.as_raw() }.clone())
                .map(|value| match value {
                    Value::Struct(st) => st,
                    other => panic!("expected struct element, got {other:?}"),
                })
                .collect(),
            Value::Struct(st) => vec![st],
            other => panic!("expected struct or struct array, got {other:?}"),
        }
    }
}
