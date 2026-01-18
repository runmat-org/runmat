//! MATLAB-compatible `isfield` builtin that reports whether structs contain a field.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use runmat_builtins::{CellArray, LogicalArray, StructValue, Value};
use runmat_macros::runtime_builtin;
use std::collections::HashSet;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "isfield",
        builtin_path = "crate::builtins::structs::core::isfield"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "isfield"
category: "structs/core"
keywords: ["isfield", "struct", "struct array", "field existence", "metadata"]
summary: "Test whether a struct or struct array defines specific field names."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Metadata-only; executes entirely on the host. GPU tensors stored inside structs remain resident."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::structs::core::isfield::tests"
  integration: "runmat_ignition::functions::struct_isfield_multi_and_fieldnames"
---

# What does the `isfield` function do in MATLAB / RunMat?
`tf = isfield(S, name)` returns a logical value that indicates whether the struct `S`
defines the field `name`. When `name` is a string array or a cell array of names,
`isfield` returns a logical array with the same size, reporting the result for each
requested field.

## How does the `isfield` function behave in MATLAB / RunMat?
- Works with scalar structs and struct arrays created with `struct`, `load`, or similar builtins.
- Accepts character vectors, string scalars, string arrays, or cell arrays containing character vectors and/or string scalars for the `name` argument.
- Returns a scalar logical (`true`/`false`) when `name` is a single string or char vector.
- Returns a logical array that matches the size of `name` when it is a string array or cell array.
- If the first argument is not a struct or struct array, the result is `false` (or a logical array of `false` values) without raising an error.
- Empty struct arrays yield `false` for every queried name because they do not carry field metadata once the elements have been removed.

## `isfield` Function GPU Execution Behaviour
`isfield` performs metadata checks entirely on the host. It never gathers or copies GPU
tensors that may be stored inside the struct; it only inspects the host-side descriptors
that record field names. No acceleration provider hooks are required.

## Examples of using the `isfield` function in MATLAB / RunMat

### Checking whether a struct defines a single field
```matlab
s = struct("name", "Ada", "score", 42);
hasScore = isfield(s, "score");
```

Expected output:
```matlab
hasScore =
  logical
   1
```

### Testing multiple field names at once
```matlab
s = struct("name", "Ada", "score", 42);
names = {"name", "department"; "score", "email"};
mask = isfield(s, names);
```

Expected output:
```matlab
mask =
  2×2 logical array
     1     0
     1     0
```

### Inspecting a struct array that shares a common schema
```matlab
people = struct("name", {"Ada", "Grace"}, "id", {101, 102});
idxMask = isfield(people, ["id", "department"]);
```

Expected output:
```matlab
idxMask =
  1×2 logical array
     1     0
```

### Using `isfield` before accessing optional configuration fields
```matlab
cfg = struct("mode", "fast");
if ~isfield(cfg, "rate")
    cfg.rate = 60;
end
```

### Mixing string scalars and character vectors in a cell array
```matlab
opts = struct("Solver", "cg", "MaxIter", 200);
queries = {"Solver", "Tolerances"; "MaxIter", "History"};
present = isfield(opts, queries);
```

### Behaviour when the input is not a struct
```matlab
value = 42;
tf = isfield(value, "anything");
```

Expected output:
```matlab
tf =
  logical
   0
```

## GPU residency in RunMat (Do I need `gpuArray`?)
`isfield` is metadata-only. It neither moves nor inspects the contents of GPU tensors that
might live inside the struct. You do not need to call `gpuArray` or `gather` to use the
builtin; residency is preserved automatically.

## FAQ

### What argument types does `isfield` accept for field names?
Character vectors, string scalars, string arrays, and cell arrays that contain character
vectors or string scalars. Passing other types raises an error.

### Does `isfield` error when the first input is not a struct?
No. It returns `false` (or a logical array of `false`) to mirror MATLAB's behaviour. Use
`isstruct` if you need to guard against non-struct inputs before calling `isfield`.

### How do struct arrays affect the result?
Every element of the struct array must contain the queried field. If any element is
missing the field, the result is `false` for that name.

### What happens with empty struct arrays?
Empty struct arrays do not retain field metadata in RunMat yet, so `isfield` returns
`false` for all queries. This matches MATLAB when the struct has no defined fields.

### Are field name comparisons case-sensitive?
Yes. Field names are compared using exact, case-sensitive matches, just like MATLAB.

### Does `isfield` gather GPU data?
No. The builtin only inspects field metadata and leaves GPU-resident tensors untouched.

## See Also
[fieldnames](./fieldnames), [struct](./struct), [isprop](./isprop), [getfield](./getfield), [setfield](./setfield)
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::structs::core::isfield")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isfield",
    op_kind: GpuOpKind::Custom("isfield"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host-only metadata check; acceleration providers do not participate.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::structs::core::isfield")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isfield",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Acts as a fusion barrier because it inspects struct metadata on the host.",
};

fn isfield_flow(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("isfield")
        .build()
}

#[runtime_builtin(
    name = "isfield",
    category = "structs/core",
    summary = "Test whether a struct or struct array defines specific field names.",
    keywords = "isfield,struct,field existence",
    builtin_path = "crate::builtins::structs::core::isfield"
)]
fn isfield_builtin(target: Value, names: Value) -> BuiltinResult<Value> {
    let context = classify_struct(&target)?;
    let parsed = parse_field_names(names)?;
    match context {
        StructContext::Struct(struct_value) => evaluate_scalar(struct_value, parsed),
        StructContext::StructArray(cell) => evaluate_struct_array(cell, parsed),
        StructContext::NonStruct => evaluate_non_struct(parsed),
    }
}

#[derive(Clone, Copy)]
enum StructContext<'a> {
    Struct(&'a StructValue),
    StructArray(&'a CellArray),
    NonStruct,
}

fn classify_struct<'a>(value: &'a Value) -> BuiltinResult<StructContext<'a>> {
    match value {
        Value::Struct(st) => Ok(StructContext::Struct(st)),
        Value::Cell(cell) => {
            if cell.data.is_empty() {
                return Ok(StructContext::StructArray(cell));
            }
            if cell
                .data
                .iter()
                .all(|handle| matches!(unsafe { &*handle.as_raw() }, Value::Struct(_)))
            {
                Ok(StructContext::StructArray(cell))
            } else {
                Ok(StructContext::NonStruct)
            }
        }
        _ => Ok(StructContext::NonStruct),
    }
}

enum ParsedNames {
    Scalar(String),
    Array {
        names: Vec<String>,
        shape: Vec<usize>,
    },
}

fn parse_field_names(names: Value) -> BuiltinResult<ParsedNames> {
    match names {
        Value::String(s) => Ok(ParsedNames::Scalar(s)),
        Value::CharArray(ca) => {
            if ca.rows == 1 {
                Ok(ParsedNames::Scalar(ca.data.iter().collect()))
            } else {
                Err(field_name_type_error())
            }
        }
        Value::StringArray(sa) => Ok(ParsedNames::Array {
            names: sa.data.clone(),
            shape: sa.shape.clone(),
        }),
        Value::Cell(cell) => Ok(ParsedNames::Array {
            names: collect_cell_names(&cell)?,
            shape: if cell.shape.is_empty() {
                vec![cell.rows, cell.cols]
            } else {
                cell.shape.clone()
            },
        }),
        other => match try_single_field_name(&other)? {
            Some(name) => Ok(ParsedNames::Scalar(name)),
            None => Err(field_name_type_error()),
        },
    }
}

fn try_single_field_name(value: &Value) -> BuiltinResult<Option<String>> {
    match value {
        Value::String(s) => Ok(Some(s.clone())),
        Value::CharArray(ca) => {
            if ca.rows == 1 {
                Ok(Some(ca.data.iter().collect()))
            } else {
                Err(field_name_type_error())
            }
        }
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                Ok(Some(sa.data[0].clone()))
            } else {
                Err(field_name_type_error())
            }
        }
        _ => Ok(None),
    }
}

fn evaluate_scalar(struct_value: &StructValue, names: ParsedNames) -> BuiltinResult<Value> {
    match names {
        ParsedNames::Scalar(name) => Ok(Value::Bool(struct_value.fields.contains_key(&name))),
        ParsedNames::Array { names, shape } => {
            let mut bits = Vec::with_capacity(names.len());
            for name in names {
                bits.push(if struct_value.fields.contains_key(&name) {
                    1
                } else {
                    0
                });
            }
            let logical = LogicalArray::new(bits, shape).map_err(|e| isfield_flow(format!("isfield: {e}")))?;
            Ok(Value::LogicalArray(logical))
        }
    }
}

fn evaluate_struct_array(cell: &CellArray, names: ParsedNames) -> BuiltinResult<Value> {
    let fields = struct_array_field_intersection(cell)?;
    match names {
        ParsedNames::Scalar(name) => Ok(Value::Bool(fields.contains(&name))),
        ParsedNames::Array { names, shape } => {
            let mut bits = Vec::with_capacity(names.len());
            for name in names {
                bits.push(if fields.contains(&name) { 1 } else { 0 });
            }
            let logical = LogicalArray::new(bits, shape).map_err(|e| isfield_flow(format!("isfield: {e}")))?;
            Ok(Value::LogicalArray(logical))
        }
    }
}

fn evaluate_non_struct(names: ParsedNames) -> BuiltinResult<Value> {
    match names {
        ParsedNames::Scalar(_) => Ok(Value::Bool(false)),
        ParsedNames::Array { names, shape } => {
            let logical = LogicalArray::new(vec![0; names.len()], shape)
                .map_err(|e| isfield_flow(format!("isfield: {e}")))?;
            Ok(Value::LogicalArray(logical))
        }
    }
}

fn struct_array_field_intersection(cell: &CellArray) -> BuiltinResult<HashSet<String>> {
    if cell.data.is_empty() {
        return Ok(HashSet::new());
    }

    let mut iter = cell.data.iter();
    let first = unsafe { &*iter.next().unwrap().as_raw() };
    let Value::Struct(first_struct) = first else {
        return Err(isfield_flow("isfield: struct array elements must be structs"));
    };
    let mut fields: HashSet<String> = first_struct.fields.keys().cloned().collect();

    for handle in iter {
        let value = unsafe { &*handle.as_raw() };
        let Value::Struct(struct_value) = value else {
            return Err(isfield_flow("isfield: struct array elements must be structs"));
        };
        fields.retain(|name| struct_value.fields.contains_key(name));
        if fields.is_empty() {
            break;
        }
    }

    Ok(fields)
}

fn collect_cell_names(cell: &CellArray) -> BuiltinResult<Vec<String>> {
    let total = cell.data.len();
    if total == 0 {
        return Ok(Vec::new());
    }

    let shape = if cell.shape.is_empty() {
        vec![cell.rows, cell.cols]
    } else {
        cell.shape.clone()
    };

    let mut names = Vec::with_capacity(total);
    let row_strides = row_major_strides(&shape);
    for idx in 0..total {
        let coords = column_major_coordinates(idx, &shape);
        let mut row_index = 0usize;
        for (coord, stride) in coords.iter().zip(row_strides.iter()) {
            row_index += coord * stride;
        }
        let value = unsafe { &*cell.data[row_index].as_raw() };
        names.push(value_to_field_name(value)?);
    }
    Ok(names)
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut strides = vec![0; shape.len()];
    let mut stride = 1usize;
    for (i, dim) in shape.iter().enumerate().rev() {
        strides[i] = stride;
        stride = stride.saturating_mul(*dim.max(&1));
    }
    strides
}

fn column_major_coordinates(mut index: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut coords = vec![0usize; shape.len()];
    for (i, dim) in shape.iter().enumerate() {
        if *dim == 0 {
            coords[i] = 0;
            continue;
        }
        coords[i] = index % dim;
        index /= dim;
    }
    coords
}

fn value_to_field_name(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) => {
            if ca.rows == 1 {
                Ok(ca.data.iter().collect())
            } else {
                Err(field_name_type_error())
            }
        }
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                Ok(sa.data[0].clone())
            } else {
                Err(field_name_type_error())
            }
        }
        other => Err(isfield_flow(format!(
            "isfield: cell array elements must be character vectors or strings (got {other:?})"
        ))),
    }
}

fn field_name_type_error() -> RuntimeError {
    isfield_flow(
        "isfield: field names must be strings, string arrays, or cell arrays of character vectors",
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_builtins::{CellArray, CharArray, StringArray, StructValue};

    use crate::builtins::common::test_support;

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfield_scalar_struct_single_name() {
        let mut st = StructValue::new();
        st.fields.insert("name".to_string(), Value::from("Ada"));
        assert_eq!(
            isfield_builtin(Value::Struct(st.clone()), Value::from("name")).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            isfield_builtin(Value::Struct(st), Value::from("score")).unwrap(),
            Value::Bool(false)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfield_char_array_single_row() {
        let mut st = StructValue::new();
        st.fields.insert("alpha".into(), Value::Num(1.0));
        let chars = CharArray::new("alpha".chars().collect(), 1, 5).unwrap();
        let result = isfield_builtin(Value::Struct(st), Value::CharArray(chars)).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfield_struct_cell_names_produces_logical_array() {
        let mut st = StructValue::new();
        st.fields.insert("name".to_string(), Value::from("Ada"));
        st.fields.insert("score".to_string(), Value::from(42.0));
        let names = CellArray::new(
            vec![
                Value::from("name"),
                Value::from("department"),
                Value::from("score"),
                Value::from("email"),
            ],
            2,
            2,
        )
        .unwrap();
        let result = isfield_builtin(Value::Struct(st), Value::Cell(names)).expect("isfield");
        let expected = LogicalArray::new(vec![1, 1, 0, 0], vec![2, 2]).expect("logical array");
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfield_cell_mixed_string_types() {
        let mut st = StructValue::new();
        st.fields.insert("name".into(), Value::from("Ada"));
        st.fields.insert("id".into(), Value::from(7.0));
        let id_chars = CharArray::new("id".chars().collect(), 1, 2).unwrap();
        let cell = CellArray::new(
            vec![
                Value::from("name"),
                Value::CharArray(id_chars),
                Value::from("department"),
            ],
            1,
            3,
        )
        .unwrap();
        let result = isfield_builtin(Value::Struct(st), Value::Cell(cell)).unwrap();
        let expected = LogicalArray::new(vec![1, 1, 0], vec![1, 3]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfield_struct_array_intersection() {
        let mut first = StructValue::new();
        first.fields.insert("name".to_string(), Value::from("Ada"));
        first.fields.insert("id".to_string(), Value::from(101.0));

        let mut second = StructValue::new();
        second
            .fields
            .insert("name".to_string(), Value::from("Grace"));

        let struct_array = CellArray::new_with_shape(
            vec![Value::Struct(first), Value::Struct(second)],
            vec![1, 2],
        )
        .unwrap();

        let res_id =
            isfield_builtin(Value::Cell(struct_array.clone()), Value::from("id")).expect("isfield");
        assert_eq!(res_id, Value::Bool(false));

        let res_name =
            isfield_builtin(Value::Cell(struct_array), Value::from("name")).expect("isfield");
        assert_eq!(res_name, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfield_non_struct_returns_false() {
        let result = isfield_builtin(Value::Num(5.0), Value::from("field")).unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfield_string_array_names() {
        let mut st = StructValue::new();
        st.fields.insert("alpha".into(), Value::Num(1.0));
        st.fields.insert("beta".into(), Value::Num(2.0));
        let names = StringArray::new(vec!["alpha".into(), "gamma".into()], vec![2, 1]).unwrap();
        let result = isfield_builtin(Value::Struct(st), Value::StringArray(names)).unwrap();
        let expected = LogicalArray::new(vec![1, 0], vec![2, 1]).expect("logical array");
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfield_invalid_name_type_errors() {
        let mut st = StructValue::new();
        st.fields.insert("alpha".into(), Value::Num(1.0));
        let err = error_message(isfield_builtin(Value::Struct(st), Value::from(5_i32)).unwrap_err());
        assert!(err.contains("field names must be strings"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isfield_char_matrix_errors() {
        let mut st = StructValue::new();
        st.fields.insert("alpha".into(), Value::Num(1.0));
        let matrix = CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).unwrap();
        let err = error_message(
            isfield_builtin(Value::Struct(st), Value::CharArray(matrix)).unwrap_err(),
        );
        assert!(err.contains("field names must be strings"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
