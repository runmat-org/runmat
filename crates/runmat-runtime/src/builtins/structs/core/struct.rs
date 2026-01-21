//! MATLAB-compatible `struct` builtin.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use runmat_builtins::{CellArray, CharArray, StructValue, Value};
use runmat_macros::runtime_builtin;

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

    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::HostTensorView;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_empty() {
        let Value::Struct(s) = struct_builtin(Vec::new()).expect("struct") else {
            panic!("expected struct value");
        };
        assert!(s.fields.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_rejects_odd_arguments() {
        let err = struct_builtin(vec![Value::from("name")]).unwrap_err();
        assert!(err.contains("name/value pairs"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_rejects_invalid_field_name() {
        let err =
            struct_builtin(vec![Value::from("1bad"), Value::Int(IntValue::I32(1))]).unwrap_err();
        assert!(err.contains("begin with a letter or underscore"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_rejects_non_text_field_name() {
        let err = struct_builtin(vec![Value::Num(1.0), Value::Int(IntValue::I32(1))]).unwrap_err();
        assert!(err.contains("strings or character vectors"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_accepts_char_vector_name() {
        let chars = CharArray::new("field".chars().collect(), 1, 5).unwrap();
        let args = vec![Value::CharArray(chars), Value::Num(1.0)];
        let Value::Struct(s) = struct_builtin(args).expect("struct") else {
            panic!("expected struct value");
        };
        assert!(s.fields.contains_key("field"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_accepts_string_scalar_name() {
        let sa = StringArray::new(vec!["field".to_string()], vec![1]).unwrap();
        let args = vec![Value::StringArray(sa), Value::Num(1.0)];
        let Value::Struct(s) = struct_builtin(args).expect("struct") else {
            panic!("expected struct value");
        };
        assert!(s.fields.contains_key("field"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_allows_existing_struct_copy() {
        let mut base = StructValue::new();
        base.fields
            .insert("id".to_string(), Value::Int(IntValue::I32(7)));
        let copy = struct_builtin(vec![Value::Struct(base.clone())]).expect("struct");
        assert_eq!(copy, Value::Struct(base));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_rejects_cell_argument_without_structs() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let err = struct_builtin(vec![Value::Cell(cell)]).unwrap_err();
        assert!(err.contains("must contain structs"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
