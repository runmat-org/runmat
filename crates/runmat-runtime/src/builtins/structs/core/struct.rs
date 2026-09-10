//! MATLAB-compatible `struct` builtin.

use crate::builtins::common::identifiers::is_valid_varname;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_builtins::{
    STRUCT_ERROR_ASSEMBLE_FAILED, STRUCT_ERROR_CELL_SIZE_MISMATCH, STRUCT_ERROR_DUPLICATE_FIELD,
    STRUCT_ERROR_EMPTY_ARRAY_FAILED, STRUCT_ERROR_FIELD_NAME_CHAR_VECTOR,
    STRUCT_ERROR_FIELD_NAME_EMPTY, STRUCT_ERROR_FIELD_NAME_SCALAR,
    STRUCT_ERROR_FIELD_NAME_START_CHAR, STRUCT_ERROR_FIELD_NAME_TYPE,
    STRUCT_ERROR_INVALID_SINGLE_INPUT, STRUCT_ERROR_NAME_VALUE_PAIRS, STRUCT_ERROR_SIZE_OVERFLOW,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, StructArray, StructValue, Value};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

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

const BUILTIN_NAME: &str = "struct";

fn struct_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    struct_error_with_message(error.message, error)
}

fn struct_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "struct",
    binding_variant = "default",
    builtin_path = "crate::builtins::structs::core::r#struct"
)]
async fn struct_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    match rest.len() {
        0 => Ok(Value::Struct(StructValue::new())),
        1 => match rest.into_iter().next().unwrap() {
            Value::Struct(existing) => Ok(Value::Struct(existing)),
            Value::StructArray(existing) => Ok(Value::StructArray(existing)),
            Value::Tensor(tensor) if tensor::tensor_element_len(&tensor) == 0 => {
                empty_struct_array()
            }
            Value::LogicalArray(logical) if logical.data.is_empty() => empty_struct_array(),
            other => Err(struct_error_with_message(
                format!(
                    "{} (got {other:?})",
                    STRUCT_ERROR_INVALID_SINGLE_INPUT.message
                ),
                &STRUCT_ERROR_INVALID_SINGLE_INPUT,
            )),
        },
        len if len % 2 == 0 => build_from_pairs(rest),
        _ => Err(struct_error(&STRUCT_ERROR_NAME_VALUE_PAIRS)),
    }
}

fn build_from_pairs(args: Vec<Value>) -> BuiltinResult<Value> {
    let mut entries: Vec<FieldEntry> = Vec::new();
    let mut target_shape: Option<Vec<usize>> = None;
    let mut field_names = std::collections::HashSet::with_capacity(args.len() / 2);

    let mut iter = args.into_iter();
    while let (Some(name_value), Some(field_value)) = (iter.next(), iter.next()) {
        let field_name = parse_field_name(&name_value)?;
        if !field_names.insert(field_name.clone()) {
            return Err(struct_error(&STRUCT_ERROR_DUPLICATE_FIELD));
        }
        match field_value {
            Value::Cell(cell) => {
                let shape = cell.shape.clone();
                if let Some(existing) = &target_shape {
                    if *existing != shape {
                        return Err(struct_error(&STRUCT_ERROR_CELL_SIZE_MISMATCH));
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

fn build_scalar_struct(entries: Vec<FieldEntry>) -> BuiltinResult<Value> {
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

fn build_struct_array(entries: Vec<FieldEntry>, shape: Vec<usize>) -> BuiltinResult<Value> {
    let total_len = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| struct_error(&STRUCT_ERROR_SIZE_OVERFLOW))?;

    for entry in &entries {
        if let FieldValue::Cell(cell) = &entry.value {
            if cell.data.len() != total_len {
                return Err(struct_error(&STRUCT_ERROR_CELL_SIZE_MISMATCH));
            }
        }
    }

    let column_major_cells = entries
        .iter()
        .map(|entry| match &entry.value {
            FieldValue::Cell(cell) => Some(cell.to_column_major()),
            FieldValue::Single(_) => None,
        })
        .collect::<Vec<_>>();
    let mut structs = Vec::with_capacity(total_len);
    for idx in 0..total_len {
        let mut fields = StructValue::new();
        for (entry, column_major) in entries.iter().zip(&column_major_cells) {
            let value = match &entry.value {
                FieldValue::Single(val) => val.clone(),
                FieldValue::Cell(_) => column_major
                    .as_ref()
                    .and_then(|values| values.get(idx))
                    .cloned()
                    .ok_or_else(|| struct_error(&STRUCT_ERROR_CELL_SIZE_MISMATCH))?,
            };
            fields.fields.insert(entry.name.clone(), value);
        }
        structs.push(fields);
    }

    let field_names = entries.iter().map(|entry| entry.name.clone()).collect();
    StructArray::normalize(field_names, structs, shape).map_err(|e| {
        struct_error_with_message(
            format!("{}: {e}", STRUCT_ERROR_ASSEMBLE_FAILED.message),
            &STRUCT_ERROR_ASSEMBLE_FAILED,
        )
    })
}

fn empty_struct_array() -> BuiltinResult<Value> {
    StructArray::empty(Vec::new(), vec![0, 0])
        .map(Value::StructArray)
        .map_err(|e| {
            struct_error_with_message(
                format!("{}: {e}", STRUCT_ERROR_EMPTY_ARRAY_FAILED.message),
                &STRUCT_ERROR_EMPTY_ARRAY_FAILED,
            )
        })
}

fn parse_field_name(value: &Value) -> BuiltinResult<String> {
    let text = match value {
        Value::String(s) => s.clone(),
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                sa.data[0].clone()
            } else {
                return Err(struct_error(&STRUCT_ERROR_FIELD_NAME_SCALAR));
            }
        }
        Value::CharArray(ca) => char_array_to_string(ca)?,
        _ => return Err(struct_error(&STRUCT_ERROR_FIELD_NAME_TYPE)),
    };

    validate_field_name(&text)?;
    Ok(text)
}

fn char_array_to_string(ca: &CharArray) -> BuiltinResult<String> {
    if ca.rows > 1 {
        return Err(struct_error(&STRUCT_ERROR_FIELD_NAME_CHAR_VECTOR));
    }
    let mut out = String::with_capacity(ca.data.len());
    for ch in &ca.data {
        out.push(*ch);
    }
    Ok(out)
}

fn validate_field_name(name: &str) -> BuiltinResult<()> {
    if name.is_empty() {
        return Err(struct_error(&STRUCT_ERROR_FIELD_NAME_EMPTY));
    }

    if !is_valid_varname(name) {
        return Err(struct_error_with_message(
            format!(
                "{} (got '{name}')",
                STRUCT_ERROR_FIELD_NAME_START_CHAR.message
            ),
            &STRUCT_ERROR_FIELD_NAME_START_CHAR,
        ));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::identifiers::MATLAB_NAME_LENGTH_MAX;
    use runmat_accelerate_api::GpuTensorHandle;
    use runmat_value::{CellArray, IntValue, StringArray, StructArray, StructValue, Tensor};

    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::HostTensorView;

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    fn run_struct(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(struct_builtin(args))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_empty() {
        let Value::Struct(s) = run_struct(Vec::new()).expect("struct") else {
            panic!("expected struct value");
        };
        assert!(s.fields.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_empty_from_empty_matrix() {
        let tensor = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let value = run_struct(vec![Value::Tensor(tensor)]).expect("struct([])");
        match value {
            Value::StructArray(array) => {
                assert_eq!(array.shape(), &[0, 0]);
                assert!(array.is_empty());
            }
            other => panic!("expected empty struct array, got {other:?}"),
        }
    }

    #[test]
    fn struct_uses_typed_integer_storage_to_detect_empty_input() {
        let tensor =
            Tensor::new_integer(runmat_value::IntegerStorage::U64(Vec::new()), vec![0, 0]).unwrap();

        assert!(matches!(
            run_struct(vec![Value::Tensor(tensor)]).unwrap(),
            Value::StructArray(_)
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_name_value_pairs() {
        let args = vec![
            Value::from("name"),
            Value::from("entry-a"),
            Value::from("score"),
            Value::Int(IntValue::I32(42)),
        ];
        let Value::Struct(s) = run_struct(args).expect("struct") else {
            panic!("expected struct value");
        };
        assert_eq!(s.fields.len(), 2);
        assert!(matches!(s.fields.get("name"), Some(Value::String(v)) if v == "entry-a"));
        assert!(matches!(
            s.fields.get("score"),
            Some(Value::Int(IntValue::I32(42)))
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_struct_array_from_cells() {
        let names =
            CellArray::new(vec![Value::from("entry-a"), Value::from("entry-b")], 1, 2).unwrap();
        let ages = CellArray::new(
            vec![Value::Int(IntValue::I32(36)), Value::Int(IntValue::I32(45))],
            1,
            2,
        )
        .unwrap();
        let result = run_struct(vec![
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
            Some(Value::String(v)) if v == "entry-a"
        ));
        assert!(matches!(
            structs[1].fields.get("age"),
            Some(Value::Int(IntValue::I32(45)))
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_struct_array_replicates_scalars() {
        let names =
            CellArray::new(vec![Value::from("entry-a"), Value::from("entry-b")], 1, 2).unwrap();
        let result = run_struct(vec![
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
        let names =
            CellArray::new(vec![Value::from("entry-a"), Value::from("entry-b")], 1, 2).unwrap();
        let scores = CellArray::new(vec![Value::Int(IntValue::I32(1))], 1, 1).unwrap();
        let err = error_message(
            run_struct(vec![
                Value::from("name"),
                Value::Cell(names),
                Value::from("score"),
                Value::Cell(scores),
            ])
            .unwrap_err(),
        );
        assert!(err.contains("matching sizes"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_rejects_duplicate_scalar_fields() {
        let args = vec![
            Value::from("version"),
            Value::Int(IntValue::I32(1)),
            Value::from("version"),
            Value::Int(IntValue::I32(2)),
        ];
        let error = run_struct(args).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:struct:DuplicateField"));
    }

    #[test]
    fn struct_rejects_duplicate_array_fields() {
        let values = || CellArray::new(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap();
        let error = run_struct(vec![
            Value::from("version"),
            Value::Cell(values()),
            Value::from("version"),
            Value::Cell(values()),
        ])
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:struct:DuplicateField"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_rejects_odd_arguments() {
        let err = error_message(run_struct(vec![Value::from("name")]).unwrap_err());
        assert!(err.contains("name/value pairs"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_rejects_invalid_field_name() {
        let err = error_message(
            run_struct(vec![Value::from("1bad"), Value::Int(IntValue::I32(1))]).unwrap_err(),
        );
        assert!(err.contains("valid MATLAB identifiers"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_field_names_follow_shared_identifier_rules() {
        let max_len = "a".repeat(MATLAB_NAME_LENGTH_MAX);
        let Value::Struct(s) = run_struct(vec![
            Value::from(max_len.clone()),
            Value::Int(IntValue::I32(1)),
        ])
        .expect("max length field name") else {
            panic!("expected struct value");
        };
        assert!(s.fields.contains_key(&max_len));

        for bad in [
            "_hidden".to_string(),
            "for".to_string(),
            "éclair".to_string(),
            "a".repeat(MATLAB_NAME_LENGTH_MAX + 1),
        ] {
            let err =
                error_message(run_struct(vec![Value::from(bad), Value::Num(1.0)]).unwrap_err());
            assert!(err.contains("valid MATLAB identifiers"));
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_rejects_non_text_field_name() {
        let err = error_message(
            run_struct(vec![Value::Num(1.0), Value::Int(IntValue::I32(1))]).unwrap_err(),
        );
        assert!(err.contains("strings or character vectors"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_accepts_char_vector_name() {
        let chars = CharArray::new("field".chars().collect(), 1, 5).unwrap();
        let args = vec![Value::CharArray(chars), Value::Num(1.0)];
        let Value::Struct(s) = run_struct(args).expect("struct") else {
            panic!("expected struct value");
        };
        assert!(s.fields.contains_key("field"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_accepts_string_scalar_name() {
        let sa = StringArray::new(vec!["field".to_string()], vec![1]).unwrap();
        let args = vec![Value::StringArray(sa), Value::Num(1.0)];
        let Value::Struct(s) = run_struct(args).expect("struct") else {
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
        let copy = run_struct(vec![Value::Struct(base.clone())]).expect("struct");
        assert_eq!(copy, Value::Struct(base));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_copies_struct_array_argument() {
        let mut proto = StructValue::new();
        proto
            .fields
            .insert("id".into(), Value::Int(IntValue::I32(7)));
        let struct_array = StructArray::new(vec![proto.clone(); 3], vec![1, 3]).unwrap();
        let original = struct_array.clone();
        let result =
            run_struct(vec![Value::StructArray(struct_array)]).expect("struct array clone");
        assert_eq!(result, Value::StructArray(original));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_rejects_cell_argument_without_structs() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        assert!(run_struct(vec![Value::Cell(cell)]).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn struct_preserves_gpu_tensor_handles() {
        let handle = GpuTensorHandle {
            shape: vec![2, 2],
            device_id: 1,
            buffer_id: 99,
            descriptor: Default::default(),
        };
        let args = vec![Value::from("data"), Value::GpuTensor(handle.clone())];
        let Value::Struct(s) = run_struct(args).expect("struct") else {
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
            descriptor: Default::default(),
        };
        let second = GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 2,
            buffer_id: 12,
            descriptor: Default::default(),
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
        let result = run_struct(vec![Value::from("payload"), Value::Cell(cell)])
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
        let Value::Struct(s) = run_struct(args).expect("struct") else {
            panic!("expected struct value");
        };
        assert!(matches!(s.fields.get("gpu"), Some(Value::GpuTensor(h)) if h == &handle));
    }

    fn expect_struct_array(value: Value) -> Vec<StructValue> {
        match value {
            Value::StructArray(array) => array.into_elements(),
            Value::Struct(st) => vec![st],
            other => panic!("expected struct or struct array, got {other:?}"),
        }
    }
}
