//! MATLAB-compatible `rmfield` builtin that removes fields from structs and struct arrays.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::structs::type_resolvers::rmfield_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{CellArray, StringArray, StructValue, Value};
use runmat_macros::runtime_builtin;
use std::collections::HashSet;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::structs::core::rmfield")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "rmfield",
    op_kind: GpuOpKind::Custom("rmfield"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host-only struct metadata update; acceleration providers are not consulted.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::structs::core::rmfield")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "rmfield",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Metadata mutation forces fusion planners to flush pending groups on the host.",
};

fn rmfield_flow(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("rmfield").build()
}

#[runtime_builtin(
    name = "rmfield",
    category = "structs/core",
    summary = "Remove one or more fields from scalar structs or struct arrays.",
    keywords = "rmfield,struct,remove field,struct array",
    type_resolver(rmfield_type),
    builtin_path = "crate::builtins::structs::core::rmfield"
)]
async fn rmfield_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let names = parse_field_names(&rest)?;
    if names.is_empty() {
        return Ok(value);
    }

    match value {
        Value::Struct(st) => {
            let updated = remove_fields_from_struct_owned(st, &names)?;
            Ok(Value::Struct(updated))
        }
        Value::Cell(cell) if is_struct_array(&cell) => {
            let updated = remove_fields_from_struct_array(&cell, &names)?;
            Ok(Value::Cell(updated))
        }
        other => Err(rmfield_flow(format!(
            "rmfield: expected struct or struct array, got {other:?}"
        ))),
    }
}

fn parse_field_names(args: &[Value]) -> BuiltinResult<Vec<String>> {
    if args.is_empty() {
        return Err(rmfield_flow("rmfield: not enough input arguments"));
    }
    let mut names: Vec<String> = Vec::new();
    for value in args {
        names.extend(collect_field_names(value)?);
    }
    Ok(names)
}

fn collect_field_names(value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(_) | Value::CharArray(_) => expect_scalar_name(value)
            .map(|name| vec![name])
            .map_err(|err| rmfield_flow(format!("rmfield: {}", describe_field_name_error(err)))),
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                expect_scalar_name(value)
                    .map(|name| vec![name])
                    .map_err(|err| {
                        rmfield_flow(format!("rmfield: {}", describe_field_name_error(err)))
                    })
            } else {
                string_array_to_names(sa)
            }
        }
        Value::Cell(cell) => cell_to_names(cell),
        other => Err(rmfield_flow(format!(
            "rmfield: field names must be strings or character vectors (got {other:?})"
        ))),
    }
}

fn string_array_to_names(array: &StringArray) -> BuiltinResult<Vec<String>> {
    let mut names = Vec::with_capacity(array.data.len());
    for (index, name) in array.data.iter().enumerate() {
        if name.is_empty() {
            return Err(rmfield_flow(format!(
                "rmfield: field names must be nonempty character vectors or strings (string array element {})",
                index + 1
            )));
        }
        names.push(name.clone());
    }
    Ok(names)
}

fn cell_to_names(cell: &CellArray) -> BuiltinResult<Vec<String>> {
    let mut output = Vec::with_capacity(cell.data.len());
    for (index, handle) in cell.data.iter().enumerate() {
        let value = unsafe { &*handle.as_raw() };
        let name = expect_scalar_name(value).map_err(|err| {
            rmfield_flow(format!(
                "rmfield: {} (cell element {})",
                describe_field_name_error(err),
                index + 1
            ))
        })?;
        output.push(name);
    }
    Ok(output)
}

#[derive(Clone, Copy)]
enum FieldNameError {
    Type,
    Empty,
}

fn describe_field_name_error(kind: FieldNameError) -> &'static str {
    match kind {
        FieldNameError::Type => {
            "field names must be string scalars, character vectors, or single-element string arrays"
        }
        FieldNameError::Empty => "field names must be nonempty character vectors or strings",
    }
}

fn expect_scalar_name(value: &Value) -> Result<String, FieldNameError> {
    match value {
        Value::String(s) => {
            if s.is_empty() {
                Err(FieldNameError::Empty)
            } else {
                Ok(s.clone())
            }
        }
        Value::CharArray(ca) => {
            if ca.rows != 1 {
                return Err(FieldNameError::Type);
            }
            let text: String = ca.data.iter().collect();
            if text.is_empty() {
                Err(FieldNameError::Empty)
            } else {
                Ok(text)
            }
        }
        Value::StringArray(sa) => {
            if sa.data.len() != 1 {
                return Err(FieldNameError::Type);
            }
            let text = sa.data[0].clone();
            if text.is_empty() {
                Err(FieldNameError::Empty)
            } else {
                Ok(text)
            }
        }
        _ => Err(FieldNameError::Type),
    }
}

fn remove_fields_from_struct_owned(
    mut st: StructValue,
    names: &[String],
) -> BuiltinResult<StructValue> {
    let mut seen: HashSet<&str> = HashSet::new();
    for name in names {
        if !seen.insert(name.as_str()) {
            continue;
        }
        if st.remove(name).is_none() {
            return Err(missing_field_error(name));
        }
    }
    Ok(st)
}

fn remove_fields_from_struct_array(
    array: &CellArray,
    names: &[String],
) -> BuiltinResult<CellArray> {
    if array.data.is_empty() {
        return Ok(array.clone());
    }

    let mut updated: Vec<Value> = Vec::with_capacity(array.data.len());
    for handle in &array.data {
        let value = unsafe { &*handle.as_raw() };
        let Value::Struct(st) = value else {
            return Err(rmfield_flow(
                "rmfield: expected struct array contents to be structs",
            ));
        };
        let revised = remove_fields_from_struct_owned(st.clone(), names)?;
        updated.push(Value::Struct(revised));
    }
    CellArray::new_with_shape(updated, array.shape.clone())
        .map_err(|e| rmfield_flow(format!("rmfield: failed to rebuild struct array: {e}")))
}

fn missing_field_error(name: &str) -> RuntimeError {
    rmfield_flow(format!("Reference to non-existent field '{name}'."))
}

fn is_struct_array(cell: &CellArray) -> bool {
    cell.data
        .iter()
        .all(|handle| matches!(unsafe { &*handle.as_raw() }, Value::Struct(_)))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_builtins::{CellArray, CharArray, StringArray, StructValue, Value};

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    fn run_rmfield(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(rmfield_builtin(value, rest))
    }
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::HostTensorView;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmfield_removes_single_field_from_scalar_struct() {
        let mut st = StructValue::new();
        st.fields.insert("name".to_string(), Value::from("Ada"));
        st.fields.insert("score".to_string(), Value::Num(42.0));
        let result = run_rmfield(Value::Struct(st), vec![Value::from("score")]).expect("rmfield");
        let Value::Struct(updated) = result else {
            panic!("expected struct result");
        };
        assert!(!updated.fields.contains_key("score"));
        assert!(updated.fields.contains_key("name"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmfield_accepts_cell_array_of_field_names() {
        let mut st = StructValue::new();
        st.fields.insert("left".to_string(), Value::Num(1.0));
        st.fields.insert("right".to_string(), Value::Num(2.0));
        st.fields.insert("top".to_string(), Value::Num(3.0));
        let cell =
            CellArray::new(vec![Value::from("left"), Value::from("top")], 1, 2).expect("cell");
        let result = run_rmfield(Value::Struct(st), vec![Value::Cell(cell)]).expect("rmfield");
        let Value::Struct(updated) = result else {
            panic!("expected struct result");
        };
        assert!(!updated.fields.contains_key("left"));
        assert!(!updated.fields.contains_key("top"));
        assert!(updated.fields.contains_key("right"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmfield_supports_string_array_names() {
        let mut st = StructValue::new();
        st.fields.insert("alpha".to_string(), Value::Num(1.0));
        st.fields.insert("beta".to_string(), Value::Num(2.0));
        st.fields.insert("gamma".to_string(), Value::Num(3.0));
        let strings = StringArray::new(vec!["alpha".into(), "gamma".into()], vec![1, 2]).unwrap();
        let result =
            run_rmfield(Value::Struct(st), vec![Value::StringArray(strings)]).expect("rmfield");
        let Value::Struct(updated) = result else {
            panic!("expected struct result");
        };
        assert!(!updated.fields.contains_key("alpha"));
        assert!(!updated.fields.contains_key("gamma"));
        assert!(updated.fields.contains_key("beta"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmfield_errors_when_field_missing() {
        let mut st = StructValue::new();
        st.fields.insert("name".to_string(), Value::from("Ada"));
        let err =
            error_message(run_rmfield(Value::Struct(st), vec![Value::from("id")]).unwrap_err());
        assert!(
            err.contains("Reference to non-existent field 'id'."),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmfield_struct_array_roundtrip() {
        let mut first = StructValue::new();
        first.fields.insert("name".to_string(), Value::from("Ada"));
        first.fields.insert("score".to_string(), Value::Num(90.0));

        let mut second = StructValue::new();
        second
            .fields
            .insert("name".to_string(), Value::from("Grace"));
        second.fields.insert("score".to_string(), Value::Num(95.0));

        let array = CellArray::new_with_shape(
            vec![Value::Struct(first), Value::Struct(second)],
            vec![1, 2],
        )
        .expect("struct array");

        let result = run_rmfield(Value::Cell(array), vec![Value::from("score")]).expect("rmfield");
        let Value::Cell(updated) = result else {
            panic!("expected struct array");
        };
        for handle in &updated.data {
            let value = unsafe { &*handle.as_raw() };
            let Value::Struct(st) = value else {
                panic!("expected struct element");
            };
            assert!(!st.fields.contains_key("score"));
            assert!(st.fields.contains_key("name"));
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmfield_struct_array_missing_field_errors() {
        let mut first = StructValue::new();
        first.fields.insert("id".to_string(), Value::Num(1.0));
        let mut second = StructValue::new();
        second.fields.insert("id".to_string(), Value::Num(2.0));
        second.fields.insert("extra".to_string(), Value::Num(3.0));

        let array = CellArray::new_with_shape(
            vec![Value::Struct(first), Value::Struct(second)],
            vec![1, 2],
        )
        .expect("struct array");

        let err = error_message(
            run_rmfield(Value::Cell(array), vec![Value::from("missing")]).unwrap_err(),
        );
        assert!(
            err.contains("Reference to non-existent field 'missing'."),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmfield_rejects_non_struct_inputs() {
        let err =
            error_message(run_rmfield(Value::Num(1.0), vec![Value::from("field")]).unwrap_err());
        assert!(
            err.contains("expected struct or struct array"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmfield_produces_error_for_empty_field_name() {
        let mut st = StructValue::new();
        st.fields.insert("data".to_string(), Value::Num(1.0));
        let err = error_message(run_rmfield(Value::Struct(st), vec![Value::from("")]).unwrap_err());
        assert!(
            err.contains("field names must be nonempty"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmfield_accepts_multiple_argument_forms() {
        let mut st = StructValue::new();
        st.fields.insert("alpha".to_string(), Value::Num(1.0));
        st.fields.insert("beta".to_string(), Value::Num(2.0));
        st.fields.insert("gamma".to_string(), Value::Num(3.0));
        st.fields.insert("delta".to_string(), Value::Num(4.0));

        let char_name = CharArray::new_row("beta");
        let string_array =
            StringArray::new(vec!["gamma".into()], vec![1, 1]).expect("string scalar array");
        let cell = CellArray::new(vec![Value::from("delta")], 1, 1).expect("cell array of strings");

        let result = run_rmfield(
            Value::Struct(st),
            vec![
                Value::from("alpha"),
                Value::CharArray(char_name),
                Value::StringArray(string_array),
                Value::Cell(cell),
            ],
        )
        .expect("rmfield");

        let Value::Struct(updated) = result else {
            panic!("expected struct result");
        };

        assert!(updated.fields.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmfield_ignores_duplicate_field_names() {
        let mut st = StructValue::new();
        st.fields.insert("keep".to_string(), Value::Num(1.0));
        st.fields.insert("drop".to_string(), Value::Num(2.0));
        let result = run_rmfield(
            Value::Struct(st),
            vec![Value::from("drop"), Value::from("drop")],
        )
        .expect("rmfield");
        let Value::Struct(updated) = result else {
            panic!("expected struct result");
        };
        assert!(!updated.fields.contains_key("drop"));
        assert!(updated.fields.contains_key("keep"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmfield_returns_original_when_no_names_supplied() {
        let mut st = StructValue::new();
        st.fields.insert("value".to_string(), Value::Num(10.0));
        let empty = CellArray::new(Vec::new(), 0, 0).expect("empty cell array");
        let original = st.clone();
        let result =
            run_rmfield(Value::Struct(st), vec![Value::Cell(empty)]).expect("rmfield empty");
        assert_eq!(result, Value::Struct(original));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmfield_requires_field_names() {
        let mut st = StructValue::new();
        st.fields.insert("value".to_string(), Value::Num(10.0));
        let err = error_message(run_rmfield(Value::Struct(st), Vec::new()).unwrap_err());
        assert!(
            err.contains("rmfield: not enough input arguments"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn rmfield_preserves_gpu_handles() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = HostTensorView {
            data: &[1.0, 2.0],
            shape: &[2, 1],
        };
        let handle = provider.upload(&view).expect("upload");

        let mut st = StructValue::new();
        st.fields
            .insert("gpu".to_string(), Value::GpuTensor(handle.clone()));
        st.fields.insert("remove".to_string(), Value::Num(5.0));

        let result = run_rmfield(Value::Struct(st), vec![Value::from("remove")]).expect("rmfield");

        let Value::Struct(updated) = result else {
            panic!("expected struct result");
        };

        assert!(matches!(
            updated.fields.get("gpu"),
            Some(Value::GpuTensor(h)) if h == &handle
        ));
        assert!(!updated.fields.contains_key("remove"));
    }
}
