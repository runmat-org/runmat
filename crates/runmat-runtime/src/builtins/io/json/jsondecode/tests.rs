use super::*;
use futures::executor::block_on;
use runmat_builtins::BuiltinIntegerAuditKind;
use runmat_value::{IntValue, Tensor};

use crate::RuntimeError;

fn char_row(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}

fn error_message(err: RuntimeError) -> String {
    err.message().to_string()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsondecode_descriptor_signature_covers_core_form() {
    let labels: Vec<&str> = JSONDECODE_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"value = jsondecode(text)"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsondecode_scalar_number() {
    let result = block_on(jsondecode_builtin(char_row("42"))).expect("jsondecode");
    assert_eq!(result, Value::Num(42.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsondecode_boolean_array() {
    let result = block_on(jsondecode_builtin(char_row("[true,false,true]"))).expect("jsondecode");
    match result {
        Value::LogicalArray(array) => {
            assert_eq!(array.shape, vec![3]);
            assert_eq!(array.data, vec![1, 0, 1]);
        }
        other => panic!("expected logical array, got {:?}", other),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsondecode_matrix_to_tensor() {
    let result =
        block_on(jsondecode_builtin(char_row("[[1,2,3],[4,5,6]]"))).expect("jsondecode matrix");
    match result {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[2, 3]);
            assert_eq!(tensor.materialize_f64(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        }
        other => panic!("expected tensor, got {:?}", other),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsondecode_numeric_tensor_3d() {
    let json = "[[[1,2],[3,4]],[[5,6],[7,8]]]";
    let result = block_on(jsondecode_builtin(char_row(json))).expect("jsondecode 3d tensor");
    match result {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[2, 2, 2]);
            assert_eq!(
                tensor.materialize_f64(),
                vec![1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0]
            );
        }
        other => panic!("expected tensor, got {:?}", other),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsondecode_numeric_singleton_array_retains_tensor() {
    let result =
        block_on(jsondecode_builtin(char_row("[42]"))).expect("jsondecode singleton numeric array");
    match result {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[1]);
            assert_eq!(tensor.rows(), 1);
            assert_eq!(tensor.cols(), 1);
            assert_eq!(tensor.materialize_f64(), vec![42.0]);
        }
        other => panic!("expected tensor, got {:?}", other),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsondecode_object_to_struct() {
    let result = block_on(jsondecode_builtin(char_row(
        "{\"name\":\"RunMat\",\"year\":2025}",
    )))
    .expect("jsondecode struct");
    match result {
        Value::Struct(struct_value) => {
            assert_eq!(
                struct_value.fields.get("name"),
                Some(&Value::CharArray(CharArray::new_row("RunMat")))
            );
            match struct_value.fields.get("year") {
                Some(Value::Num(year)) => assert_eq!(*year, 2025.0),
                Some(Value::Int(IntValue::I32(year))) => assert_eq!(*year, 2025),
                other => panic!("unexpected year field {other:?}"),
            }
        }
        other => panic!("expected struct, got {:?}", other),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsondecode_string_array() {
    let result = block_on(jsondecode_builtin(char_row(
        "[\"alpha\",\"beta\",\"gamma\"]",
    )))
    .expect("jsondecode string array");
    match result {
        Value::StringArray(array) => {
            assert_eq!(array.shape, vec![3]);
            assert_eq!(array.rows, 1);
            assert_eq!(array.cols, 3);
            assert_eq!(
                array.data(),
                vec![
                    String::from("alpha"),
                    String::from("beta"),
                    String::from("gamma"),
                ]
            );
        }
        other => panic!("expected string array, got {:?}", other),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsondecode_mixed_array_returns_cell() {
    let result =
        block_on(jsondecode_builtin(char_row("[\"RunMat\",42,true]"))).expect("jsondecode mixed");
    match result {
        Value::Cell(cell) => {
            assert_eq!(cell.rows, 1);
            assert_eq!(cell.cols, 3);
            let first = cell.get(0, 0).expect("cell");
            assert_eq!(first, Value::CharArray(CharArray::new_row("RunMat")));
            let second = cell.get(0, 1).expect("cell");
            assert_eq!(second, Value::Num(42.0));
            let third = cell.get(0, 2).expect("cell");
            assert_eq!(third, Value::Bool(true));
        }
        other => panic!("expected cell array, got {:?}", other),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsondecode_rectangular_cell_array_preserves_layout() {
    let text = "[[1,true],[false,null]]";
    let result = block_on(jsondecode_builtin(char_row(text)))
        .expect("jsondecode rectangular heterogeneous array");
    match result {
        Value::Cell(cell) => {
            assert_eq!(cell.rows, 2);
            assert_eq!(cell.cols, 2);
            assert_eq!(cell.get(0, 0).unwrap(), Value::Num(1.0));
            assert_eq!(cell.get(0, 1).unwrap(), Value::Bool(true));
            assert_eq!(cell.get(1, 0).unwrap(), Value::Bool(false));
            match cell.get(1, 1).unwrap() {
                Value::Tensor(t) => {
                    assert_eq!(t.shape(), &[0, 0]);
                    assert!(t.materialize_f64().is_empty());
                }
                other => panic!("expected empty tensor, got {:?}", other),
            }
        }
        other => panic!("expected cell array, got {:?}", other),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsondecode_array_of_objects_returns_typed_column_vector() {
    let text = "[{\"id\":1,\"name\":\"first\"},{\"name\":\"second\",\"id\":2}]";
    let result = block_on(jsondecode_builtin(char_row(text))).expect("jsondecode object array");
    let Value::StructArray(array) = result else {
        panic!("expected structure array");
    };
    assert_eq!(array.shape(), [2, 1]);
    assert_eq!(
        array
            .field_names()
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>(),
        ["id", "name"]
    );
    assert_eq!(
        array.get_linear(0).unwrap().fields.get("id"),
        Some(&Value::Num(1.0))
    );
    assert_eq!(
        array.get_linear(1).unwrap().fields.get("id"),
        Some(&Value::Num(2.0))
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsondecode_rectangular_object_array_uses_column_major_order() {
    let text = "[[{\"id\":1},{\"id\":2}],[{\"id\":3},{\"id\":4}]]";
    let Value::StructArray(array) =
        block_on(jsondecode_builtin(char_row(text))).expect("rectangular object array")
    else {
        panic!("expected structure array");
    };
    assert_eq!(array.shape(), [2, 2]);
    let ids = array
        .elements()
        .map(|element| element.fields["id"].clone())
        .collect::<Vec<_>>();
    assert_eq!(
        ids,
        [
            Value::Num(1.0),
            Value::Num(3.0),
            Value::Num(2.0),
            Value::Num(4.0)
        ]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsondecode_nd_object_array_uses_column_major_order() {
    let text = concat!(
        "[[[{\"id\":1},{\"id\":2}],[{\"id\":3},{\"id\":4}]],",
        "[[{\"id\":5},{\"id\":6}],[{\"id\":7},{\"id\":8}]]]"
    );
    let Value::StructArray(array) =
        block_on(jsondecode_builtin(char_row(text))).expect("N-D object array")
    else {
        panic!("expected structure array");
    };
    assert_eq!(array.shape(), [2, 2, 2]);
    let ids = array
        .elements()
        .map(|element| element.fields["id"].clone())
        .collect::<Vec<_>>();
    assert_eq!(
        ids,
        [
            Value::Num(1.0),
            Value::Num(5.0),
            Value::Num(3.0),
            Value::Num(7.0),
            Value::Num(2.0),
            Value::Num(6.0),
            Value::Num(4.0),
            Value::Num(8.0),
        ]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsondecode_heterogeneous_object_fields_remain_a_cell() {
    let text = "[{\"left\":1},{\"right\":2}]";
    let Value::Cell(cell) =
        block_on(jsondecode_builtin(char_row(text))).expect("heterogeneous object array")
    else {
        panic!("expected cell array");
    };
    assert_eq!(cell.shape, [2, 1]);
    assert!(matches!(cell.data(), [Value::Struct(_), Value::Struct(_)]));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn json_object_array_schema_uses_deterministic_canonical_key_order() {
    let text = concat!(
        "[[[{\"zeta\":1,\"ä\":2,\"alpha\":3}]],",
        "[[{\"ä\":4,\"alpha\":5,\"zeta\":6}]]]"
    );
    let Value::StructArray(array) =
        block_on(jsondecode_builtin(char_row(text))).expect("N-D object array")
    else {
        panic!("expected structure array")
    };
    assert_eq!(array.shape(), [2, 1, 1]);
    assert_eq!(
        array.field_names().collect::<Vec<_>>(),
        ["alpha", "zeta", "ä"]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsondecode_null_returns_empty_double() {
    let result = block_on(jsondecode_builtin(char_row("null"))).expect("jsondecode null");
    match result {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[0, 0]);
            assert!(tensor.materialize_f64().is_empty());
        }
        other => panic!("expected empty tensor, got {:?}", other),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsondecode_invalid_text_reports_error() {
    let err = block_on(jsondecode_builtin(char_row("{not json}"))).expect_err("expected failure");
    let err = error_message(err);
    assert!(
        err.starts_with(JSONDECODE_ERROR_PARSE.message),
        "unexpected error message: {err}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsondecode_rejects_multirow_char_input() {
    let chars =
        CharArray::from_unicode_scalars(vec!['a', 'b', 'c', 'd'], 2, 2).expect("char array");
    let err =
        block_on(jsondecode_builtin(Value::CharArray(chars))).expect_err("expected type error");
    assert_eq!(error_message(err), JSONDECODE_ERROR_INPUT_TYPE.message);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsondecode_accepts_string_input() {
    let result = block_on(jsondecode_builtin(Value::String(
        "[1,2]".to_string().into(),
    )))
    .expect("jsondecode");
    match result {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[2]);
            assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0]);
        }
        other => panic!("expected tensor, got {:?}", other),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsondecode_accepts_string_array_scalar_input() {
    let array = StringArray::new(vec!["[1,2]".to_string()], vec![1, 1]).expect("string scalar");
    let result = block_on(jsondecode_builtin(Value::StringArray(array))).expect("jsondecode");
    match result {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[2]);
            assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0]);
        }
        other => panic!("expected tensor, got {:?}", other),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsondecode_round_trip_with_jsonencode() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).expect("tensor");
    let encoded = block_on(crate::call_builtin_async(
        "jsonencode",
        &[Value::Tensor(tensor.clone())],
    ))
    .expect("encode");
    let decoded = block_on(jsondecode_builtin(encoded.value().clone())).expect("decode");
    match decoded {
        Value::Tensor(result) => {
            assert_eq!(
                result.materialize_f64().len(),
                tensor.materialize_f64().len()
            );
            assert_eq!(result.materialize_f64(), tensor.materialize_f64());
        }
        other => panic!("expected tensor, got {:?}", other),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsondecode_integer_audit_is_explicitly_inapplicable() {
    assert_eq!(
        JSONDECODE_INTEGER_AUDIT.kind,
        BuiltinIntegerAuditKind::NotApplicable
    );
    assert!(JSONDECODE_INTEGER_AUDIT.canonical_builtin.is_none());
    assert!(JSONDECODE_INTEGER_AUDIT.notes.contains("text only"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn jsondecode_rejects_integer_and_resident_numeric_before_provider_access() {
    let integer = block_on(jsondecode_builtin(Value::Int(IntValue::U64(u64::MAX))))
        .expect_err("integer input must reject");
    assert_eq!(integer.message(), JSONDECODE_ERROR_INPUT_TYPE.message);

    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("genuine unregistered allocation");
    let original = allocation.handle.wire_descriptor();
    assert!(
        crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
            .is_none()
    );
    let resident = Value::GpuTensor(allocation.handle.clone());
    let error = block_on(jsondecode_builtin(resident))
        .expect_err("resident numeric input must reject before owner lookup");
    assert_eq!(error.message(), JSONDECODE_ERROR_INPUT_TYPE.message);
    assert_eq!(allocation.handle.wire_descriptor(), original);
    assert_eq!(allocation.probe.retirements(), 0);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}
