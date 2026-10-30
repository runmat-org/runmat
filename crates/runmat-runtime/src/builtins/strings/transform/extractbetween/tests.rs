#![allow(non_snake_case)]

use super::*;
use runmat_value::{CellArray, IntegerStorage, NumericStorage, StructArray, Tensor};

fn extract_between_builtin(
    text: Value,
    start: Value,
    stop: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    futures::executor::block_on(super::extract_between_builtin(
        text,
        start,
        stop,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
}

#[test]
fn extractBetween_position_vectors_read_typed_integer_storage_exactly() {
    for storage in [
        IntegerStorage::I8(vec![2, 4]),
        IntegerStorage::I16(vec![2, 4]),
        IntegerStorage::I32(vec![2, 4]),
        IntegerStorage::I64(vec![2, 4]),
        IntegerStorage::U8(vec![2, 4]),
        IntegerStorage::U16(vec![2, 4]),
        IntegerStorage::U32(vec![2, 4]),
        IntegerStorage::U64(vec![2, 4]),
    ] {
        let positions = Tensor::new_integer(storage, vec![1, 2]).expect("positions");
        let parsed = BoundaryPositions::from_value(Value::Tensor(positions)).unwrap();
        assert_eq!(parsed.data, vec![2, 4]);
        assert_eq!(parsed.shape, vec![1, 2]);
    }
}

#[test]
fn extractBetween_position_vectors_read_native_single_storage() {
    let positions = Tensor::from_numeric_storage(NumericStorage::F32(vec![2.0, 4.0]), vec![1, 2])
        .expect("positions");
    let parsed = BoundaryPositions::from_value(Value::Tensor(positions)).unwrap();
    assert_eq!(parsed.data, vec![2, 4]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn extractBetween_rank_zero_position_retains_boundary_domain_shape() {
    let positions = || Tensor::new(vec![2.0], Vec::new()).expect("rank-zero position");
    let parsed = BoundaryPositions::from_value(Value::Tensor(positions())).expect("positions");

    assert_eq!(parsed.data, vec![2]);
    assert_eq!(parsed.shape, vec![0, 1]);

    let text = StringArray::new(Vec::new(), vec![0, 1]).expect("empty text array");
    let output = extract_between_builtin(
        Value::StringArray(text),
        Value::Tensor(positions()),
        Value::Tensor(positions()),
        Vec::new(),
    )
    .expect("extractBetween rank-zero positions");
    let Value::StringArray(output) = output else {
        panic!("expected string array");
    };
    assert_eq!(output.shape, vec![0, 1]);
}

#[test]
fn extractBetween_position_vectors_reject_nonpositive_integer_storage() {
    let positions =
        Tensor::new_integer(IntegerStorage::I16(vec![1, 0]), vec![1, 2]).expect("positions");

    assert!(BoundaryPositions::from_value(Value::Tensor(positions)).is_err());
}

#[test]
fn extractBetween_position_vectors_reject_oversized_double_positions() {
    assert!(BoundaryPositions::from_value(Value::Num(1.0e300)).is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn extractBetween_basic_string() {
    let result = extract_between_builtin(
        Value::String("RunMat accelerates MATLAB".into()),
        Value::String("RunMat ".into()),
        Value::String(" MATLAB".into()),
        Vec::new(),
    )
    .expect("extractBetween");
    assert_eq!(result, Value::String("accelerates".into()));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn extractBetween_inclusive_option() {
    let result = extract_between_builtin(
        Value::String("a[b]c".into()),
        Value::String("[".into()),
        Value::String("]".into()),
        vec![
            Value::String("Boundaries".into()),
            Value::String("inclusive".into()),
        ],
    )
    .expect("extractBetween");
    assert_eq!(result, Value::String("[b]".into()));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn extractBetween_numeric_positions() {
    let result = extract_between_builtin(
        Value::String("Accelerator".into()),
        Value::Num(3.0),
        Value::Num(7.0),
        Vec::new(),
    )
    .expect("extractBetween");
    assert_eq!(result, Value::String("celer".into()));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn extractBetween_numeric_positions_exclusive_option() {
    let result = extract_between_builtin(
        Value::String("Accelerator".into()),
        Value::Num(3.0),
        Value::Num(7.0),
        vec![
            Value::String("Boundaries".into()),
            Value::String("exclusive".into()),
        ],
    )
    .expect("extractBetween");
    assert_eq!(result, Value::String("ele".into()));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn extractBetween_numeric_positions_clamps_stop() {
    let result = extract_between_builtin(
        Value::String("Accelerator".into()),
        Value::Num(3.0),
        Value::Num(100.0),
        Vec::new(),
    )
    .expect("extractBetween");
    assert_eq!(result, Value::String("celerator".into()));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn extractBetween_numeric_positions_start_past_length() {
    let result = extract_between_builtin(
        Value::String("abc".into()),
        Value::Num(10.0),
        Value::Num(12.0),
        Vec::new(),
    )
    .expect("extractBetween");
    assert_eq!(result, Value::String(String::new().into()));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn extractBetween_string_array_broadcast() {
    let array = StringArray::new(
        vec!["runmat_accel.rs".into(), "runmat_gc.rs".into()],
        vec![2, 1],
    )
    .unwrap();
    let result = extract_between_builtin(
        Value::StringArray(array),
        Value::String("runmat_".into()),
        Value::String(".rs".into()),
        Vec::new(),
    )
    .expect("extractBetween");
    match result {
        Value::StringArray(sa) => {
            assert_eq!(sa.data(), vec!["accel".to_string(), "gc".to_string()]);
            assert_eq!(sa.shape, vec![2, 1]);
        }
        other => panic!("expected string array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn extractBetween_character_vector_returns_scalar_cellstr() {
    let chars = CharArray::from_unicode_scalars(
        "GPUAccelerateVM".chars().collect(),
        1,
        "GPUAccelerateVM".len(),
    )
    .unwrap();
    let result = extract_between_builtin(
        Value::CharArray(chars),
        Value::String("GPU".into()),
        Value::String("VM".into()),
        Vec::new(),
    )
    .expect("extractBetween");
    match result {
        Value::Cell(out) => {
            assert_eq!(out.shape, vec![1, 1]);
            let Value::CharArray(chars) = &out.data()[0] else {
                panic!("expected character vector element")
            };
            let text: String =
                runmat_value::decode_utf16_code_units(chars.data()).expect("valid UTF-16 fixture");
            assert_eq!(text.trim_end(), "Accelerate");
        }
        other => panic!("expected scalar cellstr, got {other:?}"),
    }
}

#[test]
fn extractBetween_integer_capabilities_and_strict_shape_are_declared() {
    assert_eq!(EXTRACT_BETWEEN_INTEGER_CAPABILITIES.len(), 2);
    assert_eq!(
        EXTRACT_BETWEEN_INTEGER_CAPABILITIES[0].inputs[0]
            .classes
            .len(),
        8
    );
    let text = Value::StringArray(
        StringArray::new(vec!["abcd".into(), "wxyz".into()], vec![2, 1]).unwrap(),
    );
    let start =
        Value::Tensor(Tensor::new_integer(IntegerStorage::U64(vec![1, 2]), vec![1, 2]).unwrap());
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = extract_between_builtin(text, start, Value::Num(3.0), Vec::new())
        .expect_err("non-same-size expansion must be gated");
    assert_eq!(
        error.identifier(),
        FULL_BROADCAST_EXTENSION.error_identifier
    );
}

#[test]
fn extractBetween_strict_mode_gates_resident_positions_before_gather() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let resident = Value::GpuTensor(allocation.handle.clone());
    let error = extract_between_builtin(
        Value::String("abcd".into()),
        resident,
        Value::Num(3.0),
        Vec::new(),
    )
    .expect_err("resident position gate");
    assert_eq!(
        error.identifier(),
        RESIDENT_POSITION_EXTENSION.error_identifier
    );
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
    assert_eq!(allocation.probe.retirements(), 0);
}

#[test]
fn extractBetween_rejects_malformed_text_before_resident_boundary_access() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("resident fixture");
    let malformed =
        Value::Cell(CellArray::new(vec![Value::Int(IntValue::U8(7))], 1, 1).expect("cell"));
    let error = extract_between_builtin(
        malformed,
        Value::GpuTensor(allocation.handle.clone()),
        Value::Num(2.0),
        Vec::new(),
    )
    .expect_err("malformed source must reject before gather");
    assert_eq!(
        error.identifier(),
        EXTRACT_BETWEEN_ERROR_INVALID_INPUT.identifier
    );
    assert_eq!(allocation.probe.retirements(), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn extractBetween_cell_array_preserves_types() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let cell = CellArray::new(
        vec![
            Value::CharArray(CharArray::new_row("A[B]C")),
            Value::String("Planner<GPU>".into()),
        ],
        1,
        2,
    )
    .unwrap();
    let result = extract_between_builtin(
        Value::Cell(cell),
        Value::String("[".into()),
        Value::String("]".into()),
        Vec::new(),
    )
    .expect("extractBetween");
    match result {
        Value::Cell(out) => {
            let first = out.get(0, 0).unwrap();
            let second = out.get(0, 1).unwrap();
            assert_eq!(first, Value::CharArray(CharArray::new_row("B")));
            assert_eq!(second, Value::String(String::new().into()));
        }
        other => panic!("expected cell array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn extractBetween_cell_and_structure_identity_does_not_depend_on_contents() {
    let empty_cell = Value::Cell(CellArray::new(Vec::new(), 0, 0).unwrap());
    let result = extract_between_builtin(
        empty_cell,
        Value::String("<".into()),
        Value::String(">".into()),
        Vec::new(),
    )
    .expect("empty cell remains a cell text container");
    assert!(matches!(result, Value::Cell(cell) if cell.shape == [0, 0]));

    let empty_structure = Value::StructArray(StructArray::empty(Vec::new(), vec![0, 0]).unwrap());
    let error = extract_between_builtin(
        empty_structure,
        Value::String("<".into()),
        Value::String(">".into()),
        Vec::new(),
    )
    .expect_err("typed structure array is not a text cell, even when empty");
    assert_eq!(
        error.identifier(),
        EXTRACT_BETWEEN_ERROR_INVALID_INPUT.identifier
    );

    let mut structure = runmat_value::UnboundedStructValueBuilder::new();
    structure.insert("payload", Value::CharArray(CharArray::new_row("A<B>")));
    let structure_cell =
        Value::Cell(CellArray::new(vec![Value::Struct(structure.finish())], 1, 1).unwrap());
    let error = extract_between_builtin(
        structure_cell.clone(),
        Value::String("<".into()),
        Value::String(">".into()),
        Vec::new(),
    )
    .expect_err("a cell containing a structure is still a cell with an invalid text element");
    assert_eq!(
        error.identifier(),
        EXTRACT_BETWEEN_ERROR_INVALID_INPUT.identifier
    );

    let error = extract_between_builtin(
        Value::String("A<B>".into()),
        structure_cell,
        Value::String(">".into()),
        Vec::new(),
    )
    .expect_err("a structure in a boundary cell is an invalid cell element");
    assert_eq!(
        error.identifier(),
        EXTRACT_BETWEEN_ERROR_CELL_ELEMENT.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn extractBetween_missing_string_propagates() {
    let strings = StringArray::from_elements_unbounded(
        vec![runmat_value::StringElement::Missing],
        vec![1, 1],
    )
    .unwrap();
    let result = extract_between_builtin(
        Value::StringArray(strings),
        Value::String("[".into()),
        Value::String("]".into()),
        Vec::new(),
    )
    .expect("extractBetween");
    assert_eq!(
        result,
        Value::StringArray(
            StringArray::from_elements_unbounded(
                vec![runmat_value::StringElement::Missing],
                vec![1, 1]
            )
            .unwrap()
        )
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn extractBetween_position_type_error() {
    let err = extract_between_builtin(
        Value::String("abc".into()),
        Value::Num(0.5),
        Value::Num(2.0),
        Vec::new(),
    )
    .unwrap_err();
    assert_eq!(err.to_string(), EXTRACT_BETWEEN_ERROR_POSITION_TYPE.message);
    assert_eq!(
        err.identifier(),
        EXTRACT_BETWEEN_ERROR_POSITION_TYPE.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn extractBetween_mixed_boundary_error() {
    let err = extract_between_builtin(
        Value::String("abc".into()),
        Value::String("a".into()),
        Value::Num(3.0),
        Vec::new(),
    )
    .unwrap_err();
    assert_eq!(err.to_string(), EXTRACT_BETWEEN_ERROR_BOUNDARY_TYPE.message);
    assert_eq!(
        err.identifier(),
        EXTRACT_BETWEEN_ERROR_BOUNDARY_TYPE.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn extractBetween_numeric_tensor_broadcast() {
    let text = StringArray::new(vec!["abcd".into(), "wxyz".into()], vec![2, 1]).unwrap();
    let start = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let stop = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
    let result = extract_between_builtin(
        Value::StringArray(text),
        Value::Tensor(start),
        Value::Tensor(stop),
        Vec::new(),
    )
    .expect("extractBetween");
    match result {
        Value::StringArray(sa) => {
            assert_eq!(sa.data(), vec!["abc".to_string(), "xyz".to_string()]);
            assert_eq!(sa.shape, vec![2, 1]);
        }
        other => panic!("expected string array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn extractBetween_option_invalid_value() {
    let err = extract_between_builtin(
        Value::String("abc".into()),
        Value::String("a".into()),
        Value::String("c".into()),
        vec![
            Value::String("Boundaries".into()),
            Value::String("middle".into()),
        ],
    )
    .unwrap_err();
    assert_eq!(err.to_string(), EXTRACT_BETWEEN_ERROR_OPTION_VALUE.message);
    assert_eq!(
        err.identifier(),
        EXTRACT_BETWEEN_ERROR_OPTION_VALUE.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn extractBetween_option_name_error() {
    let err = extract_between_builtin(
        Value::String("abc".into()),
        Value::String("a".into()),
        Value::String("c".into()),
        vec![
            Value::String("Padding".into()),
            Value::String("inclusive".into()),
        ],
    )
    .unwrap_err();
    assert_eq!(err.to_string(), EXTRACT_BETWEEN_ERROR_OPTION_NAME.message);
    assert_eq!(
        err.identifier(),
        EXTRACT_BETWEEN_ERROR_OPTION_NAME.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn extractBetween_option_pair_error() {
    let err = extract_between_builtin(
        Value::String("abc".into()),
        Value::String("a".into()),
        Value::String("b".into()),
        vec![Value::String("Boundaries".into())],
    )
    .unwrap_err();
    assert_eq!(
        err.to_string(),
        EXTRACT_BETWEEN_ERROR_NAME_VALUE_PAIR.message
    );
    assert_eq!(
        err.identifier(),
        EXTRACT_BETWEEN_ERROR_NAME_VALUE_PAIR.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn extractBetween_missing_boundary_propagates() {
    let result = extract_between_builtin(
        Value::String("Planner<GPU>".into()),
        Value::String(runmat_value::StringElement::Missing),
        Value::String(">".into()),
        Vec::new(),
    )
    .expect("extractBetween");
    assert_eq!(result, Value::String(runmat_value::StringElement::Missing));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn extractBetween_cell_boundary_arguments() {
    let text = CellArray::new(vec![Value::CharArray(CharArray::new_row("A<GPU>"))], 1, 1).unwrap();
    let start = CellArray::new(vec![Value::CharArray(CharArray::new_row("<"))], 1, 1).unwrap();
    let stop = CellArray::new(vec![Value::CharArray(CharArray::new_row(">"))], 1, 1).unwrap();
    let result = extract_between_builtin(
        Value::Cell(text),
        Value::Cell(start),
        Value::Cell(stop),
        Vec::new(),
    )
    .expect("extractBetween");
    match result {
        Value::Cell(out) => {
            let value = out.get(0, 0).unwrap();
            assert_eq!(value, Value::CharArray(CharArray::new_row("GPU")));
        }
        other => panic!("expected cell array, got {other:?}"),
    }
}
