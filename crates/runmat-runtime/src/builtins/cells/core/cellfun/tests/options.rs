use super::*;
use runmat_builtins::{CELLFUN_ERROR_INVALID_INPUT, CELLFUN_ERROR_UNIFORM_OUTPUT};
use runmat_value::{IntegerStorage, LogicalArray};

#[test]
fn padded_names_and_first_duplicate_control_collection() {
    let output = call(
        Value::FunctionHandle("abs".into()),
        vec![
            cell(vec![Value::Num(-1.0)], &[1, 1]),
            Value::String(" UniformOutput ".into()),
            Value::Bool(false),
            Value::String("UniformOutput".into()),
            Value::Bool(true),
        ],
    )
    .unwrap();
    let Value::Cell(output) = output.value() else {
        panic!("first option must select cell collection")
    };
    assert_eq!(output.to_column_major(), vec![Value::Num(1.0)]);
}

#[test]
fn uniform_output_accepts_logical_and_legacy_double_controls() {
    for (control, expected_cell) in [
        (Value::Bool(true), false),
        (Value::Bool(false), true),
        (Value::Num(1.0), false),
        (Value::Num(0.0), true),
        (
            Value::LogicalArray(LogicalArray::new(vec![1], vec![1, 1]).unwrap()),
            false,
        ),
    ] {
        let output = call(
            Value::FunctionHandle("sin".into()),
            vec![
                cell(vec![Value::Num(0.0)], &[1, 1]),
                Value::String("UniformOutput".into()),
                control,
            ],
        )
        .unwrap();
        assert_eq!(matches!(output.value(), Value::Cell(_)), expected_cell);
    }
}

#[test]
fn uniform_output_rejects_every_typed_integer_class() {
    for storage in [
        IntegerStorage::I8(vec![1]),
        IntegerStorage::I16(vec![1]),
        IntegerStorage::I32(vec![1]),
        IntegerStorage::I64(vec![1]),
        IntegerStorage::U8(vec![1]),
        IntegerStorage::U16(vec![1]),
        IntegerStorage::U32(vec![1]),
        IntegerStorage::U64(vec![1]),
    ] {
        let error = call(
            Value::FunctionHandle("sin".into()),
            vec![
                cell(vec![Value::Num(0.0)], &[1, 1]),
                Value::String("UniformOutput".into()),
                Value::Int(storage.value_at(0).unwrap()),
            ],
        )
        .unwrap_err();
        assert_eq!(error.identifier(), CELLFUN_ERROR_UNIFORM_OUTPUT.identifier);
    }
}

#[test]
fn rejects_unknown_name_value_options() {
    let error = call(
        Value::FunctionHandle("sin".into()),
        vec![
            cell(vec![Value::Num(0.0)], &[1, 1]),
            Value::String("Mystery".into()),
            Value::Bool(true),
        ],
    )
    .unwrap_err();
    assert_eq!(error.identifier(), CELLFUN_ERROR_INVALID_INPUT.identifier);
}
