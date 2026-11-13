use super::*;
use crate::builtins::common::test_support::builtin_arguments;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn stored_seconds_preserve_empty_shape_and_signed_calendar_fraction() {
    let empty =
        datetime_object_from_serials(None, Vec::new(), vec![0, 2, 1], "yyyy-MM-dd").unwrap();
    let extracted =
        futures::executor::block_on(super::super::second_builtin::second_builtin(None, empty))
            .unwrap();
    assert!(
        matches!(extracted, Value::Tensor(tensor) if tensor.shape() == [0, 2, 1] && tensor.is_empty())
    );

    let instant = NaiveDate::from_ymd_opt(1960, 2, 29)
        .unwrap()
        .and_hms_milli_opt(12, 30, 5, 250)
        .unwrap();
    let stored = datetime_object_from_serials(
        None,
        vec![datenum_from_naive(instant)],
        vec![1, 1],
        "yyyy-MM-dd HH:mm:ss",
    )
    .unwrap();
    let extracted =
        futures::executor::block_on(super::super::second_builtin::second_builtin(None, stored))
            .unwrap();
    assert!(matches!(extracted, Value::Num(value) if (value - 5.25).abs() < 1e-4));
    let _outputs = crate::output_count::push_output_count(Some(2)).unwrap();
    let error = futures::executor::block_on(super::super::second_builtin::second_builtin(
        None,
        Value::Num(0.0),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        runmat_builtins::SECOND_ERROR_TOO_MANY_OUTPUTS.identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dateshift_scalar_expansion_preserves_empty_geometry_and_format() {
    for shape in [vec![0, 2], vec![2, 0], vec![0, 2, 1]] {
        let empty =
            datetime_object_from_serials(None, Vec::new(), shape.clone(), "yyyy-MM-dd").unwrap();
        let output = futures::executor::block_on(dateshift_builtin(
            None,
            empty,
            Value::from("start"),
            Value::from("month"),
            builtin_arguments(Vec::new()),
        ))
        .unwrap();
        assert_eq!(serials_from_datetime_value(&output).unwrap().shape(), shape);
        assert_eq!(datetime_format_from_value(&output).unwrap(), "yyyy-MM-dd");
    }
    let scalar = datetime_object_from_serials(
        None,
        vec![serial_for_date(2024, 2, 10)],
        vec![1, 1],
        "yyyy-MM-dd",
    )
    .unwrap();
    for day_target in [false, true] {
        let controls = integer_tensor(runmat_value::IntegerStorage::I64(Vec::new()), vec![0, 3]);
        let (boundary, unit, rest) = if day_target {
            (
                Value::from("dayofweek"),
                controls,
                builtin_arguments(Vec::new()),
            )
        } else {
            (
                Value::from("end"),
                Value::from("month"),
                builtin_arguments(vec![controls]),
            )
        };
        let output = futures::executor::block_on(dateshift_builtin(
            None,
            scalar.clone(),
            boundary,
            unit,
            rest,
        ))
        .unwrap();
        assert_eq!(
            serials_from_datetime_value(&output).unwrap().shape(),
            [0, 3]
        );
        assert_eq!(datetime_format_from_value(&output).unwrap(), "yyyy-MM-dd");
    }
    let empty = datetime_object_from_serials(None, Vec::new(), vec![0, 2], "yyyy-MM-dd").unwrap();
    let incompatible = integer_tensor(runmat_value::IntegerStorage::I64(Vec::new()), vec![0, 3]);
    let error = futures::executor::block_on(dateshift_builtin(
        None,
        empty,
        Value::from("start"),
        Value::from("month"),
        builtin_arguments(vec![incompatible]),
    ))
    .expect_err("empty geometry is not interchangeable");
    assert!(error.message().contains("matching sizes"));
    let _outputs = crate::output_count::push_output_count(Some(2)).unwrap();
    let error = futures::executor::block_on(dateshift_builtin(
        None,
        Value::Num(0.0),
        Value::from("start"),
        Value::from("day"),
        builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        runmat_builtins::DATESHIFT_ERROR_TOO_MANY_OUTPUTS.identifier
    );
}
