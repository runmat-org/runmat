use super::*;
use futures::executor::block_on;
use runmat_value::{IntegerStorage, NumericDType, NumericScalar, Value};

use crate::builtins::table::{table_from_columns, table_variables};

fn call(value: Value) -> BuiltinResult<Value> {
    block_on(macd_builtin(None, value))
}

fn call_with_mode(value: Value, extensions_enabled: bool) -> BuiltinResult<Value> {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(extensions_enabled);
    block_on(macd_builtin(None, value))
}

fn matrix(rows: usize, columns: [[f64; 4]; 5]) -> Value {
    let mut data = Vec::with_capacity(rows * 4);
    for col in 0..4 {
        for row_values in columns.iter().take(rows) {
            data.push(row_values[col]);
        }
    }
    Value::Tensor(Tensor::new(data, vec![rows, 4]).unwrap())
}

fn integer_matrix(rows: usize, columns: [[i16; 4]; 5]) -> Value {
    let mut data = Vec::with_capacity(rows * 4);
    for col in 0..4 {
        for row_values in columns.iter().take(rows) {
            data.push(row_values[col]);
        }
    }
    let tensor =
        Tensor::new_integer(IntegerStorage::I16(data), vec![rows, 4]).expect("integer matrix");
    Value::Tensor(tensor)
}

fn expect_tensor(value: Value) -> Tensor {
    match value {
        Value::Tensor(tensor) => tensor,
        other => panic!("expected tensor, got {other:?}"),
    }
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn matrix_input_returns_macd_line_by_default() {
    let out = expect_tensor(
        call(matrix(
            5,
            [
                [2.0, 0.5, 1.0, 1.0],
                [3.0, 1.5, 2.0, 2.0],
                [4.0, 2.5, 3.0, 4.0],
                [5.0, 3.5, 4.0, 8.0],
                [6.0, 4.5, 5.0, 16.0],
            ],
        ))
        .unwrap(),
    );
    assert_eq!(out.shape(), vec![5, 1]);
    assert_close(out.materialize_f64()[0], 0.0);
    assert_close(out.materialize_f64()[1], 0.075);
    assert_close(out.materialize_f64()[2], 0.283125);
    assert_close(out.materialize_f64()[3], 0.743578125);
    assert_close(out.materialize_f64()[4], 1.697244140625);
}

#[test]
fn matrix_input_reads_typed_integer_storage_exactly_as_double() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = expect_tensor(
        call(integer_matrix(
            5,
            [
                [2, 1, 1, 1],
                [3, 2, 2, 2],
                [4, 3, 3, 4],
                [5, 4, 4, 8],
                [6, 5, 5, 16],
            ],
        ))
        .unwrap(),
    );
    assert_eq!(out.shape(), vec![5, 1]);
    assert!(out.integer_storage().is_none());
    assert_close(out.materialize_f64()[0], 0.0);
    assert_close(out.materialize_f64()[1], 0.075);
    assert_close(out.materialize_f64()[2], 0.283125);
    assert_close(out.materialize_f64()[3], 0.743578125);
    assert_close(out.materialize_f64()[4], 1.697244140625);
}

#[test]
fn raw_nondouble_matrix_follows_compatibility_mode() {
    let integer = || integer_matrix(1, [[2, 1, 1, 1], [0; 4], [0; 4], [0; 4], [0; 4]]);
    let error =
        call_with_mode(integer(), false).expect_err("MATLAB mode rejects raw integer macd matrix");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:MacdNondoubleMatrixExtension")
    );
    call_with_mode(integer(), true).expect("RunMat mode accepts raw integer macd matrix");

    let single = Tensor::from_f32(vec![2.0, 1.0, 1.0, 1.0], vec![1, 4]).expect("single matrix");
    let error = call_with_mode(Value::Tensor(single), false)
        .expect_err("MATLAB mode rejects raw single macd matrix");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:MacdNondoubleMatrixExtension")
    );
}

#[test]
fn integer_prices_reject_before_a_lossy_binary64_boundary() {
    let wide = (1_u64 << 53) + 1;
    let matrix = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![wide; 4]), vec![1, 4]).expect("wide matrix"),
    );
    let error = call_with_mode(matrix, true).expect_err("lossy raw price must reject");
    assert!(error.message().contains("exactly representable"));

    let table = table_from_columns(
        vec![
            "High".to_string(),
            "Low".to_string(),
            "Open".to_string(),
            "Close".to_string(),
        ],
        vec![
            Value::Num(2.0),
            Value::Num(1.0),
            Value::Num(1.0),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1])
                    .expect("wide close"),
            ),
        ],
    )
    .expect("table");
    let error = call(table).expect_err("lossy table price must reject");
    assert!(error.message().contains("exactly representable"));
}

#[test]
fn macd_integer_capabilities_separate_matrix_extension_and_table_data() {
    assert_eq!(MACD_INTEGER_CAPABILITIES.len(), 2);
    assert_eq!(
        MACD_INTEGER_CAPABILITIES[0].inputs[0].availability,
        BuiltinIntegerInputAvailability::RunMatOnly
    );
    assert_eq!(
        MACD_INTEGER_CAPABILITIES[1].inputs[0].availability,
        BuiltinIntegerInputAvailability::Documented
    );
}

#[test]
fn resident_nondouble_matrix_uses_the_same_compatibility_gate() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let double = Tensor::new(vec![2.0, 1.0, 1.0, 1.0], vec![1, 4]).expect("matrix");
        let double_handle =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &double).expect("upload");
        call_with_mode(Value::GpuTensor(double_handle), false)
            .expect("provider precision does not change documented double class");

        let upload = || {
            let matrix = Tensor::from_f32(vec![2.0, 1.0, 1.0, 1.0], vec![1, 4]).expect("matrix");
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &matrix).expect("upload")
        };
        let error = call_with_mode(Value::GpuTensor(upload()), false)
            .expect_err("MATLAB mode rejects resident single matrix");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:MacdNondoubleMatrixExtension")
        );
        call_with_mode(Value::GpuTensor(upload()), true)
            .expect("RunMat mode accepts resident single matrix");
    });
}

#[test]
fn resident_integer_matrix_is_downloaded_once_and_checked_after_gather() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let exact = Tensor::new_integer(IntegerStorage::U64(vec![8, 7, 6, 5]), vec![1, 4])
            .expect("integer matrix");
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &exact)
            .expect("integer upload");
        provider.reset_telemetry();
        let output = call_with_mode(Value::GpuTensor(handle), true).expect("exact integer MACD");
        assert_eq!(expect_tensor(output).shape(), vec![1, 1]);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 4 * 8);

        let lossy =
            Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1; 4]), vec![1, 4])
                .expect("wide matrix");
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &lossy)
            .expect("wide upload");
        provider.reset_telemetry();
        let failure = call_with_mode(Value::GpuTensor(handle), true)
            .expect_err("lossy resident integer must reject");
        assert_eq!(failure.identifier(), Some("RunMat:macd:InvalidInput"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 4 * 8);
    });
}

#[test]
fn tensor_shape_for_reads_scalar_typed_integer_storage_without_mirror() {
    let input = Tensor::new_integer(IntegerStorage::U8(vec![7]), Vec::new()).unwrap();

    assert_eq!(tensor_shape_for(&input), vec![1, 1]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn rank_one_macd_input_keeps_its_full_shape() {
    let input = Tensor::from_f32(vec![7.0, 8.0], vec![2]).unwrap();
    assert_eq!(tensor_shape_for(&input), vec![2]);
}

#[test]
fn two_outputs_return_macd_and_signal() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let result = call(matrix(
        5,
        [
            [2.0, 0.5, 1.0, 1.0],
            [3.0, 1.5, 2.0, 2.0],
            [4.0, 2.5, 3.0, 4.0],
            [5.0, 3.5, 4.0, 8.0],
            [6.0, 4.5, 5.0, 16.0],
        ],
    ))
    .unwrap();
    let Value::OutputList(values) = result else {
        panic!("expected outputs");
    };
    assert_eq!(values.len(), 2);
    let macd = expect_tensor(values[0].clone());
    let signal = expect_tensor(values[1].clone());
    assert_eq!(macd.shape(), vec![5, 1]);
    assert_eq!(signal.shape(), vec![5, 1]);
    assert_close(signal.materialize_f64()[0], 0.0);
    assert_close(signal.materialize_f64()[1], 0.015);
    assert_close(signal.materialize_f64()[2], 0.068625);
    assert_close(signal.materialize_f64()[3], 0.203615625);
    assert_close(signal.materialize_f64()[4], 0.502341328125);
}

#[test]
fn zero_outputs_return_empty_without_validating_input() {
    let _guard = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let result = call(Value::String("not data".to_string().into())).unwrap();
    assert_eq!(result, Value::output_list(Vec::new()));
}

#[test]
fn rejects_more_than_two_outputs_before_evaluation() {
    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let err = call(matrix(
        1,
        [[2.0, 0.5, 1.0, 1.0], [0.0; 4], [0.0; 4], [0.0; 4], [0.0; 4]],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:macd:OutputCount"));
}

#[test]
fn empty_matrix_returns_empty_series() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let result = call(Value::Tensor(Tensor::new(vec![], vec![0, 4]).unwrap())).unwrap();
    let Value::OutputList(values) = result else {
        panic!("expected outputs");
    };
    let macd = expect_tensor(values[0].clone());
    let signal = expect_tensor(values[1].clone());
    assert_eq!(macd.shape(), vec![0, 1]);
    assert_eq!(signal.shape(), vec![0, 1]);
    assert!(macd.materialize_f64().is_empty());
    assert!(signal.materialize_f64().is_empty());
}

#[test]
fn nan_prices_propagate_through_ema_recurrence() {
    let out = expect_tensor(
        call(matrix(
            4,
            [
                [2.0, 0.5, 1.0, 1.0],
                [3.0, 1.5, 2.0, f64::NAN],
                [4.0, 2.5, 3.0, 4.0],
                [5.0, 3.5, 4.0, 8.0],
                [0.0; 4],
            ],
        ))
        .unwrap(),
    );
    assert_eq!(out.shape(), vec![4, 1]);
    assert_eq!(out.materialize_f64()[0], 0.0);
    assert!(out.materialize_f64()[1].is_nan());
    assert!(out.materialize_f64()[2].is_nan());
    assert!(out.materialize_f64()[3].is_nan());
}

#[test]
fn table_input_is_case_insensitive_and_returns_table_with_close() {
    let table = table_from_columns(
        vec![
            "high".to_string(),
            "LOW".to_string(),
            "Open".to_string(),
            "close".to_string(),
        ],
        vec![
            Value::Tensor(Tensor::new(vec![2.0, 3.0, 4.0], vec![3, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.5, 1.5, 2.5], vec![3, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 4.0], vec![3, 1]).unwrap()),
        ],
    )
    .unwrap();
    let out = call(table).unwrap();
    let Value::Object(object) = out else {
        panic!("expected table output");
    };
    assert_eq!(object.class_name().display_name(), "table");
    let variables = table_variables(&object).unwrap();
    assert_eq!(variables.fields.len(), 1);
    let close = expect_tensor(variables.fields.get("Close").cloned().unwrap());
    assert_eq!(close.shape(), vec![3, 1]);
    assert_close(close.materialize_f64()[1], 0.075);
    assert_close(close.materialize_f64()[2], 0.283125);
}

#[test]
fn table_input_preserves_native_single_close_class() {
    let single_column = |values: Vec<f64>| {
        Value::Tensor(
            Tensor::new_with_dtype(values, vec![3, 1], NumericDType::F32)
                .expect("single table variable"),
        )
    };
    let table = table_from_columns(
        vec![
            "High".to_string(),
            "Low".to_string(),
            "Open".to_string(),
            "Close".to_string(),
        ],
        vec![
            single_column(vec![2.0, 3.0, 4.0]),
            single_column(vec![0.5, 1.5, 2.5]),
            single_column(vec![1.0, 2.0, 3.0]),
            single_column(vec![1.0, 2.0, 4.0]),
        ],
    )
    .expect("single table");

    let Value::Object(object) = call(table).expect("macd") else {
        panic!("expected table output");
    };
    let variables = table_variables(&object).expect("output variables");
    let close = expect_tensor(variables.fields.get("Close").cloned().expect("Close"));
    assert_eq!(close.numeric_dtype(), NumericDType::F32);
    for (index, expected) in [0.0_f32, 0.075, 0.283125].into_iter().enumerate() {
        assert_eq!(
            close.numeric_value_at(index),
            Some(NumericScalar::F32(expected))
        );
    }
}

#[test]
fn rejects_matrix_without_four_columns() {
    let err = call(Value::Tensor(
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap(),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:macd:InvalidInput"));
}

#[test]
fn rejects_higher_rank_matrix_input() {
    let err = call(Value::Tensor(
        Tensor::new((1..=16).map(|value| value as f64).collect(), vec![2, 4, 2]).unwrap(),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:macd:InvalidInput"));
}

#[test]
fn rejects_table_missing_required_price_variable() {
    let table = table_from_columns(
        vec!["High".to_string(), "Low".to_string(), "Open".to_string()],
        vec![
            Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.5, 1.5], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
        ],
    )
    .unwrap();
    let err = call(table).unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:macd:InvalidInput"));
}

#[test]
fn rejects_nonnumeric_price_variables() {
    let table = table_from_columns(
        vec![
            "High".to_string(),
            "Low".to_string(),
            "Open".to_string(),
            "Close".to_string(),
        ],
        vec![
            Value::String("not numeric".to_string().into()),
            Value::Tensor(Tensor::new(vec![0.5], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![2.0], vec![1, 1]).unwrap()),
        ],
    )
    .unwrap();
    let err = call(table).unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:macd:InvalidInput"));
}

#[test]
fn rejects_multicolumn_price_variables() {
    let table = table_from_columns(
        vec![
            "High".to_string(),
            "Low".to_string(),
            "Open".to_string(),
            "Close".to_string(),
        ],
        vec![
            Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.5], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![2.0], vec![1, 1]).unwrap()),
        ],
    )
    .unwrap();
    let err = call(table).unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:macd:InvalidInput"));
}
