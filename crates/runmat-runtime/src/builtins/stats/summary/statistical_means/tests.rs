use super::{data, geomean, harmmean, options, rms, rmse};
use futures::executor::block_on;
use runmat_value::{IntValue, NumericDType, Tensor, Value};

fn tensor(values: Vec<f64>, shape: Vec<usize>, dtype: NumericDType) -> Value {
    Value::Tensor(Tensor::new_with_dtype(values, shape, dtype).unwrap())
}

#[test]
fn single_precision_is_preserved_by_all_four_reductions() {
    let input = tensor(vec![1.0, 4.0, 9.0, 16.0], vec![2, 2], NumericDType::F32);
    for output in [
        block_on(geomean::geomean_builtin(input.clone(), vec![])).unwrap(),
        block_on(harmmean::harmmean_builtin(input.clone(), vec![])).unwrap(),
        block_on(rms::rms_builtin(input.clone(), vec![])).unwrap(),
        block_on(rmse::rmse_builtin(input.clone(), input.clone(), vec![])).unwrap(),
    ] {
        let Value::Tensor(output) = output else {
            panic!("expected typed tensor result")
        };
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
    }
}

#[test]
fn control_and_weight_extensions_are_classified_from_typed_values() {
    assert!(data::integer_control(&Value::Int(IntValue::U64(u64::MAX))));
    assert!(data::exact_integer_weight(&Value::Int(IntValue::I8(1))));
    assert!(!data::integer_data(&Value::Num(1.0)));
}

#[test]
fn malformed_weighted_selector_is_rejected() {
    let result = block_on(options::parse(
        options::ReductionKind::RootMeanSquaredError,
        vec![Value::from("all"), Value::from("Weights"), Value::Num(1.0)],
    ));
    assert!(result.is_err());
}
