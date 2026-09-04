use runmat_builtins::POW2_INTEGER_UNARY_EXPONENT_EXTENSION;
use runmat_value::{IntegerComplexStorage, IntegerStorage, NumericStorage, Tensor, Value};

use super::call;

#[test]
fn unary_and_binary_integer_extensions_use_checked_double_boundaries() {
    let _mode = crate::compatibility::push_runmat_extensions_enabled(true);
    let exponent =
        Tensor::new_integer(IntegerStorage::I16(vec![-1, 0, 3]), vec![1, 3]).expect("exponent");
    let Value::Tensor(output) = call(Value::Tensor(exponent), vec![]).expect("unary") else {
        panic!("expected tensor");
    };
    assert_eq!(
        output.into_numeric_storage().expect("storage"),
        NumericStorage::F64(vec![0.5, 1.0, 8.0])
    );

    let significand =
        Tensor::new_integer(IntegerStorage::I32(vec![5]), vec![1, 1]).expect("significand");
    let exponent = Tensor::new_integer(IntegerStorage::U16(vec![3]), vec![1, 1]).expect("exponent");
    assert_eq!(
        call(Value::Tensor(significand), vec![Value::Tensor(exponent)]).expect("binary"),
        Value::Num(40.0)
    );
}

#[test]
fn compatibility_mode_rejects_integer_extensions() {
    let _mode = crate::compatibility::push_runmat_extensions_enabled(false);
    let exponent = Tensor::new_integer(IntegerStorage::I16(vec![3]), vec![1, 1]).expect("exponent");
    let error = call(Value::Tensor(exponent), vec![]).expect_err("extension must reject");
    assert_eq!(
        error.identifier(),
        POW2_INTEGER_UNARY_EXPONENT_EXTENSION.error_identifier
    );
}

#[test]
fn lossy_wide_values_and_complex_integers_reject() {
    let _mode = crate::compatibility::push_runmat_extensions_enabled(true);
    let wide = Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
        .expect("wide");
    let error = call(Value::Tensor(wide), vec![]).expect_err("lossy value must reject");
    assert!(error.message().contains("exactly representable as double"));

    let storage =
        IntegerComplexStorage::new(IntegerStorage::U64(vec![1]), IntegerStorage::U64(vec![1]))
            .expect("storage");
    let complex =
        runmat_value::ComplexTensor::new_integer(storage, vec![1, 1]).expect("complex integer");
    let error =
        call(Value::ComplexTensor(complex), vec![]).expect_err("complex integer must reject");
    assert!(error
        .message()
        .contains("complex numbers with integer types"));
}
