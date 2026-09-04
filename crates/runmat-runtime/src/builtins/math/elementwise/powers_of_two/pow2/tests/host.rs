use runmat_builtins::{
    POW2_ERROR_INVALID_ARGUMENT, POW2_ERROR_INVALID_INPUT, POW2_ERROR_SIZE_MISMATCH,
    POW2_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_value::{CharArray, ComplexStorage, ComplexTensor, NumericStorage, Tensor, Value};

use super::call;

#[test]
fn unary_scalar_tensor_and_special_values_follow_the_contract() {
    assert_eq!(
        call(Value::Num(3.0), vec![]).expect("scalar"),
        Value::Num(8.0)
    );
    let tensor = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], vec![2, 2]).expect("tensor");
    let Value::Tensor(output) = call(Value::Tensor(tensor), vec![]).expect("tensor pow2") else {
        panic!("expected tensor");
    };
    assert_eq!(output.shape, vec![2, 2]);
    assert_eq!(output.materialize_f64(), vec![0.5, 1.0, 2.0, 4.0]);
    assert!(
        matches!(call(Value::Num(f64::INFINITY), vec![]), Ok(Value::Num(value)) if value.is_infinite())
    );
    assert!(matches!(call(Value::Num(f64::NAN), vec![]), Ok(Value::Num(value)) if value.is_nan()));
}

#[test]
fn unary_preserves_single_and_complex_precision() {
    let single = Tensor::from_f32(vec![0.0, 3.0], vec![1, 2]).expect("single");
    let Value::Tensor(output) = call(Value::Tensor(single), vec![]).expect("single pow2") else {
        panic!("expected tensor");
    };
    assert_eq!(
        output.into_numeric_storage().expect("storage"),
        NumericStorage::F32(vec![1.0, 8.0])
    );

    let complex = ComplexTensor::from_f32(vec![(1.0, 0.5)], vec![1, 1]).expect("complex");
    let Value::ComplexTensor(output) =
        call(Value::ComplexTensor(complex), vec![]).expect("complex pow2")
    else {
        panic!("expected complex tensor");
    };
    assert!(matches!(output.complex_storage(), &ComplexStorage::F32(_)));

    let Value::Complex(real, imaginary) =
        call(Value::Complex(1.0, 2.0), vec![]).expect("complex scalar")
    else {
        panic!("expected complex scalar");
    };
    assert!(real.is_finite() && imaginary.is_finite());
    assert!((real.hypot(imaginary) - 2.0).abs() < 1e-12);
}

#[test]
fn empty_arrays_preserve_shape_and_precision() {
    let unary = Tensor::from_f32(Vec::new(), vec![0, 3]).expect("empty single");
    let Value::Tensor(output) = call(Value::Tensor(unary), vec![]).expect("unary empty") else {
        panic!("expected tensor");
    };
    assert_eq!(output.shape, vec![0, 3]);
    assert_eq!(
        output.into_numeric_storage().expect("storage"),
        NumericStorage::F32(Vec::new())
    );

    let significand =
        ComplexTensor::from_f32(Vec::new(), vec![0, 2]).expect("empty complex single");
    let exponent = Tensor::from_f32(Vec::new(), vec![0, 2]).expect("empty exponent");
    let Value::ComplexTensor(output) = call(
        Value::ComplexTensor(significand),
        vec![Value::Tensor(exponent)],
    )
    .expect("binary empty") else {
        panic!("expected complex tensor");
    };
    assert_eq!(output.shape, vec![0, 2]);
    assert!(matches!(output.complex_storage(), &ComplexStorage::F32(_)));
}

#[test]
fn binary_scaling_broadcasts_and_preserves_complex_single() {
    let significand = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).expect("significand");
    let exponent = Tensor::new(vec![2.0, 3.0], vec![1, 2]).expect("exponent");
    let Value::Tensor(output) =
        call(Value::Tensor(significand), vec![Value::Tensor(exponent)]).expect("broadcast")
    else {
        panic!("expected tensor");
    };
    assert_eq!(output.shape, vec![3, 2]);
    assert_eq!(
        output.materialize_f64(),
        vec![4.0, 8.0, 12.0, 8.0, 16.0, 24.0]
    );

    let significand = ComplexTensor::from_f32(vec![(1.0, 1.0)], vec![1, 1]).expect("complex");
    let exponent = Tensor::new(vec![2.0], vec![1, 1]).expect("exponent");
    let Value::ComplexTensor(output) = call(
        Value::ComplexTensor(significand),
        vec![Value::Tensor(exponent)],
    )
    .expect("complex scale") else {
        panic!("expected complex tensor");
    };
    assert_eq!(output.materialize_f64(), vec![(4.0, 4.0)]);
    assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F32);

    let significand = Tensor::new(vec![0.5, 1.5], vec![1, 2]).expect("double significand");
    let exponent = Tensor::from_f32(vec![3.0, 4.0], vec![1, 2]).expect("single exponent");
    let Value::Tensor(output) =
        call(Value::Tensor(significand), vec![Value::Tensor(exponent)]).expect("mixed precision")
    else {
        panic!("expected tensor");
    };
    assert_eq!(
        output.into_numeric_storage().expect("storage"),
        NumericStorage::F32(vec![4.0, 24.0])
    );
}

#[test]
fn character_and_logical_inputs_enter_the_double_domain() {
    let chars = CharArray::new("AB".chars().collect(), 1, 2).expect("chars");
    let Value::Tensor(output) = call(Value::CharArray(chars), vec![]).expect("character pow2")
    else {
        panic!("expected tensor");
    };
    assert_eq!(
        output.materialize_f64(),
        vec![65.0_f64.exp2(), 66.0_f64.exp2()]
    );
    assert_eq!(
        call(Value::Bool(true), vec![]).expect("logical"),
        Value::Num(2.0)
    );
}

#[test]
fn invalid_forms_report_stable_identifiers() {
    let error = call(Value::from("bad"), vec![]).expect_err("string must reject");
    assert_eq!(error.identifier(), POW2_ERROR_INVALID_INPUT.identifier);

    let error = call(Value::Num(1.0), vec![Value::Num(2.0), Value::Num(3.0)])
        .expect_err("arity must reject");
    assert_eq!(error.identifier(), POW2_ERROR_INVALID_ARGUMENT.identifier);

    let left = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("left");
    let right = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).expect("right");
    let error =
        call(Value::Tensor(left), vec![Value::Tensor(right)]).expect_err("shape must reject");
    assert_eq!(error.identifier(), POW2_ERROR_SIZE_MISMATCH.identifier);

    let _outputs = crate::output_count::push_output_count(Some(2));
    let error = call(Value::Num(1.0), vec![]).expect_err("outputs must reject");
    assert_eq!(error.identifier(), POW2_ERROR_TOO_MANY_OUTPUTS.identifier);
}
