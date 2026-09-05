use super::*;
use runmat_builtins::{LOG_DESCRIPTOR, LOG_ERROR_INVALID_INPUT};
use runmat_value::{
    CharArray, ComplexStorage, ComplexTensor, IntValue, IntegerStorage, NumericDType,
    NumericStorage, SymbolicExpr, Tensor,
};

#[test]
fn scalar_complex_single_and_invalid_inputs_preserve_contract() {
    assert_eq!(LOG_DESCRIPTOR.signatures[0].label, "Y = log(X)");
    let Value::Complex(real, imag) = call(Value::Num(-1.0)).expect("negative log") else {
        panic!("expected complex")
    };
    assert!(real.abs() < 1e-12);
    assert!((imag - std::f64::consts::PI).abs() < 1e-12);
    let input = Tensor::from_f32(vec![1.0, std::f32::consts::E], vec![1, 2]).unwrap();
    let Value::Tensor(output) = call(Value::Tensor(input)).expect("single log") else {
        panic!("expected tensor")
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    assert_eq!(output.shape, vec![1, 2]);
    assert_eq!(
        call(Value::from("bad"))
            .expect_err("string must fail")
            .identifier(),
        LOG_ERROR_INVALID_INPUT.identifier
    );
}

#[test]
fn exact_integer_extension_is_bounded_and_compatibility_gated() {
    for storage in [
        IntegerStorage::I8(vec![-1, 1]),
        IntegerStorage::I16(vec![-1, 1]),
        IntegerStorage::I32(vec![-1, 1]),
        IntegerStorage::I64(vec![-9_007_199_254_740_992, 9_007_199_254_740_992]),
        IntegerStorage::U8(vec![0, 1]),
        IntegerStorage::U16(vec![0, 1]),
        IntegerStorage::U32(vec![0, 1]),
        IntegerStorage::U64(vec![0, 9_007_199_254_740_992]),
    ] {
        let input = Tensor::new_integer(storage, vec![1, 2]).unwrap();
        assert!(matches!(
            call(Value::Tensor(input)),
            Ok(Value::Tensor(_)) | Ok(Value::ComplexTensor(_))
        ));
    }
    let wide =
        Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1]).unwrap();
    assert_eq!(
        call(Value::Tensor(wide)).unwrap_err().identifier(),
        LOG_ERROR_INVALID_INPUT.identifier
    );
    assert_eq!(
        call_matlab(Value::Int(IntValue::I8(1)))
            .unwrap_err()
            .identifier(),
        Some("RunMat:compatibility:LogIntegerInputExtension")
    );
}

#[test]
fn complex_table_character_and_symbolic_overloads_remain_distinct() {
    let complex = ComplexTensor::from_f32(vec![(1.0, 1.0)], vec![1, 1]).unwrap();
    let Value::ComplexTensor(output) = call(Value::ComplexTensor(complex)).expect("complex log")
    else {
        panic!("expected complex tensor")
    };
    assert!(matches!(
        output.into_complex_storage(),
        ComplexStorage::F32(_)
    ));

    let chars = CharArray::new(vec!['A'], 1, 1).unwrap();
    assert!(matches!(
        call(Value::CharArray(chars)),
        Ok(Value::Num(_)) | Ok(Value::Tensor(_))
    ));
    let table = crate::builtins::table::table_from_columns(
        vec!["X".into()],
        vec![Value::Tensor(
            Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap(),
        )],
    )
    .unwrap();
    assert!(matches!(call(table), Ok(Value::Object(_))));
    assert!(matches!(
        call(Value::Symbolic(SymbolicExpr::variable("x"))),
        Ok(Value::Symbolic(_))
    ));

    let zero = call(Value::Num(0.0)).expect("zero");
    assert!(matches!(zero, Value::Num(value) if value.is_infinite() && value.is_sign_negative()));
    let single = Tensor::from_f32(Vec::new(), vec![0, 2]).unwrap();
    let Value::Tensor(empty) = call(Value::Tensor(single)).unwrap() else {
        panic!("expected tensor")
    };
    assert_eq!(
        empty.into_numeric_storage().unwrap(),
        NumericStorage::F32(Vec::new())
    );
}
