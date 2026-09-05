use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sign_int_values() {
    let value = Value::Int(IntValue::I32(-7));
    let result = sign_builtin(value).unwrap();
    assert_eq!(result, Value::Int(IntValue::I32(-1)));
}

#[test]
fn sign_preserves_all_typed_integer_array_classes() {
    let cases = [
        (
            IntegerStorage::I8(vec![i8::MIN, 0, i8::MAX]),
            IntegerStorage::I8(vec![-1, 0, 1]),
        ),
        (
            IntegerStorage::I16(vec![i16::MIN, 0, i16::MAX]),
            IntegerStorage::I16(vec![-1, 0, 1]),
        ),
        (
            IntegerStorage::I32(vec![i32::MIN, 0, i32::MAX]),
            IntegerStorage::I32(vec![-1, 0, 1]),
        ),
        (
            IntegerStorage::I64(vec![i64::MIN, 0, i64::MAX]),
            IntegerStorage::I64(vec![-1, 0, 1]),
        ),
        (
            IntegerStorage::U8(vec![0, 1, u8::MAX]),
            IntegerStorage::U8(vec![0, 1, 1]),
        ),
        (
            IntegerStorage::U16(vec![0, 1, u16::MAX]),
            IntegerStorage::U16(vec![0, 1, 1]),
        ),
        (
            IntegerStorage::U32(vec![0, 1, u32::MAX]),
            IntegerStorage::U32(vec![0, 1, 1]),
        ),
        (
            IntegerStorage::U64(vec![0, 1, u64::MAX]),
            IntegerStorage::U64(vec![0, 1, 1]),
        ),
    ];
    for (input, expected) in cases {
        let input = Tensor::new_integer(input, vec![1, expected.len()]).expect("tensor");
        let Value::Tensor(result) = sign_builtin(Value::Tensor(input)).expect("sign") else {
            panic!("expected tensor");
        };
        assert_eq!(result.integer_storage(), Some(&expected));
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sign_bool_values() {
    let t = sign_builtin(Value::Bool(true)).unwrap();
    let f = sign_builtin(Value::Bool(false)).unwrap();
    assert_eq!(t, Value::Num(1.0));
    assert_eq!(f, Value::Num(0.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sign_infinite_values() {
    let tensor = Tensor::new(
        vec![f64::INFINITY, f64::NEG_INFINITY, 0.0, f64::NAN],
        vec![2, 2],
    )
    .unwrap();
    let result = sign_builtin(Value::Tensor(tensor)).unwrap();
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.materialize_f64()[0], 1.0);
            assert_eq!(out.materialize_f64()[1], -1.0);
            assert_eq!(out.materialize_f64()[2], 0.0);
            assert!(out.materialize_f64()[3].is_nan());
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}
