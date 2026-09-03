use super::*;
use futures::executor::block_on;
use runmat_value::{IntValue, IntegerStorage, NumericStorage, Tensor};

fn call(value: Value) -> BuiltinResult<Value> {
    block_on(factor_builtin(value, Vec::new()))
}

#[test]
fn handles_special_values_and_large_semiprime() {
    for value in [0.0, 1.0] {
        let Value::Tensor(out) = call(Value::Num(value)).unwrap() else {
            panic!()
        };
        assert_eq!(out.as_f64_slice(), Some([value].as_slice()));
    }
    let n = 4_294_967_291u64 * 4_294_967_279u64;
    let Value::Tensor(out) = call(Value::Int(IntValue::U64(n))).unwrap() else {
        panic!()
    };
    assert_eq!(
        out.integer_storage(),
        Some(&IntegerStorage::U64(vec![4_294_967_279, 4_294_967_291]))
    );
}

#[test]
fn preserves_all_numeric_classes() {
    let inputs = [
        NumericStorage::F64(vec![12.0]),
        NumericStorage::F32(vec![12.0]),
        NumericStorage::I8(vec![12]),
        NumericStorage::I16(vec![12]),
        NumericStorage::I32(vec![12]),
        NumericStorage::I64(vec![12]),
        NumericStorage::U8(vec![12]),
        NumericStorage::U16(vec![12]),
        NumericStorage::U32(vec![12]),
        NumericStorage::U64(vec![12]),
    ];
    for storage in inputs {
        let dtype = storage.numeric_dtype();
        let input = Tensor::from_numeric_storage(storage, vec![1, 1]).unwrap();
        let Value::Tensor(out) = call(Value::Tensor(input)).unwrap() else {
            panic!()
        };
        assert_eq!(out.numeric_dtype(), dtype);
        assert_eq!(out.shape, vec![1, 3]);
    }
}

#[test]
fn rejects_invalid_inputs_and_arity() {
    for value in [
        Value::Num(-1.0),
        Value::Num(1.5),
        Value::Num(u64::MAX as f64),
        Value::Complex(2.0, 0.0),
    ] {
        assert!(call(value).is_err());
    }
    assert!(block_on(factor_builtin(Value::Num(2.0), vec![Value::Num(3.0)])).is_err());
}
