use futures::executor::block_on;
use runmat_value::{ComplexStorage, IntegerStorage, Tensor, Value};

#[test]
fn complex_like_keeps_all_integer_payloads_exact_including_scalar_and_empty() {
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for storage in [
        IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
        IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
        IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
        IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
        IntegerStorage::U8(vec![0, u8::MAX]),
        IntegerStorage::U16(vec![0, u16::MAX]),
        IntegerStorage::U32(vec![0, u32::MAX]),
        IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
    ] {
        let scalar = IntegerStorage::from_scalar(storage.value_at(1).unwrap());
        let empty = storage.zeros_like(0);
        for (values, shape) in [
            (storage, vec![1, 2]),
            (scalar, vec![1, 1]),
            (empty, vec![0, 2]),
        ] {
            let expected = values.clone();
            let output = block_on(runmat_runtime::call_builtin_async_with_outputs(
                "plus",
                &[
                    Value::Tensor(Tensor::new_integer(values, shape.clone()).unwrap()),
                    Value::Num(0.0),
                    Value::String("like".into()),
                    Value::Complex(0.0, 1.0),
                ],
                1,
            ))
            .unwrap();
            let Value::ComplexTensor(tensor) = output.value() else {
                panic!("exact integer complex tensor")
            };
            assert_eq!(tensor.shape(), shape);
            let ComplexStorage::Integer(actual) = tensor.complex_storage() else {
                panic!("native integer complex storage")
            };
            assert_eq!(actual.real(), &expected);
            assert_eq!(actual.imag(), &expected.zeros_like(expected.len()));
        }
    }
}
