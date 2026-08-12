mod decode;
mod encode;
mod error;

pub use decode::decode_inline_value;
pub use encode::encode_inline_value;
pub use error::ValueCodecError;

#[cfg(test)]
mod tests {
    use crate::execution::RuntimeExecutionServices;
    use runmat_builtins::{
        CellArray, CharArray, ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage,
        MException, StringArray, StructValue, Tensor, Value,
    };

    use runmat_execution::value::{InlineValue, ValuePayload};

    use super::{decode_inline_value, encode_inline_value};

    #[test]
    fn portable_value_roundtrip_is_centralized_and_bit_exact() {
        let value = Value::Cell(
            CellArray::new(
                vec![
                    Value::Num(f64::from_bits(0x7ff8_0000_0000_0042)),
                    Value::Int(IntValue::U64(u64::MAX)),
                ],
                1,
                2,
            )
            .unwrap(),
        );
        let payload = encode_inline_value(&value).unwrap();
        let Value::Cell(decoded) = decode_inline_value(&payload).unwrap() else {
            panic!("expected decoded cell");
        };
        let Value::Num(number) = decoded.data[0] else {
            panic!("expected decoded number");
        };
        assert_eq!(number.to_bits(), 0x7ff8_0000_0000_0042);
        assert_eq!(decoded.data[1], Value::Int(IntValue::U64(u64::MAX)));
        assert_eq!(decoded.shape, vec![1, 2]);
    }

    #[test]
    fn stable_immutable_runtime_forms_round_trip_exactly() {
        let mut structure = StructValue::new();
        structure.insert(
            "tensor",
            Value::Tensor(
                Tensor::new(vec![f64::from_bits(0x8000_0000_0000_0000), 2.5], vec![2, 1]).unwrap(),
            ),
        );
        structure.insert(
            "strings",
            Value::StringArray(StringArray::new(vec!["α".into(), "β".into()], vec![1, 2]).unwrap()),
        );
        structure.insert(
            "chars",
            Value::CharArray(CharArray::new(vec!['a', 'β'], 2, 1).unwrap()),
        );
        structure.insert(
            "error",
            Value::MException(MException {
                identifier: "RunMat:test".into(),
                message: "failure".into(),
                stack: vec!["main:1".into()],
            }),
        );
        let value = Value::Struct(structure);
        let payload = encode_inline_value(&value).unwrap();
        let decoded = decode_inline_value(&payload).unwrap();
        assert_eq!(decoded, value);
    }

    #[test]
    fn native_complex_classes_round_trip_without_widening() {
        let single = Value::ComplexTensor(
            ComplexTensor::from_f32(
                vec![
                    (f32::from_bits(0x8000_0000), f32::from_bits(0x7fc0_0042)),
                    (f32::MAX, f32::MIN_POSITIVE),
                ],
                vec![1, 2],
            )
            .unwrap(),
        );
        let Value::ComplexTensor(decoded_single) =
            decode_inline_value(&encode_inline_value(&single).unwrap()).unwrap()
        else {
            panic!("expected decoded single complex tensor");
        };
        let decoded_values = decoded_single.as_f32_slice().unwrap();
        assert_eq!(decoded_values[0].0.to_bits(), 0x8000_0000);
        assert_eq!(decoded_values[0].1.to_bits(), 0x7fc0_0042);
        assert_eq!(decoded_values[1], (f32::MAX, f32::MIN_POSITIVE));
        assert_eq!(decoded_single.shape, vec![1, 2]);

        let integer_components = [
            (
                IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
                IntegerStorage::I8(vec![i8::MAX, i8::MIN]),
            ),
            (
                IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
                IntegerStorage::I16(vec![i16::MAX, i16::MIN]),
            ),
            (
                IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
                IntegerStorage::I32(vec![i32::MAX, i32::MIN]),
            ),
            (
                IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
                IntegerStorage::I64(vec![i64::MAX, i64::MIN]),
            ),
            (
                IntegerStorage::U8(vec![u8::MIN, u8::MAX]),
                IntegerStorage::U8(vec![u8::MAX, u8::MIN]),
            ),
            (
                IntegerStorage::U16(vec![u16::MIN, u16::MAX]),
                IntegerStorage::U16(vec![u16::MAX, u16::MIN]),
            ),
            (
                IntegerStorage::U32(vec![u32::MIN, u32::MAX]),
                IntegerStorage::U32(vec![u32::MAX, u32::MIN]),
            ),
            (
                IntegerStorage::U64(vec![u64::MIN, u64::MAX]),
                IntegerStorage::U64(vec![u64::MAX, u64::MIN]),
            ),
        ];
        for (real, imaginary) in integer_components {
            let value = Value::ComplexTensor(
                ComplexTensor::new_integer(
                    IntegerComplexStorage::new(real, imaginary).unwrap(),
                    vec![2, 1],
                )
                .unwrap(),
            );
            assert_eq!(
                decode_inline_value(&encode_inline_value(&value).unwrap()).unwrap(),
                value
            );
        }
    }

    #[test]
    fn callable_captures_round_trip_and_failures_name_the_local_path() {
        let closure = Value::Closure(runmat_builtins::Closure {
            function_name: "worker".into(),
            bound_function: Some(7),
            captures: vec![Value::Num(2.0)],
        });
        assert_eq!(
            decode_inline_value(&encode_inline_value(&closure).unwrap()).unwrap(),
            closure
        );
        let mut tampered = encode_inline_value(&closure).unwrap();
        let ValuePayload::Inline(ref mut inline) = tampered else {
            unreachable!("runtime encoding is inline");
        };
        let InlineValue::Callable(callable) = inline.as_mut() else {
            unreachable!("closure encoding is callable");
        };
        callable.qualified_name = "other_worker".into();
        assert!(decode_inline_value(&tampered).is_err());

        let mut structure = StructValue::new();
        let service = super::super::RuntimeExecutionService::new();
        structure.insert(
            "bad",
            Value::Future(runmat_execution::FutureHandle {
                id: runmat_execution::FutureId::derive(&[b"future"]),
                scope_id: service.scope_id(),
                outputs: runmat_execution::OutputContract {
                    requested_outputs: 1,
                },
            }),
        );
        let error = encode_inline_value(&Value::Cell(
            CellArray::new(vec![Value::Struct(structure)], 1, 1).unwrap(),
        ))
        .unwrap_err();
        assert!(error.to_string().contains("$[0].bad"));
    }

    #[test]
    fn live_execution_handles_are_never_boundary_payloads() {
        let service = super::super::RuntimeExecutionService::new();
        let future = runmat_execution::FutureHandle {
            id: runmat_execution::FutureId::derive(&[b"future"]),
            scope_id: service.scope_id(),
            outputs: runmat_execution::OutputContract {
                requested_outputs: 1,
            },
        };
        assert!(encode_inline_value(&Value::Future(future)).is_err());
    }
}
