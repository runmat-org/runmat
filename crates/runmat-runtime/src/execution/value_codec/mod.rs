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
        CellArray, CharArray, IntValue, MException, StringArray, StructValue, Tensor, Value,
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
