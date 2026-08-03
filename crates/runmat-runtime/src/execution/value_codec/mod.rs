mod decode;
mod encode;
mod error;

pub use decode::decode_inline_value;
pub use encode::encode_inline_value;
pub use error::ValueCodecError;

#[cfg(test)]
mod tests {
    use crate::execution::RuntimeExecutionServices;
    use runmat_builtins::{CellArray, IntValue, Value};

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
