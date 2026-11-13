mod bytes;
mod input;
mod shape;

use runmat_builtins::{TYPECAST_ERROR_INTERNAL, TYPECAST_ERROR_INVALID_INPUT};
use runmat_value::{ComplexTensor, LogicalArray, NumericStorage, Tensor, Value};

use crate::BuiltinResult;

use super::{error, target::OutputTarget};

pub(super) fn reinterpret(source: Value, target: OutputTarget) -> BuiltinResult<Value> {
    let source_shape = input::shape(&source)?;
    input::validate_vector(&source_shape)?;
    let encoded = bytes::encode(source)?;
    let bytes_per_output = target
        .representation
        .width()
        .checked_mul(if target.complex { 2 } else { 1 })
        .ok_or_else(|| {
            error::build(
                &TYPECAST_ERROR_INVALID_INPUT,
                "output element width overflow",
            )
        })?;
    if encoded.len() % bytes_per_output != 0 {
        return Err(error::build(
            &TYPECAST_ERROR_INVALID_INPUT,
            "input byte count is not divisible by the requested output element width",
        ));
    }
    let output_shape = shape::vector(&source_shape, encoded.len() / bytes_per_output);
    let storage = bytes::decode(&encoded, target.representation);
    materialize(storage, output_shape, target)
}

fn materialize(
    storage: NumericStorage,
    shape: Vec<usize>,
    target: OutputTarget,
) -> BuiltinResult<Value> {
    if target.complex {
        return ComplexTensor::from_complex_storage(bytes::pair_complex(storage), shape)
            .map(Value::ComplexTensor)
            .map_err(|cause| error::build(&TYPECAST_ERROR_INTERNAL, cause));
    }
    if matches!(
        target.representation,
        super::target::Representation::Logical
    ) {
        let NumericStorage::U8(values) = storage else {
            unreachable!("logical decoder emits u8 storage")
        };
        return LogicalArray::new(values, shape)
            .map(Value::LogicalArray)
            .map_err(|cause| error::build(&TYPECAST_ERROR_INTERNAL, cause));
    }
    Tensor::from_numeric_storage(storage, shape)
        .map(Value::Tensor)
        .map_err(|cause| error::build(&TYPECAST_ERROR_INTERNAL, cause))
}
