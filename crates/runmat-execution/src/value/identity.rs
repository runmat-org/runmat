use minicbor::Encoder;

use super::{ExceptionValue, InlineValue, RegisteredData, ValueLimits, ValuePayload};
use crate::{ContractError, Digest};

pub(super) fn logical_digest(payload: &ValuePayload) -> Result<Digest, ContractError> {
    payload.validate(ValueLimits::default())?;
    if let ValuePayload::Object(reference) = payload {
        return Ok(reference.logical_digest);
    }
    let mut bytes = b"runmat-logical-value-v1\0".to_vec();
    encode_payload(&mut Encoder::new(&mut bytes), payload)?;
    Ok(Digest::sha256(bytes))
}

fn encode_payload(
    encoder: &mut Encoder<&mut Vec<u8>>,
    payload: &ValuePayload,
) -> Result<(), ContractError> {
    match payload {
        ValuePayload::Inline(value) => {
            encoder
                .array(2)
                .and_then(|encoder| encoder.u8(0))
                .map_err(encoding)?;
            encode_inline(encoder, value)
        }
        ValuePayload::Object(reference) => encoder
            .array(2)
            .and_then(|encoder| encoder.u8(1))
            .and_then(|encoder| encoder.bytes(reference.logical_digest.bytes()))
            .map(|_| ())
            .map_err(encoding),
    }
}

fn encode_inline(
    encoder: &mut Encoder<&mut Vec<u8>>,
    value: &InlineValue,
) -> Result<(), ContractError> {
    match value {
        InlineValue::Null => tagged(encoder, 0),
        InlineValue::Logical(value) => {
            scalar(encoder, 1, |encoder| encoder.bool(*value).map(|_| ()))
        }
        InlineValue::F64Bits(value) => {
            scalar(encoder, 2, |encoder| encoder.u64(*value).map(|_| ()))
        }
        InlineValue::I8(value) => scalar(encoder, 3, |encoder| encoder.i8(*value).map(|_| ())),
        InlineValue::I16(value) => scalar(encoder, 4, |encoder| encoder.i16(*value).map(|_| ())),
        InlineValue::I32(value) => scalar(encoder, 5, |encoder| encoder.i32(*value).map(|_| ())),
        InlineValue::I64(value) => scalar(encoder, 6, |encoder| encoder.i64(*value).map(|_| ())),
        InlineValue::U8(value) => scalar(encoder, 7, |encoder| encoder.u8(*value).map(|_| ())),
        InlineValue::U16(value) => scalar(encoder, 8, |encoder| encoder.u16(*value).map(|_| ())),
        InlineValue::U32(value) => scalar(encoder, 9, |encoder| encoder.u32(*value).map(|_| ())),
        InlineValue::U64(value) => scalar(encoder, 10, |encoder| encoder.u64(*value).map(|_| ())),
        InlineValue::ComplexF64Bits { real, imaginary } => encoder
            .array(3)
            .and_then(|encoder| encoder.u8(11))
            .and_then(|encoder| encoder.u64(*real))
            .and_then(|encoder| encoder.u64(*imaginary))
            .map(|_| ())
            .map_err(encoding),
        InlineValue::String(value) => scalar(encoder, 12, |encoder| encoder.str(value).map(|_| ())),
        InlineValue::Char { shape, code_points } => {
            encoder
                .array(3)
                .and_then(|encoder| encoder.u8(13))
                .map_err(encoding)?;
            encode_u64_array(encoder, shape)?;
            encoder.array(code_points.len() as u64).map_err(encoding)?;
            for value in code_points {
                encoder.u32(*value).map_err(encoding)?;
            }
            Ok(())
        }
        InlineValue::StringArray { shape, values } => {
            encoder
                .array(3)
                .and_then(|encoder| encoder.u8(14))
                .map_err(encoding)?;
            encode_u64_array(encoder, shape)?;
            encoder.array(values.len() as u64).map_err(encoding)?;
            for value in values {
                encoder.str(value).map_err(encoding)?;
            }
            Ok(())
        }
        InlineValue::Dense(value) => {
            encoder
                .array(4)
                .and_then(|encoder| encoder.u8(15))
                .and_then(|encoder| encoder.u8(value.element_type as u8))
                .map_err(encoding)?;
            encode_u64_array(encoder, &value.shape)?;
            encoder
                .bytes(&value.little_endian_data)
                .map(|_| ())
                .map_err(encoding)
        }
        InlineValue::Sparse(value) => {
            encoder
                .array(7)
                .and_then(|encoder| encoder.u8(16))
                .and_then(|encoder| encoder.u8(value.element_type as u8))
                .and_then(|encoder| encoder.u64(value.rows))
                .and_then(|encoder| encoder.u64(value.columns))
                .map_err(encoding)?;
            encode_u64_array(encoder, &value.column_offsets)?;
            encode_u64_array(encoder, &value.row_indices)?;
            encoder
                .bytes(&value.little_endian_data)
                .map(|_| ())
                .map_err(encoding)
        }
        InlineValue::Symbolic(value) => encode_registered(encoder, 17, value),
        InlineValue::Cell { shape, values } => {
            encoder
                .array(3)
                .and_then(|encoder| encoder.u8(18))
                .map_err(encoding)?;
            encode_u64_array(encoder, shape)?;
            encode_payloads(encoder, values)
        }
        InlineValue::Struct(fields) => {
            encoder
                .array(2)
                .and_then(|encoder| encoder.u8(19))
                .and_then(|encoder| encoder.array(fields.len() as u64))
                .map_err(encoding)?;
            for field in fields {
                encoder
                    .array(2)
                    .and_then(|encoder| encoder.str(&field.name))
                    .map_err(encoding)?;
                encode_nested(encoder, &field.value)?;
            }
            Ok(())
        }
        InlineValue::OutputList(values) => {
            encoder
                .array(2)
                .and_then(|encoder| encoder.u8(20))
                .map_err(encoding)?;
            encode_payloads(encoder, values)
        }
        InlineValue::Exception(value) => encode_exception(encoder, value),
        InlineValue::Callable(value) => {
            encoder
                .array(5)
                .and_then(|encoder| encoder.u8(22))
                .and_then(|encoder| encoder.str(&value.owner_identity))
                .and_then(|encoder| encoder.str(&value.qualified_name))
                .and_then(|encoder| encoder.bytes(value.callable_digest.bytes()))
                .map_err(encoding)?;
            encode_payloads(encoder, &value.captures)
        }
        InlineValue::ImmutableValueClass(value) => encode_registered(encoder, 23, value),
    }
}

fn encode_registered(
    encoder: &mut Encoder<&mut Vec<u8>>,
    tag: u8,
    value: &RegisteredData,
) -> Result<(), ContractError> {
    encoder
        .array(4)
        .and_then(|encoder| encoder.u8(tag))
        .and_then(|encoder| encoder.str(&value.type_identity))
        .and_then(|encoder| encoder.u32(value.schema_version))
        .and_then(|encoder| encoder.array(value.fields.len() as u64))
        .map_err(encoding)?;
    for field in &value.fields {
        encoder
            .array(2)
            .and_then(|encoder| encoder.str(&field.name))
            .map_err(encoding)?;
        encode_nested(encoder, &field.value)?;
    }
    Ok(())
}

fn encode_exception(
    encoder: &mut Encoder<&mut Vec<u8>>,
    value: &ExceptionValue,
) -> Result<(), ContractError> {
    encoder
        .array(6)
        .and_then(|encoder| encoder.u8(21))
        .and_then(|encoder| encoder.str(&value.identifier))
        .and_then(|encoder| encoder.str(&value.message))
        .and_then(|encoder| encoder.array(value.stack.len() as u64))
        .map_err(encoding)?;
    for frame in &value.stack {
        encoder.str(frame).map_err(encoding)?;
    }
    encoder.array(value.causes.len() as u64).map_err(encoding)?;
    for cause in &value.causes {
        encode_exception(encoder, cause)?;
    }
    encoder.null().map(|_| ()).map_err(encoding)
}

fn encode_payloads(
    encoder: &mut Encoder<&mut Vec<u8>>,
    values: &[ValuePayload],
) -> Result<(), ContractError> {
    encoder.array(values.len() as u64).map_err(encoding)?;
    for value in values {
        encode_nested(encoder, value)?;
    }
    Ok(())
}

fn encode_nested(
    encoder: &mut Encoder<&mut Vec<u8>>,
    value: &ValuePayload,
) -> Result<(), ContractError> {
    encoder
        .bytes(value.logical_digest()?.bytes())
        .map(|_| ())
        .map_err(encoding)
}

fn encode_u64_array(
    encoder: &mut Encoder<&mut Vec<u8>>,
    values: &[u64],
) -> Result<(), ContractError> {
    encoder.array(values.len() as u64).map_err(encoding)?;
    for value in values {
        encoder.u64(*value).map_err(encoding)?;
    }
    Ok(())
}

fn tagged(encoder: &mut Encoder<&mut Vec<u8>>, tag: u8) -> Result<(), ContractError> {
    encoder
        .array(1)
        .and_then(|encoder| encoder.u8(tag))
        .map(|_| ())
        .map_err(encoding)
}

fn scalar(
    encoder: &mut Encoder<&mut Vec<u8>>,
    tag: u8,
    encode: impl FnOnce(
        &mut Encoder<&mut Vec<u8>>,
    ) -> Result<(), minicbor::encode::Error<std::convert::Infallible>>,
) -> Result<(), ContractError> {
    encoder
        .array(2)
        .and_then(|encoder| encoder.u8(tag))
        .map_err(encoding)?;
    encode(encoder).map_err(encoding)
}

fn encoding(error: minicbor::encode::Error<std::convert::Infallible>) -> ContractError {
    ContractError::invalid("logical value identity", error.to_string())
}
