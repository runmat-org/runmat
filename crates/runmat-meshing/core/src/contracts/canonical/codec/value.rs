use minicbor::data::Type;
use minicbor::{Decoder, Encoder};
use serde_json::{Map, Number, Value};

use super::{decoding_error, encoding_error, MeshingCanonicalLimits};
use crate::contracts::canonical::MeshingContractError;

pub(super) fn encode_value(
    encoder: &mut Encoder<&mut Vec<u8>>,
    value: &Value,
) -> Result<(), MeshingContractError> {
    match value {
        Value::Null => encoder.null().map(|_| ()).map_err(encoding_error),
        Value::Bool(value) => encoder.bool(*value).map(|_| ()).map_err(encoding_error),
        Value::Number(value) => encode_number(encoder, value),
        Value::String(value) => encoder.str(value).map(|_| ()).map_err(encoding_error),
        Value::Array(values) => {
            encoder.array(values.len() as u64).map_err(encoding_error)?;
            for value in values {
                encode_value(encoder, value)?;
            }
            Ok(())
        }
        Value::Object(values) => {
            let mut entries: Vec<_> = values.iter().collect();
            entries.sort_unstable_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));
            encoder.map(entries.len() as u64).map_err(encoding_error)?;
            for (key, value) in entries {
                encoder.str(key).map_err(encoding_error)?;
                encode_value(encoder, value)?;
            }
            Ok(())
        }
    }
}

pub(super) fn decode_value(
    decoder: &mut Decoder<'_>,
    limits: MeshingCanonicalLimits,
    depth: usize,
) -> Result<Value, MeshingContractError> {
    if depth > limits.maximum_nesting_depth {
        return Err(MeshingContractError::invalid(
            "canonical decoding",
            "nesting depth exceeds the contract limit",
        ));
    }
    match decoder.datatype().map_err(decoding_error)? {
        Type::Null => {
            decoder.null().map_err(decoding_error)?;
            Ok(Value::Null)
        }
        Type::Bool => decoder.bool().map(Value::Bool).map_err(decoding_error),
        Type::U8 | Type::U16 | Type::U32 | Type::U64 => decoder
            .u64()
            .map(|value| Value::Number(Number::from(value)))
            .map_err(decoding_error),
        Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::Int => decoder
            .i64()
            .map(|value| Value::Number(Number::from(value)))
            .map_err(decoding_error),
        Type::F16 | Type::F32 | Type::F64 => {
            let value = decoder.f64().map_err(decoding_error)?;
            Number::from_f64(value).map(Value::Number).ok_or_else(|| {
                MeshingContractError::invalid(
                    "canonical decoding",
                    "non-finite floating point values are forbidden",
                )
            })
        }
        Type::String => decode_string(decoder, limits).map(Value::String),
        Type::Array => decode_array(decoder, limits, depth),
        Type::Map => decode_object(decoder, limits, depth),
        other => Err(MeshingContractError::invalid(
            "canonical decoding",
            format!("unsupported CBOR type {other}"),
        )),
    }
}

fn encode_number(
    encoder: &mut Encoder<&mut Vec<u8>>,
    value: &Number,
) -> Result<(), MeshingContractError> {
    if let Some(value) = value.as_i64() {
        encoder.i64(value).map(|_| ()).map_err(encoding_error)
    } else if let Some(value) = value.as_u64() {
        encoder.u64(value).map(|_| ()).map_err(encoding_error)
    } else if let Some(value) = value.as_f64() {
        if !value.is_finite() {
            return Err(MeshingContractError::invalid(
                "canonical encoding",
                "non-finite floating point values are forbidden",
            ));
        }
        encoder.f64(value).map(|_| ()).map_err(encoding_error)
    } else {
        Err(MeshingContractError::invalid(
            "canonical encoding",
            "unsupported JSON number",
        ))
    }
}

fn decode_string(
    decoder: &mut Decoder<'_>,
    limits: MeshingCanonicalLimits,
) -> Result<String, MeshingContractError> {
    let value = decoder.str().map_err(decoding_error)?;
    if value.len() > limits.maximum_string_bytes {
        return Err(MeshingContractError::invalid(
            "canonical decoding",
            "string exceeds the contract limit",
        ));
    }
    Ok(value.to_owned())
}

fn decode_array(
    decoder: &mut Decoder<'_>,
    limits: MeshingCanonicalLimits,
    depth: usize,
) -> Result<Value, MeshingContractError> {
    let length = bounded_length(
        decoder.array().map_err(decoding_error)?,
        limits.maximum_collection_items,
        "array",
    )?;
    let mut values = Vec::with_capacity(length);
    for _ in 0..length {
        values.push(decode_value(decoder, limits, depth + 1)?);
    }
    Ok(Value::Array(values))
}

fn decode_object(
    decoder: &mut Decoder<'_>,
    limits: MeshingCanonicalLimits,
    depth: usize,
) -> Result<Value, MeshingContractError> {
    let length = bounded_length(
        decoder.map().map_err(decoding_error)?,
        limits.maximum_collection_items,
        "map",
    )?;
    let mut values = Map::new();
    let mut previous_key: Option<String> = None;
    for _ in 0..length {
        let key = decode_string(decoder, limits)?;
        if previous_key
            .as_ref()
            .is_some_and(|previous| previous.as_bytes() >= key.as_bytes())
        {
            return Err(MeshingContractError::invalid(
                "canonical decoding",
                "map keys must be unique and ordered by UTF-8 bytes",
            ));
        }
        let value = decode_value(decoder, limits, depth + 1)?;
        previous_key = Some(key.clone());
        values.insert(key, value);
    }
    Ok(Value::Object(values))
}

fn bounded_length(
    length: Option<u64>,
    maximum: usize,
    field: &str,
) -> Result<usize, MeshingContractError> {
    let length = length.ok_or_else(|| {
        MeshingContractError::invalid("canonical decoding", "indefinite collections are forbidden")
    })?;
    let length = usize::try_from(length).map_err(|_| {
        MeshingContractError::invalid("canonical decoding", format!("{field} length overflows"))
    })?;
    if length > maximum {
        return Err(MeshingContractError::invalid(
            "canonical decoding",
            format!("{field} exceeds the collection limit"),
        ));
    }
    Ok(length)
}
