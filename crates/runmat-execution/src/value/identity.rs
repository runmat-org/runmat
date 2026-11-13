use minicbor::Encoder;

use super::{
    CharacterEncoding, ExceptionFailureOrigin, ExceptionFrameSource, ExceptionGpuGatherRetry,
    ExceptionRuntimeMetadata, ExceptionValue, InlineValue, RegisteredData, ValueLimits,
    ValuePayload,
};
use crate::{ContractError, Digest};

pub(super) fn logical_digest(payload: &ValuePayload) -> Result<Digest, ContractError> {
    payload.validate(ValueLimits::default())?;
    match payload {
        ValuePayload::Object(reference) => return Ok(reference.logical_digest),
        ValuePayload::Distributed(handle) => {
            return Ok(handle_digest(
                b"runmat-distributed-value-v1\0",
                handle.scope_id.bytes(),
                handle.id.bytes(),
                handle.generation,
            ));
        }
        ValuePayload::Composite(handle) => {
            return Ok(handle_digest(
                b"runmat-composite-value-v1\0",
                handle.scope_id.bytes(),
                handle.id.bytes(),
                handle.generation,
            ));
        }
        ValuePayload::Inline(_) => {}
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
        ValuePayload::Distributed(handle) => encode_handle(
            encoder,
            2,
            handle.scope_id.bytes(),
            handle.id.bytes(),
            handle.generation,
        ),
        ValuePayload::Composite(handle) => encode_handle(
            encoder,
            3,
            handle.scope_id.bytes(),
            handle.id.bytes(),
            handle.generation,
        ),
    }
}

fn handle_digest(domain: &[u8], scope: &[u8], id: &[u8], generation: u64) -> Digest {
    let mut bytes = Vec::with_capacity(domain.len() + scope.len() + id.len() + 8);
    bytes.extend_from_slice(domain);
    bytes.extend_from_slice(scope);
    bytes.extend_from_slice(id);
    bytes.extend_from_slice(&generation.to_be_bytes());
    Digest::sha256(bytes)
}

fn encode_handle(
    encoder: &mut Encoder<&mut Vec<u8>>,
    tag: u8,
    scope: &[u8],
    id: &[u8],
    generation: u64,
) -> Result<(), ContractError> {
    encoder
        .array(4)
        .and_then(|encoder| encoder.u8(tag))
        .and_then(|encoder| encoder.bytes(scope))
        .and_then(|encoder| encoder.bytes(id))
        .and_then(|encoder| encoder.u64(generation))
        .map(|_| ())
        .map_err(encoding)
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
        InlineValue::String(value) => scalar(encoder, 12, |encoder| match value.as_str() {
            Some(text) => encoder.str(text).map(|_| ()),
            None => encoder.null().map(|_| ()),
        }),
        InlineValue::Char {
            encoding: character_encoding,
            shape,
            code_units,
        } => {
            encoder
                .array(4)
                .and_then(|encoder| encoder.u8(13))
                .map_err(encoding)?;
            encoder
                .u8(match character_encoding {
                    CharacterEncoding::Utf16CodeUnitsV1 => 1,
                })
                .map_err(encoding)?;
            encode_u64_array(encoder, shape)?;
            encoder.array(code_units.len() as u64).map_err(encoding)?;
            for value in code_units {
                encoder.u16(*value).map_err(encoding)?;
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
                match value.as_str() {
                    Some(text) => encoder.str(text).map_err(encoding)?,
                    None => encoder.null().map_err(encoding)?,
                };
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
        InlineValue::Exception(value) => encode_exception(encoder, value),
        InlineValue::Callable(value) => {
            encoder
                .array(8)
                .and_then(|encoder| encoder.u8(22))
                .map_err(encoding)?;
            match value.owner {
                super::CallableOwner::Workspace => {
                    encoder.array(1).and_then(|e| e.u8(0)).map_err(encoding)?;
                }
                super::CallableOwner::External => {
                    encoder.array(1).and_then(|e| e.u8(1)).map_err(encoding)?;
                }
                super::CallableOwner::Method => {
                    encoder.array(1).and_then(|e| e.u8(2)).map_err(encoding)?;
                }
                super::CallableOwner::Bound(origin) => {
                    encoder
                        .array(3)
                        .and_then(|e| e.u8(3))
                        .and_then(|e| e.bytes(origin.program.bytes()))
                        .and_then(|e| e.u32(origin.function.0))
                        .map_err(encoding)?;
                }
            }
            encoder
                .str(&value.qualified_name)
                .and_then(|encoder| encoder.bytes(value.callable_digest.bytes()))
                .and_then(|encoder| {
                    encoder.u8(match value.form {
                        super::CallableForm::Handle => 0,
                        super::CallableForm::Closure => 1,
                    })
                })
                .and_then(|encoder| encoder.u8(value.source_role as u8))
                .map_err(encoding)?;
            match &value.source_file {
                Some(path) => encoder.str(path).map_err(encoding)?,
                None => encoder.null().map_err(encoding)?,
            };
            encode_payloads(encoder, &value.captures)
        }
        InlineValue::ImmutableValueClass(value) => encode_registered(encoder, 23, value),
        InlineValue::DurableOrderedStruct(_)
        | InlineValue::DurablePlainObject(_)
        | InlineValue::DurableStructArray(_)
        | InlineValue::DurableObjectArray(_) => Err(ContractError::invalid(
            "V2 execution value",
            "durable preference aggregates require a preference document",
        )),
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
    match &value.runtime {
        Some(runtime) => encode_exception_runtime(encoder, runtime),
        None => encoder.null().map(|_| ()).map_err(encoding),
    }
}

fn encode_exception_runtime(
    encoder: &mut Encoder<&mut Vec<u8>>,
    runtime: &ExceptionRuntimeMetadata,
) -> Result<(), ContractError> {
    encoder.array(9).map_err(encoding)?;
    encode_optional_span(encoder, runtime.span)?;
    encode_optional_text(encoder, runtime.builtin.as_deref())?;
    encode_optional_text(encoder, runtime.task_id.as_deref())?;
    encoder
        .array(runtime.call_frames.len() as u64)
        .map_err(encoding)?;
    for frame in &runtime.call_frames {
        encoder
            .array(
                if matches!(&frame.source, ExceptionFrameSource::Detached(_)) {
                    6
                } else {
                    3
                },
            )
            .and_then(|encoder| encoder.str(&frame.function))
            .map_err(encoding)?;
        match &frame.source {
            ExceptionFrameSource::Admitted(source_id) => {
                encoder.u32(source_id.0).map(|_| ()).map_err(encoding)?;
            }
            ExceptionFrameSource::Unknown | ExceptionFrameSource::Detached(_) => {
                encoder.null().map(|_| ()).map_err(encoding)?;
            }
        }
        encode_optional_span(encoder, frame.span)?;
        if let ExceptionFrameSource::Detached(presentation) = &frame.source {
            encoder.str(&presentation.display_name).map_err(encoding)?;
            if let Some(line) = presentation.line {
                encoder.u64(line).map_err(encoding)?;
            } else {
                encoder.null().map_err(encoding)?;
            }
            if let Some(column) = presentation.column {
                encoder.u64(column).map_err(encoding)?;
            } else {
                encoder.null().map_err(encoding)?;
            }
        }
    }
    encoder.u64(runtime.call_frames_elided).map_err(encoding)?;
    encoder
        .array(runtime.call_stack.len() as u64)
        .map_err(encoding)?;
    for frame in &runtime.call_stack {
        encoder.str(frame).map_err(encoding)?;
    }
    encode_optional_text(encoder, runtime.phase.as_deref())?;
    encoder
        .u8(match runtime.gpu_gather_retry {
            ExceptionGpuGatherRetry::Legacy => 0,
            ExceptionGpuGatherRetry::Never => 1,
            ExceptionGpuGatherRetry::Requested => 2,
        })
        .map_err(encoding)?;
    encoder
        .u8(match runtime.failure_origin {
            ExceptionFailureOrigin::Unclassified => 0,
            ExceptionFailureOrigin::LanguageEvaluation => 1,
            ExceptionFailureOrigin::ExecutionControlCancellation => 2,
        })
        .map(|_| ())
        .map_err(encoding)
}

fn encode_optional_span(
    encoder: &mut Encoder<&mut Vec<u8>>,
    span: Option<(u64, u64)>,
) -> Result<(), ContractError> {
    match span {
        Some((start, length)) => encoder
            .array(2)
            .and_then(|encoder| encoder.u64(start))
            .and_then(|encoder| encoder.u64(length))
            .map(|_| ())
            .map_err(encoding),
        None => encoder.null().map(|_| ()).map_err(encoding),
    }
}

fn encode_optional_text(
    encoder: &mut Encoder<&mut Vec<u8>>,
    text: Option<&str>,
) -> Result<(), ContractError> {
    match text {
        Some(text) => encoder.str(text).map(|_| ()).map_err(encoding),
        None => encoder.null().map(|_| ()).map_err(encoding),
    }
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
