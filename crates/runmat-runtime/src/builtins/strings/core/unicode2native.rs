//! MATLAB-compatible `unicode2native` builtin for RunMat.

use encoding_rs::{EncoderResult, Encoding, UTF_16BE, UTF_16LE, UTF_8};
use runmat_builtins::{
    BuiltinErrorDescriptor, UNICODE2NATIVE_DESCRIPTOR, UNICODE2NATIVE_ERROR_INTERNAL,
    UNICODE2NATIVE_ERROR_INVALID_ARGUMENT, UNICODE2NATIVE_ERROR_INVALID_ENCODING,
    UNICODE2NATIVE_ERROR_INVALID_INPUT, UNICODE2NATIVE_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, IntegerStorage, StringArray, Tensor, Value};

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::common::char_row_to_string;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "unicode2native";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::core::unicode2native")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "unicode2native",
    op_kind: GpuOpKind::Custom("text-encoding"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Text encoding runs on the CPU; GPU-resident inputs are gathered before validation.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::strings::core::unicode2native"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "unicode2native",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Conversion builtin; not eligible for fusion and always materialises host uint8 bytes.",
};

fn unicode2native_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    unicode2native_error_with_message(error.message, error)
}

fn unicode2native_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn remap_unicode2native_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

#[runtime_builtin(
    name = "unicode2native",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::core::unicode2native"
)]
async fn unicode2native_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &UNICODE2NATIVE_DESCRIPTOR,
        &UNICODE2NATIVE_ERROR_TOO_MANY_OUTPUTS,
        "unicode2native",
    )?;
    if rest.len() > 1 {
        return Err(unicode2native_error_with_message(
            "unicode2native: too many input arguments",
            &UNICODE2NATIVE_ERROR_INVALID_ARGUMENT,
        ));
    }

    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(remap_unicode2native_flow)?;
    let text = value_to_text_scalar(&gathered)?;
    let encoding = match rest.into_iter().next() {
        Some(value) => {
            let gathered = gather_if_needed_async(&value)
                .await
                .map_err(remap_unicode2native_flow)?;
            EncodingSpec::parse(&value_to_encoding_name(&gathered)?)?
        }
        None => EncodingSpec::Encoding(UTF_8),
    };

    bytes_to_uint8_row(encoding.encode(&text)?)
}

#[derive(Clone, Copy)]
enum EncodingSpec {
    Encoding(&'static Encoding),
    Ascii,
    Latin1,
    Utf32 { little_endian: bool, bom: bool },
    Utf16WithBom,
}

impl EncodingSpec {
    fn parse(name: &str) -> BuiltinResult<Self> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(unicode2native_error_with_message(
                "unicode2native: encoding name cannot be empty",
                &UNICODE2NATIVE_ERROR_INVALID_ENCODING,
            ));
        }

        let lowered = normalize_encoding_name(trimmed);
        match lowered.as_str() {
            "ascii" | "us-ascii" | "ansi_x3.4-1968" | "iso-ir-6" | "iso646-us" | "us" => {
                return Ok(Self::Ascii);
            }
            "latin1" | "latin-1" | "iso-8859-1" | "iso8859-1" | "ibm819" | "cp819"
            | "csisolatin1" | "l1" => return Ok(Self::Latin1),
            "utf-16" | "utf16" => return Ok(Self::Utf16WithBom),
            "utf-32" | "utf32" => {
                return Ok(Self::Utf32 {
                    little_endian: true,
                    bom: true,
                });
            }
            "utf-32le" | "utf32le" => {
                return Ok(Self::Utf32 {
                    little_endian: true,
                    bom: false,
                });
            }
            "utf-32be" | "utf32be" => {
                return Ok(Self::Utf32 {
                    little_endian: false,
                    bom: false,
                });
            }
            "unicode" | "system" => return Ok(Self::Encoding(UTF_8)),
            _ => {}
        }

        Encoding::for_label(lowered.as_bytes())
            .map(Self::Encoding)
            .ok_or_else(|| {
                unicode2native_error_with_message(
                    format!("unicode2native: unsupported character encoding '{trimmed}'"),
                    &UNICODE2NATIVE_ERROR_INVALID_ENCODING,
                )
            })
    }

    fn encode(self, text: &str) -> BuiltinResult<Vec<u8>> {
        match self {
            Self::Ascii => {
                encode_single_byte(text, |code| if code <= 0x7F { code as u8 } else { b'?' })
            }
            Self::Latin1 => {
                encode_single_byte(text, |code| if code <= 0xFF { code as u8 } else { b'?' })
            }
            Self::Encoding(encoding) if encoding == UTF_16LE => {
                encode_utf16_without_bom(text, true)
            }
            Self::Encoding(encoding) if encoding == UTF_16BE => {
                encode_utf16_without_bom(text, false)
            }
            Self::Encoding(encoding) => encode_with_question_replacement(encoding, text),
            Self::Utf32 { little_endian, bom } => encode_utf32(text, little_endian, bom),
            Self::Utf16WithBom => {
                let capacity = text
                    .len()
                    .checked_mul(2)
                    .and_then(|len| len.checked_add(2))
                    .ok_or_else(|| unicode2native_error(&UNICODE2NATIVE_ERROR_INTERNAL))?;
                let mut bytes = Vec::new();
                bytes
                    .try_reserve_exact(capacity)
                    .map_err(|_| unicode2native_error(&UNICODE2NATIVE_ERROR_INTERNAL))?;
                bytes.extend_from_slice(&[0xFF, 0xFE]);
                bytes.extend(encode_utf16_without_bom(text, true)?);
                Ok(bytes)
            }
        }
    }
}

fn normalize_encoding_name(raw: &str) -> String {
    let lowered = raw.to_ascii_lowercase();
    match lowered.as_str() {
        "utf_16" => "utf-16".to_string(),
        "utf_16le" => "utf-16le".to_string(),
        "utf_16be" => "utf-16be".to_string(),
        "utf_32" => "utf-32".to_string(),
        "utf_32le" => "utf-32le".to_string(),
        "utf_32be" => "utf-32be".to_string(),
        "windows_1252" => "windows-1252".to_string(),
        "shift_jis" => "shift_jis".to_string(),
        _ => lowered,
    }
}

fn encode_single_byte<F>(text: &str, mut encode: F) -> BuiltinResult<Vec<u8>>
where
    F: FnMut(u32) -> u8,
{
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(text.len())
        .map_err(|_| unicode2native_error(&UNICODE2NATIVE_ERROR_INTERNAL))?;
    bytes.extend(text.chars().map(|ch| encode(ch as u32)));
    Ok(bytes)
}

fn encode_with_question_replacement(
    encoding: &'static Encoding,
    text: &str,
) -> BuiltinResult<Vec<u8>> {
    let mut encoder = encoding.new_encoder();
    let capacity = encoder
        .max_buffer_length_from_utf8_without_replacement(text.len())
        .ok_or_else(|| unicode2native_error(&UNICODE2NATIVE_ERROR_INTERNAL))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| unicode2native_error(&UNICODE2NATIVE_ERROR_INTERNAL))?;

    let mut remaining = text;
    loop {
        let chunk_capacity = encoder
            .max_buffer_length_from_utf8_without_replacement(remaining.len())
            .ok_or_else(|| unicode2native_error(&UNICODE2NATIVE_ERROR_INTERNAL))?;
        let start = bytes.len();
        bytes
            .try_reserve_exact(chunk_capacity)
            .map_err(|_| unicode2native_error(&UNICODE2NATIVE_ERROR_INTERNAL))?;
        bytes.resize(start + chunk_capacity, 0);

        let (result, read, written) =
            encoder.encode_from_utf8_without_replacement(remaining, &mut bytes[start..], true);
        bytes.truncate(start + written);
        remaining = &remaining[read..];

        match result {
            EncoderResult::InputEmpty => return Ok(bytes),
            EncoderResult::OutputFull => continue,
            EncoderResult::Unmappable(ch) => {
                bytes
                    .try_reserve_exact(1)
                    .map_err(|_| unicode2native_error(&UNICODE2NATIVE_ERROR_INTERNAL))?;
                bytes.push(b'?');
                if remaining.starts_with(ch) {
                    let skip = ch.len_utf8();
                    remaining = remaining.get(skip..).ok_or_else(|| {
                        unicode2native_error_with_message(
                            "unicode2native: internal encoding cursor error",
                            &UNICODE2NATIVE_ERROR_INTERNAL,
                        )
                    })?;
                }
            }
        }
    }
}

fn encode_utf16_without_bom(text: &str, little_endian: bool) -> BuiltinResult<Vec<u8>> {
    let capacity = text
        .len()
        .checked_mul(2)
        .ok_or_else(|| unicode2native_error(&UNICODE2NATIVE_ERROR_INTERNAL))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| unicode2native_error(&UNICODE2NATIVE_ERROR_INTERNAL))?;
    for unit in text.encode_utf16() {
        let encoded = if little_endian {
            unit.to_le_bytes()
        } else {
            unit.to_be_bytes()
        };
        bytes.extend_from_slice(&encoded);
    }
    Ok(bytes)
}

fn encode_utf32(text: &str, little_endian: bool, bom: bool) -> BuiltinResult<Vec<u8>> {
    let char_count = text.chars().count();
    let capacity = char_count
        .checked_add(usize::from(bom))
        .and_then(|units| units.checked_mul(4))
        .ok_or_else(|| unicode2native_error(&UNICODE2NATIVE_ERROR_INTERNAL))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| unicode2native_error(&UNICODE2NATIVE_ERROR_INTERNAL))?;
    if bom {
        if little_endian {
            bytes.extend_from_slice(&[0xFF, 0xFE, 0x00, 0x00]);
        } else {
            bytes.extend_from_slice(&[0x00, 0x00, 0xFE, 0xFF]);
        }
    }
    for ch in text.chars() {
        let encoded = if little_endian {
            (ch as u32).to_le_bytes()
        } else {
            (ch as u32).to_be_bytes()
        };
        bytes.extend_from_slice(&encoded);
    }
    Ok(bytes)
}

fn value_to_text_scalar(value: &Value) -> BuiltinResult<std::borrow::Cow<'_, str>> {
    match value {
        Value::String(text) => text
            .as_str()
            .map(std::borrow::Cow::Borrowed)
            .ok_or_else(|| unicode2native_error(&UNICODE2NATIVE_ERROR_INVALID_INPUT)),
        Value::StringArray(array) => string_array_scalar(array),
        Value::CharArray(array) if array.rows <= 1 => Ok(std::borrow::Cow::Owned(
            char_row_or_empty(array, &UNICODE2NATIVE_ERROR_INVALID_INPUT)?,
        )),
        Value::CharArray(_) => Err(unicode2native_error_with_message(
            "unicode2native: character input must be a row vector",
            &UNICODE2NATIVE_ERROR_INVALID_INPUT,
        )),
        _ => Err(unicode2native_error(&UNICODE2NATIVE_ERROR_INVALID_INPUT)),
    }
}

fn value_to_encoding_name(value: &Value) -> BuiltinResult<std::borrow::Cow<'_, str>> {
    match value {
        Value::String(text) => text
            .as_str()
            .map(std::borrow::Cow::Borrowed)
            .ok_or_else(|| unicode2native_error(&UNICODE2NATIVE_ERROR_INVALID_INPUT)),
        Value::StringArray(array) => string_array_scalar(array),
        Value::CharArray(array) if array.rows <= 1 => Ok(std::borrow::Cow::Owned(
            char_row_or_empty(array, &UNICODE2NATIVE_ERROR_INVALID_ENCODING)?,
        )),
        Value::CharArray(_) => Err(unicode2native_error_with_message(
            "unicode2native: encoding must be a character row vector",
            &UNICODE2NATIVE_ERROR_INVALID_ENCODING,
        )),
        _ => Err(unicode2native_error_with_message(
            "unicode2native: encoding must be a string scalar or character vector",
            &UNICODE2NATIVE_ERROR_INVALID_ENCODING,
        )),
    }
}

fn string_array_scalar(array: &StringArray) -> BuiltinResult<std::borrow::Cow<'_, str>> {
    if array.data().len() == 1 {
        array.data()[0]
            .as_str()
            .map(std::borrow::Cow::Borrowed)
            .ok_or_else(|| unicode2native_error(&UNICODE2NATIVE_ERROR_INVALID_INPUT))
    } else {
        Err(unicode2native_error_with_message(
            "unicode2native: string input must be scalar",
            &UNICODE2NATIVE_ERROR_INVALID_INPUT,
        ))
    }
}

fn char_row_or_empty(
    array: &CharArray,
    descriptor: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<String> {
    if array.rows == 0 {
        Ok(String::new())
    } else {
        char_row_to_string(array, 0).map_err(|source| {
            let mut builder =
                build_runtime_error("unicode2native: malformed UTF-16 character input")
                    .with_builtin(BUILTIN_NAME)
                    .with_source(source);
            if let Some(identifier) = descriptor.identifier {
                builder = builder.with_identifier(identifier);
            }
            builder.build()
        })
    }
}

fn bytes_to_uint8_row(bytes: Vec<u8>) -> BuiltinResult<Value> {
    let len = bytes.len();
    let tensor = Tensor::new_integer(IntegerStorage::U8(bytes), vec![1, len])
        .map_err(|_| unicode2native_error(&UNICODE2NATIVE_ERROR_INTERNAL))?;
    Ok(Value::Tensor(tensor))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{infer_catalog_call, UNICODE2NATIVE_CATALOG_ENTRY};
    use runmat_types::{
        CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
        RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
    };
    use runmat_value::NumericDType;

    fn unicode2native_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::unicode2native_builtin(
            value,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }

    fn tensor_bytes(value: Value) -> (Vec<u8>, Vec<usize>, NumericDType) {
        match value {
            Value::Tensor(tensor) => {
                let bytes = match tensor.integer_storage() {
                    Some(IntegerStorage::U8(bytes)) => bytes.clone(),
                    other => panic!("expected native uint8 storage, got {other:?}"),
                };
                let dtype = tensor.numeric_dtype();
                (bytes, tensor.shape().to_vec(), dtype)
            }
            other => panic!("expected uint8 tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unicode2native_defaults_to_utf8_row_vector() {
        let (bytes, shape, dtype) =
            tensor_bytes(unicode2native_builtin(Value::String("hé".into()), vec![]).unwrap());
        assert_eq!(bytes, vec![104, 195, 169]);
        assert_eq!(shape, vec![1, 3]);
        assert_eq!(dtype, NumericDType::U8);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unicode2native_accepts_char_vector_and_latin1_alias() {
        let text = Value::CharArray(CharArray::new_row("café"));
        let encoding = Value::CharArray(CharArray::new_row("latin1"));
        let (bytes, shape, dtype) =
            tensor_bytes(unicode2native_builtin(text, vec![encoding]).unwrap());
        assert_eq!(bytes, vec![99, 97, 102, 233]);
        assert_eq!(shape, vec![1, 4]);
        assert_eq!(dtype, NumericDType::U8);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unicode2native_accepts_shift_jis_aliases() {
        let encoding = Value::String("Shift_JIS".into());
        let (bytes, shape, dtype) = tensor_bytes(
            unicode2native_builtin(Value::String("漢".into()), vec![encoding]).unwrap(),
        );
        assert_eq!(bytes, vec![138, 191]);
        assert_eq!(shape, vec![1, 2]);
        assert_eq!(dtype, NumericDType::U8);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unicode2native_utf16_emits_little_endian_bom() {
        let (bytes, shape, dtype) = tensor_bytes(
            unicode2native_builtin(
                Value::String("hi".into()),
                vec![Value::String("UTF-16".into())],
            )
            .unwrap(),
        );
        assert_eq!(bytes, vec![255, 254, 104, 0, 105, 0]);
        assert_eq!(shape, vec![1, 6]);
        assert_eq!(dtype, NumericDType::U8);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unicode2native_utf16le_and_utf16be_skip_bom() {
        let (le, le_shape, le_dtype) = tensor_bytes(
            unicode2native_builtin(
                Value::String("A".into()),
                vec![Value::String("UTF-16LE".into())],
            )
            .unwrap(),
        );
        assert_eq!(le, vec![65, 0]);
        assert_eq!(le_shape, vec![1, 2]);
        assert_eq!(le_dtype, NumericDType::U8);

        let (be, be_shape, be_dtype) = tensor_bytes(
            unicode2native_builtin(
                Value::String("A".into()),
                vec![Value::String("UTF-16BE".into())],
            )
            .unwrap(),
        );
        assert_eq!(be, vec![0, 65]);
        assert_eq!(be_shape, vec![1, 2]);
        assert_eq!(be_dtype, NumericDType::U8);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unicode2native_ascii_replaces_unencodable_text() {
        let (bytes, shape, dtype) = tensor_bytes(
            unicode2native_builtin(
                Value::String("é".into()),
                vec![Value::String("US-ASCII".into())],
            )
            .unwrap(),
        );
        assert_eq!(bytes, vec![63]);
        assert_eq!(shape, vec![1, 1]);
        assert_eq!(dtype, NumericDType::U8);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unicode2native_windows1252_replaces_unencodable_text_with_question_mark() {
        let (bytes, shape, dtype) = tensor_bytes(
            unicode2native_builtin(
                Value::String("€☃".into()),
                vec![Value::String("windows_1252".into())],
            )
            .unwrap(),
        );
        assert_eq!(bytes, vec![128, 63]);
        assert_eq!(shape, vec![1, 2]);
        assert_eq!(dtype, NumericDType::U8);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unicode2native_utf32_aliases_encode_unicode_code_points() {
        let (bytes, shape, dtype) = tensor_bytes(
            unicode2native_builtin(
                Value::String("A".into()),
                vec![Value::String("UTF_32".into())],
            )
            .unwrap(),
        );
        assert_eq!(bytes, vec![255, 254, 0, 0, 65, 0, 0, 0]);
        assert_eq!(shape, vec![1, 8]);
        assert_eq!(dtype, NumericDType::U8);

        let (be, be_shape, be_dtype) = tensor_bytes(
            unicode2native_builtin(
                Value::String("A".into()),
                vec![Value::String("UTF-32BE".into())],
            )
            .unwrap(),
        );
        assert_eq!(be, vec![0, 0, 0, 65]);
        assert_eq!(be_shape, vec![1, 4]);
        assert_eq!(be_dtype, NumericDType::U8);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unicode2native_empty_text_preserves_uint8_empty_row() {
        let (bytes, shape, dtype) = tensor_bytes(
            unicode2native_builtin(Value::String(String::new().into()), vec![]).unwrap(),
        );
        assert!(bytes.is_empty());
        assert_eq!(shape, vec![1, 0]);
        assert_eq!(dtype, NumericDType::U8);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unicode2native_rejects_string_arrays_that_are_not_scalar() {
        let array = StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap();
        let err = unicode2native_builtin(Value::StringArray(array), vec![]).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:unicode2native:InvalidInput"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unicode2native_rejects_char_matrices() {
        let chars = CharArray::from_unicode_scalars(vec!['a', 'b', 'c', 'd'], 2, 2).unwrap();
        let err = unicode2native_builtin(Value::CharArray(chars), vec![]).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:unicode2native:InvalidInput"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unicode2native_rejects_unknown_encoding() {
        let err = unicode2native_builtin(
            Value::String("x".into()),
            vec![Value::String("not-real".into())],
        )
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            Some("RunMat:unicode2native:InvalidEncoding")
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unicode2native_rejects_extra_arguments() {
        let err = unicode2native_builtin(
            Value::String("x".into()),
            vec![Value::String("UTF-8".into()), Value::String("extra".into())],
        )
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            Some("RunMat:unicode2native:InvalidArgument")
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unicode2native_descriptor_covers_documented_forms() {
        let labels: Vec<&str> = UNICODE2NATIVE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert_eq!(
            labels,
            vec![
                "bytes = unicode2native(unicodestr)",
                "bytes = unicode2native(unicodestr, encoding)"
            ]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unicode2native_catalog_infers_native_byte_row_for_text() {
        let request = CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::String)],
            literals: LiteralContext::new(Vec::new()),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        };
        let inferred = infer_catalog_call(&UNICODE2NATIVE_CATALOG_ENTRY, &request);
        assert!(
            inferred.diagnostics.is_empty(),
            "{:?}",
            inferred.diagnostics
        );
        assert_eq!(inferred.outputs.len(), 1);
        assert_eq!(
            inferred.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt8,
                domain: NumericDomain::Real,
            })
        );
        assert_eq!(
            inferred.outputs[0].shape,
            ShapeFact::from(vec![Some(1), None])
        );
        assert_eq!(inferred.outputs[0].storage, StorageFact::Dense);
    }
}
