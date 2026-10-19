//! Strict source-byte decoding before UTF-8 event projection.
//! This intentionally retains the decoded source for xmlread's public Text field.

use super::reader::CooperativeXmlBuf;
use super::*;
use encoding_rs::{DecoderResult, Encoding, UTF_16BE, UTF_16LE, UTF_8};
use quick_xml::events::Event;
use quick_xml::Reader;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SourceSignature {
    Utf8,
    Utf16Le,
    Utf16Be,
    Unmarked,
}

#[derive(Debug)]
enum XmlSourceEncodingError {
    UnsupportedSignature,
    UnsupportedLabel(String),
    DeclarationMismatch(String),
    MalformedBytes(&'static str),
}

impl std::fmt::Display for XmlSourceEncodingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSignature => formatter.write_str("unsupported XML byte signature"),
            Self::UnsupportedLabel(label) => {
                write!(formatter, "unsupported XML encoding '{label}'")
            }
            Self::DeclarationMismatch(label) => write!(
                formatter,
                "XML declaration encoding '{label}' conflicts with source bytes"
            ),
            Self::MalformedBytes(label) => write!(formatter, "invalid bytes in {label} XML source"),
        }
    }
}

impl std::error::Error for XmlSourceEncodingError {}

pub(in crate::builtins::io::repl_fs::xml) async fn decode_document_bytes(
    bytes: &[u8],
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<String> {
    let (signature, bom_bytes) = signature(bytes)?;
    let payload = &bytes[bom_bytes..];
    let preliminary = match signature {
        SourceSignature::Utf16Le => {
            Some(decode_chunks(payload, UTF_16LE, bom_bytes > 0, cooperation).await?)
        }
        SourceSignature::Utf16Be => {
            Some(decode_chunks(payload, UTF_16BE, bom_bytes > 0, cooperation).await?)
        }
        _ => None,
    };
    let declaration = match preliminary.as_deref() {
        Some(text) => first_declaration(text.as_bytes(), cooperation).await?,
        None => first_declaration(payload, cooperation).await?,
    };
    let selected = choose_encoding(
        signature,
        declaration
            .as_ref()
            .and_then(|decl| decl.encoding.as_deref()),
    )?;
    let text = match preliminary {
        Some(text) => text,
        None => decode_chunks(payload, selected, bom_bytes > 0, cooperation).await?,
    };
    Ok(text)
}

fn signature(bytes: &[u8]) -> BuiltinResult<(SourceSignature, usize)> {
    if bytes.starts_with(&[0, 0, 0xFE, 0xFF])
        || bytes.starts_with(&[0xFF, 0xFE, 0, 0])
        || bytes.starts_with(&[0, 0, 0, b'<'])
        || bytes.starts_with(&[b'<', 0, 0, 0])
    {
        return Err(source_error(XmlSourceEncodingError::UnsupportedSignature));
    }
    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        Ok((SourceSignature::Utf8, 3))
    } else if bytes.starts_with(&[0xFF, 0xFE]) {
        Ok((SourceSignature::Utf16Le, 2))
    } else if bytes.starts_with(&[0xFE, 0xFF]) {
        Ok((SourceSignature::Utf16Be, 2))
    } else if bytes.starts_with(&[b'<', 0, b'?', 0]) {
        Ok((SourceSignature::Utf16Le, 0))
    } else if bytes.starts_with(&[0, b'<', 0, b'?']) {
        Ok((SourceSignature::Utf16Be, 0))
    } else {
        Ok((SourceSignature::Unmarked, 0))
    }
}

async fn first_declaration(
    bytes: &[u8],
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<Option<super::declaration::XmlDeclaration>> {
    let mut reader = Reader::from_reader(CooperativeXmlBuf::new(bytes, cooperation.budget_owner()));
    let mut buffer = Vec::new();
    match reader.read_event_into_async(&mut buffer).await {
        Ok(Event::Decl(declaration)) => Ok(Some(
            super::declaration::parse(&declaration, cooperation).await?,
        )),
        Ok(_) => Ok(None),
        Err(error) => Err(super::events::quick_xml_error(error, cooperation)),
    }
}

fn choose_encoding(
    signature: SourceSignature,
    declared: Option<&str>,
) -> BuiltinResult<&'static Encoding> {
    let Some(label) = declared else {
        return Ok(match signature {
            SourceSignature::Utf16Le => UTF_16LE,
            SourceSignature::Utf16Be => UTF_16BE,
            _ => UTF_8,
        });
    };
    let encoding = Encoding::for_label_no_replacement(label.as_bytes())
        .ok_or_else(|| source_error(XmlSourceEncodingError::UnsupportedLabel(label.to_owned())))?;
    let matching = match signature {
        SourceSignature::Utf8 => encoding == UTF_8,
        SourceSignature::Utf16Le => encoding == UTF_16LE,
        SourceSignature::Utf16Be => encoding == UTF_16BE || label.eq_ignore_ascii_case("utf-16"),
        SourceSignature::Unmarked => encoding != UTF_16LE && encoding != UTF_16BE,
    };
    if !matching {
        return Err(source_error(XmlSourceEncodingError::DeclarationMismatch(
            label.to_owned(),
        )));
    }
    Ok(match signature {
        SourceSignature::Utf16Be => UTF_16BE,
        _ => encoding,
    })
}

async fn decode_chunks(
    bytes: &[u8],
    encoding: &'static Encoding,
    preserve_bom: bool,
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<String> {
    let mut decoder = encoding.new_decoder_without_bom_handling();
    let mut result = String::new();
    if preserve_bom {
        result.try_reserve('\u{FEFF}'.len_utf8()).map_err(|error| {
            build_runtime_error("xmlread: cannot reserve decoded XML source")
                .with_builtin("xmlread")
                .with_source(error)
                .build()
        })?;
        result.push('\u{FEFF}');
    }
    let mut position = 0usize;
    loop {
        let end = position.saturating_add(4096).min(bytes.len());
        let last = end == bytes.len();
        let mut consumed = 0usize;
        loop {
            let mut out = [0u8; 16 * 1024];
            let (status, read, written) = decoder.decode_to_utf8_without_replacement(
                &bytes[position + consumed..end],
                &mut out,
                last,
            );
            consumed += read;
            let fragment = std::str::from_utf8(&out[..written]).map_err(|error| {
                build_runtime_error("xmlread: decoder produced invalid UTF-8")
                    .with_builtin("xmlread")
                    .with_source(error)
                    .build()
            })?;
            result.try_reserve(fragment.len()).map_err(|error| {
                build_runtime_error("xmlread: cannot reserve decoded XML source")
                    .with_builtin("xmlread")
                    .with_source(error)
                    .build()
            })?;
            result.push_str(fragment);
            cooperation.completed(read + written).await?;
            match status {
                DecoderResult::InputEmpty => break,
                DecoderResult::OutputFull => continue,
                DecoderResult::Malformed(_, _) => {
                    return Err(source_error(XmlSourceEncodingError::MalformedBytes(
                        encoding.name(),
                    )));
                }
            }
        }
        position += consumed;
        if last {
            break;
        }
    }
    Ok(result)
}

fn source_error(error: XmlSourceEncodingError) -> RuntimeError {
    build_runtime_error(format!("xmlread: malformed XML ({error})"))
        .with_builtin("xmlread")
        .with_source(error)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoded(raw: &[u8]) -> BuiltinResult<String> {
        let mut cooperation = XmlCooperation::new(None, XmlOperation::XmlRead);
        futures::executor::block_on(decode_document_bytes(raw, &mut cooperation))
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn utf8_bom_is_retained_in_public_source_but_parser_consumes_it() {
        let text = decoded(b"\xEF\xBB\xBF<?xml version='1.0' encoding='UTF-8'?><root/>")
            .expect("UTF-8 BOM source");
        assert!(text.starts_with('\u{FEFF}'));
        let mut cooperation = XmlCooperation::new(None, XmlOperation::XmlRead);
        let root = futures::executor::block_on(
            super::super::events::parse_xml_document_with_context(&text, &mut cooperation),
        )
        .expect("BOM is ignored by event projection");
        assert_eq!(root.name, "root");
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn utf16_bom_and_declared_legacy_encoding_decode_to_unicode() {
        let le = "<?xml version='1.0' encoding='UTF-16'?><root>温度</root>";
        let mut little = vec![0xFF, 0xFE];
        for unit in le.encode_utf16() {
            little.extend_from_slice(&unit.to_le_bytes());
        }
        let decoded_le = decoded(&little).expect("UTF-16LE source");
        assert_eq!(decoded_le, format!("\u{FEFF}{le}"));
        assert_eq!(decoded(&little[2..]).expect("UTF-16LE signature"), le);

        let be = "<?xml version='1.0' encoding='UTF-16BE'?><root>é</root>";
        let mut big = vec![0xFE, 0xFF];
        for unit in be.encode_utf16() {
            big.extend_from_slice(&unit.to_be_bytes());
        }
        assert_eq!(
            decoded(&big).expect("UTF-16BE source"),
            format!("\u{FEFF}{be}")
        );

        let legacy = b"<?xml version='1.0' encoding='windows-1252'?><root>caf\xE9</root>";
        assert_eq!(
            decoded(legacy).expect("declared legacy encoding"),
            "<?xml version='1.0' encoding='windows-1252'?><root>café</root>"
        );
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn mismatched_or_unsupported_source_encoding_retains_typed_cause() {
        for raw in [
            b"\xEF\xBB\xBF<?xml version='1.0' encoding='UTF-16'?><root/>".as_slice(),
            b"<?xml version='1.0' encoding='not-an-encoding'?><root/>".as_slice(),
            b"<?xml version='1.0'?><root>\xFF</root>".as_slice(),
            b"\x00\x00\xFE\xFF".as_slice(),
        ] {
            let error = decoded(raw).expect_err("invalid encoded source");
            assert!(std::error::Error::source(&error)
                .expect("typed source")
                .downcast_ref::<XmlSourceEncodingError>()
                .is_some());
        }
        let mut wrong_endian = vec![0xFE, 0xFF];
        for unit in "<?xml version='1.0' encoding='UTF-16LE'?><root/>".encode_utf16() {
            wrong_endian.extend_from_slice(&unit.to_be_bytes());
        }
        assert!(decoded(&wrong_endian).is_err());
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn long_declared_source_observes_invocation_cancellation() {
        use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
        use runmat_execution::cancellation::CancellationSource;
        use runmat_execution::CancellationReason;
        use std::num::NonZeroU64;

        let source = CancellationSource::new();
        let canceller = source.clone();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            move || {
                canceller.cancel(CancellationReason::User);
                Box::pin(async { Ok(()) })
            },
        );
        let mut cooperation = XmlCooperation::with_budget(budget, XmlOperation::XmlRead);
        let mut bytes = b"<?xml version='1.0' encoding='windows-1252'?><root>".to_vec();
        bytes.extend(std::iter::repeat_n(b'x', 16 * 1024));
        bytes.extend_from_slice(b"</root>");
        let error = futures::executor::block_on(decode_document_bytes(&bytes, &mut cooperation))
            .expect_err("source decoding must cooperate");
        assert_eq!(error.identifier(), Some("RunMat:xmlread:Cancelled"));
    }
}
