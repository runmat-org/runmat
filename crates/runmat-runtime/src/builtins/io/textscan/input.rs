use super::*;

mod decoder;
use decoder::DecodedFileText;

/// Keeps the exact registered file checked out through output publication.
/// Dropping this before `commit_file_position` restores its entry cursor.
pub(super) struct ParsedInput {
    pub(super) columns: Vec<ColumnData>,
    file_operation: Option<registry::FileOperation>,
}

impl ParsedInput {
    pub(super) fn check_authority(
        &self,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<()> {
        if let Some(operation) = &self.file_operation {
            operation.check_authority(runtime).map_err(|source| {
                textscan_error_with_source(
                    &TEXTSCAN_ERROR_FILE,
                    "textscan: file authority invalidated",
                    source,
                )
            })?;
        }
        Ok(())
    }

    pub(super) fn commit_file_position(&mut self) {
        if let Some(operation) = self.file_operation.as_mut() {
            operation.commit_cursor();
        }
    }
}

pub(super) async fn parse_input(
    runtime: Option<&crate::context::RuntimeContext>,
    value: &Value,
    format_spec: &[FormatElement],
    repeat: Option<usize>,
    options: &TextscanOptions,
    cooperation: &mut TextscanCooperation,
) -> BuiltinResult<ParsedInput> {
    if let Some(fid) = numeric_fid(value) {
        return parse_registered_file_in(runtime, fid, format_spec, repeat, options, cooperation)
            .await;
    }
    let text = string_scalar(value, "textOrFileID")?;
    let parsed = parse_textscan(&text, format_spec, repeat, options, cooperation).await?;
    Ok(ParsedInput {
        columns: parsed.columns,
        file_operation: None,
    })
}

#[cfg(test)]
pub(super) async fn parse_registered_file(
    fid: i32,
    format_spec: &[FormatElement],
    repeat: Option<usize>,
    options: &TextscanOptions,
    cooperation: &mut TextscanCooperation,
) -> BuiltinResult<ParsedInput> {
    parse_registered_file_in(None, fid, format_spec, repeat, options, cooperation).await
}

pub(super) async fn parse_registered_file_in(
    runtime: Option<&crate::context::RuntimeContext>,
    fid: i32,
    format_spec: &[FormatElement],
    repeat: Option<usize>,
    options: &TextscanOptions,
    cooperation: &mut TextscanCooperation,
) -> BuiltinResult<ParsedInput> {
    validate_fid(fid)?;
    let mut operation = registry::acquire_file_in(fid, runtime).map_err(|error| match error {
        registry::FileOperationError::InvalidIdentifier
        | registry::FileOperationError::ForeignSession => textscan_error_with(
            &TEXTSCAN_ERROR_FILE,
            format!("textscan: invalid file identifier {fid}"),
        ),
        other => {
            let source = other.into_io_error();
            textscan_error_with_source(
                &TEXTSCAN_ERROR_FILE,
                format!("textscan: file identifier {fid} is unavailable ({source})"),
                source,
            )
        }
    })?;
    if !operation.info.access.read {
        return Err(textscan_error_with(
            &TEXTSCAN_ERROR_FILE,
            format!("textscan: file identifier {fid} is not open for reading"),
        ));
    }
    let start = operation.file_mut().stream_position().map_err(|err| {
        textscan_error_with_source(
            &TEXTSCAN_ERROR_FILE,
            format!("textscan: unable to seek file identifier {fid} ({err})"),
            err,
        )
    })?;
    operation.restore_cursor_on_drop(start);
    let mut bytes = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        cooperation.completed(0).await?;
        operation.check_authority(runtime).map_err(|source| {
            textscan_error_with_source(
                &TEXTSCAN_ERROR_FILE,
                "textscan: file authority invalidated",
                source,
            )
        })?;
        let count = operation.file_mut().read(&mut chunk).map_err(|err| {
            textscan_error_with_source(
                &TEXTSCAN_ERROR_FILE,
                format!("textscan: unable to read from file identifier {fid} ({err})"),
                err,
            )
        })?;
        if count == 0 {
            break;
        }
        bytes.try_reserve(count).map_err(|err| {
            textscan_error_with_source(
                &TEXTSCAN_ERROR_FILE,
                "textscan: unable to reserve input buffer",
                err,
            )
        })?;
        bytes.extend_from_slice(&chunk[..count]);
        cooperation.completed(count).await?;
    }
    let encoding = if operation.info.encoding.trim().is_empty() {
        "UTF-8"
    } else {
        operation.info.encoding.as_str()
    };
    let decoded = DecodedFileText::decode(&bytes, encoding, cooperation).await?;
    cooperation.completed(bytes.len()).await?;
    let parsed = parse_textscan(&decoded.text, format_spec, repeat, options, cooperation).await?;
    let consumed_bytes = decoded.byte_offset_for_text_pos(parsed.consumed_text_pos)?;
    let target = checked_source_position(start, consumed_bytes)?;
    operation
        .file_mut()
        .seek(SeekFrom::Start(target))
        .map_err(|err| {
            textscan_error_with_source(
                &TEXTSCAN_ERROR_FILE,
                format!("textscan: unable to restore file position for identifier {fid} ({err})"),
                err,
            )
        })?;
    operation.check_authority(runtime).map_err(|source| {
        textscan_error_with_source(
            &TEXTSCAN_ERROR_FILE,
            "textscan: file authority invalidated",
            source,
        )
    })?;
    cooperation.completed(0).await?;
    Ok(ParsedInput {
        columns: parsed.columns,
        file_operation: Some(operation),
    })
}

fn checked_source_position(start: u64, consumed_bytes: usize) -> BuiltinResult<u64> {
    u64::try_from(consumed_bytes)
        .ok()
        .and_then(|consumed| start.checked_add(consumed))
        .ok_or_else(|| {
            textscan_error_with(
                &TEXTSCAN_ERROR_FILE,
                "textscan: consumed source position exceeds the file offset range",
            )
        })
}

pub(super) fn validate_fid(fid: i32) -> BuiltinResult<()> {
    if fid < 0 {
        return Err(textscan_error_with(
            &TEXTSCAN_ERROR_FILE,
            "textscan: file identifier must be non-negative",
        ));
    }
    if fid < 3 {
        return Err(textscan_error_with(
            &TEXTSCAN_ERROR_FILE,
            "textscan: standard input/output identifiers are not supported yet",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod position_tests {
    use super::*;
    use futures::executor::block_on;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn chunked_utf8_decode_retains_interior_bom_and_exact_boundary() {
        let mut bytes = vec![b'a'; 4096];
        bytes.extend_from_slice(&[0xEF, 0xBB, 0xBF, b'z']);
        let mut cooperation = TextscanCooperation::new(None);
        let decoded =
            block_on(DecodedFileText::decode(&bytes, "UTF-8", &mut cooperation)).expect("decode");
        assert_eq!(decoded.text.chars().count(), 4098);
        assert!(decoded.text.ends_with("\u{FEFF}z"));
        assert_eq!(
            decoded.byte_offset_for_text_pos(4096).expect("offset"),
            4096
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn initial_bom_only_and_four_byte_boundary_have_exact_offsets() {
        let mut cooperation = TextscanCooperation::new(None);
        let only_bom = block_on(DecodedFileText::decode(
            b"\xef\xbb\xbf",
            "UTF-8",
            &mut cooperation,
        ))
        .expect("BOM only");
        assert!(only_bom.text.is_empty());
        assert_eq!(only_bom.byte_offset_for_text_pos(0).expect("after BOM"), 3);

        let mut bytes = vec![b'a'; 4095];
        bytes.extend_from_slice("😀z".as_bytes());
        let decoded = block_on(DecodedFileText::decode(&bytes, "UTF-8", &mut cooperation))
            .expect("four-byte scalar crosses chunk edge");
        assert!(decoded.text.ends_with("😀z"));
        assert_eq!(decoded.byte_offset_for_text_pos(4095).expect("emoji"), 4095);
        assert_eq!(decoded.byte_offset_for_text_pos(4099).expect("z"), 4099);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn decoded_position_requires_exact_character_boundary_or_endpoint() {
        let mut cooperation = TextscanCooperation::new(None);
        let decoded = block_on(DecodedFileText::decode(
            "éz".as_bytes(),
            "UTF-8",
            &mut cooperation,
        ))
        .expect("decode");
        assert_eq!(decoded.byte_offset_for_text_pos(0).expect("first"), 0);
        assert_eq!(decoded.byte_offset_for_text_pos(2).expect("second"), 2);
        assert_eq!(decoded.byte_offset_for_text_pos(3).expect("endpoint"), 3);
        for invalid in [1, 4, usize::MAX] {
            assert_eq!(
                decoded
                    .byte_offset_for_text_pos(invalid)
                    .expect_err("invalid position")
                    .identifier(),
                TEXTSCAN_ERROR_FILE.identifier,
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn malformed_continuation_run_rejects_without_unbounded_extension() {
        let mut bytes = vec![b'a'; 4095];
        bytes.extend(std::iter::repeat_n(0x80, 8192));
        let mut cooperation = TextscanCooperation::new(None);
        let error = block_on(DecodedFileText::decode(&bytes, "UTF-8", &mut cooperation))
            .expect_err("invalid UTF-8");
        assert_eq!(error.identifier(), TEXTSCAN_ERROR_FILE.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bom_selected_utf16_and_windows1252_override_map_source_bytes() {
        let mut cooperation = TextscanCooperation::new(None);
        let utf16 = block_on(DecodedFileText::decode(
            &[0xFF, 0xFE, b'1', 0, b'\n', 0, b'2', 0],
            "UTF-8",
            &mut cooperation,
        ))
        .expect("UTF-16 BOM is honored by the prior decoder contract");
        assert_eq!(utf16.text, "1\n2");
        assert_eq!(utf16.byte_offset_for_text_pos(0).expect("first"), 2);
        assert_eq!(utf16.byte_offset_for_text_pos(2).expect("second"), 6);

        let override_encoding = block_on(DecodedFileText::decode(
            b"\xef\xbb\xbf\xc3\xa9z",
            "windows-1252",
            &mut cooperation,
        ))
        .expect("BOM switches encoding once");
        assert_eq!(override_encoding.text, "éz");
        assert_eq!(override_encoding.byte_offset_for_text_pos(2).expect("z"), 5);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn source_position_is_exact_or_rejected_not_clamped() {
        assert_eq!(checked_source_position(17, 3).expect("valid offset"), 20);
        assert_eq!(
            checked_source_position(u64::MAX, 0).expect("endpoint"),
            u64::MAX
        );
        let error = checked_source_position(u64::MAX, 1).expect_err("overflow");
        assert_eq!(error.identifier(), TEXTSCAN_ERROR_FILE.identifier);
    }
}
