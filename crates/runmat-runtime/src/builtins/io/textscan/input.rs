use super::*;

pub(super) fn parse_input(
    value: &Value,
    format_spec: &[FormatElement],
    repeat: Option<usize>,
    options: &TextscanOptions,
) -> BuiltinResult<Vec<ColumnData>> {
    if let Some(fid) = numeric_fid(value) {
        return parse_registered_file(fid, format_spec, repeat, options)
            .map(|parsed| parsed.columns);
    }
    let text = string_scalar(value, "textOrFileID")?;
    parse_textscan(&text, format_spec, repeat, options).map(|parsed| parsed.columns)
}

fn parse_registered_file(
    fid: i32,
    format_spec: &[FormatElement],
    repeat: Option<usize>,
    options: &TextscanOptions,
) -> BuiltinResult<ParsedTextscan> {
    validate_fid(fid)?;
    let info = registry::info_for(fid).ok_or_else(|| {
        textscan_error_with(
            &TEXTSCAN_ERROR_FILE,
            format!("textscan: invalid file identifier {fid}"),
        )
    })?;
    if !info.access.read {
        return Err(textscan_error_with(
            &TEXTSCAN_ERROR_FILE,
            format!("textscan: file identifier {fid} is not open for reading"),
        ));
    }
    let handle = registry::shared_handle(fid).ok_or_else(|| {
        textscan_error_with(
            &TEXTSCAN_ERROR_FILE,
            format!("textscan: invalid file identifier {fid}"),
        )
    })?;
    let mut guard = handle
        .lock()
        .map_err(|_| textscan_error_with(&TEXTSCAN_ERROR_FILE, "textscan: file handle poisoned"))?;
    let file = guard.as_mut().ok_or_else(|| {
        textscan_error_with(
            &TEXTSCAN_ERROR_FILE,
            format!("textscan: file identifier {fid} is closed"),
        )
    })?;
    let start = file.stream_position().map_err(|err| {
        textscan_error_with_source(
            &TEXTSCAN_ERROR_FILE,
            format!("textscan: unable to seek file identifier {fid} ({err})"),
            err,
        )
    })?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).map_err(|err| {
        textscan_error_with_source(
            &TEXTSCAN_ERROR_FILE,
            format!("textscan: unable to read from file identifier {fid} ({err})"),
            err,
        )
    })?;
    let encoding = if info.encoding.trim().is_empty() {
        "UTF-8"
    } else {
        info.encoding.as_str()
    };
    let decoded = DecodedFileText::decode(&bytes, encoding)?;
    let parsed = parse_textscan(&decoded.text, format_spec, repeat, options)?;
    let consumed_bytes = decoded.byte_offset_for_text_pos(parsed.consumed_text_pos)?;
    let target = start.saturating_add(consumed_bytes as u64);
    file.seek(SeekFrom::Start(target)).map_err(|err| {
        textscan_error_with_source(
            &TEXTSCAN_ERROR_FILE,
            format!("textscan: unable to restore file position for identifier {fid} ({err})"),
            err,
        )
    })?;
    Ok(parsed)
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

struct DecodedFileText {
    text: String,
    text_offsets: Vec<usize>,
    byte_offsets: Vec<usize>,
    byte_len: usize,
}

impl DecodedFileText {
    fn decode(bytes: &[u8], encoding: &str) -> BuiltinResult<Self> {
        if is_shift_jis_encoding(encoding) {
            return decode_shift_jis_with_offsets(bytes, encoding);
        }
        let chars = decode_bytes(bytes, encoding, BUILTIN_NAME).map_err(|err| {
            let message = err.message().to_string();
            textscan_error_with_source(&TEXTSCAN_ERROR_FILE, message, err)
        })?;
        let byte_width = byte_preserving_encoding_width(encoding)?;
        let mut text = String::new();
        let mut text_offsets = Vec::with_capacity(chars.len());
        let mut byte_offsets = Vec::with_capacity(chars.len());
        let mut byte_offset = 0usize;
        for ch in chars {
            text_offsets.push(text.len());
            byte_offsets.push(byte_offset);
            text.push(ch);
            byte_offset += byte_width.unwrap_or_else(|| ch.len_utf8());
        }
        Ok(Self {
            text,
            text_offsets,
            byte_offsets,
            byte_len: bytes.len(),
        })
    }

    fn byte_offset_for_text_pos(&self, text_pos: usize) -> BuiltinResult<usize> {
        if text_pos == self.text.len() {
            return Ok(self.byte_len);
        }
        match self.text_offsets.binary_search(&text_pos) {
            Ok(idx) => Ok(self.byte_offsets[idx]),
            Err(idx) if idx == self.text_offsets.len() => Ok(self.byte_len),
            Err(_) => Err(textscan_error_with(
                &TEXTSCAN_ERROR_FILE,
                "textscan: parsed position did not fall on a decoded character boundary",
            )),
        }
    }
}

fn decode_shift_jis_with_offsets(bytes: &[u8], encoding: &str) -> BuiltinResult<DecodedFileText> {
    let mut text = String::new();
    let mut text_offsets = Vec::new();
    let mut byte_offsets = Vec::new();
    let mut byte_offset = 0usize;
    while byte_offset < bytes.len() {
        let width = shift_jis_unit_width(bytes, byte_offset)?;
        let decoded = decode_bytes(
            &bytes[byte_offset..byte_offset + width],
            encoding,
            BUILTIN_NAME,
        )
        .map_err(|err| {
            let message = err.message().to_string();
            textscan_error_with_source(&TEXTSCAN_ERROR_FILE, message, err)
        })?;
        for ch in decoded {
            text_offsets.push(text.len());
            byte_offsets.push(byte_offset);
            text.push(ch);
        }
        byte_offset += width;
    }
    Ok(DecodedFileText {
        text,
        text_offsets,
        byte_offsets,
        byte_len: bytes.len(),
    })
}

fn shift_jis_unit_width(bytes: &[u8], offset: usize) -> BuiltinResult<usize> {
    let first = bytes[offset];
    if first <= 0x80 || (0xA1..=0xDF).contains(&first) {
        return Ok(1);
    }
    if (0x81..=0x9F).contains(&first) || (0xE0..=0xFC).contains(&first) {
        let Some(&second) = bytes.get(offset + 1) else {
            return Err(textscan_error_with(
                &TEXTSCAN_ERROR_FILE,
                "textscan: incomplete Shift_JIS character at end of file",
            ));
        };
        if (0x40..=0x7E).contains(&second) || (0x80..=0xFC).contains(&second) {
            return Ok(2);
        }
    }
    Err(textscan_error_with(
        &TEXTSCAN_ERROR_FILE,
        "textscan: invalid Shift_JIS byte sequence",
    ))
}

fn is_shift_jis_encoding(encoding: &str) -> bool {
    Encoding::for_label(encoding.trim().as_bytes()) == Some(SHIFT_JIS)
}

fn byte_preserving_encoding_width(encoding: &str) -> BuiltinResult<Option<usize>> {
    let label = encoding.trim();
    if label.is_empty() || label.eq_ignore_ascii_case("utf-8") || label.eq_ignore_ascii_case("utf8")
    {
        return Ok(None);
    }
    if is_shift_jis_encoding(label) {
        return Ok(None);
    }
    if label.eq_ignore_ascii_case("binary")
        || label.eq_ignore_ascii_case("latin1")
        || label.eq_ignore_ascii_case("latin-1")
        || label.eq_ignore_ascii_case("iso-8859-1")
        || label.eq_ignore_ascii_case("windows-1252")
        || label.eq_ignore_ascii_case("cp1252")
        || label.eq_ignore_ascii_case("us-ascii")
        || label.eq_ignore_ascii_case("ascii")
        || label.eq_ignore_ascii_case("us_ascii")
        || label.eq_ignore_ascii_case("usascii")
    {
        return Ok(Some(1));
    }
    Err(textscan_error_with(
        &TEXTSCAN_ERROR_FILE,
        format!(
            "textscan: file-position preserving reads do not yet support encoding '{encoding}'"
        ),
    ))
}
