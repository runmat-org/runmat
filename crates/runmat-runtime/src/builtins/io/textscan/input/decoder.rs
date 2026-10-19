use super::*;
use crate::builtins::io::filetext::helpers::decode_bytes;
use encoding_rs::{Encoding, SHIFT_JIS, UTF_16BE, UTF_16LE, UTF_8, WINDOWS_1252};

enum PositionEncoding {
    Utf8 { prefix: usize },
    Utf16 { prefix: usize, little_endian: bool },
    Latin1,
    Ascii,
    Windows1252,
    ShiftJis,
}

impl PositionEncoding {
    fn admit(label: &str, bytes: &[u8]) -> BuiltinResult<Self> {
        let label = label.trim();
        if Encoding::for_label(label.as_bytes()) == Some(SHIFT_JIS) {
            return Ok(Self::ShiftJis);
        }
        if label.is_empty()
            || label.eq_ignore_ascii_case("utf-8")
            || label.eq_ignore_ascii_case("utf8")
            || label.eq_ignore_ascii_case("windows-1252")
            || label.eq_ignore_ascii_case("cp1252")
        {
            if let Some((encoding, prefix)) = Encoding::for_bom(bytes) {
                if encoding == UTF_8 {
                    return Ok(Self::Utf8 { prefix });
                }
                if encoding == UTF_16LE || encoding == UTF_16BE {
                    return Ok(Self::Utf16 {
                        prefix,
                        little_endian: encoding == UTF_16LE,
                    });
                }
            }
            return if label.eq_ignore_ascii_case("windows-1252")
                || label.eq_ignore_ascii_case("cp1252")
            {
                Ok(Self::Windows1252)
            } else {
                Ok(Self::Utf8 { prefix: 0 })
            };
        }
        if ["binary", "latin1", "latin-1", "iso-8859-1"]
            .iter()
            .any(|name| label.eq_ignore_ascii_case(name))
        {
            return Ok(Self::Latin1);
        }
        if ["us-ascii", "ascii", "us_ascii", "usascii"]
            .iter()
            .any(|name| label.eq_ignore_ascii_case(name))
        {
            return Ok(Self::Ascii);
        }
        Err(textscan_error_with(
            &TEXTSCAN_ERROR_FILE,
            format!(
                "textscan: file-position preserving reads do not yet support encoding '{label}'"
            ),
        ))
    }
}

#[derive(Debug)]
pub(super) struct DecodedFileText {
    pub(super) text: String,
    text_offsets: Vec<usize>,
    byte_offsets: Vec<usize>,
    byte_len: usize,
}

impl DecodedFileText {
    pub(super) async fn decode(
        bytes: &[u8],
        encoding: &str,
        cooperation: &mut TextscanCooperation,
    ) -> BuiltinResult<Self> {
        let mut decoded = Self {
            text: String::new(),
            text_offsets: Vec::new(),
            byte_offsets: Vec::new(),
            byte_len: bytes.len(),
        };
        match PositionEncoding::admit(encoding, bytes)? {
            PositionEncoding::Utf8 { prefix } => {
                decoded.decode_utf8(bytes, prefix, cooperation).await?
            }
            PositionEncoding::Utf16 {
                prefix,
                little_endian,
            } => {
                decoded
                    .decode_utf16(bytes, prefix, little_endian, cooperation)
                    .await?
            }
            PositionEncoding::Latin1 => decoded.decode_latin1(bytes, false, cooperation).await?,
            PositionEncoding::Ascii => decoded.decode_latin1(bytes, true, cooperation).await?,
            PositionEncoding::Windows1252 => decoded.decode_windows1252(bytes, cooperation).await?,
            PositionEncoding::ShiftJis => {
                decoded
                    .decode_shift_jis(bytes, encoding, cooperation)
                    .await?
            }
        }
        Ok(decoded)
    }

    pub(super) fn byte_offset_for_text_pos(&self, text_pos: usize) -> BuiltinResult<usize> {
        if text_pos == self.text.len() {
            return Ok(self.byte_len);
        }
        match self.text_offsets.binary_search(&text_pos) {
            Ok(idx) => Ok(self.byte_offsets[idx]),
            Err(_) => Err(textscan_error_with(
                &TEXTSCAN_ERROR_FILE,
                "textscan: parsed position did not fall on a decoded character boundary",
            )),
        }
    }

    fn push(&mut self, ch: char, source_offset: usize) {
        self.text_offsets.push(self.text.len());
        self.byte_offsets.push(source_offset);
        self.text.push(ch);
    }

    async fn decode_utf8(
        &mut self,
        bytes: &[u8],
        prefix: usize,
        cooperation: &mut TextscanCooperation,
    ) -> BuiltinResult<()> {
        let mut offset = prefix;
        while offset < bytes.len() {
            let mut end = offset.saturating_add(4096).min(bytes.len());
            for _ in 0..3 {
                if end < bytes.len() && bytes[end] & 0xC0 == 0x80 {
                    end += 1;
                } else {
                    break;
                }
            }
            let chunk = std::str::from_utf8(&bytes[offset..end]).map_err(|source| {
                textscan_error_with_source(
                    &TEXTSCAN_ERROR_FILE,
                    "textscan: unable to decode bytes using encoding 'UTF-8'",
                    source,
                )
            })?;
            for (start, ch) in chunk.char_indices() {
                self.push(ch, offset + start);
            }
            cooperation.completed(end - offset).await?;
            offset = end;
        }
        Ok(())
    }

    async fn decode_utf16(
        &mut self,
        bytes: &[u8],
        prefix: usize,
        little_endian: bool,
        cooperation: &mut TextscanCooperation,
    ) -> BuiltinResult<()> {
        let mut offset = prefix;
        let mut work = 0usize;
        while offset < bytes.len() {
            let first = utf16_unit(bytes, offset, little_endian)?;
            let (scalar, width) = if (0xD800..=0xDBFF).contains(&first) {
                let second = utf16_unit(bytes, offset + 2, little_endian)?;
                if !(0xDC00..=0xDFFF).contains(&second) {
                    return Err(invalid_utf16());
                }
                (
                    0x10000 + ((u32::from(first) - 0xD800) << 10) + u32::from(second) - 0xDC00,
                    4,
                )
            } else if (0xDC00..=0xDFFF).contains(&first) {
                return Err(invalid_utf16());
            } else {
                (u32::from(first), 2)
            };
            let ch = char::from_u32(scalar).ok_or_else(invalid_utf16)?;
            self.push(ch, offset);
            offset += width;
            work += width;
            if work >= 4096 {
                cooperation.completed(work).await?;
                work = 0;
            }
        }
        cooperation.completed(work).await?;
        Ok(())
    }

    async fn decode_latin1(
        &mut self,
        bytes: &[u8],
        ascii_only: bool,
        cooperation: &mut TextscanCooperation,
    ) -> BuiltinResult<()> {
        for (offset, &byte) in bytes.iter().enumerate() {
            if ascii_only && byte > 0x7F {
                return Err(textscan_error_with(
                    &TEXTSCAN_ERROR_FILE,
                    format!("textscan: byte value {byte} is outside the ASCII range"),
                ));
            }
            self.push(char::from(byte), offset);
            if (offset + 1) % 4096 == 0 {
                cooperation.completed(4096).await?;
            }
        }
        cooperation.completed(bytes.len() % 4096).await?;
        Ok(())
    }

    async fn decode_windows1252(
        &mut self,
        bytes: &[u8],
        cooperation: &mut TextscanCooperation,
    ) -> BuiltinResult<()> {
        for (chunk_index, chunk) in bytes.chunks(4096).enumerate() {
            let (text, had_errors) = WINDOWS_1252.decode_without_bom_handling(chunk);
            if had_errors {
                return Err(textscan_error_with(
                    &TEXTSCAN_ERROR_FILE,
                    "textscan: unable to decode bytes using encoding 'windows-1252'",
                ));
            }
            let start = chunk_index * 4096;
            for (index, ch) in text.chars().enumerate() {
                self.push(ch, start + index);
            }
            cooperation.completed(chunk.len()).await?;
        }
        Ok(())
    }

    async fn decode_shift_jis(
        &mut self,
        bytes: &[u8],
        encoding: &str,
        cooperation: &mut TextscanCooperation,
    ) -> BuiltinResult<()> {
        let mut offset = 0usize;
        let mut work = 0usize;
        while offset < bytes.len() {
            let width = shift_jis_unit_width(bytes, offset)?;
            let decoded = decode_bytes(&bytes[offset..offset + width], encoding, BUILTIN_NAME)
                .map_err(|source| {
                    let message = source.message().to_string();
                    textscan_error_with_source(&TEXTSCAN_ERROR_FILE, message, source)
                })?;
            for ch in decoded {
                self.push(ch, offset);
            }
            offset += width;
            work += width;
            if work >= 4096 {
                cooperation.completed(work).await?;
                work = 0;
            }
        }
        cooperation.completed(work).await?;
        Ok(())
    }
}

fn utf16_unit(bytes: &[u8], offset: usize, little_endian: bool) -> BuiltinResult<u16> {
    let Some(pair) = bytes.get(offset..offset.saturating_add(2)) else {
        return Err(invalid_utf16());
    };
    let pair = [pair[0], pair[1]];
    Ok(if little_endian {
        u16::from_le_bytes(pair)
    } else {
        u16::from_be_bytes(pair)
    })
}

fn invalid_utf16() -> RuntimeError {
    textscan_error_with(
        &TEXTSCAN_ERROR_FILE,
        "textscan: unable to decode bytes using UTF-16",
    )
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
