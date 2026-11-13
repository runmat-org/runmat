//! Bounded text and attribute projection after quick-xml tokenization.

use super::*;
use quick_xml::escape::resolve_xml_entity;

pub(super) async fn append_text_bytes(
    target: &mut String,
    raw: &[u8],
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<()> {
    let mut previous_cr = false;
    append_normalized(target, raw, cooperation, false, &mut previous_cr).await
}

pub(super) async fn decode_attribute(
    raw: &[u8],
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<String> {
    let mut result = String::new();
    let mut position = 0usize;
    let mut previous_cr = false;
    while position < raw.len() {
        let next = raw[position..]
            .iter()
            .take(4096)
            .position(|byte| *byte == b'&');
        let mut plain_end = position + next.unwrap_or_else(|| (raw.len() - position).min(4096));
        while plain_end < raw.len() && raw[plain_end] & 0b1100_0000 == 0b1000_0000 {
            plain_end += 1;
        }
        append_normalized(
            &mut result,
            &raw[position..plain_end],
            cooperation,
            true,
            &mut previous_cr,
        )
        .await?;
        position = plain_end;
        if position == raw.len() {
            break;
        }
        if raw[position] == b'&' {
            position += 1;
            let start = position;
            while position < raw.len() && raw[position] != b';' {
                position += 1;
                if (position - start).is_multiple_of(4096) {
                    cooperation.completed(4096).await?;
                }
            }
            if position == raw.len() {
                return Err(compat_error(
                    "xmlread",
                    "xmlread: malformed XML (unterminated entity)",
                ));
            }
            let entity = &raw[start..position];
            position += 1;
            append_reference(&mut result, entity, cooperation).await?;
            previous_cr = false;
        }
    }
    Ok(result)
}

pub(super) async fn append_reference(
    target: &mut String,
    raw: &[u8],
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<()> {
    let value = if let Some(numeric) = raw.strip_prefix(b"#") {
        let (digits, radix) = if let Some(hex) = numeric.strip_prefix(b"x") {
            (hex, 16u32)
        } else {
            (numeric, 10u32)
        };
        if digits.is_empty() {
            return Err(compat_error(
                "xmlread",
                "xmlread: malformed XML (empty character reference)",
            ));
        }
        let mut code = 0u32;
        for (index, digit) in digits.iter().copied().enumerate() {
            let digit = (digit as char).to_digit(radix).ok_or_else(|| {
                compat_error(
                    "xmlread",
                    "xmlread: malformed XML (invalid character reference)",
                )
            })?;
            code = code
                .checked_mul(radix)
                .and_then(|value| value.checked_add(digit))
                .ok_or_else(|| {
                    compat_error(
                        "xmlread",
                        "xmlread: malformed XML (character reference overflow)",
                    )
                })?;
            if (index + 1).is_multiple_of(4096) {
                cooperation.completed(4096).await?;
            }
        }
        let character = char::from_u32(code)
            .filter(|character| xml_character(*character))
            .ok_or_else(|| {
                compat_error(
                    "xmlread",
                    "xmlread: malformed XML (invalid XML character reference)",
                )
            })?;
        let mut encoded = [0u8; 4];
        character.encode_utf8(&mut encoded).to_owned()
    } else {
        if raw.len() > 4 {
            return Err(compat_error(
                "xmlread",
                "xmlread: malformed XML (unknown XML entity)",
            ));
        }
        let name = std::str::from_utf8(raw).map_err(|error| {
            build_runtime_error("xmlread: malformed XML (invalid entity name)")
                .with_builtin("xmlread")
                .with_source(error)
                .build()
        })?;
        resolve_xml_entity(name)
            .ok_or_else(|| compat_error("xmlread", "xmlread: malformed XML (unknown XML entity)"))?
            .to_owned()
    };
    target.try_reserve(value.len()).map_err(|error| {
        build_runtime_error("xmlread: cannot reserve XML text")
            .with_builtin("xmlread")
            .with_source(error)
            .build()
    })?;
    target.push_str(&value);
    cooperation.completed(raw.len().min(4096)).await?;
    Ok(())
}

async fn append_normalized(
    target: &mut String,
    raw: &[u8],
    cooperation: &mut XmlCooperation,
    attribute: bool,
    previous_was_cr: &mut bool,
) -> BuiltinResult<()> {
    let mut position = 0usize;
    while position < raw.len() {
        let mut end = (position + 4096).min(raw.len());
        let chunk = loop {
            match std::str::from_utf8(&raw[position..end]) {
                Ok(chunk) => break chunk,
                Err(error) if error.error_len().is_none() && end < raw.len() => {
                    end = (end + 1).min(raw.len());
                }
                Err(error) => {
                    return Err(
                        build_runtime_error("xmlread: malformed XML (invalid UTF-8 text)")
                            .with_builtin("xmlread")
                            .with_source(error)
                            .build(),
                    );
                }
            }
        };
        target.try_reserve(chunk.len()).map_err(|error| {
            build_runtime_error("xmlread: cannot reserve XML text")
                .with_builtin("xmlread")
                .with_source(error)
                .build()
        })?;
        for character in chunk.chars() {
            if !xml_character(character) {
                return Err(compat_error(
                    "xmlread",
                    "xmlread: malformed XML (invalid XML character)",
                ));
            }
            if attribute && character == '<' {
                return Err(compat_error(
                    "xmlread",
                    "xmlread: malformed XML (invalid attribute character)",
                ));
            }
            if *previous_was_cr && character == '\n' {
                *previous_was_cr = false;
                continue;
            }
            if attribute && matches!(character, '\t' | '\n' | '\r') {
                target.push(' ');
            } else if character == '\r' {
                target.push('\n');
            } else {
                target.push(character);
            }
            *previous_was_cr = character == '\r';
        }
        cooperation.completed(end - position).await?;
        position = end;
    }
    Ok(())
}

fn xml_character(ch: char) -> bool {
    matches!(ch, '\t' | '\n' | '\r' | '\u{20}'..='\u{D7FF}' | '\u{E000}'..='\u{FFFD}' | '\u{10000}'..='\u{10FFFF}')
}
