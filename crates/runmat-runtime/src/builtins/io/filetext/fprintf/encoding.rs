pub(super) fn encode_output(text: &str, encoding: Option<&str>) -> Result<Vec<u8>, String> {
    let label = encoding
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .unwrap_or("utf-8");
    let lower = label.to_ascii_lowercase();
    let collapsed: String = lower
        .chars()
        .filter(|ch| !matches!(ch, '-' | '_' | ' '))
        .collect();
    if matches!(
        collapsed.as_str(),
        "utf8" | "unicode" | "auto" | "default" | "system"
    ) {
        Ok(text.as_bytes().to_vec())
    } else if matches!(collapsed.as_str(), "ascii" | "usascii" | "ansix341968") {
        encode_ascii(text)
    } else if matches!(
        collapsed.as_str(),
        "latin1" | "iso88591" | "cp819" | "ibm819"
    ) {
        encode_latin1(text, label)
    } else if matches!(collapsed.as_str(), "windows1252" | "cp1252" | "ansi") {
        encode_windows_1252(text, label)
    } else {
        Ok(text.as_bytes().to_vec())
    }
}

fn encode_ascii(text: &str) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::with_capacity(text.len());
    for ch in text.chars() {
        if ch as u32 > 0x7F {
            return Err(format!(
                "fprintf: character '{}' (U+{:04X}) cannot be encoded as ASCII",
                ch, ch as u32
            ));
        }
        bytes.push(ch as u8);
    }
    Ok(bytes)
}

fn encode_latin1(text: &str, label: &str) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::with_capacity(text.len());
    for ch in text.chars() {
        if ch as u32 > 0xFF {
            return Err(format!(
                "fprintf: character '{}' (U+{:04X}) cannot be encoded as {}",
                ch, ch as u32, label
            ));
        }
        bytes.push(ch as u8);
    }
    Ok(bytes)
}

fn encode_windows_1252(text: &str, label: &str) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::with_capacity(text.len());
    for ch in text.chars() {
        if let Some(byte) = windows_1252_byte(ch) {
            bytes.push(byte);
        } else {
            return Err(format!(
                "fprintf: character '{}' (U+{:04X}) cannot be encoded as {}",
                ch, ch as u32, label
            ));
        }
    }
    Ok(bytes)
}

fn windows_1252_byte(ch: char) -> Option<u8> {
    let code = ch as u32;
    if code <= 0x7F {
        return Some(code as u8);
    }
    if (0xA0..=0xFF).contains(&code) {
        return Some(code as u8);
    }
    match code {
        0x20AC => Some(0x80),
        0x201A => Some(0x82),
        0x0192 => Some(0x83),
        0x201E => Some(0x84),
        0x2026 => Some(0x85),
        0x2020 => Some(0x86),
        0x2021 => Some(0x87),
        0x02C6 => Some(0x88),
        0x2030 => Some(0x89),
        0x0160 => Some(0x8A),
        0x2039 => Some(0x8B),
        0x0152 => Some(0x8C),
        0x017D => Some(0x8E),
        0x2018 => Some(0x91),
        0x2019 => Some(0x92),
        0x201C => Some(0x93),
        0x201D => Some(0x94),
        0x2022 => Some(0x95),
        0x2013 => Some(0x96),
        0x2014 => Some(0x97),
        0x02DC => Some(0x98),
        0x2122 => Some(0x99),
        0x0161 => Some(0x9A),
        0x203A => Some(0x9B),
        0x0153 => Some(0x9C),
        0x017E => Some(0x9E),
        0x0178 => Some(0x9F),
        _ => None,
    }
}
