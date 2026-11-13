use lsp_types::Position;

/// LSP columns count UTF-16 code units, not source bytes. Invalid columns
/// inside a surrogate pair round down to the scalar's UTF-8 boundary.
pub fn position_to_offset(text: &str, position: &Position) -> usize {
    let mut start = 0;
    for (line_index, line) in text.split('\n').enumerate() {
        if line_index == position.line as usize {
            let content = line.strip_suffix('\r').unwrap_or(line);
            let mut column = 0;
            for (byte, ch) in content.char_indices() {
                let next = column + ch.len_utf16();
                if (position.character as usize) < next {
                    return start + byte;
                }
                column = next;
            }
            return start + content.len();
        }
        start += line.len() + 1;
    }
    text.len()
}

/// Offsets inside a UTF-8 scalar round down; offsets within CRLF clamp to
/// the preceding line's content end. A trailing LF introduces an empty line.
pub fn offset_to_position(text: &str, offset: usize) -> Position {
    let mut offset = offset.min(text.len());
    while !text.is_char_boundary(offset) {
        offset -= 1;
    }
    let mut start = 0;
    for (line_index, line) in text.split('\n').enumerate() {
        let end = start + line.len();
        if offset <= end {
            let content = line.strip_suffix('\r').unwrap_or(line);
            let column_end = offset.min(start + content.len());
            let column = text[start..column_end].encode_utf16().count();
            return Position::new(
                line_index.min(u32::MAX as usize) as u32,
                column.min(u32::MAX as usize) as u32,
            );
        }
        start = end + 1;
    }
    unreachable!("split includes the final line and the offset is bounded")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf16_positions_clamp_at_scalar_line_and_eof_boundaries() {
        // Text, possibly invalid byte offset, canonical LSP position, canonical byte offset.
        for (text, byte, position, canonical) in [
            ("", 99, Position::new(0, 0), 0),
            ("abc", 3, Position::new(0, 3), 3),
            ("abc", 99, Position::new(0, 3), 3),
            ("a\n", 2, Position::new(1, 0), 2),
            ("a\n\n", 3, Position::new(2, 0), 3),
            ("a\r\nβ😀z", 1, Position::new(0, 1), 1),
            ("a\r\nβ😀z", 2, Position::new(0, 1), 1),
            ("a\r\nβ😀z", 3, Position::new(1, 0), 3),
            ("a\r\nβ😀z", 4, Position::new(1, 0), 3),
            ("a\r\nβ😀z", 5, Position::new(1, 1), 5),
            ("a\r\nβ😀z", 7, Position::new(1, 1), 5),
            ("a\r\nβ😀z", 9, Position::new(1, 3), 9),
            ("a\r\nβ😀z", 99, Position::new(1, 4), 10),
        ] {
            assert_eq!(offset_to_position(text, byte), position, "{text:?}, {byte}");
            assert_eq!(position_to_offset(text, &position), canonical, "{text:?}");
            assert!(text.is_char_boundary(canonical));
        }
        for (text, position, expected) in [
            ("😀x", Position::new(0, 1), 0),
            ("a\r\nb", Position::new(0, 99), 1),
            ("a\r\nb", Position::new(99, 99), 4),
            ("a\n", Position::new(1, 99), 2),
        ] {
            assert_eq!(position_to_offset(text, &position), expected, "{text:?}");
            assert!(text.is_char_boundary(expected));
        }
    }
}
