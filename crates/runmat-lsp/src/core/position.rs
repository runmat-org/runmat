use lsp_types::Position;

pub fn position_to_offset(text: &str, position: &Position) -> usize {
    let mut offset = 0usize;
    let mut line = 0u32;
    for l in text.split_inclusive('\n') {
        if line == position.line {
            let col = position.character as usize;
            offset += col.min(l.len());
            return offset;
        }
        offset += l.len();
        line += 1;
    }
    offset
}

pub fn offset_to_position(text: &str, offset: usize) -> Position {
    let mut current = 0usize;
    for (line_idx, line) in text.split_inclusive('\n').enumerate() {
        let next = current + line.len();
        if offset < next {
            let character = (offset - current) as u32;
            return Position::new(line_idx as u32, character);
        }
        current = next;
    }
    Position::new(text.lines().count() as u32, 0)
}
