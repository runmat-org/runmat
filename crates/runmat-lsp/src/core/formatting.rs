use lsp_types::{Position, Range, TextEdit};

/// Simple formatter: trims trailing whitespace, normalizes newlines, and
/// collapses more than two consecutive blank lines to at most one.
pub fn formatting_edits(text: &str) -> Vec<TextEdit> {
    let formatted = format_document(text);
    if formatted == text {
        return Vec::new();
    }
    vec![TextEdit {
        range: full_range(text),
        new_text: formatted,
    }]
}

fn format_document(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut blank_run = 0usize;
    for line in text.lines() {
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            blank_run += 1;
            if blank_run > 1 {
                continue;
            }
            out.push('\n');
            continue;
        }
        blank_run = 0;
        out.push_str(trimmed);
        out.push('\n');
    }
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

fn full_range(text: &str) -> Range {
    let start = Position::new(0, 0);
    let line_count = text.lines().count() as u32;
    let last_line_len = text
        .lines()
        .last()
        .map(|l| l.len() as u32)
        .unwrap_or(0);
    let end = Position::new(line_count, last_line_len);
    Range { start, end }
}

