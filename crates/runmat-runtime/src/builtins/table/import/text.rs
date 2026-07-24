use super::*;

pub(in crate::builtins::table) fn detect_delimiter(lines: &[String]) -> Option<Delimiter> {
    let candidates = [',', '\t', ';', '|'];
    let mut best: Option<(f64, Delimiter)> = None;
    for candidate in candidates {
        let counts = lines
            .iter()
            .take(32)
            .filter(|line| line.contains(candidate))
            .map(|line| split_with_char_delim(line, candidate).len())
            .filter(|count| *count >= 2)
            .collect::<Vec<_>>();
        if counts.is_empty() {
            continue;
        }
        let avg = counts.iter().copied().sum::<usize>() as f64 / counts.len() as f64;
        if avg >= 2.0
            && best
                .as_ref()
                .map(|(best_avg, _)| avg > *best_avg)
                .unwrap_or(true)
        {
            best = Some((avg, Delimiter::Char(candidate)));
        }
    }
    best.map(|(_, delimiter)| delimiter).or_else(|| {
        lines
            .iter()
            .take(32)
            .any(|line| line.split_whitespace().count() > 1)
            .then_some(Delimiter::Whitespace)
    })
}

pub(in crate::builtins::table) fn split_with_char_delim(
    line: &str,
    delimiter: char,
) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '"' {
            if in_quotes && chars.peek() == Some(&'"') {
                current.push('"');
                chars.next();
            } else {
                in_quotes = !in_quotes;
            }
            continue;
        }
        if ch == delimiter && !in_quotes {
            out.push(current.clone());
            current.clear();
        } else {
            current.push(ch);
        }
    }
    out.push(current);
    out
}

pub(in crate::builtins::table) fn parse_text_records(
    text: &str,
    delimiter: &Delimiter,
    empty_line_rule: EmptyLineRule,
) -> Vec<Vec<ImportCell>> {
    match delimiter {
        Delimiter::Whitespace => parse_whitespace_records(text, empty_line_rule),
        Delimiter::Char(ch) => parse_delimited_records(text, &ch.to_string(), empty_line_rule),
        Delimiter::String(pattern) => parse_delimited_records(text, pattern, empty_line_rule),
    }
}

pub(in crate::builtins::table) fn parse_delimited_records(
    text: &str,
    delimiter: &str,
    empty_line_rule: EmptyLineRule,
) -> Vec<Vec<ImportCell>> {
    let mut records = Vec::new();
    let mut row = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut idx = 0usize;
    while idx < text.len() {
        let ch = text[idx..].chars().next().expect("valid char boundary");
        if ch == '"' {
            if in_quotes && text[idx + ch.len_utf8()..].starts_with('"') {
                current.push('"');
                idx += ch.len_utf8() + 1;
                continue;
            }
            in_quotes = !in_quotes;
            idx += ch.len_utf8();
            continue;
        }
        if !in_quotes && !delimiter.is_empty() && text[idx..].starts_with(delimiter) {
            row.push(ImportCell::from_text(std::mem::take(&mut current)));
            idx += delimiter.len();
            continue;
        }
        if !in_quotes && (ch == '\n' || ch == '\r') {
            row.push(ImportCell::from_text(std::mem::take(&mut current)));
            push_import_record(&mut records, std::mem::take(&mut row), empty_line_rule);
            idx += ch.len_utf8();
            if ch == '\r' && text[idx..].starts_with('\n') {
                idx += 1;
            }
            continue;
        }
        current.push(ch);
        idx += ch.len_utf8();
    }
    if !current.is_empty() || !row.is_empty() || text.ends_with(delimiter) {
        row.push(ImportCell::from_text(current));
        push_import_record(&mut records, row, empty_line_rule);
    }
    records
}

pub(in crate::builtins::table) fn parse_whitespace_records(
    text: &str,
    empty_line_rule: EmptyLineRule,
) -> Vec<Vec<ImportCell>> {
    let mut records = Vec::new();
    let mut row = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut field_open = false;
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '"' {
            if in_quotes && chars.peek() == Some(&'"') {
                current.push('"');
                chars.next();
            } else {
                in_quotes = !in_quotes;
            }
            field_open = true;
            continue;
        }
        if !in_quotes && (ch == '\n' || ch == '\r') {
            if field_open || !current.is_empty() {
                row.push(ImportCell::from_text(std::mem::take(&mut current)));
            }
            field_open = false;
            push_import_record(&mut records, std::mem::take(&mut row), empty_line_rule);
            if ch == '\r' && chars.peek() == Some(&'\n') {
                chars.next();
            }
            continue;
        }
        if !in_quotes && ch.is_whitespace() {
            if field_open || !current.is_empty() {
                row.push(ImportCell::from_text(std::mem::take(&mut current)));
                field_open = false;
            }
            continue;
        }
        current.push(ch);
        field_open = true;
    }
    if field_open || !current.is_empty() {
        row.push(ImportCell::from_text(current));
    }
    if !row.is_empty() {
        push_import_record(&mut records, row, empty_line_rule);
    }
    records
}

pub(in crate::builtins::table) fn push_import_record(
    records: &mut Vec<Vec<ImportCell>>,
    row: Vec<ImportCell>,
    empty_line_rule: EmptyLineRule,
) {
    if matches!(empty_line_rule, EmptyLineRule::Skip)
        && row.iter().all(|cell| matches!(cell, ImportCell::Empty))
    {
        return;
    }
    records.push(row);
}

pub(in crate::builtins::table) fn apply_import_range(
    rows: Vec<Vec<ImportCell>>,
    range: RangeSpec,
) -> Vec<Vec<ImportCell>> {
    if rows.is_empty() {
        return rows;
    }
    let end_row = range
        .end_row
        .unwrap_or_else(|| rows.len().saturating_sub(1));
    let max_cols = rows.iter().map(Vec::len).max().unwrap_or(0);
    let end_col = range.end_col.unwrap_or_else(|| max_cols.saturating_sub(1));
    rows.into_iter()
        .enumerate()
        .filter_map(|(idx, row)| {
            if idx < range.start_row || idx > end_row {
                return None;
            }
            let selected = (range.start_col..=end_col)
                .map(|col| row.get(col).cloned().unwrap_or(ImportCell::Empty))
                .collect::<Vec<_>>();
            Some(selected)
        })
        .collect()
}
