//! Borrowed text rows and bounded field scanning for one readmatrix call.

use std::borrow::Cow;

use super::{
    readmatrix_error_with, readmatrix_error_with_source, Delimiter, TabularCooperation,
    TabularFsPath, READMATRIX_ERROR_IO_READ, READMATRIX_ERROR_TENSOR_BUILD, ROW_WORK_CHUNK,
};
use crate::{BuiltinResult, RuntimeError};

fn field_overflow() -> RuntimeError {
    readmatrix_error_with(
        &READMATRIX_ERROR_TENSOR_BUILD,
        "readmatrix: field count exceeds platform limits",
    )
}

pub(super) async fn checked_row_work(
    cooperation: &mut TabularCooperation<'_>,
    path: &TabularFsPath,
    work: usize,
) -> BuiltinResult<()> {
    cooperation.completed(work).await?;
    path.check().map_err(|source| {
        readmatrix_error_with_source(
            &READMATRIX_ERROR_IO_READ,
            "readmatrix: captured filesystem authority was revoked",
            source,
        )
    })
}

pub(super) async fn collect_data_lines<'a>(
    text: &'a str,
    header_lines: usize,
    cooperation: &mut TabularCooperation<'_>,
    path: &TabularFsPath,
) -> BuiltinResult<Vec<(usize, &'a str)>> {
    let mut lines = Vec::new();
    let mut line_start = 0;
    let mut line_number = 1;
    let mut work = 0;
    for (index, byte) in text.bytes().enumerate() {
        work += 1;
        if byte == b'\n' {
            if line_number > header_lines {
                let line = text[line_start..index].trim_end_matches('\r');
                if has_nonwhitespace(line, cooperation, path).await? {
                    lines.push((line_number, line));
                }
            }
            line_number += 1;
            line_start = index + 1;
        }
        if work >= ROW_WORK_CHUNK {
            checked_row_work(cooperation, path, work).await?;
            work = 0;
        }
    }
    if line_start < text.len() && line_number > header_lines {
        let line = text[line_start..].trim_end_matches('\r');
        if has_nonwhitespace(line, cooperation, path).await? {
            lines.push((line_number, line));
        }
    }
    checked_row_work(cooperation, path, work).await?;
    Ok(lines)
}

pub(super) async fn has_nonwhitespace(
    line: &str,
    cooperation: &mut TabularCooperation<'_>,
    path: &TabularFsPath,
) -> BuiltinResult<bool> {
    let mut work = 0;
    for ch in line.chars() {
        if !ch.is_whitespace() {
            return Ok(true);
        }
        work += ch.len_utf8();
        if work >= ROW_WORK_CHUNK {
            checked_row_work(cooperation, path, work).await?;
            work = 0;
        }
    }
    if work != 0 {
        checked_row_work(cooperation, path, work).await?;
    }
    Ok(false)
}

pub(super) async fn detect_delimiter(
    lines: &[(usize, &str)],
    cooperation: &mut TabularCooperation<'_>,
    path: &TabularFsPath,
) -> BuiltinResult<Option<Delimiter>> {
    if lines.is_empty() {
        return Ok(None);
    }
    let sample = &lines[..lines.len().min(32)];
    let mut best: Option<(f64, Delimiter)> = None;
    for candidate in [',', '\t', ';', '|'] {
        let mut counts = [0usize; 32];
        let mut count_len = 0usize;
        for (_, line) in sample {
            let mut fields = 1;
            let mut work = 0;
            for ch in line.chars() {
                if ch == candidate {
                    fields += 1;
                }
                work += ch.len_utf8();
                if work >= ROW_WORK_CHUNK {
                    checked_row_work(cooperation, path, work).await?;
                    work = 0;
                }
            }
            if work != 0 {
                checked_row_work(cooperation, path, work).await?;
            }
            if fields >= 2 {
                counts[count_len] = fields;
                count_len += 1;
            }
        }
        if count_len == 0 {
            continue;
        }
        let average = counts[..count_len]
            .iter()
            .copied()
            .map(|count| count as u128)
            .sum::<u128>() as f64
            / count_len as f64;
        if average >= 2.0
            && best
                .as_ref()
                .map(|(best_avg, _)| average > *best_avg)
                .unwrap_or(true)
        {
            best = Some((average, Delimiter::Char(candidate)));
        }
    }
    if let Some((_, delimiter)) = best {
        return Ok(Some(delimiter));
    }
    for (_, line) in sample {
        let mut in_word = false;
        let mut words = 0;
        let mut work = 0;
        for ch in line.chars() {
            if ch.is_whitespace() {
                in_word = false;
            } else if !in_word {
                words += 1;
                in_word = true;
            }
            if words > 1 {
                return Ok(Some(Delimiter::Whitespace));
            }
            work += ch.len_utf8();
            if work >= ROW_WORK_CHUNK {
                checked_row_work(cooperation, path, work).await?;
                work = 0;
            }
        }
        if work != 0 {
            checked_row_work(cooperation, path, work).await?;
        }
    }
    Ok(None)
}

pub(super) async fn split_fields<'a>(
    line: &'a str,
    delimiter: &Delimiter,
    cooperation: &mut TabularCooperation<'_>,
    path: &TabularFsPath,
) -> BuiltinResult<Vec<Cow<'a, str>>> {
    let mut fields = Vec::new();
    for_each_field(line, delimiter, cooperation, path, |_, field| {
        fields.push(field);
        Ok(())
    })
    .await?;
    Ok(fields)
}

/// A single tokenizer authority for both the legacy collected route and the
/// bounded scan/fill route. The latter need not retain a row-sized field Vec.
pub(super) async fn for_each_field<'a>(
    line: &'a str,
    delimiter: &Delimiter,
    cooperation: &mut TabularCooperation<'_>,
    path: &TabularFsPath,
    mut visit: impl FnMut(usize, Cow<'a, str>) -> BuiltinResult<()>,
) -> BuiltinResult<usize> {
    match delimiter {
        Delimiter::Char(ch) => visit_char_fields(line, *ch, cooperation, path, &mut visit).await,
        Delimiter::String(pattern) => {
            visit_string_fields(line, pattern, cooperation, path, &mut visit).await
        }
        Delimiter::Whitespace => visit_whitespace_fields(line, cooperation, path, &mut visit).await,
    }
}

async fn visit_char_fields<'a>(
    line: &'a str,
    delimiter: char,
    cooperation: &mut TabularCooperation<'_>,
    path: &TabularFsPath,
    visit: &mut impl FnMut(usize, Cow<'a, str>) -> BuiltinResult<()>,
) -> BuiltinResult<usize> {
    let mut count = 0usize;
    let mut start = 0;
    let mut segment_start = 0;
    let mut owned: Option<String> = None;
    let mut in_quotes = false;
    let mut chars = line.char_indices().peekable();
    let mut work = 0;
    while let Some((index, ch)) = chars.next() {
        work += ch.len_utf8();
        if ch == '"' {
            let value = owned.get_or_insert_with(String::new);
            value.push_str(&line[segment_start..index]);
            if in_quotes && chars.peek().is_some_and(|(_, next)| *next == '"') {
                let (_, escaped) = chars.next().expect("peeked escaped quote");
                work += escaped.len_utf8();
                value.push('"');
                segment_start = index + ch.len_utf8() + escaped.len_utf8();
            } else {
                in_quotes = !in_quotes;
                segment_start = index + ch.len_utf8();
            }
        } else if ch == delimiter && !in_quotes {
            let field = match owned.take() {
                Some(mut value) => {
                    value.push_str(&line[segment_start..index]);
                    Cow::Owned(value)
                }
                None => Cow::Borrowed(&line[start..index]),
            };
            visit(count, field)?;
            count = count.checked_add(1).ok_or_else(field_overflow)?;
            start = index + ch.len_utf8();
            segment_start = start;
        }
        if work >= ROW_WORK_CHUNK {
            checked_row_work(cooperation, path, work).await?;
            work = 0;
        }
    }
    let field = match owned {
        Some(mut value) => {
            value.push_str(&line[segment_start..]);
            Cow::Owned(value)
        }
        None => Cow::Borrowed(&line[start..]),
    };
    visit(count, field)?;
    count = count.checked_add(1).ok_or_else(field_overflow)?;
    checked_row_work(cooperation, path, work).await?;
    Ok(count)
}

async fn visit_string_fields<'a>(
    line: &'a str,
    pattern: &str,
    cooperation: &mut TabularCooperation<'_>,
    path: &TabularFsPath,
    visit: &mut impl FnMut(usize, Cow<'a, str>) -> BuiltinResult<()>,
) -> BuiltinResult<usize> {
    let mut count = 0usize;
    let mut start = 0;
    let mut index = 0;
    let mut work = 0;
    while index < line.len() {
        if line[index..].starts_with(pattern) {
            visit(count, Cow::Borrowed(&line[start..index]))?;
            count = count.checked_add(1).ok_or_else(field_overflow)?;
            index += pattern.len();
            start = index;
            work += pattern.len();
        } else {
            let width = line[index..]
                .chars()
                .next()
                .expect("remaining text")
                .len_utf8();
            index += width;
            work += width;
        }
        if work >= ROW_WORK_CHUNK {
            checked_row_work(cooperation, path, work).await?;
            work = 0;
        }
    }
    visit(count, Cow::Borrowed(&line[start..]))?;
    count = count.checked_add(1).ok_or_else(field_overflow)?;
    checked_row_work(cooperation, path, work).await?;
    Ok(count)
}

async fn visit_whitespace_fields<'a>(
    line: &'a str,
    cooperation: &mut TabularCooperation<'_>,
    path: &TabularFsPath,
    visit: &mut impl FnMut(usize, Cow<'a, str>) -> BuiltinResult<()>,
) -> BuiltinResult<usize> {
    let mut count = 0usize;
    let mut start = None;
    let mut work = 0;
    for (index, ch) in line.char_indices() {
        if ch.is_whitespace() {
            if let Some(begin) = start.take() {
                visit(count, Cow::Borrowed(&line[begin..index]))?;
                count = count.checked_add(1).ok_or_else(field_overflow)?;
            }
        } else if start.is_none() {
            start = Some(index);
        }
        work += ch.len_utf8();
        if work >= ROW_WORK_CHUNK {
            checked_row_work(cooperation, path, work).await?;
            work = 0;
        }
    }
    if let Some(begin) = start {
        visit(count, Cow::Borrowed(&line[begin..]))?;
        count = count.checked_add(1).ok_or_else(field_overflow)?;
    }
    checked_row_work(cooperation, path, work).await?;
    Ok(count)
}
