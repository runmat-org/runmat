//! Borrowed text rows and bounded field scanning for one readmatrix call.

use super::{
    readmatrix_error_with, readmatrix_error_with_source, Delimiter, TabularCooperation,
    TabularReadPath, READMATRIX_ERROR_IO_READ, READMATRIX_ERROR_TENSOR_BUILD, ROW_WORK_CHUNK,
};
use crate::{BuiltinResult, RuntimeError};

mod fields;
pub(super) use fields::{for_each_field_ref, split_fields};

fn field_overflow() -> RuntimeError {
    readmatrix_error_with(
        &READMATRIX_ERROR_TENSOR_BUILD,
        "readmatrix: field count exceeds platform limits",
    )
}

pub(super) async fn checked_row_work(
    cooperation: &mut TabularCooperation<'_>,
    path: &impl TabularReadPath,
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
    path: &impl TabularReadPath,
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
    path: &impl TabularReadPath,
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
    path: &impl TabularReadPath,
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
