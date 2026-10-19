//! One delimiter/quote grammar for collected legacy fields and short-lived
//! staged field views. A staged scratch never escapes its callback.

use super::*;
use crate::context::host_effect::ExactHostEffectBytes;
use std::borrow::Cow;

pub(in crate::builtins::io::tabular::readmatrix::reader) enum FieldView<'line, 'scratch> {
    Borrowed(&'line str),
    Owned(String),
    Staged(&'scratch str),
}

impl<'line, 'scratch> FieldView<'line, 'scratch> {
    pub(in crate::builtins::io::tabular::readmatrix::reader) fn as_str(&self) -> &str {
        match self {
            Self::Borrowed(value) => value,
            Self::Staged(value) => value,
            Self::Owned(value) => value,
        }
    }

    fn into_cow(self) -> Cow<'line, str> {
        match self {
            Self::Borrowed(value) => Cow::Borrowed(value),
            Self::Owned(value) => Cow::Owned(value),
            Self::Staged(value) => Cow::Owned(value.to_owned()),
        }
    }
}

pub(in crate::builtins::io::tabular::readmatrix::reader) async fn split_fields<'a>(
    line: &'a str,
    delimiter: &Delimiter,
    cooperation: &mut TabularCooperation<'_>,
    path: &impl TabularReadPath,
) -> BuiltinResult<Vec<Cow<'a, str>>> {
    let mut fields = Vec::new();
    for_each_field(line, delimiter, cooperation, path, |_, field| {
        fields.push(field);
        Ok(())
    })
    .await?;
    Ok(fields)
}

pub(in crate::builtins::io::tabular::readmatrix::reader) async fn for_each_field<'a>(
    line: &'a str,
    delimiter: &Delimiter,
    cooperation: &mut TabularCooperation<'_>,
    path: &impl TabularReadPath,
    mut visit: impl FnMut(usize, Cow<'a, str>) -> BuiltinResult<()>,
) -> BuiltinResult<usize> {
    for_each_field_ref(line, delimiter, cooperation, path, None, |index, field| {
        visit(index, field.into_cow())
    })
    .await
}

pub(in crate::builtins::io::tabular::readmatrix::reader) async fn for_each_field_ref<'a>(
    line: &'a str,
    delimiter: &Delimiter,
    cooperation: &mut TabularCooperation<'_>,
    path: &impl TabularReadPath,
    mut staged_scratch: Option<&mut ExactHostEffectBytes>,
    mut visit: impl for<'field> FnMut(usize, FieldView<'a, 'field>) -> BuiltinResult<()>,
) -> BuiltinResult<usize> {
    if let Some(scratch) = staged_scratch.as_mut() {
        scratch.clear();
    }
    match delimiter {
        Delimiter::Char(ch) => {
            visit_char_fields(line, *ch, cooperation, path, staged_scratch, &mut visit).await
        }
        Delimiter::String(pattern) => {
            visit_string_fields(line, pattern, cooperation, path, &mut visit).await
        }
        Delimiter::Whitespace => visit_whitespace_fields(line, cooperation, path, &mut visit).await,
    }
}

async fn append_segment(
    staged: &mut Option<&mut ExactHostEffectBytes>,
    owned: &mut Option<String>,
    text: &str,
    cooperation: &mut TabularCooperation<'_>,
    path: &impl TabularReadPath,
) -> BuiltinResult<()> {
    let mut start = 0;
    while start < text.len() {
        let mut end = start.saturating_add(ROW_WORK_CHUNK).min(text.len());
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        let part = &text[start..end];
        if let Some(scratch) = staged.as_deref_mut() {
            scratch.append(part.as_bytes()).map_err(|source| {
                readmatrix_error_with_source(
                    &READMATRIX_ERROR_TENSOR_BUILD,
                    "readmatrix: quoted field exceeded its admitted workspace",
                    source,
                )
            })?;
        } else {
            let result = owned.get_or_insert_with(String::new);
            result.try_reserve(part.len()).map_err(|source| {
                readmatrix_error_with_source(
                    &READMATRIX_ERROR_TENSOR_BUILD,
                    "readmatrix: unable to reserve quoted field",
                    source,
                )
            })?;
            result.push_str(part);
        }
        checked_row_work(cooperation, path, part.len()).await?;
        start = end;
    }
    Ok(())
}

fn append_quote(
    staged: &mut Option<&mut ExactHostEffectBytes>,
    owned: &mut Option<String>,
) -> BuiltinResult<()> {
    if let Some(scratch) = staged.as_deref_mut() {
        scratch.append(b"\"").map_err(|source| {
            readmatrix_error_with_source(
                &READMATRIX_ERROR_TENSOR_BUILD,
                "readmatrix: quoted field exceeded its admitted workspace",
                source,
            )
        })?;
    } else {
        owned.get_or_insert_with(String::new).push('"');
    }
    Ok(())
}

async fn emit_field<'a>(
    line: &'a str,
    start: usize,
    segment_start: usize,
    end: usize,
    quoted: bool,
    staged: &mut Option<&mut ExactHostEffectBytes>,
    owned: &mut Option<String>,
    cooperation: &mut TabularCooperation<'_>,
    path: &impl TabularReadPath,
    index: usize,
    visit: &mut impl for<'field> FnMut(usize, FieldView<'a, 'field>) -> BuiltinResult<()>,
) -> BuiltinResult<()> {
    if !quoted {
        return visit(index, FieldView::Borrowed(&line[start..end]));
    }
    append_segment(staged, owned, &line[segment_start..end], cooperation, path).await?;
    if let Some(scratch) = staged.as_deref_mut() {
        let text = std::str::from_utf8(scratch.as_bytes()).expect("UTF-8 field segments");
        visit(index, FieldView::Staged(text))?;
        scratch.clear();
    } else {
        visit(index, FieldView::Owned(owned.take().unwrap_or_default()))?;
    }
    Ok(())
}

async fn visit_char_fields<'a>(
    line: &'a str,
    delimiter: char,
    cooperation: &mut TabularCooperation<'_>,
    path: &impl TabularReadPath,
    mut staged: Option<&mut ExactHostEffectBytes>,
    visit: &mut impl for<'field> FnMut(usize, FieldView<'a, 'field>) -> BuiltinResult<()>,
) -> BuiltinResult<usize> {
    let mut count = 0usize;
    let mut start = 0;
    let mut segment_start = 0;
    let mut owned: Option<String> = None;
    let mut quoted = false;
    let mut in_quotes = false;
    let mut chars = line.char_indices().peekable();
    let mut work = 0;
    while let Some((index, ch)) = chars.next() {
        work += ch.len_utf8();
        if ch == '"' {
            quoted = true;
            append_segment(
                &mut staged,
                &mut owned,
                &line[segment_start..index],
                cooperation,
                path,
            )
            .await?;
            if in_quotes && chars.peek().is_some_and(|(_, next)| *next == '"') {
                let (_, escaped) = chars.next().expect("peeked escaped quote");
                work += escaped.len_utf8();
                append_quote(&mut staged, &mut owned)?;
                segment_start = index + ch.len_utf8() + escaped.len_utf8();
            } else {
                in_quotes = !in_quotes;
                segment_start = index + ch.len_utf8();
            }
        } else if ch == delimiter && !in_quotes {
            emit_field(
                line,
                start,
                segment_start,
                index,
                quoted,
                &mut staged,
                &mut owned,
                cooperation,
                path,
                count,
                visit,
            )
            .await?;
            count = count.checked_add(1).ok_or_else(field_overflow)?;
            start = index + ch.len_utf8();
            segment_start = start;
            quoted = false;
        }
        if work >= ROW_WORK_CHUNK {
            checked_row_work(cooperation, path, work).await?;
            work = 0;
        }
    }
    emit_field(
        line,
        start,
        segment_start,
        line.len(),
        quoted,
        &mut staged,
        &mut owned,
        cooperation,
        path,
        count,
        visit,
    )
    .await?;
    count = count.checked_add(1).ok_or_else(field_overflow)?;
    checked_row_work(cooperation, path, work).await?;
    Ok(count)
}

async fn visit_string_fields<'a>(
    line: &'a str,
    pattern: &str,
    cooperation: &mut TabularCooperation<'_>,
    path: &impl TabularReadPath,
    visit: &mut impl for<'field> FnMut(usize, FieldView<'a, 'field>) -> BuiltinResult<()>,
) -> BuiltinResult<usize> {
    let mut count = 0usize;
    let mut start = 0;
    let mut index = 0;
    let mut work = 0;
    while index < line.len() {
        if line[index..].starts_with(pattern) {
            visit(count, FieldView::Borrowed(&line[start..index]))?;
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
    visit(count, FieldView::Borrowed(&line[start..]))?;
    count = count.checked_add(1).ok_or_else(field_overflow)?;
    checked_row_work(cooperation, path, work).await?;
    Ok(count)
}

async fn visit_whitespace_fields<'a>(
    line: &'a str,
    cooperation: &mut TabularCooperation<'_>,
    path: &impl TabularReadPath,
    visit: &mut impl for<'field> FnMut(usize, FieldView<'a, 'field>) -> BuiltinResult<()>,
) -> BuiltinResult<usize> {
    let mut count = 0usize;
    let mut start = None;
    let mut work = 0;
    for (index, ch) in line.char_indices() {
        if ch.is_whitespace() {
            if let Some(begin) = start.take() {
                visit(count, FieldView::Borrowed(&line[begin..index]))?;
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
        visit(count, FieldView::Borrowed(&line[begin..]))?;
        count = count.checked_add(1).ok_or_else(field_overflow)?;
    }
    checked_row_work(cooperation, path, work).await?;
    Ok(count)
}
