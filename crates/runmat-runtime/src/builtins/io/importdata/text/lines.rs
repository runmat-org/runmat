//! Cooperative source-line discovery with Rust `str::lines` CRLF semantics.

use super::*;

pub(super) async fn collect_nonempty_lines<'a>(
    text: &'a str,
    cooperation: &mut cooperation::ImportCooperation,
) -> BuiltinResult<Vec<(usize, &'a str)>> {
    let mut nonempty = Vec::new();
    let mut start = 0usize;
    let mut line_index = 0usize;
    for (offset, byte) in text.as_bytes().iter().enumerate() {
        if *byte == b'\n' {
            let line = &text[start..offset];
            let line = line.strip_suffix('\r').unwrap_or(line);
            if line_has_nonwhitespace(line, cooperation).await? {
                nonempty.push((line_index, line));
            }
            line_index += 1;
            start = offset + 1;
        }
        if offset % 4096 == 4095 {
            cooperation.completed(4096).await?;
        }
    }
    cooperation.completed(text.len() % 4096).await?;
    if start < text.len() {
        let line = &text[start..];
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line_has_nonwhitespace(line, cooperation).await? {
            nonempty.push((line_index, line));
        }
    }
    Ok(nonempty)
}

async fn line_has_nonwhitespace(
    line: &str,
    cooperation: &mut cooperation::ImportCooperation,
) -> BuiltinResult<bool> {
    let mut scanned = 0usize;
    for ch in line.chars() {
        if !ch.is_whitespace() {
            cooperation.completed(scanned).await?;
            return Ok(true);
        }
        scanned += ch.len_utf8();
        if scanned >= 4096 {
            cooperation.completed(scanned).await?;
            scanned = 0;
        }
    }
    cooperation.completed(scanned).await?;
    Ok(false)
}
