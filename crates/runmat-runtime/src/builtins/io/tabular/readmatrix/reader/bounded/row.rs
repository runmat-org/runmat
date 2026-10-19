//! One retained bounded UTF-8 row owner, including chunk validation and
//! cooperative line termination. The staged variant never grows its buffer.

use super::super::scan;
use super::*;
use sha2::{Digest as _, Sha256};
use std::io::BufRead;

mod source;
use source::BoundedByteReader;

pub(super) enum LineBuffer {
    Ordinary(String),
    Staged(ExactHostEffectBytes),
}

impl LineBuffer {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Ordinary(line) => line.len(),
            Self::Staged(line) => line.len(),
        }
    }

    fn clear(&mut self) {
        match self {
            Self::Ordinary(line) => line.clear(),
            Self::Staged(line) => line.clear(),
        }
    }

    pub(super) fn as_str(&self) -> &str {
        match self {
            Self::Ordinary(line) => line,
            Self::Staged(line) => {
                std::str::from_utf8(line.as_bytes()).expect("validated UTF-8 line chunks")
            }
        }
    }

    fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Ordinary(line) => line.as_bytes(),
            Self::Staged(line) => line.as_bytes(),
        }
    }

    pub(super) fn append_validated(&mut self, text: &str) -> BuiltinResult<()> {
        match self {
            Self::Ordinary(line) => {
                line.try_reserve(text.len()).map_err(|source| {
                    readmatrix_error_with_source(
                        &READMATRIX_ERROR_TENSOR_BUILD,
                        "readmatrix: unable to reserve decoded input line",
                        source,
                    )
                })?;
                line.push_str(text);
            }
            Self::Staged(line) => line.append(text.as_bytes()).map_err(|source| {
                readmatrix_error_with_source(
                    &READMATRIX_ERROR_TENSOR_BUILD,
                    "readmatrix: decoded input line exceeded its admitted workspace",
                    source,
                )
            })?,
        }
        Ok(())
    }

    fn pop_ascii(&mut self) {
        match self {
            Self::Ordinary(line) => {
                line.pop();
            }
            Self::Staged(line) => {
                line.truncate(line.len() - 1)
                    .expect("ASCII line suffix has one byte");
            }
        }
    }
}

pub(super) struct RowReader<'a> {
    source: BoundedByteReader<'a>,
    line: LineBuffer,
    line_number: usize,
    first_data_line: bool,
    text_start: usize,
    digest: Sha256,
    diagnostic: Option<&'a dyn DiagnosticReservation>,
}

impl<'a> RowReader<'a> {
    pub(super) fn new(
        source: &'a mut FsBoundedInput,
        staged_line: Option<ExactHostEffectBytes>,
        diagnostic: Option<&'a dyn DiagnosticReservation>,
    ) -> Self {
        Self {
            source: BoundedByteReader::new(source),
            line: staged_line
                .map_or_else(|| LineBuffer::Ordinary(String::new()), LineBuffer::Staged),
            line_number: 0,
            first_data_line: true,
            text_start: 0,
            digest: Sha256::new(),
            diagnostic,
        }
    }

    pub(super) fn text(&self) -> &str {
        &self.line.as_str()[self.text_start..]
    }

    pub(super) fn finish_digest(self) -> [u8; 32] {
        self.digest.finalize().into()
    }

    pub(super) async fn next(
        &mut self,
        cooperation: &mut TabularCooperation<'_>,
        path: &impl TabularReadPath,
        header_lines: usize,
    ) -> BuiltinResult<Option<usize>> {
        loop {
            self.line.clear();
            let mut pending = [0_u8; 3];
            let mut pending_len = 0usize;
            let mut saw_bytes = false;
            loop {
                let available = self.source.fill_buf().map_err(|source| {
                    readmatrix_error_with_source(
                        &READMATRIX_ERROR_IO_READ,
                        format!(
                            "readmatrix: error reading '{}': {source}",
                            path.path().display()
                        ),
                        source,
                    )
                })?;
                if available.is_empty() {
                    if pending_len != 0 {
                        return Err(invalid_utf8(self.diagnostic));
                    }
                    break;
                }
                let count = available
                    .iter()
                    .position(|byte| *byte == b'\n')
                    .map_or(available.len(), |index| index + 1);
                let complete_line = available[count - 1] == b'\n';
                append_utf8_chunk(
                    &mut self.line,
                    &mut pending,
                    &mut pending_len,
                    &available[..count],
                    self.diagnostic,
                )?;
                self.digest.update(&available[..count]);
                self.source.consume(count);
                saw_bytes = true;
                checked_row_work(cooperation, path, count).await?;
                if complete_line {
                    if pending_len != 0 {
                        return Err(invalid_utf8(self.diagnostic));
                    }
                    break;
                }
            }
            if !saw_bytes {
                return Ok(None);
            }
            self.line_number = self
                .line_number
                .checked_add(1)
                .ok_or_else(|| FixedReaderDiagnostic::LineOverflow.render(self.diagnostic))?;
            if self.line.as_bytes().last() == Some(&b'\n') {
                self.line.pop_ascii();
            }
            let mut trailing_cr_work = 0;
            while self.line.as_bytes().last() == Some(&b'\r') {
                self.line.pop_ascii();
                trailing_cr_work += 1;
                if trailing_cr_work >= ROW_WORK_CHUNK {
                    checked_row_work(cooperation, path, trailing_cr_work).await?;
                    trailing_cr_work = 0;
                }
            }
            if trailing_cr_work != 0 {
                checked_row_work(cooperation, path, trailing_cr_work).await?;
            }
            // Validate here rather than repeatedly during trailing CR/BOM
            // trimming. The bytewise suffix scan above cannot split UTF-8.
            let line_text = self.line.as_str();
            let mut text_start = if self.line_number == 1 && line_text.starts_with('\u{FEFF}') {
                '\u{FEFF}'.len_utf8()
            } else {
                0
            };
            let text = &line_text[text_start..];
            if self.line_number <= header_lines
                || !scan::has_nonwhitespace(text, cooperation, path).await?
            {
                continue;
            }
            if self.first_data_line {
                self.first_data_line = false;
                let mut bom_work = 0;
                while line_text[text_start..].starts_with('\u{FEFF}') {
                    text_start += '\u{FEFF}'.len_utf8();
                    bom_work += '\u{FEFF}'.len_utf8();
                    if bom_work >= ROW_WORK_CHUNK {
                        checked_row_work(cooperation, path, bom_work).await?;
                        bom_work = 0;
                    }
                }
                if bom_work != 0 {
                    checked_row_work(cooperation, path, bom_work).await?;
                }
            }
            self.text_start = text_start;
            return Ok(Some(self.line_number));
        }
    }
}

fn invalid_utf8(reservation: Option<&dyn DiagnosticReservation>) -> RuntimeError {
    FixedReaderDiagnostic::InvalidUtf8.render(reservation)
}

fn append_utf8_chunk(
    line: &mut LineBuffer,
    pending: &mut [u8; 3],
    pending_len: &mut usize,
    chunk: &[u8],
    diagnostic: Option<&dyn DiagnosticReservation>,
) -> BuiltinResult<()> {
    let mut bytes = [0_u8; 4099];
    let total = pending_len
        .checked_add(chunk.len())
        .ok_or_else(|| matrix_overflow_with(diagnostic))?;
    if total > bytes.len() {
        return Err(FixedReaderDiagnostic::ChunkOverflow.render(diagnostic));
    }
    bytes[..*pending_len].copy_from_slice(&pending[..*pending_len]);
    bytes[*pending_len..total].copy_from_slice(chunk);
    let candidate = &bytes[..total];
    let (text, tail) = match std::str::from_utf8(candidate) {
        Ok(text) => (text, &[][..]),
        Err(source) if source.error_len().is_none() => {
            let valid = source.valid_up_to();
            let text = std::str::from_utf8(&candidate[..valid]).expect("validated UTF-8 prefix");
            (text, &candidate[valid..])
        }
        Err(source) => {
            return Err(FixedReaderDiagnostic::invalid_utf8_with_source(
                source, diagnostic,
            ));
        }
    };
    line.append_validated(text)?;
    if tail.len() > pending.len() {
        return Err(invalid_utf8(diagnostic));
    }
    pending[..tail.len()].copy_from_slice(tail);
    *pending_len = tail.len();
    Ok(())
}
