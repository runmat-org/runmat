//! Two-pass UTF-8 numeric text scanning on one retained seekable source.
//!
//! This owner does not itself certify a finite complete-call peak. The
//! staged caller must promote its original reservation for the opened extent
//! before invoking it and charge its live line, field, and output allocations.

use super::*;
use runmat_filesystem::FsBoundedInput;
use sha2::{Digest as _, Sha256};
use std::io::{BufRead, BufReader};

mod fill;
pub(super) use fill::fill_utf8;

pub(super) struct BoundedScanGeometry {
    pub(super) rows: usize,
    pub(super) columns: usize,
    pub(super) delimiter: Delimiter,
    source_digest: [u8; 32],
}

struct RowReader<'a> {
    source: BufReader<&'a mut FsBoundedInput>,
    line: String,
    line_number: usize,
    first_data_line: bool,
    text_start: usize,
    digest: Sha256,
}

impl<'a> RowReader<'a> {
    fn new(source: &'a mut FsBoundedInput) -> Self {
        Self {
            source: BufReader::with_capacity(4096, source),
            line: String::new(),
            line_number: 0,
            first_data_line: true,
            text_start: 0,
            digest: Sha256::new(),
        }
    }

    fn text(&self) -> &str {
        &self.line[self.text_start..]
    }

    async fn next(
        &mut self,
        cooperation: &mut TabularCooperation<'_>,
        path: &TabularFsPath,
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
                        return Err(readmatrix_error_with(
                            &READMATRIX_ERROR_IO_READ,
                            "readmatrix: file contains bytes that are invalid for Encoding 'utf-8'",
                        ));
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
                )?;
                self.digest.update(&available[..count]);
                self.source.consume(count);
                saw_bytes = true;
                checked_row_work(cooperation, path, count).await?;
                if complete_line {
                    if pending_len != 0 {
                        return Err(readmatrix_error_with(
                            &READMATRIX_ERROR_IO_READ,
                            "readmatrix: file contains bytes that are invalid for Encoding 'utf-8'",
                        ));
                    }
                    break;
                }
            }
            if !saw_bytes {
                return Ok(None);
            }
            self.line_number = self.line_number.checked_add(1).ok_or_else(|| {
                readmatrix_error_with(
                    &READMATRIX_ERROR_IO_READ,
                    "readmatrix: input line number exceeds platform limits",
                )
            })?;
            if self.line.ends_with('\n') {
                self.line.pop();
            }
            let mut trailing_cr_work = 0;
            while self.line.ends_with('\r') {
                self.line.pop();
                trailing_cr_work += 1;
                if trailing_cr_work >= ROW_WORK_CHUNK {
                    checked_row_work(cooperation, path, trailing_cr_work).await?;
                    trailing_cr_work = 0;
                }
            }
            if trailing_cr_work != 0 {
                checked_row_work(cooperation, path, trailing_cr_work).await?;
            }
            let mut text_start = if self.line_number == 1 && self.line.starts_with('\u{FEFF}') {
                '\u{FEFF}'.len_utf8()
            } else {
                0
            };
            let text = &self.line[text_start..];
            if self.line_number <= header_lines
                || !scan::has_nonwhitespace(text, cooperation, path).await?
            {
                continue;
            }
            if self.first_data_line {
                self.first_data_line = false;
                let mut bom_work = 0;
                while self.line[text_start..].starts_with('\u{FEFF}') {
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

fn append_utf8_chunk(
    line: &mut String,
    pending: &mut [u8; 3],
    pending_len: &mut usize,
    chunk: &[u8],
) -> BuiltinResult<()> {
    let mut bytes = [0_u8; 4099];
    let total = pending_len
        .checked_add(chunk.len())
        .ok_or_else(matrix_overflow)?;
    if total > bytes.len() {
        return Err(readmatrix_error_with(
            &READMATRIX_ERROR_IO_READ,
            "readmatrix: decoded input chunk exceeds the bounded reader capacity",
        ));
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
            return Err(readmatrix_error_with_source(
                &READMATRIX_ERROR_IO_READ,
                "readmatrix: file contains bytes that are invalid for Encoding 'utf-8'",
                source,
            ));
        }
    };
    line.try_reserve(text.len()).map_err(|source| {
        readmatrix_error_with_source(
            &READMATRIX_ERROR_TENSOR_BUILD,
            "readmatrix: unable to reserve decoded input line",
            source,
        )
    })?;
    line.push_str(text);
    if tail.len() > pending.len() {
        return Err(readmatrix_error_with(
            &READMATRIX_ERROR_IO_READ,
            "readmatrix: file contains bytes that are invalid for Encoding 'utf-8'",
        ));
    }
    pending[..tail.len()].copy_from_slice(tail);
    *pending_len = tail.len();
    Ok(())
}

fn empty_tensor() -> BuiltinResult<Tensor> {
    Tensor::new(Vec::new(), vec![0, 0]).map_err(|source| {
        readmatrix_error_with(
            &READMATRIX_ERROR_TENSOR_BUILD,
            format!("readmatrix: {source}"),
        )
    })
}

pub(super) async fn scan_utf8(
    source: &mut FsBoundedInput,
    cooperation: &mut TabularCooperation<'_>,
    path: &TabularFsPath,
    options: &ReadMatrixOptions,
) -> BuiltinResult<BoundedScanGeometry> {
    let mut reader = RowReader::new(source);
    let mut sample = Vec::new();
    while sample.len() < 32 {
        let Some(number) = reader
            .next(cooperation, path, options.num_header_lines)
            .await?
        else {
            break;
        };
        sample.push((number, reader.text().to_owned()));
    }
    let delimiter = match options.delimiter.clone() {
        Some(delimiter) => delimiter,
        None => {
            let borrowed: Vec<_> = sample
                .iter()
                .map(|(number, text)| (*number, text.as_str()))
                .collect();
            detect_delimiter(&borrowed, cooperation, path)
                .await?
                .unwrap_or(Delimiter::Whitespace)
        }
    };
    let integer_dtype = integer_output_dtype(&options.output_template);
    let mut rows = 0usize;
    let mut columns = 0usize;
    // The legacy decoder checks the whole source for invalid encoding before
    // numeric tokens. Keep the first semantic error, but finish UTF-8
    // validation on this same opened handle before reporting it.
    let mut semantic_error = None;
    for (number, text) in &sample {
        let result = validate_row(
            text,
            *number,
            &delimiter,
            integer_dtype,
            options,
            cooperation,
            path,
        )
        .await;
        match result {
            Ok(count) if count != 0 => {
                rows = rows.checked_add(1).ok_or_else(matrix_overflow)?;
                columns = columns.max(count);
            }
            Err(error) if semantic_error.is_none() => semantic_error = Some(error),
            _ => {}
        }
    }
    while let Some(number) = reader
        .next(cooperation, path, options.num_header_lines)
        .await?
    {
        if semantic_error.is_none() {
            match validate_row(
                reader.text(),
                number,
                &delimiter,
                integer_dtype,
                options,
                cooperation,
                path,
            )
            .await
            {
                Ok(count) if count != 0 => {
                    rows = rows.checked_add(1).ok_or_else(matrix_overflow)?;
                    columns = columns.max(count);
                }
                Err(error) => semantic_error = Some(error),
                _ => {}
            }
        }
    }
    if let Some(error) = semantic_error {
        return Err(error);
    }
    let source_digest = reader.digest.finalize().into();
    Ok(BoundedScanGeometry {
        rows,
        columns,
        delimiter,
        source_digest,
    })
}

async fn validate_row(
    text: &str,
    line_number: usize,
    delimiter: &Delimiter,
    integer_dtype: Option<NumericDType>,
    options: &ReadMatrixOptions,
    cooperation: &mut TabularCooperation<'_>,
    path: &TabularFsPath,
) -> BuiltinResult<usize> {
    scan::for_each_field(text, delimiter, cooperation, path, |index, field| {
        if let Some(dtype) = integer_dtype {
            parse_integer_numeric_token(&field, options, dtype, line_number, index + 1)?;
        } else {
            validate_numeric_token(&field, options, line_number, index + 1)?;
        }
        Ok(())
    })
    .await
}

fn matrix_overflow() -> RuntimeError {
    readmatrix_error_with(
        &READMATRIX_ERROR_TENSOR_BUILD,
        "readmatrix: matrix dimensions exceed addressable storage",
    )
}
