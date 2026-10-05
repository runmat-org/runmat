//! Two-pass UTF-8 numeric text scanning on one retained seekable source.
//!
//! This owner does not itself certify a finite complete-call peak. The
//! staged caller must promote its original reservation for the opened extent
//! before invoking it and charge its live line, field, and output allocations.

use super::*;
use runmat_filesystem::FsBoundedInput;
use sha2::{Digest as _, Sha256};
use std::io::{BufRead, BufReader, Seek, SeekFrom};

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
    digest: Sha256,
}

impl<'a> RowReader<'a> {
    fn new(source: &'a mut FsBoundedInput) -> Self {
        Self {
            source: BufReader::with_capacity(4096, source),
            line: String::new(),
            line_number: 0,
            first_data_line: true,
            digest: Sha256::new(),
        }
    }

    async fn next<'b>(
        &'b mut self,
        cooperation: &mut TabularCooperation<'_>,
        path: &TabularFsPath,
        header_lines: usize,
    ) -> BuiltinResult<Option<(usize, &'b str)>> {
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
            let text = self.line.as_str();
            let text = if self.line_number == 1 {
                strip_utf8_bom(text)
            } else {
                text
            };
            if self.line_number <= header_lines
                || !scan::has_nonwhitespace(text, cooperation, path).await?
            {
                continue;
            }
            let text = if self.first_data_line {
                self.first_data_line = false;
                let mut stripped = text;
                let mut bom_work = 0;
                while let Some(without_bom) = stripped.strip_prefix('\u{FEFF}') {
                    stripped = without_bom;
                    bom_work += '\u{FEFF}'.len_utf8();
                    if bom_work >= ROW_WORK_CHUNK {
                        checked_row_work(cooperation, path, bom_work).await?;
                        bom_work = 0;
                    }
                }
                if bom_work != 0 {
                    checked_row_work(cooperation, path, bom_work).await?;
                }
                stripped
            } else {
                text
            };
            return Ok(Some((self.line_number, text)));
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
        let Some((number, text)) = reader
            .next(cooperation, path, options.num_header_lines)
            .await?
        else {
            break;
        };
        sample.push((number, text.to_owned()));
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
    while let Some((number, text)) = reader
        .next(cooperation, path, options.num_header_lines)
        .await?
    {
        if semantic_error.is_none() {
            match validate_row(
                text,
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
    let fields = split_fields(text, delimiter, cooperation, path).await?;
    for (index, field) in fields.iter().enumerate() {
        if let Some(dtype) = integer_dtype {
            parse_integer_numeric_token(field, options, dtype, line_number, index + 1)?;
        } else {
            validate_numeric_token(field, options, line_number, index + 1)?;
        }
        if (index + 1) % ROW_WORK_CHUNK == 0 {
            checked_row_work(cooperation, path, ROW_WORK_CHUNK).await?;
        }
    }
    if !fields.len().is_multiple_of(ROW_WORK_CHUNK) {
        checked_row_work(cooperation, path, fields.len() % ROW_WORK_CHUNK).await?;
    }
    Ok(fields.len())
}

fn matrix_overflow() -> RuntimeError {
    readmatrix_error_with(
        &READMATRIX_ERROR_TENSOR_BUILD,
        "readmatrix: matrix dimensions exceed addressable storage",
    )
}

fn selected_geometry(
    geometry: &BoundedScanGeometry,
    range: Option<&RangeSpec>,
) -> BuiltinResult<(usize, usize, usize, usize)> {
    if geometry.rows == 0 || geometry.columns == 0 {
        return Ok((0, 0, 0, 0));
    }
    let Some(range) = range else {
        return Ok((0, 0, geometry.rows, geometry.columns));
    };
    if range.start_row >= geometry.rows || range.start_col >= geometry.columns {
        return Ok((0, 0, 0, 0));
    }
    let last_row = range
        .end_row
        .unwrap_or(geometry.rows - 1)
        .min(geometry.rows - 1);
    let last_col = range
        .end_col
        .unwrap_or(geometry.columns - 1)
        .min(geometry.columns - 1);
    if last_row < range.start_row || last_col < range.start_col {
        return Ok((0, 0, 0, 0));
    }
    Ok((
        range.start_row,
        range.start_col,
        last_row - range.start_row + 1,
        last_col - range.start_col + 1,
    ))
}

pub(super) async fn fill_utf8(
    source: &mut FsBoundedInput,
    geometry: &BoundedScanGeometry,
    cooperation: &mut TabularCooperation<'_>,
    path: &TabularFsPath,
    options: &ReadMatrixOptions,
) -> BuiltinResult<Tensor> {
    let (first_row, first_col, rows, cols) = selected_geometry(geometry, options.range.as_ref())?;
    if rows == 0 || cols == 0 {
        return empty_tensor();
    }
    let count = rows.checked_mul(cols).ok_or_else(matrix_overflow)?;
    source.seek(SeekFrom::Start(0)).map_err(|source| {
        readmatrix_error_with_source(
            &READMATRIX_ERROR_IO_READ,
            "readmatrix: unable to replay the retained input source",
            source,
        )
    })?;
    path.check().map_err(|source| {
        readmatrix_error_with_source(
            &READMATRIX_ERROR_IO_READ,
            "readmatrix: captured filesystem authority was revoked",
            source,
        )
    })?;
    let integer_dtype = integer_output_dtype(&options.output_template);
    let mut integer = integer_dtype.map(|dtype| NumericStorage::zeros(dtype, count));
    let mut floating = integer_dtype
        .is_none()
        .then(|| vec![options.empty_value_f64(); count]);
    let mut reader = RowReader::new(source);
    let mut data_row = 0usize;
    while let Some((line_number, text)) = reader
        .next(cooperation, path, options.num_header_lines)
        .await?
    {
        let fields = split_fields(text, &geometry.delimiter, cooperation, path).await?;
        if fields.is_empty() {
            continue;
        }
        if fields.len() > geometry.columns || data_row >= geometry.rows {
            return Err(changed_source());
        }
        if data_row >= first_row && data_row - first_row < rows {
            let output_row = data_row - first_row;
            for output_col in 0..cols {
                let token = fields
                    .get(first_col + output_col)
                    .map(|field| field.as_ref())
                    .unwrap_or("");
                let index = output_col * rows + output_row;
                if let (Some(dtype), Some(storage)) = (integer_dtype, integer.as_mut()) {
                    let value = parse_integer_numeric_token(
                        token,
                        options,
                        dtype,
                        line_number,
                        first_col + output_col + 1,
                    )?;
                    storage.set_value(index, value).map_err(|source| {
                        readmatrix_error_with(
                            &READMATRIX_ERROR_TENSOR_BUILD,
                            format!("readmatrix: {source}"),
                        )
                    })?;
                } else if let Some(storage) = floating.as_mut() {
                    storage[index] = parse_numeric_token(
                        token,
                        options,
                        line_number,
                        first_col + output_col + 1,
                    )?;
                }
                if (output_col + 1) % ROW_WORK_CHUNK == 0 {
                    checked_row_work(cooperation, path, ROW_WORK_CHUNK).await?;
                }
            }
            if !cols.is_multiple_of(ROW_WORK_CHUNK) {
                checked_row_work(cooperation, path, cols % ROW_WORK_CHUNK).await?;
            }
        }
        data_row = data_row.checked_add(1).ok_or_else(matrix_overflow)?;
    }
    if data_row != geometry.rows {
        return Err(changed_source());
    }
    let replay_digest: [u8; 32] = reader.digest.finalize().into();
    if replay_digest != geometry.source_digest {
        return Err(changed_source());
    }
    if let Some(storage) = integer {
        Tensor::from_numeric_storage(storage, vec![rows, cols]).map_err(|source| {
            readmatrix_error_with(
                &READMATRIX_ERROR_TENSOR_BUILD,
                format!("readmatrix: {source}"),
            )
        })
    } else {
        Tensor::new(floating.expect("floating storage"), vec![rows, cols]).map_err(|source| {
            readmatrix_error_with(
                &READMATRIX_ERROR_TENSOR_BUILD,
                format!("readmatrix: {source}"),
            )
        })
    }
}

fn changed_source() -> RuntimeError {
    readmatrix_error_with(
        &READMATRIX_ERROR_IO_READ,
        "readmatrix: opened source changed between validation and output",
    )
}
