//! Two-pass UTF-8 numeric text scanning on one retained seekable source.
//!
//! This owner does not itself certify a finite complete-call peak. The
//! staged caller must promote its original reservation for the opened extent
//! before invoking it and charge its live line, field, and output allocations.

use super::*;
use crate::context::host_effect::ExactHostEffectBytes;
use runmat_filesystem::FsBoundedInput;
use runmat_value::ExactHostAllocationPermit;

mod fill;
pub(super) use fill::fill_utf8;
mod row;
use row::{LineBuffer, RowReader};

#[derive(Default)]
pub(super) struct ScanResources<'a> {
    pub(super) line: Option<ExactHostEffectBytes>,
    pub(super) samples: Option<ExactHostEffectBytes>,
    pub(super) field: Option<ExactHostEffectBytes>,
    pub(super) diagnostic: Option<&'a dyn DiagnosticReservation>,
}

#[derive(Default)]
pub(super) struct FillResources<'a> {
    pub(super) line: Option<ExactHostEffectBytes>,
    pub(super) field: Option<ExactHostEffectBytes>,
    pub(super) output: Option<&'a dyn ExactHostAllocationPermit>,
    pub(super) diagnostic: Option<&'a dyn DiagnosticReservation>,
}

pub(super) struct BoundedScanGeometry {
    pub(super) rows: usize,
    pub(super) columns: usize,
    pub(super) delimiter: Delimiter,
    source_digest: [u8; 32],
}

#[derive(Clone, Copy, Default)]
struct SampleLine {
    number: usize,
    start: usize,
    end: usize,
}

struct SampleRows {
    text: LineBuffer,
    lines: [SampleLine; 32],
    count: usize,
}

impl SampleRows {
    fn new(mut staged: Option<ExactHostEffectBytes>) -> Self {
        if let Some(buffer) = staged.as_mut() {
            // A claimed workspace may be reused after a previous parse. Its
            // initialized prefix is not assumed to contain valid UTF-8.
            buffer.clear();
        }
        Self {
            text: staged.map_or_else(|| LineBuffer::Ordinary(String::new()), LineBuffer::Staged),
            lines: [SampleLine::default(); 32],
            count: 0,
        }
    }

    fn push(&mut self, number: usize, text: &str) -> BuiltinResult<()> {
        let start = self.text.len();
        self.text.append_validated(text)?;
        self.lines[self.count] = SampleLine {
            number,
            start,
            end: self.text.len(),
        };
        self.count += 1;
        Ok(())
    }
}

pub(super) async fn scan_utf8(
    source: &mut FsBoundedInput,
    cooperation: &mut TabularCooperation<'_>,
    path: &impl TabularReadPath,
    options: &ReadMatrixOptions,
    mut resources: ScanResources<'_>,
) -> BuiltinResult<BoundedScanGeometry> {
    let mut reader = RowReader::new(source, resources.line.take(), resources.diagnostic);
    let mut sample = SampleRows::new(resources.samples.take());
    while sample.count < sample.lines.len() {
        let Some(number) = reader
            .next(cooperation, path, options.num_header_lines)
            .await?
        else {
            break;
        };
        sample.push(number, reader.text())?;
    }
    let delimiter = match options.delimiter.clone() {
        Some(delimiter) => delimiter,
        None => {
            let sample_text = sample.text.as_str();
            let mut borrowed = [(0usize, ""); 32];
            for (index, slot) in borrowed.iter_mut().enumerate().take(sample.count) {
                let line = sample.lines[index];
                *slot = (line.number, &sample_text[line.start..line.end]);
            }
            detect_delimiter(&borrowed[..sample.count], cooperation, path)
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
    let sample_text = sample.text.as_str();
    for index in 0..sample.count {
        let line = sample.lines[index];
        let number = line.number;
        let text = &sample_text[line.start..line.end];
        let result = validate_row(
            text,
            number,
            &delimiter,
            integer_dtype,
            options,
            cooperation,
            path,
            resources.field.as_mut(),
            resources.diagnostic,
        )
        .await;
        match result {
            Ok(count) if count != 0 => {
                rows = rows
                    .checked_add(1)
                    .ok_or_else(|| matrix_overflow_with(resources.diagnostic))?;
                columns = columns.max(count);
            }
            Err(error) if semantic_error.is_none() => semantic_error = Some(error),
            _ => {}
        }
    }
    drop(sample);
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
                resources.field.as_mut(),
                resources.diagnostic,
            )
            .await
            {
                Ok(count) if count != 0 => {
                    rows = rows
                        .checked_add(1)
                        .ok_or_else(|| matrix_overflow_with(resources.diagnostic))?;
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
    let source_digest = reader.finish_digest();
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
    path: &impl TabularReadPath,
    staged_field: Option<&mut ExactHostEffectBytes>,
    diagnostic_reservation: Option<&dyn DiagnosticReservation>,
) -> BuiltinResult<usize> {
    scan::for_each_field_ref(
        text,
        delimiter,
        cooperation,
        path,
        staged_field,
        |index, field| {
            if let Some(dtype) = integer_dtype {
                parse_integer_numeric_token_with_reservation(
                    field.as_str(),
                    options,
                    dtype,
                    line_number,
                    options.column_from_source(index),
                    diagnostic_reservation,
                )?;
            } else {
                validate_numeric_token_with_reservation(
                    field.as_str(),
                    options,
                    line_number,
                    options.column_from_source(index),
                    diagnostic_reservation,
                )?;
            }
            Ok(())
        },
    )
    .await
}

pub(super) fn matrix_overflow() -> RuntimeError {
    matrix_overflow_with(None)
}

pub(super) fn matrix_overflow_with(
    reservation: Option<&dyn DiagnosticReservation>,
) -> RuntimeError {
    FixedReaderDiagnostic::MatrixOverflow.render(reservation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reused_sample_workspace_discards_invalid_old_bytes_before_utf8_access() {
        let mut workspace = ExactHostEffectBytes::test_buffer(16);
        workspace.append(&[0xff]).expect("previous invalid bytes");
        let mut samples = SampleRows::new(Some(workspace));
        samples.push(3, "1,2").expect("fresh sample");
        let line = samples.lines[0];
        assert_eq!(line.number, 3);
        assert_eq!(&samples.text.as_str()[line.start..line.end], "1,2");
    }
}
