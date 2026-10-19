#[cfg(test)]
use super::super::path::TabularFsPath;
use super::admission;
use super::*;
use crate::context::host_effect::{
    select_host_effect_after_stage, HostEffectFinalFacts, HostEffectLiveAllocationKind,
    HostEffectStagePermit,
};
use bounded::matrix_overflow;
use runmat_builtins::{ReadMatrixResultClass, ReadMatrixTextResourceFacts};
use runmat_execution::WholeWorkObservationOutcome;
use std::borrow::Cow;

mod scan;
mod bounded;
use scan::{checked_row_work, collect_data_lines, detect_delimiter, split_fields};

// Bounds synchronous work between checks; the execution service still owns
// the decision to yield and the cancellation policy.
const ROW_WORK_CHUNK: usize = 4 * 1024;

pub(super) async fn read_default_staged(
    runtime: &crate::context::RuntimeContext,
    cooperation: &mut TabularCooperation<'_>,
    path: &impl TabularReadPath,
    filesystem: &std::sync::Arc<dyn runmat_filesystem::FsProvider>,
    stage: HostEffectStagePermit,
) -> BuiltinResult<Value> {
    let mut source = await_io(Some(runtime), BUILTIN_NAME, path.open_bounded_input())
        .await?
        .map_err(|source| {
            readmatrix_error_with_source(
                &READMATRIX_ERROR_IO_OPEN,
                "readmatrix: unable to open admitted bounded source",
                source,
            )
        })?
        .ok_or_else(|| {
            readmatrix_error_with(
                &READMATRIX_ERROR_IO_OPEN,
                "readmatrix: selected source has no bounded input contract",
            )
        })?;
    cooperation.completed(0).await?;
    stage.checkpoint()?;
    let extent = source.opened_extent();
    if extent > MAX_READMATRIX_TEXT_BYTES {
        return Err(readmatrix_error_with(
            &READMATRIX_ERROR_IO_READ,
            format!("readmatrix: file exceeds maximum supported text size of {MAX_READMATRIX_TEXT_BYTES} bytes"),
        ));
    }
    // These three live owners can overlap during the scan: one complete
    // source line, the first 32 sampled lines, and one quoted-field scratch.
    // Each holds at most bytes from this opened extent. The sample owner is
    // dropped before replay; replay needs only line and field scratch.
    let capacity = usize::try_from(extent).map_err(|_| matrix_overflow())?;
    let diagnostic_bytes = maximum_numeric_diagnostic_bytes(capacity)
        .and_then(|numeric| maximum_fixed_diagnostic_bytes().map(|fixed| numeric.max(fixed)))
        .ok_or_else(matrix_overflow)?;
    let diagnostic_bytes = u64::try_from(diagnostic_bytes).map_err(|_| matrix_overflow())?;
    let claim_bytes =
        u64::try_from(stage.live_claim_metadata_bytes()).map_err(|_| matrix_overflow())?;
    let scan_buffers = extent
        .checked_add(claim_bytes)
        .and_then(|bytes| bytes.checked_mul(3))
        .ok_or_else(matrix_overflow)?;
    let promoted = stage
        .reserved_bytes()
        .checked_add(scan_buffers)
        .and_then(|bytes| bytes.checked_add(diagnostic_bytes))
        .and_then(|bytes| bytes.checked_add(claim_bytes))
        .ok_or_else(matrix_overflow)?;
    stage
        .promote_scan_envelope(promoted)
        .map_err(|source| admission::allocation_error(BUILTIN_NAME, source))?;
    let workspace = || {
        stage
            .allocate_exact_bytes(HostEffectLiveAllocationKind::CodecWorkspace, capacity)
            .map_err(|source| admission::allocation_error(BUILTIN_NAME, source))
    };
    let options = ReadMatrixOptions::default();
    let geometry = bounded::scan_utf8(
        &mut source,
        cooperation,
        path,
        &options,
        bounded::ScanResources {
            line: Some(workspace()?),
            samples: Some(workspace()?),
            field: Some(workspace()?),
            diagnostic: Some(&stage),
        },
    )
    .await?;
    let fill_line = workspace()?;
    let fill_field = workspace()?;
    let facts = ReadMatrixTextResourceFacts {
        opened_extent: extent,
        source_resource_revision: *source.resource_revision().bytes(),
        rows: u64::try_from(geometry.rows).map_err(|_| matrix_overflow())?,
        columns: u64::try_from(geometry.columns).map_err(|_| matrix_overflow())?,
        result_class: ReadMatrixResultClass::Numeric(runmat_types::NumericClass::Double),
    };
    let receipt = select_host_effect_after_stage(
        runtime,
        admission::request(
            path.path(),
            filesystem,
            admission::controls_digest(&options).expect("default controls are complete"),
            admission::digest_serializable(&facts),
        ),
        HostEffectFinalFacts::ReadMatrixText(&facts),
        stage,
    )?;
    let result = async {
        let output_permit = receipt.output_allocation_permit().ok_or_else(|| {
            readmatrix_error_with(
                &READMATRIX_ERROR_TENSOR_BUILD,
                "readmatrix: selected output has no exact allocation authority",
            )
        })?;
        let tensor = bounded::fill_utf8(
            &mut source,
            &geometry,
            cooperation,
            path,
            &options,
            bounded::FillResources {
                line: Some(fill_line),
                field: Some(fill_field),
                output: Some(output_permit.as_ref()),
                diagnostic: Some(&receipt),
            },
        )
        .await?;
        cooperation.completed(0).await?;
        path.check().map_err(|source| {
            readmatrix_error_with_source(
                &READMATRIX_ERROR_IO_READ,
                "readmatrix: captured filesystem authority was revoked",
                source,
            )
        })?;
        receipt.checkpoint()?;
        Ok(Value::Tensor(tensor))
    }
    .await;
    // The stage precharge also owns the opened provider handle's requested
    // heap bytes. Release that handle before retiring the stage reservation.
    drop(source);
    receipt.finish(if runtime.is_cancelled() {
        WholeWorkObservationOutcome::Cancelled
    } else if result.is_ok() {
        WholeWorkObservationOutcome::Success
    } else {
        WholeWorkObservationOutcome::OperationalFailure
    });
    result
}

pub(super) async fn read_numeric_matrix(
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut TabularCooperation<'_>,
    path: &impl TabularReadPath,
    options: &ReadMatrixOptions,
) -> BuiltinResult<Tensor> {
    let file = await_io(runtime, BUILTIN_NAME, path.open_read_preferred())
        .await?
        .map_err(|source| {
            readmatrix_error_with_source(
                &READMATRIX_ERROR_IO_OPEN,
                format!(
                    "readmatrix: unable to read '{}': {source}",
                    path.path().display()
                ),
                source,
            )
        })?;
    cooperation.completed(0).await?;
    path.check().map_err(|source| {
        readmatrix_error_with_source(
            &READMATRIX_ERROR_IO_READ,
            "readmatrix: captured filesystem authority was revoked",
            source,
        )
    })?;
    match file {
        TabularReadSource::Bounded(mut source)
            if normalized_encoding_label(options.encoding.trim()).eq_ignore_ascii_case("utf-8") =>
        {
            if source.opened_extent() > MAX_READMATRIX_TEXT_BYTES {
                return Err(readmatrix_error_with(
                    &READMATRIX_ERROR_IO_READ,
                    format!(
                        "readmatrix: file exceeds maximum supported text size of {MAX_READMATRIX_TEXT_BYTES} bytes"
                    ),
                ));
            }
            let geometry = bounded::scan_utf8(
                &mut source,
                cooperation,
                path,
                options,
                bounded::ScanResources::default(),
            )
            .await?;
            bounded::fill_utf8(
                &mut source,
                &geometry,
                cooperation,
                path,
                options,
                bounded::FillResources::default(),
            )
            .await
        }
        other => read_numeric_matrix_from_bytes(cooperation, path, options, other).await,
    }
}

async fn read_numeric_matrix_from_bytes(
    cooperation: &mut TabularCooperation<'_>,
    path: &impl TabularReadPath,
    options: &ReadMatrixOptions,
    file: TabularReadSource,
) -> BuiltinResult<Tensor> {
    let bytes = read_file_bytes(cooperation, path, file).await?;
    let decoded = decode_text_bytes(&bytes, &options.encoding)?;
    drop(bytes);
    let text = strip_utf8_bom(&decoded);
    let mut data_lines =
        collect_data_lines(text, options.num_header_lines, cooperation, path).await?;

    if data_lines.is_empty() {
        return Tensor::new(Vec::new(), vec![0, 0]).map_err(|e| {
            readmatrix_error_with(&READMATRIX_ERROR_TENSOR_BUILD, format!("readmatrix: {e}"))
        });
    }

    if let Some((_, first_line)) = data_lines.first_mut() {
        let stripped = (*first_line).trim_start_matches('\u{FEFF}');
        *first_line = stripped;
    }

    let delimiter = if let Some(delimiter) = options.delimiter.clone() {
        delimiter
    } else {
        detect_delimiter(&data_lines, cooperation, path)
            .await?
            .unwrap_or(Delimiter::Whitespace)
    };
    let integer_dtype = integer_output_dtype(&options.output_template);

    let mut rows: Vec<Vec<Cow<'_, str>>> = Vec::new();
    let mut max_cols = 0usize;

    for (line_number, text) in &data_lines {
        let fields = split_fields(text, &delimiter, cooperation, path).await?;
        if fields.is_empty() {
            continue;
        }
        for (index, field) in fields.iter().enumerate() {
            if let Some(dtype) = integer_dtype {
                parse_integer_numeric_token(
                    field,
                    options,
                    dtype,
                    *line_number,
                    options.column_from_source(index),
                )?;
            } else {
                validate_numeric_token(
                    field,
                    options,
                    *line_number,
                    options.column_from_source(index),
                )?;
            }
            if (index + 1) % ROW_WORK_CHUNK == 0 {
                checked_row_work(cooperation, path, ROW_WORK_CHUNK).await?;
            }
        }
        if fields.len() % ROW_WORK_CHUNK != 0 {
            checked_row_work(cooperation, path, fields.len() % ROW_WORK_CHUNK).await?;
        }
        max_cols = max_cols.max(fields.len());
        rows.push(fields);
    }

    if rows.is_empty() {
        return Tensor::new(Vec::new(), vec![0, 0]).map_err(|e| {
            readmatrix_error_with(&READMATRIX_ERROR_TENSOR_BUILD, format!("readmatrix: {e}"))
        });
    }

    if let Some(range) = &options.range {
        let (subset_rows, subset_cols) = apply_range(rows, max_cols, range, Cow::Borrowed(""));
        rows = subset_rows;
        max_cols = subset_cols;
    }

    if rows.is_empty() || max_cols == 0 {
        return Tensor::new(Vec::new(), vec![0, 0]).map_err(|e| {
            readmatrix_error_with(&READMATRIX_ERROR_TENSOR_BUILD, format!("readmatrix: {e}"))
        });
    }
    let row_count = rows.len();
    let element_count = row_count.checked_mul(max_cols).ok_or_else(|| {
        readmatrix_error_with(
            &READMATRIX_ERROR_TENSOR_BUILD,
            "readmatrix: matrix dimensions exceed addressable storage",
        )
    })?;
    if let Some(dtype) = integer_dtype {
        let mut storage = NumericStorage::zeros(dtype, element_count);
        for (row_index, row) in rows.iter().enumerate() {
            for col_index in 0..max_cols {
                let token = row.get(col_index).map(|field| field.as_ref()).unwrap_or("");
                let value = parse_integer_numeric_token(
                    token,
                    options,
                    dtype,
                    row_index + 1,
                    options.column_from_selected(col_index),
                )?;
                storage
                    .set_value(col_index * row_count + row_index, value)
                    .map_err(|error| {
                        readmatrix_error_with(
                            &READMATRIX_ERROR_TENSOR_BUILD,
                            format!("readmatrix: {error}"),
                        )
                    })?;
                if (col_index + 1) % ROW_WORK_CHUNK == 0 {
                    checked_row_work(cooperation, path, ROW_WORK_CHUNK).await?;
                }
            }
            if !max_cols.is_multiple_of(ROW_WORK_CHUNK) {
                checked_row_work(cooperation, path, max_cols % ROW_WORK_CHUNK).await?;
            }
        }
        return Tensor::from_numeric_storage(storage, vec![row_count, max_cols]).map_err(|error| {
            readmatrix_error_with(
                &READMATRIX_ERROR_TENSOR_BUILD,
                format!("readmatrix: {error}"),
            )
        });
    }

    let default_fill = options.empty_value_f64();
    let mut data = vec![default_fill; element_count];
    for (row_index, row) in rows.iter().enumerate() {
        for col_index in 0..max_cols {
            let token = row.get(col_index).map(|field| field.as_ref()).unwrap_or("");
            data[col_index * row_count + row_index] = parse_numeric_token(
                token,
                options,
                row_index + 1,
                options.column_from_selected(col_index),
            )?;
            if (col_index + 1) % ROW_WORK_CHUNK == 0 {
                checked_row_work(cooperation, path, ROW_WORK_CHUNK).await?;
            }
        }
        if !max_cols.is_multiple_of(ROW_WORK_CHUNK) {
            checked_row_work(cooperation, path, max_cols % ROW_WORK_CHUNK).await?;
        }
    }
    Tensor::new(data, vec![row_count, max_cols]).map_err(|error| {
        readmatrix_error_with(
            &READMATRIX_ERROR_TENSOR_BUILD,
            format!("readmatrix: {error}"),
        )
    })
}

async fn read_file_bytes(
    cooperation: &mut TabularCooperation<'_>,
    path: &impl TabularReadPath,
    file: TabularReadSource,
) -> BuiltinResult<Vec<u8>> {
    path.check().map_err(|source| {
        readmatrix_error_with_source(
            &READMATRIX_ERROR_IO_READ,
            "readmatrix: captured filesystem authority was revoked",
            source,
        )
    })?;
    if file
        .opened_extent()
        .is_some_and(|extent| extent > MAX_READMATRIX_TEXT_BYTES)
    {
        return Err(readmatrix_error_with(
            &READMATRIX_ERROR_IO_READ,
            format!(
                "readmatrix: file exceeds maximum supported text size of {MAX_READMATRIX_TEXT_BYTES} bytes"
            ),
        ));
    }
    let mut limited = file.take(MAX_READMATRIX_TEXT_BYTES + 1);
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 64 * 1024];
    loop {
        let count = limited.read(&mut chunk).map_err(|err| {
            readmatrix_error_with_source(
                &READMATRIX_ERROR_IO_READ,
                format!(
                    "readmatrix: error reading '{}': {err}",
                    path.path().display()
                ),
                err,
            )
        })?;
        if count == 0 {
            break;
        }
        bytes.extend_from_slice(&chunk[..count]);
        cooperation.completed(count).await?;
        path.check().map_err(|source| {
            readmatrix_error_with_source(
                &READMATRIX_ERROR_IO_READ,
                "readmatrix: captured filesystem authority was revoked",
                source,
            )
        })?;
    }
    if bytes.len() as u64 > MAX_READMATRIX_TEXT_BYTES {
        return Err(readmatrix_error_with(
            &READMATRIX_ERROR_IO_READ,
            format!(
                "readmatrix: file exceeds maximum supported text size of {MAX_READMATRIX_TEXT_BYTES} bytes"
            ),
        ));
    }
    Ok(bytes)
}

pub(super) fn validate_encoding_label(label: &str) -> BuiltinResult<()> {
    let label = label.trim();
    if label.eq_ignore_ascii_case("auto") || label.eq_ignore_ascii_case("unicode") {
        return Ok(());
    }
    decode_bytes(&[], normalized_encoding_label(label), BUILTIN_NAME)
        .map(|_| ())
        .map_err(|_| {
            readmatrix_error_with(
                &READMATRIX_ERROR_OPTION_VALUE,
                format!("readmatrix: unsupported Encoding '{label}'"),
            )
        })
}

fn decode_text_bytes(bytes: &[u8], encoding: &str) -> BuiltinResult<String> {
    let label = encoding.trim();
    let normalized = normalized_encoding_label(label);
    let chars = if normalized.eq_ignore_ascii_case("auto") {
        if let Some((encoding, bom_len)) = Encoding::for_bom(bytes) {
            decode_bom_bytes(bytes, encoding, bom_len)
        } else {
            decode_bytes(bytes, "utf-8", BUILTIN_NAME)
        }
    } else {
        decode_bytes(bytes, normalized, BUILTIN_NAME)
    }
    .map_err(|err| {
        let message = err.message().to_string();
        readmatrix_error_with_source(&READMATRIX_ERROR_IO_READ, message, err)
    })?;
    Ok(chars.into_iter().collect())
}

fn decode_bom_bytes(
    bytes: &[u8],
    encoding: &'static Encoding,
    bom_len: usize,
) -> BuiltinResult<Vec<char>> {
    let (decoded, had_errors) = encoding.decode_without_bom_handling(&bytes[bom_len..]);
    if had_errors {
        return Err(readmatrix_error_with(
            &READMATRIX_ERROR_IO_READ,
            "readmatrix: file contains bytes that are invalid for the detected Encoding",
        ));
    }
    Ok(decoded.chars().collect())
}

fn normalized_encoding_label(label: &str) -> &str {
    if label.eq_ignore_ascii_case("unicode") {
        "utf-8"
    } else if label.eq_ignore_ascii_case("default") || label.eq_ignore_ascii_case("native") {
        "system"
    } else {
        label
    }
}

fn strip_utf8_bom(text: &str) -> &str {
    text.strip_prefix('\u{FEFF}').unwrap_or(text)
}

mod diagnostic;
use diagnostic::{maximum_fixed_diagnostic_bytes, DiagnosticReservation, FixedReaderDiagnostic};
mod numeric;
pub(super) use numeric::parse_integer_numeric_token;
use numeric::{
    integer_output_dtype, maximum_numeric_diagnostic_bytes,
    parse_integer_numeric_token_with_reservation, parse_numeric_token,
    parse_numeric_token_with_reservation, validate_numeric_token,
    validate_numeric_token_with_reservation,
};

#[cfg(test)]
#[path = "reader/tests.rs"]
mod tests;
