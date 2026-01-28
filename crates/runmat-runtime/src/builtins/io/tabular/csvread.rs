//! MATLAB-compatible `csvread` builtin for RunMat.
//!
//! `csvread` is largely superseded by `readmatrix`, but MATLAB users still rely on
//! its terse API for numeric CSV imports. This implementation mirrors MATLAB's
//! zero-based range semantics while integrating with the modern builtin template.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use runmat_builtins::{Tensor, Value};
use runmat_filesystem::File;
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "csvread";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::tabular::csvread")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "csvread",
    op_kind: GpuOpKind::Custom("io-csvread"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Runs entirely on the host; acceleration providers are not involved.",
};

fn csvread_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn csvread_error_with_source<E>(message: impl Into<String>, source: E) -> RuntimeError
where
    E: std::error::Error + Send + Sync + 'static,
{
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source)
        .build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(|value| value.to_string());
    let message = err.message().to_string();
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::tabular::csvread")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "csvread",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion; executes as a standalone host operation.",
};

#[runtime_builtin(
    name = "csvread",
    category = "io/tabular",
    summary = "Read numeric data from a comma-separated text file.",
    keywords = "csvread,csv,dlmread,numeric import,range",
    accel = "cpu",
    builtin_path = "crate::builtins::io::tabular::csvread"
)]
async fn csvread_builtin(path: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let gathered_path = gather_if_needed_async(&path)
        .await
        .map_err(map_control_flow)?;
    let options = parse_arguments(&rest).await?;
    let resolved = resolve_path(&gathered_path)?;
    let (rows, max_cols) = read_csv_rows(&resolved)?;
    let subset = if let Some(range) = options.range {
        apply_range(&rows, max_cols, &range, 0.0)
    } else {
        apply_offsets(&rows, max_cols, options.start_row, options.start_col, 0.0)
    };
    let tensor = rows_to_tensor(subset.rows, subset.row_count, subset.col_count, 0.0)?;
    Ok(Value::Tensor(tensor))
}

#[derive(Debug, Default)]
struct CsvReadOptions {
    start_row: usize,
    start_col: usize,
    range: Option<RangeSpec>,
}

async fn parse_arguments(args: &[Value]) -> BuiltinResult<CsvReadOptions> {
    let mut gathered = Vec::with_capacity(args.len());
    for value in args {
        gathered.push(
            gather_if_needed_async(value)
                .await
                .map_err(map_control_flow)?,
        );
    }
    match gathered.len() {
        0 => Ok(CsvReadOptions::default()),
        2 => {
            let start_row = value_to_start_index(&gathered[0], "row")?;
            let start_col = value_to_start_index(&gathered[1], "col")?;
            Ok(CsvReadOptions {
                start_row,
                start_col,
                range: None,
            })
        }
        3 => {
            let start_row = value_to_start_index(&gathered[0], "row")?;
            let start_col = value_to_start_index(&gathered[1], "col")?;
            let range = parse_range(&gathered[2])?;
            Ok(CsvReadOptions {
                start_row,
                start_col,
                range: Some(range),
            })
        }
        _ => Err(csvread_error(
            "csvread: expected csvread(filename[, row, col[, range]])",
        )),
    }
}

fn value_to_start_index(value: &Value, name: &str) -> BuiltinResult<usize> {
    match value {
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < 0 {
                return Err(csvread_error(format!(
                    "csvread: {name} must be a non-negative integer"
                )));
            }
            usize::try_from(raw).map_err(|_| csvread_error(format!("csvread: {name} is too large")))
        }
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(csvread_error(format!(
                    "csvread: {name} must be a finite integer"
                )));
            }
            if *n < 0.0 {
                return Err(csvread_error(format!(
                    "csvread: {name} must be a non-negative integer"
                )));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(csvread_error(format!("csvread: {name} must be an integer")));
            }
            usize::try_from(rounded as i64)
                .map_err(|_| csvread_error(format!("csvread: {name} is too large")))
        }
        _ => Err(csvread_error(format!(
            "csvread: expected {name} as a numeric scalar, got {value:?}"
        ))),
    }
}

fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
    match value {
        Value::String(s) => normalize_path(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            normalize_path(&text)
        }
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                normalize_path(&sa.data[0])
            } else {
                Err(csvread_error("csvread: string array inputs must be scalar"))
            }
        }
        Value::CharArray(_) => Err(csvread_error(
            "csvread: expected a 1-by-N character vector for the file name",
        )),
        other => Err(csvread_error(format!(
            "csvread: expected filename as string scalar or character vector, got {other:?}"
        ))),
    }
}

fn normalize_path(raw: &str) -> BuiltinResult<PathBuf> {
    if raw.trim().is_empty() {
        return Err(csvread_error("csvread: filename must not be empty"));
    }
    let expanded = expand_user_path(raw, BUILTIN_NAME).map_err(csvread_error)?;
    Ok(Path::new(&expanded).to_path_buf())
}

fn read_csv_rows(path: &Path) -> BuiltinResult<(Vec<Vec<f64>>, usize)> {
    let file = File::open(path).map_err(|err| {
        csvread_error_with_source(
            format!("csvread: unable to open '{}': {err}", path.display()),
            err,
        )
    })?;
    let mut reader = BufReader::new(file);
    let mut buffer = String::new();
    let mut rows = Vec::new();
    let mut max_cols = 0usize;
    let mut line_index = 0usize;

    loop {
        buffer.clear();
        let bytes = reader.read_line(&mut buffer).map_err(|err| {
            csvread_error_with_source(
                format!("csvread: failed to read '{}': {err}", path.display()),
                err,
            )
        })?;
        if bytes == 0 {
            break;
        }
        line_index += 1;
        if buffer.trim().is_empty() {
            continue;
        }
        if buffer.ends_with('\n') {
            buffer.pop();
            if buffer.ends_with('\r') {
                buffer.pop();
            }
        } else if buffer.ends_with('\r') {
            buffer.pop();
        }
        let parsed = parse_csv_row(&buffer, line_index)?;
        max_cols = max_cols.max(parsed.len());
        rows.push(parsed);
    }

    Ok((rows, max_cols))
}

fn parse_csv_row(line: &str, line_index: usize) -> BuiltinResult<Vec<f64>> {
    let mut values = Vec::new();
    for (col_index, raw_field) in line.split(',').enumerate() {
        let trimmed = raw_field.trim();
        if trimmed.is_empty() {
            values.push(0.0);
            continue;
        }
        let unwrapped = if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2
        {
            &trimmed[1..trimmed.len() - 1]
        } else {
            trimmed
        };
        let lowered = unwrapped.to_ascii_lowercase();
        let value = match lowered.as_str() {
            "nan" => f64::NAN,
            "inf" | "+inf" => f64::INFINITY,
            "-inf" => f64::NEG_INFINITY,
            _ => unwrapped.parse::<f64>().map_err(|_| {
                csvread_error(format!(
                    "csvread: nonnumeric token '{}' at row {}, column {}",
                    unwrapped,
                    line_index,
                    col_index + 1
                ))
            })?,
        };
        values.push(value);
    }
    Ok(values)
}

#[derive(Clone, Copy, Debug)]
struct RangeSpec {
    start_row: usize,
    start_col: usize,
    end_row: Option<usize>,
    end_col: Option<usize>,
}

fn parse_range(value: &Value) -> BuiltinResult<RangeSpec> {
    match value {
        Value::String(s) => parse_range_string(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            parse_range_string(&text)
        }
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                parse_range_string(&sa.data[0])
            } else {
                Err(csvread_error(
                    "csvread: Range string array inputs must be scalar",
                ))
            }
        }
        Value::Tensor(_) => parse_range_numeric(value),
        _ => Err(csvread_error(
            "csvread: Range must be provided as a string or numeric vector",
        )),
    }
}

fn parse_range_string(text: &str) -> BuiltinResult<RangeSpec> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(csvread_error("csvread: Range string cannot be empty"));
    }
    let parts: Vec<&str> = trimmed.split(':').collect();
    if parts.len() > 2 {
        return Err(csvread_error(format!(
            "csvread: invalid Range specification '{trimmed}'"
        )));
    }
    let start = parse_cell_reference(parts[0])?;
    if start.col.is_none() {
        return Err(csvread_error(
            "csvread: Range must specify a starting column",
        ));
    }
    let end = if parts.len() == 2 {
        Some(parse_cell_reference(parts[1])?)
    } else {
        None
    };
    if let Some(ref end_ref) = end {
        if end_ref.col.is_none() {
            return Err(csvread_error(
                "csvread: Range end must include a column reference",
            ));
        }
    }
    let start_row = start.row.unwrap_or(0);
    let start_col = start.col.unwrap();
    let end_row = end.as_ref().and_then(|r| r.row);
    let end_col = end.as_ref().and_then(|r| r.col);
    Ok(RangeSpec {
        start_row,
        start_col,
        end_row,
        end_col,
    })
}

fn parse_range_numeric(value: &Value) -> BuiltinResult<RangeSpec> {
    let elements = match value {
        Value::Tensor(t) => t.data.clone(),
        _ => {
            return Err(csvread_error(
                "csvread: numeric Range must be provided as a vector with 2 or 4 elements",
            ))
        }
    };
    if elements.len() != 2 && elements.len() != 4 {
        return Err(csvread_error(
            "csvread: numeric Range must contain exactly 2 or 4 elements",
        ));
    }
    let mut indices = Vec::with_capacity(elements.len());
    for (idx, element) in elements.iter().enumerate() {
        indices.push(non_negative_index(*element, idx)?);
    }
    let start_row = indices[0];
    let start_col = indices[1];
    let (end_row, end_col) = if indices.len() == 4 {
        (Some(indices[2]), Some(indices[3]))
    } else {
        (None, None)
    };
    Ok(RangeSpec {
        start_row,
        start_col,
        end_row,
        end_col,
    })
}

fn non_negative_index(value: f64, position: usize) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(csvread_error("csvread: Range indices must be finite"));
    }
    if value < 0.0 {
        return Err(csvread_error("csvread: Range indices must be non-negative"));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(csvread_error("csvread: Range indices must be integers"));
    }
    usize::try_from(rounded as i64).map_err(|_| {
        csvread_error(format!(
            "csvread: Range index {} is too large to fit in usize",
            position + 1
        ))
    })
}

#[derive(Clone, Copy)]
struct CellReference {
    row: Option<usize>,
    col: Option<usize>,
}

fn parse_cell_reference(token: &str) -> BuiltinResult<CellReference> {
    let mut letters = String::new();
    let mut digits = String::new();
    for ch in token.trim().chars() {
        if ch == '$' {
            continue;
        }
        if ch.is_ascii_alphabetic() {
            letters.push(ch.to_ascii_uppercase());
        } else if ch.is_ascii_digit() {
            digits.push(ch);
        } else {
            return Err(csvread_error(format!(
                "csvread: invalid Range component '{token}'"
            )));
        }
    }
    if letters.is_empty() && digits.is_empty() {
        return Err(csvread_error("csvread: Range references cannot be empty"));
    }
    let col = if letters.is_empty() {
        None
    } else {
        Some(column_index_from_letters(&letters)?)
    };
    let row = if digits.is_empty() {
        None
    } else {
        let parsed = digits.parse::<usize>().map_err(|_| {
            csvread_error(format!(
                "csvread: invalid row index '{}' in Range component '{token}'",
                digits
            ))
        })?;
        if parsed == 0 {
            return Err(csvread_error("csvread: Range rows must be >= 1"));
        }
        Some(parsed - 1)
    };
    Ok(CellReference { row, col })
}

fn column_index_from_letters(letters: &str) -> BuiltinResult<usize> {
    let mut value: usize = 0;
    for ch in letters.chars() {
        if !ch.is_ascii_uppercase() {
            return Err(csvread_error(format!(
                "csvread: invalid column designator '{letters}' in Range"
            )));
        }
        let digit = (ch as u8 - b'A' + 1) as usize;
        value = value
            .checked_mul(26)
            .and_then(|v| v.checked_add(digit))
            .ok_or_else(|| csvread_error("csvread: Range column index overflowed"))?;
    }
    value
        .checked_sub(1)
        .ok_or_else(|| csvread_error("csvread: Range column index underflowed"))
}

struct SubsetResult {
    rows: Vec<Vec<f64>>,
    row_count: usize,
    col_count: usize,
}

fn apply_offsets(
    rows: &[Vec<f64>],
    max_cols: usize,
    start_row: usize,
    start_col: usize,
    default_fill: f64,
) -> SubsetResult {
    if rows.is_empty() || max_cols == 0 {
        return SubsetResult {
            rows: Vec::new(),
            row_count: 0,
            col_count: 0,
        };
    }
    if start_row >= rows.len() {
        return SubsetResult {
            rows: Vec::new(),
            row_count: 0,
            col_count: 0,
        };
    }
    if start_col >= max_cols {
        return SubsetResult {
            rows: Vec::new(),
            row_count: 0,
            col_count: 0,
        };
    }

    let mut subset_rows = Vec::new();
    let mut col_count = 0usize;
    for row in rows.iter().skip(start_row) {
        if start_col >= row.len() && row.len() < max_cols {
            // Entire row missing required columns; fill zeros of remaining width.
            let width = max_cols - start_col;
            subset_rows.push(vec![default_fill; width]);
            col_count = col_count.max(width);
            continue;
        }
        let mut extracted = Vec::with_capacity(max_cols - start_col);
        for col_idx in start_col..max_cols {
            let value = row.get(col_idx).copied().unwrap_or(default_fill);
            extracted.push(value);
        }
        col_count = col_count.max(extracted.len());
        subset_rows.push(extracted);
    }
    let row_count = subset_rows.len();
    SubsetResult {
        rows: subset_rows,
        row_count,
        col_count,
    }
}

fn apply_range(
    rows: &[Vec<f64>],
    max_cols: usize,
    range: &RangeSpec,
    default_fill: f64,
) -> SubsetResult {
    if rows.is_empty() || max_cols == 0 {
        return SubsetResult {
            rows: Vec::new(),
            row_count: 0,
            col_count: 0,
        };
    }
    if range.start_row >= rows.len() || range.start_col >= max_cols {
        return SubsetResult {
            rows: Vec::new(),
            row_count: 0,
            col_count: 0,
        };
    }
    let last_row = rows.len().saturating_sub(1);
    let mut end_row = range.end_row.unwrap_or(last_row);
    if end_row > last_row {
        end_row = last_row;
    }
    if end_row < range.start_row {
        return SubsetResult {
            rows: Vec::new(),
            row_count: 0,
            col_count: 0,
        };
    }

    let last_col = max_cols.saturating_sub(1);
    let mut end_col = range.end_col.unwrap_or(last_col);
    if end_col > last_col {
        end_col = last_col;
    }
    if end_col < range.start_col {
        return SubsetResult {
            rows: Vec::new(),
            row_count: 0,
            col_count: 0,
        };
    }

    let mut subset_rows = Vec::new();
    let mut col_count = 0usize;
    for row_idx in range.start_row..=end_row {
        if row_idx >= rows.len() {
            break;
        }
        let row = &rows[row_idx];
        let mut extracted = Vec::with_capacity(end_col - range.start_col + 1);
        for col_idx in range.start_col..=end_col {
            if col_idx >= max_cols {
                break;
            }
            let value = row.get(col_idx).copied().unwrap_or(default_fill);
            extracted.push(value);
        }
        col_count = col_count.max(extracted.len());
        subset_rows.push(extracted);
    }
    let row_count = subset_rows.len();
    SubsetResult {
        rows: subset_rows,
        row_count,
        col_count,
    }
}

fn rows_to_tensor(
    rows: Vec<Vec<f64>>,
    row_count: usize,
    col_count: usize,
    default_fill: f64,
) -> BuiltinResult<Tensor> {
    if row_count == 0 || col_count == 0 {
        return Tensor::new(Vec::new(), vec![0, 0])
            .map_err(|e| csvread_error(format!("csvread: {e}")));
    }
    let mut data = vec![default_fill; row_count * col_count];
    for (row_idx, row) in rows.iter().enumerate().take(row_count) {
        for col_idx in 0..col_count {
            let value = row.get(col_idx).copied().unwrap_or(default_fill);
            data[row_idx + col_idx * row_count] = value;
        }
    }
    Tensor::new(data, vec![row_count, col_count])
        .map_err(|e| csvread_error(format!("csvread: {e}")))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_time::unix_timestamp_ns;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use runmat_builtins::{CharArray, IntValue, Tensor as BuiltinTensor};

    use crate::builtins::common::test_support;

    fn csvread_builtin(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::csvread_builtin(path, rest))
    }

    static UNIQUE_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn unique_path(prefix: &str) -> PathBuf {
        let nanos = unix_timestamp_ns();
        let seq = UNIQUE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "runmat_csvread_{prefix}_{}_{}_{}",
            std::process::id(),
            nanos,
            seq
        ));
        path
    }

    fn write_temp_file(lines: &[&str]) -> PathBuf {
        let path = unique_path("input").with_extension("csv");
        let contents = lines.join("\n");
        fs::write(&path, contents).expect("write temp csv");
        path
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn csvread_basic_csv_roundtrip() {
        let path = write_temp_file(&["1,2,3", "4,5,6"]);
        let result = csvread_builtin(Value::from(path.to_string_lossy().to_string()), Vec::new())
            .expect("csvread");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3]);
                assert_eq!(t.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        fs::remove_file(path).ok();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn csvread_with_offsets() {
        let path = write_temp_file(&["0,1,2", "3,4,5", "6,7,8"]);
        let args = vec![Value::Int(IntValue::I32(1)), Value::Int(IntValue::I32(1))];
        let result =
            csvread_builtin(Value::from(path.to_string_lossy().to_string()), args).expect("csv");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![4.0, 7.0, 5.0, 8.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        fs::remove_file(path).ok();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn csvread_with_numeric_range() {
        let path = write_temp_file(&["1,2,3", "4,5,6", "7,8,9"]);
        let args = vec![
            Value::Int(IntValue::I32(0)),
            Value::Int(IntValue::I32(0)),
            Value::from(BuiltinTensor::new(vec![1.0, 1.0, 2.0, 2.0], vec![4, 1]).expect("tensor")),
        ];
        let result =
            csvread_builtin(Value::from(path.to_string_lossy().to_string()), args).expect("csv");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![5.0, 8.0, 6.0, 9.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        fs::remove_file(path).ok();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn csvread_with_string_range() {
        let path = write_temp_file(&["1,2,3", "4,5,6", "7,8,9"]);
        let args = vec![
            Value::Int(IntValue::I32(0)),
            Value::Int(IntValue::I32(0)),
            Value::from("B2:C3"),
        ];
        let result =
            csvread_builtin(Value::from(path.to_string_lossy().to_string()), args).expect("csv");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![5.0, 8.0, 6.0, 9.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        fs::remove_file(path).ok();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn csvread_empty_fields_become_zero() {
        let path = write_temp_file(&["1,,3", ",5,", "7,8,"]);
        let result = csvread_builtin(Value::from(path.to_string_lossy().to_string()), Vec::new())
            .expect("csv");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                assert_eq!(t.data, vec![1.0, 0.0, 7.0, 0.0, 5.0, 8.0, 3.0, 0.0, 0.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        fs::remove_file(path).ok();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn csvread_errors_on_text() {
        let path = write_temp_file(&["1,2,3", "4,error,6"]);
        let err = csvread_builtin(Value::from(path.to_string_lossy().to_string()), Vec::new())
            .expect_err("should fail");
        let message = err.message().to_string();
        assert!(
            message.contains("nonnumeric token 'error'"),
            "unexpected error: {message}"
        );
        fs::remove_file(path).ok();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn csvread_accepts_char_array_filename() {
        let path = write_temp_file(&["1,2"]);
        let path_string = path.to_string_lossy().to_string();
        let data: Vec<char> = path_string.chars().collect();
        let cols = data.len();
        let chars = CharArray::new(data, 1, cols).expect("char array");
        let result = csvread_builtin(Value::CharArray(chars), Vec::new()).expect("csv");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.data, vec![1.0, 2.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        fs::remove_file(path).ok();
    }
}
