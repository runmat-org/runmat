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
use crate::gather_if_needed;

#[cfg(feature = "doc_export")]
#[runmat_macros::register_doc_text(name = "csvread")]
pub const DOC_MD: &str = r#"---
title: "csvread"
category: "io/tabular"
keywords: ["csvread", "csv", "comma-separated values", "numeric import", "range", "header"]
summary: "Read numeric data from a comma-separated text file with MATLAB-compatible zero-based ranges."
references:
  - https://www.mathworks.com/help/matlab/ref/csvread.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Performs host-side file I/O and parsing. Acceleration providers are not involved, and results remain on the CPU."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::tabular::csvread::tests"
  integration: "builtins::io::tabular::csvread::tests::csvread_basic_csv_roundtrip"
---

# What does the `csvread` function do in MATLAB / RunMat?
`csvread(filename)` reads numeric data from a comma-separated text file and returns a dense double-precision matrix. It is a legacy convenience wrapper preserved for MATLAB compatibility, and RunMat intentionally mirrors the original zero-based semantics.

## How does the `csvread` function behave in MATLAB / RunMat?
- Accepts character vectors or string scalars for the file name. String arrays must contain exactly one element.
- `csvread(filename, row, col)` starts reading at the zero-based row `row` and column `col`, skipping any data before that offset.
- `csvread(filename, row, col, range)` reads only the rectangle described by `range`. Numeric ranges must contain four elements `[r1 c1 r2 c2]` (zero-based, inclusive). Excel-style ranges use the familiar `"B2:D6"` A1 notation, which RunMat converts to zero-based indices internally.
- Empty fields (two consecutive commas or a trailing comma) are interpreted as `0`. Tokens such as `NaN`, `Inf`, and `-Inf` are accepted (case-insensitive).
- Any other nonnumeric token raises an error that identifies the offending row and column.
- Results are dense double-precision tensors using column-major layout. An empty file produces a `0×0` tensor.
- Paths can contain `~` to reference the home directory; RunMat expands the token before opening the file.

## `csvread` Function GPU Execution Behaviour
`csvread` performs all work on the host CPU. Arguments are gathered from the GPU when necessary, and the resulting tensor is returned in host memory. To keep data on the GPU, call `gpuArray` on the output or switch to `readmatrix` with the `'Like'` option. No provider hooks are required.

## Examples of using the `csvread` function in MATLAB / RunMat

### Import Entire CSV File
```matlab
writematrix([1 2 3; 4 5 6], "scores.csv");
M = csvread("scores.csv");
delete("scores.csv");
```
Expected output:
```matlab
M =
     1     2     3
     4     5     6
```

### Skip Header Row And Column Using Zero-Based Offsets
```matlab
fid = fopen("with_header.csv", "w");
fprintf(fid, "Name,Jan,Feb\nalpha,1,2\nbeta,3,4\n");
fclose(fid);

M = csvread("with_header.csv", 1, 1);
delete("with_header.csv");
```
Expected output:
```matlab
M =
     1     2
     3     4
```

### Read A Specific Range With Numeric Vector Syntax
```matlab
fid = fopen("measurements.csv", "w");
fprintf(fid, "10,11,12,13\n14,15,16,17\n18,19,20,21\n22,23,24,25\n");
fclose(fid);

block = csvread("measurements.csv", 0, 0, [1 1 2 3]);
delete("measurements.csv");
```
Expected output:
```matlab
block =
    15    16    17
    19    20    21
```

### Read A Block Using Excel-Style Range Notation
```matlab
fid = fopen("measurements2.csv", "w");
fprintf(fid, "10,11,12\n14,15,16\n18,19,20\n");
fclose(fid);

sub = csvread("measurements2.csv", 0, 0, "B2:C3");
delete("measurements2.csv");
```
Expected output:
```matlab
sub =
    15    16
    19    20
```

### Handle Empty Fields As Zeros
```matlab
fid = fopen("with_blanks.csv", "w");
fprintf(fid, "1,,3\n,5,\n7,8,\n");
fclose(fid);

M = csvread("with_blanks.csv");
delete("with_blanks.csv");
```
Expected output:
```matlab
M =
     1     0     3
     0     5     0
     7     8     0
```

### Read Numeric Data From A File In The Home Directory
```matlab
homePath = fullfile(getenv("HOME"), "runmat_csvread_home.csv");
fid = fopen(homePath, "w");
fprintf(fid, "9,10\n11,12\n");
fclose(fid);

M = csvread(fullfile("~", "runmat_csvread_home.csv"));
delete(homePath);
```
Expected output:
```matlab
M =
     9    10
    11    12
```

### Detect Errors When Text Appears In Numeric Columns
```matlab
fid = fopen("bad.csv", "w");
fprintf(fid, "1,2,3\n4,error,6\n");
fclose(fid);

try
    csvread("bad.csv");
catch err
    disp(err.message);
end
delete("bad.csv");
```
Expected output:
```matlab
csvread: nonnumeric token 'error' at row 2, column 2
```

## GPU residency in RunMat (Do I need `gpuArray`?)

`csvread` always returns a host-resident tensor because it performs file I/O and parsing on the CPU. If you need the data on the GPU, wrap the call with `gpuArray(csvread(...))` or switch to `readmatrix` with the `'Like'` option so that RunMat can place the result directly on the desired device.

## FAQ

### Why does `csvread` complain about text data?
`csvread` is limited to numeric CSV content. If a field contains letters, quoted strings, or other tokens that cannot be parsed as numbers, the builtin raises an error. Switch to `readmatrix` or `readtable` when the file mixes text and numbers.

### Are the row and column offsets zero-based?
Yes. `csvread(filename, row, col)` treats `row` and `col` as zero-based counts to skip from the start of the file before reading results.

### How are Excel-style ranges interpreted?
Excel ranges such as `"B2:D5"` use the familiar 1-based row numbering and column letters. The builtin converts them internally to zero-based indices and includes both endpoints.

### Can I read files with quoted numeric fields?
Quoted numeric fields are not supported. Remove quotes before calling `csvread`, or switch to `readmatrix`, which has full CSV parsing support.

### What happens to empty cells?
Empty cells (two consecutive commas or a trailing delimiter) become zero, matching MATLAB's `csvread` behaviour.

### Does `csvread` support custom delimiters?
No. `csvread` always uses comma separation. Use `dlmread` or `readmatrix` for other delimiters.

### How do I keep the results on the GPU?
`csvread` returns a host tensor. Call `gpuArray(csvread(...))` after reading, or prefer `readmatrix` with `'Like', gpuArray.zeros(1)` to keep residency on the GPU automatically.

### What if the file is empty?
An empty file results in a `0×0` double tensor. MATLAB behaves the same way.

### Does `csvread` change the working directory?
No. Relative paths are resolved against the current working directory and do not modify it.

## See Also
[readmatrix](./readmatrix), [writematrix](./writematrix), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `csvread` function is available at: [`crates/runmat-runtime/src/builtins/io/tabular/csvread.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/tabular/csvread.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec]
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

#[runmat_macros::register_fusion_spec]
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
    accel = "cpu"
)]
fn csvread_builtin(path: Value, rest: Vec<Value>) -> Result<Value, String> {
    let gathered_path = gather_if_needed(&path).map_err(|e| format!("csvread: {e}"))?;
    let options = parse_arguments(&rest)?;
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

fn parse_arguments(args: &[Value]) -> Result<CsvReadOptions, String> {
    let mut gathered = Vec::with_capacity(args.len());
    for value in args {
        gathered.push(gather_if_needed(value).map_err(|e| format!("csvread: {e}"))?);
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
        _ => Err("csvread: expected csvread(filename[, row, col[, range]])".to_string()),
    }
}

fn value_to_start_index(value: &Value, name: &str) -> Result<usize, String> {
    match value {
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < 0 {
                return Err(format!("csvread: {name} must be a non-negative integer"));
            }
            usize::try_from(raw).map_err(|_| format!("csvread: {name} is too large"))
        }
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(format!("csvread: {name} must be a finite integer"));
            }
            if *n < 0.0 {
                return Err(format!("csvread: {name} must be a non-negative integer"));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(format!("csvread: {name} must be an integer"));
            }
            usize::try_from(rounded as i64).map_err(|_| format!("csvread: {name} is too large"))
        }
        _ => Err(format!(
            "csvread: expected {name} as a numeric scalar, got {value:?}"
        )),
    }
}

fn resolve_path(value: &Value) -> Result<PathBuf, String> {
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
                Err("csvread: string array inputs must be scalar".to_string())
            }
        }
        Value::CharArray(_) => {
            Err("csvread: expected a 1-by-N character vector for the file name".to_string())
        }
        other => Err(format!(
            "csvread: expected filename as string scalar or character vector, got {other:?}"
        )),
    }
}

fn normalize_path(raw: &str) -> Result<PathBuf, String> {
    if raw.trim().is_empty() {
        return Err("csvread: filename must not be empty".to_string());
    }
    let expanded = expand_user_path(raw, "csvread").map_err(|e| format!("csvread: {e}"))?;
    Ok(Path::new(&expanded).to_path_buf())
}

fn read_csv_rows(path: &Path) -> Result<(Vec<Vec<f64>>, usize), String> {
    let file = File::open(path)
        .map_err(|e| format!("csvread: unable to open '{}': {e}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut buffer = String::new();
    let mut rows = Vec::new();
    let mut max_cols = 0usize;
    let mut line_index = 0usize;

    loop {
        buffer.clear();
        let bytes = reader
            .read_line(&mut buffer)
            .map_err(|e| format!("csvread: failed to read '{}': {}", path.display(), e))?;
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

fn parse_csv_row(line: &str, line_index: usize) -> Result<Vec<f64>, String> {
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
                format!(
                    "csvread: nonnumeric token '{}' at row {}, column {}",
                    unwrapped,
                    line_index,
                    col_index + 1
                )
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

fn parse_range(value: &Value) -> Result<RangeSpec, String> {
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
                Err("csvread: Range string array inputs must be scalar".to_string())
            }
        }
        Value::Tensor(_) => parse_range_numeric(value),
        _ => Err("csvread: Range must be provided as a string or numeric vector".to_string()),
    }
}

fn parse_range_string(text: &str) -> Result<RangeSpec, String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err("csvread: Range string cannot be empty".to_string());
    }
    let parts: Vec<&str> = trimmed.split(':').collect();
    if parts.len() > 2 {
        return Err(format!("csvread: invalid Range specification '{trimmed}'"));
    }
    let start = parse_cell_reference(parts[0])?;
    if start.col.is_none() {
        return Err("csvread: Range must specify a starting column".to_string());
    }
    let end = if parts.len() == 2 {
        Some(parse_cell_reference(parts[1])?)
    } else {
        None
    };
    if let Some(ref end_ref) = end {
        if end_ref.col.is_none() {
            return Err("csvread: Range end must include a column reference".to_string());
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

fn parse_range_numeric(value: &Value) -> Result<RangeSpec, String> {
    let elements = match value {
        Value::Tensor(t) => t.data.clone(),
        _ => {
            return Err(
                "csvread: numeric Range must be provided as a vector with 2 or 4 elements"
                    .to_string(),
            )
        }
    };
    if elements.len() != 2 && elements.len() != 4 {
        return Err("csvread: numeric Range must contain exactly 2 or 4 elements".to_string());
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

fn non_negative_index(value: f64, position: usize) -> Result<usize, String> {
    if !value.is_finite() {
        return Err("csvread: Range indices must be finite".to_string());
    }
    if value < 0.0 {
        return Err("csvread: Range indices must be non-negative".to_string());
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err("csvread: Range indices must be integers".to_string());
    }
    usize::try_from(rounded as i64).map_err(|_| {
        format!(
            "csvread: Range index {} is too large to fit in usize",
            position + 1
        )
    })
}

#[derive(Clone, Copy)]
struct CellReference {
    row: Option<usize>,
    col: Option<usize>,
}

fn parse_cell_reference(token: &str) -> Result<CellReference, String> {
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
            return Err(format!("csvread: invalid Range component '{token}'"));
        }
    }
    if letters.is_empty() && digits.is_empty() {
        return Err("csvread: Range references cannot be empty".to_string());
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
            format!(
                "csvread: invalid row index '{}' in Range component '{token}'",
                digits
            )
        })?;
        if parsed == 0 {
            return Err("csvread: Range rows must be >= 1".to_string());
        }
        Some(parsed - 1)
    };
    Ok(CellReference { row, col })
}

fn column_index_from_letters(letters: &str) -> Result<usize, String> {
    let mut value: usize = 0;
    for ch in letters.chars() {
        if !ch.is_ascii_uppercase() {
            return Err(format!(
                "csvread: invalid column designator '{letters}' in Range"
            ));
        }
        let digit = (ch as u8 - b'A' + 1) as usize;
        value = value
            .checked_mul(26)
            .and_then(|v| v.checked_add(digit))
            .ok_or_else(|| "csvread: Range column index overflowed".to_string())?;
    }
    value
        .checked_sub(1)
        .ok_or_else(|| "csvread: Range column index underflowed".to_string())
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
) -> Result<Tensor, String> {
    if row_count == 0 || col_count == 0 {
        return Tensor::new(Vec::new(), vec![0, 0]).map_err(|e| format!("csvread: {e}"));
    }
    let mut data = vec![default_fill; row_count * col_count];
    for (row_idx, row) in rows.iter().enumerate().take(row_count) {
        for col_idx in 0..col_count {
            let value = row.get(col_idx).copied().unwrap_or(default_fill);
            data[row_idx + col_idx * row_count] = value;
        }
    }
    Tensor::new(data, vec![row_count, col_count]).map_err(|e| format!("csvread: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use runmat_builtins::{CharArray, IntValue, Tensor as BuiltinTensor};

    use crate::builtins::common::test_support;

    static UNIQUE_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn unique_path(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
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

    #[test]
    fn csvread_errors_on_text() {
        let path = write_temp_file(&["1,2,3", "4,error,6"]);
        let err = csvread_builtin(Value::from(path.to_string_lossy().to_string()), Vec::new())
            .expect_err("should fail");
        assert!(
            err.contains("nonnumeric token 'error'"),
            "unexpected error: {err}"
        );
        fs::remove_file(path).ok();
    }

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

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
