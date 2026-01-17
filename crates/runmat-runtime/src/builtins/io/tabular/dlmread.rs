//! MATLAB-compatible `dlmread` builtin for RunMat.
//!
//! `dlmread` predates `readmatrix` but is still widely used in MATLAB
//! codebases for quick numeric imports with custom delimiters. This
//! implementation mirrors MATLAB's zero-based range semantics and
//! accepts the same mix of delimiter forms: characters, string scalars,
//! and numeric codes corresponding to ASCII delimiters.

use std::char;
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
use crate::{gather_if_needed, build_runtime_error, BuiltinResult, RuntimeControlFlow};

const BUILTIN_NAME: &str = "dlmread";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "dlmread",
        builtin_path = "crate::builtins::io::tabular::dlmread"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "dlmread"
category: "io/tabular"
keywords: ["dlmread", "delimiter", "numeric import", "ascii", "range"]
summary: "Read numeric data from a delimiter-separated text file with MATLAB-compatible zero-based offsets."
references:
  - https://www.mathworks.com/help/matlab/ref/dlmread.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "All parsing occurs on the CPU. Acceleration providers are not involved and the result remains in host memory."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::tabular::dlmread::tests"
  integration: "builtins::io::tabular::dlmread::tests::dlmread_semicolon_roundtrip"
---

# What does the `dlmread` function do in MATLAB / RunMat?
`dlmread(filename)` reads numeric data from an ASCII text file that uses a consistent delimiter between fields. Unlike `csvread`, `dlmread` lets you supply any single-character delimiter (including tabs or semicolons) or an ASCII code.

## How does the `dlmread` function behave in MATLAB / RunMat?
- Accepts character vectors or string scalars for `filename`. String arrays must contain exactly one element.
- The delimiter is optional; when omitted, a comma (`,`) is used. Supply a character (for example `';'`), a string scalar (`"\t"`), or a numeric ASCII code (`9` for tab). Empty delimiter inputs are not allowed.
- `dlmread(filename, R, C)` and `dlmread(filename, delimiter, R, C)` start reading at zero-based offsets `R` and `C`, skipping any earlier rows or columns.
- `dlmread(filename, range)` and `dlmread(filename, delimiter, range)` accept a numeric vector `[r1 c1 r2 c2]` or `[r1 c1]` describing a rectangular slice. The indices are zero-based and inclusive. Excel-style addresses such as `"B2:D6"` are also accepted for compatibility with `csvread`.
- Empty fields (for example two adjacent delimiters) are interpreted as `0`. Tokens such as `NaN`, `Inf`, and `-Inf` are accepted (case-insensitive).
- Any other nonnumeric token raises an error that identifies the offending row and column using one-based indices (matching MATLAB diagnostic messages).
- Results are dense double-precision tensors laid out in column-major order. Empty files yield a `0×0` tensor.
- Leading UTF-8 byte order marks (BOM) are stripped automatically to match MATLAB's handling of spreadsheets that emit BOM-prefixed text files.
- Paths may include `~` to reference the home directory; RunMat expands the token before opening the file.

## `dlmread` Function GPU Execution Behavior
`dlmread` performs file I/O and parsing on the CPU. Arguments are gathered from the GPU when needed, and the output tensor lives in host memory. Wrap the result in `gpuArray` if you need residency on a device. No provider hooks are involved.

## Examples of using the `dlmread` function in MATLAB / RunMat

### Reading comma-delimited data by default
```matlab
writematrix([1 2 3; 4 5 6], "samples.csv");
M = dlmread("samples.csv");
delete("samples.csv");
```
Expected output:
```matlab
M =
     1     2     3
     4     5     6
```

### Importing semicolon-separated values
```matlab
fid = fopen("scores.txt", "w");
fprintf(fid, "1;2;3\n4;5;6\n");
fclose(fid);

M = dlmread("scores.txt", ";");
delete("scores.txt");
```
Expected output:
```matlab
M =
     1     2     3
     4     5     6
```

### Using tab characters as the delimiter
```matlab
fid = fopen("tabs.txt", "w");
fprintf(fid, "10\t11\t12\n13\t14\t15\n");
fclose(fid);

M = dlmread("tabs.txt", "\t");
delete("tabs.txt");
```
Expected output:
```matlab
M =
    10    11    12
    13    14    15
```

### Skipping a header row and column (zero-based offsets)
```matlab
fid = fopen("with_header.txt", "w");
fprintf(fid, "Label,Jan,Feb\nalpha,1,2\nbeta,3,4\n");
fclose(fid);

M = dlmread("with_header.txt", ",", 1, 1);
delete("with_header.txt");
```
Expected output:
```matlab
M =
     1     2
     3     4
```

### Extracting a rectangular range
```matlab
fid = fopen("block.txt", "w");
fprintf(fid, "10,11,12,13\n14,15,16,17\n18,19,20,21\n");
fclose(fid);

sub = dlmread("block.txt", ",", [1 1 2 3]);
delete("block.txt");
```
Expected output:
```matlab
sub =
    15    16    17
    19    20    21
```

### Treating empty fields as zeros
```matlab
fid = fopen("blanks.txt", "w");
fprintf(fid, "1,,3\n,5,\n7,8,\n");
fclose(fid);

M = dlmread("blanks.txt");
delete("blanks.txt");
```
Expected output:
```matlab
M =
     1     0     3
     0     5     0
     7     8     0
```

### Using a numeric ASCII code for the delimiter
```matlab
fid = fopen("pipe.txt", "w");
fprintf(fid, "5|6|7\n8|9|10\n");
fclose(fid);

M = dlmread("pipe.txt", double('|'));
delete("pipe.txt");
```
Expected output:
```matlab
M =
     5     6     7
     8     9    10
```

## GPU residency in RunMat (Do I need `gpuArray`?)

`dlmread` always creates a CPU-resident tensor because the function performs file I/O synchronously on the host. If you need the data on the GPU, call `gpuArray(dlmread(...))` or switch to `readmatrix` with the `'Like'` option to direct the result to a device.

## FAQ

### Can I omit the delimiter argument?
Yes. When you omit the delimiter, `dlmread` assumes a comma. If you need another delimiter, pass it explicitly as a character, string scalar, or numeric ASCII code.

### Are row and column offsets zero-based like MATLAB?
Yes. The `R` and `C` arguments count from zero. `dlmread(filename, delimiter, 1, 2)` skips the first row and the first two columns before reading data.

### How do I specify a range?
Provide a numeric vector `[r1 c1 r2 c2]` (zero-based, inclusive) or an Excel-style address such as `"B2:D5"`. You can pass the range with or without a delimiter argument.

### What happens if the file contains text tokens?
Non-numeric fields trigger an error that includes the one-based row and column number of the offending token. Use `readmatrix` or `readtable` when you need to mix text and numbers.

### How does `dlmread` treat empty cells?
Empty cells evaluate to `0`, matching MATLAB behavior. This includes consecutive delimiters and trailing delimiters.

### Can I use whitespace as the delimiter?
Yes. Pass `" "` (space), `"\t"` (tab), or the corresponding ASCII code. Multiple consecutive delimiters produce zeros where the values are missing.

### Does `dlmread` respect locale-specific decimal separators?
No. Parsing always uses `.` as the decimal separator, consistent with MATLAB.

### Does `dlmread` change the working directory?
No. Relative paths are resolved against the current working directory. `dlmread` never mutates global process state.

### Why does the output stay on the CPU?
`dlmread` performs synchronous file I/O, so the result resides in host memory. Wrap the result with `gpuArray` if you want a device-resident tensor.

## See Also
[csvread](./csvread), [readmatrix](./readmatrix), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- The full source code for the implementation of the `dlmread` function is available at: [`crates/runmat-runtime/src/builtins/io/tabular/dlmread.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/tabular/dlmread.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::tabular::dlmread")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "dlmread",
    op_kind: GpuOpKind::Custom("io-dlmread"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Runs entirely on the host CPU; providers are not involved.",
};

fn dlmread_error(message: impl Into<String>) -> RuntimeControlFlow {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
        .into()
}

fn dlmread_error_with_source<E>(message: impl Into<String>, source: E) -> RuntimeControlFlow
where
    E: std::error::Error + Send + Sync + 'static,
{
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source)
        .build()
        .into()
}

fn map_control_flow(flow: RuntimeControlFlow) -> RuntimeControlFlow {
    match flow {
        RuntimeControlFlow::Suspend(pending) => RuntimeControlFlow::Suspend(pending),
        RuntimeControlFlow::Error(err) => {
            let identifier = err.identifier().map(|value| value.to_string());
            let message = err.message().to_string();
            let mut builder = build_runtime_error(message)
                .with_builtin(BUILTIN_NAME)
                .with_source(err);
            if let Some(identifier) = identifier {
                builder = builder.with_identifier(identifier);
            }
            builder.build().into()
        }
    }
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::tabular::dlmread")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "dlmread",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Standalone host operation; not eligible for fusion.",
};

#[runtime_builtin(
    name = "dlmread",
    category = "io/tabular",
    summary = "Read numeric data from a delimiter-separated text file.",
    keywords = "dlmread,delimiter,ascii import,range",
    accel = "cpu",
    builtin_path = "crate::builtins::io::tabular::dlmread"
)]
fn dlmread_builtin(path: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let gathered_path = gather_if_needed(&path).map_err(map_control_flow)?;
    let options = parse_arguments(&rest)?;
    let resolved = resolve_path(&gathered_path)?;
    let (rows, max_cols) = read_dlm_rows(&resolved, &options.delimiter)?;
    let subset = if let Some(range) = options.range {
        apply_range(&rows, max_cols, &range, 0.0)
    } else {
        apply_offsets(&rows, max_cols, options.start_row, options.start_col, 0.0)
    };
    let tensor = rows_to_tensor(subset.rows, subset.row_count, subset.col_count, 0.0)?;
    Ok(Value::Tensor(tensor))
}

#[derive(Clone, Debug)]
enum DelimiterSpec {
    Char(char),
    String(String),
}

impl DelimiterSpec {
    fn new_from_string(raw: &str) -> BuiltinResult<Self> {
        if raw.is_empty() {
            return Err(dlmread_error("dlmread: delimiter must not be empty"));
        }
        if raw == r"\t" {
            return Ok(DelimiterSpec::Char('\t'));
        }
        if raw == r"\n" {
            return Ok(DelimiterSpec::Char('\n'));
        }
        if raw == r"\r" {
            return Ok(DelimiterSpec::Char('\r'));
        }
        let mut chars = raw.chars();
        if let Some(first) = chars.next() {
            if chars.next().is_none() {
                return Ok(DelimiterSpec::Char(first));
            }
        }
        Ok(DelimiterSpec::String(raw.to_string()))
    }

    fn split<'a>(&self, line: &'a str) -> Vec<&'a str> {
        match self {
            DelimiterSpec::Char(ch) => line.split(*ch).collect(),
            DelimiterSpec::String(pattern) => line.split(pattern.as_str()).collect(),
        }
    }
}

#[derive(Debug)]
struct DlmReadOptions {
    delimiter: DelimiterSpec,
    start_row: usize,
    start_col: usize,
    range: Option<RangeSpec>,
}

impl Default for DlmReadOptions {
    fn default() -> Self {
        Self {
            delimiter: DelimiterSpec::Char(','),
            start_row: 0,
            start_col: 0,
            range: None,
        }
    }
}

fn parse_arguments(args: &[Value]) -> BuiltinResult<DlmReadOptions> {
    let mut gathered = Vec::with_capacity(args.len());
    for value in args {
        gathered.push(gather_if_needed(value).map_err(map_control_flow)?);
    }

    let mut options = DlmReadOptions::default();

    match gathered.len() {
        0 => Ok(options),
        1 => {
            if is_range_candidate(&gathered[0]) {
                options.range = Some(parse_range(&gathered[0])?);
            } else {
                options.delimiter = parse_delimiter(&gathered[0])?;
            }
            Ok(options)
        }
        2 => {
            if is_range_candidate(&gathered[1]) {
                options.delimiter = parse_delimiter(&gathered[0])?;
                options.range = Some(parse_range(&gathered[1])?);
            } else {
                options.start_row = value_to_start_index(&gathered[0], "row")?;
                options.start_col = value_to_start_index(&gathered[1], "col")?;
            }
            Ok(options)
        }
        3 => {
            if is_range_candidate(&gathered[2]) {
                options.start_row = value_to_start_index(&gathered[0], "row")?;
                options.start_col = value_to_start_index(&gathered[1], "col")?;
                options.range = Some(parse_range(&gathered[2])?);
            } else if is_delimiter_value(&gathered[0]) {
                options.delimiter = parse_delimiter(&gathered[0])?;
                options.start_row = value_to_start_index(&gathered[1], "row")?;
                options.start_col = value_to_start_index(&gathered[2], "col")?;
            } else {
                return Err(dlmread_error(
                    "dlmread: expected dlmread(filename[, delimiter][, row, col][, range])",
                ));
            }
            Ok(options)
        }
        4 => {
            if !is_range_candidate(&gathered[3]) {
                return Err(dlmread_error("dlmread: expected Range as final argument"));
            }
            options.delimiter = parse_delimiter(&gathered[0])?;
            options.start_row = value_to_start_index(&gathered[1], "row")?;
            options.start_col = value_to_start_index(&gathered[2], "col")?;
            options.range = Some(parse_range(&gathered[3])?);
            Ok(options)
        }
        _ => Err(dlmread_error(
            "dlmread: expected dlmread(filename[, delimiter][, row, col][, range])",
        )),
    }
}

fn is_delimiter_value(value: &Value) -> bool {
    match value {
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => true,
        Value::Int(_) | Value::Num(_) => true,
        Value::Tensor(t) => t.data.len() == 1,
        _ => false,
    }
}

fn is_range_candidate(value: &Value) -> bool {
    match value {
        Value::String(s) => looks_like_range_string(s),
        Value::CharArray(ca) => {
            if ca.rows != 1 {
                return false;
            }
            let text: String = ca.data.iter().collect();
            looks_like_range_string(&text)
        }
        Value::StringArray(sa) => {
            if sa.data.len() != 1 {
                return false;
            }
            looks_like_range_string(&sa.data[0])
        }
        Value::Tensor(t) => t.data.len() == 2 || t.data.len() == 4,
        _ => false,
    }
}

fn looks_like_range_string(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    trimmed.chars().any(|ch| ch.is_ascii_digit()) || trimmed.contains(':')
}

fn parse_delimiter(value: &Value) -> BuiltinResult<DelimiterSpec> {
    match value {
        Value::String(s) => DelimiterSpec::new_from_string(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            DelimiterSpec::new_from_string(&text)
        }
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                DelimiterSpec::new_from_string(&sa.data[0])
            } else {
                Err(dlmread_error(
                    "dlmread: string array delimiters must be scalar",
                ))
            }
        }
        Value::Int(i) => delimiter_from_ascii(i.to_i64()),
        Value::Num(n) => delimiter_from_numeric(*n),
        Value::Tensor(t) if t.data.len() == 1 => delimiter_from_numeric(t.data[0]),
        _ => Err(dlmread_error(format!(
            "dlmread: unsupported delimiter value {value:?}"
        ))),
    }
}

fn delimiter_from_numeric(value: f64) -> BuiltinResult<DelimiterSpec> {
    if !value.is_finite() {
        return Err(dlmread_error("dlmread: delimiter code must be finite"));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(dlmread_error(
            "dlmread: delimiter code must be an integer",
        ));
    }
    delimiter_from_ascii(rounded as i64)
}

fn delimiter_from_ascii(value: i64) -> BuiltinResult<DelimiterSpec> {
    if value < 0 || value > char::MAX as i64 {
        return Err(dlmread_error(
            "dlmread: delimiter code must be within Unicode range",
        ));
    }
    let ch = char::from_u32(value as u32).ok_or_else(|| {
        dlmread_error("dlmread: delimiter code does not map to a Unicode scalar")
    })?;
    Ok(DelimiterSpec::Char(ch))
}

fn value_to_start_index(value: &Value, name: &str) -> BuiltinResult<usize> {
    match value {
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < 0 {
                return Err(dlmread_error(format!(
                    "dlmread: {name} must be a non-negative integer"
                )));
            }
            usize::try_from(raw)
                .map_err(|_| dlmread_error(format!("dlmread: {name} is too large")))
        }
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(dlmread_error(format!("dlmread: {name} must be finite")));
            }
            if *n < 0.0 {
                return Err(dlmread_error(format!(
                    "dlmread: {name} must be a non-negative integer"
                )));
            }
            let rounded = n.round();
            if (rounded - n).abs() > f64::EPSILON {
                return Err(dlmread_error(format!("dlmread: {name} must be an integer")));
            }
            usize::try_from(rounded as i64)
                .map_err(|_| dlmread_error(format!("dlmread: {name} is too large")))
        }
        Value::Tensor(t) if t.data.len() == 1 => value_to_start_index(&Value::Num(t.data[0]), name),
        _ => Err(dlmread_error(format!(
            "dlmread: expected numeric scalar for {name}, got {value:?}"
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
                Err(dlmread_error(
                    "dlmread: string array filename inputs must be scalar",
                ))
            }
        }
        Value::CharArray(_) => Err(dlmread_error(
            "dlmread: expected a 1-by-N character vector for the file name",
        )),
        other => Err(dlmread_error(format!(
            "dlmread: expected filename as string scalar or character vector, got {other:?}"
        ))),
    }
}

fn normalize_path(raw: &str) -> BuiltinResult<PathBuf> {
    if raw.trim().is_empty() {
        return Err(dlmread_error("dlmread: filename must not be empty"));
    }
    let expanded = expand_user_path(raw, BUILTIN_NAME).map_err(dlmread_error)?;
    Ok(Path::new(&expanded).to_path_buf())
}

fn read_dlm_rows(
    path: &Path,
    delimiter: &DelimiterSpec,
) -> BuiltinResult<(Vec<Vec<f64>>, usize)> {
    let file = File::open(path).map_err(|err| {
        dlmread_error_with_source(
            format!("dlmread: unable to open '{}': {err}", path.display()),
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
            dlmread_error_with_source(
                format!("dlmread: failed to read '{}': {err}", path.display()),
                err,
            )
        })?;
        if bytes == 0 {
            break;
        }
        // Trim standard newline endings.
        if buffer.ends_with('\n') {
            buffer.pop();
            if buffer.ends_with('\r') {
                buffer.pop();
            }
        } else if buffer.ends_with('\r') {
            buffer.pop();
        }
        let mut view: &str = &buffer;
        if line_index == 0 && view.starts_with('\u{FEFF}') {
            view = &view['\u{FEFF}'.len_utf8()..];
        }
        let parsed = parse_dlm_row(view, delimiter, line_index)?;
        max_cols = max_cols.max(parsed.len());
        rows.push(parsed);
        line_index += 1;
    }

    Ok((rows, max_cols))
}

fn parse_dlm_row(
    line: &str,
    delimiter: &DelimiterSpec,
    line_index: usize,
) -> BuiltinResult<Vec<f64>> {
    let mut values = Vec::new();
    let tokens = delimiter.split(line);
    for (col_index, raw_field) in tokens.into_iter().enumerate() {
        let trimmed = raw_field.trim();
        if trimmed.is_empty() {
            values.push(0.0);
            continue;
        }
        let lowered = trimmed.to_ascii_lowercase();
        let value = match lowered.as_str() {
            "nan" => f64::NAN,
            "inf" | "+inf" => f64::INFINITY,
            "-inf" => f64::NEG_INFINITY,
            _ => trimmed.parse::<f64>().map_err(|_| {
                dlmread_error(format!(
                    "dlmread: nonnumeric token '{}' at row {}, column {}",
                    trimmed,
                    line_index + 1,
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

fn validate_range(spec: RangeSpec) -> BuiltinResult<RangeSpec> {
    if let Some(end_row) = spec.end_row {
        if end_row < spec.start_row {
            return Err(dlmread_error(
                "dlmread: Range must satisfy R1 <= R2 and C1 <= C2",
            ));
        }
    }
    if let Some(end_col) = spec.end_col {
        if end_col < spec.start_col {
            return Err(dlmread_error(
                "dlmread: Range must satisfy R1 <= R2 and C1 <= C2",
            ));
        }
    }
    Ok(spec)
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
                Err(dlmread_error(
                    "dlmread: Range string array inputs must be scalar",
                ))
            }
        }
        Value::Tensor(_) => parse_range_numeric(value),
        _ => Err(dlmread_error(
            "dlmread: Range must be provided as a string or numeric vector",
        )),
    }
}

fn parse_range_string(text: &str) -> BuiltinResult<RangeSpec> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(dlmread_error("dlmread: Range string cannot be empty"));
    }
    let parts: Vec<&str> = trimmed.split(':').collect();
    if parts.len() > 2 {
        return Err(dlmread_error(format!(
            "dlmread: invalid Range specification '{trimmed}'"
        )));
    }
    let start = parse_cell_reference(parts[0])?;
    if start.col.is_none() {
        return Err(dlmread_error(
            "dlmread: Range must specify a starting column",
        ));
    }
    let end = if parts.len() == 2 {
        Some(parse_cell_reference(parts[1])?)
    } else {
        None
    };
    if let Some(ref end_ref) = end {
        if end_ref.col.is_none() {
            return Err(dlmread_error(
                "dlmread: Range end must include a column reference",
            ));
        }
    }
    let start_row = start.row.unwrap_or(0);
    let start_col = start.col.unwrap();
    let end_row = end.as_ref().and_then(|r| r.row);
    let end_col = end.as_ref().and_then(|r| r.col);
    let spec = RangeSpec {
        start_row,
        start_col,
        end_row,
        end_col,
    };
    validate_range(spec)
}

fn parse_range_numeric(value: &Value) -> BuiltinResult<RangeSpec> {
    let elements = match value {
        Value::Tensor(t) => t.data.clone(),
        _ => {
            return Err(dlmread_error(
                "dlmread: numeric Range must be provided as a vector with 2 or 4 elements",
            ))
        }
    };
    if elements.len() != 2 && elements.len() != 4 {
        return Err(dlmread_error(
            "dlmread: numeric Range must contain exactly 2 or 4 elements",
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
    let spec = RangeSpec {
        start_row,
        start_col,
        end_row,
        end_col,
    };
    validate_range(spec)
}

fn non_negative_index(value: f64, position: usize) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(dlmread_error("dlmread: Range indices must be finite"));
    }
    if value < 0.0 {
        return Err(dlmread_error(
            "dlmread: Range indices must be non-negative",
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(dlmread_error("dlmread: Range indices must be integers"));
    }
    usize::try_from(rounded as i64).map_err(|_| {
        dlmread_error(format!(
            "dlmread: Range index {} is too large to fit in usize",
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
            return Err(dlmread_error(format!(
                "dlmread: invalid Range component '{token}'"
            )));
        }
    }
    if letters.is_empty() && digits.is_empty() {
        return Err(dlmread_error("dlmread: Range references cannot be empty"));
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
            dlmread_error(format!(
                "dlmread: invalid row index '{}' in Range component '{token}'",
                digits
            ))
        })?;
        if parsed == 0 {
            return Err(dlmread_error("dlmread: Range rows must be >= 1"));
        }
        Some(parsed - 1)
    };
    Ok(CellReference { row, col })
}

fn column_index_from_letters(letters: &str) -> BuiltinResult<usize> {
    let mut value: usize = 0;
    for ch in letters.chars() {
        if !ch.is_ascii_uppercase() {
            return Err(dlmread_error(format!(
                "dlmread: invalid column designator '{letters}' in Range"
            )));
        }
        let digit = (ch as u8 - b'A' + 1) as usize;
        value = value
            .checked_mul(26)
            .and_then(|v| v.checked_add(digit))
            .ok_or_else(|| dlmread_error("dlmread: Range column index overflowed"))?;
    }
    value
        .checked_sub(1)
        .ok_or_else(|| dlmread_error("dlmread: Range column index underflowed"))
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
    if start_row >= rows.len() || start_col >= max_cols {
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
    let row_span = end_row.saturating_sub(range.start_row).saturating_add(1);
    for row in rows.iter().skip(range.start_row).take(row_span) {
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
            .map_err(|e| dlmread_error(format!("dlmread: {e}")));
    }
    let mut data = vec![default_fill; row_count * col_count];
    for (row_idx, row) in rows.iter().enumerate().take(row_count) {
        for col_idx in 0..col_count {
            let value = row.get(col_idx).copied().unwrap_or(default_fill);
            data[row_idx + col_idx * row_count] = value;
        }
    }
    Tensor::new(data, vec![row_count, col_count])
        .map_err(|e| dlmread_error(format!("dlmread: {e}")))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_time::unix_timestamp_ns;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use runmat_builtins::{CharArray, IntValue, Tensor as BuiltinTensor};

    use crate::builtins::common::test_support;

    static UNIQUE_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn unique_path(prefix: &str) -> PathBuf {
        let nanos = unix_timestamp_ns();
        let seq = UNIQUE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "runmat_dlmread_{prefix}_{}_{}_{}",
            std::process::id(),
            nanos,
            seq
        ));
        path
    }

    fn write_temp_file(lines: &[&str]) -> PathBuf {
        let path = unique_path("input").with_extension("txt");
        let contents = lines.join("\n");
        fs::write(&path, contents).expect("write temp file");
        path
    }

    fn write_temp_file_bytes(bytes: &[u8]) -> PathBuf {
        let path = unique_path("input_bytes").with_extension("txt");
        fs::write(&path, bytes).expect("write temp file bytes");
        path
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmread_default_delimiter() {
        let path = write_temp_file(&["1,2,3", "4,5,6"]);
        let result = dlmread_builtin(Value::from(path.to_string_lossy().to_string()), Vec::new())
            .expect("dlmread");
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
    fn dlmread_semicolon_roundtrip() {
        let path = write_temp_file(&["1;2;3", "4;5;6"]);
        let args = vec![Value::from(";")];
        let result = dlmread_builtin(Value::from(path.to_string_lossy().to_string()), args)
            .expect("dlmread");
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
    fn dlmread_ascii_code_delimiter() {
        let path = write_temp_file(&["5|6|7", "8|9|10"]);
        let args = vec![Value::Int(IntValue::I32('|' as i32))];
        let result = dlmread_builtin(Value::from(path.to_string_lossy().to_string()), args)
            .expect("dlmread");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3]);
                assert_eq!(t.data, vec![5.0, 8.0, 6.0, 9.0, 7.0, 10.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        fs::remove_file(path).ok();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmread_char_array_filename() {
        let path = write_temp_file(&["1,2", "3,4"]);
        let path_string = path.to_string_lossy().to_string();
        let chars: Vec<char> = path_string.chars().collect();
        let char_array = CharArray::new(chars, 1, path_string.chars().count()).expect("char array");
        let result = dlmread_builtin(Value::CharArray(char_array), Vec::new()).expect("dlmread");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![1.0, 3.0, 2.0, 4.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        fs::remove_file(path).ok();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmread_handles_utf8_bom() {
        let bytes = b"\xEF\xBB\xBF1,2\n3,4\n";
        let path = write_temp_file_bytes(bytes);
        let result = dlmread_builtin(Value::from(path.to_string_lossy().to_string()), Vec::new())
            .expect("dlmread");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![1.0, 3.0, 2.0, 4.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        fs::remove_file(path).ok();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmread_empty_file_returns_empty_tensor() {
        let path = write_temp_file(&[]);
        let result = dlmread_builtin(Value::from(path.to_string_lossy().to_string()), Vec::new())
            .expect("dlmread");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        fs::remove_file(path).ok();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmread_with_offsets() {
        let path = write_temp_file(&["0,1,2", "3,4,5", "6,7,8"]);
        let args = vec![Value::Int(IntValue::I32(1)), Value::Int(IntValue::I32(1))];
        let result = dlmread_builtin(Value::from(path.to_string_lossy().to_string()), args)
            .expect("dlmread");
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
    fn dlmread_with_numeric_range() {
        let path = write_temp_file(&["1,2,3", "4,5,6", "7,8,9"]);
        let range = BuiltinTensor::new(vec![1.0, 1.0, 2.0, 2.0], vec![4, 1]).expect("tensor");
        let args = vec![Value::from(","), Value::Tensor(range)];
        let result = dlmread_builtin(Value::from(path.to_string_lossy().to_string()), args)
            .expect("dlmread");
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
    fn dlmread_numeric_range_two_elements() {
        let path = write_temp_file(&["1,2,3", "4,5,6", "7,8,9"]);
        let range = BuiltinTensor::new(vec![1.0, 1.0], vec![2, 1]).expect("tensor");
        let args = vec![Value::Tensor(range)];
        let result = dlmread_builtin(Value::from(path.to_string_lossy().to_string()), args)
            .expect("dlmread");
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
    fn dlmread_excel_style_range_string() {
        let path = write_temp_file(&["1,2,3,4", "5,6,7,8", "9,10,11,12"]);
        let args = vec![Value::from(","), Value::from("B2:C3")];
        let result = dlmread_builtin(Value::from(path.to_string_lossy().to_string()), args)
            .expect("dlmread");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![6.0, 10.0, 7.0, 11.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        fs::remove_file(path).ok();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmread_range_without_delimiter() {
        let path = write_temp_file(&["1,2,3", "4,5,6", "7,8,9"]);
        let range = BuiltinTensor::new(vec![1.0, 0.0, 2.0, 1.0], vec![4, 1]).expect("tensor");
        let args = vec![Value::Tensor(range)];
        let result = dlmread_builtin(Value::from(path.to_string_lossy().to_string()), args)
            .expect("dlmread");
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
    fn dlmread_nonnumeric_token_error() {
        let path = write_temp_file(&["1,foo"]);
        let err = dlmread_builtin(Value::from(path.to_string_lossy().to_string()), Vec::new())
            .expect_err("dlmread should fail");
        let message = match err {
            RuntimeControlFlow::Error(err) => err.message().to_string(),
            RuntimeControlFlow::Suspend(_) => panic!("unexpected suspend from dlmread"),
        };
        assert!(
            message.contains("nonnumeric token 'foo' at row 1, column 2"),
            "unexpected error message: {message}"
        );
        fs::remove_file(path).ok();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmread_invalid_range_error() {
        let path = write_temp_file(&["1,2,3", "4,5,6"]);
        let range = BuiltinTensor::new(vec![2.0, 1.0, 1.0, 3.0], vec![4, 1]).expect("tensor");
        let args = vec![Value::from(","), Value::Tensor(range)];
        let err = dlmread_builtin(Value::from(path.to_string_lossy().to_string()), args)
            .expect_err("dlmread should fail");
        let message = match err {
            RuntimeControlFlow::Error(err) => err.message().to_string(),
            RuntimeControlFlow::Suspend(_) => panic!("unexpected suspend from dlmread"),
        };
        assert!(
            message.contains("Range must satisfy R1 <= R2 and C1 <= C2"),
            "unexpected error message: {message}"
        );
        fs::remove_file(path).ok();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmread_space_delimiter() {
        let path = write_temp_file(&["1  3", " 4 5 ", "6 7  "]);
        let args = vec![Value::from(" ")];
        let result = dlmread_builtin(Value::from(path.to_string_lossy().to_string()), args)
            .expect("dlmread");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 4]);
                assert_eq!(
                    t.data,
                    vec![
                        1.0, 0.0, 6.0, // column 0
                        0.0, 4.0, 7.0, // column 1
                        3.0, 5.0, 0.0, // column 2
                        0.0, 0.0, 0.0, // column 3 (trailing blanks)
                    ]
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        fs::remove_file(path).ok();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
