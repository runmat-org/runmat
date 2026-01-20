//! MATLAB-compatible `writematrix` builtin for emitting delimited text files.

use std::io::Write;
use std::path::{Path, PathBuf};

use runmat_builtins::{Tensor, Value};
use runmat_filesystem::OpenOptions;
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "writematrix";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "writematrix",
        builtin_path = "crate::builtins::io::tabular::writematrix"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "writematrix"
category: "io/tabular"
keywords: ["writematrix", "csv", "delimited text", "write", "append", "quote strings", "decimal separator"]
summary: "Write numeric or string matrices to delimited text files with MATLAB-compatible defaults."
references:
  - https://www.mathworks.com/help/matlab/ref/writematrix.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs entirely on the CPU. gpuArray inputs are gathered automatically before serialisation."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::tabular::writematrix::tests"
  integration:
    - "builtins::io::tabular::writematrix::tests::writematrix_writes_space_delimited_txt"
    - "builtins::io::tabular::writematrix::tests::writematrix_defaults_to_comma_for_csv"
    - "builtins::io::tabular::writematrix::tests::writematrix_honours_write_mode_append"
    - "builtins::io::tabular::writematrix::tests::writematrix_quotes_strings_by_default"
    - "builtins::io::tabular::writematrix::tests::writematrix_accepts_gpu_tensor_inputs"
---

# What does the `writematrix` function do in MATLAB / RunMat?
`writematrix(A, filename)` serialises numeric or string matrices to a text file using
MATLAB-compatible defaults. Column-major ordering is preserved, delimiters adjust to the
filename extension, and optional name/value pairs customise quoting, decimal separators,
line endings, and write mode. The builtin mirrors MATLAB behaviour for the supported options,
raising descriptive errors for unsupported combinations like spreadsheet output.

## How does the `writematrix` function behave in MATLAB / RunMat?
- Accepts real numeric, logical, or string arrays with up to two dimensions (trailing singleton
  dimensions are ignored). Heterogeneous data requires `writecell` instead.
- Default delimiters follow MATLAB: comma for `.csv`, tab for `.tsv`, whitespace for `.txt` and
  `.dat`, and comma for other extensions. Specify `'Delimiter', value` to override.
- `'WriteMode'` supports `'overwrite'` (default) and `'append'`. Overwrite truncates existing
  files; append writes new rows to the end without inserting extra delimiters.
- `'QuoteStrings'` controls whether text fields are wrapped in double quotes. When enabled (the
  default) embedded quotes are doubled (`"Alice"` becomes `""Alice""`).
- `'DecimalSeparator'` swaps the decimal point for locales that require `','` or other
  characters. Thousands separators are not inserted automatically, matching MATLAB.
- `'LineEnding'` supports `'auto'` (normalized `\n` for cross-platform portability), `'unix'`,
  `'pc'`/`'windows'`, and `'mac'`.
- `'FileType'` accepts `'delimitedtext'` or `'text'`. Spreadsheet output (`'spreadsheet'`)
  is not yet available and triggers a descriptive error consistent with MATLAB's messaging.
- Unsupported options are ignored for forward compatibility. Errors indicate which argument was
  invalid when the value cannot be interpreted.

## `writematrix` Function GPU Execution Behaviour
`writematrix` always executes on the host. When the input matrix resides on a GPU, RunMat gathers
the array via the active acceleration provider before formatting. No provider-specific hooks or
GPU kernels are required, and the result of the write remains a host-side side effect (file on
disk). If no provider is registered, the builtin emits the same gather error reported by other
residency sinks.

## Examples of using the `writematrix` function in MATLAB / RunMat

### Save a numeric matrix to a CSV file
```matlab
A = [1 2 3; 4 5 6];
writematrix(A, "results.csv");
```
Expected contents of `results.csv`:
```matlab
1,2,3
4,5,6
```

### Export data with a custom semicolon delimiter
```matlab
writematrix(A, "results.dat", 'Delimiter', ';');
```
Expected contents of `results.dat`:
```matlab
1;2;3
4;5;6
```

### Append additional rows to an existing report
```matlab
writematrix([7 8 9], "report.txt");
writematrix([10 11 12], "report.txt", 'WriteMode', 'append');
```
Expected contents of `report.txt`:
```matlab
7 8 9
10 11 12
```

### Write string data with quoting disabled
```matlab
names = ["Alice" "Bob" "Charlie"];
writematrix(names, "names.csv", 'QuoteStrings', false);
```
Expected contents of `names.csv`:
```matlab
Alice,Bob,Charlie
```

### Use a European decimal separator and explicit line ending
```matlab
vals = [12.34; 56.78];
writematrix(vals, "eu.csv", 'DecimalSeparator', ',', 'LineEnding', 'unix');
```
Expected contents of `eu.csv`:
```matlab
12,34
56,78
```

### Write GPU-resident data transparently
```matlab
G = gpuArray(rand(2, 3));
writematrix(G, "gpu_output.csv");
```
Expected behaviour:
```matlab
% Data is gathered from the GPU automatically and written to disk as CSV.
```

## GPU residency in RunMat (Do I need `gpuArray`?)
No additional steps are required. `writematrix` treats GPU arrays as residency sinks: the data is
gathered before formatting, ensuring the file on disk reflects host memory. This mirrors MATLAB,
where `writematrix` operates on numeric values regardless of their original residency.

## FAQ

### Which data types does `writematrix` support?
Real numeric, logical, and string arrays up to two dimensions. For heterogenous content (mixed
numbers and text) use `writecell` or `writetable`.

### How are empty matrices handled?
Zero-row or zero-column inputs produce either an empty file or blank line endings per MATLAB's
behaviour. The file is still created or truncated according to `'WriteMode'`.

### Can I write OTA spreadsheet formats like `.xlsx`?
Not yet. Passing `'FileType','spreadsheet'` raises a descriptive error. Use MATLAB's table-based
workflows or export delimited text instead.

### How are embedded quotes in text fields escaped?
When `'QuoteStrings'` is `true`, embedded double quotes are doubled (`"She said ""hi"""`),
conforming to RFC 4180 and MATLAB's implementation. With quoting disabled, the characters are
written verbatim.

### What happens if I choose `','` as both delimiter and decimal separator?
RunMat mirrors MATLAB by honouring your request without modification, even though the output may
be ambiguous. Choose a different delimiter when writing locale-specific decimals.

### Does `writematrix` change the working directory?
No. Relative paths are resolved against the current MATLAB working directory, and only the target
file is touched.

### How do I include header rows or variable names?
`writematrix` focuses on numeric/string matrices. For labelled data use `writetable` (planned) or
prepend header lines manually with `fprintf` before calling `writematrix` in `'append'` mode.

## See Also
[readmatrix](./readmatrix), [writecell](./writecell), [fprintf](./fprintf), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- The full source code for the implementation of the `writematrix` function is available at: [`crates/runmat-runtime/src/builtins/io/tabular/writematrix.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/tabular/writematrix.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::tabular::writematrix")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "writematrix",
    op_kind: GpuOpKind::Custom("io-writematrix"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Runs entirely on the host; gpuArray inputs are gathered before serialisation.",
};

fn writematrix_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn writematrix_error_with_source<E>(message: impl Into<String>, source: E) -> RuntimeError
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::tabular::writematrix")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "writematrix",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion; performs host-side file I/O.",
};

#[runtime_builtin(
    name = "writematrix",
    category = "io/tabular",
    summary = "Write numeric or string matrices to delimited text files with MATLAB-compatible defaults.",
    keywords = "writematrix,csv,delimited text,write,append,quote strings",
    accel = "cpu",
    builtin_path = "crate::builtins::io::tabular::writematrix"
)]
fn writematrix_builtin(data: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.is_empty() {
        return Err(writematrix_error("writematrix: filename is required"));
    }

    let filename_value = gather_if_needed(&rest[0]).map_err(map_control_flow)?;
    let path = resolve_path(&filename_value)?;

    let options = parse_options(&rest[1..])?;

    let gathered = gather_if_needed(&data).map_err(map_control_flow)?;
    let matrix = MatrixData::from_value(gathered)?;

    let bytes_written = write_matrix(&path, &matrix, &options)?;

    Ok(Value::Num(bytes_written as f64))
}

#[derive(Debug, Clone)]
struct WriteMatrixOptions {
    delimiter: Option<String>,
    write_mode: WriteMode,
    quote_strings: bool,
    line_ending: LineEnding,
    decimal_separator: char,
    file_type: FileType,
}

impl Default for WriteMatrixOptions {
    fn default() -> Self {
        Self {
            delimiter: None,
            write_mode: WriteMode::Overwrite,
            quote_strings: true,
            line_ending: LineEnding::Auto,
            decimal_separator: '.',
            file_type: FileType::DelimitedText,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriteMode {
    Overwrite,
    Append,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LineEnding {
    Auto,
    Unix,
    Windows,
    Mac,
}

impl LineEnding {
    fn as_str(&self) -> &'static str {
        match self {
            LineEnding::Auto | LineEnding::Unix => "\n",
            LineEnding::Windows => "\r\n",
            LineEnding::Mac => "\r",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileType {
    DelimitedText,
    Text,
}

fn parse_options(args: &[Value]) -> BuiltinResult<WriteMatrixOptions> {
    if args.is_empty() {
        return Ok(WriteMatrixOptions::default());
    }
    if !args.len().is_multiple_of(2) {
        return Err(writematrix_error(
            "writematrix: name/value inputs must appear in pairs",
        ));
    }

    let mut options = WriteMatrixOptions::default();
    let mut index = 0usize;
    while index < args.len() {
        let name_value = gather_if_needed(&args[index]).map_err(map_control_flow)?;
        let name = option_name_from_value(&name_value)?;
        let value = gather_if_needed(&args[index + 1]).map_err(map_control_flow)?;
        apply_option(&mut options, &name, &value)?;
        index += 2;
    }
    Ok(options)
}

fn apply_option(options: &mut WriteMatrixOptions, name: &str, value: &Value) -> BuiltinResult<()> {
    if name.eq_ignore_ascii_case("Delimiter") {
        let delimiter = parse_delimiter(value)?;
        options.delimiter = Some(delimiter);
        return Ok(());
    }
    if name.eq_ignore_ascii_case("WriteMode") {
        options.write_mode = parse_write_mode(value)?;
        return Ok(());
    }
    if name.eq_ignore_ascii_case("QuoteStrings") {
        options.quote_strings = parse_bool_like(value, "QuoteStrings")?;
        return Ok(());
    }
    if name.eq_ignore_ascii_case("DecimalSeparator") {
        options.decimal_separator = parse_decimal_separator(value)?;
        return Ok(());
    }
    if name.eq_ignore_ascii_case("LineEnding") {
        options.line_ending = parse_line_ending(value)?;
        return Ok(());
    }
    if name.eq_ignore_ascii_case("FileType") {
        options.file_type = parse_file_type(value)?;
        return Ok(());
    }
    // Unsupported or future options are ignored for compatibility with MATLAB's permissive behaviour.
    Ok(())
}

fn option_name_from_value(value: &Value) -> BuiltinResult<String> {
    value_to_string_scalar(value, "option name")
}

fn parse_delimiter(value: &Value) -> BuiltinResult<String> {
    let text = value_to_string_scalar(value, "Delimiter")?;
    if text.is_empty() {
        return Err(writematrix_error("writematrix: Delimiter cannot be empty"));
    }
    let trimmed = text.trim();
    let lowered = trimmed.to_ascii_lowercase();
    match lowered.as_str() {
        "tab" => Ok("\t".to_string()),
        "space" | "whitespace" => Ok(" ".to_string()),
        "comma" => Ok(",".to_string()),
        "semicolon" => Ok(";".to_string()),
        "pipe" => Ok("|".to_string()),
        _ => Ok(trimmed.to_string()),
    }
}

fn parse_write_mode(value: &Value) -> BuiltinResult<WriteMode> {
    let text = value_to_string_scalar(value, "WriteMode")?;
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "overwrite" => Ok(WriteMode::Overwrite),
        "append" => Ok(WriteMode::Append),
        _ => Err(writematrix_error(
            "writematrix: WriteMode must be 'overwrite' or 'append'",
        )),
    }
}

fn parse_bool_like(value: &Value, context: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::Int(i) => {
            let raw = i.to_i64();
            match raw {
                0 => Ok(false),
                1 => Ok(true),
                _ => Err(writematrix_error(format!(
                    "writematrix: {context} must be logical (0 or 1)"
                ))),
            }
        }
        Value::Num(n) => {
            if (*n - 0.0).abs() < f64::EPSILON {
                Ok(false)
            } else if (*n - 1.0).abs() < f64::EPSILON {
                Ok(true)
            } else {
                Err(writematrix_error(format!(
                    "writematrix: {context} must be logical (0 or 1)"
                )))
            }
        }
        _ => {
            let text = value_to_string_scalar(value, context)?;
            let lowered = text.trim().to_ascii_lowercase();
            match lowered.as_str() {
                "on" | "true" | "yes" | "1" => Ok(true),
                "off" | "false" | "no" | "0" => Ok(false),
                _ => Err(writematrix_error(format!(
                    "writematrix: {context} must be logical (true/on or false/off)"
                ))),
            }
        }
    }
}

fn parse_decimal_separator(value: &Value) -> BuiltinResult<char> {
    let text = value_to_string_scalar(value, "DecimalSeparator")?;
    let mut chars = text.chars();
    let ch = chars.next().ok_or_else(|| {
        writematrix_error("writematrix: DecimalSeparator must be a single character")
    })?;
    if chars.next().is_some() {
        return Err(writematrix_error(
            "writematrix: DecimalSeparator must be a single character",
        ));
    }
    if ch == '\n' || ch == '\r' {
        return Err(writematrix_error(
            "writematrix: DecimalSeparator cannot be a newline character",
        ));
    }
    Ok(ch)
}

fn parse_line_ending(value: &Value) -> BuiltinResult<LineEnding> {
    let text = value_to_string_scalar(value, "LineEnding")?;
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "auto" => Ok(LineEnding::Auto),
        "unix" => Ok(LineEnding::Unix),
        "pc" | "windows" => Ok(LineEnding::Windows),
        "mac" => Ok(LineEnding::Mac),
        _ => Err(writematrix_error(
            "writematrix: LineEnding must be 'auto', 'unix', 'pc', or 'mac'",
        )),
    }
}

fn parse_file_type(value: &Value) -> BuiltinResult<FileType> {
    let text = value_to_string_scalar(value, "FileType")?;
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "delimitedtext" | "text" => {
            if lowered == "text" {
                Ok(FileType::Text)
            } else {
                Ok(FileType::DelimitedText)
            }
        }
        "spreadsheet" => Err(writematrix_error(
            "writematrix: FileType 'spreadsheet' is not supported; export delimited text instead",
        )),
        _ => Err(writematrix_error("writematrix: unsupported FileType")),
    }
}

fn value_to_string_scalar(value: &Value, context: &str) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(writematrix_error(format!(
            "writematrix: expected {context} as a string scalar or character vector"
        ))),
    }
}

enum MatrixData {
    Numeric(Tensor),
    Text {
        rows: usize,
        cols: usize,
        data: Vec<String>,
    },
}

impl MatrixData {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::StringArray(sa) => {
                ensure_matrix_shape(&sa.shape, "string")?;
                let (rows, cols) = if sa.shape.is_empty() {
                    (1, 1)
                } else {
                    (sa.rows, sa.cols)
                };
                Ok(MatrixData::Text {
                    rows,
                    cols,
                    data: sa.data.clone(),
                })
            }
            Value::String(s) => Ok(MatrixData::Text {
                rows: 1,
                cols: 1,
                data: vec![s],
            }),
            Value::CharArray(_) => Err(writematrix_error(
                "writematrix: character arrays are not supported; convert to string arrays or use writecell",
            )),
            Value::Cell(_) => Err(writematrix_error(
                "writematrix: cell arrays are not supported; use writecell for heterogeneous content",
            )),
            Value::ComplexTensor(_) | Value::Complex(_, _) => Err(writematrix_error(
                "writematrix: complex values are not supported; write real and imaginary parts separately",
            )),
            other => {
                let tensor = tensor::value_into_tensor_for("writematrix", other)
                    .map_err(writematrix_error)?;
                ensure_matrix_shape(&tensor.shape, "numeric")?;
                Ok(MatrixData::Numeric(tensor))
            }
        }
    }

    fn rows(&self) -> usize {
        match self {
            MatrixData::Numeric(t) => t.rows(),
            MatrixData::Text { rows, .. } => *rows,
        }
    }

    fn cols(&self) -> usize {
        match self {
            MatrixData::Numeric(t) => t.cols(),
            MatrixData::Text { cols, .. } => *cols,
        }
    }

    fn format_cell(
        &self,
        row: usize,
        col: usize,
        options: &WriteMatrixOptions,
        delimiter: &str,
    ) -> String {
        match self {
            MatrixData::Numeric(tensor) => {
                let rows = tensor.rows();
                let idx = row + col * rows;
                let value = tensor.data[idx];
                format_numeric(value, options.decimal_separator)
            }
            MatrixData::Text { rows, data, .. } => {
                if *rows == 0 {
                    return String::new();
                }
                let idx = row + col * rows;
                format_string(&data[idx], options.quote_strings, delimiter)
            }
        }
    }
}

fn ensure_matrix_shape(shape: &[usize], context: &str) -> BuiltinResult<()> {
    if shape.len() <= 2 {
        return Ok(());
    }
    if shape[2..].iter().all(|&dim| dim == 1) {
        Ok(())
    } else {
        Err(writematrix_error(format!(
            "writematrix: {context} input must be 2-D; reshape or use writecell for higher dimensions"
        )))
    }
}

fn write_matrix(
    path: &Path,
    matrix: &MatrixData,
    options: &WriteMatrixOptions,
) -> BuiltinResult<usize> {
    let delimiter = options.resolve_delimiter(path);
    let line_ending = options.line_ending.as_str();

    let mut open_options = OpenOptions::new();
    open_options.create(true);
    match options.write_mode {
        WriteMode::Overwrite => {
            open_options.write(true).truncate(true);
        }
        WriteMode::Append => {
            open_options.write(true).append(true);
        }
    }

    let mut file = open_options.open(path).map_err(|err| {
        writematrix_error_with_source(
            format!(
                "writematrix: unable to open \"{}\" for writing ({err})",
                path.display()
            ),
            err,
        )
    })?;

    let mut bytes_written = 0usize;
    let rows = matrix.rows();
    let cols = matrix.cols();

    if rows == 0 {
        // Nothing else to do; file was truncated or created above.
        file.flush().map_err(|err| {
            writematrix_error_with_source(
                format!("writematrix: failed to flush output ({err})"),
                err,
            )
        })?;
        return Ok(0);
    }

    for row in 0..rows {
        for col in 0..cols {
            if col > 0 {
                file.write_all(delimiter.as_bytes()).map_err(|err| {
                    writematrix_error_with_source(
                        format!("writematrix: failed to write delimiter ({err})"),
                        err,
                    )
                })?;
                bytes_written += delimiter.len();
            }
            let cell = matrix.format_cell(row, col, options, &delimiter);
            if !cell.is_empty() {
                file.write_all(cell.as_bytes()).map_err(|err| {
                    writematrix_error_with_source(
                        format!("writematrix: failed to write value ({err})"),
                        err,
                    )
                })?;
                bytes_written += cell.len();
            }
        }
        file.write_all(line_ending.as_bytes()).map_err(|err| {
            writematrix_error_with_source(
                format!("writematrix: failed to write line ending ({err})"),
                err,
            )
        })?;
        bytes_written += line_ending.len();
    }

    file.flush().map_err(|err| {
        writematrix_error_with_source(format!("writematrix: failed to flush output ({err})"), err)
    })?;

    Ok(bytes_written)
}

impl WriteMatrixOptions {
    fn resolve_delimiter(&self, path: &Path) -> String {
        if let Some(custom) = &self.delimiter {
            return custom.clone();
        }
        match self.file_type {
            FileType::Text => " ".to_string(),
            FileType::DelimitedText => default_delimiter_for_path(path),
        }
    }
}

fn default_delimiter_for_path(path: &Path) -> String {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase());
    match ext.as_deref() {
        Some("csv") => ",".to_string(),
        Some("tsv") | Some("tab") => "\t".to_string(),
        Some("txt") | Some("dat") | Some("dlm") => " ".to_string(),
        _ => ",".to_string(),
    }
}

fn format_numeric(value: f64, decimal_separator: char) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-Inf".to_string()
        } else {
            "Inf".to_string()
        };
    }

    let abs = value.abs();
    let scientific = abs != 0.0 && !(1e-4..1e15).contains(&abs);
    let raw = if scientific {
        format!("{:.15e}", value)
    } else {
        format!("{:.15}", value)
    };

    let mut formatted = trim_trailing_zeros(raw);
    if formatted == "-0" {
        formatted = "0".to_string();
    }
    if decimal_separator != '.' && formatted.contains('.') {
        formatted = formatted.replace('.', &decimal_separator.to_string());
    }
    formatted
}

fn trim_trailing_zeros(mut value: String) -> String {
    if let Some(exp_pos) = value.find(['e', 'E']) {
        let exponent = value.split_off(exp_pos);
        while value.ends_with('0') {
            value.pop();
        }
        if value.ends_with('.') {
            value.pop();
        }
        value.push_str(&exponent);
        value
    } else {
        if value.contains('.') {
            while value.ends_with('0') {
                value.pop();
            }
            if value.ends_with('.') {
                value.pop();
            }
        }
        if value.is_empty() {
            "0".to_string()
        } else {
            value
        }
    }
}

fn format_string(value: &str, quote: bool, _delimiter: &str) -> String {
    if !quote {
        return value.to_string();
    }
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for ch in value.chars() {
        if ch == '"' {
            escaped.push('"');
            escaped.push('"');
        } else {
            escaped.push(ch);
        }
    }
    escaped.push('"');
    escaped
}

fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
    match value {
        Value::String(s) => normalize_path(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            normalize_path(&text)
        }
        Value::CharArray(_) => Err(writematrix_error(
            "writematrix: expected a 1-by-N character vector for the filename",
        )),
        Value::StringArray(sa) if sa.data.len() == 1 => normalize_path(&sa.data[0]),
        Value::StringArray(_) => Err(writematrix_error(
            "writematrix: filename string array inputs must be scalar",
        )),
        other => Err(writematrix_error(format!(
            "writematrix: expected filename as string scalar or character vector, got {other:?}"
        ))),
    }
}

fn normalize_path(raw: &str) -> BuiltinResult<PathBuf> {
    if raw.trim().is_empty() {
        return Err(writematrix_error("writematrix: filename must not be empty"));
    }
    Ok(Path::new(raw).to_path_buf())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_time::unix_timestamp_ms;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{StringArray, Tensor};

    use crate::builtins::common::test_support;

    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    fn temp_path(ext: &str) -> PathBuf {
        let millis = unix_timestamp_ms();
        let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "runmat_writematrix_{}_{}_{}.{}",
            std::process::id(),
            millis,
            unique,
            ext
        ));
        path
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn writematrix_writes_space_delimited_txt() {
        let path = temp_path("txt");
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let filename = path.to_string_lossy().into_owned();

        writematrix_builtin(Value::Tensor(tensor), vec![Value::from(filename)])
            .expect("writematrix");

        let contents = fs::read_to_string(&path).expect("read contents");
        assert_eq!(contents, "1 2 3\n4 5 6\n");
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn writematrix_defaults_to_comma_for_csv() {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let filename = path.to_string_lossy().into_owned();

        writematrix_builtin(Value::Tensor(tensor), vec![Value::from(filename)])
            .expect("writematrix");

        let contents = fs::read_to_string(&path).expect("read contents");
        assert_eq!(contents, "1,2,3\n");
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn writematrix_honours_write_mode_append() {
        let path = temp_path("txt");
        let first = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let second = Tensor::new(vec![4.0, 5.0, 6.0], vec![1, 3]).unwrap();
        let filename = path.to_string_lossy().into_owned();

        writematrix_builtin(Value::Tensor(first), vec![Value::from(filename.clone())])
            .expect("initial write");

        writematrix_builtin(
            Value::Tensor(second),
            vec![
                Value::from(filename.clone()),
                Value::from("WriteMode"),
                Value::from("append"),
            ],
        )
        .expect("append write");

        let contents = fs::read_to_string(&path).expect("read contents");
        assert_eq!(contents, "1 2 3\n4 5 6\n");
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn writematrix_quotes_strings_by_default() {
        let path = temp_path("csv");
        let strings = StringArray::new(vec!["Alice".to_string(), "Bob".to_string()], vec![1, 2])
            .expect("string array");
        let filename = path.to_string_lossy().into_owned();

        writematrix_builtin(Value::StringArray(strings), vec![Value::from(filename)])
            .expect("writematrix");

        let contents = fs::read_to_string(&path).expect("read contents");
        assert_eq!(contents, "\"Alice\",\"Bob\"\n");
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn writematrix_accepts_gpu_tensor_inputs() {
        test_support::with_test_provider(|provider| {
            let path = temp_path("csv");
            let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let filename = path.to_string_lossy().into_owned();

            writematrix_builtin(Value::GpuTensor(handle), vec![Value::from(filename)])
                .expect("writematrix");

            let contents = fs::read_to_string(&path).expect("read contents");
            assert_eq!(contents, "1,2\n");
            let _ = fs::remove_file(path);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
