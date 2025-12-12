//! MATLAB-compatible `readmatrix` builtin for RunMat.

use std::collections::HashSet;
use std::convert::TryFrom;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{LogicalArray, Tensor, Value};
use runmat_filesystem::File;
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::gather_if_needed;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "readmatrix",
        builtin_path = "crate::builtins::io::tabular::readmatrix"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "readmatrix"
category: "io/tabular"
keywords: ["readmatrix", "csv", "delimited text", "numeric import", "table", "range", "logical"]
summary: "Import numeric data from delimited text files into a RunMat matrix."
references:
  - https://www.mathworks.com/help/matlab/ref/readmatrix.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Performs host-side file I/O and parsing. GPU providers are not involved; gathered results remain CPU-resident."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::tabular::readmatrix::tests"
  integration: "builtins::io::tabular::readmatrix::tests::readmatrix_reads_csv_data"
---

# What does the `readmatrix` function do in MATLAB / RunMat?
`readmatrix(filename)` reads numeric data from text or delimited files and returns a dense double-precision matrix. RunMat mirrors MATLAB defaults: the function automatically detects common delimiters, skips leading header lines only when requested, infers a rectangular output, and treats empty fields as missing values.

## How does the `readmatrix` function behave in MATLAB / RunMat?
- Accepts character vectors or string scalars for the file name. String arrays must contain exactly one element.
- Supports option structs (from `detectImportOptions`) as well as name/value arguments such as `'Delimiter'`, `'NumHeaderLines'`, `'TreatAsMissing'`, `'DecimalSeparator'`, `'ThousandsSeparator'`, and `'EmptyValue'`.
- Accepts `'Range'` as either an Excel-style address (`"B2:E10"`) or a numeric vector `[rowStart colStart rowEnd colEnd]` to slice the imported data.
- Automatically detects comma, tab, semicolon, pipe, or whitespace delimiters when none are specified. Detection is based on the first few non-empty data lines.
- Parses numeric values using MATLAB-compatible rules, recognising `NaN`, `Inf`, `-Inf`, and locale-specific decimal/thousands separators.
- Treats empty fields as `NaN` by default; specify `'EmptyValue', value` to inject a replacement scalar, or `'TreatAsMissing', tokens` to mark additional strings as missing.
- `'OutputType','logical'` coalesces non-zero numeric values (including `NaN`) to logical true, mirroring MATLAB's casting behaviour.
- `'Like', prototype` matches the output class and residency of an existing array. Supplying a GPU tensor keeps the parsed matrix on the device when an acceleration provider is active.
- Tolerates ragged rows by padding trailing elements with the configured empty value (default `NaN`).
- Raises descriptive errors when the file cannot be read or when a field cannot be parsed as a numeric value.

## `readmatrix` Function GPU Execution Behaviour
`readmatrix` always executes on the host CPU. If the file name or option arguments are GPU-resident scalars, RunMat gathers them automatically before accessing the filesystem. The resulting matrix is created in host memory unless you pass `'Like', gpuPrototype`, in which case the parsed tensor is uploaded to the same provider so subsequent operations remain on the device. Acceleration providers do not need bespoke hooks for this builtin.

## Examples of using the `readmatrix` function in MATLAB / RunMat

### Read Comma-Separated Values With Automatic Delimiter Detection
```matlab
M = readmatrix("data/scores.csv");
```
Expected output:
```matlab
% Returns a numeric matrix containing the CSV data.
```

### Skip Header Lines Before Reading Numeric Data
```matlab
M = readmatrix("data/sensor_log.txt", 'NumHeaderLines', 2);
```
Expected output:
```matlab
% The first two lines are skipped; the remaining numeric rows are returned.
```

### Import Tab-Delimited Text By Specifying The Delimiter
```matlab
M = readmatrix("data/report.tsv", 'Delimiter', 'tab');
```
Expected output:
```matlab
% Numeric matrix representing the tab-delimited values.
```

### Treat Custom Tokens As Missing Values
```matlab
M = readmatrix("data/results.csv", 'TreatAsMissing', ["NA", "missing"]);
```
Expected output:
```matlab
% Entries equal to "NA" or "missing" become NaN in the output matrix.
```

### Use European Decimal And Thousands Separators
```matlab
M = readmatrix("data/europe.csv", 'Delimiter', ';', 'DecimalSeparator', ',', 'ThousandsSeparator', '.');
```
Expected output:
```matlab
% Values like "1.234,56" are interpreted as 1234.56.
```

### Replace Empty Numeric Fields With A Custom Value
```matlab
M = readmatrix("data/with_blanks.csv", 'EmptyValue', 0);
```
Expected output:
```matlab
% Blank entries become 0 instead of NaN.
```

### Import A Specific Range Of Cells
```matlab
M = readmatrix("data/quarterly.csv", 'Range', 'B2:D5');
```
Expected output:
```matlab
% Returns only the rows and columns covered by the specified range.
```

### Convert The Result To A Logical Matrix
```matlab
flags = readmatrix("data/thresholds.csv", 'OutputType', 'logical');
```
Expected output:
```matlab
% Non-zero entries (including NaN) become logical true, zero stays false.
```

### Keep The Result On The GPU By Matching A Prototype
```matlab
proto = gpuArray.zeros(1);        % simple prototype to establish residency
G = readmatrix("data/heavy.csv", 'Like', proto);
```
Expected output:
```matlab
% The parsed matrix is uploaded to the same GPU device as the prototype.
```

### Provide Options Using A Struct From detectImportOptions
```matlab
opts = struct('Delimiter', ',', 'NumHeaderLines', 1);
M = readmatrix("data/measurements.csv", opts);
```
Expected output:
```matlab
% Reads the file using the supplied options struct.
```

## FAQ

### What file encodings does `readmatrix` support?
The builtin reads UTF-8 text files by default. If a file starts with a UTF-8 byte-order mark, it is ignored automatically. For other encodings, convert the file with `fileread`/`string` or external tools before calling `readmatrix`.

### Does `readmatrix` support Excel or binary files?
This implementation focuses on delimited text files. For MAT-files use `load`, and for spreadsheets use the `readtable` / `readcell` family (planned).

### How are missing values represented?
Empty fields become NaN unless `'EmptyValue'` is supplied. Additional tokens can be marked missing with `'TreatAsMissing'`, which also converts them to NaN.

### What happens when rows have different numbers of columns?
RunMat pads short rows with the empty value (default NaN) so the output remains rectangular.

### Can I import only part of the file?
Yes. Pass `'Range', 'B3:F20'` (Excel-style addresses) or `'Range', [3 2 10 6]` (row/column indices) to slice the data before it is materialised. Rows or columns outside the range are ignored entirely.

### Are comment lines supported?
Lines that are entirely blank are ignored. Use `'NumHeaderLines'` to skip introductory text or call `detectImportOptions` for more control.

### How do I read files stored on the GPU?
File paths are always gathered to the CPU before reading. By default the parsed matrix is created on the host; supply `'Like', gpuArray.zeros(1)` (or any GPU prototype) to upload the result automatically, or call `gpuArray` afterwards to move it manually.

### Can I request single-precision output?
`readmatrix` currently returns double-precision arrays, matching MATLAB defaults. Cast the result with `single(...)` when you need single precision.

### How are delimiters detected automatically?
The builtin inspects the first few non-empty data lines and chooses the candidate delimiter (comma, tab, semicolon, pipe, or whitespace) that produces the most columns consistently. Explicit `'Delimiter'` settings override detection.

### How are thousands separators handled?
Specify `'ThousandsSeparator'` to strip that character before parsing, e.g. `'.'` for European locales. The thousands and decimal separators must be different.

### Does `readmatrix` modify the current directory?
No. Relative paths are resolved against the current working directory, exactly like MATLAB.

## See Also
[fileread](../../filetext/fileread), [load](../../mat/load), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `readmatrix` function is available at: [`crates/runmat-runtime/src/builtins/io/tabular/readmatrix.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/tabular/readmatrix.rs)
- Found a bug or behavioral difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::tabular::readmatrix")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "readmatrix",
    op_kind: GpuOpKind::Custom("io-readmatrix"),
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::tabular::readmatrix")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "readmatrix",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion; executes as a standalone host operation.",
};

#[runtime_builtin(
    name = "readmatrix",
    category = "io/tabular",
    summary = "Import numeric data from delimited text files into a RunMat matrix.",
    keywords = "readmatrix,csv,delimited text,numeric import,table",
    accel = "cpu",
    builtin_path = "crate::builtins::io::tabular::readmatrix"
)]
fn readmatrix_builtin(path: Value, rest: Vec<Value>) -> Result<Value, String> {
    let path_value = gather_if_needed(&path).map_err(|e| format!("readmatrix: {e}"))?;
    let options = parse_options(&rest)?;
    options.validate()?;
    let resolved = resolve_path(&path_value)?;
    let tensor = read_numeric_matrix(&resolved, &options)?;
    finalize_output(tensor, &options)
}

fn parse_options(args: &[Value]) -> Result<ReadMatrixOptions, String> {
    let mut options = ReadMatrixOptions::default();
    let mut index = 0usize;
    if let Some(Value::Struct(struct_value)) = args.get(index) {
        parse_struct_options(struct_value, &mut options)?;
        index += 1;
    }
    while index < args.len() {
        if index + 1 >= args.len() {
            return Err("readmatrix: name/value inputs must appear in pairs".to_string());
        }
        let name_value = gather_if_needed(&args[index]).map_err(|e| format!("readmatrix: {e}"))?;
        let name = option_name_from_value(&name_value)?;
        let value = &args[index + 1];
        apply_option(&mut options, &name, value)?;
        index += 2;
    }
    Ok(options)
}

fn parse_struct_options(
    struct_value: &runmat_builtins::StructValue,
    options: &mut ReadMatrixOptions,
) -> Result<(), String> {
    for (name, value) in &struct_value.fields {
        apply_option(options, name, value)?;
    }
    Ok(())
}

fn apply_option(options: &mut ReadMatrixOptions, name: &str, value: &Value) -> Result<(), String> {
    let lowered = name.trim().to_ascii_lowercase();
    let is_like = lowered == "like";
    let effective_value = if is_like {
        value.clone()
    } else {
        gather_if_needed(value).map_err(|e| format!("readmatrix: {e}"))?
    };
    if name.eq_ignore_ascii_case("Delimiter") {
        let delimiter = parse_delimiter(&effective_value)?;
        options.delimiter = Some(delimiter);
        return Ok(());
    }
    if name.eq_ignore_ascii_case("NumHeaderLines") {
        let header_lines = value_to_usize(&effective_value, "NumHeaderLines")?;
        options.num_header_lines = header_lines;
        return Ok(());
    }
    if name.eq_ignore_ascii_case("TreatAsMissing") {
        let tokens = parse_treat_as_missing(&effective_value)?;
        for token in tokens {
            options.add_missing_token(&token);
        }
        return Ok(());
    }
    if name.eq_ignore_ascii_case("DecimalSeparator") {
        let sep = parse_separator_char(&effective_value, "DecimalSeparator")?;
        options.decimal_separator = sep;
        return Ok(());
    }
    if name.eq_ignore_ascii_case("ThousandsSeparator") {
        let sep = parse_separator_char(&effective_value, "ThousandsSeparator")?;
        options.thousands_separator = Some(sep);
        return Ok(());
    }
    if name.eq_ignore_ascii_case("EmptyValue") {
        let numeric = value_to_f64(&effective_value, "EmptyValue")?;
        options.empty_value = Some(numeric);
        return Ok(());
    }
    if name.eq_ignore_ascii_case("OutputType") {
        let text = value_to_string_scalar(&effective_value, "OutputType")?;
        options.set_output_type(&text)?;
        return Ok(());
    }
    if name.eq_ignore_ascii_case("Range") {
        let range = parse_range(&effective_value)?;
        options.range = Some(range);
        return Ok(());
    }
    if is_like {
        options.set_like(effective_value)?;
        return Ok(());
    }
    // Unknown options are ignored for forward compatibility.
    Ok(())
}

fn option_name_from_value(value: &Value) -> Result<String, String> {
    value_to_string_scalar(value, "option name")
}

fn parse_delimiter(value: &Value) -> Result<Delimiter, String> {
    let text = value_to_string_scalar(value, "Delimiter")?;
    if text.is_empty() {
        return Err("readmatrix: Delimiter cannot be empty".to_string());
    }
    let trimmed_lower = text.trim().to_ascii_lowercase();
    match trimmed_lower.as_str() {
        "tab" => Ok(Delimiter::Char('\t')),
        "space" | "whitespace" => Ok(Delimiter::Whitespace),
        "comma" => Ok(Delimiter::Char(',')),
        "semicolon" => Ok(Delimiter::Char(';')),
        "pipe" => Ok(Delimiter::Char('|')),
        _ => {
            if text.chars().count() == 1 {
                Ok(Delimiter::Char(text.chars().next().unwrap()))
            } else {
                Ok(Delimiter::String(text))
            }
        }
    }
}

fn parse_separator_char(value: &Value, option_name: &str) -> Result<char, String> {
    let text = value_to_string_scalar(value, option_name)?;
    if text.is_empty() {
        return Err(format!(
            "readmatrix: {option_name} must be a single character"
        ));
    }
    let mut chars = text.chars();
    let ch = chars.next().unwrap();
    if chars.next().is_some() {
        return Err(format!(
            "readmatrix: {option_name} must be a single character"
        ));
    }
    if ch == '\n' || ch == '\r' {
        return Err(format!(
            "readmatrix: {option_name} cannot be a newline character"
        ));
    }
    Ok(ch)
}

fn value_to_string_scalar(value: &Value, context: &str) -> Result<String, String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                Ok(sa.data[0].clone())
            } else {
                Err(format!(
                    "readmatrix: {context} must be a scalar string array"
                ))
            }
        }
        _ => Err(format!(
            "readmatrix: expected {context} as a string scalar or character vector"
        )),
    }
}

fn value_to_usize(value: &Value, context: &str) -> Result<usize, String> {
    match value {
        Value::Int(i) => {
            let num = i.to_i64();
            if num < 0 {
                Err(format!(
                    "readmatrix: {context} must be a non-negative integer"
                ))
            } else {
                Ok(num as usize)
            }
        }
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(format!(
                    "readmatrix: {context} must be a finite non-negative integer"
                ));
            }
            if *n < 0.0 {
                return Err(format!(
                    "readmatrix: {context} must be a non-negative integer"
                ));
            }
            if (n.round() - n).abs() > f64::EPSILON {
                return Err(format!("readmatrix: {context} must be an integer value"));
            }
            Ok(n.round() as usize)
        }
        _ => Err(format!(
            "readmatrix: {context} must be provided as a numeric scalar"
        )),
    }
}

fn value_to_f64(value: &Value, context: &str) -> Result<f64, String> {
    match value {
        Value::Num(n) => {
            if n.is_finite() {
                Ok(*n)
            } else {
                Err(format!(
                    "readmatrix: {context} must be a finite numeric scalar"
                ))
            }
        }
        Value::Int(i) => Ok(i.to_f64()),
        Value::Tensor(t) => {
            if t.data.len() == 1 {
                let v = t.data[0];
                if v.is_finite() {
                    Ok(v)
                } else {
                    Err(format!(
                        "readmatrix: {context} must be a finite numeric scalar"
                    ))
                }
            } else {
                Err(format!("readmatrix: {context} must be a numeric scalar"))
            }
        }
        _ => Err(format!("readmatrix: {context} must be a numeric scalar")),
    }
}

fn parse_treat_as_missing(value: &Value) -> Result<Vec<String>, String> {
    match value {
        Value::String(s) => Ok(vec![s.clone()]),
        Value::CharArray(ca) if ca.rows == 1 => Ok(vec![ca.data.iter().collect()]),
        Value::StringArray(sa) => Ok(sa.data.clone()),
        Value::Num(n) => Ok(vec![format_numeric_token(*n)]),
        Value::Int(i) => Ok(vec![format!("{}", i.to_i64())]),
        Value::Tensor(t) => {
            if t.data.len() == 1 {
                Ok(vec![format_numeric_token(t.data[0])])
            } else {
                Err("readmatrix: TreatAsMissing entries must be scalar values".to_string())
            }
        }
        Value::Cell(_) => {
            let nested = Vec::<Value>::try_from(value).map_err(|_| {
                "readmatrix: TreatAsMissing cell arrays must contain scalars".to_string()
            })?;
            let mut out = Vec::new();
            for entry in nested {
                let gathered = gather_if_needed(&entry).map_err(|e| format!("readmatrix: {e}"))?;
                for token in parse_treat_as_missing(&gathered)? {
                    out.push(token);
                }
            }
            Ok(out)
        }
        _ => {
            Err("readmatrix: TreatAsMissing values must be strings or numeric scalars".to_string())
        }
    }
}

fn format_numeric_token(value: f64) -> String {
    if value == 0.0 {
        "0".to_string()
    } else {
        format!("{}", value)
    }
}

#[derive(Clone)]
struct ReadMatrixOptions {
    delimiter: Option<Delimiter>,
    num_header_lines: usize,
    decimal_separator: char,
    thousands_separator: Option<char>,
    treat_as_missing: HashSet<String>,
    empty_value: Option<f64>,
    range: Option<RangeSpec>,
    output_template: OutputTemplate,
}

impl Default for ReadMatrixOptions {
    fn default() -> Self {
        Self {
            delimiter: None,
            num_header_lines: 0,
            decimal_separator: '.',
            thousands_separator: None,
            treat_as_missing: HashSet::new(),
            empty_value: None,
            range: None,
            output_template: OutputTemplate::Double,
        }
    }
}

impl ReadMatrixOptions {
    fn add_missing_token(&mut self, token: &str) {
        let normalized = normalize_missing_token(token);
        self.treat_as_missing.insert(normalized);
    }

    fn is_missing_token(&self, token: &str) -> bool {
        if self.treat_as_missing.is_empty() {
            return false;
        }
        let norm = normalize_missing_token(token);
        self.treat_as_missing.contains(&norm)
    }

    fn empty_value(&self) -> f64 {
        self.empty_value.unwrap_or(f64::NAN)
    }

    fn validate(&self) -> Result<(), String> {
        if let Some(range) = &self.range {
            range.validate()?;
        }
        if let Some(sep) = self.thousands_separator {
            if sep == self.decimal_separator {
                return Err(
                    "readmatrix: DecimalSeparator and ThousandsSeparator must differ".to_string(),
                );
            }
        }
        Ok(())
    }

    fn set_output_type(&mut self, spec: &str) -> Result<(), String> {
        if matches!(self.output_template, OutputTemplate::Like(_)) {
            return Err("readmatrix: cannot combine 'Like' with OutputType".to_string());
        }
        if spec.eq_ignore_ascii_case("double") {
            self.output_template = OutputTemplate::Double;
            return Ok(());
        }
        if spec.eq_ignore_ascii_case("logical") {
            self.output_template = OutputTemplate::Logical;
            return Ok(());
        }
        Err(format!("readmatrix: unsupported OutputType '{}'", spec))
    }

    fn set_like(&mut self, proto: Value) -> Result<(), String> {
        if matches!(self.output_template, OutputTemplate::Like(_)) {
            return Err("readmatrix: multiple 'Like' specifications are not supported".to_string());
        }
        if !matches!(self.output_template, OutputTemplate::Double) {
            return Err("readmatrix: cannot combine 'Like' with OutputType overrides".to_string());
        }
        self.output_template = OutputTemplate::Like(proto);
        Ok(())
    }
}

#[derive(Clone)]
enum Delimiter {
    Char(char),
    String(String),
    Whitespace,
}

#[derive(Clone)]
enum OutputTemplate {
    Double,
    Logical,
    Like(Value),
}

#[derive(Clone)]
struct RangeSpec {
    start_row: usize,
    start_col: usize,
    end_row: Option<usize>,
    end_col: Option<usize>,
}

impl RangeSpec {
    fn validate(&self) -> Result<(), String> {
        if let Some(end_row) = self.end_row {
            if end_row < self.start_row {
                return Err("readmatrix: Range end row must be >= start row".to_string());
            }
        }
        if let Some(end_col) = self.end_col {
            if end_col < self.start_col {
                return Err("readmatrix: Range end column must be >= start column".to_string());
            }
        }
        Ok(())
    }
}

fn normalize_missing_token(token: &str) -> String {
    token.trim().to_ascii_lowercase()
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
                Err("readmatrix: Range string array inputs must be scalar".to_string())
            }
        }
        Value::Tensor(_) => parse_range_numeric(value),
        _ => Err("readmatrix: Range must be provided as a string or numeric vector".to_string()),
    }
}

fn parse_range_string(text: &str) -> Result<RangeSpec, String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err("readmatrix: Range string cannot be empty".to_string());
    }
    let parts: Vec<&str> = trimmed.split(':').collect();
    if parts.len() > 2 {
        return Err(format!(
            "readmatrix: invalid Range specification '{}'",
            text
        ));
    }
    let start = parse_cell_reference(parts[0])?;
    if start.col.is_none() {
        return Err("readmatrix: Range must specify a starting column".to_string());
    }
    let end_ref = if parts.len() == 2 {
        Some(parse_cell_reference(parts[1])?)
    } else {
        None
    };
    if let Some(ref end) = end_ref {
        if end.col.is_none() {
            return Err("readmatrix: Range end must include a column reference".to_string());
        }
    }
    let start_row = start.row.unwrap_or(0);
    let start_col = start.col.unwrap();
    let end_row = end_ref.as_ref().and_then(|r| r.row);
    let end_col = end_ref.as_ref().and_then(|r| r.col);
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
                "readmatrix: numeric Range must be provided as a vector with 2 or 4 elements"
                    .to_string(),
            )
        }
    };
    if elements.len() != 2 && elements.len() != 4 {
        return Err("readmatrix: numeric Range must contain exactly 2 or 4 elements".to_string());
    }
    let mut indices = Vec::with_capacity(elements.len());
    for (idx, value) in elements.iter().enumerate() {
        let converted = positive_index(*value, idx)?;
        indices.push(converted);
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

fn positive_index(value: f64, position: usize) -> Result<usize, String> {
    if !value.is_finite() {
        return Err("readmatrix: Range indices must be finite".to_string());
    }
    if value < 1.0 {
        return Err("readmatrix: Range indices must be >= 1".to_string());
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err("readmatrix: Range indices must be integers".to_string());
    }
    let zero_based = (rounded as i64) - 1;
    if zero_based < 0 {
        return Err("readmatrix: Range indices must be >= 1".to_string());
    }
    usize::try_from(zero_based).map_err(|_| {
        format!(
            "readmatrix: Range index {} is too large to fit in usize",
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
            return Err(format!("readmatrix: invalid Range component '{}'", token));
        }
    }
    if letters.is_empty() && digits.is_empty() {
        return Err("readmatrix: Range references cannot be empty".to_string());
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
                "readmatrix: invalid row index '{}' in Range component '{}'",
                digits, token
            )
        })?;
        if parsed == 0 {
            return Err("readmatrix: Range rows must be >= 1".to_string());
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
                "readmatrix: invalid column designator '{}' in Range",
                letters
            ));
        }
        let digit = (ch as u8 - b'A' + 1) as usize;
        value = value
            .checked_mul(26)
            .and_then(|v| v.checked_add(digit))
            .ok_or_else(|| "readmatrix: Range column index overflowed".to_string())?;
    }
    value
        .checked_sub(1)
        .ok_or_else(|| "readmatrix: Range column index underflowed".to_string())
}

fn apply_range(
    rows: &[Vec<f64>],
    max_cols: usize,
    range: &RangeSpec,
    default_fill: f64,
) -> (Vec<Vec<f64>>, usize) {
    if rows.is_empty() || max_cols == 0 {
        return (Vec::new(), 0);
    }
    if range.start_row >= rows.len() || range.start_col >= max_cols {
        return (Vec::new(), 0);
    }
    let last_row = rows.len().saturating_sub(1);
    let mut end_row = range.end_row.unwrap_or(last_row);
    if end_row > last_row {
        end_row = last_row;
    }
    if end_row < range.start_row {
        return (Vec::new(), 0);
    }

    let last_col = max_cols.saturating_sub(1);
    let mut end_col = range.end_col.unwrap_or(last_col);
    if end_col > last_col {
        end_col = last_col;
    }
    if end_col < range.start_col {
        return (Vec::new(), 0);
    }

    let mut subset = Vec::new();
    let mut subset_max_cols = 0usize;
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
        subset_max_cols = subset_max_cols.max(extracted.len());
        subset.push(extracted);
    }

    if subset_max_cols == 0 {
        (Vec::new(), 0)
    } else {
        (subset, subset_max_cols)
    }
}

fn finalize_output(tensor: Tensor, options: &ReadMatrixOptions) -> Result<Value, String> {
    match &options.output_template {
        OutputTemplate::Double => Ok(Value::Tensor(tensor)),
        OutputTemplate::Logical => tensor_to_logical(tensor),
        OutputTemplate::Like(proto) => finalize_like(tensor, proto),
    }
}

fn tensor_to_logical(tensor: Tensor) -> Result<Value, String> {
    let mut data = Vec::with_capacity(tensor.data.len());
    for value in &tensor.data {
        let bit = if *value == 0.0 { 0 } else { 1 };
        data.push(bit);
    }
    let logical =
        LogicalArray::new(data, tensor.shape.clone()).map_err(|e| format!("readmatrix: {e}"))?;
    Ok(Value::LogicalArray(logical))
}

fn finalize_like(tensor: Tensor, proto: &Value) -> Result<Value, String> {
    match proto {
        Value::LogicalArray(_) | Value::Bool(_) => tensor_to_logical(tensor),
        Value::GpuTensor(handle) => tensor_to_gpu(tensor, handle),
        Value::Tensor(_) | Value::Num(_) | Value::Int(_) => Ok(Value::Tensor(tensor)),
        Value::ComplexTensor(_) | Value::Complex(_, _) => Ok(Value::Tensor(tensor)),
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) => Ok(Value::Tensor(tensor)),
        Value::Cell(_) => Ok(Value::Tensor(tensor)),
        _ => Ok(Value::Tensor(tensor)),
    }
}

fn tensor_to_gpu(
    tensor: Tensor,
    _handle: &runmat_accelerate_api::GpuTensorHandle,
) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        if let Ok(uploaded) = provider.upload(&view) {
            return Ok(Value::GpuTensor(uploaded));
        }
    }
    Ok(Value::Tensor(tensor))
}

fn read_numeric_matrix(path: &Path, options: &ReadMatrixOptions) -> Result<Tensor, String> {
    let file = File::open(path)
        .map_err(|err| format!("readmatrix: unable to read '{}': {}", path.display(), err))?;
    let reader = BufReader::new(file);
    let mut data_lines: Vec<(usize, String)> = Vec::new();
    for (idx, line_result) in reader.lines().enumerate() {
        let line_number = idx + 1;
        let line = line_result
            .map_err(|err| format!("readmatrix: error reading '{}': {}", path.display(), err))?;
        let cleaned = line.trim_end_matches('\r');
        if line_number <= options.num_header_lines {
            continue;
        }
        if cleaned.trim().is_empty() {
            continue;
        }
        data_lines.push((line_number, cleaned.to_string()));
    }

    if data_lines.is_empty() {
        return Tensor::new(Vec::new(), vec![0, 0]).map_err(|e| format!("readmatrix: {e}"));
    }

    if let Some((_, first_line)) = data_lines.first_mut() {
        if first_line.starts_with('\u{FEFF}') {
            let stripped = first_line.trim_start_matches('\u{FEFF}').to_string();
            *first_line = stripped;
        }
    }

    let delimiter = options
        .delimiter
        .clone()
        .or_else(|| detect_delimiter(&data_lines))
        .unwrap_or(Delimiter::Whitespace);

    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut max_cols = 0usize;

    for (line_number, text) in &data_lines {
        let fields = split_fields(text, &delimiter);
        if fields.is_empty() {
            continue;
        }
        let mut row = Vec::with_capacity(fields.len());
        for (index, field) in fields.iter().enumerate() {
            let value = parse_numeric_token(field, options, *line_number, index + 1)?;
            row.push(value);
        }
        if row.len() > max_cols {
            max_cols = row.len();
        }
        rows.push(row);
    }

    if rows.is_empty() {
        return Tensor::new(Vec::new(), vec![0, 0]).map_err(|e| format!("readmatrix: {e}"));
    }

    let default_fill = options.empty_value();
    if let Some(range) = &options.range {
        let (subset_rows, subset_cols) = apply_range(&rows, max_cols, range, default_fill);
        rows = subset_rows;
        max_cols = subset_cols;
    }

    if rows.is_empty() || max_cols == 0 {
        return Tensor::new(Vec::new(), vec![0, 0]).map_err(|e| format!("readmatrix: {e}"));
    }
    let row_count = rows.len();
    let mut data = vec![default_fill; row_count * max_cols];

    for (row_index, row) in rows.iter().enumerate() {
        for col_index in 0..max_cols {
            let value = row.get(col_index).copied().unwrap_or(default_fill);
            data[col_index * row_count + row_index] = value;
        }
    }

    Tensor::new(data, vec![row_count, max_cols]).map_err(|e| format!("readmatrix: {e}"))
}

fn detect_delimiter(lines: &[(usize, String)]) -> Option<Delimiter> {
    if lines.is_empty() {
        return None;
    }
    let sample: Vec<&str> = lines
        .iter()
        .take(32)
        .map(|(_, line)| line.as_str())
        .collect();
    let candidates = [',', '\t', ';', '|'];
    let mut best: Option<(f64, Delimiter)> = None;

    for candidate in candidates {
        let mut counts = Vec::new();
        for line in &sample {
            if !line.contains(candidate) {
                continue;
            }
            let fields = line.split(candidate).count();
            if fields >= 2 {
                counts.push(fields);
            }
        }
        if counts.is_empty() {
            continue;
        }
        let average = counts.iter().copied().sum::<usize>() as f64 / counts.len() as f64;
        if average < 2.0 {
            continue;
        }
        if best
            .as_ref()
            .map(|(best_avg, _)| average > *best_avg)
            .unwrap_or(true)
        {
            best = Some((average, Delimiter::Char(candidate)));
        }
    }

    if let Some((_, delimiter)) = best {
        return Some(delimiter);
    }

    let mut whitespace_hits = 0usize;
    for line in &sample {
        if line.split_whitespace().count() > 1 {
            whitespace_hits += 1;
        }
    }
    if whitespace_hits > 0 {
        Some(Delimiter::Whitespace)
    } else {
        None
    }
}

fn split_fields(line: &str, delimiter: &Delimiter) -> Vec<String> {
    match delimiter {
        Delimiter::Char(ch) => split_with_char_delim(line, *ch),
        Delimiter::String(pattern) => line.split(pattern).map(|s| s.to_string()).collect(),
        Delimiter::Whitespace => line.split_whitespace().map(|s| s.to_string()).collect(),
    }
}

fn split_with_char_delim(line: &str, delimiter: char) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '"' {
            if in_quotes && chars.peek() == Some(&'"') {
                current.push('"');
                chars.next();
            } else {
                in_quotes = !in_quotes;
            }
            continue;
        }
        if ch == delimiter && !in_quotes {
            fields.push(current.clone());
            current.clear();
        } else {
            current.push(ch);
        }
    }
    fields.push(current);
    fields
}

fn parse_numeric_token(
    token: &str,
    options: &ReadMatrixOptions,
    line_number: usize,
    column_number: usize,
) -> Result<f64, String> {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return Ok(options.empty_value());
    }
    let unquoted = unquote(trimmed);
    let inner = unquoted.trim();
    if inner.is_empty() {
        return Ok(options.empty_value());
    }
    if options.is_missing_token(inner) {
        return Ok(f64::NAN);
    }
    let normalized = normalize_numeric_token(inner, options);
    if normalized.is_empty() {
        return Ok(options.empty_value());
    }
    let lower = normalized.to_ascii_lowercase();
    if lower == "nan" {
        return Ok(f64::NAN);
    }
    if matches!(lower.as_str(), "inf" | "+inf" | "infinity" | "+infinity") {
        return Ok(f64::INFINITY);
    }
    if matches!(lower.as_str(), "-inf" | "-infinity") {
        return Ok(f64::NEG_INFINITY);
    }
    normalized.parse::<f64>().map_err(|_| {
        format!(
            "readmatrix: unable to parse numeric value '{}' on line {} column {}",
            inner, line_number, column_number
        )
    })
}

fn normalize_numeric_token(token: &str, options: &ReadMatrixOptions) -> String {
    let mut text = token.to_string();
    if let Some(thousands) = options.thousands_separator {
        if thousands != options.decimal_separator {
            text = text.chars().filter(|ch| *ch != thousands).collect();
        }
    }
    if options.decimal_separator != '.' {
        text = text.replace(options.decimal_separator, ".");
    }
    text
}

fn unquote(token: &str) -> &str {
    if token.len() >= 2 {
        let bytes = token.as_bytes();
        if (bytes[0] == b'"' && bytes[token.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[token.len() - 1] == b'\'')
        {
            return &token[1..token.len() - 1];
        }
    }
    token
}

fn resolve_path(value: &Value) -> Result<PathBuf, String> {
    match value {
        Value::String(s) => normalize_path(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            normalize_path(&text)
        }
        Value::CharArray(_) => {
            Err("readmatrix: expected a 1-by-N character vector for the file name".to_string())
        }
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                normalize_path(&sa.data[0])
            } else {
                Err("readmatrix: string array inputs must be scalar".to_string())
            }
        }
        other => Err(format!(
            "readmatrix: expected filename as string scalar or character vector, got {other:?}"
        )),
    }
}

fn normalize_path(raw: &str) -> Result<PathBuf, String> {
    if raw.is_empty() {
        return Err("readmatrix: filename must not be empty".to_string());
    }
    Ok(Path::new(raw).to_path_buf())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_time::unix_timestamp_ms;
    use std::fs;

    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{CharArray, IntValue, LogicalArray, StringArray, Tensor};

    use crate::builtins::common::test_support;

    fn unique_path(prefix: &str) -> PathBuf {
        let millis = unix_timestamp_ms();
        let mut path = std::env::temp_dir();
        path.push(format!("runmat_{prefix}_{}_{}", std::process::id(), millis));
        path
    }

    #[test]
    fn readmatrix_reads_csv_data() {
        let path = unique_path("readmatrix_csv");
        fs::write(&path, "1,2,3\n4,5,6\n").expect("write sample file");
        let result =
            readmatrix_builtin(Value::from(path.to_string_lossy().to_string()), Vec::new())
                .expect("readmatrix");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3]);
                assert_eq!(t.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn readmatrix_skips_header_lines() {
        let path = unique_path("readmatrix_header");
        fs::write(&path, "time,value\n0,10\n1,12\n").expect("write sample file");
        let args = vec![Value::from("NumHeaderLines"), Value::Int(IntValue::I32(1))];
        let result = readmatrix_builtin(Value::from(path.to_string_lossy().to_string()), args)
            .expect("readmatrix");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![0.0, 1.0, 10.0, 12.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn readmatrix_respects_delimiter_option() {
        let path = unique_path("readmatrix_tab");
        fs::write(&path, "1\t2\t3\n4\t5\t6\n").expect("write sample file");
        let args = vec![Value::from("Delimiter"), Value::from("tab")];
        let result = readmatrix_builtin(Value::from(path.to_string_lossy().to_string()), args)
            .expect("readmatrix");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3]);
                assert_eq!(t.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn readmatrix_respects_range_string() {
        let path = unique_path("readmatrix_range_string");
        fs::write(&path, "11,12,13\n21,22,23\n31,32,33\n").expect("write sample file");
        let args = vec![Value::from("Range"), Value::from("B2:C3")];
        let result = readmatrix_builtin(Value::from(path.to_string_lossy().to_string()), args)
            .expect("readmatrix");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![22.0, 32.0, 23.0, 33.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn readmatrix_respects_range_numeric_vector() {
        let path = unique_path("readmatrix_range_numeric");
        fs::write(&path, "11,12,13\n21,22,23\n31,32,33\n").expect("write sample file");
        let range = Tensor::new(vec![2.0, 2.0, 3.0, 3.0], vec![1, 4]).expect("range tensor");
        let args = vec![Value::from("Range"), Value::Tensor(range)];
        let result = readmatrix_builtin(Value::from(path.to_string_lossy().to_string()), args)
            .expect("readmatrix");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![22.0, 32.0, 23.0, 33.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn readmatrix_treats_custom_missing_tokens() {
        let path = unique_path("readmatrix_missing");
        fs::write(&path, "1,NA,3\nNA,5,missing\n").expect("write file");
        let strings = StringArray::new(vec!["NA".to_string(), "missing".to_string()], vec![1, 2])
            .expect("string array");
        let args = vec![Value::from("TreatAsMissing"), Value::StringArray(strings)];
        let result = readmatrix_builtin(Value::from(path.to_string_lossy().to_string()), args)
            .expect("readmatrix");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3]);
                assert!(t.data[1].is_nan()); // column 1, row 2
                assert!(t.data[2].is_nan()); // column 2, row 1
                assert!(t.data[5].is_nan()); // column 3, row 2
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn readmatrix_uses_decimal_and_thousands_separators() {
        let path = unique_path("readmatrix_decimal");
        fs::write(&path, "1.234,56;7.890,12\n").expect("write sample file");
        let args = vec![
            Value::from("Delimiter"),
            Value::from(";"),
            Value::from("DecimalSeparator"),
            Value::from(","),
            Value::from("ThousandsSeparator"),
            Value::from("."),
        ];
        let result = readmatrix_builtin(Value::from(path.to_string_lossy().to_string()), args)
            .expect("readmatrix");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert!((t.data[0] - 1234.56).abs() < 1e-9);
                assert!((t.data[1] - 7890.12).abs() < 1e-9);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn readmatrix_applies_empty_value() {
        let path = unique_path("readmatrix_empty_value");
        fs::write(&path, "1,,3\n4,,6\n").expect("write sample file");
        let args = vec![Value::from("EmptyValue"), Value::Num(0.0)];
        let result = readmatrix_builtin(Value::from(path.to_string_lossy().to_string()), args)
            .expect("readmatrix");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3]);
                assert_eq!(t.data, vec![1.0, 4.0, 0.0, 0.0, 3.0, 6.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn readmatrix_accepts_struct_options() {
        let path = unique_path("readmatrix_struct_opts");
        fs::write(&path, "header1,header2\n9,10\n11,12\n").expect("write sample file");
        let mut options_struct = runmat_builtins::StructValue::new();
        options_struct
            .fields
            .insert("Delimiter".to_string(), Value::from(","));
        options_struct
            .fields
            .insert("NumHeaderLines".to_string(), Value::Int(IntValue::I32(1)));
        let args = vec![Value::Struct(options_struct)];
        let result = readmatrix_builtin(Value::from(path.to_string_lossy().to_string()), args)
            .expect("readmatrix");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![9.0, 11.0, 10.0, 12.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn readmatrix_errors_on_non_numeric_field() {
        let path = unique_path("readmatrix_error");
        fs::write(&path, "1,abc,3\n").expect("write sample file");
        let err = readmatrix_builtin(Value::from(path.to_string_lossy().to_string()), Vec::new())
            .expect_err("readmatrix should fail");
        assert!(
            err.contains("unable to parse numeric value"),
            "unexpected error message: {err}"
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn readmatrix_returns_empty_on_no_data() {
        let path = unique_path("readmatrix_empty");
        File::create(&path).expect("create file");
        let result =
            readmatrix_builtin(Value::from(path.to_string_lossy().to_string()), Vec::new())
                .expect("readmatrix");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn readmatrix_output_type_logical() {
        let path = unique_path("readmatrix_output_logical");
        fs::write(&path, "0,1,-3\nNaN,0,5\n").expect("write sample file");
        let args = vec![Value::from("OutputType"), Value::from("logical")];
        let result = readmatrix_builtin(Value::from(path.to_string_lossy().to_string()), args)
            .expect("readmatrix");
        match result {
            Value::LogicalArray(arr) => {
                assert_eq!(arr.shape, vec![2, 3]);
                assert_eq!(arr.data, vec![0, 1, 1, 0, 1, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn readmatrix_like_logical_proto() {
        let path = unique_path("readmatrix_like_logical");
        fs::write(&path, "1,0\n0,5\n").expect("write sample file");
        let proto = LogicalArray::new(vec![1], vec![1]).expect("logical prototype");
        let args = vec![Value::from("Like"), Value::LogicalArray(proto)];
        let result = readmatrix_builtin(Value::from(path.to_string_lossy().to_string()), args)
            .expect("readmatrix");
        match result {
            Value::LogicalArray(arr) => {
                assert_eq!(arr.shape, vec![2, 2]);
                assert_eq!(arr.data, vec![1, 0, 0, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn readmatrix_like_gpu_proto() {
        test_support::with_test_provider(|provider| {
            let path = unique_path("readmatrix_like_gpu");
            fs::write(&path, "1,2\n3,4\n").expect("write sample file");
            let proto_tensor = Tensor::new(vec![0.0, 0.0], vec![1, 2]).expect("tensor");
            let view = HostTensorView {
                data: &proto_tensor.data,
                shape: &proto_tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload prototype");
            let args = vec![Value::from("Like"), Value::GpuTensor(handle.clone())];
            let result = readmatrix_builtin(Value::from(path.to_string_lossy().to_string()), args)
                .expect("readmatrix");
            assert!(
                matches!(result, Value::GpuTensor(_)),
                "expected GPU tensor result, got {result:?}"
            );
            let gathered = test_support::gather(result).expect("gather result");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![1.0, 3.0, 2.0, 4.0]);
            let _ = fs::remove_file(&path);
        });
    }

    #[test]
    fn readmatrix_accepts_character_vector_path() {
        let path = unique_path("readmatrix_char_path");
        fs::write(&path, "1 2 3\n").expect("write sample file");
        let text = path.to_string_lossy().to_string();
        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();
        let char_array = CharArray::new(chars, 1, len).expect("char array");
        let result =
            readmatrix_builtin(Value::CharArray(char_array), Vec::new()).expect("readmatrix");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.data, vec![1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn readmatrix_handles_quoted_fields() {
        let path = unique_path("readmatrix_quotes");
        fs::write(&path, "\"1\",\"2\",\"3\"\n").expect("write sample file");
        let result =
            readmatrix_builtin(Value::from(path.to_string_lossy().to_string()), Vec::new())
                .expect("readmatrix");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.data, vec![1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn readmatrix_preserves_negative_infinity() {
        let path = unique_path("readmatrix_infinity");
        fs::write(&path, "-Inf,Inf,NaN\n").expect("write sample file");
        let result =
            readmatrix_builtin(Value::from(path.to_string_lossy().to_string()), Vec::new())
                .expect("readmatrix");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert!(t.data[0].is_infinite() && t.data[0].is_sign_negative());
                assert!(t.data[1].is_infinite() && t.data[1].is_sign_positive());
                assert!(t.data[2].is_nan());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn readmatrix_supports_whitespace_delimiter() {
        let path = unique_path("readmatrix_whitespace");
        fs::write(&path, "1 2 3\n4 5 6\n").expect("write sample file");
        let result =
            readmatrix_builtin(Value::from(path.to_string_lossy().to_string()), Vec::new())
                .expect("readmatrix");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3]);
                assert_eq!(t.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
        let _ = fs::remove_file(&path);
    }
}
