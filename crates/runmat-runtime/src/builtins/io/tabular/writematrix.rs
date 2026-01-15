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
use crate::gather_if_needed;

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
fn writematrix_builtin(data: Value, rest: Vec<Value>) -> Result<Value, String> {
    if rest.is_empty() {
        return Err("writematrix: filename is required".to_string());
    }

    let filename_value = gather_if_needed(&rest[0]).map_err(|e| format!("writematrix: {e}"))?;
    let path = resolve_path(&filename_value)?;

    let options = parse_options(&rest[1..])?;

    let gathered = gather_if_needed(&data).map_err(|e| format!("writematrix: {e}"))?;
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

fn parse_options(args: &[Value]) -> Result<WriteMatrixOptions, String> {
    if args.is_empty() {
        return Ok(WriteMatrixOptions::default());
    }
    if !args.len().is_multiple_of(2) {
        return Err("writematrix: name/value inputs must appear in pairs".to_string());
    }

    let mut options = WriteMatrixOptions::default();
    let mut index = 0usize;
    while index < args.len() {
        let name_value = gather_if_needed(&args[index]).map_err(|e| format!("writematrix: {e}"))?;
        let name = option_name_from_value(&name_value)?;
        let value = gather_if_needed(&args[index + 1]).map_err(|e| format!("writematrix: {e}"))?;
        apply_option(&mut options, &name, &value)?;
        index += 2;
    }
    Ok(options)
}

fn apply_option(options: &mut WriteMatrixOptions, name: &str, value: &Value) -> Result<(), String> {
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

fn option_name_from_value(value: &Value) -> Result<String, String> {
    value_to_string_scalar(value, "option name")
}

fn parse_delimiter(value: &Value) -> Result<String, String> {
    let text = value_to_string_scalar(value, "Delimiter")?;
    if text.is_empty() {
        return Err("writematrix: Delimiter cannot be empty".to_string());
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

fn parse_write_mode(value: &Value) -> Result<WriteMode, String> {
    let text = value_to_string_scalar(value, "WriteMode")?;
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "overwrite" => Ok(WriteMode::Overwrite),
        "append" => Ok(WriteMode::Append),
        _ => Err("writematrix: WriteMode must be 'overwrite' or 'append'".to_string()),
    }
}

fn parse_bool_like(value: &Value, context: &str) -> Result<bool, String> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::Int(i) => {
            let raw = i.to_i64();
            match raw {
                0 => Ok(false),
                1 => Ok(true),
                _ => Err(format!("writematrix: {context} must be logical (0 or 1)")),
            }
        }
        Value::Num(n) => {
            if (*n - 0.0).abs() < f64::EPSILON {
                Ok(false)
            } else if (*n - 1.0).abs() < f64::EPSILON {
                Ok(true)
            } else {
                Err(format!("writematrix: {context} must be logical (0 or 1)"))
            }
        }
        _ => {
            let text = value_to_string_scalar(value, context)?;
            let lowered = text.trim().to_ascii_lowercase();
            match lowered.as_str() {
                "on" | "true" | "yes" | "1" => Ok(true),
                "off" | "false" | "no" | "0" => Ok(false),
                _ => Err(format!(
                    "writematrix: {context} must be logical (true/on or false/off)"
                )),
            }
        }
    }
}

fn parse_decimal_separator(value: &Value) -> Result<char, String> {
    let text = value_to_string_scalar(value, "DecimalSeparator")?;
    let mut chars = text.chars();
    let ch = chars
        .next()
        .ok_or_else(|| "writematrix: DecimalSeparator must be a single character".to_string())?;
    if chars.next().is_some() {
        return Err("writematrix: DecimalSeparator must be a single character".to_string());
    }
    if ch == '\n' || ch == '\r' {
        return Err("writematrix: DecimalSeparator cannot be a newline character".to_string());
    }
    Ok(ch)
}

fn parse_line_ending(value: &Value) -> Result<LineEnding, String> {
    let text = value_to_string_scalar(value, "LineEnding")?;
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "auto" => Ok(LineEnding::Auto),
        "unix" => Ok(LineEnding::Unix),
        "pc" | "windows" => Ok(LineEnding::Windows),
        "mac" => Ok(LineEnding::Mac),
        _ => Err("writematrix: LineEnding must be 'auto', 'unix', 'pc', or 'mac'".to_string()),
    }
}

fn parse_file_type(value: &Value) -> Result<FileType, String> {
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
        "spreadsheet" => Err(
            "writematrix: FileType 'spreadsheet' is not supported; export delimited text instead"
                .to_string(),
        ),
        _ => Err("writematrix: unsupported FileType".to_string()),
    }
}

fn value_to_string_scalar(value: &Value, context: &str) -> Result<String, String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        _ => Err(format!(
            "writematrix: expected {context} as a string scalar or character vector"
        )),
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
    fn from_value(value: Value) -> Result<Self, String> {
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
            Value::CharArray(_) => Err(
                "writematrix: character arrays are not supported; convert to string arrays or use writecell"
                    .to_string(),
            ),
            Value::Cell(_) => Err(
                "writematrix: cell arrays are not supported; use writecell for heterogeneous content"
                    .to_string(),
            ),
            Value::ComplexTensor(_) | Value::Complex(_, _) => Err(
                "writematrix: complex values are not supported; write real and imaginary parts separately"
                    .to_string(),
            ),
            other => {
                let tensor = tensor::value_into_tensor_for("writematrix", other)?;
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

fn ensure_matrix_shape(shape: &[usize], context: &str) -> Result<(), String> {
    if shape.len() <= 2 {
        return Ok(());
    }
    if shape[2..].iter().all(|&dim| dim == 1) {
        Ok(())
    } else {
        Err(format!(
            "writematrix: {context} input must be 2-D; reshape or use writecell for higher dimensions"
        ))
    }
}

fn write_matrix(
    path: &Path,
    matrix: &MatrixData,
    options: &WriteMatrixOptions,
) -> Result<usize, String> {
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
        format!(
            "writematrix: unable to open \"{}\" for writing ({err})",
            path.display()
        )
    })?;

    let mut bytes_written = 0usize;
    let rows = matrix.rows();
    let cols = matrix.cols();

    if rows == 0 {
        // Nothing else to do; file was truncated or created above.
        file.flush()
            .map_err(|err| format!("writematrix: failed to flush output ({err})"))?;
        return Ok(0);
    }

    for row in 0..rows {
        for col in 0..cols {
            if col > 0 {
                file.write_all(delimiter.as_bytes())
                    .map_err(|err| format!("writematrix: failed to write delimiter ({err})"))?;
                bytes_written += delimiter.len();
            }
            let cell = matrix.format_cell(row, col, options, &delimiter);
            if !cell.is_empty() {
                file.write_all(cell.as_bytes())
                    .map_err(|err| format!("writematrix: failed to write value ({err})"))?;
                bytes_written += cell.len();
            }
        }
        file.write_all(line_ending.as_bytes())
            .map_err(|err| format!("writematrix: failed to write line ending ({err})"))?;
        bytes_written += line_ending.len();
    }

    file.flush()
        .map_err(|err| format!("writematrix: failed to flush output ({err})"))?;

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

fn resolve_path(value: &Value) -> Result<PathBuf, String> {
    match value {
        Value::String(s) => normalize_path(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            normalize_path(&text)
        }
        Value::CharArray(_) => {
            Err("writematrix: expected a 1-by-N character vector for the filename".to_string())
        }
        Value::StringArray(sa) if sa.data.len() == 1 => normalize_path(&sa.data[0]),
        Value::StringArray(_) => {
            Err("writematrix: filename string array inputs must be scalar".to_string())
        }
        other => Err(format!(
            "writematrix: expected filename as string scalar or character vector, got {other:?}"
        )),
    }
}

fn normalize_path(raw: &str) -> Result<PathBuf, String> {
    if raw.trim().is_empty() {
        return Err("writematrix: filename must not be empty".to_string());
    }
    Ok(Path::new(raw).to_path_buf())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_time::unix_timestamp_ms;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{StringArray, Tensor};

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
}
