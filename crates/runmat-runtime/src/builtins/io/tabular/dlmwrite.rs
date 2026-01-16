//! MATLAB-compatible `dlmwrite` builtin for delimiter-separated exports.

#[cfg(not(target_arch = "wasm32"))]
use core::ffi::{c_char, c_int};
#[cfg(any(
    all(not(target_arch = "wasm32"), not(windows)),
    all(windows, target_env = "gnu")
))]
use libc;
use runmat_builtins::{Tensor, Value};
use runmat_filesystem::{self as vfs, File, OpenOptions};
use runmat_macros::runtime_builtin;
#[cfg(not(target_arch = "wasm32"))]
use std::ffi::CString;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::gather_if_needed;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::tabular::dlmwrite")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "dlmwrite",
    op_kind: GpuOpKind::Custom("io-dlmwrite"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Runs entirely on the host; gpuArray inputs are gathered before formatting.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::tabular::dlmwrite")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "dlmwrite",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion; performs synchronous file I/O.",
};

#[runtime_builtin(
    name = "dlmwrite",
    category = "io/tabular",
    summary = "Write numeric matrices to delimiter-separated text files.",
    keywords = "dlmwrite,delimiter,precision,append,roffset,coffset",
    accel = "cpu",
    builtin_path = "crate::builtins::io::tabular::dlmwrite"
)]
fn dlmwrite_builtin(filename: Value, data: Value, rest: Vec<Value>) -> Result<Value, String> {
    let gathered_path = gather_if_needed(&filename).map_err(|e| format!("dlmwrite: {e}"))?;
    let path = resolve_path(&gathered_path)?;

    let mut gathered_args = Vec::with_capacity(rest.len());
    for value in &rest {
        gathered_args.push(gather_if_needed(value).map_err(|e| format!("dlmwrite: {e}"))?);
    }
    let options = parse_arguments(&gathered_args)?;

    let gathered_data = gather_if_needed(&data).map_err(|e| format!("dlmwrite: {e}"))?;
    let tensor = tensor::value_into_tensor_for("dlmwrite", gathered_data)?;
    ensure_matrix_shape(&tensor)?;

    let bytes = write_dlm(&path, &tensor, &options)?;
    Ok(Value::Num(bytes as f64))
}

#[derive(Clone, Debug)]
struct DlmWriteOptions {
    delimiter: String,
    newline: LineEnding,
    roffset: usize,
    coffset: usize,
    precision: PrecisionSpec,
    append: bool,
}

impl Default for DlmWriteOptions {
    fn default() -> Self {
        Self {
            delimiter: ",".to_string(),
            newline: LineEnding::platform_default(),
            roffset: 0,
            coffset: 0,
            precision: PrecisionSpec::Significant(5),
            append: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum LineEnding {
    Unix,
    Pc,
    Mac,
}

impl LineEnding {
    fn as_str(&self) -> &'static str {
        match self {
            LineEnding::Unix => "\n",
            LineEnding::Pc => "\r\n",
            LineEnding::Mac => "\r",
        }
    }

    fn platform_default() -> Self {
        if cfg!(windows) {
            LineEnding::Pc
        } else {
            LineEnding::Unix
        }
    }
}

#[derive(Clone, Debug)]
enum PrecisionSpec {
    Significant(u32),
    Format(String),
}

fn parse_arguments(args: &[Value]) -> Result<DlmWriteOptions, String> {
    let mut options = DlmWriteOptions::default();
    if args.is_empty() {
        return Ok(options);
    }

    let mut idx = 0usize;
    // Consume any leading '-append' flags.
    while idx < args.len() {
        if is_append_flag(&args[idx]) {
            options.append = true;
            idx += 1;
        } else {
            break;
        }
    }

    // Optional positional delimiter.
    if idx < args.len() && is_positional_delimiter_candidate(&args[idx]) {
        options.delimiter = parse_delimiter_value(&args[idx])?;
        idx += 1;
        // Optional positional row/col offsets if both numeric scalars follow.
        if idx + 1 < args.len()
            && is_numeric_scalar(&args[idx])
            && is_numeric_scalar(&args[idx + 1])
        {
            options.roffset = parse_offset_value(&args[idx], "row offset")?;
            options.coffset = parse_offset_value(&args[idx + 1], "column offset")?;
            idx += 2;
        }
        // Consume any subsequent '-append'.
        while idx < args.len() {
            if is_append_flag(&args[idx]) {
                options.append = true;
                idx += 1;
            } else {
                break;
            }
        }
    }

    // Remaining arguments should be name-value pairs (allowing additional '-append').
    while idx < args.len() {
        if is_append_flag(&args[idx]) {
            options.append = true;
            idx += 1;
            continue;
        }

        let name = value_to_lowercase_string(&args[idx])
            .ok_or_else(|| format!("dlmwrite: expected name-value pair, got {:?}", args[idx]))?;
        idx += 1;
        if idx >= args.len() {
            return Err("dlmwrite: name-value arguments must appear in pairs".to_string());
        }
        let value = &args[idx];
        idx += 1;

        match name.as_str() {
            "delimiter" => {
                options.delimiter = parse_delimiter_value(value)?;
            }
            "precision" => {
                options.precision = parse_precision_value(value)?;
            }
            "newline" => {
                options.newline = parse_newline_value(value)?;
            }
            "roffset" => {
                options.roffset = parse_offset_value(value, "row offset")?;
            }
            "coffset" => {
                options.coffset = parse_offset_value(value, "column offset")?;
            }
            "append" => {
                options.append = parse_append_value(value)?;
            }
            other => {
                return Err(format!("dlmwrite: unsupported name-value pair '{other}'"));
            }
        }
    }

    Ok(options)
}

fn is_append_flag(value: &Value) -> bool {
    match value {
        Value::String(s) => s.trim().eq_ignore_ascii_case("-append"),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            text.trim().eq_ignore_ascii_case("-append")
        }
        Value::StringArray(sa) if sa.data.len() == 1 => {
            sa.data[0].trim().eq_ignore_ascii_case("-append")
        }
        _ => false,
    }
}

fn is_positional_delimiter_candidate(value: &Value) -> bool {
    match value {
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => {
            if let Some(lower) = value_to_lowercase_string(value) {
                match lower.as_str() {
                    "delimiter" | "precision" | "newline" | "roffset" | "coffset" | "append"
                    | "-append" => return false,
                    _ => {}
                }
            }
            true
        }
        _ => false,
    }
}

fn parse_delimiter_value(value: &Value) -> Result<String, String> {
    match value {
        Value::String(s) => interpret_delimiter_string(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            interpret_delimiter_string(&text)
        }
        Value::StringArray(sa) if sa.data.len() == 1 => interpret_delimiter_string(&sa.data[0]),
        _ => Err("dlmwrite: delimiter must be a string scalar or character vector".to_string()),
    }
}

fn interpret_delimiter_string(raw: &str) -> Result<String, String> {
    if raw.is_empty() {
        return Err("dlmwrite: delimiter must not be empty".to_string());
    }
    if raw == r"\t" {
        return Ok("\t".to_string());
    }
    if raw == r"\n" {
        return Ok("\n".to_string());
    }
    if raw == r"\r" {
        return Ok("\r".to_string());
    }
    Ok(raw.to_string())
}

fn is_numeric_scalar(value: &Value) -> bool {
    match value {
        Value::Int(_) | Value::Num(_) | Value::Bool(_) => true,
        Value::Tensor(t) => t.data.len() == 1,
        Value::LogicalArray(logical) => logical.data.len() == 1,
        _ => false,
    }
}

fn parse_offset_value(value: &Value, context: &str) -> Result<usize, String> {
    let scalar = extract_scalar(value).map_err(|e| format!("dlmwrite: {context} {e}"))?;
    if !scalar.is_finite() {
        return Err(format!("dlmwrite: {context} must be finite"));
    }
    let rounded = scalar.round();
    if (rounded - scalar).abs() > 1e-9 {
        return Err(format!(
            "dlmwrite: {context} must be an integer, got {scalar}"
        ));
    }
    if rounded < 0.0 {
        return Err(format!("dlmwrite: {context} must be >= 0"));
    }
    Ok(rounded as usize)
}

fn parse_precision_value(value: &Value) -> Result<PrecisionSpec, String> {
    match value {
        Value::Int(i) => {
            let digits = i.to_i64();
            if digits <= 0 {
                return Err("dlmwrite: precision must be a positive integer".to_string());
            }
            Ok(PrecisionSpec::Significant(digits as u32))
        }
        Value::Num(n) => {
            if !n.is_finite() {
                return Err("dlmwrite: precision scalar must be finite".to_string());
            }
            let rounded = n.round();
            if (rounded - n).abs() > 1e-9 {
                return Err("dlmwrite: precision scalar must be an integer".to_string());
            }
            if rounded <= 0.0 {
                return Err("dlmwrite: precision must be a positive integer".to_string());
            }
            Ok(PrecisionSpec::Significant(rounded as u32))
        }
        Value::Tensor(t) if t.data.len() == 1 => parse_precision_value(&Value::Num(t.data[0])),
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            if logical.data[0] != 0 {
                Ok(PrecisionSpec::Significant(1))
            } else {
                Err("dlmwrite: precision must be a positive integer".to_string())
            }
        }
        Value::Bool(b) => {
            if *b {
                Ok(PrecisionSpec::Significant(1))
            } else {
                Err("dlmwrite: precision must be a positive integer".to_string())
            }
        }
        Value::String(s) => parse_precision_format(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            parse_precision_format(&text)
        }
        Value::StringArray(sa) if sa.data.len() == 1 => parse_precision_format(&sa.data[0]),
        _ => Err("dlmwrite: precision must be numeric or a format string".to_string()),
    }
}

fn parse_precision_format(text: &str) -> Result<PrecisionSpec, String> {
    if text.is_empty() {
        return Err("dlmwrite: precision format string must not be empty".to_string());
    }
    Ok(PrecisionSpec::Format(text.to_string()))
}

fn parse_newline_value(value: &Value) -> Result<LineEnding, String> {
    let text = value_to_lowercase_string(value).ok_or_else(|| {
        "dlmwrite: newline must be a string scalar or character vector".to_string()
    })?;
    match text.as_str() {
        "pc" | "windows" | "crlf" => Ok(LineEnding::Pc),
        "unix" | "lf" => Ok(LineEnding::Unix),
        "mac" | "cr" => Ok(LineEnding::Mac),
        other => Err(format!(
            "dlmwrite: unsupported newline setting '{other}' (expected 'pc' or 'unix')"
        )),
    }
}

fn parse_append_value(value: &Value) -> Result<bool, String> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::Int(i) => Ok(i.to_i64() != 0),
        Value::Num(n) => Ok(*n != 0.0),
        Value::String(s) => parse_bool_string(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            parse_bool_string(&text)
        }
        Value::StringArray(sa) if sa.data.len() == 1 => parse_bool_string(&sa.data[0]),
        _ => Err("dlmwrite: append value must be logical".to_string()),
    }
}

fn parse_bool_string(text: &str) -> Result<bool, String> {
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "true" | "on" | "yes" | "1" => Ok(true),
        "false" | "off" | "no" | "0" => Ok(false),
        _ => Err("dlmwrite: append value must be logical".to_string()),
    }
}

fn extract_scalar(value: &Value) -> Result<f64, String> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if t.data.len() == 1 => Ok(t.data[0]),
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(if logical.data[0] != 0 { 1.0 } else { 0.0 })
        }
        _ => Err("must be numeric scalar".to_string()),
    }
}

fn value_to_lowercase_string(value: &Value) -> Option<String> {
    tensor::value_to_string(value).map(|s| s.trim().to_ascii_lowercase())
}

fn resolve_path(value: &Value) -> Result<PathBuf, String> {
    let raw = match value {
        Value::String(s) => s.clone(),
        Value::CharArray(ca) if ca.rows == 1 => ca.data.iter().collect(),
        Value::StringArray(sa) if sa.data.len() == 1 => sa.data[0].clone(),
        _ => {
            return Err(
                "dlmwrite: filename must be a string scalar or character vector".to_string(),
            )
        }
    };
    if raw.trim().is_empty() {
        return Err("dlmwrite: filename must not be empty".to_string());
    }
    let expanded = expand_user_path(&raw, "dlmwrite").map_err(|e| format!("dlmwrite: {e}"))?;
    Ok(Path::new(&expanded).to_path_buf())
}

fn ensure_matrix_shape(tensor: &Tensor) -> Result<(), String> {
    if tensor.shape.len() <= 2 {
        return Ok(());
    }
    if tensor.shape[2..].iter().all(|&dim| dim == 1) {
        return Ok(());
    }
    Err("dlmwrite: input must be 2-D; reshape before writing".to_string())
}

fn write_dlm(path: &Path, tensor: &Tensor, options: &DlmWriteOptions) -> Result<usize, String> {
    let rows = tensor.rows();
    let cols = tensor.cols();
    let newline = options.newline.as_str();

    let (existing_nonempty, ends_with_newline) = if options.append {
        match vfs::metadata(path) {
            Ok(meta) if !meta.is_empty() => {
                let ends = file_ends_with_newline(path).map_err(|e| {
                    format!(
                        "dlmwrite: failed to inspect existing file \"{}\" ({e})",
                        path.display()
                    )
                })?;
                (true, ends)
            }
            Ok(_) => (false, false),
            Err(err) => {
                if err.kind() == io::ErrorKind::NotFound {
                    (false, false)
                } else {
                    return Err(format!(
                        "dlmwrite: unable to inspect \"{}\" ({err})",
                        path.display()
                    ));
                }
            }
        }
    } else {
        (false, false)
    };

    let mut open = OpenOptions::new();
    open.create(true);
    if options.append {
        open.append(true);
    } else {
        open.write(true).truncate(true);
    }

    let mut file = open.open(path).map_err(|err| {
        format!(
            "dlmwrite: unable to open \"{}\" for writing ({err})",
            path.display()
        )
    })?;

    let mut bytes = 0usize;

    if options.append && existing_nonempty && !ends_with_newline {
        file.write_all(newline.as_bytes())
            .map_err(|e| format!("dlmwrite: failed to insert newline before append ({e})"))?;
        bytes += newline.len();
    }

    for _ in 0..options.roffset {
        bytes += write_blank_row(
            &mut file,
            cols,
            options.coffset,
            &options.delimiter,
            newline,
        )?;
    }

    if rows == 0 || cols == 0 {
        file.flush()
            .map_err(|e| format!("dlmwrite: failed to flush output ({e})"))?;
        return Ok(bytes);
    }

    for row in 0..rows {
        let mut fields = Vec::with_capacity(options.coffset + cols);
        for _ in 0..options.coffset {
            fields.push(String::new());
        }
        for col in 0..cols {
            let idx = row + col * rows;
            let value = tensor.data[idx];
            fields.push(format_numeric(value, &options.precision)?);
        }
        let line = fields.join(&options.delimiter);
        if !line.is_empty() {
            file.write_all(line.as_bytes())
                .map_err(|e| format!("dlmwrite: failed to write data ({e})"))?;
            bytes += line.len();
        }
        file.write_all(newline.as_bytes())
            .map_err(|e| format!("dlmwrite: failed to write newline ({e})"))?;
        bytes += newline.len();
    }

    file.flush()
        .map_err(|e| format!("dlmwrite: failed to flush output ({e})"))?;
    Ok(bytes)
}

fn write_blank_row(
    file: &mut File,
    cols: usize,
    coffset: usize,
    delimiter: &str,
    newline: &str,
) -> Result<usize, String> {
    let mut bytes = 0usize;
    if coffset == 0 && cols == 0 {
        file.write_all(newline.as_bytes())
            .map_err(|e| format!("dlmwrite: failed to write offset newline ({e})"))?;
        return Ok(newline.len());
    }
    let mut fields = Vec::with_capacity(coffset + cols);
    for _ in 0..coffset {
        fields.push(String::new());
    }
    for _ in 0..cols {
        fields.push(String::new());
    }
    let line = fields.join(delimiter);
    if !line.is_empty() {
        file.write_all(line.as_bytes())
            .map_err(|e| format!("dlmwrite: failed to write offset row ({e})"))?;
        bytes += line.len();
    }
    file.write_all(newline.as_bytes())
        .map_err(|e| format!("dlmwrite: failed to write offset newline ({e})"))?;
    bytes += newline.len();
    Ok(bytes)
}

fn file_ends_with_newline(path: &Path) -> io::Result<bool> {
    let metadata = vfs::metadata(path)?;
    let len = metadata.len();
    if len == 0 {
        return Ok(false);
    }
    let mut file = File::open(path)?;
    let to_read = len.min(2) as usize;
    file.seek(SeekFrom::End(-(to_read as i64)))?;
    let mut buffer = vec![0u8; to_read];
    file.read_exact(&mut buffer)?;
    Ok(buffer.contains(&b'\n') || buffer.contains(&b'\r'))
}

fn format_numeric(value: f64, precision: &PrecisionSpec) -> Result<String, String> {
    if value.is_nan() {
        return Ok("NaN".to_string());
    }
    if value.is_infinite() {
        return Ok(if value.is_sign_negative() {
            "-Inf".to_string()
        } else {
            "Inf".to_string()
        });
    }
    match precision {
        PrecisionSpec::Significant(digits) => {
            if *digits == 0 {
                return Err("dlmwrite: precision must be positive".to_string());
            }
            let fmt = format!("%.{digits}g");
            let mut rendered = c_format(value, &fmt)?;
            if value == 0.0 || rendered == "-0" {
                rendered = "0".to_string();
            }
            Ok(rendered)
        }
        PrecisionSpec::Format(spec) => {
            let rendered = c_format(value, spec)?;
            if value == 0.0 && rendered.starts_with("-") {
                Ok(rendered.trim_start_matches('-').to_string())
            } else {
                Ok(rendered)
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn c_format(value: f64, spec: &str) -> Result<String, String> {
    let fmt = CString::new(spec.as_bytes()).map_err(|_| {
        "dlmwrite: precision format must not contain embedded null bytes".to_string()
    })?;
    let mut size: usize = 128;
    loop {
        let mut buffer = vec![0u8; size];
        let written = unsafe {
            platform_snprintf(
                buffer.as_mut_ptr() as *mut c_char,
                size,
                fmt.as_ptr(),
                value,
            )
        };
        if written < 0 {
            return Err("dlmwrite: failed to apply precision format string".to_string());
        }
        let written = written as usize;
        if written >= size {
            size = written + 1;
            continue;
        }
        buffer.truncate(written);
        return String::from_utf8(buffer)
            .map_err(|_| "dlmwrite: formatted output was not valid UTF-8".to_string());
    }
}

#[cfg(target_arch = "wasm32")]
fn c_format(value: f64, spec: &str) -> Result<String, String> {
    wasm_format_float(value, spec)
}

#[cfg(all(not(target_arch = "wasm32"), not(windows)))]
unsafe fn platform_snprintf(
    buffer: *mut c_char,
    size: usize,
    fmt: *const c_char,
    value: f64,
) -> c_int {
    libc::snprintf(buffer, size as libc::size_t, fmt, value)
}

#[cfg(all(windows, target_env = "msvc"))]
extern "C" {
    fn _snprintf(buffer: *mut c_char, size: usize, fmt: *const c_char, ...) -> c_int;
}

#[cfg(all(windows, target_env = "msvc"))]
unsafe fn platform_snprintf(
    buffer: *mut c_char,
    size: usize,
    fmt: *const c_char,
    value: f64,
) -> c_int {
    _snprintf(buffer, size, fmt, value)
}

#[cfg(all(windows, target_env = "gnu"))]
unsafe fn platform_snprintf(
    buffer: *mut c_char,
    size: usize,
    fmt: *const c_char,
    value: f64,
) -> c_int {
    libc::snprintf(buffer, size as libc::size_t, fmt, value)
}

#[cfg(any(target_arch = "wasm32", test))]
fn wasm_format_float(value: f64, spec: &str) -> Result<String, String> {
    let parsed = ParsedFormat::parse(spec)?;
    Ok(parsed.render(value))
}

#[cfg(any(target_arch = "wasm32", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FloatSpecifier {
    Fixed,
    Exponent,
    General,
}

#[cfg(any(target_arch = "wasm32", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SignFlag {
    None,
    Plus,
    Space,
}

#[cfg(any(target_arch = "wasm32", test))]
#[derive(Clone, Copy, Debug)]
struct ParsedFormat {
    specifier: FloatSpecifier,
    uppercase: bool,
    alternate: bool,
    sign: SignFlag,
    left_adjust: bool,
    zero_pad: bool,
    width: Option<usize>,
    precision: Option<usize>,
}

#[cfg(any(target_arch = "wasm32", test))]
impl ParsedFormat {
    fn parse(input: &str) -> Result<Self, String> {
        use std::iter::Peekable;
        use std::str::Chars;

        fn parse_number(
            chars: &mut Peekable<Chars<'_>>,
            label: &str,
        ) -> Result<Option<usize>, String> {
            let mut value: usize = 0;
            let mut saw_digit = false;
            while let Some(&ch) = chars.peek() {
                if ch.is_ascii_digit() {
                    saw_digit = true;
                    value = value
                        .checked_mul(10)
                        .and_then(|v| v.checked_add((ch as u8 - b'0') as usize))
                        .ok_or_else(|| {
                            format!("dlmwrite: {label} too large in precision format")
                        })?;
                    chars.next();
                } else {
                    break;
                }
            }
            Ok(if saw_digit { Some(value) } else { None })
        }

        let mut chars = input.chars().peekable();
        match chars.next() {
            Some('%') => {}
            _ => {
                return Err("dlmwrite: precision format must start with '%'".to_string());
            }
        }

        let mut left_adjust = false;
        let mut sign = SignFlag::None;
        let mut zero_pad = false;
        let mut alternate = false;
        while let Some(&ch) = chars.peek() {
            match ch {
                '-' => {
                    left_adjust = true;
                    zero_pad = false;
                    chars.next();
                }
                '+' => {
                    sign = SignFlag::Plus;
                    chars.next();
                }
                ' ' => {
                    if sign != SignFlag::Plus {
                        sign = SignFlag::Space;
                    }
                    chars.next();
                }
                '0' => {
                    if !left_adjust {
                        zero_pad = true;
                    }
                    chars.next();
                }
                '#' => {
                    alternate = true;
                    chars.next();
                }
                _ => break,
            }
        }

        let width = parse_number(&mut chars, "field width")?;
        let precision = if matches!(chars.peek(), Some('.')) {
            chars.next();
            parse_number(&mut chars, "precision")?.or(Some(0))
        } else {
            None
        };

        if matches!(chars.peek(), Some('l' | 'L' | 'h')) {
            return Err(
                "dlmwrite: length modifiers are not supported in precision formats".to_string(),
            );
        }

        let spec_ch = chars
            .next()
            .ok_or_else(|| "dlmwrite: incomplete precision format".to_string())?;
        if chars.next().is_some() {
            return Err("dlmwrite: unexpected trailing characters in precision format".to_string());
        }

        let (specifier, uppercase) = match spec_ch {
            'f' => (FloatSpecifier::Fixed, false),
            'F' => (FloatSpecifier::Fixed, true),
            'e' => (FloatSpecifier::Exponent, false),
            'E' => (FloatSpecifier::Exponent, true),
            'g' => (FloatSpecifier::General, false),
            'G' => (FloatSpecifier::General, true),
            other => {
                return Err(format!(
                    "dlmwrite: unsupported precision format specifier '{other}'"
                ));
            }
        };

        Ok(Self {
            specifier,
            uppercase,
            alternate,
            sign,
            left_adjust,
            zero_pad: zero_pad && !left_adjust,
            width,
            precision,
        })
    }

    fn render(&self, value: f64) -> String {
        let negative = value.is_sign_negative();
        let magnitude = if negative { -value } else { value };
        let mut body = match self.specifier {
            FloatSpecifier::Fixed => {
                format_fixed_body(magnitude, self.precision.unwrap_or(6), self.alternate)
            }
            FloatSpecifier::Exponent => format_exponential_body(
                magnitude,
                self.precision.unwrap_or(6),
                self.alternate,
                self.uppercase,
            ),
            FloatSpecifier::General => format_general_body(
                magnitude,
                self.precision.unwrap_or(6),
                self.alternate,
                self.uppercase,
            ),
        };
        if self.uppercase {
            body.make_ascii_uppercase();
        }

        let mut prefix = String::new();
        if negative {
            prefix.push('-');
        } else {
            match self.sign {
                SignFlag::Plus => prefix.push('+'),
                SignFlag::Space => prefix.push(' '),
                SignFlag::None => {}
            }
        }

        let total_len = prefix.len() + body.len();
        if let Some(width) = self.width {
            if width > total_len {
                let pad = width - total_len;
                if self.left_adjust {
                    let mut result = prefix;
                    result.push_str(&body);
                    result.extend(std::iter::repeat_n(' ', pad));
                    return result;
                } else if self.zero_pad {
                    let mut result = String::with_capacity(width);
                    result.push_str(&prefix);
                    result.extend(std::iter::repeat_n('0', pad));
                    result.push_str(&body);
                    return result;
                } else {
                    let mut result = String::with_capacity(width);
                    result.extend(std::iter::repeat_n(' ', pad));
                    result.push_str(&prefix);
                    result.push_str(&body);
                    return result;
                }
            }
        }

        let mut result = prefix;
        result.push_str(&body);
        result
    }
}

#[cfg(any(target_arch = "wasm32", test))]
fn format_fixed_body(value: f64, precision: usize, alternate: bool) -> String {
    let mut s = format!("{:.*}", precision, value);
    if precision == 0 && alternate && !s.contains('.') {
        s.push('.');
    }
    s
}

#[cfg(any(target_arch = "wasm32", test))]
fn format_exponential_body(
    value: f64,
    precision: usize,
    alternate: bool,
    uppercase: bool,
) -> String {
    let mut s = format!("{:.*e}", precision, value);
    normalize_exponent_notation(&mut s);
    if uppercase {
        s.make_ascii_uppercase();
    }
    if precision == 0 && alternate {
        insert_decimal_point(&mut s);
    }
    s
}

#[cfg(any(target_arch = "wasm32", test))]
fn format_general_body(value: f64, precision: usize, alternate: bool, uppercase: bool) -> String {
    let effective_precision = precision.max(1);
    let abs_val = value.abs();
    if abs_val == 0.0 {
        return if alternate {
            let mut s = "0.".to_string();
            s.extend(std::iter::repeat_n('0', effective_precision - 1));
            s
        } else {
            "0".to_string()
        };
    }

    let exponent = abs_val.log10().floor() as i32;
    let force_exponent = uppercase && alternate;
    let use_exponent = force_exponent || exponent < -4 || exponent >= effective_precision as i32;
    let mut s = if use_exponent {
        let frac = effective_precision.saturating_sub(1);
        let mut out = format!("{:.*e}", frac, abs_val);
        normalize_exponent_notation(&mut out);
        if uppercase {
            out.make_ascii_uppercase();
        }
        out
    } else {
        let frac = {
            let diff = effective_precision as isize - (exponent + 1) as isize;
            if diff < 0 {
                0
            } else {
                diff as usize
            }
        };
        let mut out = format!("{:.*}", frac, abs_val);
        if uppercase {
            out.make_ascii_uppercase();
        }
        out
    };

    if alternate {
        insert_decimal_point(&mut s);
    } else {
        trim_trailing_zeros(&mut s);
    }
    s
}

#[cfg(any(target_arch = "wasm32", test))]
fn insert_decimal_point(s: &mut String) {
    if s.contains('.') {
        return;
    }
    if let Some(idx) = find_exponent_index(s) {
        s.insert(idx, '.');
    } else {
        s.push('.');
    }
}

#[cfg(any(target_arch = "wasm32", test))]
fn trim_trailing_zeros(s: &mut String) {
    if let Some(idx) = find_exponent_index(s) {
        let exponent = s[idx..].to_string();
        let mut mantissa = s[..idx].to_string();
        trim_fraction(&mut mantissa);
        s.clear();
        s.push_str(&mantissa);
        s.push_str(&exponent);
        normalize_exponent_notation(s);
    } else {
        trim_fraction(s);
    }
}

#[cfg(any(target_arch = "wasm32", test))]
fn trim_fraction(s: &mut String) {
    if let Some(dot_idx) = s.find('.') {
        let mut idx = s.len();
        while idx > dot_idx + 1 && matches!(s.as_bytes().get(idx - 1), Some(b'0')) {
            idx -= 1;
        }
        if idx == dot_idx + 1 {
            idx -= 1;
        }
        s.truncate(idx);
    }
}

#[cfg(any(target_arch = "wasm32", test))]
fn find_exponent_index(s: &str) -> Option<usize> {
    s.find('e').or_else(|| s.find('E'))
}

#[cfg(any(target_arch = "wasm32", test))]
fn normalize_exponent_notation(s: &mut String) {
    if let Some(idx) = find_exponent_index(s) {
        let marker = s.as_bytes()[idx] as char;
        let suffix = &s[idx + 1..];
        let (sign, digits) = if let Some(first) = suffix.chars().next() {
            if first == '+' || first == '-' {
                (first, suffix.get(1..).unwrap_or("").to_string())
            } else {
                ('+', suffix.to_string())
            }
        } else {
            ('+', String::from("0"))
        };
        let mut normalized_digits = if digits.is_empty() {
            String::from("0")
        } else {
            digits
        };
        if normalized_digits.is_empty() {
            normalized_digits.push('0');
        }
        if normalized_digits.len() < 2 {
            normalized_digits = format!("{:0>2}", normalized_digits);
        }
        let mut rebuilt = String::with_capacity(idx + 1 + 1 + normalized_digits.len());
        rebuilt.push_str(&s[..idx]);
        rebuilt.push(marker);
        rebuilt.push(sign);
        rebuilt.push_str(&normalized_digits);
        *s = rebuilt;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_time::unix_timestamp_ms;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_provider;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::IntValue;

    use crate::builtins::common::fs as fs_helpers;

    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    fn temp_path(ext: &str) -> PathBuf {
        let millis = unix_timestamp_ms();
        let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "runmat_dlmwrite_{}_{}_{}.{}",
            std::process::id(),
            millis,
            unique,
            ext
        ));
        path
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn wasm_precision_parser_handles_common_specs() {
        fn fmt(value: f64, spec: &str) -> String {
            super::wasm_format_float(value, spec).expect("formatting failed")
        }

        assert_eq!(fmt(12.3456, "%.2f"), "12.35");
        assert_eq!(fmt(-12.3456, "%+08.1f"), "-00012.3");
        assert_eq!(fmt(0.001234, "%.4g"), "0.001234");
        assert_eq!(fmt(12345.0, "%.3g"), "1.23e+04");
        assert_eq!(fmt(1.5, "%#.0f"), "2.");
        assert_eq!(fmt(1.5, "%#.2e"), "1.50e+00");
        assert_eq!(fmt(1.5, "%#.2G"), "1.5E+00");
    }

    fn platform_newline() -> &'static str {
        LineEnding::platform_default().as_str()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_writes_default_comma() {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let filename = path.to_string_lossy().into_owned();

        dlmwrite_builtin(Value::from(filename), Value::Tensor(tensor), Vec::new()).unwrap();

        let contents = fs::read_to_string(&path).unwrap();
        let nl = platform_newline();
        assert_eq!(contents, format!("1,2,3{nl}4,5,6{nl}"));
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_accepts_positional_delimiter_and_offsets() {
        let path = temp_path("txt");
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let filename = path.to_string_lossy().into_owned();
        dlmwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![
                Value::from(";"),
                Value::Int(IntValue::I32(1)),
                Value::Int(IntValue::I32(2)),
            ],
        )
        .unwrap();

        let contents = fs::read_to_string(&path).unwrap();
        let nl = platform_newline();
        assert_eq!(contents, format!(";;;;{nl};;1;2;3{nl};;4;5;6{nl}"));
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_supports_append_and_offsets() {
        let path = temp_path("csv");
        let tensor_a = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let tensor_b = Tensor::new(vec![10.0, 13.0, 11.0, 14.0, 12.0, 15.0], vec![3, 2]).unwrap();
        let filename = path.to_string_lossy().into_owned();
        dlmwrite_builtin(
            Value::from(filename.clone()),
            Value::Tensor(tensor_a),
            Vec::new(),
        )
        .unwrap();
        dlmwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor_b),
            vec![
                Value::from("-append"),
                Value::from("roffset"),
                Value::Int(IntValue::I32(1)),
                Value::from("coffset"),
                Value::Int(IntValue::I32(1)),
            ],
        )
        .unwrap();
        let contents = fs::read_to_string(&path).unwrap();
        let nl = platform_newline();
        assert_eq!(
            contents,
            format!("1,2,3{nl},,{nl},10,14{nl},13,12{nl},11,15{nl}")
        );
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_precision_digits() {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![12.34567, std::f64::consts::PI], vec![1, 2]).unwrap();
        let filename = path.to_string_lossy().into_owned();
        dlmwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![Value::from("precision"), Value::Int(IntValue::I32(3))],
        )
        .unwrap();
        let contents = fs::read_to_string(&path).unwrap();
        let nl = platform_newline();
        assert_eq!(contents, format!("12.3,3.14{nl}"));
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_precision_format_string() {
        let path = temp_path("txt");
        let tensor = Tensor::new(vec![0.25, 0.5, 0.75, 1.5], vec![2, 2]).unwrap();
        let filename = path.to_string_lossy().into_owned();
        dlmwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![
                Value::from("precision"),
                Value::from("%.4f"),
                Value::from("delimiter"),
                Value::from("\t"),
            ],
        )
        .unwrap();
        let contents = fs::read_to_string(&path).unwrap();
        let nl = platform_newline();
        assert_eq!(contents, format!("0.2500\t0.7500{nl}0.5000\t1.5000{nl}"));
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_newline_pc() {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let filename = path.to_string_lossy().into_owned();
        dlmwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![Value::from("newline"), Value::from("pc")],
        )
        .unwrap();
        let contents = fs::read_to_string(&path).unwrap();
        let nl = LineEnding::Pc.as_str();
        assert_eq!(contents, format!("1{nl}2{nl}"));
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_coffset_inserts_empty_fields() {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let filename = path.to_string_lossy().into_owned();
        dlmwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![Value::from("coffset"), Value::Int(IntValue::I32(2))],
        )
        .unwrap();
        let contents = fs::read_to_string(&path).unwrap();
        let nl = platform_newline();
        assert_eq!(contents, format!(",,1,2{nl},,3,4{nl}"));
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_handles_gpu_tensors() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).unwrap();
            let path = temp_path("csv");
            let filename = path.to_string_lossy().into_owned();
            dlmwrite_builtin(Value::from(filename), Value::GpuTensor(handle), Vec::new()).unwrap();
            let contents = fs::read_to_string(&path).unwrap();
            let nl = platform_newline();
            assert_eq!(contents, format!("1,2,3{nl}"));
            let _ = fs::remove_file(path);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn dlmwrite_handles_wgpu_provider_gather() {
        let _ =
            wgpu_provider::register_wgpu_provider(wgpu_provider::WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).unwrap();
        let path = temp_path("csv");
        let filename = path.to_string_lossy().into_owned();
        dlmwrite_builtin(Value::from(filename), Value::GpuTensor(handle), Vec::new()).unwrap();
        let contents = fs::read_to_string(&path).unwrap();
        let nl = platform_newline();
        assert_eq!(contents, format!("1,2{nl}"));
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_interprets_control_sequence_delimiters() {
        let path = temp_path("txt");
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let filename = path.to_string_lossy().into_owned();
        dlmwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![Value::from("delimiter"), Value::from("\\t")],
        )
        .unwrap();
        let contents = fs::read_to_string(&path).unwrap();
        let nl = platform_newline();
        assert_eq!(contents, format!("1\t2{nl}"));
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_rejects_negative_offsets() {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let filename = path.to_string_lossy().into_owned();
        let err = dlmwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![Value::from("roffset"), Value::Int(IntValue::I32(-1))],
        )
        .unwrap_err();
        assert!(err.to_ascii_lowercase().contains("row offset"));
        assert!(!path.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_rejects_fractional_offsets() {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let filename = path.to_string_lossy().into_owned();
        let err = dlmwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![Value::from("coffset"), Value::Num(1.5)],
        )
        .unwrap_err();
        assert!(err.to_ascii_lowercase().contains("integer"));
        assert!(!path.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_rejects_empty_delimiter() {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let filename = path.to_string_lossy().into_owned();
        let err = dlmwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![Value::from("")],
        )
        .unwrap_err();
        assert!(err.to_ascii_lowercase().contains("delimiter"));
        assert!(!path.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_precision_zero_error() {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let filename = path.to_string_lossy().into_owned();
        let err = dlmwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![Value::from("precision"), Value::Int(IntValue::I32(0))],
        )
        .unwrap_err();
        assert!(err.to_ascii_lowercase().contains("precision"));
        assert!(!path.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_requires_name_value_pairs() {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let filename = path.to_string_lossy().into_owned();
        let err = dlmwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![Value::from("delimiter")],
        )
        .unwrap_err();
        assert!(err
            .to_ascii_lowercase()
            .contains("name-value arguments must appear in pairs"));
        assert!(!path.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_expands_home_directory() {
        let Some(mut home) = fs_helpers::home_directory() else {
            return;
        };
        let filename = format!(
            "runmat_dlmwrite_home_{}_{}.csv",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        );
        home.push(&filename);

        let tilde_path = format!("~/{}", filename);
        let tensor = Tensor::new(vec![42.0], vec![1, 1]).unwrap();

        dlmwrite_builtin(Value::from(tilde_path), Value::Tensor(tensor), Vec::new()).unwrap();

        let contents = fs::read_to_string(&home).unwrap();
        let nl = platform_newline();
        assert_eq!(contents, format!("42{nl}"));
        let _ = fs::remove_file(home);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dlmwrite_rejects_non_numeric_inputs() {
        let path = temp_path("csv");
        let filename = path.to_string_lossy().into_owned();
        let err = dlmwrite_builtin(
            Value::from(filename),
            Value::String("abc".into()),
            Vec::new(),
        )
        .unwrap_err();
        assert!(err.contains("dlmwrite"));
    }
}
