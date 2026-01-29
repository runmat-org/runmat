//! MATLAB-compatible `filewrite` builtin for RunMat.

use std::io::Write;
use std::path::{Path, PathBuf};

use runmat_builtins::{CharArray, IntValue, LogicalArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};
use runmat_filesystem::OpenOptions;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::filetext::filewrite")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "filewrite",
    op_kind: GpuOpKind::Custom("io-file-write"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Performs synchronous host file I/O; GPU providers do not participate.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::filetext::filewrite")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "filewrite",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Standalone host-side operation; never fused with other kernels.",
};

const BUILTIN_NAME: &str = "filewrite";

fn filewrite_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(str::to_string);
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_control_flow_with_context(err: RuntimeError, context: &str) -> RuntimeError {
    let identifier = err.identifier().map(str::to_string);
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {context}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileEncoding {
    Auto,
    Utf8,
    Ascii,
    Latin1,
    Raw,
}

impl FileEncoding {
    fn from_label(label: &str) -> Option<Self> {
        match label.trim().to_ascii_lowercase().as_str() {
            "auto" | "default" | "system" | "native" => Some(FileEncoding::Auto),
            "utf-8" | "utf8" | "utf_8" | "unicode" => Some(FileEncoding::Utf8),
            "ascii" | "us-ascii" | "us_ascii" | "usascii" => Some(FileEncoding::Ascii),
            "latin1" | "latin-1" | "latin_1" | "iso-8859-1" | "iso8859-1" | "iso88591" => {
                Some(FileEncoding::Latin1)
            }
            "raw" | "bytes" | "byte" | "binary" => Some(FileEncoding::Raw),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriteMode {
    Overwrite,
    Append,
}

impl WriteMode {
    fn from_label(label: &str) -> Option<Self> {
        match label.trim().to_ascii_lowercase().as_str() {
            "overwrite" | "replace" | "truncate" => Some(WriteMode::Overwrite),
            "append" | "add" => Some(WriteMode::Append),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FilewriteOptions {
    encoding: FileEncoding,
    write_mode: WriteMode,
}

impl Default for FilewriteOptions {
    fn default() -> Self {
        Self {
            encoding: FileEncoding::Auto,
            write_mode: WriteMode::Overwrite,
        }
    }
}

#[runtime_builtin(
    name = "filewrite",
    category = "io/filetext",
    summary = "Write text or raw bytes to a file.",
    keywords = "filewrite,io,write file,text file,append,encoding",
    accel = "cpu",
    builtin_path = "crate::builtins::io::filetext::filewrite"
)]
async fn filewrite_builtin(
    path: Value,
    data: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let path = gather_if_needed_async(&path)
        .await
        .map_err(map_control_flow)?;
    let data = gather_if_needed_async(&data)
        .await
        .map_err(map_control_flow)?;
    let rest = gather_values(&rest).await?;
    let options = parse_options(&rest)?;
    let resolved = resolve_path(&path)?;
    let payload = prepare_payload(&data, options.encoding)?;
    let written = write_bytes(&resolved, &payload, options.write_mode)?;
    Ok(Value::Num(written as f64))
}

async fn gather_values(values: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(
            gather_if_needed_async(value)
                .await
                .map_err(map_control_flow)?,
        );
    }
    Ok(out)
}

fn parse_options(args: &[Value]) -> BuiltinResult<FilewriteOptions> {
    if args.is_empty() {
        return Ok(FilewriteOptions::default());
    }

    let mut options = FilewriteOptions::default();
    let mut idx = 0usize;
    let mut encoding_specified = false;
    let mut write_mode_specified = false;

    if !args.is_empty() && !is_keyword(&args[0])? {
        match encoding_from_value(&args[0]) {
            Ok(enc) => {
                options.encoding = enc;
                encoding_specified = true;
                idx = 1;
            }
            Err(err) => {
                if args.len() == 1 {
                    return Err(err);
                }
            }
        }
    }

    if !(args.len() - idx).is_multiple_of(2) {
        return Err(filewrite_error(
            "filewrite: expected keyword/value argument pairs",
        ));
    }

    while idx < args.len() {
        let key = keyword_name(&args[idx])?;
        let value = &args[idx + 1];
        if key.eq_ignore_ascii_case("encoding") {
            if encoding_specified {
                return Err(filewrite_error("filewrite: duplicate 'Encoding' argument"));
            }
            options.encoding = encoding_from_value(value)?;
            encoding_specified = true;
        } else if key.eq_ignore_ascii_case("writemode") {
            if write_mode_specified {
                return Err(filewrite_error("filewrite: duplicate 'WriteMode' argument"));
            }
            options.write_mode = write_mode_from_value(value)?;
            write_mode_specified = true;
        } else {
            return Err(filewrite_error(format!(
                "filewrite: unrecognised option '{}'",
                key
            )));
        }
        idx += 2;
    }

    Ok(options)
}

fn is_keyword(value: &Value) -> BuiltinResult<bool> {
    let text = keyword_name(value)?;
    Ok(text.eq_ignore_ascii_case("encoding") || text.eq_ignore_ascii_case("writemode"))
}

fn keyword_name(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::CharArray(_) => Err(filewrite_error(
            "filewrite: keyword names must be 1-by-N character vectors",
        )),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        Value::StringArray(_) => Err(filewrite_error(
            "filewrite: keyword inputs must be scalar string arrays",
        )),
        other => Err(filewrite_error(format!(
            "filewrite: expected keyword as string scalar or character vector, got {other:?}"
        ))),
    }
}

fn encoding_from_value(value: &Value) -> BuiltinResult<FileEncoding> {
    let label = keyword_name(value)?;
    match FileEncoding::from_label(&label) {
        Some(enc) => Ok(enc),
        None if label.trim().is_empty() => Err(filewrite_error(
            "filewrite: encoding name must not be empty",
        )),
        None => Err(filewrite_error(format!(
            "filewrite: unsupported encoding '{}'",
            label
        ))),
    }
}

fn write_mode_from_value(value: &Value) -> BuiltinResult<WriteMode> {
    let label = keyword_name(value)?;
    match WriteMode::from_label(&label) {
        Some(mode) => Ok(mode),
        None if label.trim().is_empty() => {
            Err(filewrite_error("filewrite: write mode must not be empty"))
        }
        None => Err(filewrite_error(format!(
            "filewrite: unsupported write mode '{}'; use 'overwrite' or 'append'",
            label
        ))),
    }
}

fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
    match value {
        Value::String(s) => normalize_path(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let path: String = ca.data.iter().collect();
            normalize_path(&path)
        }
        Value::CharArray(_) => Err(filewrite_error(
            "filewrite: filename must be a 1-by-N character vector",
        )),
        Value::StringArray(sa) if sa.data.len() == 1 => normalize_path(&sa.data[0]),
        Value::StringArray(_) => Err(filewrite_error(
            "filewrite: string array filename inputs must be scalar",
        )),
        other => Err(filewrite_error(format!(
            "filewrite: expected filename as string scalar or character vector, got {other:?}"
        ))),
    }
}

fn normalize_path(raw: &str) -> BuiltinResult<PathBuf> {
    if raw.is_empty() {
        return Err(filewrite_error("filewrite: filename must not be empty"));
    }
    Ok(Path::new(raw).to_path_buf())
}

enum Payload {
    Text(Vec<char>),
    Bytes(Vec<u8>),
}

fn prepare_payload(data: &Value, encoding: FileEncoding) -> BuiltinResult<Vec<u8>> {
    let payload = extract_payload(data)?;
    match payload {
        Payload::Text(chars) => encode_text(chars, encoding),
        Payload::Bytes(bytes) => encode_bytes(bytes, encoding),
    }
}

fn extract_payload(data: &Value) -> BuiltinResult<Payload> {
    match data {
        Value::CharArray(ca) => Ok(Payload::Text(char_array_to_text(ca))),
        Value::String(s) => Ok(Payload::Text(s.chars().collect())),
        Value::StringArray(sa) => Ok(Payload::Text(string_array_to_text(sa))),
        Value::Num(n) => {
            let byte = float_to_byte(*n)
                .map_err(|err| map_control_flow_with_context(err, "filewrite: numeric value"))?;
            Ok(Payload::Bytes(vec![byte]))
        }
        Value::Int(i) => {
            let byte = int_value_to_byte(i)
                .map_err(|err| map_control_flow_with_context(err, "filewrite: integer value"))?;
            Ok(Payload::Bytes(vec![byte]))
        }
        Value::Bool(flag) => Ok(Payload::Bytes(vec![if *flag { 1 } else { 0 }])),
        Value::Tensor(t) => Ok(Payload::Bytes(tensor_to_bytes(t)?)),
        Value::LogicalArray(la) => Ok(Payload::Bytes(logical_to_bytes(la))),
        Value::Cell(_) => Err(filewrite_error(
            "filewrite: cell arrays are not supported inputs",
        )),
        Value::GpuTensor(_) => Err(filewrite_error(
            "filewrite: internal error: GPU tensor should be gathered",
        )),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(filewrite_error(
            "filewrite: complex data must be converted to text before writing",
        )),
        other => Err(filewrite_error(format!(
            "filewrite: unsupported data type {other:?}; expected text or uint8-compatible array"
        ))),
    }
}

fn char_array_to_text(ca: &CharArray) -> Vec<char> {
    if ca.rows <= 1 {
        return ca.data.clone();
    }
    let mut out = Vec::with_capacity(ca.rows * (ca.cols + 1));
    for row in 0..ca.rows {
        for col in 0..ca.cols {
            out.push(ca.data[row * ca.cols + col]);
        }
        if row + 1 < ca.rows {
            out.push('\n');
        }
    }
    out
}

fn string_array_to_text(sa: &StringArray) -> Vec<char> {
    if sa.data.is_empty() {
        return Vec::new();
    }
    let mut combined = String::new();
    for (idx, entry) in sa.data.iter().enumerate() {
        combined.push_str(entry);
        if idx + 1 < sa.data.len() {
            combined.push('\n');
        }
    }
    combined.chars().collect()
}

fn tensor_to_bytes(tensor: &Tensor) -> BuiltinResult<Vec<u8>> {
    let mut out = Vec::with_capacity(tensor.data.len());
    for (idx, value) in tensor.data.iter().enumerate() {
        match float_to_byte(*value) {
            Ok(byte) => out.push(byte),
            Err(msg) => {
                return Err(filewrite_error(format!(
                    "filewrite: numeric element {} {msg}",
                    idx
                )));
            }
        }
    }
    Ok(out)
}

fn logical_to_bytes(array: &LogicalArray) -> Vec<u8> {
    array
        .data
        .iter()
        .map(|value| if *value != 0 { 1 } else { 0 })
        .collect()
}

fn float_to_byte(value: f64) -> BuiltinResult<u8> {
    if !value.is_finite() {
        return Err(filewrite_error(format!(
            "value {value} is not finite; cannot write as raw byte"
        )));
    }
    let rounded = value.round();
    if (value - rounded).abs() > 1e-9 {
        return Err(filewrite_error(format!(
            "value {value} is not an integer in the range 0..255"
        )));
    }
    let int = rounded as i128;
    if !(0..=255).contains(&int) {
        return Err(filewrite_error(format!(
            "value {value} is not in the range 0..255"
        )));
    }
    Ok(int as u8)
}

fn int_value_to_byte(value: &IntValue) -> BuiltinResult<u8> {
    match value {
        IntValue::I8(v) => signed_to_byte(*v as i64),
        IntValue::I16(v) => signed_to_byte(*v as i64),
        IntValue::I32(v) => signed_to_byte(*v as i64),
        IntValue::I64(v) => signed_to_byte(*v),
        IntValue::U8(v) => Ok(*v),
        IntValue::U16(v) => unsigned_to_byte(*v as u64),
        IntValue::U32(v) => unsigned_to_byte(*v as u64),
        IntValue::U64(v) => unsigned_to_byte(*v),
    }
}

fn signed_to_byte(value: i64) -> BuiltinResult<u8> {
    if !(0..=255).contains(&value) {
        return Err(filewrite_error(format!(
            "value {value} is not in the range 0..255"
        )));
    }
    Ok(value as u8)
}

fn unsigned_to_byte(value: u64) -> BuiltinResult<u8> {
    if value > 255 {
        return Err(filewrite_error(format!(
            "value {value} is not in the range 0..255"
        )));
    }
    Ok(value as u8)
}

fn encode_text(chars: Vec<char>, encoding: FileEncoding) -> BuiltinResult<Vec<u8>> {
    match encoding {
        FileEncoding::Auto | FileEncoding::Utf8 => {
            Ok(chars.iter().collect::<String>().into_bytes())
        }
        FileEncoding::Ascii => encode_ascii_chars(&chars),
        FileEncoding::Latin1 | FileEncoding::Raw => encode_latin_chars(&chars, encoding),
    }
}

fn encode_ascii_chars(chars: &[char]) -> BuiltinResult<Vec<u8>> {
    let mut out = Vec::with_capacity(chars.len());
    for &ch in chars {
        if ch as u32 > 0x7F {
            return Err(filewrite_error(format!(
                "filewrite: character '{}' (U+{:04X}) cannot be encoded as ASCII",
                ch, ch as u32
            )));
        }
        out.push(ch as u8);
    }
    Ok(out)
}

fn encode_latin_chars(chars: &[char], encoding: FileEncoding) -> BuiltinResult<Vec<u8>> {
    let mut out = Vec::with_capacity(chars.len());
    for &ch in chars {
        if ch as u32 > 0xFF {
            return Err(filewrite_error(format!(
                "filewrite: character '{}' (U+{:04X}) cannot be encoded as {}",
                ch,
                ch as u32,
                match encoding {
                    FileEncoding::Latin1 => "Latin-1",
                    FileEncoding::Raw => "raw bytes",
                    _ => unreachable!(),
                }
            )));
        }
        out.push(ch as u8);
    }
    Ok(out)
}

fn encode_bytes(bytes: Vec<u8>, encoding: FileEncoding) -> BuiltinResult<Vec<u8>> {
    if matches!(encoding, FileEncoding::Ascii) {
        for (idx, byte) in bytes.iter().enumerate() {
            if *byte > 0x7F {
                return Err(filewrite_error(format!(
                    "filewrite: byte 0x{byte:02X} at index {idx} cannot be encoded as ASCII"
                )));
            }
        }
    }
    Ok(bytes)
}

fn write_bytes(path: &Path, payload: &[u8], mode: WriteMode) -> BuiltinResult<usize> {
    let mut options = OpenOptions::new();
    options.create(true);
    match mode {
        WriteMode::Overwrite => {
            options.write(true).truncate(true);
        }
        WriteMode::Append => {
            options.write(true).append(true);
        }
    }

    let mut file = options.open(path).map_err(|err| {
        build_runtime_error(format!(
            "filewrite: unable to open '{}': {}",
            path.display(),
            err
        ))
        .with_builtin(BUILTIN_NAME)
        .with_source(err)
        .build()
    })?;

    file.write_all(payload).map_err(|err| {
        build_runtime_error(format!(
            "filewrite: unable to write to '{}': {}",
            path.display(),
            err
        ))
        .with_builtin(BUILTIN_NAME)
        .with_source(err)
        .build()
    })?;

    file.flush().map_err(|err| {
        build_runtime_error(format!(
            "filewrite: unable to flush '{}': {}",
            path.display(),
            err
        ))
        .with_builtin(BUILTIN_NAME)
        .with_source(err)
        .build()
    })?;

    Ok(payload.len())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::RuntimeError;
    use runmat_filesystem as fs;
    use runmat_time::unix_timestamp_ms;
    use std::io::Read;

    fn unwrap_error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    fn run_filewrite(path: Value, data: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(filewrite_builtin(path, data, rest))
    }

    fn unique_path(prefix: &str) -> PathBuf {
        let millis = unix_timestamp_ms();
        let mut path = std::env::temp_dir();
        path.push(format!("runmat_{prefix}_{}_{}", std::process::id(), millis));
        path
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filewrite_writes_text_content() {
        let path = unique_path("filewrite_text");
        let contents = "RunMat filewrite\nLine two\n";

        let result = run_filewrite(
            Value::from(path.to_string_lossy().to_string()),
            Value::from(contents),
            Vec::new(),
        )
        .expect("filewrite");

        match result {
            Value::Num(n) => assert_eq!(n as usize, contents.len()),
            other => panic!("expected numeric byte count, got {other:?}"),
        }

        let written = fs::read_to_string(&path).expect("read filewrite output");
        assert_eq!(written, contents);

        let _ = fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filewrite_appends_when_requested() {
        let path = unique_path("filewrite_append");
        fs::write(&path, "first\n").expect("write baseline");

        run_filewrite(
            Value::from(path.to_string_lossy().to_string()),
            Value::from("second\n"),
            vec![Value::from("WriteMode"), Value::from("append")],
        )
        .expect("filewrite append");

        let written = fs::read_to_string(&path).expect("read appended file");
        assert_eq!(written, "first\nsecond\n");

        let _ = fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filewrite_errors_on_invalid_ascii() {
        let path = unique_path("filewrite_ascii_error");
        let err = unwrap_error_message(
            run_filewrite(
                Value::from(path.to_string_lossy().to_string()),
                Value::from("café"),
                vec![Value::from("Encoding"), Value::from("ascii")],
            )
            .unwrap_err(),
        );
        assert!(
            err.contains("cannot be encoded as ASCII"),
            "unexpected error message: {err}"
        );
        assert!(!path.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filewrite_writes_raw_bytes_from_tensor() {
        let path = unique_path("filewrite_raw_bytes");
        let tensor = Tensor::new(vec![0.0, 127.0, 255.0], vec![3, 1]).expect("tensor");
        run_filewrite(
            Value::from(path.to_string_lossy().to_string()),
            Value::Tensor(tensor),
            vec![Value::from("Encoding"), Value::from("raw")],
        )
        .expect("filewrite raw");

        let mut bytes = Vec::new();
        fs::File::open(&path)
            .expect("open raw file")
            .read_to_end(&mut bytes)
            .expect("read raw file");
        assert_eq!(bytes, vec![0u8, 127u8, 255u8]);

        let _ = fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filewrite_numeric_scalar_writes_byte() {
        let path = unique_path("filewrite_numeric_scalar");
        run_filewrite(
            Value::from(path.to_string_lossy().to_string()),
            Value::Num(65.0),
            Vec::new(),
        )
        .expect("filewrite numeric scalar");

        let bytes = fs::read(&path).expect("read numeric scalar file");
        assert_eq!(bytes, vec![65u8]);

        let _ = fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filewrite_bool_scalar_writes_byte() {
        let path = unique_path("filewrite_bool_scalar");
        run_filewrite(
            Value::from(path.to_string_lossy().to_string()),
            Value::Bool(true),
            Vec::new(),
        )
        .expect("filewrite bool scalar");

        let bytes = fs::read(&path).expect("read bool scalar file");
        assert_eq!(bytes, vec![1u8]);

        let _ = fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filewrite_writes_logical_array_bytes() {
        let path = unique_path("filewrite_logical_array");
        let logical = LogicalArray::new(vec![0, 1, 2], vec![3]).expect("logical array");
        run_filewrite(
            Value::from(path.to_string_lossy().to_string()),
            Value::LogicalArray(logical),
            Vec::new(),
        )
        .expect("filewrite logical array");

        let bytes = fs::read(&path).expect("read logical array file");
        assert_eq!(bytes, vec![0u8, 1u8, 1u8]);

        let _ = fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filewrite_errors_on_numeric_out_of_range() {
        let path = unique_path("filewrite_out_of_range");
        let err = unwrap_error_message(
            run_filewrite(
                Value::from(path.to_string_lossy().to_string()),
                Value::Num(300.0),
                Vec::new(),
            )
            .unwrap_err(),
        );
        assert!(
            err.contains("range 0..255"),
            "unexpected error message: {err}"
        );
        assert!(!path.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filewrite_errors_on_non_integer_numeric() {
        let path = unique_path("filewrite_non_integer");
        let err = unwrap_error_message(
            run_filewrite(
                Value::from(path.to_string_lossy().to_string()),
                Value::Num(std::f64::consts::PI),
                Vec::new(),
            )
            .unwrap_err(),
        );
        assert!(
            err.contains("not an integer"),
            "unexpected error message: {err}"
        );
        assert!(!path.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filewrite_rejects_ascii_numeric_bytes_above_range() {
        let path = unique_path("filewrite_ascii_bytes");
        let tensor = Tensor::new(vec![255.0], vec![1, 1]).expect("tensor");
        let err = unwrap_error_message(
            run_filewrite(
                Value::from(path.to_string_lossy().to_string()),
                Value::Tensor(tensor),
                vec![Value::from("Encoding"), Value::from("ascii")],
            )
            .unwrap_err(),
        );
        assert!(
            err.contains("cannot be encoded as ASCII"),
            "unexpected error message: {err}"
        );
        assert!(!path.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filewrite_positional_encoding_argument() {
        let path = unique_path("filewrite_positional_encoding");
        run_filewrite(
            Value::from(path.to_string_lossy().to_string()),
            Value::from("Espa\u{00F1}a"),
            vec![Value::from("latin1")],
        )
        .expect("filewrite positional encoding");

        let bytes = fs::read(&path).expect("read latin1 file");
        assert_eq!(bytes, b"Espa\xF1a");

        let _ = fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filewrite_utf8_encoding_allows_arbitrary_bytes() {
        let path = unique_path("filewrite_utf8_numeric");
        let tensor = Tensor::new(vec![0.0, 255.0], vec![2, 1]).expect("tensor");
        run_filewrite(
            Value::from(path.to_string_lossy().to_string()),
            Value::Tensor(tensor),
            vec![Value::from("Encoding"), Value::from("utf-8")],
        )
        .expect("filewrite utf8 numeric");

        let bytes = fs::read(&path).expect("read utf8 numeric file");
        assert_eq!(bytes, vec![0u8, 255u8]);

        let _ = fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filewrite_rejects_unknown_option() {
        let path = unique_path("filewrite_unknown_option");
        let err = unwrap_error_message(
            run_filewrite(
                Value::from(path.to_string_lossy().to_string()),
                Value::from("data"),
                vec![Value::from("Mode"), Value::from("append")],
            )
            .unwrap_err(),
        );
        assert!(
            err.contains("unrecognised option"),
            "unexpected error message: {err}"
        );
        assert!(!path.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filewrite_rejects_duplicate_encoding() {
        let path = unique_path("filewrite_duplicate_encoding");
        let err = unwrap_error_message(
            run_filewrite(
                Value::from(path.to_string_lossy().to_string()),
                Value::from("data"),
                vec![
                    Value::from("utf-8"),
                    Value::from("Encoding"),
                    Value::from("ascii"),
                ],
            )
            .unwrap_err(),
        );
        assert!(
            err.contains("duplicate 'Encoding'"),
            "unexpected error message: {err}"
        );
        assert!(!path.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filewrite_rejects_duplicate_writemode() {
        let path = unique_path("filewrite_duplicate_writemode");
        let err = unwrap_error_message(
            run_filewrite(
                Value::from(path.to_string_lossy().to_string()),
                Value::from("data"),
                vec![
                    Value::from("WriteMode"),
                    Value::from("append"),
                    Value::from("WriteMode"),
                    Value::from("overwrite"),
                ],
            )
            .unwrap_err(),
        );
        assert!(
            err.contains("duplicate 'WriteMode'"),
            "unexpected error message: {err}"
        );
        assert!(!path.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filewrite_rejects_invalid_writemode_value() {
        let path = unique_path("filewrite_invalid_writemode");
        let err = unwrap_error_message(
            run_filewrite(
                Value::from(path.to_string_lossy().to_string()),
                Value::from("data"),
                vec![Value::from("WriteMode"), Value::from("invalid")],
            )
            .unwrap_err(),
        );
        assert!(
            err.contains("unsupported write mode"),
            "unexpected error message: {err}"
        );
        assert!(!path.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filewrite_rejects_invalid_encoding_value() {
        let path = unique_path("filewrite_invalid_encoding");
        let err = unwrap_error_message(
            run_filewrite(
                Value::from(path.to_string_lossy().to_string()),
                Value::from("data"),
                vec![Value::from("Encoding"), Value::from("utf-32")],
            )
            .unwrap_err(),
        );
        assert!(
            err.contains("unsupported encoding"),
            "unexpected error message: {err}"
        );
        assert!(!path.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filewrite_accepts_char_array_filename() {
        let path = unique_path("filewrite_char_path");
        let path_str = path.to_string_lossy();
        let chars: Vec<char> = path_str.chars().collect();
        let char_array = CharArray::new(chars, 1, path_str.len()).expect("char array path");

        run_filewrite(
            Value::CharArray(char_array),
            Value::from("hello"),
            Vec::new(),
        )
        .expect("filewrite char path");

        let written = fs::read_to_string(&path).expect("read char path file");
        assert_eq!(written, "hello");

        let _ = fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn filewrite_string_array_stores_newlines() {
        let path = unique_path("filewrite_string_array");
        let array = StringArray::new(vec!["a".into(), "b".into(), "c".into()], vec![3, 1])
            .expect("string array");
        run_filewrite(
            Value::from(path.to_string_lossy().to_string()),
            Value::StringArray(array),
            Vec::new(),
        )
        .expect("filewrite string array");

        let written = fs::read_to_string(&path).expect("read string array file");
        assert_eq!(written, "a\nb\nc");

        let _ = fs::remove_file(&path);
    }
}
