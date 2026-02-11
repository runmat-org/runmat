//! MATLAB-compatible `fgets` builtin for RunMat.

use std::io::{Read, Seek, SeekFrom};

use encoding_rs::{Encoding, UTF_8};
use runmat_builtins::{CharArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::io::filetext::registry;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};
use runmat_filesystem::File;

const INVALID_IDENTIFIER_MESSAGE: &str =
    "Invalid file identifier. Use fopen to generate a valid file ID.";
const BUILTIN_NAME: &str = "fgets";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::filetext::fgets")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fgets",
    op_kind: GpuOpKind::Custom("file-io"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host-only file I/O; arguments gathered from the GPU when necessary.",
};

fn fgets_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let message = err.message().to_string();
    let identifier = err.identifier().map(|value| value.to_string());
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {message}"))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::filetext::fgets")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fgets",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "File I/O calls are not eligible for fusion.",
};

#[runtime_builtin(
    name = "fgets",
    category = "io/filetext",
    summary = "Read the next line from a file, including newline characters.",
    keywords = "fgets,file,io,line,newline",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::fgets_type),
    builtin_path = "crate::builtins::io::filetext::fgets"
)]
async fn fgets_builtin(fid: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(&fid, &rest).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            eval.outputs(),
        ));
    }
    Ok(eval.first_output())
}

#[derive(Clone, Debug)]
pub struct FgetsEval {
    line: Value,
    terminators: Value,
}

impl FgetsEval {
    fn new(line: Value, terminators: Value) -> Self {
        Self { line, terminators }
    }

    fn end_of_file() -> Self {
        Self {
            line: Value::Num(-1.0),
            terminators: Value::Num(-1.0),
        }
    }

    pub fn first_output(&self) -> Value {
        self.line.clone()
    }

    pub fn outputs(&self) -> Vec<Value> {
        vec![self.line.clone(), self.terminators.clone()]
    }
}

pub async fn evaluate(fid_value: &Value, rest: &[Value]) -> BuiltinResult<FgetsEval> {
    if rest.len() > 1 {
        return Err(fgets_error("fgets: too many input arguments"));
    }

    let fid_host = gather_value(fid_value).await?;
    let fid = parse_fid(&fid_host)?;
    if fid < 0 {
        return Err(fgets_error("fgets: file identifier must be non-negative"));
    }
    if fid < 3 {
        return Err(fgets_error(
            "fgets: standard input/output identifiers are not supported yet",
        ));
    }

    let info = registry::info_for(fid)
        .ok_or_else(|| fgets_error(format!("fgets: {INVALID_IDENTIFIER_MESSAGE}")))?;
    if !permission_allows_read(&info.permission) {
        return Err(fgets_error(
            "fgets: file identifier is not open for reading",
        ));
    }
    let handle = registry::take_handle(fid)
        .ok_or_else(|| fgets_error(format!("fgets: {INVALID_IDENTIFIER_MESSAGE}")))?;
    let mut file = handle
        .lock()
        .map_err(|_| fgets_error("fgets: failed to lock file handle (poisoned mutex)"))?;

    let limit = parse_nchar(rest).await?;
    let read = read_line(&mut file, limit)?;
    if read.eof_before_any {
        return Ok(FgetsEval::end_of_file());
    }

    let encoding = if info.encoding.trim().is_empty() {
        "UTF-8".to_string()
    } else {
        info.encoding.clone()
    };

    let line_value = bytes_to_char_array(&read.data, &encoding)?;
    let terminators_value = if read.terminators.is_empty() {
        empty_numeric_row()
    } else {
        numeric_row(&read.terminators)?
    };

    Ok(FgetsEval::new(line_value, terminators_value))
}

async fn gather_value(value: &Value) -> BuiltinResult<Value> {
    gather_if_needed_async(value)
        .await
        .map_err(map_control_flow)
}

fn parse_fid(value: &Value) -> BuiltinResult<i32> {
    match value {
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(fgets_error("fgets: file identifier must be finite"));
            }
            if (n.fract()).abs() > f64::EPSILON {
                return Err(fgets_error(
                    "fgets: file identifier must be an integer scalar",
                ));
            }
            Ok(*n as i32)
        }
        Value::Int(i) => Ok(i.to_i64() as i32),
        Value::Tensor(t) if t.data.len() == 1 => {
            let n = t.data[0];
            if !n.is_finite() {
                return Err(fgets_error("fgets: file identifier must be finite"));
            }
            if (n.fract()).abs() > f64::EPSILON {
                return Err(fgets_error(
                    "fgets: file identifier must be an integer scalar",
                ));
            }
            Ok(n as i32)
        }
        _ => Err(fgets_error(
            "fgets: file identifier must be a numeric scalar",
        )),
    }
}

async fn parse_nchar(args: &[Value]) -> BuiltinResult<Option<usize>> {
    if args.is_empty() {
        return Ok(None);
    }
    let value = gather_value(&args[0]).await?;
    match value {
        Value::Num(n) => {
            if !n.is_finite() {
                if n.is_sign_positive() {
                    return Ok(None);
                }
                return Err(fgets_error(
                    "fgets: nchar must be a non-negative integer scalar",
                ));
            }
            if n < 0.0 {
                return Err(fgets_error(
                    "fgets: nchar must be a non-negative integer scalar",
                ));
            }
            if (n.fract()).abs() > f64::EPSILON {
                return Err(fgets_error(
                    "fgets: nchar must be a non-negative integer scalar",
                ));
            }
            Ok(Some(n as usize))
        }
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < 0 {
                return Err(fgets_error(
                    "fgets: nchar must be a non-negative integer scalar",
                ));
            }
            Ok(Some(raw as usize))
        }
        Value::Tensor(t) if t.data.len() == 1 => {
            let n = t.data[0];
            if !n.is_finite() {
                if n.is_sign_positive() {
                    return Ok(None);
                }
                return Err(fgets_error(
                    "fgets: nchar must be a non-negative integer scalar",
                ));
            }
            if n < 0.0 {
                return Err(fgets_error(
                    "fgets: nchar must be a non-negative integer scalar",
                ));
            }
            if (n.fract()).abs() > f64::EPSILON {
                return Err(fgets_error(
                    "fgets: nchar must be a non-negative integer scalar",
                ));
            }
            Ok(Some(n as usize))
        }
        _ => Err(fgets_error(
            "fgets: nchar must be a non-negative integer scalar",
        )),
    }
}

fn permission_allows_read(permission: &str) -> bool {
    let lower = permission.to_ascii_lowercase();
    lower.starts_with('r') || lower.contains('+')
}

struct LineRead {
    data: Vec<u8>,
    terminators: Vec<u8>,
    eof_before_any: bool,
}

fn read_line(file: &mut File, limit: Option<usize>) -> BuiltinResult<LineRead> {
    let mut data = Vec::new();
    let mut terminators = Vec::new();
    let mut eof_before_any = false;

    let max_bytes = limit.unwrap_or(usize::MAX);
    if max_bytes == 0 {
        return Ok(LineRead {
            data,
            terminators,
            eof_before_any,
        });
    }

    let mut first_attempt = true;
    let mut buffer = [0u8; 1];
    loop {
        if data.len() >= max_bytes {
            break;
        }

        let read = file.read(&mut buffer).map_err(|err| {
            build_runtime_error(format!("fgets: failed to read from file: {err}"))
                .with_builtin(BUILTIN_NAME)
                .with_source(err)
                .build()
        })?;
        if read == 0 {
            if data.is_empty() && first_attempt {
                eof_before_any = true;
            }
            break;
        }
        first_attempt = false;
        let byte = buffer[0];

        if byte == b'\n' {
            if data.len().saturating_add(1) > max_bytes {
                file.seek(SeekFrom::Current(-1)).map_err(|err| {
                    build_runtime_error(format!("fgets: failed to seek in file: {err}"))
                        .with_builtin(BUILTIN_NAME)
                        .with_source(err)
                        .build()
                })?;
            } else {
                data.push(b'\n');
                terminators.push(b'\n');
            }
            break;
        } else if byte == b'\r' {
            let mut newline = [0u8; 2];
            newline[0] = b'\r';
            let mut newline_len = 1usize;
            let mut consumed = 1i64;

            let mut next = [0u8; 1];
            let read_next = file.read(&mut next).map_err(|err| {
                build_runtime_error(format!("fgets: failed to read from file: {err}"))
                    .with_builtin(BUILTIN_NAME)
                    .with_source(err)
                    .build()
            })?;
            if read_next > 0 {
                if next[0] == b'\n' {
                    newline[1] = b'\n';
                    newline_len = 2;
                    consumed = 2;
                } else {
                    file.seek(SeekFrom::Current(-1)).map_err(|err| {
                        build_runtime_error(format!("fgets: failed to seek in file: {err}"))
                            .with_builtin(BUILTIN_NAME)
                            .with_source(err)
                            .build()
                    })?;
                }
            }

            if data.len().saturating_add(newline_len) > max_bytes {
                file.seek(SeekFrom::Current(-consumed)).map_err(|err| {
                    build_runtime_error(format!("fgets: failed to seek in file: {err}"))
                        .with_builtin(BUILTIN_NAME)
                        .with_source(err)
                        .build()
                })?;
            } else {
                data.extend_from_slice(&newline[..newline_len]);
                terminators.extend_from_slice(&newline[..newline_len]);
            }
            break;
        } else {
            data.push(byte);
        }
    }

    Ok(LineRead {
        data,
        terminators,
        eof_before_any,
    })
}

fn bytes_to_char_array(bytes: &[u8], encoding: &str) -> BuiltinResult<Value> {
    let chars = decode_bytes(bytes, encoding)?;
    let cols = chars.len();
    let char_array = CharArray::new(chars, 1, cols)
        .map_err(|e| fgets_error(format!("fgets: failed to build char array: {e}")))?;
    Ok(Value::CharArray(char_array))
}

fn decode_bytes(bytes: &[u8], encoding: &str) -> BuiltinResult<Vec<char>> {
    let label = encoding.trim();
    if label.is_empty() || label.eq_ignore_ascii_case("utf-8") || label.eq_ignore_ascii_case("utf8")
    {
        return decode_with_encoding(bytes, UTF_8);
    }
    if label.eq_ignore_ascii_case("binary") {
        return Ok(bytes
            .iter()
            .map(|&b| char::from_u32(b as u32).unwrap())
            .collect());
    }
    if label.eq_ignore_ascii_case("latin1")
        || label.eq_ignore_ascii_case("latin-1")
        || label.eq_ignore_ascii_case("iso-8859-1")
    {
        return Ok(bytes
            .iter()
            .map(|&b| char::from_u32(b as u32).unwrap())
            .collect());
    }
    if label.eq_ignore_ascii_case("windows-1252") || label.eq_ignore_ascii_case("cp1252") {
        return decode_with_encoding(bytes, encoding_rs::WINDOWS_1252);
    }
    if label.eq_ignore_ascii_case("shift_jis")
        || label.eq_ignore_ascii_case("shift-jis")
        || label.eq_ignore_ascii_case("sjis")
    {
        return decode_with_encoding(bytes, encoding_rs::SHIFT_JIS);
    }
    if label.eq_ignore_ascii_case("us-ascii")
        || label.eq_ignore_ascii_case("ascii")
        || label.eq_ignore_ascii_case("us_ascii")
        || label.eq_ignore_ascii_case("usascii")
    {
        return decode_ascii(bytes);
    }
    if label.eq_ignore_ascii_case("system") {
        let fallback = system_default_encoding_label();
        if fallback.eq_ignore_ascii_case("binary") {
            return Ok(bytes
                .iter()
                .map(|&b| char::from_u32(b as u32).unwrap())
                .collect());
        }
        return decode_bytes(bytes, fallback);
    }

    if let Some(enc) = Encoding::for_label(label.as_bytes()) {
        return decode_with_encoding(bytes, enc);
    }

    Err(fgets_error(format!(
        "fgets: unsupported encoding '{encoding}'"
    )))
}

fn decode_with_encoding(bytes: &[u8], enc: &'static Encoding) -> BuiltinResult<Vec<char>> {
    let (cow, _, had_errors) = enc.decode(bytes);
    if had_errors {
        return Err(fgets_error(format!(
            "fgets: unable to decode bytes using encoding '{}'",
            enc.name()
        )));
    }
    Ok(cow.chars().collect())
}

fn decode_ascii(bytes: &[u8]) -> BuiltinResult<Vec<char>> {
    if let Some(byte) = bytes.iter().find(|&&b| b > 0x7F) {
        return Err(fgets_error(format!(
            "fgets: byte value {} is outside the ASCII range",
            byte
        )));
    }
    Ok(bytes
        .iter()
        .map(|&b| char::from_u32(b as u32).unwrap())
        .collect())
}

fn numeric_row(bytes: &[u8]) -> BuiltinResult<Value> {
    let data: Vec<f64> = bytes.iter().map(|&b| b as f64).collect();
    let tensor = Tensor::new(data, vec![1, bytes.len()])
        .map_err(|e| fgets_error(format!("fgets: failed to construct numeric array: {e}")))?;
    Ok(Value::Tensor(tensor))
}

fn empty_numeric_row() -> Value {
    let tensor = Tensor::new(Vec::new(), vec![0, 0]).unwrap_or_else(|_| Tensor::zeros(vec![0, 0]));
    Value::Tensor(tensor)
}

fn system_default_encoding_label() -> &'static str {
    #[cfg(windows)]
    {
        "windows-1252"
    }
    #[cfg(not(windows))]
    {
        "utf-8"
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::builtins::io::filetext::{fopen, registry};
    use crate::RuntimeError;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::IntValue;
    use runmat_filesystem as fs;
    use runmat_time::system_time_now;
    use std::path::{Path, PathBuf};
    use std::time::UNIX_EPOCH;

    fn unwrap_error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    fn run_evaluate(fid_value: &Value, rest: &[Value]) -> BuiltinResult<FgetsEval> {
        futures::executor::block_on(evaluate(fid_value, rest))
    }

    fn run_fopen(args: &[Value]) -> BuiltinResult<fopen::FopenEval> {
        futures::executor::block_on(fopen::evaluate(args))
    }

    fn registry_guard() -> std::sync::MutexGuard<'static, ()> {
        registry::test_guard()
    }

    fn unique_path(prefix: &str) -> PathBuf {
        let now = system_time_now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards");
        let filename = format!("{}_{}_{}.tmp", prefix, now.as_secs(), now.subsec_nanos());
        std::env::temp_dir().join(filename)
    }

    fn fopen_path(path: &Path) -> FopenHandle {
        let eval = run_fopen(&[Value::from(path.to_string_lossy().to_string())]).expect("fopen");
        let open = eval.as_open().expect("open outputs");
        assert!(open.fid >= 3.0);
        FopenHandle {
            fid: open.fid as i32,
        }
    }

    struct FopenHandle {
        fid: i32,
    }

    impl Drop for FopenHandle {
        fn drop(&mut self) {
            let _ = registry::close(self.fid);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgets_reads_line_with_newline() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgets_line");
        fs::write(&path, "Hello world\nSecond line\n").unwrap();

        let handle = fopen_path(&path);
        let eval = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("fgets");
        let line = eval.first_output();
        match line {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, "Hello world\n");
            }
            other => panic!("expected char array, got {other:?}"),
        }
        let ltout = eval.outputs()[1].clone();
        match ltout {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![10.0]);
                assert_eq!(t.shape, vec![1, 1]);
            }
            other => panic!("expected numeric tensor, got {other:?}"),
        }

        fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgets_returns_minus_one_at_eof() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgets_eof");
        fs::write(&path, "line\n").unwrap();
        let handle = fopen_path(&path);

        let _ = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("first read");
        let eval = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("second read");
        assert_eq!(eval.first_output(), Value::Num(-1.0));
        assert_eq!(eval.outputs()[1], Value::Num(-1.0));

        fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgets_honours_nchar_limit() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgets_limit");
        fs::write(&path, "abcdefghij\nrest\n").unwrap();
        let handle = fopen_path(&path);

        let eval =
            run_evaluate(&Value::Num(handle.fid as f64), &[Value::Num(5.0)]).expect("limited read");
        match eval.first_output() {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, "abcde");
            }
            other => panic!("expected char array, got {other:?}"),
        }
        match &eval.outputs()[1] {
            Value::Tensor(t) => {
                assert!(t.data.is_empty());
            }
            other => panic!("expected empty numeric tensor, got {other:?}"),
        }

        fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgets_errors_for_write_only_identifier() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgets_write_only");
        fs::write(&path, "payload").unwrap();
        let eval = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w"),
        ])
        .expect("fopen");
        let open = eval.as_open().expect("open outputs");
        assert!(open.fid >= 3.0);
        let err = unwrap_error_message(run_evaluate(&Value::Num(open.fid), &[]).unwrap_err());
        assert_eq!(err, "fgets: file identifier is not open for reading");
        fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgets_respects_limit_before_crlf_sequence() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgets_limit_crlf");
        fs::write(&path, b"ABCDE\r\nnext\n").unwrap();
        let handle = fopen_path(&path);

        let first =
            run_evaluate(&Value::Num(handle.fid as f64), &[Value::Num(3.0)]).expect("first");
        match first.first_output() {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, "ABC");
            }
            other => panic!("expected char array, got {other:?}"),
        }
        match &first.outputs()[1] {
            Value::Tensor(t) => assert!(t.data.is_empty()),
            other => panic!("expected empty numeric tensor, got {other:?}"),
        }

        let second = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("second");
        match second.first_output() {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, "DE\r\n");
            }
            other => panic!("expected char array, got {other:?}"),
        }
        match &second.outputs()[1] {
            Value::Tensor(t) => assert_eq!(t.data, vec![13.0, 10.0]),
            other => panic!("expected CRLF terminators, got {other:?}"),
        }

        let third = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("third");
        match third.first_output() {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, "next\n");
            }
            other => panic!("expected char array, got {other:?}"),
        }

        fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgets_handles_crlf_newlines() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgets_crlf");
        fs::write(&path, b"first line\r\nsecond\r\n").unwrap();
        let handle = fopen_path(&path);

        let eval = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("fgets");
        let outputs = eval.outputs();
        match &outputs[0] {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, "first line\r\n");
            }
            other => panic!("expected char array, got {other:?}"),
        }
        match &outputs[1] {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![13.0, 10.0]);
            }
            other => panic!("expected numeric tensor, got {other:?}"),
        }

        fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgets_decodes_latin1() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgets_latin1");
        fs::write(&path, [0x48u8, 0x6f, 0x6c, 0x61, 0x20, 0xf1, b'\n']).unwrap();
        let eval = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("r"),
            Value::from("native"),
            Value::from("latin1"),
        ])
        .expect("fopen");
        let open = eval.as_open().expect("open outputs");
        let fid = open.fid as i32;

        let read = run_evaluate(&Value::Num(fid as f64), &[]).expect("fgets");
        match read.first_output() {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, "Hola ñ\n");
            }
            other => panic!("expected char array, got {other:?}"),
        }

        let _ = registry::close(fid);
        fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgets_nchar_zero_returns_empty_char() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgets_zero");
        fs::write(&path, "hello\n").unwrap();
        let handle = fopen_path(&path);

        let eval = run_evaluate(
            &Value::Num(handle.fid as f64),
            &[Value::Int(IntValue::I32(0))],
        )
        .expect("fgets");
        match eval.first_output() {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 0);
                assert!(ca.data.is_empty());
            }
            other => panic!("expected empty char array, got {other:?}"),
        }

        fs::remove_file(&path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fgets_gathers_gpu_scalar_arguments() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path = unique_path("fgets_gpu_args");
        fs::write(&path, b"abcdef\nextra").unwrap();
        let handle = fopen_path(&path);

        test_support::with_test_provider(|provider| {
            let fid_host = [handle.fid as f64];
            let fid_view = HostTensorView {
                data: &fid_host,
                shape: &[1, 1],
            };
            let fid_gpu = Value::GpuTensor(provider.upload(&fid_view).expect("upload fid"));

            let limit_host = [3.0f64];
            let limit_view = HostTensorView {
                data: &limit_host,
                shape: &[1, 1],
            };
            let limit_gpu = Value::GpuTensor(provider.upload(&limit_view).expect("upload limit"));

            let eval = run_evaluate(&fid_gpu, &[limit_gpu]).expect("fgets");
            match eval.first_output() {
                Value::CharArray(ca) => {
                    let text: String = ca.data.iter().collect();
                    assert_eq!(text, "abc");
                }
                other => panic!("expected char array, got {other:?}"),
            }
        });

        fs::remove_file(&path).unwrap();
    }
}
