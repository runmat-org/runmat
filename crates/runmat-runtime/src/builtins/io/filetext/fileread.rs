//! MATLAB-compatible `fileread` builtin for RunMat.

use std::path::{Path, PathBuf};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};
use runmat_filesystem as fs;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::filetext::fileread")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fileread",
    op_kind: GpuOpKind::Custom("io-file-read"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Performs synchronous host file I/O; acceleration providers are not involved.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::filetext::fileread")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fileread",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion; executes as a standalone host operation.",
};

const BUILTIN_NAME: &str = "fileread";

const FILEREAD_OUTPUT_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "text",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File contents as a 1-by-N character vector.",
}];
const FILEREAD_INPUTS_FILENAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Path to a readable file.",
}];
const FILEREAD_INPUTS_FILENAME_ENCODING: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Path to a readable file.",
    },
    BuiltinParamDescriptor {
        name: "encoding",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"auto\""),
        description: "Encoding label (for example 'utf-8', 'latin1', 'ascii', 'raw').",
    },
];
const FILEREAD_INPUTS_FILENAME_ENCODING_PAIR: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Path to a readable file.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Optional,
        default: Some("\"Encoding\""),
        description: "Name of supported option; currently only 'Encoding'.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::PropertyValue,
        arity: BuiltinParamArity::Optional,
        default: Some("\"auto\""),
        description: "Option value for the provided option name.",
    },
];
const FILEREAD_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "text = fileread(filename)",
        inputs: &FILEREAD_INPUTS_FILENAME,
        outputs: &FILEREAD_OUTPUT_TEXT,
    },
    BuiltinSignatureDescriptor {
        label: "text = fileread(filename, encoding)",
        inputs: &FILEREAD_INPUTS_FILENAME_ENCODING,
        outputs: &FILEREAD_OUTPUT_TEXT,
    },
    BuiltinSignatureDescriptor {
        label: "text = fileread(filename, \"Encoding\", encoding)",
        inputs: &FILEREAD_INPUTS_FILENAME_ENCODING_PAIR,
        outputs: &FILEREAD_OUTPUT_TEXT,
    },
];

const FILEREAD_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILEREAD.INVALID_INPUT",
    identifier: Some("RunMat:fileread:InvalidInput"),
    when: "Filename or argument cardinality/type constraints are violated.",
    message: "fileread: invalid input arguments",
};
const FILEREAD_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILEREAD.INVALID_OPTION",
    identifier: Some("RunMat:fileread:InvalidOption"),
    when: "Encoding option syntax or value is invalid.",
    message: "fileread: invalid option configuration",
};
const FILEREAD_ERROR_DECODE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILEREAD.DECODE",
    identifier: Some("RunMat:fileread:DecodeFailed"),
    when: "Requested decoding of bytes fails (for example UTF-8 or ASCII mismatch).",
    message: "fileread: unable to decode file contents",
};
const FILEREAD_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILEREAD.IO",
    identifier: Some("RunMat:fileread:IoFailure"),
    when: "Filesystem read operation fails.",
    message: "fileread: file read failed",
};
const FILEREAD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILEREAD.INTERNAL",
    identifier: None,
    when: "Internal runtime control-flow or conversion fails.",
    message: "fileread: internal error",
};
const FILEREAD_ERRORS: [BuiltinErrorDescriptor; 5] = [
    FILEREAD_ERROR_INVALID_INPUT,
    FILEREAD_ERROR_INVALID_OPTION,
    FILEREAD_ERROR_DECODE,
    FILEREAD_ERROR_IO,
    FILEREAD_ERROR_INTERNAL,
];
pub const FILEREAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FILEREAD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FILEREAD_ERRORS,
};

fn fileread_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    fileread_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn fileread_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn fileread_error_with_source(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
    source: impl std::error::Error + Send + Sync + 'static,
) -> RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = FILEREAD_ERROR_INTERNAL.identifier {
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
        let trimmed = label.trim();
        if trimmed.is_empty() {
            return None;
        }
        match trimmed.to_ascii_lowercase().as_str() {
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

#[runtime_builtin(
    name = "fileread",
    category = "io/filetext",
    summary = "Read the entire contents of a text file into a 1-by-N character vector.",
    keywords = "fileread,io,read file,text file,character vector",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::fileread_type),
    descriptor(crate::builtins::io::filetext::fileread::FILEREAD_DESCRIPTOR),
    builtin_path = "crate::builtins::io::filetext::fileread"
)]
async fn fileread_builtin(path: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let gathered_path = gather_if_needed_async(&path)
        .await
        .map_err(map_control_flow)?;
    let gathered_rest = gather_values(&rest).await?;
    let encoding = parse_encoding_args(&gathered_rest)?;
    let resolved = resolve_path(&gathered_path)?;
    let bytes = read_all(&resolved).await?;
    let chars = decode_bytes(bytes, encoding)?;
    let cols = chars.len();
    let char_array = CharArray::new(chars, 1, cols)
        .map_err(|e| fileread_error_with_detail(&FILEREAD_ERROR_INTERNAL, &e))?;
    Ok(Value::CharArray(char_array))
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

fn parse_encoding_args(args: &[Value]) -> BuiltinResult<FileEncoding> {
    match args.len() {
        0 => Ok(FileEncoding::Auto),
        1 => {
            if is_encoding_keyword(&args[0])? {
                return Err(fileread_error_with_detail(
                    &FILEREAD_ERROR_INVALID_OPTION,
                    "missing encoding value after 'Encoding' keyword",
                ));
            }
            encoding_from_value(&args[0])
        }
        2 => {
            if !is_encoding_keyword(&args[0])? {
                return Err(fileread_error_with_detail(
                    &FILEREAD_ERROR_INVALID_OPTION,
                    "expected 'Encoding' keyword before encoding value",
                ));
            }
            encoding_from_value(&args[1])
        }
        _ => Err(fileread_error_with_detail(
            &FILEREAD_ERROR_INVALID_INPUT,
            "too many input arguments",
        )),
    }
}

fn encoding_from_value(value: &Value) -> BuiltinResult<FileEncoding> {
    let label = encoding_name(value)?;
    match FileEncoding::from_label(&label) {
        Some(enc) => Ok(enc),
        None => {
            if label.trim().is_empty() {
                Err(fileread_error_with_detail(
                    &FILEREAD_ERROR_INVALID_OPTION,
                    "encoding name must not be empty",
                ))
            } else {
                Err(fileread_error_with_detail(
                    &FILEREAD_ERROR_INVALID_OPTION,
                    format!("unsupported encoding '{}'", label),
                ))
            }
        }
    }
}

fn is_encoding_keyword(value: &Value) -> BuiltinResult<bool> {
    let text = encoding_name(value)?;
    Ok(text.eq_ignore_ascii_case("encoding"))
}

fn encoding_name(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::CharArray(_) => Err(fileread_error_with_detail(
            &FILEREAD_ERROR_INVALID_OPTION,
            "encoding name must be a 1-by-N character vector",
        )),
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                Ok(sa.data[0].clone())
            } else {
                Err(fileread_error_with_detail(
                    &FILEREAD_ERROR_INVALID_OPTION,
                    "encoding inputs must be scalar string arrays",
                ))
            }
        }
        other => Err(fileread_error_with_detail(
            &FILEREAD_ERROR_INVALID_OPTION,
            format!("expected encoding as string scalar or character vector, got {other:?}"),
        )),
    }
}

fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
    match value {
        Value::String(s) => normalize_path(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let path: String = ca.data.iter().collect();
            normalize_path(&path)
        }
        Value::CharArray(_) => Err(fileread_error_with_detail(
            &FILEREAD_ERROR_INVALID_INPUT,
            "expected a 1-by-N character vector for the file name",
        )),
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                normalize_path(&sa.data[0])
            } else {
                Err(fileread_error_with_detail(
                    &FILEREAD_ERROR_INVALID_INPUT,
                    "string array inputs must be scalar",
                ))
            }
        }
        other => Err(fileread_error_with_detail(
            &FILEREAD_ERROR_INVALID_INPUT,
            format!("expected filename as string scalar or character vector, got {other:?}"),
        )),
    }
}

fn normalize_path(raw: &str) -> BuiltinResult<PathBuf> {
    if raw.is_empty() {
        return Err(fileread_error_with_detail(
            &FILEREAD_ERROR_INVALID_INPUT,
            "filename must not be empty",
        ));
    }
    Ok(Path::new(raw).to_path_buf())
}

async fn read_all(path: &Path) -> BuiltinResult<Vec<u8>> {
    fs::read_async(path).await.map_err(|err| {
        fileread_error_with_source(
            format!(
                "{}: unable to read '{}': {}",
                FILEREAD_ERROR_IO.message,
                path.display(),
                err
            ),
            &FILEREAD_ERROR_IO,
            err,
        )
    })
}

fn decode_bytes(bytes: Vec<u8>, encoding: FileEncoding) -> BuiltinResult<Vec<char>> {
    match encoding {
        FileEncoding::Auto => match String::from_utf8(bytes) {
            Ok(text) => Ok(text.chars().collect()),
            Err(err) => Ok(bytes_to_chars(err.into_bytes())),
        },
        FileEncoding::Utf8 => decode_utf8(bytes),
        FileEncoding::Ascii => decode_ascii(bytes),
        FileEncoding::Latin1 | FileEncoding::Raw => Ok(bytes_to_chars(bytes)),
    }
}

fn decode_utf8(bytes: Vec<u8>) -> BuiltinResult<Vec<char>> {
    match String::from_utf8(bytes) {
        Ok(text) => Ok(text.chars().collect()),
        Err(err) => {
            let utf8_err = err.utf8_error();
            let offset = utf8_err.valid_up_to();
            let detail = match utf8_err.error_len() {
                Some(len) => {
                    let plural = if len == 1 { "" } else { "s" };
                    format!(
                        "invalid UTF-8 sequence of {len} byte{plural} starting at offset {offset}"
                    )
                }
                None => format!("incomplete UTF-8 sequence at end of data (after offset {offset})"),
            };
            Err(fileread_error_with_detail(
                &FILEREAD_ERROR_DECODE,
                format!("unable to decode file as UTF-8: {detail}"),
            ))
        }
    }
}

fn decode_ascii(bytes: Vec<u8>) -> BuiltinResult<Vec<char>> {
    for (idx, byte) in bytes.iter().enumerate() {
        if *byte > 0x7F {
            return Err(fileread_error_with_detail(
                &FILEREAD_ERROR_DECODE,
                format!("byte 0x{byte:02X} at offset {idx} is not valid ASCII"),
            ));
        }
    }
    Ok(bytes_to_chars(bytes))
}

fn bytes_to_chars(bytes: Vec<u8>) -> Vec<char> {
    bytes.into_iter().map(char::from).collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_time::unix_timestamp_ms;
    use std::io::Write;

    fn unwrap_error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    fn run_fileread(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(fileread_builtin(path, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fileread_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = FILEREAD_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"text = fileread(filename)"));
        assert!(labels.contains(&"text = fileread(filename, encoding)"));
        assert!(labels.contains(&"text = fileread(filename, \"Encoding\", encoding)"));
    }

    fn unique_path(prefix: &str) -> PathBuf {
        let millis = unix_timestamp_ms();
        let mut path = std::env::temp_dir();
        path.push(format!("runmat_{prefix}_{}_{}", std::process::id(), millis));
        path
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fileread_reads_text_file() {
        let path = unique_path("fileread_text");
        let contents = "RunMat fileread\nLine two\n";
        test_support::fs::write(&path, contents).expect("write sample file");

        let value = Value::from(path.to_string_lossy().to_string());
        let result = run_fileread(value, Vec::new()).expect("fileread result");

        match result {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, contents.chars().count());
                assert_eq!(text, contents);
            }
            other => panic!("expected char array, got {other:?}"),
        }

        let _ = test_support::fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fileread_accepts_char_array_input() {
        let path = unique_path("fileread_char_input");
        test_support::fs::write(&path, "abc").expect("write sample file");

        let path_str = path.to_string_lossy();
        let chars: Vec<char> = path_str.chars().collect();
        let len = chars.len();
        let char_array = CharArray::new(chars, 1, len).expect("char array from path string");
        let value = Value::CharArray(char_array);

        let result = run_fileread(value, Vec::new()).expect("fileread");
        match result {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, "abc");
            }
            other => panic!("expected char array, got {other:?}"),
        }

        let _ = test_support::fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fileread_accepts_string_array_scalar() {
        let path = unique_path("fileread_string_scalar");
        test_support::fs::write(&path, "xyz").expect("write sample file");

        let path_str = path.to_string_lossy().to_string();
        let string_array = runmat_builtins::StringArray::new(vec![path_str.clone()], vec![1, 1])
            .expect("string array scalar");
        let value = Value::StringArray(string_array);

        let result = run_fileread(value, Vec::new()).expect("fileread");
        match result {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, "xyz");
            }
            other => panic!("expected char array, got {other:?}"),
        }

        let _ = test_support::fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fileread_returns_empty_for_empty_file() {
        let path = unique_path("fileread_empty");
        fs::File::create(&path).expect("create empty file");

        let result = run_fileread(Value::from(path.to_string_lossy().to_string()), Vec::new())
            .expect("fileread");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 0);
                assert!(ca.data.is_empty());
            }
            other => panic!("expected char array, got {other:?}"),
        }

        let _ = test_support::fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fileread_errors_when_file_missing() {
        let path = unique_path("fileread_missing");
        let err = unwrap_error_message(
            run_fileread(Value::from(path.to_string_lossy().to_string()), Vec::new()).unwrap_err(),
        );
        assert!(
            err.contains("unable to read"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fileread_preserves_non_utf8_bytes() {
        let path = unique_path("fileread_raw_bytes");
        let mut file = fs::File::create(&path).expect("create file");
        file.write_all(&[0xFF, 0x61, 0x00]).expect("write bytes");
        drop(file);

        let result = run_fileread(Value::from(path.to_string_lossy().to_string()), Vec::new())
            .expect("fileread");
        match result {
            Value::CharArray(ca) => {
                let collected: Vec<u32> = ca.data.iter().map(|ch| *ch as u32).collect();
                assert_eq!(collected, vec![0x00FF, 0x0061, 0x0000]);
            }
            other => panic!("expected char array, got {other:?}"),
        }

        let _ = test_support::fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fileread_supports_encoding_keyword() {
        let path = unique_path("fileread_utf8_keyword");
        let contents = "UTF-8 ✓";
        test_support::fs::write(&path, contents).expect("write sample file");

        let result = run_fileread(
            Value::from(path.to_string_lossy().to_string()),
            vec![Value::from("Encoding"), Value::from("utf-8")],
        )
        .expect("fileread");
        match result {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, contents);
            }
            other => panic!("expected char array, got {other:?}"),
        }

        let _ = test_support::fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fileread_supports_single_encoding_argument() {
        let path = unique_path("fileread_latin1");
        let bytes = [0xC0u8, 0x20, 0x41];
        test_support::fs::write(&path, bytes).expect("write latin1 data");

        let result = run_fileread(
            Value::from(path.to_string_lossy().to_string()),
            vec![Value::from("latin1")],
        )
        .expect("fileread");
        match result {
            Value::CharArray(ca) => {
                let codes: Vec<u32> = ca.data.iter().map(|ch| *ch as u32).collect();
                assert_eq!(codes, vec![0x00C0, 0x0020, 0x0041]);
            }
            other => panic!("expected char array, got {other:?}"),
        }

        let _ = test_support::fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fileread_raw_encoding_returns_bytes() {
        let path = unique_path("fileread_raw_encoding");
        let bytes = [0x01u8, 0xFF, 0x7F];
        test_support::fs::write(&path, bytes).expect("write raw bytes");

        let result = run_fileread(
            Value::from(path.to_string_lossy().to_string()),
            vec![Value::from("raw")],
        )
        .expect("fileread");
        match result {
            Value::CharArray(ca) => {
                let codes: Vec<u8> = ca.data.iter().map(|ch| *ch as u8).collect();
                assert_eq!(codes, bytes);
            }
            other => panic!("expected char array, got {other:?}"),
        }

        let _ = test_support::fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fileread_ascii_encoding_errors_on_invalid_bytes() {
        let path = unique_path("fileread_ascii_error");
        test_support::fs::write(&path, [0x41, 0x80]).expect("write bytes");

        let err = unwrap_error_message(
            run_fileread(
                Value::from(path.to_string_lossy().to_string()),
                vec![Value::from("Encoding"), Value::from("ascii")],
            )
            .unwrap_err(),
        );
        assert!(
            err.contains("not valid ASCII"),
            "unexpected error message: {err}"
        );

        let _ = test_support::fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fileread_encoding_keyword_missing_value_errors() {
        let path = unique_path("fileread_encoding_missing");
        test_support::fs::write(&path, "abc").expect("write file");

        let err = unwrap_error_message(
            run_fileread(
                Value::from(path.to_string_lossy().to_string()),
                vec![Value::from("Encoding")],
            )
            .unwrap_err(),
        );
        assert!(
            err.contains("missing encoding value"),
            "unexpected error message: {err}"
        );

        let _ = test_support::fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fileread_too_many_arguments_errors() {
        let path = unique_path("fileread_too_many");
        test_support::fs::write(&path, "abc").expect("write file");

        let err = unwrap_error_message(
            run_fileread(
                Value::from(path.to_string_lossy().to_string()),
                vec![
                    Value::from("Encoding"),
                    Value::from("utf-8"),
                    Value::from("extra"),
                ],
            )
            .unwrap_err(),
        );
        assert!(
            err.contains("too many input arguments"),
            "unexpected error message: {err}"
        );

        let _ = test_support::fs::remove_file(&path);
    }
}
