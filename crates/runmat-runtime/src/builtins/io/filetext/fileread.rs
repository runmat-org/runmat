//! MATLAB-compatible `fileread` builtin for RunMat.

use std::path::{Path, PathBuf};

use runmat_builtins::{CharArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::gather_if_needed;
use runmat_filesystem as fs;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "fileread",
        builtin_path = "crate::builtins::io::filetext::fileread"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "fileread"
category: "io/filetext"
keywords: ["fileread", "io", "read file", "text file", "character vector"]
summary: "Read the entire contents of a text file into a 1-by-N character vector."
references:
  - https://www.mathworks.com/help/matlab/ref/fileread.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs entirely on the host CPU. When the file path lives on the GPU, RunMat gathers it first; the file contents remain host-resident."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::filetext::fileread::tests"
  integration: "builtins::io::filetext::fileread::tests::fileread_reads_text_file"
---

# What does the `fileread` function do in MATLAB / RunMat?
`fileread(filename)` loads the complete contents of a text file into memory and returns a row vector of characters that mirrors MATLAB's behaviour. The builtin preserves all bytes, including newlines and extended ASCII characters, so downstream code can parse, split, or convert the text as needed.

## How does the `fileread` function behave in MATLAB / RunMat?
- Accepts paths provided as character vectors or string scalars. String arrays must contain exactly one element.
- Supports an optional encoding argument: `fileread(filename, encoding)` or `fileread(filename, 'Encoding', encoding)`. Recognised values include `auto` (default), `utf-8`, `ascii`, `latin1`, and `raw`.
- Resolves relative paths with respect to the current working directory of the RunMat process, just like MATLAB.
- Returns a `1×N` character array. Empty files yield a `1×0` character vector.
- Leaves line endings untouched (`\r`, `\n`, or `\r\n`) so scripts can inspect original formatting.
- When `auto` decoding detects invalid UTF-8 sequences, RunMat maps each byte to the corresponding extended-ASCII code point so that callers can recover the raw data.
- Throws a descriptive error when the file cannot be opened or read.

## `fileread` Function GPU Execution Behaviour
`fileread` performs synchronous host I/O and never dispatches GPU work. If the provided file name lives on the GPU (for example, produced by a GPU array that was gathered lazily), RunMat gathers that scalar first. File contents are returned as an ordinary character array that resides on the CPU. Providers do not need to implement any hooks for this builtin.

## Examples of using the `fileread` function in MATLAB / RunMat

### Read Entire File Into A Character Vector
```matlab
text = fileread("LICENSE.md");
```
Expected output:
```matlab
% Character vector containing the full license text
```

### Read A File Using A Relative Path
```matlab
text = fileread("data/config.json");
```
Expected output:
```matlab
% Returns the JSON file contents as a character vector
```

### Preserve Extended ASCII Bytes
```matlab
bytes = fileread("fixtures/high_ascii.txt");
double_values = double(bytes);
```
Expected output:
```matlab
% double_values contains the numeric codes for every byte in the file
```

### Convert File Contents To A String Scalar
```matlab
raw = fileread("README.md");
doc = string(raw);
```
Expected output:
```matlab
% doc is a string scalar ready for further processing
```

### Read A File With UTF-8 Decoding Explicitly
```matlab
text = fileread("data/report.txt", 'Encoding', 'utf-8');
```
Expected output:
```matlab
% Character vector decoded using UTF-8. An error is thrown if the bytes are not valid UTF-8.
```

### Handle Missing Files With Try/Catch
```matlab
try
    fileread("missing.txt");
catch err
    disp(err.message);
end
```
Expected output:
```matlab
% Prints a descriptive error such as:
% "fileread: unable to read 'missing.txt' (No such file or directory)"
```

## FAQ

### What does `fileread` return?
It returns a `1×N` character vector containing every byte from the file. Convert it to a string with `string(...)` when you prefer string scalars.

### Does `fileread` change line endings?
No. The builtin preserves whatever newline sequence the file uses so downstream tools can handle formatting explicitly.

### Can `fileread` read binary data?
While designed for text, `fileread` will happily return any bytes. The result is a character vector whose numeric codes match the file's bytes.

### How are encodings handled?
The default `auto` mode attempts UTF-8 decoding and, if the data is not valid UTF-8, falls back to mapping each byte to its extended-ASCII code point (`latin1`). Provide an explicit encoding such as `'utf-8'`, `'latin1'`, `'ascii'`, or `'raw'` to control the conversion. Explicit encodings raise descriptive errors when the bytes are incompatible with the requested format.

### Can I force raw byte behaviour?
Yes. Specify `'raw'` (or `'bytes'`) as the encoding argument to receive a character vector whose code points equal the file's bytes.

### How do relative paths resolve?
Relative paths are evaluated against the current working directory of the RunMat process. Use `pwd` or `cd` to control where `fileread` looks.

## See Also
[fopen](./fopen), [fread](./fread), [string](./string), [strlength](./strlength)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/io/filetext/fileread.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/filetext/fileread.rs)
- Found an issue? [Open a GitHub ticket](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

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
    builtin_path = "crate::builtins::io::filetext::fileread"
)]
fn fileread_builtin(path: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let gathered_path = gather_if_needed(&path).map_err(|e| format!("fileread: {e}"))?;
    let gathered_rest = gather_values(&rest)?;
    let encoding = parse_encoding_args(&gathered_rest)?;
    let resolved = resolve_path(&gathered_path)?;
    let bytes = read_all(&resolved)?;
    let chars = decode_bytes(bytes, encoding)?;
    let cols = chars.len();
    let char_array = CharArray::new(chars, 1, cols).map_err(|e| format!("fileread: {e}"))?;
    Ok(Value::CharArray(char_array))
}

fn gather_values(values: &[Value]) -> Result<Vec<Value>, String> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(gather_if_needed(value).map_err(|e| format!("fileread: {e}"))?);
    }
    Ok(out)
}

fn parse_encoding_args(args: &[Value]) -> Result<FileEncoding, String> {
    match args.len() {
        0 => Ok(FileEncoding::Auto),
        1 => {
            if is_encoding_keyword(&args[0])? {
                return Err("fileread: missing encoding value after 'Encoding' keyword".to_string());
            }
            encoding_from_value(&args[0])
        }
        2 => {
            if !is_encoding_keyword(&args[0])? {
                return Err(
                    "fileread: expected 'Encoding' keyword before encoding value".to_string(),
                );
            }
            encoding_from_value(&args[1])
        }
        _ => Err("fileread: too many input arguments".to_string()),
    }
}

fn encoding_from_value(value: &Value) -> Result<FileEncoding, String> {
    let label = encoding_name(value)?;
    match FileEncoding::from_label(&label) {
        Some(enc) => Ok(enc),
        None => {
            if label.trim().is_empty() {
                Err("fileread: encoding name must not be empty".to_string())
            } else {
                Err(format!("fileread: unsupported encoding '{}'", label))
            }
        }
    }
}

fn is_encoding_keyword(value: &Value) -> Result<bool, String> {
    let text = encoding_name(value)?;
    Ok(text.eq_ignore_ascii_case("encoding"))
}

fn encoding_name(value: &Value) -> Result<String, String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::CharArray(_) => {
            Err("fileread: encoding name must be a 1-by-N character vector".to_string())
        }
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                Ok(sa.data[0].clone())
            } else {
                Err("fileread: encoding inputs must be scalar string arrays".to_string())
            }
        }
        other => Err(format!(
            "fileread: expected encoding as string scalar or character vector, got {other:?}"
        )),
    }
}

fn resolve_path(value: &Value) -> Result<PathBuf, String> {
    match value {
        Value::String(s) => normalize_path(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let path: String = ca.data.iter().collect();
            normalize_path(&path)
        }
        Value::CharArray(_) => {
            Err("fileread: expected a 1-by-N character vector for the file name".to_string())
        }
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                normalize_path(&sa.data[0])
            } else {
                Err("fileread: string array inputs must be scalar".to_string())
            }
        }
        other => Err(format!(
            "fileread: expected filename as string scalar or character vector, got {other:?}"
        )),
    }
}

fn normalize_path(raw: &str) -> Result<PathBuf, String> {
    if raw.is_empty() {
        return Err("fileread: filename must not be empty".to_string());
    }
    Ok(Path::new(raw).to_path_buf())
}

fn read_all(path: &Path) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|err| format!("fileread: unable to read '{}': {}", path.display(), err))
}

fn decode_bytes(bytes: Vec<u8>, encoding: FileEncoding) -> Result<Vec<char>, String> {
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

fn decode_utf8(bytes: Vec<u8>) -> Result<Vec<char>, String> {
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
            Err(format!(
                "fileread: unable to decode file as UTF-8: {detail}"
            ))
        }
    }
}

fn decode_ascii(bytes: Vec<u8>) -> Result<Vec<char>, String> {
    for (idx, byte) in bytes.iter().enumerate() {
        if *byte > 0x7F {
            return Err(format!(
                "fileread: byte 0x{byte:02X} at offset {idx} is not valid ASCII"
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
    use runmat_filesystem as fs;
    use runmat_time::unix_timestamp_ms;
    use std::io::Write;

    use crate::builtins::common::test_support;

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
        fs::write(&path, contents).expect("write sample file");

        let value = Value::from(path.to_string_lossy().to_string());
        let result = fileread_builtin(value, Vec::new()).expect("fileread result");

        match result {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, contents.chars().count());
                assert_eq!(text, contents);
            }
            other => panic!("expected char array, got {other:?}"),
        }

        let _ = fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fileread_accepts_char_array_input() {
        let path = unique_path("fileread_char_input");
        fs::write(&path, "abc").expect("write sample file");

        let path_str = path.to_string_lossy();
        let chars: Vec<char> = path_str.chars().collect();
        let len = chars.len();
        let char_array = CharArray::new(chars, 1, len).expect("char array from path string");
        let value = Value::CharArray(char_array);

        let result = fileread_builtin(value, Vec::new()).expect("fileread");
        match result {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, "abc");
            }
            other => panic!("expected char array, got {other:?}"),
        }

        let _ = fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fileread_accepts_string_array_scalar() {
        let path = unique_path("fileread_string_scalar");
        fs::write(&path, "xyz").expect("write sample file");

        let path_str = path.to_string_lossy().to_string();
        let string_array = runmat_builtins::StringArray::new(vec![path_str.clone()], vec![1, 1])
            .expect("string array scalar");
        let value = Value::StringArray(string_array);

        let result = fileread_builtin(value, Vec::new()).expect("fileread");
        match result {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, "xyz");
            }
            other => panic!("expected char array, got {other:?}"),
        }

        let _ = fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fileread_returns_empty_for_empty_file() {
        let path = unique_path("fileread_empty");
        fs::File::create(&path).expect("create empty file");

        let result = fileread_builtin(Value::from(path.to_string_lossy().to_string()), Vec::new())
            .expect("fileread");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 0);
                assert!(ca.data.is_empty());
            }
            other => panic!("expected char array, got {other:?}"),
        }

        let _ = fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fileread_errors_when_file_missing() {
        let path = unique_path("fileread_missing");
        let err = fileread_builtin(Value::from(path.to_string_lossy().to_string()), Vec::new())
            .expect_err("missing file should error");
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

        let result = fileread_builtin(Value::from(path.to_string_lossy().to_string()), Vec::new())
            .expect("fileread");
        match result {
            Value::CharArray(ca) => {
                let collected: Vec<u32> = ca.data.iter().map(|ch| *ch as u32).collect();
                assert_eq!(collected, vec![0x00FF, 0x0061, 0x0000]);
            }
            other => panic!("expected char array, got {other:?}"),
        }

        let _ = fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fileread_supports_encoding_keyword() {
        let path = unique_path("fileread_utf8_keyword");
        let contents = "UTF-8 ✓";
        fs::write(&path, contents).expect("write sample file");

        let result = fileread_builtin(
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

        let _ = fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fileread_supports_single_encoding_argument() {
        let path = unique_path("fileread_latin1");
        let bytes = [0xC0u8, 0x20, 0x41];
        fs::write(&path, bytes).expect("write latin1 data");

        let result = fileread_builtin(
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

        let _ = fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fileread_raw_encoding_returns_bytes() {
        let path = unique_path("fileread_raw_encoding");
        let bytes = [0x01u8, 0xFF, 0x7F];
        fs::write(&path, bytes).expect("write raw bytes");

        let result = fileread_builtin(
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

        let _ = fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fileread_ascii_encoding_errors_on_invalid_bytes() {
        let path = unique_path("fileread_ascii_error");
        fs::write(&path, [0x41, 0x80]).expect("write bytes");

        let err = fileread_builtin(
            Value::from(path.to_string_lossy().to_string()),
            vec![Value::from("Encoding"), Value::from("ascii")],
        )
        .expect_err("invalid ASCII should error");
        assert!(
            err.contains("not valid ASCII"),
            "unexpected error message: {err}"
        );

        let _ = fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fileread_encoding_keyword_missing_value_errors() {
        let path = unique_path("fileread_encoding_missing");
        fs::write(&path, "abc").expect("write file");

        let err = fileread_builtin(
            Value::from(path.to_string_lossy().to_string()),
            vec![Value::from("Encoding")],
        )
        .expect_err("missing encoding value should error");
        assert!(
            err.contains("missing encoding value"),
            "unexpected error message: {err}"
        );

        let _ = fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fileread_too_many_arguments_errors() {
        let path = unique_path("fileread_too_many");
        fs::write(&path, "abc").expect("write file");

        let err = fileread_builtin(
            Value::from(path.to_string_lossy().to_string()),
            vec![
                Value::from("Encoding"),
                Value::from("utf-8"),
                Value::from("extra"),
            ],
        )
        .expect_err("too many arguments should error");
        assert!(
            err.contains("too many input arguments"),
            "unexpected error message: {err}"
        );

        let _ = fs::remove_file(&path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(
            !blocks.is_empty(),
            "expected matlab code blocks in documentation"
        );
    }
}
