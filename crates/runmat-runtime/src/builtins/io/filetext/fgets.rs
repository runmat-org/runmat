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
use crate::{gather_if_needed, register_builtin_fusion_spec, register_builtin_gpu_spec};
use runmat_filesystem::File;

#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;

const INVALID_IDENTIFIER_MESSAGE: &str =
    "Invalid file identifier. Use fopen to generate a valid file ID.";

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "fgets"
category: "io/filetext"
keywords: ["fgets", "text file", "line input", "newline"]
summary: "Read the next line from a file, keeping newline characters in the result."
references:
  - https://www.mathworks.com/help/matlab/ref/fgets.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs entirely on the host CPU. When arguments live on the GPU, RunMat gathers them before performing host I/O."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::filetext::fgets::tests"
---

# What does the `fgets` function do in MATLAB / RunMat?
`fgets(fid)` returns the next line from the text file associated with `fid`, preserving any newline
characters that terminate the line. It mirrors MATLAB's behaviour for optional length limits,
line-terminator reporting, and end-of-file handling so existing scripts work without modification.

## How does the `fgets` function behave in MATLAB / RunMat?
- `tline = fgets(fid)` reads from the current position to the next newline (including the newline) or
  to end-of-file when no terminator is found. The result is a character row vector with MATLAB's
  column-major semantics.
- `tline = fgets(fid, nchar)` limits the read to at most `nchar` characters (rounded toward zero). The
  call stops early if a newline is encountered before the limit. When the newline appears beyond the
  limit, the delimiter is left unread for the next call.
- `[tline, ltout] = fgets(___)` additionally returns the line terminators as a row vector of doubles.
  For Windows newlines (`\r\n`) the second output is `[13 10]`, for Unix newlines (`\n`) it is `10`, and
  it is empty when the line has no terminator.
- When the file is empty or the file position indicator is already at end-of-file, `tline` (and `ltout`,
  if requested) return `-1`.
- Lines are decoded using the text encoding recorded by `fopen`. UTF-8, US-ASCII, ISO-8859-1
  (`latin1`), Windows-1252, Shift_JIS, and binary mode are recognised without additional user work.

## `fgets` Function GPU Execution Behaviour
`fgets` is a host-only operation. File identifiers live in the host registry created by `fopen`, so
arguments that arrive as GPU-resident scalars are gathered back to the CPU before the read occurs.
The returned line and optional terminator vector are regular host values; no GPU residency is tracked.

## Examples of using the `fgets` function in MATLAB / RunMat

### Read the first line of a file
```matlab
fname = tempname;
fid = fopen(fname, 'w');
fprintf(fid, 'RunMat\nSecond line\n');
fclose(fid);

fid = fopen(fname, 'r');
line = fgets(fid);
fclose(fid);
delete(fname);

double(line)
```
Expected output:
```matlab
ans =
    82   117   110    77    97   116    10
```

### Limit the number of characters returned
```matlab
fname = tempname;
fid = fopen(fname, 'w');
fprintf(fid, 'Example line\n');
fclose(fid);

fid = fopen(fname, 'r');
snippet = fgets(fid, 5);
fclose(fid);
delete(fname);

snippet
double(snippet)
```
Expected output:
```matlab
snippet =
    'Exam'
ans =
    69   120    97   109
```

### Retrieve line terminators separately
```matlab
fname = tempname;
fid = fopen(fname, 'w');
fprintf(fid, 'Windows line\r\n');
fclose(fid);

fid = fopen(fname, 'r');
[line, ltout] = fgets(fid);
fclose(fid);
delete(fname);

content = line(1:end-numel(ltout));
terminators = double(ltout);

content
terminators
```
Expected output:
```matlab
content =
    'Windows line'
terminators =
    13    10
```

### Handle lines without a trailing newline
```matlab
fname = tempname;
fid = fopen(fname, 'w');
fprintf(fid, 'last line');
fclose(fid);

fid = fopen(fname, 'r');
[line1, lt1] = fgets(fid);
line2 = fgets(fid);
fclose(fid);
delete(fname);

line1
lt1
line2
```
Expected output:
```matlab
line1 =
    'last line'
lt1 =
     []
line2 =
    -1
```

### Detect end of file using the return value
```matlab
fname = tempname;
fid = fopen(fname, 'w');
fprintf(fid, 'one\n');
fprintf(fid, 'two\n');
fclose(fid);

fid = fopen(fname, 'r');
tline = fgets(fid);
while tline ~= -1
    fprintf('> %s', tline);
    tline = fgets(fid);
end
fclose(fid);
delete(fname);
```
Expected output:
```matlab
> one
> two
```

### Read Latin-1 encoded text
```matlab
fname = tempname;
fid = fopen(fname, 'w', 'n', 'latin1');
fprintf(fid, 'Español\n');
fclose(fid);

fid = fopen(fname, 'r', 'n', 'latin1');
line = fgets(fid);
fclose(fid);
delete(fname);

text = line(1:end-1);
codes = double(text);

text
codes
```
Expected output:
```matlab
text =
    'Español'
codes =
    69   115   112    97   241   111   108
```

## FAQ

### What does `fgets` return at end-of-file?
When no characters can be read because the file position indicator is at end-of-file, `fgets`
returns the numeric value `-1`. If you request two outputs, `ltout` is also `-1` in this case.

### Does `fgets` strip newline characters?
No. Unlike `fgetl`, `fgets` keeps any newline bytes in the returned character vector so you can
distinguish empty lines from lines that end in a newline.

### How do I interpret the second output `ltout`?
`ltout` contains the numeric codes for the terminators that ended the line. Use `char(ltout)` or
`double(ltout)` to inspect them. On Windows you typically see `[13 10]`, on Unix-like platforms `10`,
`[]` when the line has no terminator, and `-1` when end-of-file is reached.

### What happens if I specify an `nchar` limit?
RunMat reads at most `nchar` characters. If the newline characters appear before reaching the limit,
they are included in the output. If the limit is reached first, the newline remains unread and `ltout`
is empty.

### Which encodings are supported?
`fgets` honours the encoding recorded by `fopen`. UTF-8, US-ASCII, Latin-1 (ISO-8859-1), Windows-1252,
Shift_JIS, and binary mode are recognised. On most Unix-like systems the `system` encoding resolves to
UTF-8; on Windows it defaults to Windows-1252.

### How does `fgets` differ from `fgetl`?
`fgetl` removes newline characters, while `fgets` keeps them. Use `fgetl` when you want newline-free
strings and `fgets` when you need to preserve the exact bytes that appear in the file.

### Can I call `fgets` on files opened for writing only?
No. The file must be opened with read permission (for example `'r'`, `'r+'`, or `'w+'`). Calling
`fgets` on a write-only identifier raises an error.

## See Also
[fopen](./fopen), [fclose](./fclose), [feof](./feof), [fread](./fread), [fileread](./fileread)

## Source & Feedback
- Implementation: `crates/runmat-runtime/src/builtins/io/filetext/fgets.rs`
- Found a bug? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal repro.
"#;

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

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fgets",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "File I/O calls are not eligible for fusion.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("fgets", DOC_MD);

#[runtime_builtin(
    name = "fgets",
    category = "io/filetext",
    summary = "Read the next line from a file, including newline characters.",
    keywords = "fgets,file,io,line,newline",
    accel = "cpu"
)]
fn fgets_builtin(fid: Value, rest: Vec<Value>) -> Result<Value, String> {
    let eval = evaluate(&fid, &rest)?;
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

pub fn evaluate(fid_value: &Value, rest: &[Value]) -> Result<FgetsEval, String> {
    if rest.len() > 1 {
        return Err("fgets: too many input arguments".to_string());
    }

    let fid_host = gather_value(fid_value)?;
    let fid = parse_fid(&fid_host)?;
    if fid < 0 {
        return Err("fgets: file identifier must be non-negative".to_string());
    }
    if fid < 3 {
        return Err("fgets: standard input/output identifiers are not supported yet".to_string());
    }

    let info =
        registry::info_for(fid).ok_or_else(|| format!("fgets: {INVALID_IDENTIFIER_MESSAGE}"))?;
    if !permission_allows_read(&info.permission) {
        return Err("fgets: file identifier is not open for reading".to_string());
    }
    let handle =
        registry::take_handle(fid).ok_or_else(|| format!("fgets: {INVALID_IDENTIFIER_MESSAGE}"))?;
    let mut file = handle
        .lock()
        .map_err(|_| "fgets: failed to lock file handle (poisoned mutex)".to_string())?;

    let limit = parse_nchar(rest)?;
    let read = read_line(&mut *file, limit).map_err(|err| format!("fgets: {err}"))?;
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

fn gather_value(value: &Value) -> Result<Value, String> {
    gather_if_needed(value).map_err(|e| format!("fgets: {e}"))
}

fn parse_fid(value: &Value) -> Result<i32, String> {
    match value {
        Value::Num(n) => {
            if !n.is_finite() {
                return Err("fgets: file identifier must be finite".to_string());
            }
            if (n.fract()).abs() > f64::EPSILON {
                return Err("fgets: file identifier must be an integer scalar".to_string());
            }
            Ok(*n as i32)
        }
        Value::Int(i) => Ok(i.to_i64() as i32),
        Value::Tensor(t) if t.data.len() == 1 => {
            let n = t.data[0];
            if !n.is_finite() {
                return Err("fgets: file identifier must be finite".to_string());
            }
            if (n.fract()).abs() > f64::EPSILON {
                return Err("fgets: file identifier must be an integer scalar".to_string());
            }
            Ok(n as i32)
        }
        _ => Err("fgets: file identifier must be a numeric scalar".to_string()),
    }
}

fn parse_nchar(args: &[Value]) -> Result<Option<usize>, String> {
    if args.is_empty() {
        return Ok(None);
    }
    let value = gather_value(&args[0])?;
    match value {
        Value::Num(n) => {
            if !n.is_finite() {
                if n.is_sign_positive() {
                    return Ok(None);
                }
                return Err("fgets: nchar must be a non-negative integer scalar".to_string());
            }
            if n < 0.0 {
                return Err("fgets: nchar must be a non-negative integer scalar".to_string());
            }
            if (n.fract()).abs() > f64::EPSILON {
                return Err("fgets: nchar must be a non-negative integer scalar".to_string());
            }
            Ok(Some(n as usize))
        }
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < 0 {
                return Err("fgets: nchar must be a non-negative integer scalar".to_string());
            }
            Ok(Some(raw as usize))
        }
        Value::Tensor(t) if t.data.len() == 1 => {
            let n = t.data[0];
            if !n.is_finite() {
                if n.is_sign_positive() {
                    return Ok(None);
                }
                return Err("fgets: nchar must be a non-negative integer scalar".to_string());
            }
            if n < 0.0 {
                return Err("fgets: nchar must be a non-negative integer scalar".to_string());
            }
            if (n.fract()).abs() > f64::EPSILON {
                return Err("fgets: nchar must be a non-negative integer scalar".to_string());
            }
            Ok(Some(n as usize))
        }
        _ => Err("fgets: nchar must be a non-negative integer scalar".to_string()),
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

fn read_line(file: &mut File, limit: Option<usize>) -> Result<LineRead, String> {
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

        let read = file
            .read(&mut buffer)
            .map_err(|err| format!("failed to read from file: {err}"))?;
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
                file.seek(SeekFrom::Current(-1))
                    .map_err(|err| format!("failed to seek in file: {err}"))?;
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
            let read_next = file
                .read(&mut next)
                .map_err(|err| format!("failed to read from file: {err}"))?;
            if read_next > 0 {
                if next[0] == b'\n' {
                    newline[1] = b'\n';
                    newline_len = 2;
                    consumed = 2;
                } else {
                    file.seek(SeekFrom::Current(-1))
                        .map_err(|err| format!("failed to seek in file: {err}"))?;
                }
            }

            if data.len().saturating_add(newline_len) > max_bytes {
                file.seek(SeekFrom::Current(-consumed))
                    .map_err(|err| format!("failed to seek in file: {err}"))?;
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

fn bytes_to_char_array(bytes: &[u8], encoding: &str) -> Result<Value, String> {
    let chars = decode_bytes(bytes, encoding)?;
    let cols = chars.len();
    let char_array = CharArray::new(chars, 1, cols)
        .map_err(|e| format!("fgets: failed to build char array: {e}"))?;
    Ok(Value::CharArray(char_array))
}

fn decode_bytes(bytes: &[u8], encoding: &str) -> Result<Vec<char>, String> {
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

    Err(format!("fgets: unsupported encoding '{encoding}'"))
}

fn decode_with_encoding(bytes: &[u8], enc: &'static Encoding) -> Result<Vec<char>, String> {
    let (cow, _, had_errors) = enc.decode(bytes);
    if had_errors {
        return Err(format!(
            "fgets: unable to decode bytes using encoding '{}'",
            enc.name()
        ));
    }
    Ok(cow.chars().collect())
}

fn decode_ascii(bytes: &[u8]) -> Result<Vec<char>, String> {
    if let Some(byte) = bytes.iter().find(|&&b| b > 0x7F) {
        return Err(format!(
            "fgets: byte value {} is outside the ASCII range",
            byte
        ));
    }
    Ok(bytes
        .iter()
        .map(|&b| char::from_u32(b as u32).unwrap())
        .collect())
}

fn numeric_row(bytes: &[u8]) -> Result<Value, String> {
    let data: Vec<f64> = bytes.iter().map(|&b| b as f64).collect();
    let tensor = Tensor::new(data, vec![1, bytes.len()])
        .map_err(|e| format!("fgets: failed to construct numeric array: {e}"))?;
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
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::builtins::io::filetext::{fopen, registry};
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::IntValue;
    use runmat_filesystem as fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_path(prefix: &str) -> PathBuf {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards");
        let filename = format!("{}_{}_{}.tmp", prefix, now.as_secs(), now.subsec_nanos());
        std::env::temp_dir().join(filename)
    }

    fn fopen_path(path: &Path) -> FopenHandle {
        let eval =
            fopen::evaluate(&[Value::from(path.to_string_lossy().to_string())]).expect("fopen");
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

    #[test]
    fn fgets_reads_line_with_newline() {
        registry::reset_for_tests();
        let path = unique_path("fgets_line");
        fs::write(&path, "Hello world\nSecond line\n").unwrap();

        let handle = fopen_path(&path);
        let eval = evaluate(&Value::Num(handle.fid as f64), &[]).expect("fgets");
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

    #[test]
    fn fgets_returns_minus_one_at_eof() {
        registry::reset_for_tests();
        let path = unique_path("fgets_eof");
        fs::write(&path, "line\n").unwrap();
        let handle = fopen_path(&path);

        let _ = evaluate(&Value::Num(handle.fid as f64), &[]).expect("first read");
        let eval = evaluate(&Value::Num(handle.fid as f64), &[]).expect("second read");
        assert_eq!(eval.first_output(), Value::Num(-1.0));
        assert_eq!(eval.outputs()[1], Value::Num(-1.0));

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn fgets_honours_nchar_limit() {
        registry::reset_for_tests();
        let path = unique_path("fgets_limit");
        fs::write(&path, "abcdefghij\nrest\n").unwrap();
        let handle = fopen_path(&path);

        let eval =
            evaluate(&Value::Num(handle.fid as f64), &[Value::Num(5.0)]).expect("limited read");
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

    #[test]
    fn fgets_errors_for_write_only_identifier() {
        registry::reset_for_tests();
        let path = unique_path("fgets_write_only");
        fs::write(&path, "payload").unwrap();
        let eval = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w"),
        ])
        .expect("fopen");
        let open = eval.as_open().expect("open outputs");
        assert!(open.fid >= 3.0);
        let err = evaluate(&Value::Num(open.fid), &[]).expect_err("fgets should fail");
        assert_eq!(err, "fgets: file identifier is not open for reading");
        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn fgets_respects_limit_before_crlf_sequence() {
        registry::reset_for_tests();
        let path = unique_path("fgets_limit_crlf");
        fs::write(&path, b"ABCDE\r\nnext\n").unwrap();
        let handle = fopen_path(&path);

        let first = evaluate(&Value::Num(handle.fid as f64), &[Value::Num(3.0)]).expect("first");
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

        let second = evaluate(&Value::Num(handle.fid as f64), &[]).expect("second");
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

        let third = evaluate(&Value::Num(handle.fid as f64), &[]).expect("third");
        match third.first_output() {
            Value::CharArray(ca) => {
                let text: String = ca.data.iter().collect();
                assert_eq!(text, "next\n");
            }
            other => panic!("expected char array, got {other:?}"),
        }

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn fgets_handles_crlf_newlines() {
        registry::reset_for_tests();
        let path = unique_path("fgets_crlf");
        fs::write(&path, b"first line\r\nsecond\r\n").unwrap();
        let handle = fopen_path(&path);

        let eval = evaluate(&Value::Num(handle.fid as f64), &[]).expect("fgets");
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

    #[test]
    fn fgets_decodes_latin1() {
        registry::reset_for_tests();
        let path = unique_path("fgets_latin1");
        fs::write(&path, [0x48u8, 0x6f, 0x6c, 0x61, 0x20, 0xf1, b'\n']).unwrap();
        let eval = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("r"),
            Value::from("native"),
            Value::from("latin1"),
        ])
        .expect("fopen");
        let open = eval.as_open().expect("open outputs");
        let fid = open.fid as i32;

        let read = evaluate(&Value::Num(fid as f64), &[]).expect("fgets");
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

    #[test]
    fn fgets_nchar_zero_returns_empty_char() {
        registry::reset_for_tests();
        let path = unique_path("fgets_zero");
        fs::write(&path, "hello\n").unwrap();
        let handle = fopen_path(&path);

        let eval = evaluate(
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

    #[test]
    fn fgets_gathers_gpu_scalar_arguments() {
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

            let eval = evaluate(&fid_gpu, &[limit_gpu]).expect("fgets");
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

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
