//! MATLAB-compatible `fprintf` builtin enabling formatted text output to files and standard streams.

use std::io::{self, Write};
use std::sync::{Arc, Mutex as StdMutex};

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::builtins::common::format::{
    decode_escape_sequences, flatten_arguments, format_variadic_with_cursor, ArgCursor,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::io::filetext::registry::{self, FileInfo};
use crate::console::{record_console_output, ConsoleStream};
use crate::gather_if_needed;
use runmat_filesystem::File;

const INVALID_IDENTIFIER_MESSAGE: &str =
    "fprintf: Invalid file identifier. Use fopen to generate a valid file ID.";
const MISSING_FORMAT_MESSAGE: &str = "fprintf: missing format string";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "fprintf",
        builtin_path = "crate::builtins::io::filetext::fprintf"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "fprintf"
category: "io/filetext"
keywords: ["fprintf", "format", "printf", "write file", "stdout", "stderr", "encoding"]
summary: "Write formatted text to files or standard output/error using MATLAB-compatible semantics."
references:
  - https://www.mathworks.com/help/matlab/ref/fprintf.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Formatting and I/O execute on the CPU. GPU tensors are gathered automatically before substitution."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 3
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::filetext::fprintf::tests"
  integration:
    - "builtins::io::filetext::fprintf::tests::fprintf_matrix_column_major"
    - "builtins::io::filetext::fprintf::tests::fprintf_ascii_encoding_errors"
    - "builtins::io::filetext::fprintf::tests::fprintf_gpu_gathers_values"
---

# What does the `fprintf` function do in MATLAB / RunMat?
`fprintf` formats data according to a printf-style template and writes the result to a file,
standard output (`stdout`), or standard error (`stderr`). The builtin mirrors MATLAB behaviour,
including repetition of the format string, column-major traversal of matrix inputs, support for
the special stream names `'stdout'`/`'stderr'`, and the same set of numeric and text conversions
available to `sprintf`.

## How does the `fprintf` function behave in MATLAB / RunMat?
- `fprintf(formatSpec, A, ...)` writes to standard output. The format repeats automatically until
  every element from the argument list has been consumed, traversing arrays in column-major order.
- `fprintf(fid, formatSpec, A, ...)` writes to a file identifier returned by `fopen`. Identifiers
  `1`/`"stdout"` and `2`/`"stderr"` refer to the process streams. Identifiers must be finite,
  non-negative integers.
- The return value is the number of bytes written as a double scalar. Omitting the output argument
  discards it without affecting the write.
- Text arguments (character vectors, string scalars, string arrays, cell arrays of text) are
  expanded in column-major order, matching MATLAB's behaviour.
- Numeric arrays (double, integer, logical, or gpuArray) are flattened column-first and substituted
  element-by-element into the format string. Star (`*`) width and precision arguments are also
  drawn from the flattened stream.
- The text encoding recorded by `fopen` is honoured. ASCII and Latin-1 encodings raise descriptive
  errors when characters cannot be represented. Binary/RAW encodings treat the output as UTF-8,
  mirroring MATLAB's default on modern systems.
- Arguments that reside on the GPU are gathered to the host before formatting. Formatting itself is
  always executed on the CPU.

## `fprintf` Function GPU Execution Behaviour
`fprintf` is a residency sink. Any argument containing `gpuArray` data is gathered via the active
acceleration provider before formatting. No GPU kernels are launched. When no provider is
registered, the builtin raises the same descriptive error used by other sinks (`gather: no
acceleration provider registered`).

## Examples of using the `fprintf` function in MATLAB / RunMat

### Write Formatted Text To A File
```matlab
[fid, msg] = fopen('report.txt', 'w');
assert(fid ~= -1, msg);
fprintf(fid, 'Total: %d (%.2f%%)\n', 42, 87.5);
fclose(fid);
```
Expected contents of `report.txt`:
```matlab
Total: 42 (87.50%)
```

### Use Standard Output Without An Explicit File Identifier
```matlab
fprintf('Processing %s ...\n', datestr(now, 0));
```
Expected console output:
```matlab
Processing 07-Jan-2025 23:14:55 ...
```

### Write To Standard Error Using The Stream Name
```matlab
fprintf('stderr', 'Warning: iteration limit reached (%d steps)\n', iter);
```
Expected console output (sent to stderr):
```matlab
Warning: iteration limit reached (250 steps)
```

### Format A Matrix In Column-Major Order
```matlab
A = [1 2 3; 4 5 6];
fprintf('%d %d\n', A);
```
Expected console output:
```matlab
1 4
2 5
3 6
```

### Respect File Encoding Constraints
```matlab
[fid, msg] = fopen('ascii.txt', 'w', 'native', 'ascii');
if fid == -1, error(msg); end
try
    fprintf(fid, 'café\n');
catch err
    disp(err.message);
end
fclose(fid);
```
Expected console output:
```matlab
fprintf: character 'é' (U+00E9) cannot be encoded as ASCII
```

### Format GPU-Resident Data Transparently
```matlab
G = gpuArray([1.2 3.4 5.6]);
[fid, msg] = fopen('gpu.txt', 'w');
assert(fid ~= -1, msg);
fprintf(fid, '%.1f,', G);
fclose(fid);
```
Expected contents of `gpu.txt`:
```matlab
1.2,3.4,5.6,
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You can pass `gpuArray` inputs directly—`fprintf` gathers them back to host memory before formatting.
No provider-specific hooks are required and outputs always reside on the CPU. This mirrors MATLAB,
where explicit `gather` calls are unnecessary when writing to files or console streams.

## FAQ

### Does `fprintf` return the number of characters or bytes?
It returns the number of bytes written. This may differ from the number of characters when using
multi-byte encodings such as UTF-8.

### Can I use `'stdout'` or `'stderr'` instead of numeric identifiers?
Yes. The strings `'stdout'` and `'stderr'` (any case) map to identifiers `1` and `2` respectively,
matching MATLAB.

### What happens if the file was opened read-only?
`fprintf` raises `fprintf: file is not open for writing`. Ensure the permission string passed to
`fopen` includes `'w'`, `'a'`, or `'+'`.

### Which encodings are supported?
`fprintf` honours the encoding recorded by `fopen`. UTF-8 (default), ASCII, and Latin-1 are
supported explicitly. Other labels fall back to UTF-8 behaviour.

### How are multi-dimensional arrays handled?
Arguments are flattened in column-major order. The format string repeats until every element has
been consumed, just like MATLAB.

### Does `fprintf` flush the stream?
The builtin delegates to Rust's buffered writers. Files are flushed when closed; standard streams
inherit the host buffering policy.

### What if the format string contains no conversions?
Literal format strings are written once. Supplying additional arguments raises
`fprintf: formatSpec contains no conversion specifiers but additional arguments were supplied`.

### Are cell arrays supported?
Yes. Cell arrays containing supported scalar or text values are flattened in column-major order
before formatting.

### Can I mix numeric and text arguments?
Absolutely. Numeric, logical, and text inputs can be interleaved. Star width/precision arguments
use the same flattened stream.

### How do I suppress the return value?
Ignore it, just as in MATLAB. Omitting the output argument does not change the write behaviour.

## See Also
[sprintf](./sprintf), [compose](./compose),
[fopen](./fopen), [fclose](./fclose), [fwrite](./fwrite), [fileread](./fileread)

## Source & Feedback
- Implementation: `crates/runmat-runtime/src/builtins/io/filetext/fprintf.rs`
- Found a behavioural difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose)
  with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::filetext::fprintf")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fprintf",
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
    notes: "Host-only text I/O. Arguments residing on the GPU are gathered before formatting.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::filetext::fprintf")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fprintf",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Formatting is a side-effecting sink and never participates in fusion.",
};

/// Result of evaluating `fprintf`.
#[derive(Debug)]
pub struct FprintfEval {
    bytes_written: usize,
}

impl FprintfEval {
    /// Number of bytes emitted by the write.
    pub fn bytes_written(&self) -> usize {
        self.bytes_written
    }
}

/// Evaluate the `fprintf` builtin without going through the dispatcher.
pub fn evaluate(args: &[Value]) -> Result<FprintfEval, String> {
    if args.is_empty() {
        return Err("fprintf: not enough input arguments".to_string());
    }

    // Gather all arguments to host first
    let mut all: Vec<Value> = Vec::with_capacity(args.len());
    for v in args {
        all.push(gather_value(v)?);
    }

    // Locate the first valid formatSpec anywhere in the list
    let mut fmt_idx: Option<usize> = None;
    let mut format_string_val: Option<String> = None;
    for (i, value) in all.iter().enumerate() {
        // Never interpret a stream label ('stdout'/'stderr') as the format string
        if match_stream_label(value).is_some() {
            continue;
        }
        if let Some(Value::String(s)) = coerce_to_format_string(value)? {
            fmt_idx = Some(i);
            format_string_val = Some(s);
            break;
        }
    }
    let fmt_idx = fmt_idx.ok_or_else(|| MISSING_FORMAT_MESSAGE.to_string())?;
    let raw_format = format_string_val.unwrap();

    // Determine output target by scanning only arguments BEFORE the format
    let mut target_idx: Option<usize> = None;
    let mut target: OutputTarget = OutputTarget::Stdout;
    // Prefer explicit stream labels over numeric fids if both appear
    let mut first_stream: Option<(usize, SpecialStream)> = None;
    for (i, value) in all.iter().enumerate().take(fmt_idx) {
        if let Some(stream) = match_stream_label(value) {
            first_stream = Some((i, stream));
            break;
        }
    }
    if let Some((idx, stream)) = first_stream {
        target_idx = Some(idx);
        target = match stream {
            SpecialStream::Stdout => OutputTarget::Stdout,
            SpecialStream::Stderr => OutputTarget::Stderr,
        };
    } else {
        // Try to parse a numeric fid that appears before the format
        for (i, value) in all.iter().enumerate().take(fmt_idx) {
            if matches!(value, Value::Num(_) | Value::Int(_) | Value::Tensor(_)) {
                if let Ok(fid) = parse_fid(value) {
                    target_idx = Some(i);
                    target = target_from_fid(fid)?;
                    break;
                }
            }
        }
    }

    // Remaining arguments are data, excluding the chosen target and the format
    let mut data_args: Vec<Value> = Vec::with_capacity(all.len().saturating_sub(1));
    for (i, v) in all.into_iter().enumerate() {
        if i == fmt_idx {
            continue;
        }
        if let Some(tidx) = target_idx {
            if i == tidx {
                continue;
            }
        }
        data_args.push(v);
    }

    let format_string = decode_escape_sequences("fprintf", &raw_format)?;
    let flattened_args = flatten_arguments(&data_args, "fprintf")?;
    let rendered = format_with_repetition(&format_string, &flattened_args)?;
    let bytes = encode_output(&rendered, target.encoding_label())?;
    target.write(&bytes)?;
    Ok(FprintfEval {
        bytes_written: bytes.len(),
    })
}

// kind_of was used for debugging logs; removed to avoid dead code in production builds.

fn try_tensor_char_row_as_string(value: &Value) -> Option<Result<String, String>> {
    match value {
        Value::Tensor(t) => {
            let is_row = (t.shape.len() == 2 && t.shape[0] == 1 && t.data.len() == t.shape[1])
                || (t.shape.len() == 1 && t.data.len() == t.shape[0]);
            if is_row {
                let mut out = String::with_capacity(t.data.len());
                for &code in &t.data {
                    if !code.is_finite() {
                        return Some(Err(
                            "fprintf: formatSpec must be a character row vector or string scalar"
                                .to_string(),
                        ));
                    }
                    let v = code as u32;
                    // Allow full Unicode range; MATLAB chars are UTF-16 but format strings are ASCII-compatible typically
                    if let Some(ch) = char::from_u32(v) {
                        out.push(ch);
                    } else {
                        return Some(Err(
                            "fprintf: formatSpec contains invalid character code".to_string()
                        ));
                    }
                }
                return Some(Ok(out));
            }
            None
        }
        _ => None,
    }
}

fn coerce_to_format_string(value: &Value) -> Result<Option<Value>, String> {
    match value {
        Value::String(s) => Ok(Some(Value::String(s.clone()))),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(Some(Value::String(sa.data[0].clone()))),
        Value::CharArray(ca) => {
            let s: String = ca.data.iter().collect();
            Ok(Some(Value::String(s)))
        }
        Value::Tensor(t) => {
            // Only accept numeric codepoint vectors of length >= 2 as formatSpec.
            // This avoids misinterpreting stray 1x1 numerics (e.g., accidental stack values)
            // as a valid format string.
            if t.data.len() >= 2 {
                match try_tensor_char_row_as_string(value) {
                    Some(Ok(s)) => Ok(Some(Value::String(s))),
                    Some(Err(e)) => Err(e),
                    None => Ok(None),
                }
            } else {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}

#[runtime_builtin(
    name = "fprintf",
    category = "io/filetext",
    summary = "Write formatted text to files or standard streams.",
    keywords = "fprintf,format,printf,io",
    accel = "cpu",
    sink = true,
    suppress_auto_output = true,
    builtin_path = "crate::builtins::io::filetext::fprintf"
)]
fn fprintf_builtin(first: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let mut args = Vec::with_capacity(rest.len() + 1);
    args.push(first);
    args.extend(rest);
    let eval = evaluate(&args)?;
    Ok(Value::Num(eval.bytes_written() as f64))
}

#[derive(Clone, Copy)]
enum SpecialStream {
    Stdout,
    Stderr,
}

enum OutputTarget {
    Stdout,
    Stderr,
    File {
        handle: Arc<StdMutex<File>>,
        encoding: String,
    },
}

impl OutputTarget {
    fn encoding_label(&self) -> Option<&str> {
        match self {
            OutputTarget::Stdout | OutputTarget::Stderr => None,
            OutputTarget::File { encoding, .. } => Some(encoding.as_str()),
        }
    }

    fn write(&self, bytes: &[u8]) -> Result<(), String> {
        match self {
            OutputTarget::Stdout => {
                let mut stdout = io::stdout().lock();
                stdout
                    .write_all(bytes)
                    .map_err(|err| format!("fprintf: failed to write to stdout ({err})"))?;
                record_console_chunk(ConsoleStream::Stdout, bytes);
                Ok(())
            }
            OutputTarget::Stderr => {
                let mut stderr = io::stderr().lock();
                stderr
                    .write_all(bytes)
                    .map_err(|err| format!("fprintf: failed to write to stderr ({err})"))?;
                record_console_chunk(ConsoleStream::Stderr, bytes);
                Ok(())
            }
            OutputTarget::File { handle, .. } => {
                let mut guard = handle.lock().map_err(|_| {
                    "fprintf: failed to lock file handle (poisoned mutex)".to_string()
                })?;
                guard
                    .write_all(bytes)
                    .map_err(|err| format!("fprintf: failed to write to file ({err})"))
            }
        }
    }
}

fn record_console_chunk(stream: ConsoleStream, bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    let text = String::from_utf8_lossy(bytes).to_string();
    record_console_output(stream, text);
}

fn gather_value(value: &Value) -> Result<Value, String> {
    gather_if_needed(value).map_err(|e| format!("fprintf: {e}"))
}

#[allow(dead_code)]
fn resolve_target<'a>(
    first: &'a Value,
    rest: &'a [Value],
) -> Result<(OutputTarget, &'a Value, &'a [Value]), String> {
    if let Some(stream) = match_stream_label(first) {
        if rest.is_empty() {
            return Err(MISSING_FORMAT_MESSAGE.to_string());
        }
        let target = match stream {
            SpecialStream::Stdout => OutputTarget::Stdout,
            SpecialStream::Stderr => OutputTarget::Stderr,
        };
        return Ok((target, &rest[0], &rest[1..]));
    }

    match first {
        Value::Num(_) | Value::Int(_) => {
            let fid = parse_fid(first)?;
            resolve_fid_target(fid, rest)
        }
        Value::Tensor(t) => {
            // If this looks like a 1xN row of character codes, treat it as a format string to stdout
            if t.shape.len() == 2 && t.shape[0] == 1 && t.data.len() == t.shape[1] {
                return Ok((OutputTarget::Stdout, first, rest));
            }
            // Otherwise only scalar numeric tensors are valid as fids
            let fid = parse_fid(first)?;
            resolve_fid_target(fid, rest)
        }
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => {
            let target = OutputTarget::Stdout;
            Ok((target, first, rest))
        }
        // Be permissive: if it's not a numeric fid or stream label, interpret as format string to stdout
        _ => Ok((OutputTarget::Stdout, first, rest)),
    }
}

fn resolve_fid_target(
    fid: i32,
    rest: &[Value],
) -> Result<(OutputTarget, &Value, &[Value]), String> {
    if rest.is_empty() {
        return Err(MISSING_FORMAT_MESSAGE.to_string());
    }
    if fid < 0 {
        return Err("fprintf: file identifier must be non-negative".to_string());
    }
    match fid {
        0 => Err("fprintf: file identifier 0 (stdin) is not writable".to_string()),
        1 => Ok((OutputTarget::Stdout, &rest[0], &rest[1..])),
        2 => Ok((OutputTarget::Stderr, &rest[0], &rest[1..])),
        _ => {
            let info =
                registry::info_for(fid).ok_or_else(|| INVALID_IDENTIFIER_MESSAGE.to_string())?;
            ensure_writable(&info)?;
            let handle =
                registry::take_handle(fid).ok_or_else(|| INVALID_IDENTIFIER_MESSAGE.to_string())?;
            Ok((
                OutputTarget::File {
                    handle,
                    encoding: info.encoding.clone(),
                },
                &rest[0],
                &rest[1..],
            ))
        }
    }
}

fn target_from_fid(fid: i32) -> Result<OutputTarget, String> {
    if fid < 0 {
        return Err("fprintf: file identifier must be non-negative".to_string());
    }
    match fid {
        0 => Err("fprintf: file identifier 0 (stdin) is not writable".to_string()),
        1 => Ok(OutputTarget::Stdout),
        2 => Ok(OutputTarget::Stderr),
        _ => {
            let info =
                registry::info_for(fid).ok_or_else(|| INVALID_IDENTIFIER_MESSAGE.to_string())?;
            ensure_writable(&info)?;
            let handle =
                registry::take_handle(fid).ok_or_else(|| INVALID_IDENTIFIER_MESSAGE.to_string())?;
            Ok(OutputTarget::File {
                handle,
                encoding: info.encoding.clone(),
            })
        }
    }
}

fn parse_fid(value: &Value) -> Result<i32, String> {
    let scalar = match value {
        Value::Num(n) => *n,
        Value::Int(int) => int.to_f64(),
        Value::Tensor(t) => {
            if t.shape == vec![1, 1] && t.data.len() == 1 {
                t.data[0]
            } else {
                return Err("fprintf: file identifier must be numeric".to_string());
            }
        }
        _ => return Err("fprintf: file identifier must be numeric".to_string()),
    };
    if !scalar.is_finite() {
        return Err("fprintf: file identifier must be finite".to_string());
    }
    if (scalar.fract().abs()) > f64::EPSILON {
        return Err("fprintf: file identifier must be an integer".to_string());
    }
    Ok(scalar as i32)
}

fn ensure_writable(info: &FileInfo) -> Result<(), String> {
    let permission = info.permission.to_ascii_lowercase();
    if permission.contains('w') || permission.contains('a') || permission.contains('+') {
        Ok(())
    } else {
        Err("fprintf: file is not open for writing".to_string())
    }
}

fn match_stream_label(value: &Value) -> Option<SpecialStream> {
    let candidate = match value {
        Value::String(s) => s.trim().to_string(),
        Value::CharArray(ca) if ca.rows == 1 => {
            ca.data.iter().collect::<String>().trim().to_string()
        }
        Value::StringArray(sa) if sa.data.len() == 1 => sa.data[0].trim().to_string(),
        _ => return None,
    };
    match candidate.to_ascii_lowercase().as_str() {
        "stdout" => Some(SpecialStream::Stdout),
        "stderr" => Some(SpecialStream::Stderr),
        _ => None,
    }
}

fn format_with_repetition(format: &str, args: &[Value]) -> Result<String, String> {
    let mut cursor = ArgCursor::new(args);
    let mut out = String::new();
    loop {
        let step = format_variadic_with_cursor(format, &mut cursor).map_err(remap_format_error)?;
        out.push_str(&step.output);
        if step.consumed == 0 {
            if cursor.remaining() > 0 {
                return Err("fprintf: formatSpec contains no conversion specifiers but additional arguments were supplied".to_string());
            }
            break;
        }
        if cursor.remaining() == 0 {
            break;
        }
    }
    Ok(out)
}

fn remap_format_error(err: String) -> String {
    if err.contains("sprintf") {
        err.replace("sprintf", "fprintf")
    } else {
        err
    }
}

fn encode_output(text: &str, encoding: Option<&str>) -> Result<Vec<u8>, String> {
    let label = encoding
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .unwrap_or("utf-8");
    let lower = label.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "utf-8" | "utf8" | "unicode" | "auto" | "default" | "system"
    ) {
        Ok(text.as_bytes().to_vec())
    } else if matches!(
        lower.as_str(),
        "ascii" | "us-ascii" | "us_ascii" | "usascii"
    ) {
        encode_ascii(text)
    } else if matches!(
        lower.as_str(),
        "latin1" | "latin-1" | "latin_1" | "iso-8859-1" | "iso8859-1" | "iso88591"
    ) {
        encode_latin1(text, label)
    } else {
        Ok(text.as_bytes().to_vec())
    }
}

fn encode_ascii(text: &str) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::with_capacity(text.len());
    for ch in text.chars() {
        if ch as u32 > 0x7F {
            return Err(format!(
                "fprintf: character '{}' (U+{:04X}) cannot be encoded as ASCII",
                ch, ch as u32
            ));
        }
        bytes.push(ch as u8);
    }
    Ok(bytes)
}

fn encode_latin1(text: &str, label: &str) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::with_capacity(text.len());
    for ch in text.chars() {
        if ch as u32 > 0xFF {
            return Err(format!(
                "fprintf: character '{}' (U+{:04X}) cannot be encoded as {}",
                ch, ch as u32, label
            ));
        }
        bytes.push(ch as u8);
    }
    Ok(bytes)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::builtins::io::filetext::{fclose, fopen, registry};
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, Tensor};
    use runmat_filesystem::{self as fs, File};
    use runmat_time::system_time_now;
    use std::io::Read;
    use std::path::PathBuf;
    use std::time::UNIX_EPOCH;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fprintf_matrix_column_major() {
        registry::reset_for_tests();
        let path = unique_path("fprintf_matrix");
        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let args = vec![
            Value::Num(fid as f64),
            Value::String("%d %d\n".to_string()),
            Value::Tensor(tensor),
        ];
        let eval = evaluate(&args).expect("fprintf");
        assert_eq!(eval.bytes_written(), 12);

        fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();

        let contents = fs::read_to_string(&path).expect("read");
        assert_eq!(contents, "1 4\n2 5\n3 6\n");
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fprintf_ascii_encoding_errors() {
        registry::reset_for_tests();
        let path = unique_path("fprintf_ascii");
        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w"),
            Value::from("native"),
            Value::from("ascii"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let args = vec![
            Value::Num(fid as f64),
            Value::String("%s".to_string()),
            Value::String("café".to_string()),
        ];
        let err = evaluate(&args).expect_err("fprintf should reject ASCII-incompatible text");
        assert!(err.contains("cannot be encoded as ASCII"), "{err}");

        fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fprintf_gpu_gathers_values() {
        registry::reset_for_tests();
        let path = unique_path("fprintf_gpu");

        test_support::with_test_provider(|provider| {
            registry::reset_for_tests();
            let open = fopen::evaluate(&[
                Value::from(path.to_string_lossy().to_string()),
                Value::from("w"),
            ])
            .expect("fopen");
            let fid = open.as_open().unwrap().fid as i32;

            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let args = vec![
                Value::Num(fid as f64),
                Value::String("%.1f,".to_string()),
                Value::GpuTensor(handle),
            ];
            let eval = evaluate(&args).expect("fprintf");
            assert_eq!(eval.bytes_written(), 12);

            fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();
        });

        let mut file = File::open(&path).expect("open");
        let mut contents = String::new();
        file.read_to_string(&mut contents).expect("read");
        assert_eq!(contents, "1.0,2.0,3.0,");
        fs::remove_file(path).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fprintf_missing_format_errors() {
        let err = evaluate(&[Value::Num(1.0)]).expect_err("fprintf should require format");
        assert!(err.contains("missing format string"), "{err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fprintf_literal_with_extra_args_errors() {
        let err = evaluate(&[
            Value::String("literal text".to_string()),
            Value::Int(IntValue::I32(1)),
        ])
        .expect_err("fprintf should reject extra args without conversions");
        assert!(err.contains("contains no conversion specifiers"), "{err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fprintf_invalid_identifier_errors() {
        let err = evaluate(&[Value::Num(99.0), Value::String("value".to_string())])
            .expect_err("fprintf should reject unknown fid");
        assert!(err.contains("Invalid file identifier"), "{err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fprintf_read_only_error() {
        registry::reset_for_tests();
        let path = unique_path("fprintf_read_only");
        fs::write(&path, b"readonly").unwrap();
        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("r"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;
        let err = evaluate(&[Value::Num(fid as f64), Value::String("text".to_string())])
            .expect_err("fprintf should reject read-only handles");
        assert!(err.contains("not open for writing"), "{err}");

        fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fprintf_doc_examples_parse() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    fn unique_path(prefix: &str) -> PathBuf {
        let nanos = system_time_now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let filename = format!("runmat_{prefix}_{nanos}.txt");
        std::env::temp_dir().join(filename)
    }
}
