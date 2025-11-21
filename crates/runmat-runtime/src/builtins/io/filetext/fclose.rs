//! MATLAB-compatible `fclose` builtin for RunMat.
//!
//! Mirrors MATLAB semantics for closing individual files, vectors of file
//! identifiers, or all open files. The implementation integrates with the
//! shared file registry managed by `fopen` and always executes on the host.

use runmat_builtins::{CharArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::io::filetext::registry;
use crate::{gather_if_needed, register_builtin_fusion_spec, register_builtin_gpu_spec};

#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;

const INVALID_IDENTIFIER_MESSAGE: &str =
    "Invalid file identifier. Use fopen to generate a valid file ID.";

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "fclose"
category: "io/filetext"
keywords: ["fclose", "file", "close", "io", "identifier"]
summary: "Close one file, multiple files, or all files opened with fopen."
references:
  - https://www.mathworks.com/help/matlab/ref/fclose.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs entirely on the CPU. Arguments residing on the GPU are gathered automatically; file handles live on the host."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::filetext::fclose::tests"
  integration: "builtins::io::filetext::fclose::tests::fclose_all_closes_everything"
---

# What does the `fclose` function do in MATLAB / RunMat?
`fclose` closes files that were previously opened with `fopen`. Pass a file
identifier, an array of identifiers, or the keyword `'all'` to close the desired
handles. The first output (`status`) is `0` when the close succeeds and `-1`
otherwise. The optional second output (`message`) contains diagnostic text when
an identifier is invalid.

## How does the `fclose` function behave in MATLAB / RunMat?
- `status = fclose(fid)` closes the file represented by the numeric identifier
  `fid`. The status is `0` on success and `-1` if the identifier is invalid.
- `status = fclose([fid1 fid2 ...])` closes a vector of identifiers. If any
  identifier is invalid, status is `-1` and `message` explains the failure.
- `status = fclose('all')` (or `fclose()` with no arguments) closes every open
  file except the standard streams (0, 1, 2).
- `[status, message] = fclose(...)` returns the diagnostic message. On success,
  `message` is an empty character vector.
- Identifiers 0, 1, and 2 refer to standard input, output, and error. `fclose`
  treats them as already-open handles and simply returns success.
- RunMat keeps file metadata in a registry shared with `fopen`, ensuring
  MATLAB-compatible behaviour across subsequent I/O builtins.

## Examples of using the `fclose` function in MATLAB / RunMat

### Close a file after writing data
```matlab
[fid, msg] = fopen('results.txt', 'w');
if fid == -1
    error('Failed to open file: %s', msg);
end
fprintf(fid, 'Simulation complete\n');
status = fclose(fid);
```

Expected output:
```matlab
status = 0;
```

### Close all open files at once
```matlab
% Close every file except stdin/stdout/stderr
status = fclose('all');
```

Expected output:
```matlab
status = 0;
```

### Handle invalid file identifiers gracefully
```matlab
[status, message] = fclose(9999);
if status == -1
    fprintf('Close failed: %s\n', message);
end
```

Expected output:
```matlab
status = -1;
message = 'Invalid file identifier. Use fopen to generate a valid file ID.';
```

### Close multiple file identifiers together
```matlab
fids = fopen('all');
status = fclose(fids);
```

Expected output:
```matlab
status = 0;
```

### Detect failures with the second output
```matlab
[fid, msg] = fopen('data.bin', 'r');
assert(fid ~= -1, msg);
fclose(fid);
[status, message] = fclose(fid);  % closes again, returns -1 and an error string
```

Expected output:
```matlab
status = -1;
message = 'Invalid file identifier. Use fopen to generate a valid file ID.';
```

### Close files using the no-argument form
```matlab
% Equivalent to fclose('all')
status = fclose();
```

Expected output:
```matlab
status = 0;
```

## `fclose` GPU Execution Behaviour
`fclose` does not perform GPU computation. If the argument resides on a GPU
array (for example, `'all'` stored in a `gpuArray`), RunMat gathers the value to
host memory before dispatching the host-only close logic.

## FAQ

### What values can I pass to `fclose`?
Pass a numeric file identifier (scalar or array) returned by `fopen`, or the
keyword `'all'`. Calling `fclose()` with no arguments is equivalent to
`fclose('all')`.

### What does the status code mean?
`0` indicates that every requested identifier was successfully processed.
`-1` means that at least one identifier was invalid; the optional second output
contains the diagnostic message.

### Does `fclose` close standard input/output?
Identifiers 0, 1, and 2 always refer to the process standard streams. `fclose`
accepts them but leaves the streams open, returning a success status.

### Can I call `fclose` multiple times on the same identifier?
Yes. The first call closes the file and subsequent calls return status `-1`
with the message `"Invalid file identifier. Use fopen to generate a valid file ID."`

### Does `fclose` flush buffered writes?
Closing a file flushes buffered writes and releases the underlying operating
system descriptor, matching MATLAB behaviour.

### Do I need to close files explicitly when using GPU arrays?
Yes. GPU residency does not change the lifecycle of file handles. Use `fclose`
to release identifiers created with `fopen` regardless of where the arguments
reside.

### Can `fclose` close files opened by other processes?
No. It only closes identifiers that the current RunMat process opened via
`fopen`.

## See Also
[fopen](./fopen), [fread](./fread), [fwrite](./fwrite), [fileread](./fileread), [filewrite](./filewrite)

## Source & Feedback
- The implementation lives at `crates/runmat-runtime/src/builtins/io/filetext/fclose.rs`.
- Found a behavioural mismatch? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal repro.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fclose",
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
    notes:
        "Host-only operation: closes identifiers stored in the shared file registry; GPU inputs are gathered automatically.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fclose",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "File I/O is not eligible for fusion; metadata is registered for completeness.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("fclose", DOC_MD);

#[runtime_builtin(
    name = "fclose",
    category = "io/filetext",
    summary = "Close one file, multiple files, or all files opened with fopen.",
    keywords = "fclose,file,close,io,identifier",
    accel = "cpu"
)]
fn fclose_builtin(args: Vec<Value>) -> Result<Value, String> {
    let eval = evaluate(&args)?;
    Ok(eval.first_output())
}

#[derive(Debug, Clone)]
pub struct FcloseEval {
    status: f64,
    message: String,
}

impl FcloseEval {
    fn success() -> Self {
        Self {
            status: 0.0,
            message: String::new(),
        }
    }

    fn failure(message: String) -> Self {
        Self {
            status: -1.0,
            message,
        }
    }

    pub fn first_output(&self) -> Value {
        Value::Num(self.status)
    }

    pub fn outputs(&self) -> Vec<Value> {
        vec![Value::Num(self.status), char_array_value(&self.message)]
    }

    #[cfg(test)]
    pub(crate) fn status(&self) -> f64 {
        self.status
    }

    #[cfg(test)]
    pub(crate) fn message(&self) -> &str {
        &self.message
    }
}

pub fn evaluate(args: &[Value]) -> Result<FcloseEval, String> {
    let gathered = gather_args(args)?;
    match gathered.len() {
        0 => Ok(close_all()),
        1 => handle_single_argument(&gathered[0]),
        _ => Err("fclose: too many input arguments".to_string()),
    }
}

fn handle_single_argument(value: &Value) -> Result<FcloseEval, String> {
    if matches_keyword(value, "all") {
        return Ok(close_all());
    }
    let fids = collect_file_ids(value).map_err(|err| format!("fclose: {err}"))?;
    Ok(close_fids(&fids))
}

fn close_all() -> FcloseEval {
    let infos = registry::list_infos();
    for info in infos {
        if info.id >= 3 {
            let _ = registry::close(info.id);
        }
    }
    FcloseEval::success()
}

fn close_fids(fids: &[i32]) -> FcloseEval {
    if fids.is_empty() {
        return FcloseEval::success();
    }
    let mut status_ok = true;
    let mut message = String::new();
    for &fid in fids {
        if fid < 0 {
            status_ok = false;
            if message.is_empty() {
                message = INVALID_IDENTIFIER_MESSAGE.to_string();
            }
            continue;
        }
        if fid < 3 {
            continue;
        }
        if registry::close(fid).is_none() {
            status_ok = false;
            if message.is_empty() {
                message = INVALID_IDENTIFIER_MESSAGE.to_string();
            }
        }
    }
    if status_ok {
        FcloseEval::success()
    } else {
        FcloseEval::failure(message)
    }
}

fn collect_file_ids(value: &Value) -> Result<Vec<i32>, String> {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok(vec![parse_scalar_fid(value)?]),
        Value::Tensor(t) => {
            let mut ids = Vec::with_capacity(t.data.len());
            for &n in &t.data {
                ids.push(parse_fid_from_f64(n)?);
            }
            Ok(ids)
        }
        Value::LogicalArray(la) => {
            let mut ids = Vec::with_capacity(la.data.len());
            for &b in &la.data {
                let v = if b != 0 { 1 } else { 0 };
                ids.push(v);
            }
            Ok(ids)
        }
        Value::Cell(ca) => {
            let mut ids = Vec::with_capacity(ca.data.len());
            for ptr in &ca.data {
                let nested = collect_file_ids(ptr)?;
                ids.extend(nested);
            }
            Ok(ids)
        }
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) => {
            Err("file identifier must be numeric or 'all'".to_string())
        }
        _ => Err("file identifier must be numeric or 'all'".to_string()),
    }
}

fn parse_scalar_fid(value: &Value) -> Result<i32, String> {
    match value {
        Value::Int(i) => {
            let v = i.to_i64();
            if v < i32::MIN as i64 || v > i32::MAX as i64 {
                return Err("file identifier is out of range".to_string());
            }
            Ok(v as i32)
        }
        Value::Num(n) => parse_fid_from_f64(*n),
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        _ => Err("file identifier must be numeric or 'all'".to_string()),
    }
}

fn parse_fid_from_f64(value: f64) -> Result<i32, String> {
    if !value.is_finite() {
        return Err("file identifier must be finite".to_string());
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err("file identifier must be an integer".to_string());
    }
    if rounded < i32::MIN as f64 || rounded > i32::MAX as f64 {
        return Err("file identifier is out of range".to_string());
    }
    Ok(rounded as i32)
}

fn gather_args(args: &[Value]) -> Result<Vec<Value>, String> {
    let mut gathered = Vec::with_capacity(args.len());
    for value in args {
        gathered.push(gather_if_needed(value).map_err(|e| format!("fclose: {e}"))?);
    }
    Ok(gathered)
}

fn matches_keyword(value: &Value, keyword: &str) -> bool {
    extract_scalar_string(value)
        .map(|text| text.eq_ignore_ascii_case(keyword))
        .unwrap_or(false)
}

fn extract_scalar_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        _ => None,
    }
}

fn char_array_value(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::io::filetext::{fopen, registry};
    use once_cell::sync::Lazy;
    use runmat_builtins::{CellArray, LogicalArray, StringArray, Tensor};
    use std::fs;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::{Mutex, MutexGuard};
    use std::time::{SystemTime, UNIX_EPOCH};

    static REGISTRY_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    fn registry_guard() -> MutexGuard<'static, ()> {
        REGISTRY_LOCK.lock().unwrap()
    }

    fn unique_path(prefix: &str) -> PathBuf {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards");
        let filename = format!("{}_{}_{}.tmp", prefix, now.as_secs(), now.subsec_nanos());
        std::env::temp_dir().join(filename)
    }

    fn open_temp_file(prefix: &str) -> (f64, PathBuf) {
        let path = unique_path(prefix);
        {
            let mut file = fs::File::create(&path).unwrap();
            writeln!(&mut file, "data").unwrap();
        }
        let eval =
            fopen::evaluate(&[Value::from(path.to_string_lossy().to_string())]).expect("fopen");
        let fid = eval.as_open().unwrap().fid;
        assert!(fid >= 3.0, "expected valid file identifier");
        (fid, path)
    }

    #[test]
    fn fclose_closes_single_file() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let (fid, path) = open_temp_file("fclose_single");
        let eval = evaluate(&[Value::Num(fid)]).expect("fclose");
        assert_eq!(eval.status(), 0.0);
        assert!(eval.message().is_empty());
        assert!(registry::info_for(fid as i32).is_none());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn fclose_invalid_identifier_returns_error() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let eval = evaluate(&[Value::Num(9999.0)]).expect("fclose");
        assert_eq!(eval.status(), -1.0);
        assert_eq!(eval.message(), INVALID_IDENTIFIER_MESSAGE);
    }

    #[test]
    fn fclose_all_closes_everything() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let (fid, path) = open_temp_file("fclose_all");
        let eval = evaluate(&[Value::from("all")]).expect("fclose all");
        assert_eq!(eval.status(), 0.0);
        assert!(registry::info_for(fid as i32).is_none());
        let infos = registry::list_infos();
        assert!(infos.iter().all(|info| info.id < 3));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn fclose_no_args_closes_all() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let (fid, path) = open_temp_file("fclose_no_args");
        let eval = evaluate(&[]).expect("fclose");
        assert_eq!(eval.status(), 0.0);
        assert!(registry::info_for(fid as i32).is_none());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn fclose_vector_of_fids_closes_each() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let path1 = unique_path("fclose_vec1");
        fs::write(&path1, "a").unwrap();
        let fid1 = fopen::evaluate(&[Value::from(path1.to_string_lossy().to_string())])
            .expect("open 1")
            .as_open()
            .unwrap()
            .fid;
        let path2 = unique_path("fclose_vec2");
        fs::write(&path2, "b").unwrap();
        let fid2 = fopen::evaluate(&[Value::from(path2.to_string_lossy().to_string())])
            .expect("open 2")
            .as_open()
            .unwrap()
            .fid;
        let tensor = Tensor::new(vec![fid1, fid2], vec![2, 1]).expect("tensor construction");
        let eval = evaluate(&[Value::Tensor(tensor)]).expect("fclose");
        assert_eq!(eval.status(), 0.0);
        assert!(registry::info_for(fid1 as i32).is_none());
        assert!(registry::info_for(fid2 as i32).is_none());
        fs::remove_file(path1).unwrap();
        fs::remove_file(path2).unwrap();
    }

    #[test]
    fn fclose_repeat_returns_error_message() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let (fid, path) = open_temp_file("fclose_repeat");
        let first = evaluate(&[Value::Num(fid)]).expect("fclose");
        assert_eq!(first.status(), 0.0);
        let second = evaluate(&[Value::Num(fid)]).expect("fclose second");
        assert_eq!(second.status(), -1.0);
        assert_eq!(second.message(), INVALID_IDENTIFIER_MESSAGE);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn fclose_standard_stream_bool_argument() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let eval = evaluate(&[Value::Bool(true)]).expect("fclose stdout");
        assert_eq!(eval.status(), 0.0);
        assert!(eval.message().is_empty());
        let outputs = eval.outputs();
        assert_eq!(outputs.len(), 2);
        assert!(matches!(outputs[1], Value::CharArray(ref ca) if ca.rows == 1 && ca.cols == 0));
    }

    #[test]
    fn fclose_logical_array_converts_to_numeric_ids() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let logical = LogicalArray::new(vec![1u8, 0u8, 1u8], vec![3]).expect("logical array");
        let eval = evaluate(&[Value::LogicalArray(logical)]).expect("fclose logical");
        assert_eq!(eval.status(), 0.0);
        assert!(eval.message().is_empty());
    }

    #[test]
    fn fclose_cell_array_closes_each_entry() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let (fid1, path1) = open_temp_file("fclose_cell1");
        let (fid2, path2) = open_temp_file("fclose_cell2");
        let cell = CellArray::new(vec![Value::Num(fid1), Value::Num(fid2)], 1, 2).expect("cell");
        let eval = evaluate(&[Value::Cell(cell)]).expect("fclose cell");
        assert_eq!(eval.status(), 0.0);
        assert!(registry::info_for(fid1 as i32).is_none());
        assert!(registry::info_for(fid2 as i32).is_none());
        fs::remove_file(path1).unwrap();
        fs::remove_file(path2).unwrap();
    }

    #[test]
    fn fclose_tensor_with_non_integer_entries_errors() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let tensor = Tensor::new(vec![1.5], vec![1, 1]).expect("tensor");
        let err = evaluate(&[Value::Tensor(tensor)]).unwrap_err();
        assert_eq!(err, "fclose: file identifier must be an integer");
    }

    #[test]
    fn fclose_string_array_all_equivalent() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let strings = StringArray::new(vec!["all".to_string()], vec![1]).expect("string array");
        let eval = evaluate(&[Value::StringArray(strings)]).expect("fclose all");
        assert_eq!(eval.status(), 0.0);
        assert!(eval.message().is_empty());
    }

    #[test]
    fn fclose_accepts_empty_tensor() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let tensor = Tensor::new(Vec::new(), vec![0, 0]).expect("tensor");
        let eval = evaluate(&[Value::Tensor(tensor)]).expect("fclose");
        assert_eq!(eval.status(), 0.0);
        assert!(eval.message().is_empty());
    }

    #[test]
    fn fclose_errors_on_non_numeric_input() {
        let _guard = registry_guard();
        registry::reset_for_tests();
        let err = evaluate(&[Value::from("not-a-fid")]).unwrap_err();
        assert_eq!(err, "fclose: file identifier must be numeric or 'all'");
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let _guard = registry_guard();
        let blocks = crate::builtins::common::test_support::doc_examples(super::DOC_MD);
        assert!(!blocks.is_empty());
    }
}
