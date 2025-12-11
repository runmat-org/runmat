//! MATLAB-compatible `feof` builtin for RunMat.
//!
//! Provides end-of-file detection for file identifiers opened via `fopen`.
//! The implementation mirrors MATLAB semantics for host-side files and
//! integrates with the shared registry used by the other text I/O builtins.

use std::io::{ErrorKind, Seek, SeekFrom};

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::io::filetext::registry;
use crate::gather_if_needed;

const INVALID_IDENTIFIER_MESSAGE: &str =
    "Invalid file identifier. Use fopen to generate a valid file ID.";
const IDENTIFIER_TYPE_ERROR: &str = "feof: file identifier must be a numeric scalar";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(name = "feof")
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "feof"
category: "io/filetext"
keywords: ["feof", "end of file", "io", "file identifier"]
summary: "Query whether the file position indicator of a file identifier is at end-of-file."
references:
  - https://www.mathworks.com/help/matlab/ref/feof.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs entirely on the CPU. Arguments that live on the GPU are gathered automatically, and file handles remain host-resident."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::filetext::feof::tests"
  integration: "builtins::io::filetext::feof::tests::feof_returns_true_after_reading_to_end"
---

# What does the `feof` function do in MATLAB / RunMat?
`feof(fid)` reports whether the file position indicator associated with `fid`
has reached the end of the file. It is most commonly used to terminate loops
that repeatedly call `fread`, `fgets`, or `fscanf`.

## How does the `feof` function behave in MATLAB / RunMat?
- `tf = feof(fid)` returns logical `1` when the file position indicator is at
  end-of-file and logical `0` otherwise.
- `fid` must be a non-negative integer returned by `fopen`. Passing an invalid
  identifier raises the same error message as other file I/O builtins.
- Standard input/output identifiers (`0`, `1`, `2`) always report `false` because
  RunMat does not yet expose end-of-file state for those streams.
- `feof` preserves the file position indicator; checking the flag does not
  modify where subsequent reads continue.
- When a file is empty, `feof` immediately returns `true` because the initial
  file position coincides with the end of the file.

## `feof` Function GPU Execution Behaviour
`feof` is a host-only query. If `fid` resides on the GPU (for example inside a
`gpuArray`), RunMat gathers the scalar to host memory before consulting the file
registry. File handles always remain on the CPU, so there are no provider hooks
or fusion paths for this builtin. The result is produced on the host as a
logical scalar that mirrors MATLAB behaviour.

## GPU residency in RunMat (Do I need `gpuArray`?)
No. File identifiers are always owned by the host-side registry maintained by
`fopen`. Wrapping a scalar identifier in `gpuArray` does not keep the file handle
on the GPU—the runtime gathers the scalar before performing the lookup. This
means you can freely mix host and GPU code when driving file I/O loops, and
`feof` remains inexpensive regardless of execution tier.

## Examples of using the `feof` function in MATLAB / RunMat

### Check for end-of-file while reading line by line
```matlab
[fid, msg] = fopen('log.txt', 'r');
assert(fid ~= -1, msg);
while ~feof(fid)
    line = fgets(fid);
    disp(line);
end
fclose(fid);
```

### Detect the end of a binary stream read with fread
```matlab
[fid, msg] = fopen('payload.bin', 'rb');
assert(fid ~= -1, msg);
while ~feof(fid)
    chunk = fread(fid, 1024, 'uint8');
    process(chunk);
end
fclose(fid);
```

### Handle empty files gracefully
```matlab
[fid, msg] = fopen('empty.txt', 'r');
assert(fid ~= -1, msg);
isAtEnd = feof(fid);   % returns true immediately because the file is empty
fclose(fid);
```

### Protect against invalid identifiers
```matlab
try
    feof(9999);
catch err
    disp(err.message);
end
```

### Guard reads that rely on gpuArray identifiers
```matlab
fid = gpuArray(int32(fopen('samples.dat', 'rb')));
tf = feof(fid);        % argument is gathered automatically before the check
```

## FAQ

### What kind of input does `feof` accept?
Pass a scalar numeric file identifier returned by `fopen`. The identifier must
be non-negative and finite. Character vectors or strings are not accepted.

### Does `feof` move the file position indicator?
No. The builtin queries the current position and restores it before returning,
so subsequent reads resume where they left off.

### How does `feof` behave for standard input/output?
RunMat currently reports `false` for identifiers `0`, `1`, and `2` because the
standard streams are not integrated with the file registry. Use other platform
APIs to detect end-of-file on those streams if necessary.

### Does `feof` require a previous read attempt to return true?
No. RunMat compares the current position against the end of the file. As soon
as the indicator reaches the end (for example after reading all bytes or when
the file is empty), `feof` returns `true`.

### Can I call `feof` on closed files?
No. Passing a closed or never-opened identifier raises
`"Invalid file identifier. Use fopen to generate a valid file ID."`

### How does `feof` interact with GPU execution?
`feof` itself never runs on the GPU. When the identifier is stored in a GPU
array, the runtime gathers it before performing the host-only check.

## See Also
[fopen](./fopen), [fclose](./fclose), [fread](./fread), [fwrite](./fwrite), [fileread](./fileread)

## Source & Feedback
- Source: `crates/runmat-runtime/src/builtins/io/filetext/feof.rs`
- Found a behavioural difference? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "feof",
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
    notes: "Host-only file I/O query; providers are not involved.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "feof",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "File I/O queries are not eligible for fusion; metadata registered for completeness.",
};

#[runtime_builtin(
    name = "feof",
    category = "io/filetext",
    summary = "Query whether a file identifier is positioned at end-of-file.",
    keywords = "feof,end of file,io,file identifier",
    accel = "cpu"
)]
fn feof_builtin(fid: Value) -> Result<Value, String> {
    let at_end = evaluate(&fid)?;
    Ok(Value::Bool(at_end))
}

/// Evaluate the `feof` builtin without invoking the runtime dispatcher.
pub fn evaluate(fid_value: &Value) -> Result<bool, String> {
    let fid_host = gather_if_needed(fid_value).map_err(|e| format!("feof: {e}"))?;
    let fid = parse_fid(&fid_host)?;
    if fid < 0 {
        return Err("feof: file identifier must be non-negative".to_string());
    }
    if fid < 3 {
        return Ok(false);
    }

    let handle =
        registry::take_handle(fid).ok_or_else(|| format!("feof: {INVALID_IDENTIFIER_MESSAGE}"))?;
    let mut file = handle
        .lock()
        .map_err(|_| "feof: failed to lock file handle (poisoned mutex)".to_string())?;

    let position = file
        .seek(SeekFrom::Current(0))
        .map_err(|err| format!("feof: failed to query file position: {err}"))?;

    let end_position = match file.seek(SeekFrom::End(0)) {
        Ok(pos) => pos,
        Err(err) => {
            if err.kind() == ErrorKind::Unsupported {
                let _ = file.seek(SeekFrom::Start(position));
                return Ok(false);
            }
            return Err(format!("feof: failed to query file length: {err}"));
        }
    };

    if let Err(err) = file.seek(SeekFrom::Start(position)) {
        return Err(format!("feof: failed to restore file position: {err}"));
    }

    Ok(position >= end_position)
}

fn parse_fid(value: &Value) -> Result<i32, String> {
    match value {
        Value::Num(n) => parse_scalar_fid(*n),
        Value::Int(int) => {
            let v = int.to_f64();
            parse_scalar_fid(v)
        }
        Value::Tensor(t) => {
            if t.data.len() == 1 {
                parse_scalar_fid(t.data[0])
            } else {
                Err(IDENTIFIER_TYPE_ERROR.to_string())
            }
        }
        Value::LogicalArray(la) if la.data.len() == 1 => {
            let v = if la.data[0] != 0 { 1.0 } else { 0.0 };
            parse_scalar_fid(v)
        }
        Value::LogicalArray(_) => Err(IDENTIFIER_TYPE_ERROR.to_string()),
        Value::Bool(b) => parse_scalar_fid(if *b { 1.0 } else { 0.0 }),
        _ => Err(IDENTIFIER_TYPE_ERROR.to_string()),
    }
}

fn parse_scalar_fid(value: f64) -> Result<i32, String> {
    if !value.is_finite() {
        return Err("feof: file identifier must be finite".to_string());
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err("feof: file identifier must be an integer".to_string());
    }
    if rounded < i32::MIN as f64 || rounded > i32::MAX as f64 {
        return Err("feof: file identifier is out of range".to_string());
    }
    Ok(rounded as i32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::io::filetext::{fclose, fopen, fread, registry};
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{Tensor, Value};
    use runmat_filesystem::{self as fs, File};
    use runmat_time::system_time_now;
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::UNIX_EPOCH;

    #[test]
    fn feof_returns_false_before_reading() {
        registry::reset_for_tests();
        let path = unique_path("feof_false_before_read");
        {
            let mut file = File::create(&path).expect("create");
            file.write_all(b"abc").expect("write");
        }

        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let at_end = evaluate(&Value::Num(fid as f64)).expect("feof");
        assert!(!at_end);

        fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn feof_returns_true_after_reading_to_end() {
        registry::reset_for_tests();
        let path = unique_path("feof_true_after_read");
        {
            let mut file = File::create(&path).expect("create");
            file.write_all(&[1u8, 2, 3]).expect("write");
        }

        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        // Read the entire file to advance the file position to EOF.
        let _ = fread::evaluate(&Value::Num(fid as f64), &Vec::new()).expect("fread");

        let at_end = evaluate(&Value::Num(fid as f64)).expect("feof");
        assert!(at_end);

        fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn feof_empty_file_is_true() {
        registry::reset_for_tests();
        let path = unique_path("feof_empty_file");
        File::create(&path).expect("create empty");

        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let at_end = evaluate(&Value::Num(fid as f64)).expect("feof");
        assert!(at_end);

        fclose::evaluate(&[Value::Num(fid as f64)]).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn feof_invalid_identifier_errors() {
        registry::reset_for_tests();
        let err = evaluate(&Value::Num(42.0)).unwrap_err();
        assert!(err.contains("Invalid file identifier"));
    }

    #[test]
    fn feof_rejects_non_integer_identifier() {
        registry::reset_for_tests();
        let err = evaluate(&Value::Num(1.5)).unwrap_err();
        assert_eq!(err, "feof: file identifier must be an integer");
    }

    #[test]
    fn feof_rejects_nan_identifier() {
        registry::reset_for_tests();
        let err = evaluate(&Value::Num(f64::NAN)).unwrap_err();
        assert_eq!(err, "feof: file identifier must be finite");
    }

    #[test]
    fn feof_rejects_negative_identifier() {
        registry::reset_for_tests();
        let err = evaluate(&Value::Num(-1.0)).unwrap_err();
        assert_eq!(err, "feof: file identifier must be non-negative");
    }

    #[test]
    fn feof_rejects_non_numeric_inputs() {
        registry::reset_for_tests();
        let err = evaluate(&Value::from("abc")).unwrap_err();
        assert_eq!(err, IDENTIFIER_TYPE_ERROR);
    }

    #[test]
    fn feof_accepts_scalar_tensor_identifier() {
        registry::reset_for_tests();
        let path = unique_path("feof_tensor_identifier");
        {
            let mut file = File::create(&path).expect("create");
            file.write_all(b"data").expect("write");
        }

        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as f64;

        let tensor = Tensor::new(vec![fid], vec![1]).unwrap();
        let at_end = evaluate(&Value::Tensor(tensor)).expect("feof");
        assert!(!at_end);

        fclose::evaluate(&[Value::Num(fid)]).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn feof_errors_on_closed_identifier() {
        registry::reset_for_tests();
        let path = unique_path("feof_closed_identifier");
        {
            let mut file = File::create(&path).expect("create");
            file.write_all(b"x").expect("write");
        }

        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as f64;

        fclose::evaluate(&[Value::Num(fid)]).unwrap();

        let err = evaluate(&Value::Num(fid)).unwrap_err();
        assert!(err.contains("Invalid file identifier"));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn feof_accepts_gpu_identifier_via_gather() {
        registry::reset_for_tests();
        let path = unique_path("feof_gpu_identifier");
        {
            let mut file = File::create(&path).expect("create");
            file.write_all(b"xyz").expect("write");
        }

        let open = fopen::evaluate(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as f64;

        crate::builtins::common::test_support::with_test_provider(|provider| {
            let data = [fid];
            let shape = [1usize];
            let view = HostTensorView {
                data: &data,
                shape: &shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let value = Value::GpuTensor(handle.clone());

            let at_end = evaluate(&value).expect("feof");
            assert!(!at_end);

            provider.free(&handle).expect("free");
        });

        fclose::evaluate(&[Value::Num(fid)]).unwrap();
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn feof_standard_identifier_returns_false() {
        registry::reset_for_tests();
        let result = evaluate(&Value::Num(0.0)).expect("feof");
        assert!(!result);
    }

    #[test]
    fn doc_examples_present() {
        let blocks = crate::builtins::common::test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    fn unique_path(prefix: &str) -> PathBuf {
        let now = system_time_now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let filename = format!("{prefix}_{now}.tmp");
        std::env::temp_dir().join(filename)
    }
}
