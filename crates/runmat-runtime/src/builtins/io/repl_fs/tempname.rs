//! MATLAB-compatible `tempname` builtin for RunMat.

use runmat_time::system_time_now;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::UNIX_EPOCH;

use runmat_builtins::{CharArray, Value};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::{expand_user_path, path_to_string};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const ERR_TOO_MANY_INPUTS: &str = "tempname: too many input arguments";
const ERR_FOLDER_TYPE: &str = "tempname: folder name must be a character vector or string scalar";
const ERR_FOLDER_EMPTY: &str = "tempname: folder name must not be empty";
const ERR_TEMP_DIR_UNAVAILABLE: &str = "tempname: unable to determine temporary directory";
const ERR_UNABLE_TO_GENERATE: &str = "tempname: unable to generate a unique name";

const MAX_ATTEMPTS: usize = 64;
static UNIQUE_COUNTER: AtomicU64 = AtomicU64::new(0);

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "tempname",
        builtin_path = "crate::builtins::io::repl_fs::tempname"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "tempname"
category: "io/repl_fs"
keywords: ["tempname", "temporary file", "unique name", "filesystem", "temp directory"]
summary: "Return a unique temporary file path in the system temp folder or a user-specified directory."
references:
  - https://www.mathworks.com/help/matlab/ref/tempname.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs entirely on the host. When callers provide GPU-resident strings, RunMat gathers them before computing the file name."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::repl_fs::tempname::tests"
  integration: "builtins::io::repl_fs::tempname::tests::tempname_generates_unique_names"
---

# What does the `tempname` function do in MATLAB / RunMat?
`tempname` returns a unique path that you can reserve for a temporary file or folder. By default it chooses the system temporary directory. When you provide a folder argument, the path is generated inside that folder instead.

## How does the `tempname` function behave in MATLAB / RunMat?
- `tempname()` returns a character row vector (`1×N`) containing an absolute path inside the system temporary directory.
- `tempname(folder)` returns a path inside `folder`, which can be absolute or relative. RunMat expands leading `~` to the home directory for convenience.
- The returned path never corresponds to an existing file or directory at the time of the call.
- `tempname` does not create files or directories. Use the returned value with builtins such as `fopen`, `mkdir`, or `movefile`.
- The generated token begins with the MATLAB-compatible `tp` prefix followed by hexadecimal entropy, making it human-recognisable while avoiding collisions.
- Providing more than one input argument raises `tempname: too many input arguments`. Non-text inputs raise a descriptive type error.

## `tempname` Function GPU Execution Behaviour
`tempname` performs all computation on the CPU. When scripts pass GPU-resident strings (for example, `gpuArray("scratch")`), RunMat automatically gathers those scalars to host memory before determining the result. Acceleration providers do not implement hooks for this builtin, and there is no GPU kernel to warm up.

## GPU residency in RunMat (Do I need `gpuArray`?)
No. `tempname` only manipulates paths and never benefits from GPU execution. It always returns a host-resident character array. If you accidentally store a folder name on the GPU, RunMat gathers it transparently.

## Examples of using the `tempname` function in MATLAB / RunMat

### Generate A Unique Temporary File Name
```matlab
fname = tempname();
fprintf("Saving intermediate results to %s\n", fname);
```
Expected output:
```matlab
% Prints the unique file name inside the system temporary folder.
```

### Create A Temporary File Name In A Custom Folder
```matlab
logDir = fullfile(pwd(), "logs");
fname = tempname(logDir);
```
Expected output:
```matlab
% fname starts with the logs folder and ends with a tp******** token.
```

### Append A File Extension To Tempname Results
```matlab
csvPath = [tempname(), ".csv"];
```
Expected output:
```matlab
% csvPath is a unique .csv file path you can pass to writematrix or fprintf.
```

### Reserve A Temporary Folder Path
```matlab
scratch = tempname();
mkdir(scratch);
cleanupObj = onCleanup(@() rmdir(scratch, "s"));
```
Expected output:
```matlab
% scratch now exists on disk and is removed automatically via onCleanup.
```

### Use Tempname With fopen To Write Temporary Data
```matlab
tmpFile = tempname();
[fid, message] = fopen(tmpFile, "w");
if fid == -1
    error("Failed to open temp file: %s", message);
end
fprintf(fid, "Temporary output\n");
fclose(fid);
```
Expected output:
```matlab
% Creates a file, writes text, and leaves it for later processing.
```

### Combine Tempname With gpuArray Inputs
```matlab
folder = gpuArray("scratch");
fname = tempname(folder);
```
Expected output:
```matlab
% RunMat gathers the string and returns a host-side character vector.
```

## FAQ

### Does `tempname` create the file or folder for me?
No. It only reserves a unique path. Call `fopen`, `mkdir`, or other functions to create the resource.

### Can I call `tempname` with relative folders?
Yes. RunMat honours relative paths and joins the generated token using the platform’s path separator.

### What happens if the target folder does not exist yet?
`tempname` still returns a path under that folder. It is up to your code to create intermediate directories if needed.

### Why does the result start with `tp`?
MATLAB prefixes the token with `tp` for temporary paths. RunMat follows the same convention for familiarity.

### Is the result guaranteed to be unique?
The builtin combines monotonic process-wide counters with high-resolution timestamps and the process ID. The result does not exist at the moment of generation; collisions are exceedingly unlikely.

### Can I request multiple names at once?
Call `tempname` repeatedly. Each invocation returns a fresh token.

### Does `tempname` support Unicode folder names?
Yes. Paths are stored as UTF-16 internally on Windows and UTF-8 on Unix-like systems. RunMat converts between encodings automatically.

### How do I convert the result to a string scalar?
Wrap the output in `string(tempname())` or `string(tempname(folder))`.

### Will GPU acceleration change the output?
No. The builtin is host-only and ignores GPU providers entirely.

## See Also
[tempdir](./tempdir), [mkdir](./mkdir), [fopen](./fopen), [delete](./delete), [movefile](./movefile)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/io/repl_fs/tempname.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/tempname.rs)
- Issues: [Open a GitHub ticket](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::tempname")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "tempname",
    op_kind: GpuOpKind::Custom("io"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host-only path generation. Providers are not expected to supply kernels for temporary name creation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::tempname")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "tempname",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "I/O builtins are not eligible for fusion; metadata is registered for completeness.",
};

const BUILTIN_NAME: &str = "tempname";

fn tempname_error(message: impl Into<String>) -> RuntimeError {
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

#[runtime_builtin(
    name = "tempname",
    category = "io/repl_fs",
    summary = "Return a unique temporary file path.",
    keywords = "tempname,temporary file,unique name,temp directory",
    accel = "cpu",
    builtin_path = "crate::builtins::io::repl_fs::tempname"
)]
async fn tempname_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    match args.len() {
        0 => {
            let base = default_temp_directory()?;
            Ok(path_to_value(&generate_unique_path(&base)?))
        }
        1 => {
            let gathered = gather_argument(&args[0]).await?;
            let folder = parse_folder_argument(&gathered)?;
            Ok(path_to_value(&generate_unique_path(&folder)?))
        }
        _ => Err(tempname_error(ERR_TOO_MANY_INPUTS)),
    }
}

fn default_temp_directory() -> BuiltinResult<PathBuf> {
    let path = std::env::temp_dir();
    if path.as_os_str().is_empty() {
        Err(tempname_error(ERR_TEMP_DIR_UNAVAILABLE))
    } else {
        Ok(path)
    }
}

async fn gather_argument(value: &Value) -> BuiltinResult<Value> {
    gather_if_needed_async(value).await.map_err(map_control_flow)
}

fn parse_folder_argument(value: &Value) -> BuiltinResult<PathBuf> {
    let text = match value {
        Value::String(s) => {
            if s.is_empty() {
                return Err(tempname_error(ERR_FOLDER_EMPTY));
            }
            s.clone()
        }
        Value::CharArray(array) => {
            if array.rows != 1 {
                return Err(tempname_error(ERR_FOLDER_TYPE));
            }
            let collected: String = array.data.iter().collect();
            if collected.is_empty() {
                return Err(tempname_error(ERR_FOLDER_EMPTY));
            }
            collected
        }
        Value::StringArray(array) => {
            if array.data.len() != 1 {
                return Err(tempname_error(ERR_FOLDER_TYPE));
            }
            let collected = array.data[0].clone();
            if collected.is_empty() {
                return Err(tempname_error(ERR_FOLDER_EMPTY));
            }
            collected
        }
        _ => return Err(tempname_error(ERR_FOLDER_TYPE)),
    };

    let expanded = expand_user_path(&text, "tempname").map_err(|err| tempname_error(err))?;
    if expanded.is_empty() {
        Err(tempname_error(ERR_FOLDER_EMPTY))
    } else {
        Ok(PathBuf::from(expanded))
    }
}

fn generate_unique_path(base: &Path) -> BuiltinResult<PathBuf> {
    for _ in 0..MAX_ATTEMPTS {
        let token = unique_token();
        let candidate = if base.as_os_str().is_empty() {
            PathBuf::from(&token)
        } else {
            base.join(&token)
        };
        if !path_exists(&candidate) {
            return Ok(candidate);
        }
    }
    Err(tempname_error(ERR_UNABLE_TO_GENERATE))
}

fn unique_token() -> String {
    let now = system_time_now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let nanos = now.subsec_nanos();
    let pid = std::process::id() as u64;
    let counter = UNIQUE_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("tp{:016x}{:08x}{:08x}{:016x}", secs, nanos, pid, counter)
}

fn path_to_value(path: &Path) -> Value {
    let text = path_to_string(path);
    Value::CharArray(CharArray::new_row(&text))
}

fn path_exists(path: &Path) -> bool {
    vfs::metadata(path).is_ok()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::REPL_FS_TEST_LOCK;
    use super::*;
    use crate::builtins::common::fs::home_directory;
    use runmat_builtins::StringArray;
    use std::convert::TryFrom;
    use std::path::{Path, PathBuf};
    use tempfile::tempdir;

    fn tempname_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::tempname_builtin(args))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempname_generates_unique_names() {
        let _lock = REPL_FS_TEST_LOCK.lock().unwrap();
        let first = tempname_builtin(Vec::new()).expect("tempname");
        let second = tempname_builtin(Vec::new()).expect("tempname");
        let first_str = String::try_from(&first).expect("first string");
        let second_str = String::try_from(&second).expect("second string");
        assert_ne!(
            first_str, second_str,
            "tempname should return unique values"
        );
        assert!(!path_exists(Path::new(&first_str)), "first path exists");
        assert!(!path_exists(Path::new(&second_str)), "second path exists");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempname_returns_char_row_vector() {
        let value = tempname_builtin(Vec::new()).expect("tempname");
        match value {
            Value::CharArray(CharArray { rows, cols, .. }) => {
                assert_eq!(rows, 1);
                assert!(cols >= 1, "expected non-empty character row vector");
            }
            other => panic!("expected CharArray, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempname_uses_system_temp_directory_by_default() {
        let _lock = REPL_FS_TEST_LOCK.lock().unwrap();
        let value = tempname_builtin(Vec::new()).expect("tempname");
        let text = String::try_from(&value).expect("string conversion");
        let path = PathBuf::from(&text);
        let parent = path
            .parent()
            .expect("tempname should include a parent folder");
        assert_eq!(parent, std::env::temp_dir().as_path());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempname_accepts_custom_folder_char_array() {
        let _lock = REPL_FS_TEST_LOCK.lock().unwrap();
        let dir = tempdir().expect("tempdir");
        let base = dir.path();
        let base_text = base.to_string_lossy().into_owned();
        let arg = Value::CharArray(CharArray::new_row(&base_text));
        let value = tempname_builtin(vec![arg]).expect("tempname");
        let path_string = String::try_from(&value).expect("string");
        let path = PathBuf::from(&path_string);
        assert_eq!(
            path.parent(),
            Some(base),
            "expected tempname to honour the provided folder"
        );
        assert!(!path_exists(&path), "generated path should not exist yet");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempname_accepts_custom_folder_string_scalar() {
        let dir = tempdir().expect("tempdir");
        let base = dir.path().to_string_lossy().into_owned();
        let value = tempname_builtin(vec![Value::String(base.clone())]).expect("tempname");
        let path_string = String::try_from(&value).expect("string");
        assert!(
            path_string.starts_with(&base),
            "custom folder should prefix the result"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempname_allows_nonexistent_targets() {
        let dir = tempdir().expect("tempdir");
        let base = dir.path().join("nested").to_string_lossy().into_owned();
        let value = tempname_builtin(vec![Value::String(base.clone())]).expect("tempname");
        let path_string = String::try_from(&value).expect("string");
        assert!(
            path_string.starts_with(&base),
            "result should still be under the requested folder"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempname_accepts_string_array_scalar() {
        let _lock = REPL_FS_TEST_LOCK.lock().unwrap();
        let dir = tempdir().expect("tempdir");
        let base = dir.path().to_string_lossy().into_owned();
        let array = StringArray::new(vec![base.clone()], vec![1, 1]).expect("string array");
        let value = tempname_builtin(vec![Value::StringArray(array)]).expect("tempname");
        let path_string = String::try_from(&value).expect("string");
        assert!(
            path_string.starts_with(&base),
            "tempname should accept string scalar arrays"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempname_expands_tilde_prefix() {
        let home = home_directory().expect("home directory");
        let value = tempname_builtin(vec![Value::String("~".to_string())]).expect("tempname");
        let path_string = String::try_from(&value).expect("string");
        let path = PathBuf::from(&path_string);
        assert!(
            path.starts_with(&home),
            "tilde should expand to the home directory"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempname_preserves_relative_folder() {
        let base = "relative_temp";
        let value = tempname_builtin(vec![Value::String(base.to_string())]).expect("tempname");
        let path_string = String::try_from(&value).expect("string");
        let path = PathBuf::from(&path_string);
        assert!(
            path.is_relative(),
            "relative folder inputs should produce relative outputs"
        );
        assert!(
            path.starts_with(base),
            "result should start with the provided relative folder"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempname_rejects_too_many_arguments() {
        let err =
            tempname_builtin(vec![Value::Num(1.0), Value::Num(2.0)]).expect_err("expected error");
        assert_eq!(err.message(), ERR_TOO_MANY_INPUTS);

        let err = tempname_builtin(vec![Value::Num(1.0)]).expect_err("error");
        assert_eq!(err.message(), ERR_FOLDER_TYPE);

        let empty = Value::CharArray(CharArray::new_row(""));
        let err = tempname_builtin(vec![empty]).expect_err("error");
        assert_eq!(err.message(), ERR_FOLDER_EMPTY);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = crate::builtins::common::test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
