//! MATLAB-compatible `tempdir` builtin for RunMat.

use std::convert::TryFrom;
use std::env;
use std::path::Path;

use runmat_builtins::{CharArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

const ERR_TOO_MANY_INPUTS: &str = "tempdir: too many input arguments";
const ERR_UNABLE_TO_DETERMINE: &str =
    "tempdir: unable to determine temporary directory (OS returned empty path)";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "tempdir",
        builtin_path = "crate::builtins::io::repl_fs::tempdir"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "tempdir"
category: "io/repl_fs"
keywords: ["tempdir", "temporary folder", "temp directory", "system temp", "temporary path"]
summary: "Return the absolute path to the system temporary folder with a trailing file separator."
references:
  - https://www.mathworks.com/help/matlab/ref/tempdir.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs entirely on the host CPU. When scripts store path tokens on the GPU, RunMat gathers them automatically before interacting with the filesystem."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::repl_fs::tempdir::tests"
  integration: "builtins::io::repl_fs::tempdir::tests::tempdir_points_to_existing_directory"
---

# What does the `tempdir` function do in MATLAB / RunMat?
`tempdir` returns the absolute path of the operating system's temporary folder. The returned path always ends with the platform-specific file separator (`/` on macOS and Linux, `\` on Windows) so it is safe to concatenate file names directly, matching MATLAB behaviour.

## How does the `tempdir` function behave in MATLAB / RunMat?
- The builtin accepts no input arguments; providing any arguments raises `tempdir: too many input arguments`.
- The output is a character row vector (`1×N`) for compatibility with historical MATLAB code. Convert it with `string(tempdir)` if you prefer string scalars.
- The path reflects the process environment, honouring variables such as `TMPDIR` (macOS/Linux) or `TMP`/`TEMP` (Windows).
- When the OS reports a temporary folder that lacks a trailing separator, RunMat appends one automatically to mirror MATLAB.
- The directory is not created or cleaned automatically; RunMat relies on the operating system to manage the folder lifecycle.
- If RunMat cannot determine the temporary folder (rare), the builtin raises an error containing the operating system message.

## `tempdir` Function GPU Execution Behaviour
`tempdir` performs no GPU work. It queries the host environment and surfaces the result as a character array. The builtin registers a CPU-only GPU spec so the fusion planner treats it as a host-side operation. Scripts that accidentally store paths on the GPU have nothing to worry about—`tempdir` has no inputs to gather.

## GPU residency in RunMat (Do I need `gpuArray`?)
No. `tempdir` is always a host operation. There is nothing to gain by moving data to the GPU, and no acceleration provider needs to implement hooks for this builtin.

## Examples of using the `tempdir` function in MATLAB / RunMat

### Get The System Temporary Folder Path
```matlab
folder = tempdir();
disp(folder);
```
Expected output (macOS/Linux):
```matlab
/var/folders/xy/abcd/T/
```
Expected output (Windows):
```matlab
C:\Users\alex\AppData\Local\Temp\
```

### Combine `tempdir` With `tempname` To Build Unique Paths
```matlab
uniqueFile = fullfile(tempdir(), "runmat-session.log");
fid = fopen(uniqueFile, "w");
fprintf(fid, "Temporary log created at %s\n", datetime);
fclose(fid);
```
Expected output:
```matlab
% Creates runmat-session.log inside the system temporary folder
```

### Create A Temporary Working Subfolder
```matlab
workDir = fullfile(tempdir(), "mytask");
if ~exist(workDir, "dir")
    mkdir(workDir);
end
disp(workDir);
```
Expected output:
```matlab
% Prints the path to tempdir/mytask and creates the folder if needed
```

### Verify That `tempdir` Appends A File Separator
```matlab
folder = tempdir();
endsWithSeparator = folder(end) == filesep;
disp(endsWithSeparator);
```
Expected output:
```matlab
     1
```

### Use `tempdir` When Creating Temporary Zip Archives
```matlab
archive = fullfile(tempdir(), "results.zip");
zip(archive, "results");
disp(archive);
```
Expected output:
```matlab
% Displays the full path to results.zip inside the system temp folder
```

### Log Temporary Folder Usage For Debugging
```matlab
fprintf("RunMat temp folder: %s\n", tempdir());
```
Expected output:
```matlab
% Prints something like:
%   RunMat temp folder: /tmp/
```

## FAQ

### Why does `tempdir` return a character vector instead of a string scalar?
MATLAB has historically returned character vectors. RunMat mirrors that behaviour for compatibility; wrap the result in `string(...)` when you need a string scalar.

### Does the path include a trailing separator?
Yes. RunMat appends the platform-specific file separator when necessary so you can safely concatenate file names.

### What happens if the temporary folder does not exist?
RunMat reports the folder reported by the operating system. Most platforms guarantee the folder exists; if not, subsequent calls such as `mkdir` can create it.

### Can I change the temporary folder location?
Yes. Set `TMPDIR` (macOS/Linux) or `TEMP`/`TMP` (Windows) before starting RunMat. `tempdir` will honour those environment variables, just like MATLAB.

### Will `tempdir` ever create or clean files automatically?
No. The builtin only reports the path. Cleanup policies remain the responsibility of the operating system or your code.

### Is `tempdir` thread-safe?
Yes. Querying the temporary folder is read-only and does not modify global state.

### Does `tempdir` work inside deployed or sandboxed environments?
As long as the process has permission to query the environment, it returns the best-effort location supplied by the OS.

### Can I call `tempdir` on the GPU?
No. The function is host-only, but it also never transfers data to or from the GPU.

### Does `tempdir` normalize path separators?
RunMat preserves the operating system’s separators (backslash on Windows, slash elsewhere) to maintain MATLAB compatibility.

### What if I pass arguments to `tempdir` by mistake?
RunMat raises `tempdir: too many input arguments`, matching MATLAB’s diagnostic.

## See Also
[mkdir](../mkdir), [rmdir](../rmdir), [delete](../delete), [dir](../dir), [pwd](../pwd)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/io/repl_fs/tempdir.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/tempdir.rs)
- Issues: [Open a GitHub ticket](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::tempdir")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "tempdir",
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
    notes: "Host-only operation that queries the environment for the temporary folder. No provider hooks are required.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::tempdir")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "tempdir",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "I/O builtin that always executes on the host; fusion metadata is present for introspection completeness.",
};

#[runtime_builtin(
    name = "tempdir",
    category = "io/repl_fs",
    summary = "Return the absolute path to the system temporary folder.",
    keywords = "tempdir,temporary folder,temp directory,system temp",
    accel = "cpu",
    builtin_path = "crate::builtins::io::repl_fs::tempdir"
)]
fn tempdir_builtin(args: Vec<Value>) -> Result<Value, String> {
    if !args.is_empty() {
        return Err(ERR_TOO_MANY_INPUTS.to_string());
    }
    let path = env::temp_dir();
    if path.as_os_str().is_empty() {
        return Err(ERR_UNABLE_TO_DETERMINE.to_string());
    }
    let value = path_to_char_array(&path);
    if let Ok(text) = String::try_from(&value) {
        if text.is_empty() {
            return Err(ERR_UNABLE_TO_DETERMINE.to_string());
        }
    }
    Ok(value)
}

fn path_to_char_array(path: &Path) -> Value {
    let mut text = path.to_string_lossy().into_owned();
    if !text.is_empty() && !ends_with_separator(&text) {
        text.push(std::path::MAIN_SEPARATOR);
    }
    Value::CharArray(CharArray::new_row(&text))
}

fn ends_with_separator(text: &str) -> bool {
    let sep = std::path::MAIN_SEPARATOR;
    text.chars()
        .next_back()
        .is_some_and(|ch| ch == sep || (cfg!(windows) && (ch == '/' || ch == '\\')))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::convert::TryFrom;
    use std::path::{Path, PathBuf};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempdir_points_to_existing_directory() {
        let value = tempdir_builtin(Vec::new()).expect("tempdir");
        let path_string = String::try_from(&value).expect("convert to string");
        let path = PathBuf::from(&path_string);
        assert!(path.exists(), "tempdir path should exist: {}", path_string);
        assert!(
            path.is_dir(),
            "tempdir path should reference a directory: {}",
            path_string
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempdir_returns_char_array_row_vector() {
        let value = tempdir_builtin(Vec::new()).expect("tempdir");
        match value {
            Value::CharArray(CharArray { rows, cols, .. }) => {
                assert_eq!(rows, 1);
                assert!(
                    cols >= 1,
                    "expected tempdir to return at least one character"
                );
            }
            other => panic!("expected CharArray result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempdir_appends_trailing_separator() {
        let value = tempdir_builtin(Vec::new()).expect("tempdir");
        let path_string = String::try_from(&value).expect("convert to string");
        let expected_sep = std::path::MAIN_SEPARATOR;
        let last = path_string
            .chars()
            .last()
            .expect("tempdir should return non-empty path");
        if cfg!(windows) {
            assert!(
                last == '/' || last == '\\',
                "expected trailing separator, got {:?}",
                path_string
            );
        } else {
            assert_eq!(
                last, expected_sep,
                "expected trailing separator {}, got {}",
                expected_sep, path_string
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempdir_returns_consistent_result() {
        let first = tempdir_builtin(Vec::new()).expect("tempdir");
        let second = tempdir_builtin(Vec::new()).expect("tempdir");
        let first_str = String::try_from(&first).expect("first string");
        let second_str = String::try_from(&second).expect("second string");
        assert_eq!(
            first_str, second_str,
            "tempdir should be stable within a process"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempdir_errors_when_arguments_provided() {
        let err = tempdir_builtin(vec![Value::Num(1.0)]).expect_err("expected error");
        assert_eq!(err, ERR_TOO_MANY_INPUTS);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn path_to_char_array_appends_separator_when_missing() {
        let path = Path::new("runmat_tempdir_unit_path");
        let value = path_to_char_array(path);
        let text = String::try_from(&value).expect("string conversion");
        assert!(
            text.ends_with(std::path::MAIN_SEPARATOR),
            "expected trailing separator in {text:?}"
        );
        let trimmed = text.trim_end_matches(std::path::MAIN_SEPARATOR);
        assert_eq!(trimmed, path.to_string_lossy());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn path_to_char_array_preserves_existing_separator() {
        let sep = std::path::MAIN_SEPARATOR;
        let input = format!("runmat_tempdir_existing{sep}");
        let path = Path::new(&input);
        let value = path_to_char_array(path);
        let text = String::try_from(&value).expect("string conversion");
        assert_eq!(text, input);
    }

    #[cfg(windows)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ends_with_separator_accepts_forward_slash() {
        assert!(ends_with_separator("C:/Temp/"));
        assert!(ends_with_separator("temp/"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = crate::builtins::common::test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
