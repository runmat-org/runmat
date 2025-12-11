//! MATLAB-compatible `path` builtin for inspecting and updating the RunMat
//! search path.

use runmat_builtins::{CharArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::path_state::{
    current_path_string, set_path_string, PATH_LIST_SEPARATOR,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::gather_if_needed;

const ERROR_ARG_TYPE: &str = "path: arguments must be character vectors or string scalars";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "path",
        wasm_path = "crate::builtins::io::repl_fs::path"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "path"
category: "io/repl_fs"
keywords: ["path", "search path", "matlab path", "addpath", "rmpath"]
summary: "Query or replace the MATLAB search path used by RunMat."
references:
  - https://www.mathworks.com/help/matlab/ref/path.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs entirely on the CPU. gpuArray text inputs are gathered automatically before processing."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::repl_fs::path::tests"
  integration: "builtins::io::repl_fs::path::tests::path_updates_search_directories"
---

# What does the `path` function do in MATLAB / RunMat?
`path` reads or rewrites the MATLAB search path string that RunMat uses when resolving scripts, functions, and data files. The value mirrors MATLAB's notion of the path: a character row vector whose entries are separated by the platform-specific `pathsep` character.

## How does the `path` function behave in MATLAB / RunMat?
- `path()` (no inputs) returns the current search path as a character row vector. Each directory is separated by `pathsep` (`;` on Windows, `:` on Linux/macOS). The current working folder (`pwd`) is implicit and therefore is not included in the string.
- `old = path(newPath)` replaces the stored path with `newPath` and returns the previous value so it can be restored later. `newPath` may be a character row vector, a string scalar, or a 1-by-N double array of character codes. Whitespace at the ends of entries is preserved, matching MATLAB.
- `old = path(path1, path2)` sets the path to `path1` followed by `path2`. When both inputs are non-empty they are joined with `pathsep`; empty inputs are ignored so `path("", path2)` simply applies `path2`.
- All inputs must be character vectors or string scalars. Single-element string arrays are accepted. Multirow char arrays, multi-element string arrays, numeric arrays that cannot be interpreted as character codes, and other value types raise `path: arguments must be character vectors or string scalars`.
- Calling `path("")` clears the stored path while leaving `pwd` as the highest-priority location, just like MATLAB. The new value is stored in-process and mirrored to the `RUNMAT_PATH` environment variable, so `exist`, `which`, `dir`, and other filesystem-aware builtins observe the change immediately.

## `path` Function GPU Execution Behaviour
`path` operates entirely on host-side state. If an argument lives on the GPU, RunMat gathers it back to the CPU before validation. No acceleration provider hooks are required and no GPU kernels are launched.

## GPU residency in RunMat (Do I need `gpuArray`?)
No. The MATLAB path is a host-only configuration. RunMat automatically gathers any `gpuArray` text inputs, applies the request on the CPU, and returns the result as a character array. Explicitly creating `gpuArray` strings provides no benefit.

## Examples of using the `path` function in MATLAB / RunMat

### Display the current MATLAB search path
```matlab
p = path();
disp(p);
```
Expected output:
```matlab
/Users/alex/runmat/toolbox:/Users/alex/runmat/user   % Actual directories vary by installation
```

### Temporarily replace the MATLAB path
```matlab
old = path("C:\tools\runmat-addons");
% ... run code that relies on the temporary path ...
path(old);   % Restore the previous path
```
Expected output:
```matlab
old =
    '/Users/alex/runmat/toolbox:/Users/alex/runmat/user'
```

### Append folders to the end of the search path
```matlab
extra = "C:\projects\analysis";
old = path(path(), extra);
```
Expected output:
```matlab
path()
ans =
    'C:\tools\runmat-addons;C:\projects\analysis'
```

### Prepend folders ahead of the existing path
```matlab
extra = "/opt/runmat/toolbox";
old = path(extra, path());
```
Expected output:
```matlab
path()
ans =
    '/opt/runmat/toolbox:/Users/alex/runmat/toolbox:/Users/alex/runmat/user'
```

### Combine generated folder lists
```matlab
tooling = genpath("submodules/tooling");
old = path(tooling, path());
```
Expected output:
```matlab
% The directories returned by genpath now appear ahead of the previous path entries.
```

## FAQ
- **Does `path` include the current folder?** MATLAB automatically searches the current folder (`pwd`) before consulting the stored path. RunMat follows this rule; the character vector returned by `path` reflects the explicit path entries, while `pwd` remains an implicit priority.
- **Can I clear the path completely?** Yes. Call `path("")` to remove all explicit entries. The current folder is still searched first.
- **How do I append to the path without losing the existing value?** Use `path(path(), newEntry)` to append or `path(newEntry, path())` to prepend. Both return the previous value so you can restore it later.
- **Where is the path stored?** RunMat keeps the value in memory and updates the `RUNMAT_PATH` environment variable. External tooling that reads `RUNMAT_PATH` will therefore observe the latest configuration.
- **Do other builtins see the new path immediately?** Yes. `exist`, `which`, `run`, and other filesystem-aware builtins query the shared path state on each call.

## See Also
[addpath](https://www.mathworks.com/help/matlab/ref/addpath.html), [rmpath](https://www.mathworks.com/help/matlab/ref/rmpath.html), [genpath](https://www.mathworks.com/help/matlab/ref/genpath.html), [which](../../introspection/which), [exist](./exist)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/io/repl_fs/path.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/path.rs)
- Found an issue? [Open a GitHub ticket](https://github.com/runmat-org/runmat/issues/new/choose) with steps to reproduce.
"#;

#[runmat_macros::register_gpu_spec(wasm_path = "crate::builtins::io::repl_fs::path")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "path",
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
    notes: "Search-path management is a host-only operation; GPU inputs are gathered before processing.",
};

#[runmat_macros::register_fusion_spec(wasm_path = "crate::builtins::io::repl_fs::path")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "path",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "I/O builtins are not eligible for fusion; metadata registered for completeness.",
};

#[runtime_builtin(
    name = "path",
    category = "io/repl_fs",
    summary = "Query or replace the MATLAB search path used by RunMat.",
    keywords = "path,search path,matlab path,addpath,rmpath",
    accel = "cpu",
    wasm_path = "crate::builtins::io::repl_fs::path"
)]
fn path_builtin(args: Vec<Value>) -> Result<Value, String> {
    let gathered = gather_arguments(&args)?;
    match gathered.len() {
        0 => Ok(path_value()),
        1 => set_single_argument(&gathered[0]),
        2 => set_two_arguments(&gathered[0], &gathered[1]),
        _ => Err("path: too many input arguments".to_string()),
    }
}

fn path_value() -> Value {
    char_array_value(&current_path_string())
}

fn set_single_argument(arg: &Value) -> Result<Value, String> {
    let previous = current_path_string();
    let new_path = extract_text(arg)?;
    set_path_string(&new_path);
    Ok(char_array_value(&previous))
}

fn set_two_arguments(first: &Value, second: &Value) -> Result<Value, String> {
    let previous = current_path_string();
    let path1 = extract_text(first)?;
    let path2 = extract_text(second)?;
    let combined = combine_paths(&path1, &path2);
    set_path_string(&combined);
    Ok(char_array_value(&previous))
}

fn combine_paths(left: &str, right: &str) -> String {
    match (left.is_empty(), right.is_empty()) {
        (true, true) => String::new(),
        (false, true) => left.to_string(),
        (true, false) => right.to_string(),
        (false, false) => {
            let mut combined = String::with_capacity(left.len() + right.len() + 1);
            combined.push_str(left);
            combined.push(PATH_LIST_SEPARATOR);
            combined.push_str(right);
            combined
        }
    }
}

fn extract_text(value: &Value) -> Result<String, String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(StringArray { data, .. }) => {
            if data.len() != 1 {
                Err(ERROR_ARG_TYPE.to_string())
            } else {
                Ok(data[0].clone())
            }
        }
        Value::CharArray(chars) => {
            if chars.rows != 1 {
                return Err(ERROR_ARG_TYPE.to_string());
            }
            Ok(chars.data.iter().collect())
        }
        Value::Tensor(tensor) => tensor_to_string(tensor),
        Value::GpuTensor(_) => Err(ERROR_ARG_TYPE.to_string()),
        _ => Err(ERROR_ARG_TYPE.to_string()),
    }
}

fn tensor_to_string(tensor: &Tensor) -> Result<String, String> {
    if tensor.shape.len() > 2 {
        return Err(ERROR_ARG_TYPE.to_string());
    }

    let rows = tensor.rows();
    if rows > 1 {
        return Err(ERROR_ARG_TYPE.to_string());
    }

    let mut text = String::with_capacity(tensor.data.len());
    for &code in &tensor.data {
        if !code.is_finite() {
            return Err(ERROR_ARG_TYPE.to_string());
        }
        let rounded = code.round();
        if (code - rounded).abs() > 1e-6 {
            return Err(ERROR_ARG_TYPE.to_string());
        }
        let int_code = rounded as i64;
        if !(0..=0x10FFFF).contains(&int_code) {
            return Err(ERROR_ARG_TYPE.to_string());
        }
        let ch = char::from_u32(int_code as u32).ok_or_else(|| ERROR_ARG_TYPE.to_string())?;
        text.push(ch);
    }

    Ok(text)
}

fn gather_arguments(args: &[Value]) -> Result<Vec<Value>, String> {
    let mut out = Vec::with_capacity(args.len());
    for value in args {
        out.push(gather_if_needed(value).map_err(|err| format!("path: {err}"))?);
    }
    Ok(out)
}

fn char_array_value(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}

#[cfg(test)]
mod tests {
    use super::super::REPL_FS_TEST_LOCK;
    use super::*;
    use crate::builtins::common::path_search::search_directories;
    use crate::builtins::common::path_state::set_path_string;
    use std::convert::TryFrom;
    use tempfile::tempdir;

    struct PathGuard {
        previous: String,
    }

    impl PathGuard {
        fn new() -> Self {
            Self {
                previous: current_path_string(),
            }
        }
    }

    impl Drop for PathGuard {
        fn drop(&mut self) {
            set_path_string(&self.previous);
        }
    }

    #[test]
    fn path_returns_char_array() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let value = path_builtin(Vec::new()).expect("path");
        match value {
            Value::CharArray(CharArray { rows, .. }) => assert_eq!(rows, 1),
            other => panic!("expected CharArray, got {other:?}"),
        }
    }

    #[test]
    fn path_sets_new_value_and_returns_previous() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let guard = PathGuard::new();
        let previous = guard.previous.clone();

        let temp = tempdir().expect("tempdir");
        let dir_str = temp.path().to_string_lossy().into_owned();
        let new_value = Value::CharArray(CharArray::new_row(&dir_str));
        let returned = path_builtin(vec![new_value]).expect("path set");
        let returned_str = String::try_from(&returned).expect("convert");
        assert_eq!(returned_str, previous);

        let current =
            String::try_from(&path_builtin(Vec::new()).expect("path")).expect("convert current");
        assert_eq!(current, dir_str);
    }

    #[test]
    fn path_accepts_string_scalar() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let guard = PathGuard::new();
        let previous = guard.previous.clone();

        let new_value = Value::String("runmat/path/string".to_string());
        let returned = path_builtin(vec![new_value]).expect("path set");
        let returned_str = String::try_from(&returned).expect("convert");
        assert_eq!(returned_str, previous);

        let current =
            String::try_from(&path_builtin(Vec::new()).expect("path")).expect("convert current");
        assert_eq!(current, "runmat/path/string");
    }

    #[test]
    fn path_accepts_tensor_codes() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let guard = PathGuard::new();
        let previous = guard.previous.clone();

        let text = "tensor-path";
        let codes: Vec<f64> = text.chars().map(|ch| ch as u32 as f64).collect();
        let tensor = Tensor::new(codes, vec![1, text.len()]).expect("tensor");
        let returned = path_builtin(vec![Value::Tensor(tensor)]).expect("path set");
        let returned_str = String::try_from(&returned).expect("convert");
        assert_eq!(returned_str, previous);

        let current =
            String::try_from(&path_builtin(Vec::new()).expect("path")).expect("convert current");
        assert_eq!(current, text);
    }

    #[test]
    fn path_combines_two_arguments() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let dir1 = tempdir().expect("dir1");
        let dir2 = tempdir().expect("dir2");
        let dir1_str = dir1.path().to_string_lossy().to_string();
        let dir2_str = dir2.path().to_string_lossy().to_string();
        let path1 = Value::CharArray(CharArray::new_row(&dir1_str));
        let path2 = Value::CharArray(CharArray::new_row(&dir2_str));
        let _returned = path_builtin(vec![path1, path2]).expect("path set");

        let current =
            String::try_from(&path_builtin(Vec::new()).expect("path")).expect("convert current");
        let expected = format!(
            "{}{sep}{}",
            dir1.path().to_string_lossy(),
            dir2.path().to_string_lossy(),
            sep = PATH_LIST_SEPARATOR
        );
        assert_eq!(current, expected);
    }

    #[test]
    fn path_rejects_multi_row_char_array() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let chars = CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).expect("char array");
        let err = path_builtin(vec![Value::CharArray(chars)]).expect_err("expected error");
        assert_eq!(err, ERROR_ARG_TYPE);
    }

    #[test]
    fn path_rejects_multi_element_string_array() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let array = StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).expect("array");
        let err = path_builtin(vec![Value::StringArray(array)]).expect_err("expected error");
        assert_eq!(err, ERROR_ARG_TYPE);
    }

    #[test]
    fn path_rejects_invalid_argument_types() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let err = path_builtin(vec![Value::Num(1.0)]).expect_err("expected error");
        assert!(err.contains("path: arguments"));
    }

    #[test]
    fn path_updates_search_directories() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let temp = tempdir().expect("tempdir");
        let dir = temp.path().to_string_lossy().into_owned();
        let _ = path_builtin(vec![Value::CharArray(CharArray::new_row(&dir))]).expect("path");

        let search = search_directories("path test").expect("search directories");
        let search_strings: Vec<String> = search
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        assert!(
            search_strings.iter().any(|entry| entry == &dir),
            "search path should include newly added directory"
        );
    }

    #[test]
    fn doc_examples_present() {
        let blocks = crate::builtins::common::test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
