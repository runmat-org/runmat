//! MATLAB-compatible `delete` builtin for RunMat.

use runmat_filesystem as vfs;
use std::io;
use std::path::{Path, PathBuf};

use glob::{Pattern, PatternError};
use runmat_builtins::{CellArray, CharArray, StringArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::{contains_wildcards, expand_user_path, path_to_string};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::gather_if_needed;

const MESSAGE_ID_FILE_NOT_FOUND: &str = "MATLAB:DELETE:FileNotFound";
const MESSAGE_ID_IS_DIRECTORY: &str = "MATLAB:DELETE:Directories";
const MESSAGE_ID_OS_ERROR: &str = "MATLAB:DELETE:PermissionDenied";
const MESSAGE_ID_INVALID_PATTERN: &str = "MATLAB:DELETE:InvalidPattern";
const MESSAGE_ID_INVALID_INPUT: &str = "MATLAB:DELETE:InvalidInput";
const MESSAGE_ID_EMPTY_FILENAME: &str = "MATLAB:DELETE:EmptyFilename";
const MESSAGE_ID_INVALID_HANDLE: &str = "MATLAB:DELETE:InvalidHandle";

const ERR_FILENAME_ARG: &str =
    "delete: filename must be a character vector, string scalar, string array, or cell array of character vectors";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "delete",
        builtin_path = "crate::builtins::io::repl_fs::delete"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "delete"
category: "io/repl_fs"
keywords: ["delete", "remove file", "wildcard delete", "cleanup", "temporary files", "MATLAB delete"]
summary: "Remove files using MATLAB-compatible wildcard expansion, array inputs, and error diagnostics."
references:
  - https://www.mathworks.com/help/matlab/ref/delete.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Filesystem operations always execute on the host CPU. If path inputs reside on the GPU, RunMat gathers them before removing files."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::repl_fs::delete::tests"
  integration: "builtins::io::repl_fs::delete::tests::delete_removes_files_with_wildcard"
---

# What does the `delete` function do in MATLAB / RunMat?
`delete(filename)` removes files from disk, and `delete(obj)` invalidates handle objects or listeners. RunMat mirrors MATLAB behaviour: it accepts character vectors, string scalars, string arrays, char matrices, and cell arrays of names for filesystem removal, gathers GPU-resident inputs automatically, and raises MATLAB-style errors when a file cannot be removed or when handle inputs are invalid.

## How does the `delete` function behave in MATLAB / RunMat?
- Accepts individual paths, string arrays, cell arrays of character vectors, and char matrices. Each element targets one file.
- Accepts handle objects (`handle`) and event listeners, marking them invalid (`isvalid` returns `false`) without touching the filesystem when invoked with non-string inputs.
- Expands shell-style wildcards (`*` and `?`) using MATLAB-compatible rules. Patterns must resolve to existing files; otherwise, the builtin throws `MATLAB:DELETE:FileNotFound`.
- Rejects folders. When a target is a directory, RunMat raises `MATLAB:DELETE:Directories`, matching MATLAB’s “Use rmdir to remove directories” diagnostic.
- Propagates operating-system failures (for example, permission errors or read-only files) through `MATLAB:DELETE:PermissionDenied`.
- Expands `~` to the user’s home directory and resolves relative paths against the current working folder (`pwd`).
- Treats empty character vectors or empty string scalars as invalid inputs and raises `MATLAB:DELETE:EmptyFilename`.
- When passed an empty string array or empty cell array, the builtin performs no action and returns without error, just like MATLAB.

## `delete` Function GPU Execution Behaviour
`delete` performs host-side filesystem I/O. When a path argument lives on the GPU (for example, `gpuArray("scratch.log")`), RunMat gathers the scalar to CPU memory before touching the filesystem. Acceleration providers do not implement dedicated hooks for `delete`, so there are no GPU kernels or device transfers beyond the automatic gathering of inputs.

## Examples of using the `delete` function in MATLAB / RunMat

### Deleting a single temporary file
```matlab
fname = "scratch.txt";
fid = fopen(fname, "w");
fclose(fid);
delete(fname);
```

### Removing multiple files with a wildcard pattern
```matlab
logs = ["log-01.txt", "log-02.txt"];
for name = logs
    fid = fopen(name, "w");
    fclose(fid);
end
delete("log-*.txt");
```

### Deleting files listed in a string array
```matlab
files = ["stageA.dat", "stageB.dat"];
for name = files
    fid = fopen(name, "w");
    fclose(fid);
end
delete(files);
```

### Handling missing files safely with try/catch
```matlab
try
    delete("missing-file.txt");
catch err
    disp(err.identifier)
    disp(err.message)
end
```

### Cleaning up build artifacts stored under your home folder
```matlab
delete("~/runmat/build/*.o");
```

### Deleting char-matrix filenames generated programmatically
```matlab
names = char("stage1.tmp", "stage2.tmp");
for row = 1:size(names, 1)
    fname = strtrim(names(row, :));
    fid = fopen(fname, "w");
    fclose(fid);
end
delete(names);
```

### Deleting graphics handles after use
```matlab
fig = figure;
delete(fig);
tf = isvalid(fig);
```

Expected output:
```matlab
tf =
     0
```

## GPU residency in RunMat (Do I need `gpuArray`?)
No. `delete` executes on the CPU. If a script accidentally wraps path strings in `gpuArray`, RunMat gathers the scalars before issuing filesystem calls. Keeping paths on the GPU provides no benefit.

## FAQ
- **What message IDs does `delete` produce?** Missing files raise `MATLAB:DELETE:FileNotFound`, directories raise `MATLAB:DELETE:Directories`, wildcard syntax issues raise `MATLAB:DELETE:InvalidPattern`, operating-system failures raise `MATLAB:DELETE:PermissionDenied`, and invalid handle inputs raise `MATLAB:DELETE:InvalidHandle`.
- **Can I delete folders with `delete`?** No. MATLAB reserves folder deletion for `rmdir`. RunMat follows suit and throws `MATLAB:DELETE:Directories` when a target is a directory.
- **Does `delete` support multiple filenames at once?** Yes. Pass a string array, a cell array of character vectors, or a char matrix. Each element is deleted in turn.
- **How are wildcard patterns resolved?** RunMat uses MATLAB-compatible globbing: `*` matches any sequence, `?` matches a single character, and the pattern is evaluated relative to the current folder (`pwd`) unless you pass an absolute path.
- **What happens when a wildcard matches nothing?** The builtin raises `MATLAB:DELETE:FileNotFound` (just like MATLAB) and leaves the filesystem unchanged.
- **Do empty arrays raise errors?** Empty string arrays or empty cell arrays simply result in no deletions. Empty strings, however, are invalid and trigger `MATLAB:DELETE:EmptyFilename`.
- **How do GPU inputs behave?** Inputs on the GPU are gathered to the host automatically. No GPU kernels are launched.
- **Does `delete` preserve symbolic links?** Yes. RunMat delegates to the operating system: deleting a symlink removes the link itself, not the target—matching MATLAB.
- **Can I detect failures programmatically?** Wrap the call in `try/catch` and inspect `err.identifier` and `err.message` just as you would in MATLAB.
- **Will `delete` follow relative paths updated by `cd`?** Yes. Paths are interpreted using the process working directory, so calling `cd` before `delete` mirrors MATLAB’s behaviour.

## See Also
[copyfile](./copyfile), [movefile](./movefile), [rmdir](./rmdir), [dir](./dir), [pwd](./pwd)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/io/repl_fs/delete.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/delete.rs)
- Issues: [Open a GitHub ticket](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::delete")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "delete",
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
    notes:
        "Host-only filesystem operation. GPU-resident path values are gathered automatically before deletion.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::delete")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "delete",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Filesystem side-effects are executed immediately; metadata registered for completeness.",
};

#[runtime_builtin(
    name = "delete",
    category = "io/repl_fs",
    summary = "Remove files using MATLAB-compatible wildcard expansion, array inputs, and error diagnostics.",
    keywords = "delete,remove file,wildcard delete,cleanup,temporary files,MATLAB delete",
    accel = "cpu",
    sink = true,
    builtin_path = "crate::builtins::io::repl_fs::delete"
)]
fn delete_builtin(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err(runtime_error(
            MESSAGE_ID_INVALID_INPUT,
            "delete: missing filename input".to_string(),
        ));
    }
    let gathered = gather_arguments(&args)?;

    if gathered.iter().all(is_handle_input) {
        return delete_handles(&gathered);
    }

    if gathered.iter().any(contains_handle_input) {
        return Err(runtime_error(
            MESSAGE_ID_INVALID_HANDLE,
            "delete: cannot mix handle and filename inputs".to_string(),
        ));
    }

    let mut raw_targets = Vec::new();
    for value in &gathered {
        collect_targets(value, &mut raw_targets)?;
    }

    if raw_targets.is_empty() {
        return Ok(Value::Num(0.0));
    }

    for raw in raw_targets {
        delete_target(&raw)?;
    }

    Ok(Value::Num(0.0))
}

fn delete_target(raw: &str) -> Result<(), String> {
    let expanded = expand_user_path(raw, "delete")
        .map_err(|msg| runtime_error(MESSAGE_ID_INVALID_INPUT, msg))?;
    if expanded.is_empty() {
        return Err(runtime_error(
            MESSAGE_ID_EMPTY_FILENAME,
            "delete: filename cannot be empty".to_string(),
        ));
    }

    if contains_wildcards(&expanded) {
        delete_with_pattern(&expanded, raw)
    } else {
        delete_single_path(&PathBuf::from(&expanded), raw)
    }
}

fn delete_with_pattern(pattern: &str, display: &str) -> Result<(), String> {
    validate_wildcard_pattern(pattern, display)?;

    if let Err(PatternError { msg, .. }) = Pattern::new(pattern) {
        return Err(runtime_error(
            MESSAGE_ID_INVALID_PATTERN,
            format!("delete: invalid wildcard pattern '{display}' ({msg})"),
        ));
    }

    let paths = match glob::glob(pattern) {
        Ok(iter) => iter,
        Err(PatternError { msg, .. }) => {
            return Err(runtime_error(
                MESSAGE_ID_INVALID_PATTERN,
                format!("delete: invalid wildcard pattern '{display}' ({msg})"),
            ))
        }
    };

    let mut matches = Vec::new();
    for entry in paths {
        match entry {
            Ok(path) => matches.push(path),
            Err(err) => {
                let problem_path = path_to_string(err.path());
                return Err(runtime_error(
                    MESSAGE_ID_OS_ERROR,
                    format!(
                        "delete: unable to delete '{}' ({})",
                        problem_path,
                        err.error()
                    ),
                ));
            }
        }
    }

    if matches.is_empty() {
        return Err(runtime_error(
            MESSAGE_ID_FILE_NOT_FOUND,
            format!(
                "delete: cannot delete '{}' because it does not exist",
                display
            ),
        ));
    }

    for path in matches {
        let display_path = path_to_string(&path);
        delete_single_path(&path, &display_path)?;
    }
    Ok(())
}

fn delete_single_path(path: &Path, display: &str) -> Result<(), String> {
    match vfs::metadata(path) {
        Ok(meta) => {
            if meta.is_dir() {
                return Err(runtime_error(
                    MESSAGE_ID_IS_DIRECTORY,
                    format!(
                        "delete: cannot delete '{}' because it is a directory (use rmdir instead)",
                        display
                    ),
                ));
            }
            vfs::remove_file(path).map_err(|err| {
                runtime_error(
                    MESSAGE_ID_OS_ERROR,
                    format!("delete: unable to delete '{}' ({})", display, err),
                )
            })
        }
        Err(err) => {
            if err.kind() == io::ErrorKind::NotFound {
                Err(runtime_error(
                    MESSAGE_ID_FILE_NOT_FOUND,
                    format!(
                        "delete: cannot delete '{}' because it does not exist",
                        display
                    ),
                ))
            } else {
                Err(runtime_error(
                    MESSAGE_ID_OS_ERROR,
                    format!("delete: unable to delete '{}' ({})", display, err),
                ))
            }
        }
    }
}

fn validate_wildcard_pattern(pattern: &str, display: &str) -> Result<(), String> {
    if has_unbalanced(pattern, '[', ']') || has_unbalanced(pattern, '{', '}') {
        return Err(runtime_error(
            MESSAGE_ID_INVALID_PATTERN,
            format!("delete: invalid wildcard pattern '{display}'"),
        ));
    }
    Ok(())
}

fn has_unbalanced(pattern: &str, open: char, close: char) -> bool {
    let mut depth = 0usize;
    let mut chars = pattern.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            // Skip escaped characters to avoid false positives
            let _ = chars.next();
            continue;
        }
        if ch == open {
            depth += 1;
        } else if ch == close {
            if depth == 0 {
                return true;
            }
            depth -= 1;
        }
    }
    depth != 0
}

fn gather_arguments(args: &[Value]) -> Result<Vec<Value>, String> {
    let mut out = Vec::with_capacity(args.len());
    for value in args {
        out.push(gather_if_needed(value).map_err(|err| format!("delete: {err}"))?);
    }
    Ok(out)
}

fn collect_targets(value: &Value, targets: &mut Vec<String>) -> Result<(), String> {
    match value {
        Value::String(text) => push_nonempty_target(text, targets),
        Value::CharArray(array) => collect_char_array_targets(array, targets),
        Value::StringArray(array) => collect_string_array_targets(array, targets),
        Value::Cell(cell) => collect_cell_targets(cell, targets),
        _ => Err(runtime_error(
            MESSAGE_ID_INVALID_INPUT,
            ERR_FILENAME_ARG.to_string(),
        )),
    }
}

fn collect_char_array_targets(array: &CharArray, targets: &mut Vec<String>) -> Result<(), String> {
    if array.rows == 0 || array.cols == 0 {
        return Ok(());
    }
    for row in 0..array.rows {
        let mut text = String::with_capacity(array.cols);
        for col in 0..array.cols {
            text.push(array.data[row * array.cols + col]);
        }
        let trimmed = text.trim_end().to_string();
        if trimmed.is_empty() {
            return Err(runtime_error(
                MESSAGE_ID_EMPTY_FILENAME,
                "delete: filename cannot be empty".to_string(),
            ));
        }
        targets.push(trimmed);
    }
    Ok(())
}

fn collect_string_array_targets(
    array: &StringArray,
    targets: &mut Vec<String>,
) -> Result<(), String> {
    for text in &array.data {
        if text.is_empty() {
            return Err(runtime_error(
                MESSAGE_ID_EMPTY_FILENAME,
                "delete: filename cannot be empty".to_string(),
            ));
        }
        targets.push(text.clone());
    }
    Ok(())
}

fn collect_cell_targets(cell: &CellArray, targets: &mut Vec<String>) -> Result<(), String> {
    for handle in &cell.data {
        let value = unsafe { &*handle.as_raw() };
        collect_targets(value, targets)?;
    }
    Ok(())
}

fn delete_handles(values: &[Value]) -> Result<Value, String> {
    let mut mutated_last: Option<Value> = None;
    let mut total = 0usize;
    for value in values {
        total += process_handle_value(value, &mut mutated_last)?;
    }
    if total == 1 {
        Ok(mutated_last.unwrap_or(Value::Num(0.0)))
    } else {
        Ok(Value::Num(0.0))
    }
}

fn process_handle_value(value: &Value, mutated_last: &mut Option<Value>) -> Result<usize, String> {
    match value {
        Value::HandleObject(handle) => {
            let mut invalid = handle.clone();
            invalid.valid = false;
            *mutated_last = Some(Value::HandleObject(invalid));
            Ok(1)
        }
        Value::Listener(listener) => {
            let mut invalid = listener.clone();
            invalid.valid = false;
            invalid.enabled = false;
            *mutated_last = Some(Value::Listener(invalid));
            Ok(1)
        }
        Value::Cell(cell) => {
            let mut total = 0usize;
            for handle in &cell.data {
                let inner = unsafe { &*handle.as_raw() };
                total += process_handle_value(inner, mutated_last)?;
            }
            Ok(total)
        }
        other => Err(runtime_error(
            MESSAGE_ID_INVALID_HANDLE,
            format!("delete: unsupported handle input {other:?}"),
        )),
    }
}

fn is_handle_input(value: &Value) -> bool {
    match value {
        Value::HandleObject(_) | Value::Listener(_) => true,
        Value::Cell(cell) => cell
            .data
            .iter()
            .all(|ptr| is_handle_input(unsafe { &*ptr.as_raw() })),
        _ => false,
    }
}

fn contains_handle_input(value: &Value) -> bool {
    match value {
        Value::HandleObject(_) | Value::Listener(_) => true,
        Value::Cell(cell) => cell
            .data
            .iter()
            .any(|ptr| contains_handle_input(unsafe { &*ptr.as_raw() })),
        _ => false,
    }
}

fn push_nonempty_target(text: &str, targets: &mut Vec<String>) -> Result<(), String> {
    if text.is_empty() {
        Err(runtime_error(
            MESSAGE_ID_EMPTY_FILENAME,
            "delete: filename cannot be empty".to_string(),
        ))
    } else {
        targets.push(text.to_string());
        Ok(())
    }
}

fn runtime_error(message_id: &'static str, message: String) -> String {
    format!("{message_id}: {message}")
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::REPL_FS_TEST_LOCK;
    use super::*;
    use runmat_builtins::{CharArray, StringArray, Value};
    use std::fs::File;
    use tempfile::tempdir;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_removes_single_file() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let target = temp.path().join("single.txt");
        File::create(&target).expect("create");

        let result = delete_builtin(vec![Value::from(target.to_string_lossy().to_string())])
            .expect("delete");
        assert_eq!(result, Value::Num(0.0));
        assert!(!target.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_removes_files_with_wildcard() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let file_a = temp.path().join("log-01.txt");
        let file_b = temp.path().join("log-02.txt");
        File::create(&file_a).expect("create a");
        File::create(&file_b).expect("create b");

        let pattern = temp.path().join("log-*.txt");
        delete_builtin(vec![Value::from(pattern.to_string_lossy().to_string())]).expect("delete");
        assert!(!file_a.exists());
        assert!(!file_b.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_accepts_string_array() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let file_a = temp.path().join("stageA.dat");
        let file_b = temp.path().join("stageB.dat");
        File::create(&file_a).expect("create a");
        File::create(&file_b).expect("create b");

        let array = StringArray::new(
            vec![
                file_a.to_string_lossy().to_string(),
                file_b.to_string_lossy().to_string(),
            ],
            vec![2],
        )
        .expect("string array");

        delete_builtin(vec![Value::StringArray(array)]).expect("delete");
        assert!(!file_a.exists());
        assert!(!file_b.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_accepts_char_array() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let paths: Vec<_> = ["stageA.tmp", "stageB.tmp"]
            .into_iter()
            .map(|name| temp.path().join(name))
            .collect();
        let path_strings: Vec<String> = paths
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();
        let max_len = path_strings.iter().map(|s| s.len()).max().unwrap();
        let mut data: Vec<char> = Vec::with_capacity(path_strings.len() * max_len);

        for (path, path_string) in paths.iter().zip(path_strings.iter()) {
            File::create(path).expect("create file");
            let mut chars: Vec<char> = path_string.chars().collect();
            while chars.len() < max_len {
                chars.push(' ');
            }
            data.extend(&chars);
        }

        let char_array = CharArray::new(data, path_strings.len(), max_len).expect("char array");
        delete_builtin(vec![Value::CharArray(char_array)]).expect("delete");

        for path in paths {
            assert!(!path.exists(), "{path:?} should be removed");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_accepts_cell_array_of_paths() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let file_a = temp.path().join("cellA.dat");
        let file_b = temp.path().join("cellB.dat");
        File::create(&file_a).expect("create cellA");
        File::create(&file_b).expect("create cellB");

        let cell_value = crate::make_cell(
            vec![
                Value::from(file_a.to_string_lossy().to_string()),
                Value::from(file_b.to_string_lossy().to_string()),
            ],
            1,
            2,
        )
        .expect("cell");

        delete_builtin(vec![cell_value]).expect("delete");
        assert!(!file_a.exists());
        assert!(!file_b.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_empty_string_array_is_noop() {
        let array = StringArray::new(Vec::<String>::new(), vec![0]).expect("empty array");
        let result = delete_builtin(vec![Value::StringArray(array)]).expect("delete");
        assert_eq!(result, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_errors_on_empty_string_argument() {
        let err = delete_builtin(vec![Value::from(String::new())]).expect_err("empty string");
        assert!(
            err.starts_with(MESSAGE_ID_EMPTY_FILENAME),
            "expected empty filename error, got {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_errors_on_string_array_empty_element() {
        let array =
            StringArray::new(vec![String::new()], vec![1]).expect("single empty string element");
        let err = delete_builtin(vec![Value::StringArray(array)]).expect_err("empty element");
        assert!(
            err.starts_with(MESSAGE_ID_EMPTY_FILENAME),
            "expected empty filename error, got {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_errors_on_char_array_blank_row() {
        let data = vec![' '; 4];
        let char_array = CharArray::new(data, 1, 4).expect("char matrix");
        let err = delete_builtin(vec![Value::CharArray(char_array)]).expect_err("blank row");
        assert!(
            err.starts_with(MESSAGE_ID_EMPTY_FILENAME),
            "expected empty filename error, got {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_errors_on_invalid_pattern() {
        let pattern = "{invalid*";
        let err = delete_target(pattern).expect_err("invalid pattern should error");
        assert!(
            err.starts_with(MESSAGE_ID_INVALID_PATTERN),
            "expected invalid pattern error, got {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_errors_on_missing_file() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let missing = temp.path().join("missing.txt");
        let missing_str = missing.to_string_lossy().to_string();
        let err = delete_target(&missing_str).expect_err("error");
        assert!(
            err.starts_with(MESSAGE_ID_FILE_NOT_FOUND),
            "expected file not found identifier, got {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_errors_on_directory() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let dir = temp.path().join("dir");
        std::fs::create_dir(&dir).expect("create dir");
        let dir_display = dir.to_string_lossy().to_string();
        let err = delete_single_path(&dir, &dir_display).expect_err("error");
        assert!(
            err.starts_with(MESSAGE_ID_IS_DIRECTORY),
            "expected directory identifier, got {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_handle_returns_invalid_handle() {
        let handle =
            crate::new_handle_object_builtin("ReplFsDeleteTestHandle".to_string()).expect("handle");
        let result = delete_builtin(vec![handle]).expect("delete handle");
        match result {
            Value::HandleObject(h) => {
                assert!(!h.valid, "handle should be marked invalid");
                let valid_value =
                    crate::isvalid_builtin(Value::HandleObject(h.clone())).expect("isvalid");
                match valid_value {
                    Value::Bool(flag) => assert!(!flag, "isvalid should report false after delete"),
                    other => panic!("expected bool from isvalid, got {other:?}"),
                }
            }
            other => panic!("expected handle result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_rejects_mixed_handle_and_filename() {
        let handle =
            crate::new_handle_object_builtin("ReplFsDeleteTestHandle".to_string()).expect("handle");
        let err = delete_builtin(vec![
            handle,
            Value::from("mixed-handle-path.txt".to_string()),
        ])
        .expect_err("expected mixed error");
        assert!(
            err.starts_with(MESSAGE_ID_INVALID_HANDLE),
            "expected invalid handle identifier, got {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_accepts_cell_of_handles() {
        let handle_a =
            crate::new_handle_object_builtin("ReplFsDeleteTestHandle".to_string()).expect("handle");
        let handle_b =
            crate::new_handle_object_builtin("ReplFsDeleteTestHandle".to_string()).expect("handle");
        let cell = crate::make_cell(vec![handle_a, handle_b], 1, 2).expect("cell of handles");
        let result = delete_builtin(vec![cell]).expect("delete handles");
        assert_eq!(result, Value::Num(0.0));
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_runs_with_wgpu_provider_registered() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

        let temp = tempdir().expect("temp dir");
        let path = temp.path().join("wgpu-file.txt");
        File::create(&path).expect("create file");

        delete_builtin(vec![Value::from(path.to_string_lossy().to_string())]).expect("delete");
        assert!(!path.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_parse() {
        let blocks = crate::builtins::common::test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
