//! MATLAB-compatible `ls` builtin for RunMat.

use std::collections::HashSet;
use std::env;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use glob::glob;
use runmat_builtins::{CharArray, StringArray, Value};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::{
    contains_wildcards, expand_user_path, path_to_string, sort_entries,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::gather_if_needed;

#[cfg(feature = "doc_export")]
#[runmat_macros::register_doc_text(name = "ls")]
pub const DOC_MD: &str = r#"---
title: "ls"
category: "io/repl_fs"
keywords: ["ls", "list files", "folder contents", "wildcard listing", "dir"]
summary: "List files and folders in the current directory or matching a wildcard pattern."
references:
  - https://www.mathworks.com/help/matlab/ref/ls.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Executes entirely on the host CPU. When the input argument resides on the GPU, RunMat gathers it to the host before expanding any patterns."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::repl_fs::ls::tests"
  integration: "builtins::io::repl_fs::ls::tests::ls_handles_wildcard_patterns"
---

# What does the `ls` function do in MATLAB / RunMat?
`ls` lists the contents of a directory. Without inputs, it shows the files and folders in the current working directory. With an argument, it lists the items that match a path or wildcard pattern. The return value is a character array whose rows contain the matching names padded with spaces, mirroring MATLAB behaviour.

## How does the `ls` function behave in MATLAB / RunMat?
- `ls` without outputs displays the directory listing in the REPL. When you assign the result to a variable, you receive a character array with one row per item, padded with spaces to a fixed width.
- `ls(name)` accepts character vectors or string scalars. The argument may be an absolute path, a relative path, or a wildcard such as `*.m` or `build/*`.
- Paths beginning with `~` or `~/` expand to the current user's home directory, matching MATLAB desktop behaviour on every supported platform.
- Wildcards follow MATLAB rules: `*` matches any sequence of characters and `?` matches exactly one character. The pattern is evaluated relative to the current folder unless it contains a leading path.
- Directory results include the platform-specific file separator (`/` on Unix, `\` on Windows) so you can quickly distinguish folders from files, and duplicate matches are removed automatically.
- Empty matches return a `0×0` character array rather than raising an error, enabling idioms such as `isempty(ls("*.tmp"))`.
- Only scalar inputs are supported. Multi-row character arrays, multi-element string arrays, numeric values, or logical arguments raise `ls: name must be a character vector or string scalar`, mirroring MATLAB's diagnostics.

## `ls` Function GPU Execution Behaviour
`ls` is an I/O-centric builtin with no GPU execution path. When the argument is a GPU-resident scalar (for example, the output of `gather` that remained on the device), RunMat gathers it to the host before expanding wildcards. No device kernels run, and acceleration providers do not need to implement hooks for this builtin.

## GPU residency in RunMat (Do I need `gpuArray`?)
No. Directory listings are always computed on the CPU. Passing `gpuArray("*.m")` offers no benefit: RunMat will gather the value automatically before enumerating files, and results are materialised as host-resident character arrays.

## Examples of using the `ls` function in MATLAB / RunMat

### List Files In The Current Folder
```matlab
ls
```
Expected output (example):
```matlab
README.md
src/
tests/
```

### List Only MATLAB Files Using Wildcards
```matlab
ls("*.m")
```
Expected output:
```matlab
script.m
solver.m
test_helper.m
```

### Capture A Directory Listing In A Variable
```matlab
listing = ls;
disp(listing(1, :));  % Display the first entry
```
Expected output:
```matlab
README.md
```

### List The Contents Of A Specific Folder
```matlab
tempDir = fullfile(pwd, "tmp");
ls(tempDir);
```
Expected output:
```matlab
tmpfile.txt
subdir/
```

### Use Wildcards With Nested Paths
```matlab
ls("src/**/*.rs")
```
Expected output:
```matlab
src/main.rs
src/utils/helpers.rs
```

### Handle Missing Matches Gracefully
```matlab
if isempty(ls("*.tmp"))
    disp("No temporary files found.");
end
```
Expected output:
```matlab
No temporary files found.
```

### Combine `ls` With `cd` For Temporary Directory Changes
```matlab
start = cd("examples");
cleanup = onCleanup(@() cd(start));
files = ls("*.m");
```
Expected output:
```matlab
animation.m
plot_surface.m
```

## FAQ
- **What types can I pass to `ls`?** Use character vectors or string scalars. Other types – including numeric values, logical arrays, or string arrays with multiple elements – produce an error.
- **Does `ls` support wildcard patterns?** Yes. Use `*` to match any sequence and `?` for a single character. Patterns are resolved relative to the current folder unless you provide an absolute path.
- **How are directories indicated in the output?** Directory entries end with the platform-specific file separator (`/` or `\`) so you can distinguish them from files.
- **What happens if there are no matches?** The function returns a `0×0` character array. You can test for this with `isempty` or by checking `size`.
- **Can I list the contents of another drive or mounted volume?** Yes. Provide the absolute path, e.g., `ls("D:\Projects")` on Windows or `ls("/Volumes/Data")` on macOS.
- **Does `ls` follow symbolic links?** The function reports the link itself; if the link points to a directory, it receives the directory suffix but the listing does not recursively expand the target.
- **Will `ls` respect the current working directory set by `cd`?** Absolutely. `ls` always evaluates relative paths against whatever `pwd` reports.
- **Is the output sorted?** Yes. RunMat sorts case-insensitively on Windows and case-sensitively on Unix-like systems to match MATLAB conventions.
- **How do I list hidden files?** Hidden files are included when they match the pattern you specify. On Unix-like systems, prepend a dot: `ls(".*")`.
- **Can I send the output directly to another builtin?** Yes. Because the result is a character array, you can convert it to a string array with `string(ls)` or operate on individual rows using indexing.

## See Also
[pwd](./pwd), [cd](./cd), [fopen](../filetext/fopen), [fclose](../filetext/fclose)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/io/repl_fs/ls.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/ls.rs)
- Found an issue? [Open a GitHub ticket](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ls",
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
        "Host-only filesystem builtin. Providers do not participate; any GPU-resident argument is gathered before path expansion.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ls",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "I/O builtins are excluded from fusion plans; metadata registered for introspection completeness.",
};

#[runtime_builtin(
    name = "ls",
    category = "io/repl_fs",
    summary = "List files and folders in the current directory or matching a wildcard pattern.",
    keywords = "ls,list files,folder contents,wildcard listing,dir",
    accel = "cpu"
)]
fn ls_builtin(args: Vec<Value>) -> Result<Value, String> {
    let gathered = gather_arguments(&args)?;
    if gathered.len() > 1 {
        return Err("ls: too many input arguments".to_string());
    }

    let entries = if let Some(value) = gathered.first() {
        list_from_value(value)?
    } else {
        list_current_directory()?
    };

    rows_to_char_array(&entries)
}

fn list_from_value(value: &Value) -> Result<Vec<String>, String> {
    let names = patterns_from_value(value)?;
    if names.is_empty() {
        return list_current_directory();
    }

    let mut seen = HashSet::new();
    let mut combined = Vec::new();

    for pattern in names {
        let matches = list_for_pattern(&pattern)?;
        for entry in matches {
            if seen.insert(entry.clone()) {
                combined.push(entry);
            }
        }
    }

    sort_entries(&mut combined);
    Ok(combined)
}

fn list_current_directory() -> Result<Vec<String>, String> {
    let cwd = env::current_dir()
        .map_err(|err| format!("ls: unable to determine current directory ({err})"))?;
    list_directory(&cwd)
}

fn list_for_pattern(raw: &str) -> Result<Vec<String>, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return list_current_directory();
    }

    let expanded = expand_user_path(trimmed, "ls")?;

    if contains_wildcards(&expanded) {
        list_glob_pattern(&expanded, trimmed)
    } else {
        list_path(&expanded, trimmed)
    }
}

fn list_directory(dir: &Path) -> Result<Vec<String>, String> {
    let mut entries = Vec::new();
    let dir_str = path_to_string(dir);
    let read_dir =
        vfs::read_dir(dir).map_err(|err| format!("ls: unable to access '{dir_str}' ({err})"))?;

    for entry in read_dir {
        let name = entry.file_name().to_string_lossy();
        if name == "." || name == ".." {
            continue;
        }
        let mut display = name.into_owned();
        append_directory_suffix(&mut display, entry.is_dir());
        entries.push(display);
    }

    sort_entries(&mut entries);
    Ok(entries)
}

fn list_path(expanded: &str, original: &str) -> Result<Vec<String>, String> {
    let path = PathBuf::from(expanded);
    match vfs::metadata(&path) {
        Ok(metadata) => {
            if metadata.is_dir() {
                list_directory(&path)
            } else {
                let mut text = path_to_string(&path);
                append_directory_suffix(&mut text, false);
                Ok(vec![text])
            }
        }
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(Vec::new()),
        Err(err) => Err(format!("ls: unable to access '{original}' ({err})")),
    }
}

fn list_glob_pattern(expanded: &str, original: &str) -> Result<Vec<String>, String> {
    let mut entries = Vec::new();

    let matcher =
        glob(expanded).map_err(|err| format!("ls: invalid pattern '{original}' ({err})"))?;
    for item in matcher {
        match item {
            Ok(path) => {
                let is_dir = vfs::symlink_metadata(&path)
                    .map(|meta| meta.is_dir())
                    .unwrap_or(false);
                let mut name = path_to_string(&path);
                append_directory_suffix(&mut name, is_dir);
                entries.push(name);
            }
            Err(err) => {
                return Err(format!(
                    "ls: unable to enumerate matches for '{original}' ({err})"
                ))
            }
        }
    }

    Ok(entries)
}

fn rows_to_char_array(rows: &[String]) -> Result<Value, String> {
    if rows.is_empty() {
        let array = CharArray::new(Vec::new(), 0, 0).map_err(|e| format!("ls: {e}"))?;
        return Ok(Value::CharArray(array));
    }

    let width = rows
        .iter()
        .map(|row| row.chars().count())
        .max()
        .unwrap_or(0);

    let mut data = Vec::with_capacity(rows.len() * width);
    for row in rows {
        let mut chars: Vec<char> = row.chars().collect();
        while chars.len() < width {
            chars.push(' ');
        }
        data.extend(chars);
    }

    let array = CharArray::new(data, rows.len(), width).map_err(|e| format!("ls: {e}"))?;
    Ok(Value::CharArray(array))
}

fn patterns_from_value(value: &Value) -> Result<Vec<String>, String> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::StringArray(StringArray { data, .. }) => {
            if data.len() == 1 {
                Ok(vec![data[0].clone()])
            } else {
                Err("ls: name must be a character vector or string scalar".to_string())
            }
        }
        Value::CharArray(chars) => {
            if chars.rows != 1 {
                return Err("ls: name must be a character vector or string scalar".to_string());
            }
            let mut row = String::with_capacity(chars.cols);
            for c in 0..chars.cols {
                row.push(chars.data[c]);
            }
            Ok(vec![row.trim_end().to_string()])
        }
        _ => Err("ls: name must be a character vector or string scalar".to_string()),
    }
}

fn append_directory_suffix(text: &mut String, is_dir: bool) {
    if is_dir {
        let sep = std::path::MAIN_SEPARATOR;
        if !text.ends_with(sep) {
            text.push(sep);
        }
    }
}

fn gather_arguments(args: &[Value]) -> Result<Vec<Value>, String> {
    let mut out = Vec::with_capacity(args.len());
    for value in args {
        out.push(gather_if_needed(value).map_err(|err| format!("ls: {err}"))?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::REPL_FS_TEST_LOCK;
    use super::*;
    use runmat_builtins::CharArray;
    use runmat_filesystem::{self as fs, File};
    use tempfile::tempdir;

    struct DirGuard {
        original: PathBuf,
    }

    impl DirGuard {
        fn new() -> Self {
            let original = env::current_dir().expect("current dir");
            Self { original }
        }
    }

    impl Drop for DirGuard {
        fn drop(&mut self) {
            let _ = env::set_current_dir(&self.original);
        }
    }

    fn rows_from_value(value: Value) -> Vec<String> {
        match value {
            Value::CharArray(CharArray { data, rows, cols }) => {
                let mut out = Vec::with_capacity(rows);
                for r in 0..rows {
                    let mut row = String::with_capacity(cols);
                    for c in 0..cols {
                        row.push(data[r * cols + c]);
                    }
                    out.push(row.trim_end().to_string());
                }
                out
            }
            other => panic!("expected CharArray result, got {other:?}"),
        }
    }

    #[test]
    fn ls_lists_current_directory_when_no_arguments() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let guard = DirGuard::new();

        let dir = tempdir().expect("tempdir");
        env::set_current_dir(dir.path()).expect("switch temp dir");
        File::create(dir.path().join("alpha.txt")).expect("create file");
        fs::create_dir(dir.path().join("beta")).expect("create dir");

        let value = ls_builtin(Vec::new()).expect("ls");
        let mut rows = rows_from_value(value);
        rows.sort();

        let sep = std::path::MAIN_SEPARATOR.to_string();
        assert_eq!(rows.len(), 2);
        assert!(rows.contains(&"alpha.txt".to_string()));
        assert!(rows.contains(&format!("beta{sep}")));

        drop(guard);
    }

    #[test]
    fn ls_lists_specific_directory_contents() {
        let dir = tempdir().expect("tempdir");
        File::create(dir.path().join("data.csv")).expect("create file");
        fs::create_dir(dir.path().join("nested")).expect("create dir");

        let path = dir.path().to_string_lossy().to_string();
        let value = ls_builtin(vec![Value::from(path)]).expect("ls");
        let mut rows = rows_from_value(value);
        rows.sort();

        let sep = std::path::MAIN_SEPARATOR.to_string();
        assert_eq!(rows, vec!["data.csv".to_string(), format!("nested{sep}")]);
    }

    #[test]
    fn ls_handles_wildcard_patterns() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let guard = DirGuard::new();

        let dir = tempdir().expect("tempdir");
        env::set_current_dir(dir.path()).expect("switch temp dir");

        File::create("alpha.m").expect("alpha");
        File::create("beta.txt").expect("beta");
        File::create("gamma.m").expect("gamma");

        let value = ls_builtin(vec![Value::from("*.m")]).expect("ls pattern");
        let mut rows = rows_from_value(value);
        rows.sort();

        assert_eq!(rows, vec!["alpha.m".to_string(), "gamma.m".to_string()]);

        drop(guard);
    }

    #[test]
    fn ls_returns_path_for_single_file() {
        let dir = tempdir().expect("tempdir");
        let file_path = dir.path().join("report.md");
        File::create(&file_path).expect("create file");

        let value = ls_builtin(vec![Value::from(file_path.to_string_lossy().to_string())])
            .expect("ls file");
        let rows = rows_from_value(value);
        assert_eq!(rows, vec![file_path.to_string_lossy().to_string()]);
    }

    #[test]
    fn ls_returns_empty_for_missing_matches() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("*.nomatch");
        let value =
            ls_builtin(vec![Value::from(path.to_string_lossy().to_string())]).expect("ls missing");
        match value {
            Value::CharArray(array) => {
                assert_eq!(array.rows, 0);
                assert_eq!(array.cols, 0);
                assert!(array.data.is_empty());
            }
            other => panic!("expected CharArray result, got {other:?}"),
        }
    }

    #[test]
    fn ls_accepts_string_scalar_input() {
        let dir = tempdir().expect("tempdir");
        File::create(dir.path().join("file.dat")).expect("create file");

        let array = StringArray::new_2d(vec![dir.path().to_string_lossy().to_string()], 1, 1)
            .expect("string array");
        let value = ls_builtin(vec![Value::StringArray(array)]).expect("ls string");
        let rows = rows_from_value(value);
        assert!(rows.iter().any(|row| row.contains("file.dat")));
    }

    #[test]
    fn ls_rejects_numeric_argument() {
        let err = ls_builtin(vec![Value::Num(1.0)]).expect_err("expected error");
        assert_eq!(err, "ls: name must be a character vector or string scalar");
    }

    #[test]
    fn doc_examples_present() {
        let blocks = crate::builtins::common::test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
