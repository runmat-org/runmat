//! MATLAB-compatible `dir` builtin for RunMat.

use std::env;
use std::ffi::OsString;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use chrono::{DateTime, Duration, Local, NaiveDate};
use glob::glob;
use runmat_builtins::{StructValue, Value};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::{
    compare_names, contains_wildcards, expand_user_path, path_to_string,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::console::{record_console_output, ConsoleStream};
use crate::{gather_if_needed, make_cell};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "dir",
        builtin_path = "crate::builtins::io::repl_fs::dir"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "dir"
category: "io/repl_fs"
keywords: ["dir", "list files", "folder contents", "metadata", "wildcard", "struct array"]
summary: "Return file and folder information in a MATLAB-compatible struct array."
references:
  - https://www.mathworks.com/help/matlab/ref/dir.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs entirely on the CPU. If an input argument resides on the GPU, RunMat gathers it before resolving the path."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::repl_fs::dir::tests"
  integration: "builtins::io::repl_fs::dir::tests::dir_lists_current_directory"
---

# What does the `dir` function do in MATLAB / RunMat?
`dir` returns metadata for files and folders. Without arguments, it lists the current working
directory. With an argument, it can list a specific directory, match wildcard patterns, or report
information about a single file. The result is a column vector of structs matching MATLAB's
signature: each element contains `name`, `folder`, `date`, `bytes`, `isdir`, and `datenum` fields.

## How does the `dir` function behave in MATLAB / RunMat?
- `dir` with no inputs returns the contents of the current folder (including `.` and `..`).
- `dir(path)` accepts absolute or relative paths, character vectors, or string scalars. When the path
  names a directory, its contents are listed; when it names a file, metadata for that file alone is
  returned.
- `dir(pattern)` and `dir(folder, pattern)` support wildcard expansion with `*` and `?`. Patterns are
  evaluated relative to the supplied folder or the current working directory.
- The returned struct fields mirror MATLAB:
  - `name`: file or folder name.
  - `folder`: absolute path to the containing folder.
  - `date`: last-modified timestamp formatted as `dd-MMM-yyyy HH:mm:ss` in local time.
  - `bytes`: file size in bytes (`0` for directories and wildcard-only results).
  - `isdir`: logical flag indicating whether the entry is a directory.
  - `datenum`: serial date number compatible with MATLAB's `datenum`.
- Results are sorted using MATLAB-style ordering (case-insensitive on Windows, case-sensitive on
  Unix-like systems).
- Empty matches return a `0×1` struct array so idioms like `isempty(dir("*.tmp"))` continue working.
- Invalid arguments (numeric, logical, non-scalar string arrays) raise MATLAB-compatible diagnostics.

## `dir` Function GPU Execution Behaviour
Because `dir` interacts with the host filesystem, it is executed entirely on the CPU. If an input
argument resides on the GPU (for example, a scalar string created by another accelerated builtin),
RunMat gathers it to host memory before expanding patterns. Acceleration providers do not implement
hooks for `dir`, and the result always lives on the host.

## GPU residency in RunMat (Do I need `gpuArray`?)
No. `dir` is an I/O-bound builtin and always operates on host data. Passing GPU-resident strings is
supported, but RunMat gathers them automatically and there is no benefit to manually calling
`gpuArray`.

## Examples of using the `dir` function in MATLAB / RunMat

### List Files In The Current Folder
```matlab
listing = dir;
{listing.name}
```
Expected output (example):
```matlab
    {'.'}    {'..'}    {'README.md'}    {'src'}
```

### List Only MATLAB Files Using Wildcards
```matlab
scripts = dir("*.m");
cellstr({scripts.name})
```
Expected output:
```matlab
    {'solver.m'}
    {'test_helper.m'}
```

### Capture Metadata For A Specific File
```matlab
info = dir("data/results.csv");
info.bytes    % file size in bytes
info.isdir    % false
```

### List The Contents Of A Specific Folder
```matlab
tmp = dir(fullfile(pwd, "tmp"));
{tmp.name}'
```
Expected output:
```matlab
    '.'    '..'    'tmpfile.txt'    'subdir'
```

### Combine Folder And Pattern Arguments
```matlab
images = dir("assets", "*.png");
{images.name}
```
Expected output:
```matlab
    {'logo.png'}    {'splash.png'}
```

### Use Tilde Expansion For Home Directory
```matlab
home_listing = dir("~");
```
Expected output (example):
```matlab
home_listing(1).folder
ans =
    '/Users/example'
```

### Handle Missing Matches Gracefully
```matlab
if isempty(dir("*.cache"))
    disp("No cache files found.");
end
```
Expected output:
```matlab
No cache files found.
```

## FAQ
- **What types can I pass to `dir`?** Character vectors or string scalars. Other types (numeric,
  logical, multi-element string arrays, or cells) raise an error.
- **Does `dir` support recursive wildcards?** Yes. Patterns such as `"**/*.m"` are honoured through
  the standard globbing rules.
- **Why do I see `.` and `..` entries?** MATLAB includes them for directory listings; RunMat mirrors
  this behaviour so scripts relying on their presence continue to work.
- **What is the `datenum` field?** A MATLAB serial date number representing the last modification
  time in local time. Use `datetime([entry.datenum])` to convert multiple entries.
- **Are symbolic links distinguished from folders?** Symlinks are reported using the metadata
  provided by the operating system. If the link targets a directory, `isdir` is `true`.
- **Can I pass GPU-resident strings?** Yes, but RunMat gathers them automatically before computing the
  directory listing.
- **How are errors reported?** Error messages are prefixed with `dir:` and match MATLAB's argument
  diagnostics wherever possible.

## See Also
[ls](./ls), [pwd](./pwd), [cd](./cd), [fullfile](./fullfile)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/io/repl_fs/dir.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/dir.rs)
- Found an issue? [Open a GitHub ticket](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::dir")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "dir",
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
    notes: "Host-only filesystem builtin. Providers do not participate; GPU-resident inputs are gathered to host memory.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::dir")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "dir",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "I/O builtins do not participate in fusion plans; metadata registered for completeness.",
};

#[runtime_builtin(
    name = "dir",
    category = "io/repl_fs",
    summary = "Return file and folder information in a MATLAB-compatible struct array.",
    keywords = "dir,list files,folder contents,metadata,wildcard,struct array",
    accel = "cpu",
    suppress_auto_output = true,
    builtin_path = "crate::builtins::io::repl_fs::dir"
)]
fn dir_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let gathered = gather_arguments(&args)?;
    let records = match gathered.len() {
        0 => list_current_directory()?,
        1 => list_from_single_value(&gathered[0])?,
        2 => list_with_folder_and_pattern(&gathered[0], &gathered[1])?,
        _ => return Err((("dir: too many input arguments".to_string())).into()),
    };
    emit_dir_stdout(&records);
    records_to_value(records).map_err(Into::into)
}

#[derive(Clone)]
struct DirRecord {
    name: String,
    folder: String,
    date: String,
    bytes: f64,
    is_dir: bool,
    datenum: f64,
}

fn gather_arguments(args: &[Value]) -> Result<Vec<Value>, String> {
    let mut out = Vec::with_capacity(args.len());
    for value in args {
        out.push(gather_if_needed(value).map_err(|err| format!("dir: {err}"))?);
    }
    Ok(out)
}

fn list_current_directory() -> Result<Vec<DirRecord>, String> {
    let cwd = env::current_dir()
        .map_err(|err| format!("dir: unable to determine current directory ({err})"))?;
    let mut records = list_directory(&cwd, true)?;
    sort_records(&mut records);
    Ok(records)
}

fn list_from_single_value(value: &Value) -> Result<Vec<DirRecord>, String> {
    let text = scalar_text(
        value,
        "dir: name must be a character vector or string scalar",
    )?;
    list_from_text(&text)
}

fn list_with_folder_and_pattern(
    folder_value: &Value,
    pattern_value: &Value,
) -> Result<Vec<DirRecord>, String> {
    let folder_text = scalar_text(
        folder_value,
        "dir: folder must be a character vector or string scalar",
    )?;
    let pattern_text = scalar_text(
        pattern_value,
        "dir: pattern must be a character vector or string scalar",
    )?;

    let expanded_folder = expand_user_path(folder_text.trim(), "dir")?;
    if contains_wildcards(&expanded_folder) {
        return Err("dir: folder input must not contain wildcard characters".to_string());
    }

    let base_path = PathBuf::from(&expanded_folder);
    let trimmed_pattern = pattern_text.trim();
    if trimmed_pattern.is_empty() {
        let mut records = list_directory(&base_path, true)?;
        sort_records(&mut records);
        return Ok(records);
    }

    if contains_wildcards(trimmed_pattern) {
        let mut pattern_path = base_path.clone();
        pattern_path.push(trimmed_pattern);
        let pattern_str = path_to_string(&pattern_path);
        let mut records = list_glob_pattern(&pattern_str, trimmed_pattern)?;
        sort_records(&mut records);
        Ok(records)
    } else {
        let mut target_path = base_path.clone();
        target_path.push(trimmed_pattern);
        list_path(&path_to_string(&target_path), trimmed_pattern)
    }
}

fn list_from_text(text: &str) -> Result<Vec<DirRecord>, String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return list_current_directory();
    }

    let expanded = expand_user_path(trimmed, "dir")?;
    if contains_wildcards(&expanded) {
        let mut records = list_glob_pattern(&expanded, trimmed)?;
        sort_records(&mut records);
        Ok(records)
    } else {
        list_path(&expanded, trimmed)
    }
}

fn list_path(expanded: &str, original: &str) -> Result<Vec<DirRecord>, String> {
    let path = PathBuf::from(expanded);
    match vfs::metadata(&path) {
        Ok(metadata) => {
            if metadata.is_dir() {
                let mut records = list_directory(&path, true)?;
                sort_records(&mut records);
                Ok(records)
            } else {
                let folder_path = path
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| PathBuf::from("."));
                let folder = absolute_folder_string(&folder_path)?;
                let name = path
                    .file_name()
                    .map(|os| os.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path_to_string(&path));
                let (date, datenum) = timestamp_fields(metadata.modified());
                let record = DirRecord {
                    name,
                    folder,
                    date,
                    bytes: metadata.len() as f64,
                    is_dir: false,
                    datenum,
                };
                Ok(vec![record])
            }
        }
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(Vec::new()),
        Err(err) => Err(format!("dir: unable to access '{original}' ({err})")),
    }
}

fn list_glob_pattern(expanded: &str, original: &str) -> Result<Vec<DirRecord>, String> {
    let mut records = Vec::new();

    let matcher =
        glob(expanded).map_err(|err| format!("dir: invalid pattern '{original}' ({err})"))?;
    for item in matcher {
        match item {
            Ok(path) => {
                let metadata = vfs::symlink_metadata(&path).ok();
                let is_dir = metadata.as_ref().map(|m| m.is_dir()).unwrap_or(false);
                let folder_path = path
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| PathBuf::from("."));
                let folder = absolute_folder_string(&folder_path)?;
                let name = path
                    .file_name()
                    .map(|os| os.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path_to_string(&path));
                let (date, datenum) =
                    timestamp_fields(metadata.as_ref().and_then(|m| m.modified()));
                let bytes = if is_dir {
                    0.0
                } else {
                    metadata.as_ref().map(|m| m.len() as f64).unwrap_or(0.0)
                };
                records.push(DirRecord {
                    name,
                    folder,
                    date,
                    bytes,
                    is_dir,
                    datenum,
                });
            }
            Err(err) => {
                return Err(format!(
                    "dir: unable to enumerate matches for '{original}' ({err})"
                ))
            }
        }
    }

    Ok(records)
}

fn list_directory(dir: &Path, include_special: bool) -> Result<Vec<DirRecord>, String> {
    let folder = absolute_folder_string(dir)?;
    let dir_metadata = vfs::metadata(dir).ok();
    let mut records = Vec::new();

    if include_special {
        records.push(make_special(".", &folder, dir_metadata.as_ref()));
        records.push(make_special("..", &folder, dir_metadata.as_ref()));
    }

    let dir_display = folder.clone();
    let read_dir = vfs::read_dir(dir)
        .map_err(|err| format!("dir: unable to access '{dir_display}' ({err})"))?;
    for entry in read_dir {
        let name_os: &OsString = entry.file_name();
        let name = name_os.to_string_lossy().into_owned();
        if name == "." || name == ".." {
            continue;
        }

        let path = entry.path().to_path_buf();
        let metadata = vfs::metadata(&path)
            .or_else(|_| vfs::symlink_metadata(&path))
            .map_err(|err| format!("dir: unable to read metadata for '{}' ({err})", name))?;
        records.push(record_from_metadata(&folder, name, &metadata));
    }

    Ok(records)
}

fn records_to_value(records: Vec<DirRecord>) -> Result<Value, String> {
    if records.is_empty() {
        return make_cell(Vec::new(), 0, 1);
    }
    let rows = records.len();
    let mut values = Vec::with_capacity(rows);
    for record in records {
        let mut st = StructValue::new();
        st.fields
            .insert("name".to_string(), Value::String(record.name));
        st.fields
            .insert("folder".to_string(), Value::String(record.folder));
        st.fields
            .insert("date".to_string(), Value::String(record.date));
        st.fields
            .insert("bytes".to_string(), Value::Num(record.bytes));
        st.fields
            .insert("isdir".to_string(), Value::Bool(record.is_dir));
        st.fields
            .insert("datenum".to_string(), Value::Num(record.datenum));
        values.push(Value::Struct(st));
    }
    make_cell(values, rows, 1)
}

fn emit_dir_stdout(records: &[DirRecord]) {
    if records.is_empty() {
        return;
    }
    let mut lines = Vec::with_capacity(records.len());
    for record in records {
        let size_field = if record.is_dir {
            "<DIR>".to_string()
        } else {
            format!("{:>10}", record.bytes as i64)
        };
        let mut name = record.name.clone();
        if record.is_dir && !name.ends_with(std::path::MAIN_SEPARATOR) {
            name.push(std::path::MAIN_SEPARATOR);
        }
        lines.push(format!("{:<20} {:>10} {}", record.date, size_field, name));
    }
    record_console_output(ConsoleStream::Stdout, lines.join("\n"));
}

fn scalar_text(value: &Value, error: &str) -> Result<String, String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::CharArray(chars) if chars.rows == 1 => {
            let mut row = String::with_capacity(chars.cols);
            for c in 0..chars.cols {
                row.push(chars.data[c]);
            }
            Ok(row.trim_end().to_string())
        }
        _ => Err(error.to_string()),
    }
}

fn absolute_folder_string(path: &Path) -> Result<String, String> {
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()
            .map_err(|err| format!("dir: unable to determine current directory ({err})"))?
            .join(path)
    };
    let normalized = vfs::canonicalize(&joined).unwrap_or(joined);
    Ok(path_to_string(&normalized))
}

fn record_from_metadata(folder: &str, name: String, metadata: &vfs::FsMetadata) -> DirRecord {
    let is_dir = metadata.is_dir();
    let (date, datenum) = timestamp_fields(metadata.modified());
    DirRecord {
        name,
        folder: folder.to_string(),
        date,
        bytes: if is_dir { 0.0 } else { metadata.len() as f64 },
        is_dir,
        datenum,
    }
}

fn make_special(name: &str, folder: &str, metadata: Option<&vfs::FsMetadata>) -> DirRecord {
    let (date, datenum) = timestamp_fields(metadata.and_then(|m| m.modified()));
    DirRecord {
        name: name.to_string(),
        folder: folder.to_string(),
        date,
        bytes: 0.0,
        is_dir: true,
        datenum,
    }
}

fn timestamp_fields(time: Option<SystemTime>) -> (String, f64) {
    const DEFAULT_DATE: &str = "01-Jan-1970 00:00:00";
    const DEFAULT_DATENUM: f64 = 719_529.0;
    match time {
        Some(t) => {
            let datetime: DateTime<Local> = DateTime::<Local>::from(t);
            let date = datetime.format("%d-%b-%Y %H:%M:%S").to_string();
            let datenum = datenum_from_datetime(datetime);
            (date, datenum)
        }
        None => (DEFAULT_DATE.to_string(), DEFAULT_DATENUM),
    }
}

fn datenum_from_datetime(datetime: DateTime<Local>) -> f64 {
    const SECONDS_PER_DAY: f64 = 86_400.0;
    const UNIX_DN: f64 = 719_529.0;

    let naive = datetime.naive_local();
    let base_date = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    let base = base_date.and_hms_opt(0, 0, 0).unwrap();
    let duration = naive - base;

    let secs = duration.num_seconds();
    let nanos_duration = duration - Duration::seconds(secs);
    let nanos = nanos_duration.num_nanoseconds().unwrap_or(0);
    let total_seconds = secs as f64 + nanos as f64 / 1_000_000_000.0;

    total_seconds / SECONDS_PER_DAY + UNIX_DN
}

fn sort_records(records: &mut [DirRecord]) {
    records.sort_by(|a, b| compare_names(&a.name, &b.name));
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::REPL_FS_TEST_LOCK;
    use super::*;
    use runmat_builtins::{CharArray, StringArray, StructValue as TestStruct};
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

    #[cfg(not(windows))]
    struct HomeGuard {
        original: Option<String>,
    }

    #[cfg(not(windows))]
    impl HomeGuard {
        fn set_to(path: &Path) -> Self {
            let original = env::var("HOME").ok();
            env::set_var("HOME", path_to_string(path));
            Self { original }
        }
    }

    #[cfg(not(windows))]
    impl Drop for HomeGuard {
        fn drop(&mut self) {
            if let Some(ref value) = self.original {
                env::set_var("HOME", value);
            } else {
                env::remove_var("HOME");
            }
        }
    }

    fn structs_from_value(value: Value) -> Vec<TestStruct> {
        match value {
            Value::Cell(cell) => cell
                .data
                .iter()
                .map(|ptr| unsafe { &*ptr.as_raw() }.clone())
                .map(|value| match value {
                    Value::Struct(st) => st,
                    other => panic!("expected struct entry, got {other:?}"),
                })
                .collect(),
            Value::Struct(st) => vec![st],
            other => panic!("expected struct array, got {other:?}"),
        }
    }

    fn field_string(struct_value: &TestStruct, field: &str) -> Option<String> {
        struct_value
            .fields
            .get(field)
            .and_then(|value| match value {
                Value::String(s) => Some(s.clone()),
                Value::CharArray(chars) if chars.rows == 1 => {
                    let mut row = String::with_capacity(chars.cols);
                    for c in 0..chars.cols {
                        row.push(chars.data[c]);
                    }
                    Some(row.trim_end().to_string())
                }
                _ => None,
            })
    }

    fn field_bool(struct_value: &TestStruct, field: &str) -> Option<bool> {
        struct_value
            .fields
            .get(field)
            .and_then(|value| match value {
                Value::Bool(b) => Some(*b),
                _ => None,
            })
    }

    fn field_num(struct_value: &TestStruct, field: &str) -> Option<f64> {
        struct_value
            .fields
            .get(field)
            .and_then(|value| match value {
                Value::Num(n) => Some(*n),
                _ => None,
            })
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dir_lists_current_directory() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let guard = DirGuard::new();

        let dir = tempdir().expect("tempdir");
        env::set_current_dir(dir.path()).expect("switch temp dir");

        File::create("alpha.txt").expect("create file");
        fs::create_dir("beta").expect("create dir");

        let value = dir_builtin(Vec::new()).expect("dir");
        let entries = structs_from_value(value);
        let mut names: Vec<String> = entries
            .iter()
            .filter_map(|st| field_string(st, "name"))
            .collect();
        names.sort();

        assert!(names.contains(&".".to_string()));
        assert!(names.contains(&"..".to_string()));
        assert!(names.contains(&"alpha.txt".to_string()));
        assert!(names.contains(&"beta".to_string()));

        let folder_expected = dir.path().canonicalize().unwrap();
        let folder_str = folder_expected.to_string_lossy().into_owned();

        let file_entry = entries
            .iter()
            .find(|st| field_string(st, "name").as_deref() == Some("alpha.txt"))
            .expect("file entry");
        assert_eq!(field_string(file_entry, "folder").unwrap(), folder_str);
        assert_eq!(field_bool(file_entry, "isdir"), Some(false));
        assert!(field_num(file_entry, "bytes").unwrap() >= 0.0);
        assert!(field_num(file_entry, "datenum").unwrap() > 0.0);
        assert!(!field_string(file_entry, "date").unwrap().is_empty());

        let dir_entry = entries
            .iter()
            .find(|st| field_string(st, "name").as_deref() == Some("beta"))
            .expect("dir entry");
        assert_eq!(field_bool(dir_entry, "isdir"), Some(true));
        assert_eq!(field_num(dir_entry, "bytes"), Some(0.0));

        drop(guard);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dir_handles_wildcard_patterns() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let dir = tempdir().expect("tempdir");
        File::create(dir.path().join("a.m")).expect("create a.m");
        File::create(dir.path().join("b.txt")).expect("create b.txt");

        let mut entries = structs_from_value(
            dir_builtin(vec![Value::from(format!(
                "{}/*.m",
                dir.path().to_string_lossy()
            ))])
            .expect("dir pattern"),
        );
        entries.retain(|st| st.fields.contains_key("name"));
        assert_eq!(entries.len(), 1);
        assert_eq!(field_string(&entries[0], "name").as_deref(), Some("a.m"));
        assert_eq!(field_bool(&entries[0], "isdir"), Some(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dir_lists_specific_directory() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let dir = tempdir().expect("tempdir");
        File::create(dir.path().join("data.csv")).expect("create file");
        fs::create_dir(dir.path().join("nested")).expect("create dir");

        let value = dir_builtin(vec![Value::from(dir.path().to_string_lossy().to_string())])
            .expect("dir path");
        let entries = structs_from_value(value);

        let mut names: Vec<String> = entries
            .iter()
            .filter_map(|st| field_string(st, "name"))
            .collect();
        names.sort();

        assert!(names.contains(&".".to_string()));
        assert!(names.contains(&"..".to_string()));
        assert!(names.contains(&"data.csv".to_string()));
        assert!(names.contains(&"nested".to_string()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dir_folder_and_pattern_arguments() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let dir = tempdir().expect("tempdir");
        File::create(dir.path().join("keep.png")).expect("png");
        File::create(dir.path().join("skip.txt")).expect("txt");

        let folder = dir.path().to_string_lossy().to_string();
        let value = dir_builtin(vec![Value::from(folder), Value::from("*.png")]).expect("dir");
        let entries = structs_from_value(value);

        assert_eq!(entries.len(), 1);
        assert_eq!(
            field_string(&entries[0], "name").as_deref(),
            Some("keep.png")
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dir_returns_single_file_entry() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let dir = tempdir().expect("tempdir");
        let file_path = dir.path().join("solo.txt");
        File::create(&file_path).expect("create file");

        let value = dir_builtin(vec![Value::from(file_path.to_string_lossy().to_string())])
            .expect("dir file");
        let entries = structs_from_value(value);
        assert_eq!(entries.len(), 1);
        assert_eq!(
            field_string(&entries[0], "name").as_deref(),
            Some("solo.txt")
        );
        assert_eq!(field_bool(&entries[0], "isdir"), Some(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dir_accepts_char_array_input() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let dir = tempdir().expect("tempdir");
        File::create(dir.path().join("file.dat")).expect("file");

        let pattern = format!("{}/*.dat", dir.path().to_string_lossy());
        let char_array = CharArray::new_row(&pattern);
        let value = dir_builtin(vec![Value::CharArray(char_array)]).expect("dir char input");
        let entries = structs_from_value(value);
        assert_eq!(entries.len(), 1);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dir_rejects_numeric_argument() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let err = dir_builtin(vec![Value::Num(1.0)]).expect_err("expected error");
        assert_eq!(err, "dir: name must be a character vector or string scalar");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dir_rejects_multi_element_string_array() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let array = StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap();
        let err =
            dir_builtin(vec![Value::StringArray(array)]).expect_err("expected multi-string error");
        assert_eq!(err, "dir: name must be a character vector or string scalar");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dir_no_matches_returns_empty_struct_array() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let dir = tempdir().expect("tempdir");
        let pattern = format!("{}/*.nope", dir.path().to_string_lossy());
        let value = dir_builtin(vec![Value::from(pattern)]).expect("dir empty");
        match value {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 0);
                assert_eq!(cell.cols, 1);
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dir_errors_on_wildcard_folder_argument() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let err = dir_builtin(vec![Value::from("*bad"), Value::from("*.txt")])
            .expect_err("expected wildcard folder error");
        assert_eq!(
            err,
            "dir: folder input must not contain wildcard characters"
        );
    }

    #[cfg(not(windows))]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dir_expands_tilde_to_home_directory() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let dir = tempdir().expect("tempdir");
        File::create(dir.path().join("home_file.txt")).expect("home file");

        let _home_guard = HomeGuard::set_to(dir.path());

        let value = dir_builtin(vec![Value::from("~")]).expect("dir tilde");
        let entries = structs_from_value(value);
        let names: Vec<String> = entries
            .iter()
            .filter_map(|st| field_string(st, "name"))
            .collect();
        assert!(
            names.iter().any(|name| name == "home_file.txt"),
            "expected home_file.txt in listing, got {names:?}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = crate::builtins::common::test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
