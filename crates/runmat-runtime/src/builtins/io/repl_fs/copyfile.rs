//! MATLAB-compatible `copyfile` builtin for RunMat.

use runmat_filesystem as vfs;
use std::io;
use std::path::{Path, PathBuf};

use glob::PatternError;
use runmat_builtins::{CharArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::{contains_wildcards, expand_user_path};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::gather_if_needed;

const MESSAGE_ID_OS_ERROR: &str = "MATLAB:COPYFILE:OSError";
const MESSAGE_ID_SOURCE_NOT_FOUND: &str = "MATLAB:COPYFILE:FileDoesNotExist";
const MESSAGE_ID_DEST_EXISTS: &str = "MATLAB:COPYFILE:DestinationExists";
const MESSAGE_ID_DEST_MISSING: &str = "MATLAB:COPYFILE:DestinationNotFound";
const MESSAGE_ID_DEST_NOT_DIR: &str = "MATLAB:COPYFILE:DestinationNotDirectory";
const MESSAGE_ID_EMPTY_SOURCE: &str = "MATLAB:COPYFILE:EmptySource";
const MESSAGE_ID_EMPTY_DEST: &str = "MATLAB:COPYFILE:EmptyDestination";
const MESSAGE_ID_PATTERN_ERROR: &str = "MATLAB:COPYFILE:InvalidPattern";
const MESSAGE_ID_SAME_PATH: &str = "MATLAB:COPYFILE:SourceEqualsDestination";

const ERR_SOURCE_ARG: &str = "copyfile: source must be a character vector or string scalar";
const ERR_DEST_ARG: &str = "copyfile: destination must be a character vector or string scalar";
const ERR_FLAG_ARG: &str =
    "copyfile: flag must be the character 'f' supplied as a char vector or string scalar";

#[cfg(feature = "doc_export")]
#[runmat_macros::register_doc_text(name = "copyfile")]
pub const DOC_MD: &str = r#"---
title: "copyfile"
category: "io/repl_fs"
keywords: ["copyfile", "copy file", "copy folder", "filesystem", "status", "message", "messageid", "force", "overwrite"]
summary: "Copy files or folders with MATLAB-compatible status, diagnostic message, and message ID outputs."
references:
  - https://www.mathworks.com/help/matlab/ref/copyfile.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs entirely on the host CPU. When path arguments live on the GPU, RunMat gathers them to host memory before touching the filesystem."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 3
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::repl_fs::copyfile::tests"
  integration: "builtins::io::repl_fs::copyfile::tests::copyfile_copies_folder_tree"
---

# What does the `copyfile` function do in MATLAB / RunMat?
`copyfile` duplicates files and folders. It mirrors MATLAB's behavior by returning a numeric status plus optional diagnostic text instead of throwing exceptions for filesystem failures. The builtin understands wildcards, directory trees, and the optional `'f'` flag that forces overwrites of existing targets.

## How does the `copyfile` function behave in MATLAB / RunMat?
- `status = copyfile(source, destination)` copies a file or folder and returns `1` on success and `0` on failure.
- `[status, message, messageID] = copyfile(...)` adds MATLAB-style diagnostics. Successful copies return empty `1x0` character arrays.
- `copyfile(source, destination, 'f')` forces overwrites, matching MATLAB's `'f'` flag. Without the flag, existing destination files or folders prevent the copy.
- Wildcards in `source` (for example, `*.m`) copy every match. When the pattern expands to multiple items, `destination` must be an existing folder; each match is copied into that folder.
- Folder copies replicate the entire directory tree, including nested files and subfolders, honoring read-only attributes where possible.
- Inputs accept character vectors or string scalars. Other types raise MATLAB-compatible errors before any filesystem work occurs.
- Paths resolve relative to the current working directory (`pwd`), and a leading `~` expands to the user's home directory on supported platforms.

## `copyfile` Function GPU Execution Behaviour
`copyfile` performs host-side filesystem I/O. When acceleration providers are active, RunMat first gathers any GPU-resident path arguments (for example, `gpuArray("data")`) back to CPU memory, executes the copy entirely on the host, and returns host-resident outputs. Providers do not implement dedicated hooks for this builtin, so no GPU kernels are dispatched.

## GPU residency in RunMat (Do I need `gpuArray`?)
No. `copyfile` always runs on the CPU. If callers supply GPU-resident strings, RunMat automatically gathers them before touching the filesystem so existing code continues to work. Keeping paths on the GPU offers no performance benefit.

## Examples of using the `copyfile` function in MATLAB / RunMat

### Copy a file into a new name in the same folder
```matlab
fid = fopen("report.txt", "w"); fclose(fid);
status = copyfile("report.txt", "report_backup.txt");
```
Expected output:
```matlab
status =
     1
```

### Copy a file into an existing destination folder
```matlab
mkdir("archive");
fid = fopen("summary.txt", "w"); fclose(fid);
status = copyfile("summary.txt", "archive");
```
Expected output:
```matlab
status =
     1
```

### Force overwrite an existing destination file
```matlab
fid = fopen("draft.txt", "w"); fclose(fid);
fid = fopen("final.txt", "w"); fclose(fid);
[status, message, messageID] = copyfile("draft.txt", "final.txt", "f");
```
Expected output:
```matlab
status =
     1
message =

messageID =
```

### Copy an entire folder tree
```matlab
mkdir("data/raw");
fid = fopen(fullfile("data", "raw", "sample.dat"), "w"); fclose(fid);
status = copyfile("data", "data_copy");
```
Expected output:
```matlab
status =
     1
```

### Copy multiple files with a wildcard pattern
```matlab
fid = fopen("a.log", "w"); fclose(fid);
fid = fopen("b.log", "w"); fclose(fid);
mkdir("logs");
status = copyfile("*.log", "logs");
```
Expected output:
```matlab
status =
     1
```

### Handle missing sources gracefully
```matlab
[status, message, messageID] = copyfile("missing.txt", "dest.txt");
```
Expected output:
```matlab
status =
     0
message =
Source "missing.txt" does not exist.
messageID =
MATLAB:COPYFILE:FileDoesNotExist
```

## FAQ
- **What status codes does `copyfile` return?** `1` on success, `0` on failure. The numeric output is a double scalar to match MATLAB.
- **Does `copyfile` throw exceptions?** Only invalid inputs raise immediate errors. Filesystem failures surface through `status`, `message`, and `messageID`.
- **How do I overwrite existing files?** Pass `'f'` as the third argument. Without it, existing destinations prevent the copy.
- **Can I copy folders recursively?** Yes. When `source` refers to a folder, `copyfile` copies its entire contents (including subfolders) into `destination`.
- **What happens when the source and destination are the same?** The builtin reports a failure with `messageID = MATLAB:COPYFILE:SourceEqualsDestination`, mirroring MATLAB's diagnostics.
- **Does `copyfile` preserve timestamps and permissions?** It delegates to the host operating system. RunMat attempts to preserve read-only attributes where possible, but platform differences may leave some metadata unchanged.
- **Can `copyfile` create intermediate destination folders?** `copyfile` does not invent new parent folders. Ensure the destination's parent path exists before calling the builtin.
- **Will GPU acceleration speed up `copyfile`?** No. The builtin operates entirely on the host CPU. GPU-resident strings are gathered automatically so existing scripts remain compatible.
- **Are wildcards supported on all platforms?** Yes. `copyfile("*.txt", "dest")` expands using MATLAB-compatible glob semantics on every supported operating system.

## See Also
[movefile](./movefile), [mkdir](./mkdir), [rmdir](./rmdir), [dir](./dir), [pwd](./pwd)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/io/repl_fs/copyfile.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/copyfile.rs)
- Found a bug? [Open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "copyfile",
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
        "Host-only filesystem operation. GPU-resident path and flag arguments are gathered automatically before performing the copy.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "copyfile",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Filesystem side effects materialise immediately; metadata is registered for completeness.",
};

#[runtime_builtin(
    name = "copyfile",
    category = "io/repl_fs",
    summary = "Copy files or folders with MATLAB-compatible status, diagnostic message, and message ID outputs.",
    keywords = "copyfile,copy file,copy folder,filesystem,status,message,messageid,force,overwrite",
    accel = "cpu"
)]
fn copyfile_builtin(args: Vec<Value>) -> Result<Value, String> {
    let eval = evaluate(&args)?;
    Ok(eval.first_output())
}

/// Evaluate `copyfile` once and expose all outputs.
pub fn evaluate(args: &[Value]) -> Result<CopyfileResult, String> {
    let gathered = gather_arguments(args)?;
    match gathered.len() {
        0 | 1 => Err("copyfile: not enough input arguments".to_string()),
        2 => copy_operation(&gathered[0], &gathered[1], false),
        3 => {
            let force = parse_force_flag(&gathered[2])?;
            copy_operation(&gathered[0], &gathered[1], force)
        }
        _ => Err("copyfile: too many input arguments".to_string()),
    }
}

#[derive(Debug, Clone)]
pub struct CopyfileResult {
    status: f64,
    message: String,
    message_id: String,
}

impl CopyfileResult {
    fn success() -> Self {
        Self {
            status: 1.0,
            message: String::new(),
            message_id: String::new(),
        }
    }

    fn failure(message: String, message_id: &str) -> Self {
        Self {
            status: 0.0,
            message,
            message_id: message_id.to_string(),
        }
    }

    fn empty_source() -> Self {
        Self::failure(
            "Source file or folder name must not be empty.".to_string(),
            MESSAGE_ID_EMPTY_SOURCE,
        )
    }

    fn empty_destination() -> Self {
        Self::failure(
            "Destination file or folder name must not be empty.".to_string(),
            MESSAGE_ID_EMPTY_DEST,
        )
    }

    fn source_not_found(display: &str) -> Self {
        Self::failure(
            format!("Source \"{}\" does not exist.", display),
            MESSAGE_ID_SOURCE_NOT_FOUND,
        )
    }

    fn destination_exists(display: &str) -> Self {
        Self::failure(
            format!(
                "Cannot copy to \"{}\": destination already exists.",
                display
            ),
            MESSAGE_ID_DEST_EXISTS,
        )
    }

    fn destination_missing(display: &str) -> Self {
        Self::failure(
            format!(
                "Destination folder \"{}\" must exist when copying multiple sources.",
                display
            ),
            MESSAGE_ID_DEST_MISSING,
        )
    }

    fn destination_not_directory(display: &str) -> Self {
        Self::failure(
            format!("Destination \"{}\" must refer to a folder.", display),
            MESSAGE_ID_DEST_NOT_DIR,
        )
    }

    fn same_path(display: &str) -> Self {
        Self::failure(
            format!("Cannot copy \"{}\" onto itself.", display),
            MESSAGE_ID_SAME_PATH,
        )
    }

    fn glob_pattern_error(pattern: &str, err: &str) -> Self {
        Self::failure(
            format!("Invalid source pattern \"{}\": {}", pattern, err),
            MESSAGE_ID_PATTERN_ERROR,
        )
    }

    fn os_error(source: &str, target: &str, err: &io::Error) -> Self {
        Self::failure(
            format!("Unable to copy \"{}\" to \"{}\": {}", source, target, err),
            MESSAGE_ID_OS_ERROR,
        )
    }

    pub fn first_output(&self) -> Value {
        Value::Num(self.status)
    }

    pub fn outputs(&self) -> Vec<Value> {
        vec![
            Value::Num(self.status),
            char_array_value(&self.message),
            char_array_value(&self.message_id),
        ]
    }

    #[cfg(test)]
    pub(crate) fn status(&self) -> f64 {
        self.status
    }

    #[cfg(test)]
    pub(crate) fn message(&self) -> &str {
        &self.message
    }

    #[cfg(test)]
    pub(crate) fn message_id(&self) -> &str {
        &self.message_id
    }
}

fn copy_operation(
    source: &Value,
    destination: &Value,
    force: bool,
) -> Result<CopyfileResult, String> {
    let source_raw = extract_path(source, ERR_SOURCE_ARG)?;
    if source_raw.is_empty() {
        return Ok(CopyfileResult::empty_source());
    }

    let destination_raw = extract_path(destination, ERR_DEST_ARG)?;
    if destination_raw.is_empty() {
        return Ok(CopyfileResult::empty_destination());
    }

    let source_expanded = expand_user_path(&source_raw, "copyfile")?;
    let destination_expanded = expand_user_path(&destination_raw, "copyfile")?;

    if contains_wildcards(&source_expanded) {
        Ok(copy_with_pattern(
            &source_expanded,
            &destination_expanded,
            force,
        ))
    } else {
        Ok(copy_single_source(
            &source_expanded,
            &destination_expanded,
            force,
        ))
    }
}

fn copy_single_source(source: &str, destination: &str, force: bool) -> CopyfileResult {
    let source_path = PathBuf::from(source);
    let source_meta = match vfs::metadata(&source_path) {
        Ok(meta) => meta,
        Err(_) => return CopyfileResult::source_not_found(source),
    };
    let source_display = path_to_display(&source_path);

    let destination_path = PathBuf::from(destination);
    if same_physical_path(&source_path, &destination_path) {
        return CopyfileResult::same_path(&source_display);
    }

    let destination_meta = vfs::metadata(&destination_path).ok();
    let mut target_path = destination_path.clone();
    let mut remove_target = false;
    let mut remove_is_dir = false;

    if let Some(meta) = &destination_meta {
        if meta.is_dir() {
            let Some(name) = source_path.file_name() else {
                return CopyfileResult::os_error(
                    source,
                    destination,
                    &io::Error::other("Cannot determine source file name"),
                );
            };
            target_path = destination_path.join(name);
            if same_physical_path(&source_path, &target_path) {
                return CopyfileResult::same_path(&source_display);
            }
            if source_meta.is_dir() && is_descendant(&source_path, &target_path) {
                return CopyfileResult::failure(
                    "Cannot copy a folder into one of its descendants.".to_string(),
                    MESSAGE_ID_OS_ERROR,
                );
            }
            match vfs::metadata(&target_path) {
                Ok(existing) => {
                    if !force {
                        return CopyfileResult::destination_exists(&path_to_display(&target_path));
                    }
                    remove_target = true;
                    remove_is_dir = existing.is_dir();
                }
                Err(err) => {
                    if err.kind() != io::ErrorKind::NotFound {
                        return CopyfileResult::os_error(
                            source,
                            &path_to_display(&target_path),
                            &err,
                        );
                    }
                }
            }
        } else {
            if source_meta.is_dir() {
                return CopyfileResult::destination_not_directory(destination);
            }
            if !force {
                return CopyfileResult::destination_exists(destination);
            }
            remove_target = true;
            remove_is_dir = false;
        }
    } else if source_meta.is_dir() {
        // When copying a directory to a new path, ensure we don't copy into a child of itself.
        if is_descendant(&source_path, &destination_path) {
            return CopyfileResult::failure(
                "Cannot copy a folder into one of its descendants.".to_string(),
                MESSAGE_ID_OS_ERROR,
            );
        }
    }

    let target_display = path_to_display(&target_path);
    let plan = vec![CopyPlanEntry::new(
        source_path,
        source_display.clone(),
        target_path,
        target_display,
        source_meta.is_dir(),
        remove_target,
        remove_is_dir,
    )];

    match execute_plan(&plan) {
        Ok(()) => CopyfileResult::success(),
        Err(err) => CopyfileResult::os_error(&err.source_display, &err.target_display, &err.error),
    }
}

fn copy_with_pattern(pattern: &str, destination: &str, force: bool) -> CopyfileResult {
    let paths = match glob::glob(pattern) {
        Ok(iter) => iter,
        Err(PatternError { msg, .. }) => return CopyfileResult::glob_pattern_error(pattern, msg),
    };

    let mut matches = Vec::new();
    for entry in paths {
        match entry {
            Ok(path) => matches.push(path),
            Err(err) => {
                let path_display = path_to_display(err.path());
                return CopyfileResult::os_error(&path_display, destination, err.error());
            }
        }
    }

    if matches.is_empty() {
        return CopyfileResult::source_not_found(pattern);
    }

    let destination_path = PathBuf::from(destination);
    let destination_meta = match vfs::metadata(&destination_path) {
        Ok(meta) => meta,
        Err(_) => return CopyfileResult::destination_missing(destination),
    };

    if !destination_meta.is_dir() {
        return CopyfileResult::destination_not_directory(destination);
    }

    let mut plan = Vec::with_capacity(matches.len());
    for source_path in matches {
        let display_source = path_to_display(&source_path);
        let meta = match vfs::metadata(&source_path) {
            Ok(meta) => meta,
            Err(_) => return CopyfileResult::source_not_found(&display_source),
        };
        let Some(name) = source_path.file_name() else {
            return CopyfileResult::os_error(
                &display_source,
                destination,
                &io::Error::other("Cannot determine source name"),
            );
        };
        let target_path = destination_path.join(name);
        if same_physical_path(&source_path, &target_path) {
            return CopyfileResult::same_path(&display_source);
        }
        let target_display = path_to_display(&target_path);
        match vfs::metadata(&target_path) {
            Ok(existing) => {
                if !force {
                    return CopyfileResult::destination_exists(&target_display);
                }
                plan.push(CopyPlanEntry::new(
                    source_path.clone(),
                    display_source.clone(),
                    target_path.clone(),
                    target_display,
                    meta.is_dir(),
                    true,
                    existing.is_dir(),
                ));
            }
            Err(err) => {
                if err.kind() != io::ErrorKind::NotFound {
                    return CopyfileResult::os_error(&display_source, &target_display, &err);
                }
                plan.push(CopyPlanEntry::new(
                    source_path.clone(),
                    display_source,
                    target_path.clone(),
                    target_display,
                    meta.is_dir(),
                    false,
                    false,
                ));
            }
        }
    }

    if plan.is_empty() {
        return CopyfileResult::success();
    }

    match execute_plan(&plan) {
        Ok(()) => CopyfileResult::success(),
        Err(err) => CopyfileResult::os_error(&err.source_display, &err.target_display, &err.error),
    }
}

#[derive(Debug, Clone)]
struct CopyPlanEntry {
    source_path: PathBuf,
    source_display: String,
    target_path: PathBuf,
    target_display: String,
    source_is_dir: bool,
    remove_target: bool,
    remove_is_dir: bool,
}

impl CopyPlanEntry {
    fn new(
        source_path: PathBuf,
        source_display: String,
        target_path: PathBuf,
        target_display: String,
        source_is_dir: bool,
        remove_target: bool,
        remove_is_dir: bool,
    ) -> Self {
        Self {
            source_path,
            source_display,
            target_path,
            target_display,
            source_is_dir,
            remove_target,
            remove_is_dir,
        }
    }
}

struct CopyError {
    source_display: String,
    target_display: String,
    error: io::Error,
}

fn execute_plan(plan: &[CopyPlanEntry]) -> Result<(), CopyError> {
    for entry in plan {
        if entry.remove_target {
            let result = if entry.remove_is_dir {
                vfs::remove_dir_all(&entry.target_path)
            } else {
                vfs::remove_file(&entry.target_path)
            };
            if let Err(err) = result {
                if err.kind() != io::ErrorKind::NotFound {
                    return Err(CopyError {
                        source_display: entry.source_display.clone(),
                        target_display: entry.target_display.clone(),
                        error: err,
                    });
                }
            }
        }

        if let Err(err) = copy_path(&entry.source_path, &entry.target_path, entry.source_is_dir) {
            return Err(CopyError {
                source_display: entry.source_display.clone(),
                target_display: entry.target_display.clone(),
                error: err,
            });
        }
    }

    Ok(())
}

fn copy_path(source: &Path, destination: &Path, is_directory: bool) -> io::Result<()> {
    if is_directory {
        copy_directory_recursive(source, destination)
    } else {
        copy_file_to_path(source, destination)
    }
}

fn copy_directory_recursive(source: &Path, destination: &Path) -> io::Result<()> {
    ensure_parent_exists(destination)?;

    match vfs::metadata(destination) {
        Ok(meta) => {
            if !meta.is_dir() {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "Destination exists and is not a directory",
                ));
            }
        }
        Err(err) => {
            if err.kind() != io::ErrorKind::NotFound {
                return Err(err);
            }
            vfs::create_dir(destination)?;
        }
    }

    if let Ok(metadata) = vfs::metadata(source) {
        let _ = vfs::set_readonly(destination, metadata.is_readonly());
    }

    for entry in vfs::read_dir(source)? {
        let child_source = entry.path().to_path_buf();
        let child_dest = destination.join(PathBuf::from(entry.file_name()));
        let child_meta = vfs::metadata(&child_source)?;
        if child_meta.is_dir() {
            copy_directory_recursive(&child_source, &child_dest)?;
        } else {
            copy_file_to_path(&child_source, &child_dest)?;
        }
    }

    Ok(())
}

fn copy_file_to_path(source: &Path, destination: &Path) -> io::Result<()> {
    ensure_parent_exists(destination)?;

    vfs::copy_file(source, destination)?;

    if let Ok(metadata) = vfs::metadata(source) {
        let _ = vfs::set_readonly(destination, metadata.is_readonly());
    }

    Ok(())
}

fn ensure_parent_exists(path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if vfs::metadata(parent).is_err() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Destination parent \"{}\" does not exist", parent.display()),
            ));
        }
    }
    Ok(())
}

fn same_physical_path(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    match (vfs::canonicalize(a), vfs::canonicalize(b)) {
        (Ok(ca), Ok(cb)) => ca == cb,
        _ => false,
    }
}

fn is_descendant(parent: &Path, candidate: &Path) -> bool {
    if candidate.starts_with(parent) && candidate != parent {
        return true;
    }
    match (vfs::canonicalize(parent), vfs::canonicalize(candidate)) {
        (Ok(parent_canon), Ok(candidate_canon)) => {
            candidate_canon.starts_with(&parent_canon) && candidate_canon != parent_canon
        }
        _ => false,
    }
}

fn parse_force_flag(value: &Value) -> Result<bool, String> {
    let text = extract_path(value, ERR_FLAG_ARG)?;
    if text.eq_ignore_ascii_case("f") {
        Ok(true)
    } else {
        Err(ERR_FLAG_ARG.to_string())
    }
}

fn extract_path(value: &Value, error_message: &str) -> Result<String, String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(array) => {
            if array.rows == 1 {
                Ok(array.data.iter().collect())
            } else {
                Err(error_message.to_string())
            }
        }
        Value::StringArray(array) => {
            if array.data.len() == 1 {
                Ok(array.data[0].clone())
            } else {
                Err(error_message.to_string())
            }
        }
        _ => Err(error_message.to_string()),
    }
}

fn gather_arguments(args: &[Value]) -> Result<Vec<Value>, String> {
    let mut out = Vec::with_capacity(args.len());
    for value in args {
        out.push(gather_if_needed(value).map_err(|err| format!("copyfile: {err}"))?);
    }
    Ok(out)
}

fn char_array_value(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}

fn path_to_display(path: &Path) -> String {
    path.display().to_string()
}

#[cfg(test)]
mod tests {
    use super::super::REPL_FS_TEST_LOCK;
    use super::*;
    use std::fs::{self, File};
    use tempfile::tempdir;

    #[test]
    fn copyfile_copies_file_to_new_name() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let source = temp.path().join("source.txt");
        let dest = temp.path().join("dest.txt");
        File::create(&source).expect("create source");

        let eval = evaluate(&[
            Value::from(source.to_string_lossy().to_string()),
            Value::from(dest.to_string_lossy().to_string()),
        ])
        .expect("copyfile");
        assert_eq!(eval.status(), 1.0);
        assert!(source.exists());
        assert!(dest.exists());
    }

    #[test]
    fn copyfile_copies_into_existing_directory() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let source = temp.path().join("report.txt");
        let dest_dir = temp.path().join("reports");
        fs::create_dir(&dest_dir).expect("create dest dir");
        File::create(&source).expect("create source");

        let eval = evaluate(&[
            Value::from(source.to_string_lossy().to_string()),
            Value::from(dest_dir.to_string_lossy().to_string()),
        ])
        .expect("copyfile");
        assert_eq!(eval.status(), 1.0);
        assert!(source.exists());
        assert!(dest_dir.join("report.txt").exists());
    }

    #[test]
    fn copyfile_force_overwrites_existing_file() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let source = temp.path().join("draft.txt");
        let dest = temp.path().join("final.txt");
        File::create(&source).expect("create source");
        File::create(&dest).expect("create dest");

        let eval = evaluate(&[
            Value::from(source.to_string_lossy().to_string()),
            Value::from(dest.to_string_lossy().to_string()),
            Value::from("f"),
        ])
        .expect("copyfile");
        assert_eq!(eval.status(), 1.0);
        assert!(source.exists());
        assert!(dest.exists());
    }

    #[test]
    fn copyfile_without_force_fails_when_destination_exists() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let source = temp.path().join("draft.txt");
        let dest = temp.path().join("final.txt");
        File::create(&source).expect("create source");
        File::create(&dest).expect("create dest");

        let eval = evaluate(&[
            Value::from(source.to_string_lossy().to_string()),
            Value::from(dest.to_string_lossy().to_string()),
        ])
        .expect("copyfile");
        assert_eq!(eval.status(), 0.0);
        assert_eq!(eval.message_id(), MESSAGE_ID_DEST_EXISTS);
        assert!(
            eval.message().contains("destination already exists"),
            "expected descriptive destination-exists message"
        );
        assert!(source.exists());
        assert!(dest.exists());
    }

    #[test]
    fn copyfile_copies_folder_tree() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let source_dir = temp.path().join("data");
        let nested = source_dir.join("raw");
        fs::create_dir(&source_dir).expect("create source dir");
        fs::create_dir(&nested).expect("create nested dir");
        let file_path = nested.join("sample.txt");
        File::create(&file_path).expect("create sample");
        let dest_dir = temp.path().join("data_copy");

        let eval = evaluate(&[
            Value::from(source_dir.to_string_lossy().to_string()),
            Value::from(dest_dir.to_string_lossy().to_string()),
        ])
        .expect("copyfile");

        assert_eq!(eval.status(), 1.0);
        assert!(dest_dir.join("raw").exists());
        assert!(dest_dir.join("raw").join("sample.txt").exists());
        assert!(file_path.exists());
    }

    #[test]
    fn copyfile_pattern_requires_existing_directory() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let file = temp.path().join("file.log");
        File::create(&file).expect("create file");
        let pattern = temp.path().join("*.log");
        let dest = temp.path().join("missing");

        let eval = evaluate(&[
            Value::from(pattern.to_string_lossy().to_string()),
            Value::from(dest.to_string_lossy().to_string()),
        ])
        .expect("copyfile");
        assert_eq!(eval.status(), 0.0);
        assert_eq!(eval.message_id(), MESSAGE_ID_DEST_MISSING);
    }

    #[test]
    fn copyfile_pattern_copies_multiple_files() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let alpha = temp.path().join("alpha.log");
        let beta = temp.path().join("beta.log");
        File::create(&alpha).expect("create alpha");
        File::create(&beta).expect("create beta");
        let dest_dir = temp.path().join("logs");
        fs::create_dir(&dest_dir).expect("create dest dir");
        let pattern = temp.path().join("*.log");

        let eval = evaluate(&[
            Value::from(pattern.to_string_lossy().to_string()),
            Value::from(dest_dir.to_string_lossy().to_string()),
        ])
        .expect("copyfile");
        assert_eq!(eval.status(), 1.0);
        assert!(dest_dir.join("alpha.log").exists());
        assert!(dest_dir.join("beta.log").exists());
    }

    #[test]
    fn copyfile_pattern_copies_all_matches() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let dest_dir = temp.path().join("logs");
        fs::create_dir(&dest_dir).expect("create dest dir");
        let file_a = temp.path().join("a.log");
        let file_b = temp.path().join("b.log");
        File::create(&file_a).expect("create a");
        File::create(&file_b).expect("create b");
        let pattern = temp.path().join("*.log");

        let eval = evaluate(&[
            Value::from(pattern.to_string_lossy().to_string()),
            Value::from(dest_dir.to_string_lossy().to_string()),
        ])
        .expect("copyfile");
        assert_eq!(eval.status(), 1.0);
        assert!(dest_dir.join("a.log").exists());
        assert!(dest_dir.join("b.log").exists());
    }

    #[test]
    fn copyfile_reports_missing_source() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let source = temp.path().join("missing.txt");
        let dest = temp.path().join("dest.txt");

        let eval = evaluate(&[
            Value::from(source.to_string_lossy().to_string()),
            Value::from(dest.to_string_lossy().to_string()),
        ])
        .expect("copyfile");
        assert_eq!(eval.status(), 0.0);
        assert_eq!(eval.message_id(), MESSAGE_ID_SOURCE_NOT_FOUND);
    }

    #[test]
    fn copyfile_outputs_char_arrays() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let source = temp.path().join("source.txt");
        let dest = temp.path().join("dest.txt");
        File::create(&source).expect("create source");

        let eval = evaluate(&[
            Value::from(source.to_string_lossy().to_string()),
            Value::from(dest.to_string_lossy().to_string()),
        ])
        .expect("copyfile");
        let outputs = eval.outputs();
        assert_eq!(outputs.len(), 3);
        assert!(matches!(outputs[0], Value::Num(1.0)));
        assert!(matches!(outputs[1], Value::CharArray(ref ca) if ca.cols == 0));
        assert!(matches!(outputs[2], Value::CharArray(ref ca) if ca.cols == 0));
    }

    #[test]
    fn copyfile_rejects_invalid_flag() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let err = evaluate(&[Value::from("a"), Value::from("b"), Value::Num(1.0)])
            .expect_err("expected error");
        assert_eq!(err, ERR_FLAG_ARG);
    }

    #[test]
    fn copyfile_force_flag_accepts_uppercase_char_array() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let source = temp.path().join("draft.txt");
        let dest = temp.path().join("final.txt");
        File::create(&source).expect("create source");
        File::create(&dest).expect("create dest");

        let eval = evaluate(&[
            Value::from(source.to_string_lossy().to_string()),
            Value::from(dest.to_string_lossy().to_string()),
            Value::CharArray(CharArray::new_row("F")),
        ])
        .expect("copyfile");
        assert_eq!(eval.status(), 1.0);
    }

    #[test]
    fn copyfile_force_flag_accepts_uppercase_string() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let source = temp.path().join("draft.txt");
        let dest = temp.path().join("final.txt");
        File::create(&source).expect("create source");
        File::create(&dest).expect("create dest");

        let eval = evaluate(&[
            Value::from(source.to_string_lossy().to_string()),
            Value::from(dest.to_string_lossy().to_string()),
            Value::from("F"),
        ])
        .expect("copyfile");
        assert_eq!(eval.status(), 1.0);
    }

    #[test]
    fn copyfile_same_path_fails() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let source = temp.path().join("note.txt");
        File::create(&source).expect("create source");

        let eval = evaluate(&[
            Value::from(source.to_string_lossy().to_string()),
            Value::from(source.to_string_lossy().to_string()),
        ])
        .expect("copyfile");
        assert_eq!(eval.status(), 0.0);
        assert_eq!(eval.message_id(), MESSAGE_ID_SAME_PATH);
    }

    #[test]
    fn copyfile_reports_empty_source() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let eval = evaluate(&[Value::from(""), Value::from("dest.txt")]).expect("copyfile");
        assert_eq!(eval.status(), 0.0);
        assert_eq!(eval.message_id(), MESSAGE_ID_EMPTY_SOURCE);
    }

    #[test]
    fn copyfile_reports_empty_destination() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let eval = evaluate(&[Value::from("source.txt"), Value::from("")]).expect("copyfile");
        assert_eq!(eval.status(), 0.0);
        assert_eq!(eval.message_id(), MESSAGE_ID_EMPTY_DEST);
    }

    #[test]
    fn copyfile_reports_invalid_pattern() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let eval = evaluate(&[Value::from("[*.txt"), Value::from("dest")]).expect("copyfile");
        assert_eq!(eval.status(), 0.0);
        assert_eq!(eval.message_id(), MESSAGE_ID_PATTERN_ERROR);
    }

    #[test]
    fn doc_examples_present() {
        let blocks = crate::builtins::common::test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
