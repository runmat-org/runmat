//! MATLAB-compatible `mkdir` builtin for RunMat.

use std::path::{Path, PathBuf};

use runmat_builtins::{CharArray, Value};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{gather_if_needed, build_runtime_error, BuiltinResult, RuntimeError};

const MESSAGE_ID_OS_ERROR: &str = "MATLAB:MKDIR:OSError";
const MESSAGE_ID_DIRECTORY_EXISTS: &str = "MATLAB:MKDIR:DirectoryExists";
const MESSAGE_ID_INVALID_PARENT: &str = "MATLAB:MKDIR:ParentDirectoryDoesNotExist";
const MESSAGE_ID_NOT_A_DIRECTORY: &str = "MATLAB:MKDIR:ParentIsNotDirectory";
const MESSAGE_ID_EMPTY_NAME: &str = "MATLAB:MKDIR:InvalidFolderName";

const ERR_FOLDER_ARG: &str = "mkdir: folder name must be a character vector or string scalar";
const ERR_PARENT_ARG: &str = "mkdir: parent folder must be a character vector or string scalar";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "mkdir",
        builtin_path = "crate::builtins::io::repl_fs::mkdir"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "mkdir"
category: "io/repl_fs"
keywords: ["mkdir", "create directory", "folder", "filesystem", "status", "message", "messageid"]
summary: "Create folders with MATLAB-compatible status, message, and message ID outputs."
references:
  - https://www.mathworks.com/help/matlab/ref/mkdir.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs entirely on the CPU. When path inputs reside on the GPU, RunMat gathers them to host memory before creating directories."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::repl_fs::mkdir::tests"
  integration: "builtins::io::repl_fs::mkdir::tests::mkdir_captures_failure_status_and_message"
---

# What does the `mkdir` function do in MATLAB / RunMat?
`mkdir` creates new folders on disk. It mirrors MATLAB by accepting either a single path string or a parent-folder plus child-name pair, and it returns diagnostic status information instead of throwing errors for most filesystem failures.

## How does the `mkdir` function behave in MATLAB / RunMat?
- `status = mkdir(folder)` creates `folder`, returning `1` when the directory is created or already exists, and `0` on failure. Status outputs are `double` scalars for MATLAB compatibility.
- `[status, message, messageID] = mkdir(parent, child)` creates the directory `fullfile(parent, child)`. When the directory is created during this call, the message outputs are empty. If the directory already exists, MATLAB populates them with `'Directory already exists.'` and `'MATLAB:MKDIR:DirectoryExists'`. Failures return the system error text and message identifier.
- The optional `message` and `messageID` outputs are character arrays (with size `1×0` when empty), matching MATLAB’s behaviour. Callers that prefer string scalars can wrap them with `string(message)`.
- Passing a parent folder that does not exist leaves the filesystem unchanged and returns `status = 0`.
- If the target path already exists as a file, `mkdir` fails gracefully with `status = 0` and a diagnostic message; it does not overwrite files.
- Path arguments accept character vectors or string scalars. Other input types raise `mkdir: folder name must be a character vector or string scalar` (or the equivalent message for the parent argument).
- The builtin expands `~` to the user’s home directory and honours relative paths with respect to the current working folder (`pwd`).

## `mkdir` Function GPU Execution Behaviour
`mkdir` performs host-side filesystem operations. When callers supply GPU-resident scalars (for example, `gpuArray("logs")`), RunMat gathers the value back to the CPU before resolving the path. Acceleration providers do not publish hooks for this builtin, so there is no device-side implementation to enable.

## GPU residency in RunMat (Do I need `gpuArray`?)
No. `mkdir` executes entirely on the host—GPU residency provides no benefit. However, if a script accidentally stores path strings on the GPU, RunMat automatically gathers them before accessing the filesystem so the call still succeeds.

## Examples of using the `mkdir` function in MATLAB / RunMat

### Create A New Folder In The Current Directory
```matlab
status = mkdir("results");
```
Expected output:
```matlab
status =
     1
```

### Create A Nested Folder Using Parent And Child Arguments
```matlab
status = mkdir("data", "archive/2024");
```
Expected output:
```matlab
status =
     1
```

### Detect When A Folder Already Exists
```matlab
mkdir("logs");
[status, message, messageID] = mkdir("logs");
```
Expected output:
```matlab
status =
     1

message =
Directory already exists.

messageID =
MATLAB:MKDIR:DirectoryExists
```

### Handle Missing Parent Folder Gracefully
```matlab
[status, message, messageID] = mkdir("missing-parent", "child");
```
Expected output:
```matlab
status =
     0
message =
Parent folder "missing-parent" does not exist.
messageID =
MATLAB:MKDIR:ParentDirectoryDoesNotExist
```

### Capture Detailed Status And Messages
```matlab
[status, message, messageID] = mkdir("reports");
```
Expected output:
```matlab
status =
     1
message =

messageID =

```

### Create A Folder In Your Home Directory With Tilde Expansion
```matlab
status = mkdir("~", "RunMatProjects");
```
Expected output:
```matlab
status =
     1
```

### Use gpuArray Inputs For Paths
```matlab
status = mkdir(gpuArray("gpu-output"));
```
Expected output:
```matlab
status =
     1
```

## FAQ
- **What status codes does `mkdir` return?** `1` indicates success (the directory exists afterwards), and `0` indicates failure. Status values are doubles to match MATLAB.
- **Does `mkdir` overwrite existing files?** No. If the target path already exists as a regular file, `mkdir` returns `status = 0` and reports that the path is not a directory.
- **Can I create multiple levels of folders at once?** Yes when you provide a single path, because RunMat mirrors MATLAB’s behaviour of creating intermediate directories. When using the two-argument form, the parent folder must already exist.
- **Does `mkdir` support string scalars and character vectors?** Yes. String arrays must contain exactly one element; other types raise an error.
- **How are error messages returned?** Failures return descriptive messages and MATLAB-style message IDs (for example, `MATLAB:MKDIR:OSError`) in the second and third outputs. The builtin does not throw unless the inputs are invalid.
- **Are UNC paths and drive-letter paths supported on Windows?** Yes. Provide the path exactly as you would in MATLAB; RunMat forwards it to the operating system.
- **Can I run `mkdir` on the GPU?** No. The function operates on the host, but it automatically gathers GPU-resident inputs for convenience.
- **What happens if the folder already exists?** `mkdir` reports success (`status = 1`) and leaves the directory untouched, just like MATLAB.
- **Does tilde (`~`) expand to the home directory?** Yes. Both single-argument and two-argument forms expand `~` at the start of a path.
- **How do I handle errors programmatically?** Capture the optional outputs and test `status`. When it is `0`, inspect `message` and `messageID` for diagnostics.

## See Also
[cd](./cd), [pwd](./pwd), [ls](./ls), [dir](./dir)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/io/repl_fs/mkdir.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/mkdir.rs)
- Issues: [Open a GitHub ticket](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::mkdir")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "mkdir",
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
        "Host-only filesystem builtin. GPU-resident path arguments are gathered automatically before directory creation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::mkdir")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "mkdir",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Filesystem side-effects terminate fusion; metadata registered for completeness.",
};

const BUILTIN_NAME: &str = "mkdir";

fn mkdir_error(message: impl Into<String>) -> RuntimeError {
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
    name = "mkdir",
    category = "io/repl_fs",
    summary = "Create folders with MATLAB-compatible status, message, and message ID outputs.",
    keywords = "mkdir,create directory,folder,filesystem,status,message,messageid",
    accel = "cpu",
    suppress_auto_output = true,
    builtin_path = "crate::builtins::io::repl_fs::mkdir"
)]
fn mkdir_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(&args)?;
    Ok(eval.first_output())
}

/// Evaluate `mkdir` once and expose all outputs.
pub fn evaluate(args: &[Value]) -> BuiltinResult<MkdirResult> {
    let gathered = gather_arguments(args)?;
    match gathered.len() {
        0 => Err(mkdir_error("mkdir: not enough input arguments")),
        1 => create_from_single(&gathered[0]),
        2 => create_from_parent_child(&gathered[0], &gathered[1]),
        _ => Err(mkdir_error("mkdir: too many input arguments")),
    }
}

#[derive(Debug, Clone)]
pub struct MkdirResult {
    status: f64,
    message: String,
    message_id: String,
}

impl MkdirResult {
    fn success() -> Self {
        Self {
            status: 1.0,
            message: String::new(),
            message_id: String::new(),
        }
    }

    fn already_exists() -> Self {
        Self {
            status: 1.0,
            message: "Directory already exists.".to_string(),
            message_id: MESSAGE_ID_DIRECTORY_EXISTS.to_string(),
        }
    }

    fn failure(message: String, message_id: &str) -> Self {
        Self {
            status: 0.0,
            message,
            message_id: message_id.to_string(),
        }
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

fn create_from_single(value: &Value) -> BuiltinResult<MkdirResult> {
    let raw = extract_folder_name(value, ERR_FOLDER_ARG)?;
    if raw.is_empty() {
        return Ok(MkdirResult::failure(
            "Folder name must not be empty.".to_string(),
            MESSAGE_ID_EMPTY_NAME,
        ));
    }
    let expanded = expand_user_path(&raw, "mkdir").map_err(mkdir_error)?;
    let path = PathBuf::from(expanded);
    Ok(create_directory(&path))
}

fn create_from_parent_child(parent: &Value, child: &Value) -> BuiltinResult<MkdirResult> {
    let parent_raw = extract_folder_name(parent, ERR_PARENT_ARG)?;
    let child_raw = extract_folder_name(child, ERR_FOLDER_ARG)?;

    if child_raw.is_empty() {
        return Ok(MkdirResult::failure(
            "Folder name must not be empty.".to_string(),
            MESSAGE_ID_EMPTY_NAME,
        ));
    }

    let parent_expanded = if parent_raw.is_empty() {
        None
    } else {
        Some(expand_user_path(&parent_raw, "mkdir").map_err(mkdir_error)?)
    };
    let child_expanded = expand_user_path(&child_raw, "mkdir").map_err(mkdir_error)?;
    let child_path = PathBuf::from(&child_expanded);

    if child_path.is_absolute() {
        return Ok(create_directory(&child_path));
    }

    if let Some(parent_text) = parent_expanded {
        let parent_path = PathBuf::from(&parent_text);
        if !path_exists(&parent_path) {
            let message = format!("Parent folder \"{}\" does not exist.", parent_text);
            return Ok(MkdirResult::failure(message, MESSAGE_ID_INVALID_PARENT));
        }
        if !path_is_existing_directory(&parent_path) {
            let message = format!("Parent folder \"{}\" is not a directory.", parent_text);
            return Ok(MkdirResult::failure(message, MESSAGE_ID_NOT_A_DIRECTORY));
        }
        let target = parent_path.join(&child_expanded);
        return Ok(create_directory(&target));
    }

    Ok(create_directory(&PathBuf::from(&child_expanded)))
}

fn create_directory(path: &Path) -> MkdirResult {
    let display = path.display().to_string();
    if path_exists(path) {
        if path_is_existing_directory(path) {
            return MkdirResult::already_exists();
        }
        return MkdirResult::failure(
            format!(
                "Cannot create folder \"{}\". A file or non-directory item with the same name already exists.",
                display
            ),
            MESSAGE_ID_NOT_A_DIRECTORY,
        );
    }

    match vfs::create_dir_all(path) {
        Ok(_) => MkdirResult::success(),
        Err(err) => MkdirResult::failure(
            format!("Unable to create folder \"{}\": {}", display, err),
            MESSAGE_ID_OS_ERROR,
        ),
    }
}

fn path_is_existing_directory(path: &Path) -> bool {
    match vfs::metadata(path) {
        Ok(meta) => meta.is_dir(),
        Err(_) => false,
    }
}

fn path_exists(path: &Path) -> bool {
    vfs::metadata(path).is_ok()
}

fn extract_folder_name(value: &Value, error_message: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(array) => {
            if array.rows == 1 {
                Ok(array.data.iter().collect())
            } else {
                Err(mkdir_error(error_message))
            }
        }
        Value::StringArray(array) => {
            if array.data.len() == 1 {
                Ok(array.data[0].clone())
            } else {
                Err(mkdir_error(error_message))
            }
        }
        _ => Err(mkdir_error(error_message)),
    }
}

fn gather_arguments(args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for value in args {
        out.push(gather_if_needed(value).map_err(map_control_flow)?);
    }
    Ok(out)
}

fn char_array_value(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::REPL_FS_TEST_LOCK;
    use super::*;
    use runmat_builtins::Value;
    use runmat_filesystem as vfs;
    use std::fs;
    use std::fs::File;
    use tempfile::tempdir;


    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mkdir_creates_directory_with_single_argument() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let target = temp.path().join("single-arg");
        let result =
            mkdir_builtin(vec![Value::from(target.to_string_lossy().to_string())]).expect("mkdir");
        assert_eq!(result, Value::Num(1.0));
        assert!(path_is_existing_directory(&target));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mkdir_returns_success_when_directory_already_exists() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let target = temp.path().join("existing");
        vfs::create_dir(&target).expect("seed dir");

        let eval = evaluate(&[Value::from(target.to_string_lossy().to_string())]).unwrap();
        assert_eq!(eval.status(), 1.0);
        assert_eq!(eval.message(), "Directory already exists.");
        assert_eq!(eval.message_id(), MESSAGE_ID_DIRECTORY_EXISTS);
        assert!(path_is_existing_directory(&target));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mkdir_combines_parent_and_child_paths() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let parent = temp.path().join("data");
        fs::create_dir(&parent).expect("parent dir");
        let child = "archive/2024";

        let eval = evaluate(&[
            Value::from(parent.to_string_lossy().to_string()),
            Value::from(child.to_string()),
        ])
        .expect("mkdir");
        assert_eq!(eval.status(), 1.0);
        assert!(eval.message().is_empty());
        assert!(path_is_existing_directory(&parent.join(child)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mkdir_requires_string_inputs() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let err = mkdir_builtin(vec![Value::Num(42.0)]).expect_err("expected error");
        assert_eq!(err.message(), ERR_FOLDER_ARG);

        let err = evaluate(&[Value::from("parent"), Value::Num(7.0)]).expect_err("error");
        assert_eq!(err.message(), ERR_FOLDER_ARG);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mkdir_detects_missing_parent_directory() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let parent = temp.path().join("missing");
        let expected_target = parent.join("child");
        let eval = evaluate(&[
            Value::from(parent.to_string_lossy().to_string()),
            Value::from("child".to_string()),
        ])
        .expect("mkdir evaluates");
        assert_eq!(eval.status(), 0.0);
        assert_eq!(eval.message_id(), MESSAGE_ID_INVALID_PARENT);
        assert!(eval.message().contains("does not exist"));
        assert!(!path_exists(&expected_target));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mkdir_detects_parent_path_is_not_directory() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let parent_file = temp.path().join("parent.txt");
        File::create(&parent_file).expect("create file");

        let eval = evaluate(&[
            Value::from(parent_file.to_string_lossy().to_string()),
            Value::from("child".to_string()),
        ])
        .expect("evaluate");
        assert_eq!(eval.status(), 0.0);
        assert_eq!(eval.message_id(), MESSAGE_ID_NOT_A_DIRECTORY);
        assert!(eval.message().contains("not a directory"));
        let child = parent_file.with_file_name("child");
        assert!(!path_exists(&child));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mkdir_captures_failure_status_and_message() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let file_path = temp.path().join("occupied");
        File::create(&file_path).expect("create file");

        let eval =
            evaluate(&[Value::from(file_path.to_string_lossy().to_string())]).expect("evaluate");
        assert_eq!(eval.status(), 0.0);
        assert_eq!(eval.message_id(), MESSAGE_ID_NOT_A_DIRECTORY);
        assert!(eval.message().contains("non-directory"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mkdir_rejects_empty_folder_name() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let single = evaluate(&[Value::from(String::new())]).expect("evaluate");
        assert_eq!(single.status(), 0.0);
        assert_eq!(single.message_id(), MESSAGE_ID_EMPTY_NAME);
        assert!(single.message().contains("must not be empty"));

        let paired =
            evaluate(&[Value::from("parent"), Value::from(String::new())]).expect("evaluate");
        assert_eq!(paired.status(), 0.0);
        assert_eq!(paired.message_id(), MESSAGE_ID_EMPTY_NAME);
        assert!(paired.message().contains("must not be empty"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mkdir_outputs_vector_contains_message_and_id() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let target = temp.path().join("outputs");
        let eval = evaluate(&[Value::from(target.to_string_lossy().to_string())]).unwrap();
        let outputs = eval.outputs();
        assert_eq!(outputs.len(), 3);
        assert!(matches!(outputs[0], Value::Num(1.0)));
        assert!(matches!(outputs[1], Value::CharArray(ref ca) if ca.cols == 0));
        assert!(matches!(outputs[2], Value::CharArray(ref ca) if ca.cols == 0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = crate::builtins::common::test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
