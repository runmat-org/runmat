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
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const MESSAGE_ID_OS_ERROR: &str = "MATLAB:MKDIR:OSError";
const MESSAGE_ID_DIRECTORY_EXISTS: &str = "MATLAB:MKDIR:DirectoryExists";
const MESSAGE_ID_INVALID_PARENT: &str = "MATLAB:MKDIR:ParentDirectoryDoesNotExist";
const MESSAGE_ID_NOT_A_DIRECTORY: &str = "MATLAB:MKDIR:ParentIsNotDirectory";
const MESSAGE_ID_EMPTY_NAME: &str = "MATLAB:MKDIR:InvalidFolderName";

const ERR_FOLDER_ARG: &str = "mkdir: folder name must be a character vector or string scalar";
const ERR_PARENT_ARG: &str = "mkdir: parent folder must be a character vector or string scalar";

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
async fn mkdir_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(&args).await?;
    Ok(eval.first_output())
}

/// Evaluate `mkdir` once and expose all outputs.
pub async fn evaluate(args: &[Value]) -> BuiltinResult<MkdirResult> {
    let gathered = gather_arguments(args).await?;
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

async fn gather_arguments(args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for value in args {
        out.push(
            gather_if_needed_async(value)
                .await
                .map_err(map_control_flow)?,
        );
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

    fn mkdir_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::mkdir_builtin(args))
    }

    fn evaluate(args: &[Value]) -> BuiltinResult<MkdirResult> {
        futures::executor::block_on(super::evaluate(args))
    }

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
}
