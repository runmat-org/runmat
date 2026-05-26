//! MATLAB-compatible `mkdir` builtin for RunMat.

use std::path::{Path, PathBuf};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, Value,
};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

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

const MKDIR_OUTPUT_STATUS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "1 on success, 0 when creation fails.",
}];
const MKDIR_OUTPUT_STATUS_MSG_MSGID: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "status",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "1 on success, 0 when creation fails.",
    },
    BuiltinParamDescriptor {
        name: "msg",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Diagnostic message for failures.",
    },
    BuiltinParamDescriptor {
        name: "msgID",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Stable diagnostic identifier string.",
    },
];
const MKDIR_INPUTS_FOLDER: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "folderName",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Folder path to create.",
}];
const MKDIR_INPUTS_PARENT_CHILD: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "parentFolder",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Parent folder path.",
    },
    BuiltinParamDescriptor {
        name: "folderName",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Child folder path under parentFolder.",
    },
];
const MKDIR_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "status = mkdir(folderName)",
        inputs: &MKDIR_INPUTS_FOLDER,
        outputs: &MKDIR_OUTPUT_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "status = mkdir(parentFolder, folderName)",
        inputs: &MKDIR_INPUTS_PARENT_CHILD,
        outputs: &MKDIR_OUTPUT_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "[status, msg, msgID] = mkdir(folderName)",
        inputs: &MKDIR_INPUTS_FOLDER,
        outputs: &MKDIR_OUTPUT_STATUS_MSG_MSGID,
    },
    BuiltinSignatureDescriptor {
        label: "[status, msg, msgID] = mkdir(parentFolder, folderName)",
        inputs: &MKDIR_INPUTS_PARENT_CHILD,
        outputs: &MKDIR_OUTPUT_STATUS_MSG_MSGID,
    },
];
const MKDIR_ERROR_NOT_ENOUGH_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MKDIR.NOT_ENOUGH_INPUTS",
    identifier: Some("RunMat:mkdir:NotEnoughInputs"),
    when: "No input arguments are provided.",
    message: "mkdir: not enough input arguments",
};
const MKDIR_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MKDIR.TOO_MANY_INPUTS",
    identifier: Some("RunMat:mkdir:TooManyInputs"),
    when: "More than two input arguments are provided.",
    message: "mkdir: too many input arguments",
};
const MKDIR_ERROR_FOLDER_ARG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MKDIR.FOLDER_ARG",
    identifier: Some("RunMat:mkdir:FolderArgType"),
    when: "Folder argument is not a character vector or string scalar.",
    message: "mkdir: folder name must be a character vector or string scalar",
};
const MKDIR_ERROR_PARENT_ARG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MKDIR.PARENT_ARG",
    identifier: Some("RunMat:mkdir:ParentArgType"),
    when: "Parent argument is not a character vector or string scalar.",
    message: "mkdir: parent folder must be a character vector or string scalar",
};
const MKDIR_RESULT_OS_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MKDIR.OS_ERROR",
    identifier: Some("RunMat:mkdir:OSError"),
    when: "Directory creation fails due to filesystem error.",
    message: "mkdir: unable to create folder",
};
const MKDIR_RESULT_DIRECTORY_EXISTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MKDIR.DIRECTORY_EXISTS",
    identifier: Some("RunMat:mkdir:DirectoryExists"),
    when: "Target directory already exists.",
    message: "Directory already exists.",
};
const MKDIR_RESULT_INVALID_PARENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MKDIR.INVALID_PARENT",
    identifier: Some("RunMat:mkdir:ParentDirectoryDoesNotExist"),
    when: "Parent folder does not exist.",
    message: "mkdir: parent folder does not exist",
};
const MKDIR_RESULT_NOT_A_DIRECTORY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MKDIR.NOT_A_DIRECTORY",
    identifier: Some("RunMat:mkdir:ParentIsNotDirectory"),
    when: "Parent or target path exists as a non-directory entry.",
    message: "mkdir: target path is not a directory",
};
const MKDIR_RESULT_EMPTY_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MKDIR.EMPTY_NAME",
    identifier: Some("RunMat:mkdir:InvalidFolderName"),
    when: "Folder name is empty.",
    message: "Folder name must not be empty.",
};
const MKDIR_ERRORS: [BuiltinErrorDescriptor; 9] = [
    MKDIR_ERROR_NOT_ENOUGH_INPUTS,
    MKDIR_ERROR_TOO_MANY_INPUTS,
    MKDIR_ERROR_FOLDER_ARG,
    MKDIR_ERROR_PARENT_ARG,
    MKDIR_RESULT_OS_ERROR,
    MKDIR_RESULT_DIRECTORY_EXISTS,
    MKDIR_RESULT_INVALID_PARENT,
    MKDIR_RESULT_NOT_A_DIRECTORY,
    MKDIR_RESULT_EMPTY_NAME,
];
pub const MKDIR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MKDIR_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MKDIR_ERRORS,
};

fn mkdir_error_row(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

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
    type_resolver(crate::builtins::io::type_resolvers::mkdir_type),
    descriptor(crate::builtins::io::repl_fs::mkdir::MKDIR_DESCRIPTOR),
    builtin_path = "crate::builtins::io::repl_fs::mkdir"
)]
async fn mkdir_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(&args).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            eval.outputs(),
        ));
    }
    Ok(eval.first_output())
}

/// Evaluate `mkdir` once and expose all outputs.
pub async fn evaluate(args: &[Value]) -> BuiltinResult<MkdirResult> {
    let gathered = gather_arguments(args).await?;
    match gathered.len() {
        0 => Err(mkdir_error_row(&MKDIR_ERROR_NOT_ENOUGH_INPUTS)),
        1 => create_from_single(&gathered[0]).await,
        2 => create_from_parent_child(&gathered[0], &gathered[1]).await,
        _ => Err(mkdir_error_row(&MKDIR_ERROR_TOO_MANY_INPUTS)),
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
            message_id: message_identifier(&MKDIR_RESULT_DIRECTORY_EXISTS).to_string(),
        }
    }

    fn failure(message: String, error: &'static BuiltinErrorDescriptor) -> Self {
        Self {
            status: 0.0,
            message,
            message_id: message_identifier(error).to_string(),
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

async fn create_from_single(value: &Value) -> BuiltinResult<MkdirResult> {
    let raw = extract_folder_name(value, &MKDIR_ERROR_FOLDER_ARG)?;
    if raw.is_empty() {
        return Ok(MkdirResult::failure(
            "Folder name must not be empty.".to_string(),
            &MKDIR_RESULT_EMPTY_NAME,
        ));
    }
    let expanded = expand_user_path(&raw, "mkdir").map_err(mkdir_error)?;
    let path = PathBuf::from(expanded);
    Ok(create_directory(&path).await)
}

async fn create_from_parent_child(parent: &Value, child: &Value) -> BuiltinResult<MkdirResult> {
    let parent_raw = extract_folder_name(parent, &MKDIR_ERROR_PARENT_ARG)?;
    let child_raw = extract_folder_name(child, &MKDIR_ERROR_FOLDER_ARG)?;

    if child_raw.is_empty() {
        return Ok(MkdirResult::failure(
            "Folder name must not be empty.".to_string(),
            &MKDIR_RESULT_EMPTY_NAME,
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
        return Ok(create_directory(&child_path).await);
    }

    if let Some(parent_text) = parent_expanded {
        let parent_path = PathBuf::from(&parent_text);
        if !path_exists_async(&parent_path).await {
            let message = format!("Parent folder \"{}\" does not exist.", parent_text);
            return Ok(MkdirResult::failure(message, &MKDIR_RESULT_INVALID_PARENT));
        }
        if !path_is_existing_directory_async(&parent_path).await {
            let message = format!("Parent folder \"{}\" is not a directory.", parent_text);
            return Ok(MkdirResult::failure(message, &MKDIR_RESULT_NOT_A_DIRECTORY));
        }
        let target = parent_path.join(&child_expanded);
        return Ok(create_directory(&target).await);
    }

    Ok(create_directory(&PathBuf::from(&child_expanded)).await)
}

async fn create_directory(path: &Path) -> MkdirResult {
    let display = path.display().to_string();
    if path_exists_async(path).await {
        if path_is_existing_directory_async(path).await {
            return MkdirResult::already_exists();
        }
        return MkdirResult::failure(
            format!(
                "Cannot create folder \"{}\". A file or non-directory item with the same name already exists.",
                display
            ),
            &MKDIR_RESULT_NOT_A_DIRECTORY,
        );
    }

    match vfs::create_dir_all_async(path).await {
        Ok(_) => MkdirResult::success(),
        Err(err) => MkdirResult::failure(
            format!("Unable to create folder \"{}\": {}", display, err),
            &MKDIR_RESULT_OS_ERROR,
        ),
    }
}

async fn path_is_existing_directory_async(path: &Path) -> bool {
    match vfs::metadata_async(path).await {
        Ok(meta) => meta.is_dir(),
        Err(_) => false,
    }
}

async fn path_exists_async(path: &Path) -> bool {
    vfs::metadata_async(path).await.is_ok()
}

#[cfg(test)]
fn path_is_existing_directory(path: &Path) -> bool {
    futures::executor::block_on(path_is_existing_directory_async(path))
}

#[cfg(test)]
fn path_exists(path: &Path) -> bool {
    futures::executor::block_on(path_exists_async(path))
}

fn extract_folder_name(
    value: &Value,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(array) => {
            if array.rows == 1 {
                Ok(array.data.iter().collect())
            } else {
                Err(mkdir_error_row(error))
            }
        }
        Value::StringArray(array) => {
            if array.data.len() == 1 {
                Ok(array.data[0].clone())
            } else {
                Err(mkdir_error_row(error))
            }
        }
        _ => Err(mkdir_error_row(error)),
    }
}

fn message_identifier(error: &'static BuiltinErrorDescriptor) -> &'static str {
    error.identifier.unwrap_or("")
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
    fn mkdir_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = MKDIR_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"status = mkdir(folderName)"));
        assert!(labels.contains(&"status = mkdir(parentFolder, folderName)"));
        assert!(labels.contains(&"[status, msg, msgID] = mkdir(folderName)"));
        assert!(labels.contains(&"[status, msg, msgID] = mkdir(parentFolder, folderName)"));
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
        futures::executor::block_on(vfs::create_dir_async(&target)).expect("seed dir");

        let eval = evaluate(&[Value::from(target.to_string_lossy().to_string())]).unwrap();
        assert_eq!(eval.status(), 1.0);
        assert_eq!(eval.message(), "Directory already exists.");
        assert_eq!(
            eval.message_id(),
            message_identifier(&MKDIR_RESULT_DIRECTORY_EXISTS)
        );
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
        assert_eq!(err.message(), MKDIR_ERROR_FOLDER_ARG.message);

        let err = evaluate(&[Value::from("parent"), Value::Num(7.0)]).expect_err("error");
        assert_eq!(err.message(), MKDIR_ERROR_FOLDER_ARG.message);
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
        assert_eq!(
            eval.message_id(),
            message_identifier(&MKDIR_RESULT_INVALID_PARENT)
        );
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
        assert_eq!(
            eval.message_id(),
            message_identifier(&MKDIR_RESULT_NOT_A_DIRECTORY)
        );
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
        assert_eq!(
            eval.message_id(),
            message_identifier(&MKDIR_RESULT_NOT_A_DIRECTORY)
        );
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
        assert_eq!(
            single.message_id(),
            message_identifier(&MKDIR_RESULT_EMPTY_NAME)
        );
        assert!(single.message().contains("must not be empty"));

        let paired =
            evaluate(&[Value::from("parent"), Value::from(String::new())]).expect("evaluate");
        assert_eq!(paired.status(), 0.0);
        assert_eq!(
            paired.message_id(),
            message_identifier(&MKDIR_RESULT_EMPTY_NAME)
        );
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
