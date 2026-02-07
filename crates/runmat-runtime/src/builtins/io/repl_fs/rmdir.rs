//! MATLAB-compatible `rmdir` builtin for RunMat.

use runmat_filesystem as vfs;
use std::io;
use std::path::{Path, PathBuf};

use runmat_builtins::{CharArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const MESSAGE_ID_OS_ERROR: &str = "MATLAB:RMDIR:OSError";
const MESSAGE_ID_DIRECTORY_NOT_FOUND: &str = "MATLAB:RMDIR:DirectoryNotFound";
const MESSAGE_ID_NOT_A_DIRECTORY: &str = "MATLAB:RMDIR:NotADirectory";
const MESSAGE_ID_DIRECTORY_NOT_EMPTY: &str = "MATLAB:RMDIR:DirectoryNotEmpty";
const MESSAGE_ID_EMPTY_NAME: &str = "MATLAB:RMDIR:InvalidFolderName";

const ERR_FOLDER_ARG: &str = "rmdir: folder name must be a character vector or string scalar";
const ERR_FLAG_ARG: &str =
    "rmdir: flag must be the character 's' supplied as a char vector or string scalar";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::rmdir")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "rmdir",
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
        "Host-only filesystem builtin. GPU-resident path and flag arguments are gathered automatically before removal.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::rmdir")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "rmdir",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Filesystem side-effects materialise immediately; metadata registered for completeness.",
};

const BUILTIN_NAME: &str = "rmdir";

fn rmdir_error(message: impl Into<String>) -> RuntimeError {
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
    name = "rmdir",
    category = "io/repl_fs",
    summary = "Remove folders with MATLAB-compatible status, message, and message ID outputs.",
    keywords = "rmdir,remove directory,delete folder,filesystem,status,message,messageid,recursive",
    accel = "cpu",
    suppress_auto_output = true,
    type_resolver(crate::builtins::io::type_resolvers::rmdir_type),
    builtin_path = "crate::builtins::io::repl_fs::rmdir"
)]
async fn rmdir_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(&args).await?;
    Ok(eval.first_output())
}

/// Evaluate `rmdir` once and expose all outputs.
pub async fn evaluate(args: &[Value]) -> BuiltinResult<RmdirResult> {
    let gathered = gather_arguments(args).await?;
    match gathered.len() {
        0 => Err(rmdir_error("rmdir: not enough input arguments")),
        1 => remove_folder(&gathered[0], RemoveMode::NonRecursive),
        2 => {
            let mode = parse_remove_mode(&gathered[1])?;
            remove_folder(&gathered[0], mode)
        }
        _ => Err(rmdir_error("rmdir: too many input arguments")),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RemoveMode {
    NonRecursive,
    Recursive,
}

#[derive(Debug, Clone)]
pub struct RmdirResult {
    status: f64,
    message: String,
    message_id: String,
}

impl RmdirResult {
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

    fn not_found(display: &str) -> Self {
        Self::failure(
            format!("Folder \"{}\" does not exist.", display),
            MESSAGE_ID_DIRECTORY_NOT_FOUND,
        )
    }

    fn not_directory(display: &str) -> Self {
        Self::failure(
            format!("Cannot remove \"{}\": target is not a directory.", display),
            MESSAGE_ID_NOT_A_DIRECTORY,
        )
    }

    fn not_empty(display: &str) -> Self {
        Self::failure(
            format!(
                "Cannot remove folder \"{}\": directory is not empty.",
                display
            ),
            MESSAGE_ID_DIRECTORY_NOT_EMPTY,
        )
    }

    fn os_error(display: &str, err: &io::Error) -> Self {
        Self::failure(
            format!("Unable to remove folder \"{}\": {}", display, err),
            MESSAGE_ID_OS_ERROR,
        )
    }

    fn empty_name() -> Self {
        Self::failure(
            "Folder name must not be empty.".to_string(),
            MESSAGE_ID_EMPTY_NAME,
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

fn remove_folder(value: &Value, mode: RemoveMode) -> BuiltinResult<RmdirResult> {
    let raw = extract_folder_name(value)?;
    if raw.trim().is_empty() {
        return Ok(RmdirResult::empty_name());
    }
    let expanded = expand_user_path(&raw, "rmdir").map_err(rmdir_error)?;
    let path = PathBuf::from(&expanded);
    Ok(remove_directory(&path, mode))
}

fn remove_directory(path: &Path, mode: RemoveMode) -> RmdirResult {
    let display = path.display().to_string();

    let metadata = match vfs::metadata(path) {
        Ok(meta) => meta,
        Err(err) => {
            return if err.kind() == io::ErrorKind::NotFound {
                RmdirResult::not_found(&display)
            } else {
                RmdirResult::os_error(&display, &err)
            };
        }
    };

    if !metadata.is_dir() {
        return RmdirResult::not_directory(&display);
    }

    match mode {
        RemoveMode::Recursive => match vfs::remove_dir_all(path) {
            Ok(_) => RmdirResult::success(),
            Err(err) => {
                if err.kind() == io::ErrorKind::NotFound {
                    RmdirResult::not_found(&display)
                } else {
                    RmdirResult::os_error(&display, &err)
                }
            }
        },
        RemoveMode::NonRecursive => match vfs::remove_dir(path) {
            Ok(_) => RmdirResult::success(),
            Err(err) => {
                if err.kind() == io::ErrorKind::NotFound {
                    RmdirResult::not_found(&display)
                } else if err.kind() == io::ErrorKind::DirectoryNotEmpty {
                    RmdirResult::not_empty(&display)
                } else {
                    RmdirResult::os_error(&display, &err)
                }
            }
        },
    }
}

fn parse_remove_mode(value: &Value) -> BuiltinResult<RemoveMode> {
    let text = match value {
        Value::String(s) => s.clone(),
        Value::CharArray(array) => {
            if array.rows == 1 {
                array.data.iter().collect()
            } else {
                return Err(rmdir_error(ERR_FLAG_ARG));
            }
        }
        Value::StringArray(array) => {
            if array.data.len() == 1 {
                array.data[0].clone()
            } else {
                return Err(rmdir_error(ERR_FLAG_ARG));
            }
        }
        _ => return Err(rmdir_error(ERR_FLAG_ARG)),
    };

    if text.trim().eq_ignore_ascii_case("s") {
        Ok(RemoveMode::Recursive)
    } else {
        Err(rmdir_error(ERR_FLAG_ARG))
    }
}

fn extract_folder_name(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(array) => {
            if array.rows == 1 {
                Ok(array.data.iter().collect())
            } else {
                Err(rmdir_error(ERR_FOLDER_ARG))
            }
        }
        Value::StringArray(array) => {
            if array.data.len() == 1 {
                Ok(array.data[0].clone())
            } else {
                Err(rmdir_error(ERR_FOLDER_ARG))
            }
        }
        _ => Err(rmdir_error(ERR_FOLDER_ARG)),
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
    use std::fs;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    fn evaluate(args: &[Value]) -> BuiltinResult<RmdirResult> {
        futures::executor::block_on(super::evaluate(args))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmdir_removes_empty_directory() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let target = temp.path().join("empty");
        fs::create_dir(&target).expect("create folder");
        let eval = evaluate(&[Value::from(target.to_string_lossy().to_string())]).unwrap();
        assert_eq!(eval.status(), 1.0);
        assert!(eval.message().is_empty());
        assert!(eval.message_id().is_empty());
        assert!(!target.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmdir_requires_string_inputs() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let err = evaluate(&[Value::Num(1.0)]).expect_err("expected error");
        assert_eq!(err.message(), ERR_FOLDER_ARG);

        let err = evaluate(&[Value::from("path"), Value::Num(2.0)]).expect_err("error");
        assert_eq!(err.message(), ERR_FLAG_ARG);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmdir_non_recursive_fails_when_not_empty() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let target = temp.path().join("logs");
        fs::create_dir(&target).expect("create folder");
        let mut file = File::create(target.join("latest.log")).expect("create file");
        writeln!(file, "entry").expect("write");

        let eval = remove_folder(
            &Value::from(target.to_string_lossy().to_string()),
            RemoveMode::NonRecursive,
        )
        .unwrap();
        assert_eq!(eval.status(), 0.0);
        assert_eq!(eval.message_id(), MESSAGE_ID_DIRECTORY_NOT_EMPTY);
        assert!(target.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmdir_recursive_removes_contents() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let target = temp.path().join("data");
        fs::create_dir(&target).expect("create folder");
        let mut file = File::create(target.join("payload.bin")).expect("create file");
        file.write_all(b"payload").expect("write");

        let eval = evaluate(&[
            Value::from(target.to_string_lossy().to_string()),
            Value::from("s"),
        ])
        .unwrap();
        assert_eq!(eval.status(), 1.0);
        assert!(!target.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmdir_handles_missing_directory() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let target = temp.path().join("missing");
        let eval =
            evaluate(&[Value::from(target.to_string_lossy().to_string())]).expect("evaluate");
        assert_eq!(eval.status(), 0.0);
        assert_eq!(eval.message_id(), MESSAGE_ID_DIRECTORY_NOT_FOUND);
        assert!(eval.message().contains("does not exist"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmdir_rejects_files() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let target = temp.path().join("file.txt");
        File::create(&target).expect("create file");
        let eval =
            evaluate(&[Value::from(target.to_string_lossy().to_string())]).expect("evaluate");
        assert_eq!(eval.status(), 0.0);
        assert_eq!(eval.message_id(), MESSAGE_ID_NOT_A_DIRECTORY);
        assert!(eval.message().contains("not a directory"));
        assert!(target.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmdir_accepts_uppercase_flag() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let target = temp.path().join("upper");
        fs::create_dir(&target).expect("create folder");
        let eval = evaluate(&[
            Value::from(target.to_string_lossy().to_string()),
            Value::from("S"),
        ])
        .unwrap();
        assert_eq!(eval.status(), 1.0);
        assert!(!target.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmdir_outputs_produce_char_arrays() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let target = temp.path().join("outputs");
        fs::create_dir(&target).expect("create folder");
        let eval = evaluate(&[Value::from(target.to_string_lossy().to_string())]).unwrap();
        let outputs = eval.outputs();
        assert_eq!(outputs.len(), 3);
        assert!(matches!(outputs[0], Value::Num(1.0)));
        assert!(matches!(outputs[1], Value::CharArray(ref ca) if ca.cols == 0));
        assert!(matches!(outputs[2], Value::CharArray(ref ca) if ca.cols == 0));
    }
}
