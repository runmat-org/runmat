//! MATLAB-compatible `cd` builtin for RunMat.

use runmat_filesystem as vfs;
#[cfg(test)]
use std::env;
use std::path::{Path, PathBuf};

use runmat_builtins::{CharArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::cd")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cd",
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
        "Host-only operation that updates the process working folder; GPU inputs are gathered before path resolution.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::cd")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cd",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "I/O builtins are not eligible for fusion; metadata is registered for completeness.",
};

const BUILTIN_NAME: &str = "cd";

fn cd_error(message: impl Into<String>) -> RuntimeError {
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
    name = "cd",
    category = "io/repl_fs",
    summary = "Change the current working folder or query the folder that RunMat is executing in.",
    keywords = "cd,change directory,current folder,working directory,pwd",
    accel = "cpu",
    suppress_auto_output = true,
    builtin_path = "crate::builtins::io::repl_fs::cd"
)]
async fn cd_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let gathered = gather_arguments(&args).await?;
    match gathered.len() {
        0 => current_directory_value(),
        1 => change_directory(&gathered[0]),
        _ => Err(cd_error("cd: too many input arguments")),
    }
}

fn current_directory_value() -> BuiltinResult<Value> {
    let current = vfs::current_dir()
        .map_err(|err| cd_error(format!("cd: unable to determine current directory ({err})")))?;
    Ok(path_to_value(&current))
}

fn change_directory(value: &Value) -> BuiltinResult<Value> {
    let target_raw = extract_path(value)?;
    let target = expand_path(&target_raw)?;
    let previous = vfs::current_dir()
        .map_err(|err| cd_error(format!("cd: unable to determine current directory ({err})")))?;

    vfs::set_current_dir(&target).map_err(|err| {
        cd_error(format!(
            "cd: unable to change directory to '{target_raw}' ({err})"
        ))
    })?;

    let _new_path = vfs::current_dir()
        .map_err(|err| cd_error(format!("cd: unable to determine current directory ({err})")))?;
    Ok(path_to_value(&previous))
}

fn extract_path(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(text) => {
            if text.is_empty() {
                Err(cd_error("cd: folder name must not be empty"))
            } else {
                Ok(text.clone())
            }
        }
        Value::StringArray(array) => {
            if array.data.len() != 1 {
                return Err(cd_error(
                    "cd: folder name must be a character vector or string scalar",
                ));
            }
            let text = array.data[0].clone();
            if text.is_empty() {
                Err(cd_error("cd: folder name must not be empty"))
            } else {
                Ok(text)
            }
        }
        Value::CharArray(chars) => {
            if chars.rows != 1 {
                return Err(cd_error(
                    "cd: folder name must be a character vector or string scalar",
                ));
            }
            let text: String = chars.data.iter().collect();
            if text.is_empty() {
                Err(cd_error("cd: folder name must not be empty"))
            } else {
                Ok(text)
            }
        }
        _ => Err(cd_error(
            "cd: folder name must be a character vector or string scalar",
        )),
    }
}

fn expand_path(raw: &str) -> BuiltinResult<PathBuf> {
    let expanded =
        crate::builtins::common::fs::expand_user_path(raw, BUILTIN_NAME).map_err(cd_error)?;
    Ok(PathBuf::from(expanded))
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

fn path_to_value(path: &Path) -> Value {
    let text = path_to_string(path);
    char_array_value(&text)
}

fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn char_array_value(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::REPL_FS_TEST_LOCK;
    use super::*;
    use runmat_builtins::StringArray;
    use std::convert::TryFrom;
    use std::path::{Path, PathBuf};
    use tempfile::tempdir;

    fn cd_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::cd_builtin(args))
    }

    fn canonical_path(path: &Path) -> PathBuf {
        std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
    }

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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cd_returns_current_directory() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = DirGuard::new();

        let expected = env::current_dir().expect("current dir");
        let value = cd_builtin(Vec::new()).expect("cd");
        let actual = String::try_from(&value).expect("string conversion");
        assert_eq!(actual, expected.to_string_lossy());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cd_changes_directory_and_returns_previous() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = DirGuard::new();

        let original = env::current_dir().expect("current dir");
        let temp = tempdir().expect("tempdir");
        let path_str = temp.path().to_string_lossy().to_string();

        let previous = cd_builtin(vec![Value::from(path_str)]).expect("cd change");
        let previous_str = String::try_from(&previous).expect("string conversion");
        let previous_path = PathBuf::from(previous_str);
        assert_eq!(canonical_path(&previous_path), canonical_path(&original));

        let new_dir = env::current_dir().expect("current dir");
        assert_eq!(canonical_path(&new_dir), canonical_path(temp.path()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cd_supports_relative_char_array_paths() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = DirGuard::new();

        let root = tempdir().expect("root tempdir");
        let child = root.path().join("child");
        std::fs::create_dir(&child).expect("create child");

        let _ = cd_builtin(vec![Value::from(root.path().to_string_lossy().to_string())])
            .expect("cd root");

        let relative = Value::CharArray(CharArray::new_row("child"));
        let previous = cd_builtin(vec![relative]).expect("cd child");
        let previous_str = String::try_from(&previous).expect("string conversion");
        let previous_path = PathBuf::from(previous_str);
        assert_eq!(canonical_path(&previous_path), canonical_path(root.path()));
        let current = env::current_dir().expect("current dir");
        assert_eq!(canonical_path(&current), canonical_path(&child));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cd_errors_when_folder_missing() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = DirGuard::new();

        let missing = Value::from("this-directory-does-not-exist".to_string());
        let err = cd_builtin(vec![missing]).expect_err("error");
        assert!(err.message().contains("cd: unable to change directory"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cd_tilde_expands_to_home_directory() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let guard = DirGuard::new();
        let original = guard.original.clone();

        let home_text =
            crate::builtins::common::fs::expand_user_path("~", BUILTIN_NAME).expect("home");
        let home = PathBuf::from(home_text);
        let previous = cd_builtin(vec![Value::from("~")]).expect("cd ~");
        let previous_str = String::try_from(&previous).expect("string conversion");
        let previous_path = PathBuf::from(previous_str);

        assert_eq!(canonical_path(&previous_path), canonical_path(&original));
        let current = env::current_dir().expect("current dir");
        assert_eq!(canonical_path(&current), canonical_path(&home));
        drop(guard);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cd_errors_on_empty_string() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = DirGuard::new();

        let err = cd_builtin(vec![Value::from("".to_string())]).expect_err("empty string error");
        assert_eq!(err.message(), "cd: folder name must not be empty");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cd_errors_on_multi_element_string_array() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = DirGuard::new();

        let strings =
            StringArray::new(vec!["foo".to_string(), "bar".to_string()], vec![2]).expect("array");
        let err = cd_builtin(vec![Value::StringArray(strings)]).expect_err("string array error");
        assert_eq!(
            err.message(),
            "cd: folder name must be a character vector or string scalar"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cd_errors_on_multiline_char_array() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = DirGuard::new();

        let chars = CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).expect("char array");
        let err = cd_builtin(vec![Value::CharArray(chars)]).expect_err("char array error");
        assert_eq!(
            err.message(),
            "cd: folder name must be a character vector or string scalar"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cd_accepts_string_array_scalar() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let guard = DirGuard::new();

        let current = env::current_dir().expect("current dir");
        let scalar = StringArray::new(vec![current.to_string_lossy().to_string()], vec![1])
            .expect("scalar string array");
        let previous = cd_builtin(vec![Value::StringArray(scalar)]).expect("cd");
        let previous_str = String::try_from(&previous).expect("string conversion");
        assert_eq!(previous_str, current.to_string_lossy());

        drop(guard);
    }
}
