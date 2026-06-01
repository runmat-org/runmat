//! MATLAB-compatible `tempname` builtin for RunMat.

use runmat_time::system_time_now;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::UNIX_EPOCH;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, Value,
};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;

use crate::builtins::common::env as runtime_env;
use crate::builtins::common::fs::{expand_user_path, path_to_string};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const MAX_ATTEMPTS: usize = 64;
static UNIQUE_COUNTER: AtomicU64 = AtomicU64::new(0);

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::tempname")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "tempname",
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
    notes: "Host-only path generation. Providers are not expected to supply kernels for temporary name creation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::tempname")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "tempname",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "I/O builtins are not eligible for fusion; metadata is registered for completeness.",
};

const BUILTIN_NAME: &str = "tempname";

const TEMPNAME_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Unique temporary file path that does not currently exist.",
}];
const TEMPNAME_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const TEMPNAME_INPUTS_FOLDER: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "folder",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Base folder used for generated temporary file paths.",
}];
const TEMPNAME_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "filename = tempname()",
        inputs: &TEMPNAME_INPUTS_NONE,
        outputs: &TEMPNAME_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "filename = tempname(folder)",
        inputs: &TEMPNAME_INPUTS_FOLDER,
        outputs: &TEMPNAME_OUTPUT,
    },
];
const TEMPNAME_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEMPNAME.TOO_MANY_INPUTS",
    identifier: None,
    when: "More than one positional argument is supplied.",
    message: "tempname: too many input arguments",
};
const TEMPNAME_ERROR_FOLDER_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEMPNAME.FOLDER_TYPE",
    identifier: None,
    when: "Folder argument is not a character vector, string scalar, or string-array scalar.",
    message: "tempname: folder name must be a character vector or string scalar",
};
const TEMPNAME_ERROR_FOLDER_EMPTY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEMPNAME.FOLDER_EMPTY",
    identifier: None,
    when: "Folder argument (or expanded folder path) is empty.",
    message: "tempname: folder name must not be empty",
};
const TEMPNAME_ERROR_FOLDER_RESOLVE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEMPNAME.FOLDER_RESOLVE",
    identifier: None,
    when: "Folder argument cannot be resolved during home-directory expansion.",
    message: "tempname: unable to resolve folder path",
};
const TEMPNAME_ERROR_TEMP_DIR_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEMPNAME.TEMP_DIR_UNAVAILABLE",
    identifier: None,
    when: "System temporary directory cannot be determined.",
    message: "tempname: unable to determine temporary directory",
};
const TEMPNAME_ERROR_UNABLE_TO_GENERATE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEMPNAME.UNABLE_TO_GENERATE",
    identifier: None,
    when: "Unique temp name generation exhausted available attempts.",
    message: "tempname: unable to generate a unique name",
};
const TEMPNAME_ERRORS: [BuiltinErrorDescriptor; 6] = [
    TEMPNAME_ERROR_FOLDER_RESOLVE,
    TEMPNAME_ERROR_TOO_MANY_INPUTS,
    TEMPNAME_ERROR_FOLDER_TYPE,
    TEMPNAME_ERROR_FOLDER_EMPTY,
    TEMPNAME_ERROR_TEMP_DIR_UNAVAILABLE,
    TEMPNAME_ERROR_UNABLE_TO_GENERATE,
];
pub const TEMPNAME_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TEMPNAME_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TEMPNAME_ERRORS,
};

fn tempname_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    tempname_error_with_message(error.message, error)
}

fn tempname_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn tempname_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    tempname_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
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
    name = "tempname",
    category = "io/repl_fs",
    summary = "Generate unique temporary file or folder paths.",
    keywords = "tempname,temporary file,unique name,temp directory",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::tempname_type),
    descriptor(crate::builtins::io::repl_fs::tempname::TEMPNAME_DESCRIPTOR),
    builtin_path = "crate::builtins::io::repl_fs::tempname"
)]
async fn tempname_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    match args.len() {
        0 => {
            let base = default_temp_directory()?;
            Ok(path_to_value(&generate_unique_path_async(&base).await?))
        }
        1 => {
            let gathered = gather_argument(&args[0]).await?;
            let folder = parse_folder_argument(&gathered)?;
            Ok(path_to_value(&generate_unique_path_async(&folder).await?))
        }
        _ => Err(tempname_error(&TEMPNAME_ERROR_TOO_MANY_INPUTS)),
    }
}

fn default_temp_directory() -> BuiltinResult<PathBuf> {
    let path = runtime_env::temp_dir();
    if path.as_os_str().is_empty() {
        Err(tempname_error(&TEMPNAME_ERROR_TEMP_DIR_UNAVAILABLE))
    } else {
        Ok(path)
    }
}

async fn gather_argument(value: &Value) -> BuiltinResult<Value> {
    gather_if_needed_async(value)
        .await
        .map_err(map_control_flow)
}

fn parse_folder_argument(value: &Value) -> BuiltinResult<PathBuf> {
    let text = match value {
        Value::String(s) => {
            if s.is_empty() {
                return Err(tempname_error(&TEMPNAME_ERROR_FOLDER_EMPTY));
            }
            s.clone()
        }
        Value::CharArray(array) => {
            if array.rows != 1 {
                return Err(tempname_error(&TEMPNAME_ERROR_FOLDER_TYPE));
            }
            let collected: String = array.data.iter().collect();
            if collected.is_empty() {
                return Err(tempname_error(&TEMPNAME_ERROR_FOLDER_EMPTY));
            }
            collected
        }
        Value::StringArray(array) => {
            if array.data.len() != 1 {
                return Err(tempname_error(&TEMPNAME_ERROR_FOLDER_TYPE));
            }
            let collected = array.data[0].clone();
            if collected.is_empty() {
                return Err(tempname_error(&TEMPNAME_ERROR_FOLDER_EMPTY));
            }
            collected
        }
        _ => return Err(tempname_error(&TEMPNAME_ERROR_FOLDER_TYPE)),
    };

    let expanded = expand_user_path(&text, "tempname")
        .map_err(|err| tempname_error_with_detail(&TEMPNAME_ERROR_FOLDER_RESOLVE, err))?;
    if expanded.is_empty() {
        Err(tempname_error(&TEMPNAME_ERROR_FOLDER_EMPTY))
    } else {
        Ok(PathBuf::from(expanded))
    }
}

async fn generate_unique_path_async(base: &Path) -> BuiltinResult<PathBuf> {
    for _ in 0..MAX_ATTEMPTS {
        let token = unique_token();
        let candidate = if base.as_os_str().is_empty() {
            PathBuf::from(&token)
        } else {
            base.join(&token)
        };
        if !path_exists_async(&candidate).await {
            return Ok(candidate);
        }
    }
    Err(tempname_error(&TEMPNAME_ERROR_UNABLE_TO_GENERATE))
}

fn unique_token() -> String {
    let now = system_time_now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let nanos = now.subsec_nanos();
    let pid = process_id();
    let counter = UNIQUE_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("tp{:016x}{:08x}{:08x}{:016x}", secs, nanos, pid, counter)
}

fn process_id() -> u64 {
    #[cfg(target_arch = "wasm32")]
    {
        0
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::process::id() as u64
    }
}

fn path_to_value(path: &Path) -> Value {
    let text = path_to_string(path);
    Value::CharArray(CharArray::new_row(&text))
}

async fn path_exists_async(path: &Path) -> bool {
    vfs::metadata_async(path).await.is_ok()
}

#[cfg(test)]
fn path_exists(path: &Path) -> bool {
    futures::executor::block_on(path_exists_async(path))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::REPL_FS_TEST_LOCK;
    use super::*;
    use crate::builtins::common::fs::home_directory;
    use runmat_builtins::StringArray;
    use std::convert::TryFrom;
    use std::path::{Path, PathBuf};
    use tempfile::tempdir;

    fn tempname_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::tempname_builtin(args))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempname_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = TEMPNAME_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"filename = tempname()"));
        assert!(labels.contains(&"filename = tempname(folder)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempname_generates_unique_names() {
        let _lock = REPL_FS_TEST_LOCK.lock().unwrap();
        let first = tempname_builtin(Vec::new()).expect("tempname");
        let second = tempname_builtin(Vec::new()).expect("tempname");
        let first_str = String::try_from(&first).expect("first string");
        let second_str = String::try_from(&second).expect("second string");
        assert_ne!(
            first_str, second_str,
            "tempname should return unique values"
        );
        assert!(!path_exists(Path::new(&first_str)), "first path exists");
        assert!(!path_exists(Path::new(&second_str)), "second path exists");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempname_returns_char_row_vector() {
        let value = tempname_builtin(Vec::new()).expect("tempname");
        match value {
            Value::CharArray(CharArray { rows, cols, .. }) => {
                assert_eq!(rows, 1);
                assert!(cols >= 1, "expected non-empty character row vector");
            }
            other => panic!("expected CharArray, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempname_uses_system_temp_directory_by_default() {
        let _lock = REPL_FS_TEST_LOCK.lock().unwrap();
        let value = tempname_builtin(Vec::new()).expect("tempname");
        let text = String::try_from(&value).expect("string conversion");
        let path = PathBuf::from(&text);
        let parent = path
            .parent()
            .expect("tempname should include a parent folder");
        assert_eq!(parent, runtime_env::temp_dir().as_path());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempname_accepts_custom_folder_char_array() {
        let _lock = REPL_FS_TEST_LOCK.lock().unwrap();
        let dir = tempdir().expect("tempdir");
        let base = dir.path();
        let base_text = base.to_string_lossy().into_owned();
        let arg = Value::CharArray(CharArray::new_row(&base_text));
        let value = tempname_builtin(vec![arg]).expect("tempname");
        let path_string = String::try_from(&value).expect("string");
        let path = PathBuf::from(&path_string);
        assert_eq!(
            path.parent(),
            Some(base),
            "expected tempname to honour the provided folder"
        );
        assert!(!path_exists(&path), "generated path should not exist yet");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempname_accepts_custom_folder_string_scalar() {
        let dir = tempdir().expect("tempdir");
        let base = dir.path().to_string_lossy().into_owned();
        let value = tempname_builtin(vec![Value::String(base.clone())]).expect("tempname");
        let path_string = String::try_from(&value).expect("string");
        assert!(
            path_string.starts_with(&base),
            "custom folder should prefix the result"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempname_allows_nonexistent_targets() {
        let dir = tempdir().expect("tempdir");
        let base = dir.path().join("nested").to_string_lossy().into_owned();
        let value = tempname_builtin(vec![Value::String(base.clone())]).expect("tempname");
        let path_string = String::try_from(&value).expect("string");
        assert!(
            path_string.starts_with(&base),
            "result should still be under the requested folder"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempname_accepts_string_array_scalar() {
        let _lock = REPL_FS_TEST_LOCK.lock().unwrap();
        let dir = tempdir().expect("tempdir");
        let base = dir.path().to_string_lossy().into_owned();
        let array = StringArray::new(vec![base.clone()], vec![1, 1]).expect("string array");
        let value = tempname_builtin(vec![Value::StringArray(array)]).expect("tempname");
        let path_string = String::try_from(&value).expect("string");
        assert!(
            path_string.starts_with(&base),
            "tempname should accept string scalar arrays"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempname_expands_tilde_prefix() {
        let home = home_directory().expect("home directory");
        let value = tempname_builtin(vec![Value::String("~".to_string())]).expect("tempname");
        let path_string = String::try_from(&value).expect("string");
        let path = PathBuf::from(&path_string);
        assert!(
            path.starts_with(&home),
            "tilde should expand to the home directory"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempname_preserves_relative_folder() {
        let base = "relative_temp";
        let value = tempname_builtin(vec![Value::String(base.to_string())]).expect("tempname");
        let path_string = String::try_from(&value).expect("string");
        let path = PathBuf::from(&path_string);
        assert!(
            path.is_relative(),
            "relative folder inputs should produce relative outputs"
        );
        assert!(
            path.starts_with(base),
            "result should start with the provided relative folder"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempname_rejects_too_many_arguments() {
        let err =
            tempname_builtin(vec![Value::Num(1.0), Value::Num(2.0)]).expect_err("expected error");
        assert_eq!(err.message(), TEMPNAME_ERROR_TOO_MANY_INPUTS.message);

        let err = tempname_builtin(vec![Value::Num(1.0)]).expect_err("error");
        assert_eq!(err.message(), TEMPNAME_ERROR_FOLDER_TYPE.message);

        let empty = Value::CharArray(CharArray::new_row(""));
        let err = tempname_builtin(vec![empty]).expect_err("error");
        assert_eq!(err.message(), TEMPNAME_ERROR_FOLDER_EMPTY.message);
    }
}
