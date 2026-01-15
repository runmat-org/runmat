//! MATLAB-compatible `tempname` builtin for RunMat.

use runmat_time::system_time_now;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::UNIX_EPOCH;

use runmat_builtins::{CharArray, Value};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::{expand_user_path, path_to_string};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::gather_if_needed;

const ERR_TOO_MANY_INPUTS: &str = "tempname: too many input arguments";
const ERR_FOLDER_TYPE: &str = "tempname: folder name must be a character vector or string scalar";
const ERR_FOLDER_EMPTY: &str = "tempname: folder name must not be empty";
const ERR_TEMP_DIR_UNAVAILABLE: &str = "tempname: unable to determine temporary directory";
const ERR_UNABLE_TO_GENERATE: &str = "tempname: unable to generate a unique name";

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

#[runtime_builtin(
    name = "tempname",
    category = "io/repl_fs",
    summary = "Return a unique temporary file path.",
    keywords = "tempname,temporary file,unique name,temp directory",
    accel = "cpu",
    builtin_path = "crate::builtins::io::repl_fs::tempname"
)]
fn tempname_builtin(args: Vec<Value>) -> Result<Value, String> {
    match args.len() {
        0 => {
            let base = default_temp_directory()?;
            Ok(path_to_value(&generate_unique_path(&base)?))
        }
        1 => {
            let gathered = gather_argument(&args[0])?;
            let folder = parse_folder_argument(&gathered)?;
            Ok(path_to_value(&generate_unique_path(&folder)?))
        }
        _ => Err(ERR_TOO_MANY_INPUTS.to_string()),
    }
}

fn default_temp_directory() -> Result<PathBuf, String> {
    let path = std::env::temp_dir();
    if path.as_os_str().is_empty() {
        Err(ERR_TEMP_DIR_UNAVAILABLE.to_string())
    } else {
        Ok(path)
    }
}

fn gather_argument(value: &Value) -> Result<Value, String> {
    gather_if_needed(value).map_err(|err| format!("tempname: {err}"))
}

fn parse_folder_argument(value: &Value) -> Result<PathBuf, String> {
    let text = match value {
        Value::String(s) => {
            if s.is_empty() {
                return Err(ERR_FOLDER_EMPTY.to_string());
            }
            s.clone()
        }
        Value::CharArray(array) => {
            if array.rows != 1 {
                return Err(ERR_FOLDER_TYPE.to_string());
            }
            let collected: String = array.data.iter().collect();
            if collected.is_empty() {
                return Err(ERR_FOLDER_EMPTY.to_string());
            }
            collected
        }
        Value::StringArray(array) => {
            if array.data.len() != 1 {
                return Err(ERR_FOLDER_TYPE.to_string());
            }
            let collected = array.data[0].clone();
            if collected.is_empty() {
                return Err(ERR_FOLDER_EMPTY.to_string());
            }
            collected
        }
        _ => return Err(ERR_FOLDER_TYPE.to_string()),
    };

    let expanded = expand_user_path(&text, "tempname")?;
    if expanded.is_empty() {
        Err(ERR_FOLDER_EMPTY.to_string())
    } else {
        Ok(PathBuf::from(expanded))
    }
}

fn generate_unique_path(base: &Path) -> Result<PathBuf, String> {
    for _ in 0..MAX_ATTEMPTS {
        let token = unique_token();
        let candidate = if base.as_os_str().is_empty() {
            PathBuf::from(&token)
        } else {
            base.join(&token)
        };
        if !path_exists(&candidate) {
            return Ok(candidate);
        }
    }
    Err(ERR_UNABLE_TO_GENERATE.to_string())
}

fn unique_token() -> String {
    let now = system_time_now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let nanos = now.subsec_nanos();
    let pid = std::process::id() as u64;
    let counter = UNIQUE_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("tp{:016x}{:08x}{:08x}{:016x}", secs, nanos, pid, counter)
}

fn path_to_value(path: &Path) -> Value {
    let text = path_to_string(path);
    Value::CharArray(CharArray::new_row(&text))
}

fn path_exists(path: &Path) -> bool {
    vfs::metadata(path).is_ok()
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
        assert_eq!(parent, std::env::temp_dir().as_path());
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
        let err = tempname_builtin(vec![Value::Num(1.0), Value::Num(2.0)]).expect_err("error");
        assert_eq!(err, ERR_TOO_MANY_INPUTS);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempname_rejects_invalid_type() {
        let err = tempname_builtin(vec![Value::Num(1.0)]).expect_err("error");
        assert_eq!(err, ERR_FOLDER_TYPE);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tempname_rejects_empty_folder() {
        let empty = Value::CharArray(CharArray::new_row(""));
        let err = tempname_builtin(vec![empty]).expect_err("error");
        assert_eq!(err, ERR_FOLDER_EMPTY);
    }
}
