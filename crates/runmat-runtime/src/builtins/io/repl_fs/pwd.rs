//! MATLAB-compatible `pwd` builtin for RunMat.

use std::env;
use std::path::Path;

use runmat_builtins::{CharArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::console::{record_console_output, ConsoleStream};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::pwd")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "pwd",
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
    notes: "Host-only operation that queries the process working folder; no GPU provider hooks are required.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::pwd")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "pwd",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "I/O builtins are not eligible for fusion; metadata registered for introspection completeness.",
};

#[runtime_builtin(
    name = "pwd",
    category = "io/repl_fs",
    summary = "Return the absolute path to the folder where RunMat is currently executing.",
    keywords = "pwd,current directory,working folder,present working directory",
    accel = "cpu",
    suppress_auto_output = true,
    builtin_path = "crate::builtins::io::repl_fs::pwd"
)]
fn pwd_builtin(args: Vec<Value>) -> Result<Value, String> {
    if !args.is_empty() {
        return Err("pwd: too many input arguments".to_string());
    }
    let current = env::current_dir()
        .map_err(|err| format!("pwd: unable to determine current directory ({err})"))?;
    emit_path_stdout(&current);
    Ok(path_to_value(&current))
}

fn path_to_value(path: &Path) -> Value {
    let text = path.to_string_lossy().into_owned();
    Value::CharArray(CharArray::new_row(&text))
}

fn emit_path_stdout(path: &Path) {
    record_console_output(ConsoleStream::Stdout, path.to_string_lossy().into_owned());
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::REPL_FS_TEST_LOCK;
    use super::*;
    use runmat_builtins::CharArray;
    use std::convert::TryFrom;
    use std::path::PathBuf;
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pwd_returns_current_directory() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = DirGuard::new();

        let expected = env::current_dir().expect("current dir");
        let value = pwd_builtin(Vec::new()).expect("pwd");
        let actual = String::try_from(&value).expect("string conversion");
        assert_eq!(actual, expected.to_string_lossy());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pwd_reflects_directory_changes() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let guard = DirGuard::new();

        let temp = tempdir().expect("tempdir");
        env::set_current_dir(temp.path()).expect("set temp dir");

        let value = pwd_builtin(Vec::new()).expect("pwd");
        let actual = String::try_from(&value).expect("string conversion");
        let actual_path = PathBuf::from(actual);
        let expected_path =
            std::fs::canonicalize(temp.path()).unwrap_or_else(|_| temp.path().to_path_buf());
        let canonical_actual =
            std::fs::canonicalize(&actual_path).unwrap_or_else(|_| actual_path.clone());
        assert_eq!(canonical_actual, expected_path);

        drop(guard);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pwd_returns_char_array_row_vector() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = DirGuard::new();

        let value = pwd_builtin(Vec::new()).expect("pwd");
        match value {
            Value::CharArray(CharArray { rows, cols, .. }) => {
                assert_eq!(rows, 1);
                assert!(cols >= 1, "expected at least one column in pwd output");
            }
            other => panic!("expected CharArray result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pwd_errors_when_arguments_provided() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = DirGuard::new();

        let err = pwd_builtin(vec![Value::Num(1.0)]).expect_err("expected error");
        assert_eq!(err, "pwd: too many input arguments");
    }
}
