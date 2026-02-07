//! MATLAB-compatible `path` builtin for inspecting and updating the RunMat
//! search path.

use runmat_builtins::{CharArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::path_state::{
    current_path_string, set_path_string, PATH_LIST_SEPARATOR,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const ERROR_ARG_TYPE: &str = "path: arguments must be character vectors or string scalars";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::path")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "path",
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
    notes: "Search-path management is a host-only operation; GPU inputs are gathered before processing.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::path")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "path",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "I/O builtins are not eligible for fusion; metadata registered for introspection completeness.",
};

const BUILTIN_NAME: &str = "path";

fn path_error(message: impl Into<String>) -> RuntimeError {
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
    name = "path",
    category = "io/repl_fs",
    summary = "Query or replace the MATLAB search path used by RunMat.",
    keywords = "path,search path,matlab path,addpath,rmpath",
    accel = "cpu",
    suppress_auto_output = true,
    type_resolver(crate::builtins::io::type_resolvers::path_type),
    builtin_path = "crate::builtins::io::repl_fs::path"
)]
async fn path_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let gathered = gather_arguments(&args).await?;
    match gathered.len() {
        0 => Ok(path_value()),
        1 => set_single_argument(&gathered[0]),
        2 => set_two_arguments(&gathered[0], &gathered[1]),
        _ => Err(path_error("path: too many input arguments")),
    }
}

fn path_value() -> Value {
    char_array_value(&current_path_string())
}

fn set_single_argument(arg: &Value) -> BuiltinResult<Value> {
    let previous = current_path_string();
    let new_path = extract_text(arg)?;
    set_path_string(&new_path);
    Ok(char_array_value(&previous))
}

fn set_two_arguments(first: &Value, second: &Value) -> BuiltinResult<Value> {
    let previous = current_path_string();
    let path1 = extract_text(first)?;
    let path2 = extract_text(second)?;
    let combined = combine_paths(&path1, &path2);
    set_path_string(&combined);
    Ok(char_array_value(&previous))
}

fn combine_paths(left: &str, right: &str) -> String {
    match (left.is_empty(), right.is_empty()) {
        (true, true) => String::new(),
        (false, true) => left.to_string(),
        (true, false) => right.to_string(),
        (false, false) => {
            let mut combined = String::with_capacity(left.len() + right.len() + 1);
            combined.push_str(left);
            combined.push(PATH_LIST_SEPARATOR);
            combined.push_str(right);
            combined
        }
    }
}

fn extract_text(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(StringArray { data, .. }) => {
            if data.len() != 1 {
                Err(path_error(ERROR_ARG_TYPE))
            } else {
                Ok(data[0].clone())
            }
        }
        Value::CharArray(chars) => {
            if chars.rows != 1 {
                return Err(path_error(ERROR_ARG_TYPE));
            }
            Ok(chars.data.iter().collect())
        }
        Value::Tensor(tensor) => tensor_to_string(tensor),
        Value::GpuTensor(_) => Err(path_error(ERROR_ARG_TYPE)),
        _ => Err(path_error(ERROR_ARG_TYPE)),
    }
}

fn tensor_to_string(tensor: &Tensor) -> BuiltinResult<String> {
    if tensor.shape.len() > 2 {
        return Err(path_error(ERROR_ARG_TYPE));
    }

    let rows = tensor.rows();
    if rows > 1 {
        return Err(path_error(ERROR_ARG_TYPE));
    }

    let mut text = String::with_capacity(tensor.data.len());
    for &code in &tensor.data {
        if !code.is_finite() {
            return Err(path_error(ERROR_ARG_TYPE));
        }
        let rounded = code.round();
        if (code - rounded).abs() > 1e-6 {
            return Err(path_error(ERROR_ARG_TYPE));
        }
        let int_code = rounded as i64;
        if !(0..=0x10FFFF).contains(&int_code) {
            return Err(path_error(ERROR_ARG_TYPE));
        }
        let ch = char::from_u32(int_code as u32).ok_or_else(|| path_error(ERROR_ARG_TYPE))?;
        text.push(ch);
    }

    Ok(text)
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
    use crate::builtins::common::path_search::search_directories;
    use crate::builtins::common::path_state::set_path_string;
    use std::convert::TryFrom;
    use tempfile::tempdir;

    fn path_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::path_builtin(args))
    }

    struct PathGuard {
        previous: String,
    }

    impl PathGuard {
        fn new() -> Self {
            Self {
                previous: current_path_string(),
            }
        }
    }

    impl Drop for PathGuard {
        fn drop(&mut self) {
            set_path_string(&self.previous);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn path_returns_char_array() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let value = path_builtin(Vec::new()).expect("path");
        match value {
            Value::CharArray(CharArray { rows, .. }) => assert_eq!(rows, 1),
            other => panic!("expected CharArray, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn path_sets_new_value_and_returns_previous() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let guard = PathGuard::new();
        let previous = guard.previous.clone();

        let temp = tempdir().expect("tempdir");
        let dir_str = temp.path().to_string_lossy().into_owned();
        let new_value = Value::CharArray(CharArray::new_row(&dir_str));
        let returned = path_builtin(vec![new_value]).expect("path set");
        let returned_str = String::try_from(&returned).expect("convert");
        assert_eq!(returned_str, previous);

        let current =
            String::try_from(&path_builtin(Vec::new()).expect("path")).expect("convert current");
        assert_eq!(current, dir_str);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn path_accepts_string_scalar() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let guard = PathGuard::new();
        let previous = guard.previous.clone();

        let new_value = Value::String("runmat/path/string".to_string());
        let returned = path_builtin(vec![new_value]).expect("path set");
        let returned_str = String::try_from(&returned).expect("convert");
        assert_eq!(returned_str, previous);

        let current =
            String::try_from(&path_builtin(Vec::new()).expect("path")).expect("convert current");
        assert_eq!(current, "runmat/path/string");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn path_accepts_tensor_codes() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let guard = PathGuard::new();
        let previous = guard.previous.clone();

        let text = "tensor-path";
        let codes: Vec<f64> = text.chars().map(|ch| ch as u32 as f64).collect();
        let tensor = Tensor::new(codes, vec![1, text.len()]).expect("tensor");
        let returned = path_builtin(vec![Value::Tensor(tensor)]).expect("path set");
        let returned_str = String::try_from(&returned).expect("convert");
        assert_eq!(returned_str, previous);

        let current =
            String::try_from(&path_builtin(Vec::new()).expect("path")).expect("convert current");
        assert_eq!(current, text);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn path_combines_two_arguments() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let dir1 = tempdir().expect("dir1");
        let dir2 = tempdir().expect("dir2");
        let dir1_str = dir1.path().to_string_lossy().to_string();
        let dir2_str = dir2.path().to_string_lossy().to_string();
        let path1 = Value::CharArray(CharArray::new_row(&dir1_str));
        let path2 = Value::CharArray(CharArray::new_row(&dir2_str));
        let _returned = path_builtin(vec![path1, path2]).expect("path set");

        let current =
            String::try_from(&path_builtin(Vec::new()).expect("path")).expect("convert current");
        let expected = format!(
            "{}{sep}{}",
            dir1.path().to_string_lossy(),
            dir2.path().to_string_lossy(),
            sep = PATH_LIST_SEPARATOR
        );
        assert_eq!(current, expected);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn path_rejects_multi_row_char_array() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let chars = CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).expect("char array");
        let err = path_builtin(vec![Value::CharArray(chars)]).expect_err("expected error");
        assert_eq!(err.message(), ERROR_ARG_TYPE);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn path_rejects_multi_element_string_array() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let array = StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).expect("array");
        let err = path_builtin(vec![Value::StringArray(array)]).expect_err("expected error");
        assert_eq!(err.message(), ERROR_ARG_TYPE);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn path_rejects_invalid_argument_types() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let err = path_builtin(vec![Value::Num(1.0)]).expect_err("expected error");
        assert!(err.message().contains("path: arguments"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn path_updates_search_directories() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let temp = tempdir().expect("tempdir");
        let dir = temp.path().to_string_lossy().into_owned();
        let _ = path_builtin(vec![Value::CharArray(CharArray::new_row(&dir))]).expect("path");

        let search = search_directories("path test").expect("search directories");
        let search_strings: Vec<String> = search
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        assert!(
            search_strings.iter().any(|entry| entry == &dir),
            "search path should include newly added directory"
        );
    }
}
