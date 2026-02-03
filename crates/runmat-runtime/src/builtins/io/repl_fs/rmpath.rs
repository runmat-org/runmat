//! MATLAB-compatible `rmpath` builtin for manipulating the RunMat search path.

use runmat_builtins::{CharArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::{expand_user_path, path_to_string};
use crate::builtins::common::path_state::{
    current_path_segments, current_path_string, set_path_string, PATH_LIST_SEPARATOR,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

use runmat_filesystem as vfs;
use std::collections::HashSet;
use std::env;
use std::path::{Component, Path, PathBuf};

const ERROR_ARG_TYPE: &str =
    "rmpath: folder names must be character vectors, string scalars, string arrays, or cell arrays of character vectors";
const ERROR_TOO_FEW_ARGS: &str = "rmpath: at least one folder must be specified";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::rmpath")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "rmpath",
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
    notes: "Search-path manipulation is a host-only operation; GPU inputs are gathered before processing.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::rmpath")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "rmpath",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "IO builtins are not eligible for fusion; metadata registered for completeness.",
};

const BUILTIN_NAME: &str = "rmpath";

fn rmpath_error(message: impl Into<String>) -> RuntimeError {
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
    name = "rmpath",
    category = "io/repl_fs",
    summary = "Remove folders from the MATLAB search path used by RunMat.",
    keywords = "rmpath,search path,matlab path,remove folder",
    accel = "cpu",
    suppress_auto_output = true,
    type_resolver(crate::builtins::io::type_resolvers::rmpath_type),
    builtin_path = "crate::builtins::io::repl_fs::rmpath"
)]
async fn rmpath_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if args.is_empty() {
        return Err(rmpath_error(ERROR_TOO_FEW_ARGS));
    }

    let gathered = gather_arguments(&args).await?;
    let directories = parse_directories(&gathered).await?;

    let previous = current_path_string();
    apply_rmpath(directories)?;
    Ok(char_array_value(&previous))
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

async fn parse_directories(args: &[Value]) -> BuiltinResult<Vec<String>> {
    let mut directories = Vec::new();
    for value in args {
        collect_strings(value, &mut directories).await?;
    }

    if directories.is_empty() {
        return Err(rmpath_error(ERROR_TOO_FEW_ARGS));
    }

    let mut resolved = Vec::new();
    for token in directories {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            continue;
        }
        resolved.extend(split_path_list(trimmed));
    }

    if resolved.is_empty() {
        return Err(rmpath_error(ERROR_TOO_FEW_ARGS));
    }

    Ok(resolved)
}

#[async_recursion::async_recursion(?Send)]
async fn collect_strings(value: &Value, output: &mut Vec<String>) -> BuiltinResult<()> {
    match value {
        Value::String(text) => {
            output.push(text.clone());
            Ok(())
        }
        Value::StringArray(StringArray { data, .. }) => {
            for entry in data {
                output.push(entry.clone());
            }
            Ok(())
        }
        Value::CharArray(chars) => {
            if chars.rows == 1 {
                output.push(chars.data.iter().collect());
                return Ok(());
            }
            for row in 0..chars.rows {
                let mut line = String::with_capacity(chars.cols);
                for col in 0..chars.cols {
                    line.push(chars.data[row * chars.cols + col]);
                }
                output.push(line.trim_end().to_string());
            }
            Ok(())
        }
        Value::Tensor(tensor) => {
            output.push(tensor_to_string(tensor)?);
            Ok(())
        }
        Value::Cell(cell) => {
            for ptr in &cell.data {
                let inner = (*ptr).clone();
                let gathered = gather_if_needed_async(&inner)
                    .await
                    .map_err(map_control_flow)?;
                collect_strings(&gathered, output).await?;
            }
            Ok(())
        }
        Value::GpuTensor(_) => Err(rmpath_error(ERROR_ARG_TYPE)),
        _ => Err(rmpath_error(ERROR_ARG_TYPE)),
    }
}

fn split_path_list(text: &str) -> Vec<String> {
    text.split(PATH_LIST_SEPARATOR)
        .map(|segment| segment.trim())
        .filter(|segment| !segment.is_empty())
        .map(|segment| segment.to_string())
        .collect()
}

fn apply_rmpath(directories: Vec<String>) -> BuiltinResult<()> {
    let mut segments = current_path_segments();
    let mut changed = false;
    let mut seen = HashSet::new();

    for raw in directories {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }

        let dedup_key = path_identity(trimmed);
        if !seen.insert(dedup_key) {
            continue;
        }

        if remove_directory(&mut segments, trimmed)? {
            changed = true;
        }
    }

    if changed {
        let new_path = if segments.is_empty() {
            String::new()
        } else {
            join_segments(&segments)
        };
        set_path_string(&new_path);
    }

    Ok(())
}

fn remove_directory(segments: &mut Vec<String>, raw: &str) -> BuiltinResult<bool> {
    let direct_identity = path_identity(raw);
    let before = segments.len();
    segments.retain(|entry| path_identity(entry) != direct_identity);
    if segments.len() != before {
        return Ok(true);
    }

    let expanded = expand_user_path(raw, "rmpath").map_err(rmpath_error)?;
    let path = Path::new(&expanded);
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()
            .map_err(|_| rmpath_error("rmpath: unable to resolve current directory"))?
            .join(path)
    };
    let normalized = normalize_pathbuf(&joined);
    let canonical = path_to_string(&normalized);

    let canonical_identity = path_identity(&canonical);
    let before = segments.len();
    segments.retain(|entry| path_identity(entry) != canonical_identity);
    if segments.len() != before {
        return Ok(true);
    }

    match vfs::metadata(&normalized) {
        Ok(meta) => {
            if !meta.is_dir() {
                Err(rmpath_error(format!("rmpath: '{raw}' is not a folder")))
            } else {
                Err(rmpath_error(format!(
                    "rmpath: folder '{raw}' not on search path"
                )))
            }
        }
        Err(_) => Err(rmpath_error(format!("rmpath: folder '{raw}' not found"))),
    }
}

fn normalize_pathbuf(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Normal(part) => normalized.push(part),
        }
    }
    if normalized.as_os_str().is_empty() {
        path.to_path_buf()
    } else {
        normalized
    }
}

fn tensor_to_string(tensor: &Tensor) -> BuiltinResult<String> {
    if tensor.shape.len() > 2 {
        return Err(rmpath_error(ERROR_ARG_TYPE));
    }
    if tensor.rows() > 1 {
        return Err(rmpath_error(ERROR_ARG_TYPE));
    }
    let mut text = String::with_capacity(tensor.data.len());
    for &code in &tensor.data {
        if !code.is_finite() {
            return Err(rmpath_error(ERROR_ARG_TYPE));
        }
        let rounded = code.round();
        if (code - rounded).abs() > 1e-6 {
            return Err(rmpath_error(ERROR_ARG_TYPE));
        }
        let int_code = rounded as i64;
        if !(0..=0x10FFFF).contains(&int_code) {
            return Err(rmpath_error(ERROR_ARG_TYPE));
        }
        let ch = char::from_u32(int_code as u32).ok_or_else(|| rmpath_error(ERROR_ARG_TYPE))?;
        text.push(ch);
    }
    Ok(text)
}

fn path_identity(path: &str) -> String {
    #[cfg(windows)]
    {
        path.replace('/', "\\").to_ascii_lowercase()
    }
    #[cfg(not(windows))]
    {
        path.to_string()
    }
}

fn join_segments(segments: &[String]) -> String {
    let mut joined = String::new();
    for (idx, segment) in segments.iter().enumerate() {
        if idx > 0 {
            joined.push(PATH_LIST_SEPARATOR);
        }
        joined.push_str(segment);
    }
    joined
}

fn char_array_value(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::REPL_FS_TEST_LOCK;
    use super::*;
    use crate::builtins::common::path_state::{current_path_segments, set_path_string};
    use runmat_builtins::CellArray;
    use std::convert::TryFrom;
    use tempfile::tempdir;

    fn rmpath_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::rmpath_builtin(args))
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

    fn canonical(dir: &Path) -> String {
        let normalized = normalize_pathbuf(dir);
        path_to_string(&normalized)
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmpath_removes_single_entry() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let target = tempdir().expect("target");
        let keep = tempdir().expect("keep");
        let target_str = canonical(target.path());
        let keep_str = canonical(keep.path());
        let combined = format!(
            "{target}{sep}{keep}",
            target = target_str,
            keep = keep_str,
            sep = PATH_LIST_SEPARATOR
        );
        set_path_string(&combined);

        let returned = rmpath_builtin(vec![Value::String(target_str.clone())]).expect("rmpath");
        let returned_str = String::try_from(&returned).expect("convert");
        assert_eq!(returned_str, combined);

        let segments = current_path_segments();
        assert_eq!(segments, vec![keep_str]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmpath_splits_path_list_argument() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let dir1 = tempdir().expect("dir1");
        let dir2 = tempdir().expect("dir2");
        let dir3 = tempdir().expect("dir3");

        let str1 = canonical(dir1.path());
        let str2 = canonical(dir2.path());
        let str3 = canonical(dir3.path());

        let combined = format!(
            "{first}{sep}{second}{sep}{third}",
            first = str1,
            second = str2,
            third = str3,
            sep = PATH_LIST_SEPARATOR
        );
        set_path_string(&combined);

        let to_remove = format!("{str1}{sep}{str2}", sep = PATH_LIST_SEPARATOR);
        rmpath_builtin(vec![Value::String(to_remove)]).expect("rmpath");

        let segments = current_path_segments();
        assert_eq!(segments, vec![str3]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmpath_accepts_string_containers() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let dir1 = tempdir().expect("dir1");
        let dir2 = tempdir().expect("dir2");

        let str1 = canonical(dir1.path());
        let str2 = canonical(dir2.path());
        set_path_string(&format!("{str1}{sep}{str2}", sep = PATH_LIST_SEPARATOR));

        let strings = StringArray::new(vec![str1.clone()], vec![1, 1]).expect("string array");
        let chars = CharArray::new_row(str2.as_str());
        let args = vec![Value::StringArray(strings), Value::CharArray(chars)];
        rmpath_builtin(args).expect("rmpath");

        let segments = current_path_segments();
        assert!(segments.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmpath_supports_cell_array() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let dir = tempdir().expect("dir");
        let str = canonical(dir.path());
        set_path_string(&str);

        let cell = CellArray::new(vec![Value::String(str.clone())], 1, 1).expect("cell");
        rmpath_builtin(vec![Value::Cell(cell)]).expect("rmpath");
        let segments = current_path_segments();
        assert!(segments.is_empty());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmpath_errors_on_missing_folder() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        set_path_string("");
        let err = rmpath_builtin(vec![Value::String("this/folder/does/not/exist".into())])
            .expect_err("expected error");
        assert!(err.message().contains("not found"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmpath_errors_when_folder_not_on_path() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let dir = tempdir().expect("dir");
        let str = canonical(dir.path());
        set_path_string("");

        let err = rmpath_builtin(vec![Value::String(str.clone())]).expect_err("expected error");
        assert!(err.message().contains("not on search path"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rmpath_returns_previous_path() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let dir = tempdir().expect("dir");
        let str = canonical(dir.path());
        set_path_string(&str);

        let returned = rmpath_builtin(vec![Value::String(str.clone())]).expect("rmpath");
        let returned_str = String::try_from(&returned).expect("string");
        assert_eq!(returned_str, str);
    }
}
