//! MATLAB-compatible `ls` builtin for RunMat.

use std::collections::HashSet;
use std::env;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use glob::glob;
use runmat_builtins::{CharArray, StringArray, Value};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::{
    contains_wildcards, expand_user_path, path_to_string, sort_entries,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::console::{record_console_output, ConsoleStream};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::ls")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ls",
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
        "Host-only filesystem builtin. Providers do not participate; any GPU-resident argument is gathered before path expansion.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::ls")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ls",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "I/O builtins are excluded from fusion plans; metadata registered for introspection completeness.",
};

const BUILTIN_NAME: &str = "ls";

fn ls_error(message: impl Into<String>) -> RuntimeError {
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
    name = "ls",
    category = "io/repl_fs",
    summary = "List files and folders in the current directory or matching a wildcard pattern.",
    keywords = "ls,list files,folder contents,wildcard listing,dir",
    accel = "cpu",
    suppress_auto_output = true,
    builtin_path = "crate::builtins::io::repl_fs::ls"
)]
async fn ls_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let gathered = gather_arguments(&args).await?;
    if gathered.len() > 1 {
        return Err(ls_error("ls: too many input arguments"));
    }

    let entries = if let Some(value) = gathered.first() {
        list_from_value(value)?
    } else {
        list_current_directory()?
    };

    emit_listing_stdout(&entries);
    rows_to_char_array(&entries)
}

fn list_from_value(value: &Value) -> BuiltinResult<Vec<String>> {
    let names = patterns_from_value(value)?;
    if names.is_empty() {
        return list_current_directory();
    }

    let mut seen = HashSet::new();
    let mut combined = Vec::new();

    for pattern in names {
        let matches = list_for_pattern(&pattern)?;
        for entry in matches {
            if seen.insert(entry.clone()) {
                combined.push(entry);
            }
        }
    }

    sort_entries(&mut combined);
    Ok(combined)
}

fn list_current_directory() -> BuiltinResult<Vec<String>> {
    let cwd = env::current_dir()
        .map_err(|err| ls_error(format!("ls: unable to determine current directory ({err})")))?;
    list_directory(&cwd)
}

fn list_for_pattern(raw: &str) -> BuiltinResult<Vec<String>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return list_current_directory();
    }

    let expanded = expand_user_path(trimmed, "ls").map_err(ls_error)?;

    if contains_wildcards(&expanded) {
        list_glob_pattern(&expanded, trimmed)
    } else {
        list_path(&expanded, trimmed)
    }
}

fn list_directory(dir: &Path) -> BuiltinResult<Vec<String>> {
    let mut entries = Vec::new();
    let dir_str = path_to_string(dir);
    let read_dir = vfs::read_dir(dir)
        .map_err(|err| ls_error(format!("ls: unable to access '{dir_str}' ({err})")))?;

    for entry in read_dir {
        let name = entry.file_name().to_string_lossy();
        if name == "." || name == ".." {
            continue;
        }
        let mut display = name.into_owned();
        append_directory_suffix(&mut display, entry.is_dir());
        entries.push(display);
    }

    sort_entries(&mut entries);
    Ok(entries)
}

fn list_path(expanded: &str, original: &str) -> BuiltinResult<Vec<String>> {
    let path = PathBuf::from(expanded);
    match vfs::metadata(&path) {
        Ok(metadata) => {
            if metadata.is_dir() {
                list_directory(&path)
            } else {
                let mut text = path_to_string(&path);
                append_directory_suffix(&mut text, false);
                Ok(vec![text])
            }
        }
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(Vec::new()),
        Err(err) => Err(ls_error(format!(
            "ls: unable to access '{original}' ({err})"
        ))),
    }
}

fn list_glob_pattern(expanded: &str, original: &str) -> BuiltinResult<Vec<String>> {
    let mut entries = Vec::new();

    let matcher = glob(expanded)
        .map_err(|err| ls_error(format!("ls: invalid pattern '{original}' ({err})")))?;
    for item in matcher {
        match item {
            Ok(path) => {
                let is_dir = vfs::symlink_metadata(&path)
                    .map(|meta| meta.is_dir())
                    .unwrap_or(false);
                let mut name = path_to_string(&path);
                append_directory_suffix(&mut name, is_dir);
                entries.push(name);
            }
            Err(err) => {
                return Err(ls_error(format!(
                    "ls: unable to enumerate matches for '{original}' ({err})"
                )))
            }
        }
    }

    Ok(entries)
}

fn rows_to_char_array(rows: &[String]) -> BuiltinResult<Value> {
    if rows.is_empty() {
        let array = CharArray::new(Vec::new(), 0, 0).map_err(|e| ls_error(format!("ls: {e}")))?;
        return Ok(Value::CharArray(array));
    }

    let width = rows
        .iter()
        .map(|row| row.chars().count())
        .max()
        .unwrap_or(0);

    let mut data = Vec::with_capacity(rows.len() * width);
    for row in rows {
        let mut chars: Vec<char> = row.chars().collect();
        while chars.len() < width {
            chars.push(' ');
        }
        data.extend(chars);
    }

    let array =
        CharArray::new(data, rows.len(), width).map_err(|e| ls_error(format!("ls: {e}")))?;
    Ok(Value::CharArray(array))
}

fn emit_listing_stdout(rows: &[String]) {
    if rows.is_empty() {
        return;
    }
    let text = rows.join("\n");
    record_console_output(ConsoleStream::Stdout, text);
}

fn patterns_from_value(value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::StringArray(StringArray { data, .. }) => {
            if data.len() == 1 {
                Ok(vec![data[0].clone()])
            } else {
                Err(ls_error(
                    "ls: name must be a character vector or string scalar",
                ))
            }
        }
        Value::CharArray(chars) => {
            if chars.rows != 1 {
                return Err(ls_error(
                    "ls: name must be a character vector or string scalar",
                ));
            }
            let mut row = String::with_capacity(chars.cols);
            for c in 0..chars.cols {
                row.push(chars.data[c]);
            }
            Ok(vec![row.trim_end().to_string()])
        }
        _ => Err(ls_error(
            "ls: name must be a character vector or string scalar",
        )),
    }
}

fn append_directory_suffix(text: &mut String, is_dir: bool) {
    if is_dir {
        let sep = std::path::MAIN_SEPARATOR;
        if !text.ends_with(sep) {
            text.push(sep);
        }
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

#[cfg(test)]
pub(crate) mod tests {
    use super::super::REPL_FS_TEST_LOCK;
    use super::*;
    use runmat_builtins::CharArray;
    use runmat_filesystem::{self as fs, File};
    use tempfile::tempdir;

    fn ls_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::ls_builtin(args))
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

    fn rows_from_value(value: Value) -> Vec<String> {
        match value {
            Value::CharArray(CharArray { data, rows, cols }) => {
                let mut out = Vec::with_capacity(rows);
                for r in 0..rows {
                    let mut row = String::with_capacity(cols);
                    for c in 0..cols {
                        row.push(data[r * cols + c]);
                    }
                    out.push(row.trim_end().to_string());
                }
                out
            }
            other => panic!("expected CharArray result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ls_lists_current_directory_when_no_arguments() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let guard = DirGuard::new();

        let dir = tempdir().expect("tempdir");
        env::set_current_dir(dir.path()).expect("switch temp dir");
        File::create(dir.path().join("alpha.txt")).expect("create file");
        fs::create_dir(dir.path().join("beta")).expect("create dir");

        let value = ls_builtin(Vec::new()).expect("ls");
        let mut rows = rows_from_value(value);
        rows.sort();

        let sep = std::path::MAIN_SEPARATOR.to_string();
        assert_eq!(rows.len(), 2);
        assert!(rows.contains(&"alpha.txt".to_string()));
        assert!(rows.contains(&format!("beta{sep}")));

        drop(guard);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ls_lists_specific_directory_contents() {
        let dir = tempdir().expect("tempdir");
        File::create(dir.path().join("data.csv")).expect("create file");
        fs::create_dir(dir.path().join("nested")).expect("create dir");

        let path = dir.path().to_string_lossy().to_string();
        let value = ls_builtin(vec![Value::from(path)]).expect("ls");
        let mut rows = rows_from_value(value);
        rows.sort();

        let sep = std::path::MAIN_SEPARATOR.to_string();
        assert_eq!(rows, vec!["data.csv".to_string(), format!("nested{sep}")]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ls_handles_wildcard_patterns() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let guard = DirGuard::new();

        let dir = tempdir().expect("tempdir");
        env::set_current_dir(dir.path()).expect("switch temp dir");

        File::create("alpha.m").expect("alpha");
        File::create("beta.txt").expect("beta");
        File::create("gamma.m").expect("gamma");

        let value = ls_builtin(vec![Value::from("*.m")]).expect("ls pattern");
        let mut rows = rows_from_value(value);
        rows.sort();

        assert_eq!(rows, vec!["alpha.m".to_string(), "gamma.m".to_string()]);

        drop(guard);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ls_returns_path_for_single_file() {
        let dir = tempdir().expect("tempdir");
        let file_path = dir.path().join("report.md");
        File::create(&file_path).expect("create file");

        let value = ls_builtin(vec![Value::from(file_path.to_string_lossy().to_string())])
            .expect("ls file");
        let rows = rows_from_value(value);
        assert_eq!(rows, vec![file_path.to_string_lossy().to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ls_returns_empty_for_missing_matches() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("*.nomatch");
        let value =
            ls_builtin(vec![Value::from(path.to_string_lossy().to_string())]).expect("ls missing");
        match value {
            Value::CharArray(array) => {
                assert_eq!(array.rows, 0);
                assert_eq!(array.cols, 0);
                assert!(array.data.is_empty());
            }
            other => panic!("expected CharArray result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ls_accepts_string_scalar_input() {
        let dir = tempdir().expect("tempdir");
        File::create(dir.path().join("file.dat")).expect("create file");

        let array = StringArray::new_2d(vec![dir.path().to_string_lossy().to_string()], 1, 1)
            .expect("string array");
        let value = ls_builtin(vec![Value::StringArray(array)]).expect("ls string");
        let rows = rows_from_value(value);
        assert!(rows.iter().any(|row| row.contains("file.dat")));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ls_rejects_numeric_argument() {
        let err = ls_builtin(vec![Value::Num(1.0)]).expect_err("expected error");
        assert_eq!(
            err.message(),
            "ls: name must be a character vector or string scalar"
        );
    }
}
