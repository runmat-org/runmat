//! MATLAB-compatible `which` builtin for RunMat.
//!
//! Resolves the entity associated with a name, mirroring MATLAB search
//! semantics (workspace variables, builtins, classes, scripts, and folders)
//! and supporting `-all`, `-builtin`, `-var`, and `-file` options.

use std::collections::HashSet;
use std::path::Path;

use runmat_builtins::{builtin_functions, CharArray, Value};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::path_to_string;
use crate::builtins::common::path_search::{
    class_file_paths, class_folder_candidates, directory_candidates,
    find_all_files_with_extensions, CLASS_M_FILE_EXTENSIONS, GENERAL_FILE_EXTENSIONS,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::introspection::type_resolvers::which_type;
use crate::{
    build_runtime_error, dispatcher::gather_if_needed_async, make_cell, BuiltinResult, RuntimeError,
};

const ERROR_NOT_ENOUGH_ARGS: &str = "which: not enough input arguments";
const ERROR_TOO_MANY_ARGS: &str = "which: too many input arguments";
const ERROR_NAME_ARG: &str = "which: name must be a character vector or string scalar";
const ERROR_OPTION_ARG: &str = "which: option must be a character vector or string scalar";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::introspection::which")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "which",
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
        "Lookup runs on the host. Arguments are gathered from the GPU before evaluating the search.",
};

fn which_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("which").build()
}

fn which_flow(err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(|id| id.to_string());
    let mut builder = build_runtime_error(err.message().to_string())
        .with_builtin("which")
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn which_path<T>(result: Result<T, String>) -> BuiltinResult<T> {
    result.map_err(which_error)
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::introspection::which")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "which",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "I/O lookup; not eligible for fusion. Metadata registered for diagnostics.",
};

#[runtime_builtin(
    name = "which",
    category = "introspection",
    summary = "Identify which variable, builtin, script, class, or folder RunMat will execute for a given name.",
    keywords = "which,search path,builtin lookup,script path,variable shadowing",
    accel = "cpu",
    type_resolver(which_type),
    builtin_path = "crate::builtins::introspection::which"
)]
async fn which_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if args.is_empty() {
        return Err(which_error(ERROR_NOT_ENOUGH_ARGS));
    }

    let mut name: Option<String> = None;
    let mut options = WhichOptions::default();

    for arg in args {
        let gathered = gather_if_needed_async(&arg).await.map_err(which_flow)?;
        let text = value_to_string_scalar(&gathered).ok_or_else(|| {
            if name.is_none() {
                which_error(ERROR_NAME_ARG)
            } else {
                which_error(ERROR_OPTION_ARG)
            }
        })?;

        if looks_like_option(&text) {
            options.apply(&text)?;
        } else if name.is_none() {
            name = Some(text);
        } else {
            return Err(which_error(ERROR_TOO_MANY_ARGS));
        }
    }

    let name = name.ok_or_else(|| which_error(ERROR_NOT_ENOUGH_ARGS))?;
    let matches = search_matches(&name, &options)?;
    if matches.is_empty() {
        return Ok(Value::CharArray(CharArray::new_row(&format!(
            "'{name}' not found."
        ))));
    }

    if options.all {
        let mut cell_values = Vec::with_capacity(matches.len());
        for entry in &matches {
            cell_values.push(Value::CharArray(CharArray::new_row(entry)));
        }
        return make_cell(cell_values, matches.len(), 1).map_err(|err| {
            build_runtime_error(err)
                .with_builtin("which")
                .build()
                .into()
        });
    }

    Ok(Value::CharArray(CharArray::new_row(
        matches.first().expect("non-empty result"),
    )))
}

#[derive(Default, Debug)]
struct WhichOptions {
    all: bool,
    builtin_only: bool,
    var_only: bool,
    file_only: bool,
}

impl WhichOptions {
    fn apply(&mut self, option: &str) -> BuiltinResult<()> {
        let lowered = option.trim().to_ascii_lowercase();
        match lowered.as_str() {
            "-all" => {
                self.all = true;
                Ok(())
            }
            "-builtin" | "-built-in" => {
                let mut conflicts = Vec::new();
                if self.var_only {
                    conflicts.push("-var");
                }
                if self.file_only {
                    conflicts.push("-file");
                }
                if !conflicts.is_empty() {
                    return Err(which_error(format!(
                        "which: {}",
                        conflict_message("-builtin", &conflicts)
                    )));
                }
                self.builtin_only = true;
                Ok(())
            }
            "-var" | "-variable" => {
                let mut conflicts = Vec::new();
                if self.builtin_only {
                    conflicts.push("-builtin");
                }
                if self.file_only {
                    conflicts.push("-file");
                }
                if !conflicts.is_empty() {
                    return Err(which_error(format!(
                        "which: {}",
                        conflict_message("-var", &conflicts)
                    )));
                }
                self.var_only = true;
                Ok(())
            }
            "-file" => {
                let mut conflicts = Vec::new();
                if self.builtin_only {
                    conflicts.push("-builtin");
                }
                if self.var_only {
                    conflicts.push("-var");
                }
                if !conflicts.is_empty() {
                    return Err(which_error(format!(
                        "which: {}",
                        conflict_message("-file", &conflicts)
                    )));
                }
                self.file_only = true;
                Ok(())
            }
            other => Err(which_error(format!("which: unrecognized option '{other}'"))),
        }
    }
}

fn conflict_message(option: &str, conflicts: &[&str]) -> String {
    debug_assert!(!conflicts.is_empty());
    let joined = match conflicts.len() {
        1 => conflicts[0].to_string(),
        2 => format!("{} or {}", conflicts[0], conflicts[1]),
        _ => {
            let mut text = conflicts[..conflicts.len() - 1].join(", ");
            text.push_str(", or ");
            text.push_str(conflicts.last().unwrap());
            text
        }
    };
    format!("conflicting option '{option}'; cannot combine with {joined}")
}

fn search_matches(name: &str, options: &WhichOptions) -> BuiltinResult<Vec<String>> {
    if options.var_only {
        return Ok(variable_match(name).into_iter().collect());
    }
    if options.builtin_only {
        return Ok(builtin_matches(name));
    }
    if options.file_only {
        return search_file_like_matches(name, options.all);
    }

    let mut seen = HashSet::new();
    let mut results = Vec::new();

    if let Some(var_msg) = variable_match(name) {
        push_unique(&mut results, &mut seen, var_msg.clone());
        if !options.all {
            return Ok(results);
        }
    }

    for entry in builtin_matches(name) {
        push_unique(&mut results, &mut seen, entry.clone());
        if !options.all && !results.is_empty() {
            return Ok(results);
        }
    }

    let mut class_entries = class_matches(name)?;
    for entry in class_entries.drain(..) {
        push_unique(&mut results, &mut seen, entry.clone());
        if !options.all && !results.is_empty() {
            return Ok(results);
        }
    }

    let mut file_entries = file_matches(name)?;
    for entry in file_entries.drain(..) {
        push_unique(&mut results, &mut seen, entry.clone());
        if !options.all && !results.is_empty() {
            return Ok(results);
        }
    }

    let mut directory_entries = directory_matches(name)?;
    for entry in directory_entries.drain(..) {
        push_unique(&mut results, &mut seen, entry.clone());
        if !options.all && !results.is_empty() {
            return Ok(results);
        }
    }

    Ok(results)
}

fn search_file_like_matches(name: &str, gather_all: bool) -> BuiltinResult<Vec<String>> {
    let mut seen = HashSet::new();
    let mut results = Vec::new();

    for entry in class_matches(name)? {
        push_unique(&mut results, &mut seen, entry);
    }

    for entry in file_matches(name)? {
        push_unique(&mut results, &mut seen, entry);
        if !gather_all && !results.is_empty() {
            return Ok(results);
        }
    }

    for entry in directory_matches(name)? {
        push_unique(&mut results, &mut seen, entry);
        if !gather_all && !results.is_empty() {
            return Ok(results);
        }
    }

    Ok(results)
}

fn variable_match(name: &str) -> Option<String> {
    crate::workspace::lookup(name).map(|_| format!("'{name}' is a variable."))
}

fn builtin_matches(name: &str) -> Vec<String> {
    let lowered = name.to_ascii_lowercase();
    builtin_functions()
        .into_iter()
        .filter(|b| b.name.eq_ignore_ascii_case(&lowered))
        .map(|b| format!("built-in (RunMat builtin: {})", b.name))
        .collect()
}

fn class_matches(name: &str) -> BuiltinResult<Vec<String>> {
    let mut results = Vec::new();
    let mut seen = HashSet::new();

    for folder in which_path(class_folder_candidates(name, "which"))? {
        if folder.is_dir() {
            let text = format!("class folder: {}", canonical_path(&folder));
            push_unique(&mut results, &mut seen, text);
        }
    }

    for file in which_path(class_file_paths(
        name,
        CLASS_M_FILE_EXTENSIONS,
        "classdef",
        "which",
    ))? {
        let text = format!("classdef file: {}", canonical_path(&file));
        push_unique(&mut results, &mut seen, text);
    }

    Ok(results)
}

fn file_matches(name: &str) -> BuiltinResult<Vec<String>> {
    let mut results = Vec::new();
    let mut seen = HashSet::new();
    for file in which_path(find_all_files_with_extensions(
        name,
        GENERAL_FILE_EXTENSIONS,
        "which",
    ))? {
        if vfs::metadata(&file)
            .map(|meta| meta.is_file())
            .unwrap_or(false)
        {
            push_unique(&mut results, &mut seen, canonical_path(&file));
        }
    }
    Ok(results)
}

fn directory_matches(name: &str) -> BuiltinResult<Vec<String>> {
    let mut results = Vec::new();
    let mut seen = HashSet::new();
    for dir in which_path(directory_candidates(name, "which"))? {
        if vfs::metadata(&dir)
            .map(|meta| meta.is_dir())
            .unwrap_or(false)
        {
            push_unique(&mut results, &mut seen, canonical_path(&dir));
        }
    }
    Ok(results)
}

fn canonical_path(path: &Path) -> String {
    vfs::canonicalize(path)
        .map(|p| path_to_string(&p))
        .unwrap_or_else(|_| path_to_string(path))
}

fn value_to_string_scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::CharArray(array) if array.rows == 1 => Some(array.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        _ => None,
    }
}

fn looks_like_option(text: &str) -> bool {
    text.trim_start().starts_with('-')
}

fn push_unique(results: &mut Vec<String>, seen: &mut HashSet<String>, entry: String) {
    if seen.insert(entry.clone()) {
        results.push(entry);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use once_cell::sync::Lazy;
    use runmat_builtins::{CharArray, StringArray, Value};
    use runmat_thread_local::runmat_thread_local;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::Mutex;
    use tempfile::tempdir;

    static WHICH_TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    fn workspace_guard() -> std::sync::MutexGuard<'static, ()> {
        crate::workspace::test_guard()
    }

    fn test_guard() -> (
        std::sync::MutexGuard<'static, ()>,
        std::sync::MutexGuard<'static, ()>,
    ) {
        let workspace = workspace_guard();
        let which = WHICH_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        (workspace, which)
    }

    fn which_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::which_builtin(args))
    }

    runmat_thread_local! {
        static TEST_WORKSPACE: RefCell<HashMap<String, Value>> = RefCell::new(HashMap::new());
    }

    fn ensure_test_resolver() {
        crate::workspace::register_workspace_resolver(crate::workspace::WorkspaceResolver {
            lookup: |name| TEST_WORKSPACE.with(|slot| slot.borrow().get(name).cloned()),
            snapshot: || {
                let mut entries: Vec<(String, Value)> =
                    TEST_WORKSPACE.with(|slot| slot.borrow().clone().into_iter().collect());
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                entries
            },
            globals: || Vec::new(),
        });
    }

    fn set_workspace(entries: &[(&str, Value)]) {
        TEST_WORKSPACE.with(|slot| {
            let mut map = slot.borrow_mut();
            map.clear();
            for (name, value) in entries {
                map.insert((*name).to_string(), value.clone());
            }
        });
    }

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn which_reports_builtin() {
        let (_guard, _lock) = test_guard();
        let value = which_builtin(vec![Value::from("sin")]).expect("which");
        let text = String::try_from(&value).expect("string result");
        assert!(
            text.contains("built-in"),
            "expected builtin output, got {text}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn which_variable_search_respects_workspace() {
        let (_guard, _lock) = test_guard();
        ensure_test_resolver();
        set_workspace(&[("answer", Value::Num(42.0))]);

        let value = which_builtin(vec![Value::from("answer")]).expect("which");
        assert_eq!(String::try_from(&value).unwrap(), "'answer' is a variable.");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn which_finds_files() {
        let (_guard, _lock) = test_guard();
        let temp = tempdir().expect("tempdir");
        let script_path = temp.path().join("script.m");
        File::create(&script_path)
            .and_then(|mut file| writeln!(file, "disp('hi');"))
            .expect("write script");

        let guard = DirGuard::new();
        std::env::set_current_dir(temp.path()).expect("set temp dir");

        let value = which_builtin(vec![Value::from("script")]).expect("which");
        let text = String::try_from(&value).expect("string");
        assert!(
            text.ends_with("script.m"),
            "expected to end with script.m, got {text}"
        );

        drop(guard);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn which_all_returns_cell_array() {
        let (_guard, _lock) = test_guard();
        let value = which_builtin(vec![Value::from("sin"), Value::from("-all")]).expect("which");
        match value {
            Value::Cell(cell) => assert!(!cell.data.is_empty()),
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn which_not_found_message() {
        let (_guard, _lock) = test_guard();
        let value = which_builtin(vec![Value::from("definitely_missing")]).expect("which");
        let text = String::try_from(&value).expect("string");
        assert_eq!(text, "'definitely_missing' not found.");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn which_parses_leading_option() {
        let (_guard, _lock) = test_guard();
        let value = which_builtin(vec![Value::from("-all"), Value::from("sin")]).expect("which");
        match value {
            Value::Cell(cell) => assert!(!cell.data.is_empty()),
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn which_allows_uppercase_and_repeated_flags() {
        let (_guard, _lock) = test_guard();
        let value = which_builtin(vec![
            Value::from("-BUILTIN"),
            Value::from("-builtin"),
            Value::from("sin"),
        ])
        .expect("which");
        let text = String::try_from(&value).expect("string");
        assert!(
            text.contains("built-in"),
            "expected builtin output, got {text}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn which_conflicting_flags_error() {
        let (_guard, _lock) = test_guard();
        let err = which_builtin(vec![
            Value::from("-var"),
            Value::from("-builtin"),
            Value::from("sin"),
        ])
        .unwrap_err();
        let message = error_message(err);
        assert!(
            message.contains("conflicting option '-builtin'"),
            "unexpected error: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn which_invalid_flag_error() {
        let (_guard, _lock) = test_guard();
        let err = which_builtin(vec![Value::from("-nope"), Value::from("sin")]).unwrap_err();
        let message = error_message(err);
        assert!(
            message.contains("unrecognized option '-nope'"),
            "unexpected error: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn which_requires_name_argument() {
        let (_guard, _lock) = test_guard();
        let err = which_builtin(vec![]).unwrap_err();
        let message = error_message(err);
        assert_eq!(message, ERROR_NOT_ENOUGH_ARGS);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn which_errors_on_non_string_name() {
        let (_guard, _lock) = test_guard();
        let err = which_builtin(vec![Value::Num(4.0)]).unwrap_err();
        let message = error_message(err);
        assert_eq!(message, ERROR_NAME_ARG);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn which_errors_on_too_many_arguments() {
        let (_guard, _lock) = test_guard();
        let err = which_builtin(vec![
            Value::from("sin"),
            Value::from("cos"),
            Value::from("tan"),
        ])
        .unwrap_err();
        let message = error_message(err);
        assert_eq!(message, ERROR_TOO_MANY_ARGS);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn which_accepts_char_and_string_array_inputs() {
        let (_guard, _lock) = test_guard();
        let char_value = Value::CharArray(CharArray::new_row("sin"));
        let char_result = which_builtin(vec![char_value]).expect("which char");
        let char_text = String::try_from(&char_result).expect("string");
        assert!(
            char_text.contains("built-in"),
            "expected builtin output, got {char_text}"
        );

        let string_array = StringArray::new(vec!["sin".to_string()], vec![1]).unwrap();
        let string_result =
            which_builtin(vec![Value::StringArray(string_array)]).expect("which string array");
        let string_text = String::try_from(&string_result).expect("string");
        assert!(
            string_text.contains("built-in"),
            "expected builtin output, got {string_text}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn which_file_option_finds_directories() {
        let (_guard, _lock) = test_guard();
        let temp = tempdir().expect("tempdir");
        let subdir = temp.path().join("helpers");
        std::fs::create_dir_all(&subdir).expect("create dir");
        let guard = DirGuard::new();
        std::env::set_current_dir(temp.path()).expect("set temp dir");

        let value =
            which_builtin(vec![Value::from("-file"), Value::from("helpers")]).expect("which");
        let text = String::try_from(&value).expect("string");
        assert!(
            text.ends_with("helpers") || text.contains("/helpers") || text.contains("\\helpers"),
            "expected directory path, got {text}"
        );

        drop(guard);
    }

    struct DirGuard {
        original: PathBuf,
    }

    impl DirGuard {
        fn new() -> Self {
            let original = std::env::current_dir().expect("current dir");
            Self { original }
        }
    }

    impl Drop for DirGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.original);
        }
    }
}
