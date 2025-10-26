//! MATLAB-compatible `which` builtin for RunMat.
//!
//! Resolves the entity associated with a name, mirroring MATLAB search
//! semantics (workspace variables, builtins, classes, scripts, and folders)
//! and supporting `-all`, `-builtin`, `-var`, and `-file` options.

use std::collections::HashSet;
use std::path::Path;

use runmat_builtins::{builtin_functions, CharArray, Value};
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
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{
    dispatcher::gather_if_needed, make_cell, register_builtin_fusion_spec,
    register_builtin_gpu_spec,
};

const ERROR_NOT_ENOUGH_ARGS: &str = "which: not enough input arguments";
const ERROR_TOO_MANY_ARGS: &str = "which: too many input arguments";
const ERROR_NAME_ARG: &str = "which: name must be a character vector or string scalar";
const ERROR_OPTION_ARG: &str = "which: option must be a character vector or string scalar";

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "which"
category: "io/repl_fs"
keywords: ["which", "search path", "builtin lookup", "script path", "variable shadowing"]
summary: "Identify which variable, builtin, script, class, or folder RunMat will execute for a given name."
references:
  - https://www.mathworks.com/help/matlab/ref/which.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs entirely on the host CPU. Arguments that live on the GPU are gathered before evaluating the search."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::repl_fs::which::tests"
  integration: "builtins::io::repl_fs::which::tests::which_variable_search_respects_workspace"
---

# What does the `which` function do in MATLAB / RunMat?
`which name` reports exactly which entity RunMat will call when you invoke `name`. It follows MATLAB's search order:

1. Variables in the current workspace
2. Builtin functions
3. Class folders and classdef files
4. MATLAB files on the RunMat path (`.m`, `.mlx`, etc.)
5. Folders

The builtin accepts the same flags as MATLAB: `-all`, `-builtin`, `-var`, and `-file`.

## How does the `which` function behave in MATLAB / RunMat?
- Names can be supplied as character vectors or string scalars. Calling `which` with no name raises `which: not enough input arguments`.
- `which name` without options returns the first match respecting MATLAB's precedence rules.
- `which(name, "-all")` (or `which("-all", name)`) returns a cell array with every match on the search path, in discovery order, without duplicates.
- `which(..., "-builtin")` restricts the search to builtin functions. `"-var"` restricts to workspace variables, and `"-file"` restricts the search to files, classes, and folders.
- Package-qualified names like `pkg.func` automatically map to `+pkg` folders. Class lookups recognise both `@ClassName` folders and `.m` files containing `classdef`.
- Relative paths are resolved against the current working directory. Absolute paths and paths beginning with `~` or drive letters are honoured directly.

## `which` Function GPU Execution Behaviour
`which` performs string parsing and filesystem inspection on the host CPU. If you pass GPU-resident strings (for example, `gpuArray("sin")`), RunMat gathers them automatically before evaluating the request. Results are always host-resident character arrays or cell arrays. Acceleration providers do not implement kernels for this builtin.

## GPU residency in RunMat (Do I need `gpuArray`?)
No. `which` gathers GPU arguments implicitly and never produces device-resident output. There is no benefit in moving strings to the GPU before calling `which`.

## Examples of using the `which` function in MATLAB / RunMat

### Finding a built-in function's implementation
```matlab
which("sin")
```
Expected output:
```matlab
built-in (RunMat builtin: sin)
```

### Checking if a workspace variable shadows a builtin
```matlab
answer = 42;
which("answer")
```
Expected output:
```matlab
'answer' is a variable.
```

### Listing all matches on the path
```matlab
which("sum", "-all")
```
Expected output (example):
```matlab
{
    [1,1] = built-in (RunMat builtin: sum)
    [2,1] = /Users/alex/runmat/stdlib/sum.m
}
```

### Locating a script or function file
```matlab
which("helpers/process_data")
```
Expected output:
```matlab
/Users/alex/projects/runmat/helpers/process_data.m
```

### Restricting the search to variables
```matlab
which("-var", "velocity")
```
Expected output (if the variable exists):
```matlab
'velocity' is a variable.
```

### Restricting the search to files
```matlab
which("fft", "-file")
```
Expected output (example):
```matlab
/Users/alex/runmat/overrides/fft.m
```

## FAQ
- **What happens when nothing is found?** `which` returns the character vector `'<name>' not found.` just like MATLAB.
- **Are method lookups supported?** Methods defined via `@Class` folders or `classdef` files are discovered through the class search. Package-qualified methods are supported.
- **Does `which` canonicalise paths?** Yes. RunMat reports canonical absolute paths where possible; when canonicalisation fails, the original path is returned.
- **Can I combine `-all` with other options?** Yes. For example, `which("plot", "-all", "-file")` lists every file-based implementation without reporting builtins or variables.
- **Does the search respect `RUNMAT_PATH` / `MATLABPATH`?** Yes. The directory list mirrors the logic used by other REPL filesystem builtins.
- **What about Simulink models or Java classes?** File-based matches with supported extensions (`.slx`, `.mdl`, `.class`, etc.) are reported when present on the path.
- **Are duplicate results filtered?** Yes. The first occurrence of each unique path is returned.
- **Is the lookup case sensitive?** No. Matching is case-insensitive on all platforms, following MATLAB semantics.
- **Does `which` gather GPU values?** Yes. GPU-resident arguments are automatically gathered before the search begins.

## See Also
[exist](./exist), [ls](./ls), [dir](./dir), [copyfile](./copyfile)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/io/repl_fs/which.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/which.rs)
- Found an issue? [Open a GitHub ticket](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

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

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "which",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "I/O lookup; not eligible for fusion. Metadata registered for diagnostics.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("which", DOC_MD);

#[runtime_builtin(
    name = "which",
    category = "io/repl_fs",
    summary = "Identify which variable, builtin, script, class, or folder RunMat will execute for a given name.",
    keywords = "which,search path,builtin lookup,script path,variable shadowing",
    accel = "cpu"
)]
fn which_builtin(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err(ERROR_NOT_ENOUGH_ARGS.to_string());
    }

    let mut name: Option<String> = None;
    let mut options = WhichOptions::default();

    for arg in args {
        let gathered = gather_if_needed(&arg).map_err(|e| format!("which: {e}"))?;
        let text = value_to_string_scalar(&gathered).ok_or_else(|| {
            if name.is_none() {
                ERROR_NAME_ARG.to_string()
            } else {
                ERROR_OPTION_ARG.to_string()
            }
        })?;

        if looks_like_option(&text) {
            options
                .apply(&text)
                .map_err(|msg| format!("which: {msg}"))?;
        } else if name.is_none() {
            name = Some(text);
        } else {
            return Err(ERROR_TOO_MANY_ARGS.to_string());
        }
    }

    let name = name.ok_or_else(|| ERROR_NOT_ENOUGH_ARGS.to_string())?;
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
        return make_cell(cell_values, matches.len(), 1).map_err(|e| format!("which: {e}"));
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
    fn apply(&mut self, option: &str) -> Result<(), String> {
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
                    return Err(conflict_message("-builtin", &conflicts));
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
                    return Err(conflict_message("-var", &conflicts));
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
                    return Err(conflict_message("-file", &conflicts));
                }
                self.file_only = true;
                Ok(())
            }
            other => Err(format!("unrecognized option '{other}'")),
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

fn search_matches(name: &str, options: &WhichOptions) -> Result<Vec<String>, String> {
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

fn search_file_like_matches(name: &str, gather_all: bool) -> Result<Vec<String>, String> {
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

fn class_matches(name: &str) -> Result<Vec<String>, String> {
    let mut results = Vec::new();
    let mut seen = HashSet::new();

    for folder in class_folder_candidates(name, "which")? {
        if folder.is_dir() {
            let text = format!("class folder: {}", canonical_path(&folder));
            push_unique(&mut results, &mut seen, text);
        }
    }

    for file in class_file_paths(name, CLASS_M_FILE_EXTENSIONS, "classdef", "which")? {
        let text = format!("classdef file: {}", canonical_path(&file));
        push_unique(&mut results, &mut seen, text);
    }

    Ok(results)
}

fn file_matches(name: &str) -> Result<Vec<String>, String> {
    let mut results = Vec::new();
    let mut seen = HashSet::new();
    for file in find_all_files_with_extensions(name, GENERAL_FILE_EXTENSIONS, "which")? {
        if file.is_file() {
            push_unique(&mut results, &mut seen, canonical_path(&file));
        }
    }
    Ok(results)
}

fn directory_matches(name: &str) -> Result<Vec<String>, String> {
    let mut results = Vec::new();
    let mut seen = HashSet::new();
    for dir in directory_candidates(name, "which")? {
        if dir.is_dir() {
            push_unique(&mut results, &mut seen, canonical_path(&dir));
        }
    }
    Ok(results)
}

fn canonical_path(path: &Path) -> String {
    std::fs::canonicalize(path)
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
mod tests {
    use super::super::REPL_FS_TEST_LOCK;
    use super::*;
    use once_cell::sync::OnceCell;
    use runmat_builtins::{CharArray, StringArray, Value};
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;
    use tempfile::tempdir;

    thread_local! {
        static TEST_WORKSPACE: RefCell<HashMap<String, Value>> = RefCell::new(HashMap::new());
    }

    fn ensure_test_resolver() {
        static INIT: OnceCell<()> = OnceCell::new();
        INIT.get_or_init(|| {
            crate::workspace::register_workspace_resolver(crate::workspace::WorkspaceResolver {
                lookup: |name| TEST_WORKSPACE.with(|slot| slot.borrow().get(name).cloned()),
                snapshot: || {
                    let mut entries: Vec<(String, Value)> =
                        TEST_WORKSPACE.with(|slot| slot.borrow().clone().into_iter().collect());
                    entries.sort_by(|a, b| a.0.cmp(&b.0));
                    entries
                },
            });
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

    #[test]
    fn which_reports_builtin() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let value = which_builtin(vec![Value::from("sin")]).expect("which");
        let text = String::try_from(&value).expect("string result");
        assert!(
            text.contains("built-in"),
            "expected builtin output, got {text}"
        );
    }

    #[test]
    fn which_variable_search_respects_workspace() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        ensure_test_resolver();
        set_workspace(&[("answer", Value::Num(42.0))]);

        let value = which_builtin(vec![Value::from("answer")]).expect("which");
        assert_eq!(String::try_from(&value).unwrap(), "'answer' is a variable.");
    }

    #[test]
    fn which_finds_files() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
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

    #[test]
    fn which_all_returns_cell_array() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let value = which_builtin(vec![Value::from("sin"), Value::from("-all")]).expect("which");
        match value {
            Value::Cell(cell) => assert!(cell.data.len() >= 1),
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[test]
    fn which_not_found_message() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let value = which_builtin(vec![Value::from("definitely_missing")]).expect("which");
        let text = String::try_from(&value).expect("string");
        assert_eq!(text, "'definitely_missing' not found.");
    }

    #[test]
    fn which_parses_leading_option() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let value = which_builtin(vec![Value::from("-all"), Value::from("sin")]).expect("which");
        match value {
            Value::Cell(cell) => assert!(cell.data.len() >= 1),
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[test]
    fn which_allows_uppercase_and_repeated_flags() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
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

    #[test]
    fn which_conflicting_flags_error() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let err = which_builtin(vec![
            Value::from("-var"),
            Value::from("-builtin"),
            Value::from("sin"),
        ])
        .unwrap_err();
        assert!(
            err.contains("conflicting option '-builtin'"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn which_invalid_flag_error() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let err = which_builtin(vec![Value::from("-nope"), Value::from("sin")]).unwrap_err();
        assert!(
            err.contains("unrecognized option '-nope'"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn which_requires_name_argument() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let err = which_builtin(vec![]).unwrap_err();
        assert_eq!(err, ERROR_NOT_ENOUGH_ARGS);
    }

    #[test]
    fn which_errors_on_non_string_name() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let err = which_builtin(vec![Value::Num(4.0)]).unwrap_err();
        assert_eq!(err, ERROR_NAME_ARG);
    }

    #[test]
    fn which_errors_on_too_many_arguments() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let err = which_builtin(vec![
            Value::from("sin"),
            Value::from("cos"),
            Value::from("tan"),
        ])
        .unwrap_err();
        assert_eq!(err, ERROR_TOO_MANY_ARGS);
    }

    #[test]
    fn which_accepts_char_and_string_array_inputs() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
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

    #[test]
    fn which_file_option_finds_directories() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
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

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = crate::builtins::common::test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
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
