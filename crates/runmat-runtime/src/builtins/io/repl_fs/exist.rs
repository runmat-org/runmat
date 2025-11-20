//! MATLAB-compatible `exist` builtin for RunMat.
//!
//! Mirrors MATLAB semantics for checking whether a variable, file, folder,
//! builtin, class, or other entity is available in the current session.

use runmat_builtins::{builtin_functions, lookup_method, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::contains_wildcards;
use crate::builtins::common::path_search::{
    class_file_exists as path_class_file_exists,
    class_folder_candidates as path_class_folder_candidates,
    directory_candidates as path_directory_candidates,
    find_file_with_extensions as path_find_file_with_extensions, CLASS_M_FILE_EXTENSIONS,
    GENERAL_FILE_EXTENSIONS, LIB_EXTENSIONS, MEX_EXTENSIONS, PCODE_EXTENSIONS, SIMULINK_EXTENSIONS,
    THUNK_EXTENSIONS,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{
    dispatcher::gather_if_needed, register_builtin_fusion_spec, register_builtin_gpu_spec,
};

const ERROR_NAME_ARG: &str = "exist: name must be a character vector or string scalar";
const ERROR_TYPE_ARG: &str = "exist: type must be a character vector or string scalar";
const ERROR_INVALID_TYPE: &str = "exist: invalid type. Type must be one of 'var', 'variable', 'file', 'dir', 'directory', 'folder', 'builtin', 'built-in', 'class', 'handle', 'method', 'mex', 'pcode', 'simulink', 'thunk', 'lib', 'library', or 'java'";

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "exist"
category: "io/repl_fs"
keywords: ["exist", "file exists", "variable exists", "folder exists", "builtin exists"]
summary: "Determine whether a variable, file, folder, built-in function, or class exists and return MATLAB-compatible identifiers."
references:
  - https://www.mathworks.com/help/matlab/ref/exist.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs entirely on the CPU. When arguments are gpuArray values RunMat gathers them to the host before evaluating the query."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::repl_fs::exist::tests"
  integration: "builtins::io::repl_fs::exist::tests::exist_detects_workspace_variables"
---

# What does the `exist` function do in MATLAB / RunMat?
`exist(name)` returns an integer code that identifies which kind of item named `name` is currently available. The codes match MATLAB:

| Code | Meaning                                   |
| ---- | ----------------------------------------- |
| 0    | Item not found                            |
| 1    | Variable in the active workspace          |
| 2    | File (including scripts and most data files) |
| 3    | MEX-file                                  |
| 4    | Simulink® model (`.slx` / `.mdl`)         |
| 5    | Built-in MATLAB function                  |
| 6    | P-code (`.p` / `.pp`)                     |
| 7    | Folder                                    |
| 8    | Class definition                          |

Pass a second argument to restrict the lookup (for example, `'file'`, `'dir'`, `'builtin'`, `'class'`, `'mex'`, `'pcode'`, `'method'`, or `'handle'`). The builtin mirrors MATLAB semantics for string handling, search order, and numeric return values.

## How does the `exist` function behave in MATLAB / RunMat?
- Searches follow MATLAB's precedence: variables first, then built-ins, classes, compiled code (MEX/P-code), standard files, and finally folders.
- When you omit the type, `exist` returns the first match respecting this precedence.
- String handling matches MATLAB: `name` and the optional `type` must be character vectors or string scalars. String arrays must contain exactly one element.
- Paths expand `~` to the user's home folder. Relative paths resolve against the current working directory, and RunMat honours `RUNMAT_PATH` / `MATLABPATH` entries when searching for scripts and classes.
- Package-qualified names such as `pkg.func` and `pkg.Class` map to `+pkg` folders automatically. Class queries also recognise `@ClassName` folders and `.m` files that contain `classdef`.
- GPU-resident arguments (for example, `gpuArray("script")`) are gathered automatically, so callers never need to move text back to the CPU manually.
- Unsupported second-argument types surface a MATLAB-compatible error listing the accepted keywords.

## `exist` Function GPU Execution Behaviour
The builtin runs entirely on the host CPU. If any argument lives on the GPU, RunMat gathers it to host memory before performing the lookup. Providers do not implement dedicated hooks for `exist`, and the result is always returned as a host-resident double scalar. Documented behaviour is identical regardless of whether acceleration is enabled.

## Examples of using the `exist` function in MATLAB / RunMat

### How to check if a workspace variable exists
```matlab
alpha = 3.14;
status = exist("alpha", "var");
```
Expected output:
```matlab
status =
     1
```

### How to determine if a built-in function is available
```matlab
code = exist("sin");
```
Expected output:
```matlab
code =
     5
```

### How to test whether an M-file is on the MATLAB path
```matlab
status = exist("utilities/process_data", "file");
```
Expected output (assuming the file exists):
```matlab
status =
     2
```

### How to verify a folder exists before creating logs
```matlab
mkdir("logs");
status = exist("logs", "dir");
```
Expected output:
```matlab
status =
     7
```

### How to detect class definitions in packages
```matlab
status = exist("pkg.Widget", "class");
```
Expected output:
```matlab
status =
     8
```

### How to inspect compiled MEX fallbacks
```matlab
if exist("fastfft", "mex")
    disp("Using compiled fastfft.");
else
    disp("Falling back to MATLAB implementation.");
end
```
Expected output:
```matlab
% Prints which implementation will be executed based on the presence of fastfft.mex*.
```

## GPU residency in RunMat (Do I need `gpuArray`?)
No. `exist` gathers any GPU-resident string inputs transparently. The lookup, file system checks, and return value are always host-side operations. Keeping text on the GPU offers no performance benefits, so you can pass regular character vectors or string scalars.

## FAQ
- **Does `exist` support wildcards?** No. Strings containing `*` or `?` return `0`, matching MATLAB behaviour.
- **What about Simulink models or Java classes?** RunMat reports `4` for `.slx` / `.mdl` files. Java class detection is not implemented yet and currently returns `0`, mirroring MATLAB when Java support is unavailable.
- **How are packages handled?** Names like `pkg.func` translate to `+pkg/func.m`. Class queries additionally search `+pkg/@Class` folders.
- **Is the search path configurable?** Yes. RunMat honours the current working directory plus any folders listed in `RUNMAT_PATH` or `MATLABPATH` (path separator aware).
- **What's the search precedence when multiple items share a name?** Variables have highest priority, followed by built-ins, classes, compiled code (MEX/P-code), standard files, and folders last—exactly like MATLAB.
- **What happens with unsupported type keywords?** The builtin raises `exist: invalid type...` describing the accepted tokens, matching MATLAB's diagnostic.
- **Do handle queries work?** When the first argument is a RunMat handle object, `exist(handle, "handle")` returns `1`. Numeric graphics handles are not yet supported and return `0`.

## See Also
[dir](./dir), [ls](./ls), [copyfile](./copyfile), [pwd](./pwd)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/io/repl_fs/exist.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/exist.rs)
- Found an issue? [Open a GitHub ticket](https://github.com/runmat-org/runmat/issues/new/choose) with steps to reproduce.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "exist",
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
    notes: "Filesystem and workspace lookup run on the host; arguments are gathered from the GPU when necessary.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "exist",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "I/O builtins are not eligible for fusion; metadata registered for completeness.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("exist", DOC_MD);

#[runtime_builtin(
    name = "exist",
    category = "io/repl_fs",
    summary = "Determine whether a variable, file, folder, built-in, or class exists.",
    keywords = "exist,file,dir,var,builtin,class",
    accel = "cpu"
)]
fn exist_builtin(name: Value, rest: Vec<Value>) -> Result<Value, String> {
    if rest.len() > 1 {
        return Err("exist: too many input arguments".to_string());
    }

    let name_host = gather_if_needed(&name).map_err(|err| format!("exist: {err}"))?;
    let type_value = rest
        .first()
        .map(|value| gather_if_needed(value).map_err(|err| format!("exist: {err}")))
        .transpose()?;

    let query = type_value
        .as_ref()
        .map(parse_type_argument)
        .transpose()?
        .unwrap_or(ExistQuery::Any);

    let result = match query {
        ExistQuery::Handle => exist_handle(&name_host),
        _ => {
            let text = value_to_string(&name_host).ok_or_else(|| ERROR_NAME_ARG.to_string())?;
            exist_for_query(&text, query)?
        }
    };

    Ok(Value::Num(result.code()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExistQuery {
    Any,
    Var,
    File,
    Dir,
    Builtin,
    Class,
    Mex,
    Pcode,
    Method,
    Handle,
    Simulink,
    Thunk,
    Lib,
    Java,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExistResultKind {
    NotFound,
    Variable,
    File,
    Mex,
    Simulink,
    Builtin,
    Pcode,
    Directory,
    Class,
}

impl ExistResultKind {
    fn code(self) -> f64 {
        match self {
            ExistResultKind::NotFound => 0.0,
            ExistResultKind::Variable => 1.0,
            ExistResultKind::File => 2.0,
            ExistResultKind::Mex => 3.0,
            ExistResultKind::Simulink => 4.0,
            ExistResultKind::Builtin => 5.0,
            ExistResultKind::Pcode => 6.0,
            ExistResultKind::Directory => 7.0,
            ExistResultKind::Class => 8.0,
        }
    }
}

fn parse_type_argument(value: &Value) -> Result<ExistQuery, String> {
    let text = value_to_string(value).ok_or_else(|| ERROR_TYPE_ARG.to_string())?;
    match text.trim().to_ascii_lowercase().as_str() {
        "" => Ok(ExistQuery::Any),
        "var" | "variable" => Ok(ExistQuery::Var),
        "file" => Ok(ExistQuery::File),
        "dir" | "directory" | "folder" => Ok(ExistQuery::Dir),
        "builtin" | "built-in" => Ok(ExistQuery::Builtin),
        "class" => Ok(ExistQuery::Class),
        "mex" => Ok(ExistQuery::Mex),
        "pcode" | "p" => Ok(ExistQuery::Pcode),
        "handle" => Ok(ExistQuery::Handle),
        "method" => Ok(ExistQuery::Method),
        "simulink" => Ok(ExistQuery::Simulink),
        "thunk" => Ok(ExistQuery::Thunk),
        "lib" | "library" => Ok(ExistQuery::Lib),
        "java" => Ok(ExistQuery::Java),
        _ => Err(ERROR_INVALID_TYPE.to_string()),
    }
}

fn exist_for_query(name: &str, query: ExistQuery) -> Result<ExistResultKind, String> {
    if contains_wildcards(name) {
        return Ok(ExistResultKind::NotFound);
    }

    match query {
        ExistQuery::Any => evaluate_default(name),
        ExistQuery::Var => Ok(if variable_exists(name) {
            ExistResultKind::Variable
        } else {
            ExistResultKind::NotFound
        }),
        ExistQuery::Builtin => Ok(if builtin_exists(name) {
            ExistResultKind::Builtin
        } else {
            ExistResultKind::NotFound
        }),
        ExistQuery::Class => Ok(if class_exists(name)? {
            ExistResultKind::Class
        } else {
            ExistResultKind::NotFound
        }),
        ExistQuery::Dir => Ok(if directory_exists(name)? {
            ExistResultKind::Directory
        } else {
            ExistResultKind::NotFound
        }),
        ExistQuery::File => Ok(detect_file_kind(name)?.unwrap_or(ExistResultKind::NotFound)),
        ExistQuery::Mex => Ok(
            if path_find_file_with_extensions(name, MEX_EXTENSIONS, "exist")?.is_some() {
                ExistResultKind::Mex
            } else {
                ExistResultKind::NotFound
            },
        ),
        ExistQuery::Pcode => Ok(
            if path_find_file_with_extensions(name, PCODE_EXTENSIONS, "exist")?.is_some() {
                ExistResultKind::Pcode
            } else {
                ExistResultKind::NotFound
            },
        ),
        ExistQuery::Method => Ok(if method_exists(name) {
            ExistResultKind::Builtin
        } else {
            ExistResultKind::NotFound
        }),
        ExistQuery::Simulink => Ok(
            if path_find_file_with_extensions(name, SIMULINK_EXTENSIONS, "exist")?.is_some() {
                ExistResultKind::Simulink
            } else {
                ExistResultKind::NotFound
            },
        ),
        ExistQuery::Thunk => Ok(
            if path_find_file_with_extensions(name, THUNK_EXTENSIONS, "exist")?.is_some() {
                ExistResultKind::File
            } else {
                ExistResultKind::NotFound
            },
        ),
        ExistQuery::Lib => Ok(
            if path_find_file_with_extensions(name, LIB_EXTENSIONS, "exist")?.is_some() {
                ExistResultKind::File
            } else {
                ExistResultKind::NotFound
            },
        ),
        ExistQuery::Java => Ok(ExistResultKind::NotFound),
        ExistQuery::Handle => unreachable!("handle queries handled separately"),
    }
}

fn evaluate_default(name: &str) -> Result<ExistResultKind, String> {
    if variable_exists(name) {
        return Ok(ExistResultKind::Variable);
    }
    if builtin_exists(name) {
        return Ok(ExistResultKind::Builtin);
    }
    if class_exists(name)? {
        return Ok(ExistResultKind::Class);
    }
    if let Some(kind) = detect_file_kind(name)? {
        return Ok(kind);
    }
    if directory_exists(name)? {
        return Ok(ExistResultKind::Directory);
    }
    Ok(ExistResultKind::NotFound)
}

fn exist_handle(value: &Value) -> ExistResultKind {
    match value {
        Value::HandleObject(handle) => {
            if handle.valid {
                ExistResultKind::Variable
            } else {
                ExistResultKind::NotFound
            }
        }
        Value::Listener(listener) => {
            if listener.valid {
                ExistResultKind::Variable
            } else {
                ExistResultKind::NotFound
            }
        }
        _ => ExistResultKind::NotFound,
    }
}

fn variable_exists(name: &str) -> bool {
    crate::workspace::lookup(name).is_some()
}

fn builtin_exists(name: &str) -> bool {
    let lowered = name.to_ascii_lowercase();
    builtin_functions()
        .into_iter()
        .any(|b| b.name.eq_ignore_ascii_case(&lowered))
}

fn class_exists(name: &str) -> Result<bool, String> {
    if runmat_builtins::get_class(name).is_some() {
        return Ok(true);
    }
    if class_folder_exists(name)? {
        return Ok(true);
    }
    if class_file_exists(name)? {
        return Ok(true);
    }
    Ok(false)
}

fn class_folder_exists(name: &str) -> Result<bool, String> {
    Ok(path_class_folder_candidates(name, "exist")?
        .into_iter()
        .any(|path| path.is_dir()))
}

fn class_file_exists(name: &str) -> Result<bool, String> {
    path_class_file_exists(name, CLASS_M_FILE_EXTENSIONS, "classdef", "exist")
}

fn method_exists(name: &str) -> bool {
    if let Some((class_name, method_name)) = split_method_name(name) {
        lookup_method(&class_name, &method_name).is_some()
    } else {
        false
    }
}

fn directory_exists(name: &str) -> Result<bool, String> {
    Ok(path_directory_candidates(name, "exist")?
        .into_iter()
        .any(|path| path.is_dir()))
}

fn detect_file_kind(name: &str) -> Result<Option<ExistResultKind>, String> {
    if path_find_file_with_extensions(name, MEX_EXTENSIONS, "exist")?.is_some() {
        return Ok(Some(ExistResultKind::Mex));
    }
    if path_find_file_with_extensions(name, PCODE_EXTENSIONS, "exist")?.is_some() {
        return Ok(Some(ExistResultKind::Pcode));
    }
    if path_find_file_with_extensions(name, SIMULINK_EXTENSIONS, "exist")?.is_some() {
        return Ok(Some(ExistResultKind::Simulink));
    }
    if path_find_file_with_extensions(name, GENERAL_FILE_EXTENSIONS, "exist")?.is_some() {
        return Ok(Some(ExistResultKind::File));
    }
    Ok(None)
}

fn value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::CharArray(array) if array.rows == 1 => Some(array.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        _ => None,
    }
}

fn split_method_name(name: &str) -> Option<(String, String)> {
    let mut parts: Vec<&str> = name.split('.').collect();
    if parts.len() < 2 {
        return None;
    }
    let method = parts.pop()?.to_string();
    if method.is_empty() {
        return None;
    }
    let class = parts.join(".");
    if class.is_empty() {
        return None;
    }
    Some((class, method))
}

#[cfg(test)]
mod tests {
    use super::super::REPL_FS_TEST_LOCK;
    use super::*;
    use once_cell::sync::OnceCell;
    use runmat_builtins::Value;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::env;
    use std::fs::{self, File};
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
                globals: || Vec::new(),
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

    #[test]
    fn exist_detects_workspace_variables() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        ensure_test_resolver();
        set_workspace(&[("alpha", Value::Num(1.0))]);

        let value = exist_builtin(Value::from("alpha"), Vec::new()).expect("exist");
        assert_eq!(value, Value::Num(1.0));
    }

    #[test]
    fn exist_detects_builtins() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let value = exist_builtin(Value::from("sin"), Vec::new()).expect("exist");
        assert_eq!(value, Value::Num(5.0));

        let builtin =
            exist_builtin(Value::from("sin"), vec![Value::from("builtin")]).expect("exist");
        assert_eq!(builtin, Value::Num(5.0));
    }

    #[test]
    fn exist_detects_files_and_mex() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        ensure_test_resolver();

        let temp = tempdir().expect("tempdir");
        let _guard = DirGuard::new();
        env::set_current_dir(temp.path()).expect("set temp");

        File::create("script.m").expect("create m-file");
        File::create("fastfft.mexw64").expect("create mex");

        let script =
            exist_builtin(Value::from("script"), vec![Value::from("file")]).expect("exist");
        assert_eq!(script, Value::Num(2.0));

        let mex = exist_builtin(Value::from("fastfft"), vec![Value::from("file")]).expect("exist");
        assert_eq!(mex, Value::Num(3.0));

        let mex_specific =
            exist_builtin(Value::from("fastfft"), vec![Value::from("mex")]).expect("exist");
        assert_eq!(mex_specific, Value::Num(3.0));
    }

    #[test]
    fn exist_detects_directories() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("tempdir");
        let _guard = DirGuard::new();
        env::set_current_dir(temp.path()).expect("set temp");
        fs::create_dir("data").expect("mkdir data");

        let dir = exist_builtin(Value::from("data"), vec![Value::from("dir")]).expect("exist");
        assert_eq!(dir, Value::Num(7.0));

        let any = exist_builtin(Value::from("data"), Vec::new()).expect("exist");
        assert_eq!(any, Value::Num(7.0));
    }

    #[test]
    fn exist_detects_class_files_and_packages() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("tempdir");
        let _guard = DirGuard::new();
        env::set_current_dir(temp.path()).expect("set temp");

        let mut file = File::create("Widget.m").expect("create class file");
        writeln!(
            file,
            "classdef Widget\n    methods\n        function obj = Widget()\n        end\n    end\nend"
        )
        .expect("write classdef");

        fs::create_dir_all("+pkg/@Gizmo").expect("create package class folder");

        let widget =
            exist_builtin(Value::from("Widget"), vec![Value::from("class")]).expect("exist");
        assert_eq!(widget, Value::Num(8.0));

        let gizmo =
            exist_builtin(Value::from("pkg.Gizmo"), vec![Value::from("class")]).expect("exist pkg");
        assert_eq!(gizmo, Value::Num(8.0));
    }

    #[test]
    fn exist_invalid_type_raises_error() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let err = exist_builtin(Value::from("foo"), vec![Value::from("unknown")])
            .expect_err("expected error");
        assert_eq!(err, ERROR_INVALID_TYPE);
    }

    #[test]
    fn exist_errors_on_non_text_name() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let err = exist_builtin(Value::Num(5.0), Vec::new()).expect_err("expected error");
        assert_eq!(err, ERROR_NAME_ARG);
    }

    #[test]
    fn exist_handle_returns_zero_for_non_handle() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let value =
            exist_builtin(Value::Num(17.0), vec![Value::from("handle")]).expect("exist handle");
        assert_eq!(value, Value::Num(0.0));
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = crate::builtins::common::test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
