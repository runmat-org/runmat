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
    find_file_with_extensions as path_find_file_with_extensions, path_is_directory,
    CLASS_M_FILE_EXTENSIONS, GENERAL_FILE_EXTENSIONS, LIB_EXTENSIONS, MEX_EXTENSIONS,
    PCODE_EXTENSIONS, SIMULINK_EXTENSIONS, THUNK_EXTENSIONS,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const ERROR_NAME_ARG: &str = "exist: name must be a character vector or string scalar";
const ERROR_TYPE_ARG: &str = "exist: type must be a character vector or string scalar";
const ERROR_INVALID_TYPE: &str = "exist: invalid type. Type must be one of 'var', 'variable', 'file', 'dir', 'directory', 'folder', 'builtin', 'built-in', 'class', 'handle', 'method', 'mex', 'pcode', 'simulink', 'thunk', 'lib', 'library', or 'java'";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::exist")]
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::exist")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "exist",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "I/O builtins are not eligible for fusion; metadata registered for completeness.",
};

const BUILTIN_NAME: &str = "exist";

fn exist_error(message: impl Into<String>) -> RuntimeError {
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
    name = "exist",
    category = "io/repl_fs",
    summary = "Determine whether a variable, file, folder, built-in, or class exists.",
    keywords = "exist,file,dir,var,builtin,class",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::exist_type),
    builtin_path = "crate::builtins::io::repl_fs::exist"
)]
async fn exist_builtin(name: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(exist_error("exist: too many input arguments"));
    }

    let name_host = gather_if_needed_async(&name)
        .await
        .map_err(map_control_flow)?;
    let type_value = match rest.first() {
        Some(value) => Some(
            gather_if_needed_async(value)
                .await
                .map_err(map_control_flow)?,
        ),
        None => None,
    };

    let query = type_value
        .as_ref()
        .map(parse_type_argument)
        .transpose()?
        .unwrap_or(ExistQuery::Any);

    let result = match query {
        ExistQuery::Handle => exist_handle(&name_host),
        _ => {
            let text = value_to_string(&name_host).ok_or_else(|| exist_error(ERROR_NAME_ARG))?;
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

fn parse_type_argument(value: &Value) -> BuiltinResult<ExistQuery> {
    let text = value_to_string(value).ok_or_else(|| exist_error(ERROR_TYPE_ARG))?;
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
        _ => Err(exist_error(ERROR_INVALID_TYPE)),
    }
}

fn exist_for_query(name: &str, query: ExistQuery) -> BuiltinResult<ExistResultKind> {
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
            if path_find_file_with_extensions(name, MEX_EXTENSIONS, "exist")
                .map_err(exist_error)?
                .is_some()
            {
                ExistResultKind::Mex
            } else {
                ExistResultKind::NotFound
            },
        ),
        ExistQuery::Pcode => Ok(
            if path_find_file_with_extensions(name, PCODE_EXTENSIONS, "exist")
                .map_err(exist_error)?
                .is_some()
            {
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
            if path_find_file_with_extensions(name, SIMULINK_EXTENSIONS, "exist")
                .map_err(exist_error)?
                .is_some()
            {
                ExistResultKind::Simulink
            } else {
                ExistResultKind::NotFound
            },
        ),
        ExistQuery::Thunk => Ok(
            if path_find_file_with_extensions(name, THUNK_EXTENSIONS, "exist")
                .map_err(exist_error)?
                .is_some()
            {
                ExistResultKind::File
            } else {
                ExistResultKind::NotFound
            },
        ),
        ExistQuery::Lib => Ok(
            if path_find_file_with_extensions(name, LIB_EXTENSIONS, "exist")
                .map_err(exist_error)?
                .is_some()
            {
                ExistResultKind::File
            } else {
                ExistResultKind::NotFound
            },
        ),
        ExistQuery::Java => Ok(ExistResultKind::NotFound),
        ExistQuery::Handle => unreachable!("handle queries handled separately"),
    }
}

fn evaluate_default(name: &str) -> BuiltinResult<ExistResultKind> {
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

fn class_exists(name: &str) -> BuiltinResult<bool> {
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

fn class_folder_exists(name: &str) -> BuiltinResult<bool> {
    Ok(path_class_folder_candidates(name, "exist")
        .map_err(exist_error)?
        .into_iter()
        .any(|path| path_is_directory(&path)))
}

fn class_file_exists(name: &str) -> BuiltinResult<bool> {
    path_class_file_exists(name, CLASS_M_FILE_EXTENSIONS, "classdef", "exist").map_err(exist_error)
}

fn method_exists(name: &str) -> bool {
    if let Some((class_name, method_name)) = split_method_name(name) {
        lookup_method(&class_name, &method_name).is_some()
    } else {
        false
    }
}

fn directory_exists(name: &str) -> BuiltinResult<bool> {
    Ok(path_directory_candidates(name, "exist")
        .map_err(exist_error)?
        .into_iter()
        .any(|path| path_is_directory(&path)))
}

fn detect_file_kind(name: &str) -> BuiltinResult<Option<ExistResultKind>> {
    if path_find_file_with_extensions(name, MEX_EXTENSIONS, "exist")
        .map_err(exist_error)?
        .is_some()
    {
        return Ok(Some(ExistResultKind::Mex));
    }
    if path_find_file_with_extensions(name, PCODE_EXTENSIONS, "exist")
        .map_err(exist_error)?
        .is_some()
    {
        return Ok(Some(ExistResultKind::Pcode));
    }
    if path_find_file_with_extensions(name, SIMULINK_EXTENSIONS, "exist")
        .map_err(exist_error)?
        .is_some()
    {
        return Ok(Some(ExistResultKind::Simulink));
    }
    if path_find_file_with_extensions(name, GENERAL_FILE_EXTENSIONS, "exist")
        .map_err(exist_error)?
        .is_some()
    {
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
pub(crate) mod tests {
    use super::super::REPL_FS_TEST_LOCK;
    use super::*;
    use runmat_builtins::Value;
    use runmat_filesystem as vfs;
    use runmat_thread_local::runmat_thread_local;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::env;
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;
    use tempfile::tempdir;

    fn exist_builtin(name: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::exist_builtin(name, rest))
    }

    fn workspace_guard() -> std::sync::MutexGuard<'static, ()> {
        crate::workspace::test_guard()
    }

    fn test_guard() -> (
        std::sync::MutexGuard<'static, ()>,
        std::sync::MutexGuard<'static, ()>,
    ) {
        let workspace = workspace_guard();
        let fs_lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        (workspace, fs_lock)
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
    fn exist_detects_workspace_variables() {
        let (_guard, _lock) = test_guard();
        ensure_test_resolver();
        set_workspace(&[("alpha", Value::Num(1.0))]);

        let value = exist_builtin(Value::from("alpha"), Vec::new()).expect("exist");
        assert_eq!(value, Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exist_detects_builtins() {
        let (_guard, _lock) = test_guard();

        let value = exist_builtin(Value::from("sin"), Vec::new()).expect("exist");
        assert_eq!(value, Value::Num(5.0));

        let builtin =
            exist_builtin(Value::from("sin"), vec![Value::from("builtin")]).expect("exist");
        assert_eq!(builtin, Value::Num(5.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exist_detects_files_and_mex() {
        let (_guard, _lock) = test_guard();
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exist_detects_directories() {
        let (_guard, _lock) = test_guard();

        let temp = tempdir().expect("tempdir");
        let _guard = DirGuard::new();
        env::set_current_dir(temp.path()).expect("set temp");
        vfs::create_dir("data").expect("mkdir data");

        let dir = exist_builtin(Value::from("data"), vec![Value::from("dir")]).expect("exist");
        assert_eq!(dir, Value::Num(7.0));

        let any = exist_builtin(Value::from("data"), Vec::new()).expect("exist");
        assert_eq!(any, Value::Num(7.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exist_detects_class_files_and_packages() {
        let (_guard, _lock) = test_guard();

        let temp = tempdir().expect("tempdir");
        let _guard = DirGuard::new();
        env::set_current_dir(temp.path()).expect("set temp");

        let mut file = File::create("Widget.m").expect("create class file");
        writeln!(
            file,
            "classdef Widget\n    methods\n        function obj = Widget()\n        end\n    end\nend"
        )
        .expect("write classdef");

        vfs::create_dir_all("+pkg/@Gizmo").expect("create package class folder");

        let widget =
            exist_builtin(Value::from("Widget"), vec![Value::from("class")]).expect("exist");
        assert_eq!(widget, Value::Num(8.0));

        let gizmo =
            exist_builtin(Value::from("pkg.Gizmo"), vec![Value::from("class")]).expect("exist pkg");
        assert_eq!(gizmo, Value::Num(8.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exist_invalid_type_raises_error() {
        let (_guard, _lock) = test_guard();

        let err = exist_builtin(Value::from("foo"), vec![Value::from("unknown")])
            .expect_err("expected error");
        assert_eq!(err.message(), ERROR_INVALID_TYPE);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exist_errors_on_non_text_name() {
        let (_guard, _lock) = test_guard();

        let err = exist_builtin(Value::Num(5.0), Vec::new()).expect_err("expected error");
        assert_eq!(err.message(), ERROR_NAME_ARG);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exist_handle_returns_zero_for_non_handle() {
        let (_guard, _lock) = test_guard();

        let value =
            exist_builtin(Value::Num(17.0), vec![Value::from("handle")]).expect("exist handle");
        assert_eq!(value, Value::Num(0.0));
    }
}
