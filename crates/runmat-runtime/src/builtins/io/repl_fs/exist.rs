//! MATLAB-compatible `exist` builtin for RunMat.
//!
//! Mirrors MATLAB semantics for checking whether a variable, file, folder,
//! builtin, class, or other entity is available in the current session.

use runmat_builtins::{
    builtin_functions, lookup_method, BuiltinCompletionPolicy, BuiltinDescriptor,
    BuiltinErrorDescriptor, BuiltinExtensionDescriptor, BuiltinExtensionMode,
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

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
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

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
    notes: "Filesystem and workspace lookup run on the host; resident numeric arguments are invalid text and reject before provider access.",
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

const EXIST_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "typeID",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Existence code (0,1,2,3,4,5,6,7,8) following MATLAB semantics.",
}];
const EXIST_INPUTS_NAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Variable/function/file/class name to query.",
}];
const EXIST_INPUTS_NAME_TYPE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Variable/function/file/class name to query.",
    },
    BuiltinParamDescriptor {
        name: "type",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Documented query kind: builtin|class|dir|file|var.",
    },
];
const EXIST_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "exist name",
        inputs: &EXIST_INPUTS_NAME,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "exist name searchType",
        inputs: &EXIST_INPUTS_NAME_TYPE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "typeID = exist(name)",
        inputs: &EXIST_INPUTS_NAME,
        outputs: &EXIST_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "typeID = exist(name, searchType)",
        inputs: &EXIST_INPUTS_NAME_TYPE,
        outputs: &EXIST_OUTPUT,
    },
];
const EXIST_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXIST.TOO_MANY_INPUTS",
    identifier: Some("RunMat:exist:TooManyInputs"),
    when: "More than two total input arguments are provided.",
    message: "exist: too many input arguments",
};
const EXIST_ERROR_NAME_ARG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXIST.NAME_ARG",
    identifier: Some("RunMat:exist:InvalidName"),
    when: "Name input is not a character vector or string scalar/array scalar.",
    message: "exist: name must be a character vector or string scalar",
};
const EXIST_ERROR_TYPE_ARG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXIST.TYPE_ARG",
    identifier: Some("RunMat:exist:InvalidTypeArgument"),
    when: "Type input is not a character vector or string scalar/array scalar.",
    message: "exist: type must be a character vector or string scalar",
};
const EXIST_ERROR_INVALID_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EXIST.INVALID_TYPE",
    identifier: Some("RunMat:exist:InvalidType"),
    when: "Type input is not one of the supported exist query types.",
    message: "exist: invalid type. Type must be one of 'builtin', 'class', 'dir', 'file', or 'var'",
};
const EXIST_ERRORS: [BuiltinErrorDescriptor; 4] = [
    EXIST_ERROR_TOO_MANY_INPUTS,
    EXIST_ERROR_NAME_ARG,
    EXIST_ERROR_TYPE_ARG,
    EXIST_ERROR_INVALID_TYPE,
];
pub const EXIST_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &EXIST_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &EXIST_ERRORS,
};

const EXIST_SEARCH_TYPE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "exist-search-type-extension",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "exist selectors and aliases beyond builtin, class, dir, file, and var are RunMat extensions",
    error_identifier: Some("RunMat:compatibility:ExistSearchTypeExtension"),
};
const EXIST_NONTEXT_HANDLE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "exist-nontext-handle-query",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "exist(handleObject, 'handle') with a nontext first argument is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ExistNontextHandleExtension"),
};
pub const EXIST_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [EXIST_SEARCH_TYPE_EXTENSION, EXIST_NONTEXT_HANDLE_EXTENSION];
pub const EXIST_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "exist accepts textual name and searchType inputs only; all eight integer classes, logical values, and resident numeric handles reject before lookup or provider access. Its integer-valued typeID output remains a documented host double scalar, not typed-integer storage.",
};

fn exist_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn exist_error_row(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "exist",
    category = "io/repl_fs",
    summary = "Determine whether variables, files, folders, or symbols exist.",
    keywords = "exist,file,dir,var,builtin,class",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::exist_type),
    descriptor(crate::builtins::io::repl_fs::exist::EXIST_DESCRIPTOR),
    extensions(crate::builtins::io::repl_fs::exist::EXIST_EXTENSIONS),
    integer_audit(crate::builtins::io::repl_fs::exist::EXIST_INTEGER_AUDIT),
    builtin_path = "crate::builtins::io::repl_fs::exist"
)]
async fn exist_builtin(name: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(exist_error_row(&EXIST_ERROR_TOO_MANY_INPUTS));
    }

    if crate::dispatcher::value_contains_gpu(&name) {
        return Err(exist_error_row(&EXIST_ERROR_NAME_ARG));
    }
    if rest
        .first()
        .is_some_and(crate::dispatcher::value_contains_gpu)
    {
        return Err(exist_error_row(&EXIST_ERROR_TYPE_ARG));
    }
    let type_value = rest.first();

    let query = type_value
        .map(parse_type_argument)
        .transpose()?
        .unwrap_or(ExistQuery::Any);

    let result = match query {
        ExistQuery::Handle if value_to_string(&name).is_none() => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &EXIST_NONTEXT_HANDLE_EXTENSION,
                BUILTIN_NAME,
            )?;
            exist_handle(&name)
        }
        ExistQuery::Handle => ExistResultKind::NotFound,
        _ => {
            let text =
                value_to_string(&name).ok_or_else(|| exist_error_row(&EXIST_ERROR_NAME_ARG))?;
            exist_for_query(&text, query).await?
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
    let text = value_to_string(value).ok_or_else(|| exist_error_row(&EXIST_ERROR_TYPE_ARG))?;
    match text.trim() {
        "var" => Ok(ExistQuery::Var),
        "file" => Ok(ExistQuery::File),
        "dir" => Ok(ExistQuery::Dir),
        "builtin" => Ok(ExistQuery::Builtin),
        "class" => Ok(ExistQuery::Class),
        raw => parse_extension_type(raw),
    }
}

fn parse_extension_type(raw: &str) -> BuiltinResult<ExistQuery> {
    let query = match raw.to_ascii_lowercase().as_str() {
        "" => ExistQuery::Any,
        "var" | "variable" => ExistQuery::Var,
        "file" => ExistQuery::File,
        "dir" | "directory" | "folder" => ExistQuery::Dir,
        "builtin" | "built-in" => ExistQuery::Builtin,
        "class" => ExistQuery::Class,
        "mex" => ExistQuery::Mex,
        "pcode" | "p" => ExistQuery::Pcode,
        "handle" => ExistQuery::Handle,
        "method" => ExistQuery::Method,
        "simulink" => ExistQuery::Simulink,
        "thunk" => ExistQuery::Thunk,
        "lib" | "library" => ExistQuery::Lib,
        "java" => ExistQuery::Java,
        _ => return Err(exist_error_row(&EXIST_ERROR_INVALID_TYPE)),
    };
    crate::compatibility::ensure_builtin_extension_enabled(
        &EXIST_SEARCH_TYPE_EXTENSION,
        BUILTIN_NAME,
    )?;
    Ok(query)
}

async fn exist_for_query(name: &str, query: ExistQuery) -> BuiltinResult<ExistResultKind> {
    if contains_wildcards(name) {
        return Ok(ExistResultKind::NotFound);
    }

    match query {
        ExistQuery::Any => evaluate_default(name).await,
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
        ExistQuery::Class => Ok(if class_exists(name).await? {
            ExistResultKind::Class
        } else {
            ExistResultKind::NotFound
        }),
        ExistQuery::Dir => Ok(if directory_exists(name).await? {
            ExistResultKind::Directory
        } else {
            ExistResultKind::NotFound
        }),
        ExistQuery::File => {
            if directory_exists(name).await? {
                Ok(ExistResultKind::Directory)
            } else {
                Ok(detect_file_kind(name)
                    .await?
                    .unwrap_or(ExistResultKind::NotFound))
            }
        }
        ExistQuery::Mex => Ok(
            if path_find_file_with_extensions(name, MEX_EXTENSIONS, "exist")
                .await
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
                .await
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
                .await
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
                .await
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
                .await
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

async fn evaluate_default(name: &str) -> BuiltinResult<ExistResultKind> {
    if variable_exists(name) {
        return Ok(ExistResultKind::Variable);
    }
    if builtin_exists(name) {
        return Ok(ExistResultKind::Builtin);
    }
    if directory_exists(name).await? {
        return Ok(ExistResultKind::Directory);
    }
    if class_exists(name).await? {
        return Ok(ExistResultKind::Class);
    }
    if let Some(kind) = detect_file_kind(name).await? {
        return Ok(kind);
    }
    Ok(ExistResultKind::NotFound)
}

fn exist_handle(value: &Value) -> ExistResultKind {
    match value {
        Value::HandleObject(handle) => {
            if crate::is_handle_valid(handle) {
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

async fn class_exists(name: &str) -> BuiltinResult<bool> {
    if runmat_builtins::get_class(name).is_some() {
        return Ok(true);
    }
    if class_folder_exists(name).await? {
        return Ok(true);
    }
    if class_file_exists(name).await? {
        return Ok(true);
    }
    Ok(false)
}

async fn class_folder_exists(name: &str) -> BuiltinResult<bool> {
    for path in path_class_folder_candidates(name, "exist").map_err(exist_error)? {
        if path_is_directory(&path).await {
            return Ok(true);
        }
    }
    Ok(false)
}

async fn class_file_exists(name: &str) -> BuiltinResult<bool> {
    path_class_file_exists(name, CLASS_M_FILE_EXTENSIONS, "classdef", "exist")
        .await
        .map_err(exist_error)
}

fn method_exists(name: &str) -> bool {
    if let Some((class_name, method_name)) = split_method_name(name) {
        lookup_method(&class_name, &method_name).is_some()
    } else {
        false
    }
}

async fn directory_exists(name: &str) -> BuiltinResult<bool> {
    for path in path_directory_candidates(name, "exist").map_err(exist_error)? {
        if path_is_directory(&path).await {
            return Ok(true);
        }
    }
    Ok(false)
}

async fn detect_file_kind(name: &str) -> BuiltinResult<Option<ExistResultKind>> {
    if path_find_file_with_extensions(name, MEX_EXTENSIONS, "exist")
        .await
        .map_err(exist_error)?
        .is_some()
    {
        return Ok(Some(ExistResultKind::Mex));
    }
    if path_find_file_with_extensions(name, PCODE_EXTENSIONS, "exist")
        .await
        .map_err(exist_error)?
        .is_some()
    {
        return Ok(Some(ExistResultKind::Pcode));
    }
    if path_find_file_with_extensions(name, SIMULINK_EXTENSIONS, "exist")
        .await
        .map_err(exist_error)?
        .is_some()
    {
        return Ok(Some(ExistResultKind::Simulink));
    }
    if path_find_file_with_extensions(name, GENERAL_FILE_EXTENSIONS, "exist")
        .await
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
    use runmat_builtins::{ClassDef, MethodDef};
    use runmat_filesystem as vfs;
    use runmat_thread_local::runmat_thread_local;
    use runmat_value::{Access, IntValue, Value};
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::env;
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use tempfile::tempdir;

    static TEST_CLASS_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_class_name(prefix: &str) -> String {
        let id = TEST_CLASS_COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("{}_{}", prefix, id)
    }

    fn exist_builtin(name: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::exist_builtin(name, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exist_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = EXIST_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"exist name"));
        assert!(labels.contains(&"exist name searchType"));
        assert!(labels.contains(&"typeID = exist(name)"));
        assert!(labels.contains(&"typeID = exist(name, searchType)"));
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
            assign: None,
            clear: None,
            remove: None,
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
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
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
        futures::executor::block_on(vfs::create_dir_async("data")).expect("mkdir data");

        let dir = exist_builtin(Value::from("data"), vec![Value::from("dir")]).expect("exist");
        assert_eq!(dir, Value::Num(7.0));

        let any = exist_builtin(Value::from("data"), Vec::new()).expect("exist");
        assert_eq!(any, Value::Num(7.0));

        let file =
            exist_builtin(Value::from("data"), vec![Value::from("file")]).expect("exist file");
        assert_eq!(file, Value::Num(7.0));
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

        futures::executor::block_on(vfs::create_dir_all_async("+pkg/@Gizmo"))
            .expect("create package class folder");

        let widget =
            exist_builtin(Value::from("Widget"), vec![Value::from("class")]).expect("exist");
        assert_eq!(widget, Value::Num(8.0));

        futures::executor::block_on(vfs::create_dir_async("Widget"))
            .expect("create colliding folder");
        let default_widget =
            exist_builtin(Value::from("Widget"), Vec::new()).expect("folder wins over class file");
        assert_eq!(default_widget, Value::Num(7.0));

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
        assert_eq!(err.message(), EXIST_ERROR_INVALID_TYPE.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exist_errors_on_non_text_name() {
        let (_guard, _lock) = test_guard();

        let err = exist_builtin(Value::Num(5.0), Vec::new()).expect_err("expected error");
        assert_eq!(err.message(), EXIST_ERROR_NAME_ARG.message);
    }

    #[test]
    fn exist_rejects_all_integer_classes_and_logical_inputs() {
        let (_guard, _lock) = test_guard();
        for value in [
            IntValue::I8(1),
            IntValue::I16(1),
            IntValue::I32(1),
            IntValue::I64(1),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(u64::MAX),
        ] {
            let err =
                exist_builtin(Value::Int(value), Vec::new()).expect_err("integer name must reject");
            assert_eq!(err.message(), EXIST_ERROR_NAME_ARG.message);
        }
        let err =
            exist_builtin(Value::Bool(true), Vec::new()).expect_err("logical name must reject");
        assert_eq!(err.message(), EXIST_ERROR_NAME_ARG.message);
    }

    #[test]
    fn exist_rejects_resident_numeric_name_without_provider_access() {
        let (_guard, _lock) = test_guard();
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
        });
        let err = exist_builtin(resident, Vec::new()).expect_err("resident name must reject");
        assert_eq!(err.message(), EXIST_ERROR_NAME_ARG.message);
        assert!(!err.message().contains("provider"));
    }

    #[test]
    fn strict_mode_gates_nonpublic_search_types() {
        let (_guard, _lock) = test_guard();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        for selector in ["variable", "folder", "built-in", "mex", "handle", "method"] {
            let err = exist_builtin(Value::from("name"), vec![Value::from(selector)])
                .expect_err("selector extension must be gated");
            assert_eq!(
                err.identifier(),
                EXIST_SEARCH_TYPE_EXTENSION.error_identifier,
                "selector {selector}"
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exist_handle_returns_zero_for_non_handle() {
        let (_guard, _lock) = test_guard();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);

        let value =
            exist_builtin(Value::Num(17.0), vec![Value::from("handle")]).expect("exist handle");
        assert_eq!(value, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exist_method_uses_registered_class_metadata_including_inheritance() {
        let (_guard, _lock) = test_guard();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);

        let parent_name = unique_class_name("existParent");
        let child_name = unique_class_name("existChild");
        let mut parent_methods = HashMap::new();
        parent_methods.insert(
            "parentOnly".to_string(),
            MethodDef {
                name: "parentOnly".to_string(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: Access::Public,
                function_name: "parent_only_impl".to_string(),
                implicit_class_argument: None,
            },
        );
        runmat_builtins::register_class(ClassDef {
            name: parent_name.clone(),
            parent: None,
            properties: HashMap::new(),
            methods: parent_methods,
        });
        runmat_builtins::register_class(ClassDef {
            name: child_name.clone(),
            parent: Some(parent_name.clone()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });

        let direct = exist_builtin(
            Value::from(format!("{parent_name}.parentOnly")),
            vec![Value::from("method")],
        )
        .expect("direct class method lookup should succeed");
        assert_eq!(direct, Value::Num(5.0));

        let inherited = exist_builtin(
            Value::from(format!("{child_name}.parentOnly")),
            vec![Value::from("method")],
        )
        .expect("inherited class method lookup should succeed");
        assert_eq!(inherited, Value::Num(5.0));

        let missing = exist_builtin(
            Value::from(format!("{child_name}.missingMethod")),
            vec![Value::from("method")],
        )
        .expect("missing method lookup should return not found");
        assert_eq!(missing, Value::Num(0.0));
    }
}
