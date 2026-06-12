//! MATLAB-compatible `run` builtin support.
//!
//! The runtime registry owns the descriptor and filesystem resolution helper.
//! The VM owns execution because `run` mutates the caller workspace.

use std::path::{Path, PathBuf};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_hir::RUN_BUILTIN_NAME;
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::path_to_string;
use crate::builtins::common::path_search::{
    file_candidates, find_file_with_extensions, path_is_file,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const RUN_SCRIPT_EXTENSIONS: &[&str] = &[".m"];

const RUN_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "script",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Script name or path to execute in the caller workspace.",
}];

const RUN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "run(script)",
    inputs: &RUN_INPUTS,
    outputs: &[],
}];

pub const RUN_ERROR_REQUIRES_VM: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUN.REQUIRES_VM",
    identifier: Some("RunMat:run:RequiresVm"),
    when: "`run` is dispatched outside an active VM workspace frame.",
    message: "run: requires VM workspace context",
};

pub const RUN_ERROR_ARG_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUN.ARG_TYPE",
    identifier: Some("RunMat:run:InvalidScriptArgument"),
    when: "The script argument is not a character row, string scalar, or scalar string array.",
    message: "run: script must be a character vector or string scalar",
};

pub const RUN_ERROR_EMPTY_SCRIPT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUN.EMPTY_SCRIPT",
    identifier: Some("RunMat:run:EmptyScript"),
    when: "The script argument is an empty path.",
    message: "run: script path must not be empty",
};

pub const RUN_ERROR_PATH_RESOLVE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUN.PATH_RESOLVE",
    identifier: Some("RunMat:run:PathResolveFailed"),
    when: "RunMat cannot resolve the current directory, home directory, or search path.",
    message: "run: failed to resolve script path",
};

pub const RUN_ERROR_FILE_NOT_FOUND: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUN.FILE_NOT_FOUND",
    identifier: Some("RunMat:run:FileNotFound"),
    when: "No matching script file exists in the current directory or RunMat search path.",
    message: "run: script file not found",
};

pub const RUN_ERROR_FILE_READ: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUN.FILE_READ",
    identifier: Some("RunMat:run:FileReadFailed"),
    when: "The matched script file cannot be read as source text.",
    message: "run: failed to read script file",
};

pub const RUN_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RUN.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:run:TooManyOutputs"),
    when: "`run` is called with one or more requested output arguments.",
    message: "run: too many output arguments",
};

pub const RUN_ERRORS: [BuiltinErrorDescriptor; 7] = [
    RUN_ERROR_REQUIRES_VM,
    RUN_ERROR_ARG_TYPE,
    RUN_ERROR_EMPTY_SCRIPT,
    RUN_ERROR_PATH_RESOLVE,
    RUN_ERROR_FILE_NOT_FOUND,
    RUN_ERROR_FILE_READ,
    RUN_ERROR_TOO_MANY_OUTPUTS,
];

pub const RUN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RUN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RUN_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::run")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "run",
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
    notes: "Script resolution and execution run on the host. GPU-resident script path arguments are gathered before lookup.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::run")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "run",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Script execution mutates the workspace and is a fusion barrier.",
};

#[derive(Debug, Clone)]
pub struct RunScriptSource {
    pub path: PathBuf,
    pub display_name: String,
    pub text: String,
}

fn run_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    run_error_with_detail(error, "")
}

fn run_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.is_empty() {
        error.message.to_string()
    } else {
        format!("{}: {detail}", error.message)
    };
    let mut builder = build_runtime_error(message).with_builtin(RUN_BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn run_flow(err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(str::to_string);
    let mut builder = build_runtime_error(err.message().to_string())
        .with_builtin(RUN_BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn value_to_string_scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::CharArray(array) if array.rows == 1 => Some(array.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        _ => None,
    }
}

pub fn requires_vm_workspace_context() -> crate::BuiltinResult<Value> {
    Err(run_error(&RUN_ERROR_REQUIRES_VM))
}

pub fn too_many_outputs_error() -> RuntimeError {
    run_error(&RUN_ERROR_TOO_MANY_OUTPUTS)
}

fn bare_m_file_stem(script: &str) -> Option<&str> {
    if script.starts_with('~')
        || script.starts_with('@')
        || script.starts_with('+')
        || script.contains('/')
        || script.contains('\\')
    {
        return None;
    }
    let path = Path::new(script);
    if path.components().count() != 1 {
        return None;
    }
    let extension = path.extension()?.to_str()?;
    if !extension.eq_ignore_ascii_case("m") {
        return None;
    }
    path.file_stem()?.to_str().filter(|stem| !stem.is_empty())
}

async fn find_run_script(script: &str) -> Result<Option<PathBuf>, String> {
    if let Some(path) =
        find_file_with_extensions(script, RUN_SCRIPT_EXTENSIONS, RUN_BUILTIN_NAME).await?
    {
        return Ok(Some(path));
    }

    let Some(stem) = bare_m_file_stem(script) else {
        return Ok(None);
    };
    for candidate in file_candidates(stem, RUN_SCRIPT_EXTENSIONS, RUN_BUILTIN_NAME)? {
        if candidate
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("m"))
            && path_is_file(&candidate).await
        {
            return Ok(Some(candidate));
        }
    }
    Ok(None)
}

pub async fn resolve_run_source(value: &Value) -> BuiltinResult<RunScriptSource> {
    let value = gather_if_needed_async(value).await.map_err(run_flow)?;
    let script = value_to_string_scalar(&value).ok_or_else(|| run_error(&RUN_ERROR_ARG_TYPE))?;
    if script.is_empty() {
        return Err(run_error(&RUN_ERROR_EMPTY_SCRIPT));
    }

    let path = find_run_script(&script)
        .await
        .map_err(|err| run_error_with_detail(&RUN_ERROR_PATH_RESOLVE, err))?
        .ok_or_else(|| run_error_with_detail(&RUN_ERROR_FILE_NOT_FOUND, format!("'{script}'")))?;

    let text = runmat_filesystem::read_to_string_async(&path)
        .await
        .map_err(|err| {
            run_error_with_detail(&RUN_ERROR_FILE_READ, format!("{} ({err})", path.display()))
        })?;

    let display_path = runmat_filesystem::canonicalize_async(&path)
        .await
        .unwrap_or_else(|_| path.clone());
    Ok(RunScriptSource {
        path: display_path.clone(),
        display_name: path_to_string(&display_path),
        text,
    })
}

#[runtime_builtin(
    name = "run",
    category = "io/repl_fs",
    summary = "Execute a script file in the caller workspace.",
    keywords = "run,script,file,path,workspace",
    sink = true,
    suppress_auto_output = true,
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::run_type),
    descriptor(crate::builtins::io::repl_fs::run::RUN_DESCRIPTOR),
    builtin_path = "crate::builtins::io::repl_fs::run"
)]
pub fn run_builtin_registered(_args: Vec<Value>) -> crate::BuiltinResult<Value> {
    requires_vm_workspace_context()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::path_state::{current_path_string, set_path_string};
    use crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK;
    use futures::executor::block_on;
    use runmat_builtins::CharArray;
    use std::env;
    use std::path::{Path, PathBuf};

    struct CwdGuard {
        original: PathBuf,
    }

    struct PathStateGuard {
        previous: String,
    }

    impl Drop for PathStateGuard {
        fn drop(&mut self) {
            set_path_string(&self.previous);
        }
    }

    impl Drop for CwdGuard {
        fn drop(&mut self) {
            let _ = env::set_current_dir(&self.original);
        }
    }

    fn push_cwd(path: &Path) -> CwdGuard {
        let original = env::current_dir().expect("current dir");
        env::set_current_dir(path).expect("set current dir");
        CwdGuard { original }
    }

    fn push_path_state(path: &Path) -> PathStateGuard {
        let previous = current_path_string();
        let path = path.to_string_lossy().to_string();
        set_path_string(&path);
        PathStateGuard { previous }
    }

    #[test]
    fn runtime_fallback_requires_vm_context() {
        let err = run_builtin_registered(Vec::new()).expect_err("run fallback should fail");
        assert_eq!(err.identifier(), Some("RunMat:run:RequiresVm"));
    }

    #[test]
    fn resolves_script_from_current_directory_with_implicit_m_extension() {
        let _lock = REPL_FS_TEST_LOCK.lock().unwrap();
        let temp = tempfile::TempDir::new().expect("tempdir");
        std::fs::write(temp.path().join("worker.m"), "generated = 41;\n").expect("write script");
        let _cwd = push_cwd(temp.path());

        let source = block_on(resolve_run_source(&Value::from("worker"))).expect("resolve source");
        assert!(source.display_name.ends_with("worker.m"));
        assert_eq!(source.text, "generated = 41;\n");
    }

    #[test]
    fn resolves_bare_m_filename_from_search_path() {
        let _lock = REPL_FS_TEST_LOCK.lock().unwrap();
        let temp = tempfile::TempDir::new().expect("tempdir");
        let scripts = temp.path().join("scripts");
        std::fs::create_dir_all(&scripts).expect("create scripts dir");
        std::fs::write(scripts.join("path_worker.m"), "path_value = 17;\n").expect("write script");
        let _cwd = push_cwd(temp.path());
        let _path = push_path_state(&scripts);

        let source =
            block_on(resolve_run_source(&Value::from("path_worker.m"))).expect("resolve source");
        assert!(source.display_name.ends_with("path_worker.m"));
        assert_eq!(source.text, "path_value = 17;\n");
    }

    #[test]
    fn resolves_script_from_char_row_path() {
        let _lock = REPL_FS_TEST_LOCK.lock().unwrap();
        let temp = tempfile::TempDir::new().expect("tempdir");
        let path = temp.path().join("direct_script.m");
        std::fs::write(&path, "x = 1;\n").expect("write script");

        let value = Value::CharArray(CharArray::new_row(path.to_string_lossy().as_ref()));
        let source = block_on(resolve_run_source(&value)).expect("resolve source");
        assert_eq!(source.text, "x = 1;\n");
        assert!(source.path.ends_with("direct_script.m"));
    }

    #[test]
    fn missing_script_reports_stable_identifier() {
        let _lock = REPL_FS_TEST_LOCK.lock().unwrap();
        let temp = tempfile::TempDir::new().expect("tempdir");
        let _cwd = push_cwd(temp.path());

        let err = block_on(resolve_run_source(&Value::from("missing_script")))
            .expect_err("missing script should fail");
        assert_eq!(err.identifier(), Some("RunMat:run:FileNotFound"));
    }

    #[test]
    fn invalid_script_argument_reports_stable_identifier() {
        let err =
            block_on(resolve_run_source(&Value::Num(1.0))).expect_err("numeric script should fail");
        assert_eq!(err.identifier(), Some("RunMat:run:InvalidScriptArgument"));
    }
}
