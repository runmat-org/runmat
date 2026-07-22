//! MATLAB-compatible `open` builtin for files and variables.

use std::path::{Path, PathBuf};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::path_to_string;
use crate::builtins::common::path_search::{
    find_file_with_extensions, path_is_file, search_directories, should_treat_as_path,
    GENERAL_FILE_EXTENSIONS,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "open";

const OPEN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "MAT-file contents, extension-handler output, figure handle, or empty array.",
}];

const OPEN_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File or variable name to open.",
}];

const OPEN_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "open name",
        inputs: &OPEN_INPUTS,
        outputs: &OPEN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = open(name)",
        inputs: &OPEN_INPUTS,
        outputs: &OPEN_OUTPUT,
    },
];

const OPEN_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPEN.ARG_COUNT",
    identifier: Some("RunMat:open:ArgumentCount"),
    when: "The call does not provide exactly one input argument.",
    message: "open: expected exactly one input argument",
};

const OPEN_ERROR_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPEN.NAME",
    identifier: Some("RunMat:open:InvalidName"),
    when: "The name is not a string scalar or row character vector.",
    message: "open: name must be a character vector or string scalar",
};

const OPEN_ERROR_NOT_FOUND: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPEN.NOT_FOUND",
    identifier: Some("RunMat:open:NotFound"),
    when: "No matching variable or file can be resolved.",
    message: "open: file or variable was not found",
};

const OPEN_ERROR_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPEN.UNSUPPORTED",
    identifier: Some("RunMat:open:UnsupportedFileType"),
    when: "The resolved file type cannot be opened by RunMat and no extension handler exists.",
    message: "open: unsupported file type",
};

const OPEN_ERROR_HANDLER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPEN.HANDLER",
    identifier: Some("RunMat:open:HandlerFailed"),
    when: "A custom open<extension> handler fails.",
    message: "open: extension handler failed",
};

const OPEN_ERROR_OUTPUT_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPEN.OUTPUT_COUNT",
    identifier: Some("RunMat:open:TooManyOutputs"),
    when: "The call requests more than one output.",
    message: "open: expected at most one output",
};

const OPEN_ERRORS: [BuiltinErrorDescriptor; 6] = [
    OPEN_ERROR_ARG_COUNT,
    OPEN_ERROR_NAME,
    OPEN_ERROR_NOT_FOUND,
    OPEN_ERROR_UNSUPPORTED,
    OPEN_ERROR_HANDLER,
    OPEN_ERROR_OUTPUT_COUNT,
];

pub const OPEN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OPEN_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &OPEN_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::open")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "open",
    op_kind: GpuOpKind::Custom("io-open"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Runs on the host. MAT-file data is loaded into CPU-resident values; viewer/editor opens do not involve the acceleration provider.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::open")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "open",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "File and workspace opening is an I/O operation and is not eligible for fusion.",
};

#[runtime_builtin(
    name = "open",
    category = "io/repl_fs",
    summary = "Open a file or variable using MATLAB-compatible dispatch.",
    keywords = "open,file,mat,viewer,extension handler",
    accel = "cpu",
    sink = true,
    suppress_auto_output = true,
    type_resolver(crate::builtins::io::type_resolvers::open_type),
    descriptor(crate::builtins::io::repl_fs::open::OPEN_DESCRIPTOR),
    builtin_path = "crate::builtins::io::repl_fs::open"
)]
async fn open_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() != 1 {
        return Err(open_error(
            &OPEN_ERROR_ARG_COUNT,
            OPEN_ERROR_ARG_COUNT.message,
        ));
    }
    if requested_output_count() > 1 {
        return Err(open_error(
            &OPEN_ERROR_OUTPUT_COUNT,
            format!(
                "open: expected at most one output, got {}",
                requested_output_count()
            ),
        ));
    }

    let name = open_name(&args[0]).await?;
    if name.is_empty() {
        return Err(open_error(&OPEN_ERROR_NAME, "open: name must not be empty"));
    }

    if !should_treat_as_path(&name) && crate::workspace::lookup(&name).is_some() {
        return Ok(empty_array());
    }

    let path = resolve_open_path(&name).await?;
    open_path(&path).await
}

async fn open_path(path: &Path) -> BuiltinResult<Value> {
    let extension = normalized_extension(path);
    let path_text = path_to_string(path);

    if extension == "fig" {
        if let Some(handle) = try_open_runmat_figure_scene(path).await? {
            return Ok(Value::Num(handle as f64));
        }
        let eval = crate::builtins::io::mat::load::evaluate(&[Value::String(path_text)])
            .await
            .map_err(|err| open_flow_error("open", err))?;
        return Ok(eval.first_output());
    }

    if extension == "mat" {
        let eval = crate::builtins::io::mat::load::evaluate(&[Value::String(path_text)])
            .await
            .map_err(|err| open_flow_error("open", err))?;
        return Ok(eval.first_output());
    }

    if let Some(handler_result) = call_extension_handler(&extension, path).await? {
        return Ok(handler_result);
    }

    if is_viewer_or_editor_extension(&extension) {
        return Ok(empty_array());
    }

    Err(open_error(
        &OPEN_ERROR_UNSUPPORTED,
        format!(
            "open: unsupported file type '{}' for '{}'",
            extension,
            path.display()
        ),
    ))
}

async fn open_name(value: &Value) -> BuiltinResult<String> {
    let gathered = gather_if_needed_async(value)
        .await
        .map_err(|err| open_flow_error("open", err))?;
    match gathered {
        Value::String(text) => Ok(text),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect::<String>()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        _ => Err(open_error(&OPEN_ERROR_NAME, OPEN_ERROR_NAME.message)),
    }
}

async fn resolve_open_path(name: &str) -> BuiltinResult<PathBuf> {
    if let Some(path) = find_exact_open_file(name).await? {
        return Ok(vfs::canonicalize_async(&path).await.unwrap_or(path));
    }

    let path = find_file_with_extensions(name, GENERAL_FILE_EXTENSIONS, BUILTIN_NAME)
        .await
        .map_err(|err| open_error(&OPEN_ERROR_NOT_FOUND, err))?
        .ok_or_else(|| {
            open_error(
                &OPEN_ERROR_NOT_FOUND,
                format!("open: file or variable was not found: {name}"),
            )
        })?;

    Ok(vfs::canonicalize_async(&path).await.unwrap_or(path))
}

async fn find_exact_open_file(name: &str) -> BuiltinResult<Option<PathBuf>> {
    if should_treat_as_path(name) {
        return Ok(None);
    }
    let Some(candidate_name) = exact_filename_candidate(name) else {
        return Ok(None);
    };
    for dir in
        search_directories(BUILTIN_NAME).map_err(|err| open_error(&OPEN_ERROR_NOT_FOUND, err))?
    {
        let candidate = dir.join(candidate_name);
        if path_is_file(&candidate).await {
            return Ok(Some(candidate));
        }
    }
    Ok(None)
}

fn exact_filename_candidate(name: &str) -> Option<&str> {
    let path = Path::new(name);
    (path.file_name().is_some() && path.extension().is_some()).then_some(name)
}

async fn call_extension_handler(extension: &str, path: &Path) -> BuiltinResult<Option<Value>> {
    if extension.is_empty() {
        return Ok(None);
    }
    let handler_name = format!("open{extension}");
    let requested_outputs = crate::output_context::requested_output_count().unwrap_or(1);
    let requested_outputs = usize::from(requested_outputs > 0);
    let Some(result) = crate::user_functions::try_call_semantic_function_by_name(
        &handler_name,
        &[Value::String(path_to_string(path))],
        requested_outputs,
    )
    .await
    else {
        return Ok(None);
    };
    result.map(Some).map_err(|err| {
        let message = format!("{handler_name}: {}", err.message());
        let mut builder = build_runtime_error(message)
            .with_builtin(BUILTIN_NAME)
            .with_source(err);
        if let Some(identifier) = OPEN_ERROR_HANDLER.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    })
}

async fn try_open_runmat_figure_scene(path: &Path) -> BuiltinResult<Option<u32>> {
    #[cfg(feature = "plot-core")]
    {
        let text = path_to_string(path);
        match crate::builtins::plotting::import_figure_scene_from_path_async(&text).await {
            Ok(Some(handle)) => Ok(Some(handle.as_u32())),
            Ok(None) => Ok(None),
            Err(_) => Ok(None),
        }
    }
    #[cfg(not(feature = "plot-core"))]
    {
        let _ = path;
        Ok(None)
    }
}

fn normalized_extension(path: &Path) -> String {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.trim_start_matches('.').to_ascii_lowercase())
        .unwrap_or_default()
}

fn is_viewer_or_editor_extension(extension: &str) -> bool {
    matches!(
        extension,
        "m" | "mlx"
            | "mlapp"
            | "txt"
            | "csv"
            | "json"
            | "xml"
            | "dat"
            | "html"
            | "htm"
            | "pdf"
            | "doc"
            | "docx"
            | "ppt"
            | "pptx"
            | "xls"
            | "xlsx"
            | "xlsm"
            | "mdl"
            | "slx"
            | "slxc"
            | "prj"
            | "mltbx"
            | "mlappinstall"
            | "exe"
    )
}

fn requested_output_count() -> usize {
    crate::output_count::current_output_count()
        .or_else(crate::output_context::requested_output_count)
        .unwrap_or(1)
}

fn empty_array() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

fn open_error(error: &'static BuiltinErrorDescriptor, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn open_flow_error(context: &str, err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(str::to_string);
    let mut builder = build_runtime_error(format!("{context}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::io::mat::save::encode_workspace_to_mat_bytes;
    use crate::workspace::WorkspaceResolver;
    use futures::executor::block_on;
    use runmat_builtins::Tensor;
    use runmat_thread_local::runmat_thread_local;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use tempfile::tempdir;

    runmat_thread_local! {
        static TEST_WORKSPACE: RefCell<HashMap<String, Value>> = RefCell::new(HashMap::new());
    }

    fn ensure_test_resolver() {
        crate::workspace::register_workspace_resolver(WorkspaceResolver {
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

    fn workspace_guard() -> std::sync::MutexGuard<'static, ()> {
        crate::workspace::test_guard()
    }

    struct CurrentDirGuard {
        original: PathBuf,
    }

    impl CurrentDirGuard {
        fn set(path: &Path) -> Self {
            let original = std::env::current_dir().expect("current dir");
            std::env::set_current_dir(path).expect("set current dir");
            Self { original }
        }
    }

    impl Drop for CurrentDirGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.original);
        }
    }

    fn call(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(open_builtin(args))
    }

    #[test]
    fn open_mat_file_returns_loaded_struct() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("sample.mat");
        let tensor = Tensor::new_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2).expect("tensor");
        let bytes = block_on(encode_workspace_to_mat_bytes(&[(
            "A".to_string(),
            Value::Tensor(tensor),
        )]))
        .expect("encode mat");
        std::fs::write(&path, bytes).expect("write mat");

        let result = call(vec![Value::String(path.to_string_lossy().to_string())]).expect("open");
        match result {
            Value::Struct(st) => {
                assert!(matches!(st.fields.get("A"), Some(Value::Tensor(_))));
            }
            other => panic!("expected struct from open mat, got {other:?}"),
        }
    }

    #[test]
    fn open_variable_name_is_noop_editor_open() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[("answer", Value::Num(42.0))]);

        let result = call(vec![Value::from("answer")]).expect("open variable");

        assert_eq!(result, empty_array());
    }

    #[test]
    fn open_resolves_m_file_on_current_path() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[]);
        let dir = tempdir().expect("tempdir");
        std::fs::write(dir.path().join("helper.m"), "disp('ok')").expect("write m");
        let _cwd = CurrentDirGuard::set(dir.path());

        let result = call(vec![Value::from("helper")]).expect("open m-file");

        assert_eq!(result, empty_array());
    }

    #[test]
    fn open_calls_custom_extension_handler() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[]);
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("events.log");
        std::fs::write(&path, "entry").expect("write log");
        let seen = Arc::new(Mutex::new(String::new()));
        let seen_for_resolver = Arc::clone(&seen);
        let _resolver = crate::user_functions::install_semantic_function_resolver(Some(Arc::new(
            move |name| (name == "openlog").then_some(77),
        )));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |_function, args, requested_outputs| {
                assert_eq!(requested_outputs, 1);
                let Value::String(path) = &args[0] else {
                    panic!("expected handler path argument");
                };
                *seen_for_resolver.lock().unwrap() = path.clone();
                Box::pin(async { Ok(Value::String("handled".to_string())) })
            },
        )));

        let result = call(vec![Value::String(path.to_string_lossy().to_string())]).expect("open");

        assert_eq!(result, Value::String("handled".to_string()));
        assert!(seen.lock().unwrap().ends_with("events.log"));
    }

    #[test]
    fn open_resolves_custom_extension_file_on_current_path() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[]);
        let dir = tempdir().expect("tempdir");
        std::fs::write(dir.path().join("events.log"), "entry").expect("write log");
        let _cwd = CurrentDirGuard::set(dir.path());
        let _resolver = crate::user_functions::install_semantic_function_resolver(Some(Arc::new(
            move |name| (name == "openlog").then_some(77),
        )));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |_function, args, _requested_outputs| {
                let Value::String(path) = &args[0] else {
                    panic!("expected handler path argument");
                };
                assert!(path.ends_with("events.log"));
                Box::pin(async { Ok(Value::String("handled".to_string())) })
            },
        )));

        let result = call(vec![Value::from("events.log")]).expect("open");

        assert_eq!(result, Value::String("handled".to_string()));
    }

    #[test]
    fn open_resolves_known_viewer_extension_file_on_current_path() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[]);
        let dir = tempdir().expect("tempdir");
        std::fs::write(dir.path().join("report.pdf"), "pdf").expect("write pdf");
        let _cwd = CurrentDirGuard::set(dir.path());

        let result = call(vec![Value::from("report.pdf")]).expect("open pdf");

        assert_eq!(result, empty_array());
    }

    #[test]
    fn open_accepts_project_and_model_file_classes_as_viewer_requests() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[]);
        let dir = tempdir().expect("tempdir");
        for file in ["model.slx", "legacy.mdl", "project.prj", "cache.slxc"] {
            std::fs::write(dir.path().join(file), "payload").expect("write viewer file");
        }
        let _cwd = CurrentDirGuard::set(dir.path());

        for file in ["model.slx", "legacy.mdl", "project.prj", "cache.slxc"] {
            let result = call(vec![Value::from(file)]).expect("open viewer file");
            assert_eq!(result, empty_array());
        }
    }

    #[test]
    fn open_preserves_spaces_in_string_filename() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[]);
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join(" spaced name.txt");
        std::fs::write(&path, "text").expect("write spaced filename");

        let result = call(vec![Value::String(path.to_string_lossy().to_string())])
            .expect("open spaced filename");

        assert_eq!(result, empty_array());
    }

    #[test]
    fn open_rejects_more_than_one_output() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[]);
        let _outputs = crate::output_count::push_output_count(Some(2));

        let err = call(vec![Value::from("anything")]).expect_err("too many outputs");

        assert_eq!(err.identifier(), Some("RunMat:open:TooManyOutputs"));
    }

    #[test]
    fn open_unsupported_file_type_errors() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[]);
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("payload.unknown");
        std::fs::write(&path, "payload").expect("write file");

        let err =
            call(vec![Value::String(path.to_string_lossy().to_string())]).expect_err("unsupported");

        assert_eq!(err.identifier(), Some("RunMat:open:UnsupportedFileType"));
    }

    #[test]
    fn open_rejects_missing_file_or_variable() {
        let _guard = workspace_guard();
        ensure_test_resolver();
        set_workspace(&[]);

        let err =
            call(vec![Value::from("definitely_missing_open_target")]).expect_err("missing target");

        assert_eq!(err.identifier(), Some("RunMat:open:NotFound"));
    }

    #[test]
    fn open_descriptor_covers_statement_and_output_forms() {
        let labels: Vec<&str> = OPEN_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"open name"));
        assert!(labels.contains(&"A = open(name)"));
    }
}
