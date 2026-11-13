use super::*;

mod selected_context;

#[test]
fn invalid_names_reject_without_host_or_provider_access() {
    let resident = Value::GpuTensor(
        crate::builtins::io::repl_fs::test_support::unregistered_numeric_allocation(vec![1, 1])
            .handle,
    );
    let pages = Value::CharArray(
        runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2])
            .unwrap(),
    );
    for value in [resident, pages, Value::Num(1.0)] {
        assert_eq!(
            block_on(open_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![value])
            ))
            .unwrap_err()
            .identifier(),
            OPEN_ERROR_NAME.identifier
        );
    }
    let row = Value::CharArray(
        runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 2, 1])
            .unwrap(),
    );
    assert_eq!(open_name(&row).unwrap(), "ab");
}
use crate::builtins::io::mat::save::encode_workspace_to_mat_bytes;
use crate::workspace::WorkspaceResolver;
use futures::executor::block_on;
use runmat_thread_local::runmat_thread_local;
use runmat_value::Tensor;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
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

fn call(args: Vec<Value>) -> BuiltinResult<crate::execution::RootedValue> {
    let outputs = crate::output_count::current_output_count();
    let (context, _) = crate::context::document_test_support::recording_context();
    let selected = context.clone();
    block_on(context.scope(async move {
        let _outputs = crate::output_count::push_output_count(outputs).expect("output-count scope");
        open_builtin(
            Some(selected),
            crate::builtins::common::test_support::builtin_arguments(args),
        )
        .await
    }))
}

fn selected_openfig(
    invocation: crate::builtin::RuntimeBuiltinInvocation<'_>,
) -> crate::builtin::RuntimeBuiltinFuture {
    assert_eq!(invocation.requested_outputs, Some(1));
    assert!(invocation.context.is_some());
    let Some(Value::String(runmat_value::StringElement::Text(path))) = invocation.args.get(0)
    else {
        panic!("openfig must receive one resolved path");
    };
    assert_eq!(invocation.args.len(), 1);
    assert!(path.ends_with("selected.fig"));
    crate::builtin::admitted_builtin_future(None, "fixture", async {
        let sequence = runmat_value::ValueSequence::single(Value::Num(90_210.0))
            .map_err(crate::sequence::sequence_error_to_runtime)?;
        crate::execution::RootedValueSequence::new(sequence)
            .map_err(|error| error.into_runtime_error())
    })
}

struct SelectedOpenfigService;

impl crate::context::RuntimeBuiltinService for SelectedOpenfigService {
    fn bindings_by_name(&self, name: &str) -> Vec<crate::builtin::RuntimeBuiltinBinding> {
        if name == "open" {
            return crate::builtin::runtime_builtin_bindings_by_name(name);
        }
        if name != "openfig" {
            return Vec::new();
        }
        vec![crate::builtin::RuntimeBuiltinBinding::new(
            runmat_builtins::BuiltinBindingIdentity {
                builtin: runmat_builtins::BuiltinCatalogIdentity { name: "openfig" },
                variant: "default",
            },
            selected_openfig,
        )]
    }
}

#[test]
fn figure_archive_uses_the_exact_selected_openfig_binding() {
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join("selected.fig");
    std::fs::write(&path, b"selected binding fixture").expect("write fixture");
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let ports = runtime
        .service_ports()
        .clone()
        .with_builtin(Rc::new(SelectedOpenfigService));
    let runtime = runtime.with_service_ports(ports);

    let result = block_on(crate::call_builtin_async_with_outputs_in_context(
        &runtime,
        "open",
        &[Value::String(path.to_string_lossy().into_owned().into())],
        1,
    ))
    .expect("selected openfig binding");

    assert_eq!(result, Value::Num(90_210.0));
}

#[test]
fn corrupt_figure_archive_preserves_openfig_error_without_mat_fallthrough() {
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join("corrupt.fig");
    std::fs::write(&path, b"not a RunMat figure archive").expect("write fixture");

    let error = call(vec![Value::String(
        path.to_string_lossy().into_owned().into(),
    )])
    .expect_err("corrupt figure archive");

    assert_eq!(
        error.identifier(),
        Some("RunMat:figurePersistence:ImportFailed")
    );
    assert_ne!(error.identifier(), Some("RunMat:load:InvalidArgument"));
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

    let result = call(vec![Value::String(
        path.to_string_lossy().to_string().into(),
    )])
    .expect("open");
    match result.value() {
        Value::Struct(st) => {
            assert!(matches!(st.fields.get("A"), Some(Value::Tensor(_))));
        }
        other => panic!("expected struct from open mat, got {other:?}"),
    }
}

#[test]
fn open_variable_name_requests_host_editor() {
    let _guard = workspace_guard();
    ensure_test_resolver();
    set_workspace(&[("answer", Value::Num(42.0))]);

    let (context, recorder) = crate::context::document_test_support::recording_context();
    let result = block_on(context.scope(open_builtin(
        Some(context.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("answer")]),
    )))
    .expect("open variable");

    assert_eq!(result, empty_array());
    assert_eq!(
        *recorder.requests.borrow(),
        vec![crate::context::HostDocumentRequest::OpenVariable {
            name: "answer".into()
        }]
    );
}

#[test]
fn editor_file_requires_a_host_and_delivers_the_resolved_path() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("editor.m");
    std::fs::write(&path, "value = 1;\n").unwrap();
    let args = vec![Value::from(path.to_string_lossy().into_owned())];
    let error = block_on(open_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(args.clone()),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:host:DocumentInteractionUnavailable")
    );
    let (context, recorder) = crate::context::document_test_support::recording_context();
    block_on(context.scope(open_builtin(
        Some(context.clone()),
        crate::builtins::common::test_support::builtin_arguments(args),
    )))
    .unwrap();
    assert_eq!(
        *recorder.requests.borrow(),
        vec![crate::context::HostDocumentRequest::OpenFile {
            path: std::fs::canonicalize(path).unwrap()
        }]
    );
}

#[test]
fn zero_output_extension_handler_is_handled_without_viewer_fallthrough() {
    let _guard = workspace_guard();
    let directory = tempdir().unwrap();
    let path = directory.path().join("custom.audit");
    std::fs::write(&path, "fixture").unwrap();
    let _resolver =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            (name == "openaudit").then_some(88)
        })))
        .expect("call-port scope");
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |_, _, requested| {
            assert_eq!(requested, 0);
            Box::pin(async {
                crate::execution::RootedValueSequence::new(runmat_value::ValueSequence::empty())
                    .map_err(|error| error.into_runtime_error())
            })
        },
    )))
    .expect("call-port scope");
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let _semantic_outputs =
        crate::output_context::push_output_count(0).expect("output-count scope");
    assert_eq!(
        block_on(open_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from(
                path.to_string_lossy().into_owned()
            )])
        ))
        .unwrap(),
        empty_array()
    );
}

#[test]
fn open_resolves_m_file_on_current_path() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
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
    let _resolver =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(move |name| {
            (name == "openlog").then_some(77)
        })))
        .expect("call-port scope");
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |_function, args, requested_outputs| {
            assert_eq!(requested_outputs, 1);
            let Value::String(runmat_value::StringElement::Text(path)) = &args[0] else {
                panic!("expected handler path argument");
            };
            *seen_for_resolver.lock().unwrap() = path.to_string();
            Box::pin(async {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::String(
                    "handled".to_string().into(),
                ))
            })
        },
    )))
    .expect("call-port scope");

    let result = call(vec![Value::String(
        path.to_string_lossy().to_string().into(),
    )])
    .expect("open");

    assert_eq!(result, Value::String("handled".to_string().into()));
    assert!(seen.lock().unwrap().ends_with("events.log"));
}

#[test]
fn open_resolves_custom_extension_file_on_current_path() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let _guard = workspace_guard();
    ensure_test_resolver();
    set_workspace(&[]);
    let dir = tempdir().expect("tempdir");
    std::fs::write(dir.path().join("events.log"), "entry").expect("write log");
    let _cwd = CurrentDirGuard::set(dir.path());
    let _resolver =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(move |name| {
            (name == "openlog").then_some(77)
        })))
        .expect("call-port scope");
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |_function, args, _requested_outputs| {
            let Value::String(runmat_value::StringElement::Text(path)) = &args[0] else {
                panic!("expected handler path argument");
            };
            assert!(path.ends_with("events.log"));
            Box::pin(async {
                crate::user_functions::test_support::rooted_single_value_sequence(Value::String(
                    "handled".to_string().into(),
                ))
            })
        },
    )))
    .expect("call-port scope");

    let result = call(vec![Value::from("events.log")]).expect("open");

    assert_eq!(result, Value::String("handled".to_string().into()));
}

#[test]
fn open_resolves_known_viewer_extension_file_on_current_path() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
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
    let _fs_lock = runmat_filesystem::provider_override_lock();
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

    let result = call(vec![Value::String(
        path.to_string_lossy().to_string().into(),
    )])
    .expect("open spaced filename");

    assert_eq!(result, empty_array());
}

#[test]
fn open_rejects_more_than_one_output() {
    let _guard = workspace_guard();
    ensure_test_resolver();
    set_workspace(&[]);
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
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

    let err = call(vec![Value::String(
        path.to_string_lossy().to_string().into(),
    )])
    .expect_err("unsupported");

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
