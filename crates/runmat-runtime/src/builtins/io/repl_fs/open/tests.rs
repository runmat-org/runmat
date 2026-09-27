use super::*;

#[test]
fn invalid_names_reject_without_host_or_provider_access() {
    let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
        shape: vec![1, 1],
        device_id: u32::MAX,
        buffer_id: u64::MAX,
        descriptor: Default::default(),
    });
    let pages = Value::CharArray(
        runmat_value::CharArray::new_with_shape(vec!['a', 'b'], vec![1, 1, 2]).unwrap(),
    );
    for value in [resident, pages, Value::Num(1.0)] {
        assert_eq!(
            block_on(open_builtin(vec![value]))
                .unwrap_err()
                .identifier(),
            OPEN_ERROR_NAME.identifier
        );
    }
    let row = Value::CharArray(
        runmat_value::CharArray::new_with_shape(vec!['a', 'b'], vec![1, 2, 1]).unwrap(),
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
    let outputs = crate::output_count::current_output_count();
    let (context, _) = crate::context::document_test_support::recording_context();
    block_on(context.scope(async move {
        let _outputs = crate::output_count::push_output_count(outputs);
        open_builtin(args).await
    }))
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
fn open_variable_name_requests_host_editor() {
    let _guard = workspace_guard();
    ensure_test_resolver();
    set_workspace(&[("answer", Value::Num(42.0))]);

    let (context, recorder) = crate::context::document_test_support::recording_context();
    let result =
        block_on(context.scope(open_builtin(vec![Value::from("answer")]))).expect("open variable");

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
    let error = block_on(open_builtin(args.clone())).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:host:DocumentInteractionUnavailable")
    );
    let (context, recorder) = crate::context::document_test_support::recording_context();
    block_on(context.scope(open_builtin(args))).unwrap();
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
        })));
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        |_, _, requested| {
            assert_eq!(requested, 0);
            Box::pin(async { Ok(runmat_value::ValueSequence::empty()) })
        },
    )));
    let _outputs = crate::output_count::push_output_count(Some(0));
    let _semantic_outputs = crate::output_context::push_output_count(0);
    assert_eq!(
        block_on(open_builtin(vec![Value::from(
            path.to_string_lossy().into_owned()
        )]))
        .unwrap(),
        empty_array()
    );
}

#[test]
fn open_resolves_m_file_on_current_path() {
    let _guard = workspace_guard();
    let _fs_lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
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
        })));
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |_function, args, requested_outputs| {
            assert_eq!(requested_outputs, 1);
            let Value::String(path) = &args[0] else {
                panic!("expected handler path argument");
            };
            *seen_for_resolver.lock().unwrap() = path.clone();
            Box::pin(async {
                crate::sequence::single_value_sequence(Value::String("handled".to_string()))
            })
        },
    )));

    let result = call(vec![Value::String(path.to_string_lossy().to_string())]).expect("open");

    assert_eq!(result, Value::String("handled".to_string()));
    assert!(seen.lock().unwrap().ends_with("events.log"));
}

#[test]
fn open_resolves_custom_extension_file_on_current_path() {
    let _guard = workspace_guard();
    let _fs_lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    ensure_test_resolver();
    set_workspace(&[]);
    let dir = tempdir().expect("tempdir");
    std::fs::write(dir.path().join("events.log"), "entry").expect("write log");
    let _cwd = CurrentDirGuard::set(dir.path());
    let _resolver =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(move |name| {
            (name == "openlog").then_some(77)
        })));
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |_function, args, _requested_outputs| {
            let Value::String(path) = &args[0] else {
                panic!("expected handler path argument");
            };
            assert!(path.ends_with("events.log"));
            Box::pin(async {
                crate::sequence::single_value_sequence(Value::String("handled".to_string()))
            })
        },
    )));

    let result = call(vec![Value::from("events.log")]).expect("open");

    assert_eq!(result, Value::String("handled".to_string()));
}

#[test]
fn open_resolves_known_viewer_extension_file_on_current_path() {
    let _guard = workspace_guard();
    let _fs_lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
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
    let _fs_lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
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
