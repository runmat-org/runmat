use super::*;
use crate::BuiltinResult;
use runmat_filesystem as vfs;
use runmat_thread_local::runmat_thread_local;
use runmat_value::{IntValue, Value};
use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use tempfile::tempdir;

#[test]
fn output_count_rejects_before_lookup_and_zero_still_queries() {
    let (_guard, _lock) = test_guard();
    {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = exist_builtin(Value::from("sin"), vec![Value::from("builtin")]).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:exist:TooManyOutputs"));
    }
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    assert_eq!(
        exist_builtin(Value::from("sin"), vec![Value::from("builtin")]).unwrap(),
        Value::Num(5.0)
    );
    assert_eq!(
        exist_builtin(Value::Bool(true), Vec::new())
            .unwrap_err()
            .identifier(),
        Some("RunMat:exist:InvalidName")
    );
}

#[test]
fn character_pages_reject_and_trailing_singletons_preserve_names() {
    let (_guard, _lock) = test_guard();
    let pages = Value::CharArray(
        runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2])
            .unwrap(),
    );
    assert_eq!(
        exist_builtin(pages.clone(), Vec::new())
            .unwrap_err()
            .identifier(),
        Some("RunMat:exist:InvalidName")
    );
    assert_eq!(
        exist_builtin(Value::from("sin"), vec![pages])
            .unwrap_err()
            .identifier(),
        Some("RunMat:exist:InvalidTypeArgument")
    );
    let row = Value::CharArray(
        runmat_value::CharArray::from_unicode_scalars_with_shape(
            vec!['s', 'i', 'n'],
            vec![1, 3, 1],
        )
        .unwrap(),
    );
    assert_eq!(
        exist_builtin(row, vec![Value::from("builtin")]).unwrap(),
        Value::Num(5.0)
    );
}

static TEST_CLASS_COUNTER: AtomicU64 = AtomicU64::new(0);

fn unique_class_name(prefix: &str) -> String {
    let id = TEST_CLASS_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{}_{}", prefix, id)
}

fn exist_builtin(name: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    futures::executor::block_on(super::exist_builtin(
        name,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
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
    let fs_lock = runmat_filesystem::provider_override_lock();
    let workspace = workspace_guard();
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

    let builtin = exist_builtin(Value::from("sin"), vec![Value::from("builtin")]).expect("exist");
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
    File::create("data.txt").expect("create text file");
    File::create("packed.p").expect("create p-code file");

    let script = exist_builtin(Value::from("script"), vec![Value::from("file")]).expect("exist");
    assert_eq!(script, Value::Num(2.0));

    let mex = exist_builtin(Value::from("fastfft"), vec![Value::from("file")]).expect("exist");
    assert_eq!(mex, Value::Num(3.0));

    let mex_specific =
        exist_builtin(Value::from("fastfft"), vec![Value::from("mex")]).expect("exist");
    assert_eq!(mex_specific, Value::Num(3.0));

    let text =
        exist_builtin(Value::from("data.txt"), vec![Value::from("file")]).expect("exist text file");
    assert_eq!(text, Value::Num(2.0));

    let pcode = exist_builtin(Value::from("packed.p"), vec![Value::from("file")])
        .expect("exist p-code file");
    assert_eq!(pcode, Value::Num(6.0));
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

    let file = exist_builtin(Value::from("data"), vec![Value::from("file")]).expect("exist file");
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

    let widget = exist_builtin(Value::from("Widget"), vec![Value::from("class")]).expect("exist");
    assert_eq!(widget, Value::Num(8.0));

    futures::executor::block_on(vfs::create_dir_async("Widget")).expect("create colliding folder");
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
    let err = exist_builtin(Value::Bool(true), Vec::new()).expect_err("logical name must reject");
    assert_eq!(err.message(), EXIST_ERROR_NAME_ARG.message);
}

#[test]
fn exist_rejects_resident_numeric_name_without_provider_access() {
    let (_guard, _lock) = test_guard();
    let resident = Value::GpuTensor(
        crate::builtins::io::repl_fs::test_support::unregistered_numeric_allocation(vec![1, 1])
            .handle,
    );
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

    let value = exist_builtin(Value::Num(17.0), vec![Value::from("handle")]).expect("exist handle");
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
        "parentOnly".into(),
        crate::class_registry::RuntimeMethod {
            name: "parentOnly".into(),
            is_static: false,
            is_abstract: false,
            is_sealed: false,
            access: MemberAccess::Public,
            function_name: "parent_only_impl".into(),
            implicit_class_argument: None,
            role: runmat_types::ClassMethodRole::Ordinary,
        },
    );
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: parent_name.clone().into(),
        parent: None,
        properties: HashMap::new(),
        methods: parent_methods,
    });
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: child_name.clone().into(),
        parent: Some(parent_name.clone().into()),
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
