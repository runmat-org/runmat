use super::*;
use futures::executor::block_on;
use once_cell::sync::Lazy;
use runmat_thread_local::runmat_thread_local;
use runmat_value::{CharArray, StringArray, Value};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use tempfile::tempdir;

#[cfg(not(target_arch = "wasm32"))]
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, GpuTensorHandle,
    HostNumericTensorView, HostTensorView, ThreadProviderGuard,
};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(not(target_arch = "wasm32"))]
struct CountingProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    downloads: AtomicUsize,
}

#[cfg(not(target_arch = "wasm32"))]
impl CountingProvider {
    fn new() -> &'static Self {
        Box::leak(Box::new(Self {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            downloads: AtomicUsize::new(0),
        }))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl AccelProvider for CountingProvider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download(handle)
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.inner.upload_numeric_raw(host)
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download_numeric(handle)
    }
}

static WHICH_TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

fn workspace_guard() -> std::sync::MutexGuard<'static, ()> {
    crate::workspace::test_guard()
}

fn test_guard() -> (
    std::sync::MutexGuard<'static, ()>,
    std::sync::MutexGuard<'static, ()>,
) {
    crate::builtins::wasm_registry::register_all();
    let workspace = workspace_guard();
    let which = WHICH_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    (workspace, which)
}

fn which_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    block_on(super::which_builtin(
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
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

fn error_message(err: crate::RuntimeError) -> String {
    err.message().to_string()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn builtin_lookup_includes_catalog_owned_and_legacy_names_once() {
    let (_guard, _lock) = test_guard();
    assert!(runmat_builtins::builtin_catalog_entry_by_name("sin").is_some());
    assert!(runmat_builtins::builtin_function_by_name("sin").is_none());
    assert_eq!(
        block_on(search_matches("SiN", &WhichOptions::default())).unwrap(),
        ["built-in (RunMat builtin: sin)"]
    );
    assert_eq!(
        block_on(search_matches("which", &WhichOptions::default())).unwrap(),
        ["built-in (RunMat builtin: which)"]
    );
    assert!(block_on(search_matches(
        "__runmat_nonexistent_which_fixture__",
        &WhichOptions::default(),
    ))
    .unwrap()
    .is_empty());
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
fn which_observes_workspace_and_semantic_shadow_without_invocation() {
    let (_guard, _lock) = test_guard();
    ensure_test_resolver();
    set_workspace(&[("sin", Value::Num(7.0))]);
    let _resolver =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            (name == "sin").then_some(771)
        })))
        .expect("call-port scope");
    let _invoker =
        crate::user_functions::install_semantic_function_invoker(Some(Arc::new(|_, _, _| {
            panic!("which must not invoke a function")
        })))
        .expect("call-port scope");
    let resolution = block_on(observe_runtime_name(
        "sin",
        RuntimeNameQuery {
            route: RuntimeNameRoute::DynamicCall,
            filter: RuntimeNameFilter::Any,
            all: true,
        },
    ))
    .expect("read-only resolution");
    assert!(matches!(
        resolution.selected(),
        Some(RuntimeNameCandidate::WorkspaceVariable)
    ));
    assert!(matches!(
        resolution.shadowed().first(),
        Some(RuntimeNameCandidate::Semantic { function: 771, .. })
    ));
    assert!(resolution.candidates.iter().any(
        |candidate| matches!(candidate, RuntimeNameCandidate::Builtin(name) if name.0 == "sin")
    ));
    let feval = block_on(observe_runtime_name(
        "sin",
        RuntimeNameQuery {
            route: RuntimeNameRoute::NamedFeval,
            filter: RuntimeNameFilter::Any,
            all: true,
        },
    ))
    .expect("named feval resolution");
    assert!(matches!(
        feval.selected(),
        Some(RuntimeNameCandidate::Semantic { function: 771, .. })
    ));
    assert!(matches!(
        feval.shadowed().first(),
        Some(RuntimeNameCandidate::Builtin(name)) if name.0 == "sin"
    ));
    let value = which_builtin(vec![Value::from("sin")]).expect("which");
    assert_eq!(String::try_from(&value).unwrap(), "'sin' is a variable.");
    set_workspace(&[]);
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
    let _fs_lock = runmat_filesystem::provider_override_lock();
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
        Value::Cell(cell) => assert!(!cell.data().is_empty()),
        other => panic!("expected cell array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn which_all_keeps_distinct_definitions_at_the_same_display_path() {
    let (_guard, _lock) = test_guard();
    let _resolver =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            (name == "reused_definition").then_some(17)
        })))
        .expect("call-port scope");
    let _invoker =
        crate::user_functions::install_semantic_function_invoker(Some(Arc::new(|_, _, _| {
            panic!("which must not invoke either definition")
        })))
        .expect("call-port scope");
    let first = Arc::new(runmat_value::FunctionDefinitionOwner::default());
    let second = Arc::new(runmat_value::FunctionDefinitionOwner::default());
    let _candidates =
        crate::user_functions::install_semantic_name_candidates(Some(Arc::new(move |name| {
            if name != "reused_definition" {
                return Vec::new();
            }
            [Arc::clone(&first), Arc::clone(&second)]
                .into_iter()
                .map(|owner| crate::user_functions::SemanticNameCandidate {
                    function: 17,
                    source: Some(crate::user_functions::SourceFunctionInfo {
                        source_id: runmat_types::SourceId(1),
                        name: name.into(),
                        function: 17,
                        role: runmat_types::FunctionSourceRole::Primary,
                        binding_origin: None,
                        source_file: Some("same/display/path.m".into()),
                        definition_owners: None,
                    }),
                    definition_owner: Some(owner),
                })
                .collect()
        })))
        .expect("call-port scope");
    let results = block_on(search_matches(
        "reused_definition",
        &WhichOptions {
            all: true,
            mode: WhichMode::Any,
        },
    ))
    .expect("all definitions");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0], results[1]);
    assert!(results[0].ends_with("same/display/path.m"));
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
fn which_all_without_matches_returns_char_notice() {
    let (_guard, _lock) = test_guard();
    let value =
        which_builtin(vec![Value::from("-all"), Value::from("definitely_missing")]).expect("which");
    let text = String::try_from(&value).expect("char row, not a cell");
    assert_eq!(text, "'definitely_missing' not found.");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn which_decodes_padded_case_variant_all_option() {
    let (_guard, _lock) = test_guard();
    let value = which_builtin(vec![Value::from(" \t-ALL \n"), Value::from("sin")]).expect("which");
    match value {
        Value::Cell(cell) => assert!(!cell.data().is_empty()),
        other => panic!("expected cell array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn which_parses_leading_option() {
    let (_guard, _lock) = test_guard();
    let value = which_builtin(vec![Value::from("-all"), Value::from("sin")]).expect("which");
    match value {
        Value::Cell(cell) => assert!(!cell.data().is_empty()),
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
    assert_eq!(message, WHICH_ERROR_ARG_COUNT_MIN.message);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn which_errors_on_non_string_name() {
    let (_guard, _lock) = test_guard();
    let err = which_builtin(vec![Value::Num(4.0)]).unwrap_err();
    let message = error_message(err);
    assert_eq!(message, WHICH_ERROR_NAME_ARG_TYPE.message);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn numeric_gpu_name_and_option_reject_before_provider_download() {
    let (_workspace, _which) = test_guard();
    let _accel = crate::builtins::common::test_support::accel_test_lock();
    let provider = CountingProvider::new();
    let _selected = ThreadProviderGuard::set(Some(provider));
    let numeric = runmat_value::Tensor::new(vec![115.0], vec![1, 1]).expect("numeric input");
    let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &numeric)
        .expect("resident numeric input");
    assert!(handle.descriptor().element_type.is_some());
    let name_error = which_builtin(vec![Value::GpuTensor(handle.clone())])
        .expect_err("numeric GPU handle is not a text name");
    assert_eq!(name_error.message(), WHICH_ERROR_NAME_ARG_TYPE.message);
    let option_error = which_builtin(vec![Value::from("sin"), Value::GpuTensor(handle)])
        .expect_err("numeric GPU handle is not a text option");
    assert_eq!(option_error.message(), WHICH_ERROR_OPTION_ARG_TYPE.message);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
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
    assert_eq!(message, WHICH_ERROR_ARG_COUNT_MAX.message);
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
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let (_guard, _lock) = test_guard();
    let temp = tempdir().expect("tempdir");
    let subdir = temp.path().join("helpers");
    std::fs::create_dir_all(&subdir).expect("create dir");
    let guard = DirGuard::new();
    std::env::set_current_dir(temp.path()).expect("set temp dir");

    let value = which_builtin(vec![Value::from("-file"), Value::from("helpers")]).expect("which");
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
