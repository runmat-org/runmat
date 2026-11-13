use super::*;
use crate::workspace;
use runmat_builtins::SAVE_DESCRIPTOR;
use std::path::Path;

async fn save_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    super::save_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(args),
    )
    .await
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn known_invalid_option_rejects_before_resident_preparation() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let args = [Value::from("case.mat"), Value::from("-unsupported")];
    let error = preflight_known_options(crate::call::FunctionArgumentView::from_slice(&args))
        .expect_err("unsupported flag");
    assert_eq!(error.identifier(), Some("RunMat:save:InvalidOption"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn regexp_followed_by_another_option_has_no_pattern() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let args = [
        Value::from("case.mat"),
        Value::from("-regexp"),
        Value::from("-append"),
    ];
    let error = preflight_known_options(crate::call::FunctionArgumentView::from_slice(&args))
        .expect_err("missing regexp pattern");
    assert_eq!(error.identifier(), Some("RunMat:save:InvalidOption"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nested_implicit_save_does_not_inherit_outer_captured_status() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let _presentation = crate::output_context::push_output_count(1).expect("output-count scope");
    let error = block_on(crate::dispatcher::call_builtin_async(
        "save",
        &[Value::from("-unsupported")],
    ))
    .expect_err("known invalid option");
    assert_eq!(error.identifier(), Some("RunMat:save:InvalidOption"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn known_invalid_option_does_not_download_prior_resident_argument() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let owner = Box::leak(Box::new(super::super::tests::CountingProvider::new()));
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(owner));
    let resident = owner.resident();
    let error = block_on(save_builtin(vec![
        Value::GpuTensor(resident),
        Value::from("-unsupported"),
    ]))
    .expect_err("known invalid option");
    assert_eq!(error.identifier(), Some("RunMat:save:InvalidOption"));
    assert_eq!(owner.downloads(), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn numeric_resident_text_control_rejects_without_download() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let owner = Box::leak(Box::new(super::super::tests::CountingProvider::new()));
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(owner));
    let resident = owner.resident();
    let error = block_on(save_builtin(vec![Value::GpuTensor(resident)]))
        .expect_err("numeric resident handle cannot be a MAT filename");
    assert_eq!(error.identifier(), Some("RunMat:save:InvalidArgument"));
    assert_eq!(owner.downloads(), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn finite_save_without_complete_resource_evidence_rejects_before_workspace_gather_or_write() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    struct Finite;
    impl crate::context::RuntimeParallelService for Finite {
        fn supports(&self, _: crate::context::ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> crate::context::RuntimeParallelResources {
            crate::context::RuntimeParallelResources {
                memory_available_bytes: Some(1024),
                ..Default::default()
            }
        }
    }
    let filesystem = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        "/mat",
    ));
    let _selected = runmat_filesystem::replace_provider(filesystem.clone());
    runmat_filesystem::set_current_dir("/mat").expect("virtual cwd");
    let owner = Box::leak(Box::new(super::super::tests::CountingProvider::new()));
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(owner));
    set_workspace(&[("gpu_data", Value::GpuTensor(owner.resident()))]);
    let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        crate::context::RuntimeServicePorts::default().with_parallel(std::rc::Rc::new(Finite)),
    );
    let error = block_on(super::save_builtin(
        Some(runtime),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("blocked.mat"),
            Value::from("gpu_data"),
        ]),
    ))
    .expect_err("unknown complete-call memory cannot enter finite route");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    assert_eq!(owner.downloads(), 0);
    let missing = block_on(filesystem.read(std::path::Path::new("/mat/blocked.mat")))
        .expect_err("no destination effect");
    assert_eq!(missing.kind(), std::io::ErrorKind::NotFound);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn selected_outer_save_reports_write_error_once_with_source() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    use super::super::tests::receipt::Witness;
    use runmat_execution::WholeWorkObservationOutcome;

    ensure_test_resolver();
    set_workspace(&[("answer", Value::Num(42.0))]);
    let directory = "/mat";
    let filename = "/mat/readonly.mat";
    let filesystem = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        directory,
    ));
    filesystem
        .write_project_path(filename, b"original")
        .expect("seed destination");
    block_on(filesystem.set_readonly(std::path::Path::new(filename), true))
        .expect("protect destination");
    let _selected = runmat_filesystem::replace_provider(filesystem.clone());
    runmat_filesystem::set_current_dir(directory).expect("virtual cwd");
    let witness = Witness::new(filesystem.clone());
    let error = block_on(super::save_builtin(
        Some(witness.runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from(filename),
            Value::from("answer"),
        ]),
    ))
    .expect_err("provider write failure must propagate");
    assert_eq!(error.identifier(), Some("RunMat:save:Io"));
    assert!(error.source.is_some(), "retain provider IO cause");
    assert_eq!(witness.selections(), 1);
    assert_eq!(witness.probes(), 0);
    assert_eq!(
        witness.outcomes(),
        vec![WholeWorkObservationOutcome::OperationalFailure]
    );
    assert_eq!(
        block_on(filesystem.read(std::path::Path::new(filename))).expect("original bytes"),
        b"original"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn selected_outer_save_observes_only_after_published_mat_payload() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    use super::super::tests::receipt::Witness;
    use runmat_execution::WholeWorkObservationOutcome;

    ensure_test_resolver();
    set_workspace(&[("answer", Value::Num(42.0))]);
    let directory = "/mat";
    let filename = "/mat/published.mat";
    let filesystem = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        directory,
    ));
    let _selected = runmat_filesystem::replace_provider(filesystem.clone());
    runmat_filesystem::set_current_dir(directory).expect("virtual cwd");
    let witness = Witness::new(filesystem.clone());
    let status = block_on(super::save_builtin(
        Some(witness.runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from(filename),
            Value::from("answer"),
        ]),
    ))
    .expect("selected save");
    assert_eq!(status, Value::Num(0.0));
    let bytes = block_on(filesystem.read(std::path::Path::new(filename)))
        .expect("published payload before receipt completion");
    let decoded = crate::builtins::io::mat::load::decode_workspace_from_mat_bytes(&bytes)
        .expect("actual MAT payload");
    assert_eq!(
        decoded.value().0,
        vec![("answer".to_string(), Value::Num(42.0))]
    );
    assert_eq!(witness.selections(), 1);
    assert_eq!(witness.probes(), 0);
    assert_eq!(
        witness.outcomes(),
        vec![WholeWorkObservationOutcome::Success]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn revoked_mat_resource_authority_rejects_outer_save_before_write() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    use super::super::tests::receipt::Witness;

    ensure_test_resolver();
    set_workspace(&[("answer", Value::Num(42.0))]);
    let directory = "/mat";
    let filename = "/mat/revoked.mat";
    let filesystem = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        directory,
    ));
    let _selected = runmat_filesystem::replace_provider(filesystem.clone());
    runmat_filesystem::set_current_dir(directory).expect("virtual cwd");
    let witness = Witness::new(filesystem.clone());
    witness.revoke();
    let error = block_on(super::save_builtin(
        Some(witness.runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from(filename),
            Value::from("answer"),
        ]),
    ))
    .expect_err("revoked resource producer cannot select a write");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:StaleResourceAuthority")
    );
    assert_eq!(witness.selections(), 0);
    assert_eq!(witness.probes(), 0);
    assert!(witness.outcomes().is_empty());
    assert!(block_on(filesystem.read(std::path::Path::new(filename))).is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn selected_outer_save_pending_drop_and_late_cancel_release_without_feedback() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    use super::super::tests::receipt::{PendingWriteFs, Witness};
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};

    ensure_test_resolver();
    set_workspace(&[("answer", Value::Num(42.0))]);
    let directory = "/mat";
    let filename = "/mat/pending.mat";
    let filesystem = std::sync::Arc::new(PendingWriteFs::new(directory));
    let _selected = runmat_filesystem::replace_provider(filesystem.clone());
    runmat_filesystem::set_current_dir(directory).expect("virtual cwd");
    let witness = Witness::new(filesystem.clone());

    let unpolled = super::save_builtin(
        Some(witness.runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from(filename),
            Value::from("answer"),
        ]),
    );
    drop(unpolled);
    assert_eq!(witness.selections(), 0);
    assert_eq!(filesystem.writes(), 0);

    let mut abandoned = Box::pin(super::save_builtin(
        Some(witness.runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from(filename),
            Value::from("answer"),
        ]),
    ));
    assert!(matches!(
        super::super::tests::receipt::poll_once(abandoned.as_mut()),
        std::task::Poll::Pending
    ));
    assert_eq!(witness.selections(), 1);
    assert_eq!(witness.probes(), 1);
    assert_eq!(filesystem.writes(), 1);
    drop(abandoned);
    assert_eq!(filesystem.abandoned(), 1);
    assert_eq!(witness.probes(), 0);
    assert!(witness.outcomes().is_empty());

    let cancellation = CancellationSource::new();
    let runtime = witness
        .runtime
        .clone()
        .with_cancellation_token(cancellation.token());
    let mut cancelled = Box::pin(super::save_builtin(
        Some(runtime),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from(filename),
            Value::from("answer"),
        ]),
    ));
    assert!(matches!(
        super::super::tests::receipt::poll_once(cancelled.as_mut()),
        std::task::Poll::Pending
    ));
    cancellation.cancel(CancellationReason::User);
    let error = block_on(cancelled).expect_err("pending selected write is cancelled");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(witness.selections(), 2);
    assert_eq!(witness.probes(), 0);
    assert_eq!(filesystem.writes(), 2);
    assert_eq!(filesystem.abandoned(), 2);
    assert!(witness.outcomes().is_empty());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn selected_outer_save_pending_nested_resident_gather_cancels_without_write() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    use super::super::tests::pending_provider::PendingDownloadProvider;
    use super::super::tests::receipt::Witness;
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};

    ensure_test_resolver();
    let owner = Box::leak(Box::new(PendingDownloadProvider::new()));
    let _gpu = runmat_accelerate_api::ThreadProviderGuard::set(Some(owner));
    let nested = Value::Cell(
        runmat_value::CellArray::new(vec![Value::GpuTensor(owner.resident())], 1, 1)
            .expect("nested resident payload"),
    );
    set_workspace(&[("nested", nested)]);
    let directory = "/mat";
    let filename = "/mat/nested.mat";
    let filesystem = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        directory,
    ));
    let _selected = runmat_filesystem::replace_provider(filesystem.clone());
    runmat_filesystem::set_current_dir(directory).expect("virtual cwd");
    let witness = Witness::new(filesystem.clone());
    let cancellation = CancellationSource::new();
    let runtime = witness
        .runtime
        .clone()
        .with_cancellation_token(cancellation.token());
    let mut abandoned = Box::pin(super::save_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from(filename),
            Value::from("nested"),
        ]),
    ));
    assert!(matches!(
        super::super::tests::receipt::poll_once(abandoned.as_mut()),
        std::task::Poll::Pending
    ));
    assert_eq!(owner.requests(), 1);
    assert_eq!(witness.selections(), 1);
    assert_eq!(witness.probes(), 1);
    drop(abandoned);
    assert_eq!(owner.abandoned(), 1);
    assert_eq!(witness.probes(), 0);
    assert!(witness.outcomes().is_empty());
    let mut future = Box::pin(super::save_builtin(
        Some(runtime),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from(filename),
            Value::from("nested"),
        ]),
    ));
    assert!(matches!(
        super::super::tests::receipt::poll_once(future.as_mut()),
        std::task::Poll::Pending
    ));
    assert_eq!(owner.requests(), 2);
    assert_eq!(witness.selections(), 2);
    assert_eq!(witness.probes(), 1);
    cancellation.cancel(CancellationReason::User);
    let error = block_on(future).expect_err("pending nested provider readback cancels");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(owner.abandoned(), 2);
    assert_eq!(witness.probes(), 0);
    assert!(witness.outcomes().is_empty());
    assert!(block_on(filesystem.read(std::path::Path::new(filename))).is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn invalid_regexp_rejects_before_workspace_gather() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let owner = Box::leak(Box::new(super::super::tests::CountingProvider::new()));
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(owner));
    set_workspace(&[("gpu_data", Value::GpuTensor(owner.resident()))]);
    let error = block_on(save_builtin(vec![
        Value::from("case.mat"),
        Value::from("-regexp"),
        Value::from("["),
    ]))
    .expect_err("known-invalid regexp before workspace effects");
    assert_eq!(error.identifier(), Some("RunMat:save:InvalidOption"));
    assert_eq!(owner.downloads(), 0);
}
use crate::builtins::common::test_support;
use crate::builtins::common::test_support::filesystem::{exists, tempdir};
use crate::workspace::WorkspaceResolver;
use futures::executor::block_on;
#[cfg(not(target_arch = "wasm32"))]
use once_cell::sync::OnceCell;
use runmat_accelerate_api::HostTensorView;
use runmat_filesystem::{File, FsProvider};
use runmat_thread_local::runmat_thread_local;
use runmat_value::{IntValue, IntegerStorage, NumericDType, StringArray, Tensor};
use std::cell::RefCell;
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Mutex;

runmat_thread_local! {
    static TEST_WORKSPACE: RefCell<HashMap<String, Value>> = RefCell::new(HashMap::new());
}

fn ensure_test_resolver() {
    workspace::register_workspace_resolver(WorkspaceResolver {
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
        let mut map = HashMap::new();
        for (k, v) in entries {
            map.insert(k.to_string(), v.clone());
        }
        *slot.borrow_mut() = map;
    });
}

fn assert_error_contains<T>(result: crate::BuiltinResult<T>, snippet: &str) {
    match result {
        Err(err) => {
            assert!(
                err.message().contains(snippet),
                "expected error to contain '{snippet}', got '{}'",
                err.message()
            );
        }
        Ok(_) => panic!("expected error containing '{snippet}'"),
    }
}

fn assert_saved_double(path: &Path, name: &str, expected: f64) {
    let file = File::open(path).unwrap();
    let mat = matfile::MatFile::parse(file).unwrap();
    let array = mat.find_by_name(name).unwrap();
    match array.data() {
        matfile::NumericData::Double { real, .. } => assert_eq!(real, &[expected]),
        other => panic!("expected double array for {name}, got {other:?}"),
    }
}

#[test]
fn captured_status_is_a_pre_effect_runmat_extension() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    set_workspace(&[("x", Value::Num(4.0))]);
    let folder = tempdir().expect("MAT test folder");
    let path = folder.path().join("status.mat");
    let previous = crate::compatibility::runmat_extensions_enabled();
    crate::compatibility::set_runmat_extensions_enabled(false);
    let _outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let result = block_on(save_builtin(vec![Value::String(
        path.to_string_lossy().into_owned().into(),
    )]));
    crate::compatibility::set_runmat_extensions_enabled(previous);
    let error = result.expect_err("captured status requires RunMat extension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:SaveStatusOutputExtension")
    );
    assert!(!exists(&path), "rejection precedes MAT file publication");
    drop(_outputs);
    let _statement = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let status = block_on(save_builtin(vec![Value::String(
        path.to_string_lossy().into_owned().into(),
    )]))
    .expect("MATLAB-compatible statement form");
    assert_eq!(status, Value::Num(0.0));
    assert_saved_double(&path, "x", 4.0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn save_descriptor_signatures_cover_core_forms() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    let labels: Vec<&str> = SAVE_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"status = save()"));
    assert!(labels.contains(&"status = save(filename)"));
    assert!(labels.contains(&"status = save(filename, varName1, varName2, ...)"));
    assert!(labels.contains(&"status = save(filename, '-struct', structVar, field1, ...)"));
}

#[cfg(not(target_arch = "wasm32"))]
fn lock_env_override() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceCell<Mutex<()>> = OnceCell::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
}

#[cfg(not(target_arch = "wasm32"))]
struct EnvOverride {
    key: &'static str,
}

#[cfg(not(target_arch = "wasm32"))]
impl EnvOverride {
    fn set(key: &'static str, value: &str) -> Self {
        std::env::set_var(key, value);
        EnvOverride { key }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for EnvOverride {
    fn drop(&mut self) {
        std::env::remove_var(self.key);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn save_numeric_variable() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    set_workspace(&[("A", Value::Num(42.0))]);
    let dir = tempdir().unwrap();
    let path = dir.path().join("test_numeric.mat");
    let args = vec![
        Value::from(path.to_string_lossy().to_string()),
        Value::from("A"),
    ];
    block_on(save_builtin(args)).unwrap();

    let file = File::open(&path).unwrap();
    let mat = matfile::MatFile::parse(file).unwrap();
    let array = mat.find_by_name("A").unwrap();
    match array.data() {
        matfile::NumericData::Double { real, .. } => {
            assert_eq!(real, &[42.0]);
        }
        _ => panic!("expected double array"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn save_preserves_supported_numeric_classes() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    let single = Tensor::new_with_dtype(vec![1.25, 2.5], vec![1, 2], NumericDType::F32)
        .expect("single tensor");
    let int8 = Tensor::new_integer(IntegerStorage::I8(vec![i8::MIN, i8::MAX]), vec![1, 2])
        .expect("int8 tensor");
    let uint8 = Tensor::new_integer(IntegerStorage::U8(vec![0, u8::MAX]), vec![1, 2])
        .expect("uint8 tensor");
    let int16 = Tensor::new_integer(IntegerStorage::I16(vec![i16::MIN, i16::MAX]), vec![1, 2])
        .expect("int16 tensor");
    let uint16 = Tensor::new_integer(IntegerStorage::U16(vec![0, u16::MAX]), vec![1, 2])
        .expect("uint16 tensor");
    let int32 = Tensor::new_integer(IntegerStorage::I32(vec![i32::MIN, i32::MAX]), vec![1, 2])
        .expect("int32 tensor");
    let uint32 = Tensor::new_integer(IntegerStorage::U32(vec![0, u32::MAX]), vec![1, 2])
        .expect("uint32 tensor");
    let int64 = Tensor::new_integer(IntegerStorage::I64(vec![i64::MIN, i64::MAX]), vec![1, 2])
        .expect("int64 tensor");
    let uint64 = Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]), vec![1, 2])
        .expect("uint64 tensor");
    set_workspace(&[
        ("single_data", Value::Tensor(single)),
        ("int8_data", Value::Tensor(int8)),
        ("uint8_data", Value::Tensor(uint8)),
        ("int16_data", Value::Tensor(int16)),
        ("uint16_data", Value::Tensor(uint16)),
        ("int32_data", Value::Tensor(int32)),
        ("uint32_data", Value::Tensor(uint32)),
        ("int64_data", Value::Tensor(int64)),
        ("uint64_data", Value::Tensor(uint64)),
        ("i8_scalar", Value::Int(IntValue::I8(-3))),
    ]);

    let dir = tempdir().unwrap();
    let path = dir.path().join("numeric_classes.mat");
    let args = vec![
        Value::from(path.to_string_lossy().to_string()),
        Value::from("single_data"),
        Value::from("int8_data"),
        Value::from("uint8_data"),
        Value::from("int16_data"),
        Value::from("uint16_data"),
        Value::from("int32_data"),
        Value::from("uint32_data"),
        Value::from("int64_data"),
        Value::from("uint64_data"),
        Value::from("i8_scalar"),
    ];
    block_on(save_builtin(args)).unwrap();

    let values: HashMap<_, _> = block_on(crate::builtins::io::mat::load::read_mat_file(&path))
        .unwrap()
        .value()
        .0
        .iter()
        .cloned()
        .collect();
    match values.get("single_data").unwrap() {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
            assert_eq!(tensor.materialize_f64(), vec![1.25, 2.5]);
        }
        other => panic!("expected single tensor, got {other:?}"),
    }
    for (name, storage) in [
        ("int8_data", IntegerStorage::I8(vec![i8::MIN, i8::MAX])),
        ("uint8_data", IntegerStorage::U8(vec![0, u8::MAX])),
        ("int16_data", IntegerStorage::I16(vec![i16::MIN, i16::MAX])),
        ("uint16_data", IntegerStorage::U16(vec![0, u16::MAX])),
        ("int32_data", IntegerStorage::I32(vec![i32::MIN, i32::MAX])),
        ("uint32_data", IntegerStorage::U32(vec![0, u32::MAX])),
        ("int64_data", IntegerStorage::I64(vec![i64::MIN, i64::MAX])),
        (
            "uint64_data",
            IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]),
        ),
    ] {
        match values.get(name).unwrap() {
            Value::Tensor(tensor) => assert_eq!(tensor.integer_storage(), Some(&storage)),
            other => panic!("expected integer tensor for {name}, got {other:?}"),
        }
    }
    assert_eq!(values.get("i8_scalar"), Some(&Value::Int(IntValue::I8(-3))));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn save_string_array_variable_names() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    set_workspace(&[
        ("A", Value::Num(1.0)),
        ("B", Value::Num(2.0)),
        ("C", Value::Num(3.0)),
    ]);
    let dir = tempdir().unwrap();
    let path = dir.path().join("string_array.mat");
    let names = StringArray::new(vec!["A".into(), "B".into()], vec![1, 2]).unwrap();
    let args = vec![
        Value::from(path.to_string_lossy().to_string()),
        Value::StringArray(names),
    ];
    block_on(save_builtin(args)).unwrap();

    let file = File::open(&path).unwrap();
    let mat = matfile::MatFile::parse(file).unwrap();
    assert!(mat.find_by_name("A").is_some());
    assert!(mat.find_by_name("B").is_some());
    assert!(mat.find_by_name("C").is_none());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn save_char_matrix_variable_names() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    set_workspace(&[
        ("foo", Value::Num(10.0)),
        ("bar", Value::Num(20.0)),
        ("baz", Value::Num(30.0)),
    ]);
    let dir = tempdir().unwrap();
    let path = dir.path().join("char_matrix.mat");
    let chars = CharArray::from_unicode_scalars("foobar".chars().collect(), 2, 3).unwrap();
    let args = vec![
        Value::from(path.to_string_lossy().to_string()),
        Value::CharArray(chars),
    ];
    block_on(save_builtin(args)).unwrap();

    let file = File::open(&path).unwrap();
    let mat = matfile::MatFile::parse(file).unwrap();
    assert!(mat.find_by_name("foo").is_some());
    assert!(mat.find_by_name("bar").is_some());
    assert!(mat.find_by_name("baz").is_none());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn save_struct_fields() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    let mut opts_struct = runmat_value::UnboundedStructValueBuilder::new();
    opts_struct
        .fields
        .insert("foo".to_string(), Value::Num(1.0));
    opts_struct
        .fields
        .insert("bar".to_string(), Value::Num(2.0));
    set_workspace(&[("opts", Value::Struct(opts_struct.finish()))]);
    let dir = tempdir().unwrap();
    let path = dir.path().join("struct.mat");
    let args = vec![
        Value::from(path.to_string_lossy().to_string()),
        Value::from("-struct"),
        Value::from("opts"),
    ];
    block_on(save_builtin(args)).unwrap();

    let file = File::open(&path).unwrap();
    let mat = matfile::MatFile::parse(file).unwrap();
    let foo = mat.find_by_name("bar").unwrap();
    match foo.data() {
        matfile::NumericData::Double { real, .. } => assert_eq!(real, &[2.0]),
        _ => panic!("expected double"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn save_struct_field_selection() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    let mut opts_struct = runmat_value::UnboundedStructValueBuilder::new();
    opts_struct
        .fields
        .insert("foo".to_string(), Value::Num(11.0));
    opts_struct
        .fields
        .insert("bar".to_string(), Value::Num(22.0));
    set_workspace(&[("opts", Value::Struct(opts_struct.finish()))]);
    let dir = tempdir().unwrap();
    let path = dir.path().join("struct_subset.mat");
    let args = vec![
        Value::from(path.to_string_lossy().to_string()),
        Value::from("-struct"),
        Value::from("opts"),
        Value::from("bar"),
    ];
    block_on(save_builtin(args)).unwrap();

    let file = File::open(&path).unwrap();
    let mat = matfile::MatFile::parse(file).unwrap();
    assert!(mat.find_by_name("foo").is_none());
    let array = mat.find_by_name("bar").unwrap();
    match array.data() {
        matfile::NumericData::Double { real, .. } => assert_eq!(real, &[22.0]),
        _ => panic!("expected double array"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn save_missing_variable_errors() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    set_workspace(&[]);
    let result = block_on(save_builtin(vec![
        Value::from("missing.mat"),
        Value::from("x"),
    ]));
    assert_error_contains(result, "variable 'x'");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn save_regex_variable_selection() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    set_workspace(&[
        ("alpha", Value::Num(1.0)),
        ("beta", Value::Num(2.0)),
        ("gamma", Value::Num(3.0)),
    ]);
    let dir = tempdir().unwrap();
    let path = dir.path().join("regex.mat");
    let args = vec![
        Value::from(path.to_string_lossy().to_string()),
        Value::from("-regexp"),
        Value::from("^a"),
        Value::from("ma$"),
    ];
    block_on(save_builtin(args)).unwrap();

    let file = File::open(&path).unwrap();
    let mat = matfile::MatFile::parse(file).unwrap();
    assert!(mat.find_by_name("alpha").is_some());
    assert!(mat.find_by_name("gamma").is_some());
    assert!(mat.find_by_name("beta").is_none());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn save_regex_requires_pattern() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    set_workspace(&[("foo", Value::Num(1.0))]);
    let result = block_on(save_builtin(vec![Value::from("-regexp")]));
    assert_error_contains(result, "'-regexp' requires at least one pattern");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn save_append_creates_missing_file() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    set_workspace(&[("A", Value::Num(1.0))]);
    let dir = tempdir().unwrap();
    let path = dir.path().join("append_new.mat");
    let args = vec![
        Value::from(path.to_string_lossy().to_string()),
        Value::from("A"),
        Value::from("-append"),
    ];

    block_on(save_builtin(args)).unwrap();

    assert_saved_double(&path, "A", 1.0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn save_append_preserves_existing_and_replaces_duplicates() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    set_workspace(&[("A", Value::Num(1.0))]);
    let dir = tempdir().unwrap();
    let path = dir.path().join("append_replace.mat");
    block_on(save_builtin(vec![
        Value::from(path.to_string_lossy().to_string()),
        Value::from("A"),
    ]))
    .unwrap();

    set_workspace(&[("A", Value::Num(3.0)), ("B", Value::Num(2.0))]);
    block_on(save_builtin(vec![
        Value::from(path.to_string_lossy().to_string()),
        Value::from("B"),
        Value::from("A"),
        Value::from("-append"),
    ]))
    .unwrap();

    assert_saved_double(&path, "A", 3.0);
    assert_saved_double(&path, "B", 2.0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn save_unsupported_option_errors() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    set_workspace(&[("foo", Value::Num(1.0))]);
    let result = block_on(save_builtin(vec![
        Value::from("text.mat"),
        Value::from("-ascii"),
        Value::from("foo"),
    ]));
    assert_error_contains(result, "unsupported option '-ascii'");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn save_defaults_to_matlab_mat() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    #[cfg(not(target_arch = "wasm32"))]
    let _lock = lock_env_override();
    set_workspace(&[("answer", Value::Num(7.0))]);
    let dir = tempdir().unwrap();
    #[cfg(not(target_arch = "wasm32"))]
    let target = dir.path().join("matlab_default.mat");
    #[cfg(target_arch = "wasm32")]
    let target = dir.path().join("matlab.mat");
    #[cfg(not(target_arch = "wasm32"))]
    let target_str = target.to_string_lossy().to_string();
    #[cfg(not(target_arch = "wasm32"))]
    let _env = EnvOverride::set("RUNMAT_SAVE_DEFAULT_PATH", &target_str);
    block_on(save_builtin(Vec::new())).unwrap();

    assert!(
        exists(&target),
        "expected {} to be created",
        target.display()
    );
    let file = File::open(&target).unwrap();
    let mat = matfile::MatFile::parse(file).unwrap();
    let array = mat.find_by_name("answer").unwrap();
    match array.data() {
        matfile::NumericData::Double { real, .. } => assert_eq!(real, &[7.0]),
        _ => panic!("expected double array"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn save_struct_without_filename_defaults_to_matlab_mat() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    #[cfg(not(target_arch = "wasm32"))]
    let _lock = lock_env_override();
    let mut payload_struct = runmat_value::UnboundedStructValueBuilder::new();
    payload_struct
        .fields
        .insert("alpha".to_string(), Value::Num(3.0));
    set_workspace(&[("payload", Value::Struct(payload_struct.finish()))]);
    let dir = tempdir().unwrap();
    #[cfg(not(target_arch = "wasm32"))]
    let target = dir.path().join("matlab_struct.mat");
    #[cfg(target_arch = "wasm32")]
    let target = dir.path().join("matlab.mat");
    #[cfg(not(target_arch = "wasm32"))]
    let target_str = target.to_string_lossy().to_string();
    #[cfg(not(target_arch = "wasm32"))]
    let _env = EnvOverride::set("RUNMAT_SAVE_DEFAULT_PATH", &target_str);
    block_on(save_builtin(vec![
        Value::from("-struct"),
        Value::from("payload"),
    ]))
    .unwrap();

    assert!(
        exists(&target),
        "expected {} to be created",
        target.display()
    );
    let file = File::open(&target).unwrap();
    let mat = matfile::MatFile::parse(file).unwrap();
    let field = mat.find_by_name("alpha").unwrap();
    match field.data() {
        matfile::NumericData::Double { real, .. } => assert_eq!(real, &[3.0]),
        _ => panic!("expected double array"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn save_gpu_tensor_roundtrip() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload tensor");
        set_workspace(&[("gpu_data", Value::GpuTensor(handle.clone()))]);

        let dir = tempdir().unwrap();
        let path = dir.path().join("gpu_roundtrip.mat");
        block_on(save_builtin(vec![
            Value::from(path.to_string_lossy().to_string()),
            Value::from("gpu_data"),
        ]))
        .unwrap();

        let file = File::open(&path).unwrap();
        let mat = matfile::MatFile::parse(file).unwrap();
        let array = mat.find_by_name("gpu_data").unwrap();
        match array.data() {
            matfile::NumericData::Double { real, imag } => {
                assert_eq!(real, &tensor.materialize_f64());
                assert!(imag.is_none());
            }
            _ => panic!("expected double array"),
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn save_wgpu_tensor_roundtrip() {
    let _state = crate::builtins::io::mat::tests::state_guard();
    ensure_test_resolver();
    if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_err()
    {
        return;
    }
    let Some(provider) = runmat_accelerate_api::provider() else {
        return;
    };

    let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
    let view = HostTensorView {
        data: &tensor.materialize_f64(),
        shape: tensor.shape(),
    };
    let handle = provider.upload(&view).expect("upload tensor");
    set_workspace(&[("wgpu_tensor", Value::GpuTensor(handle.clone()))]);

    let dir = tempdir().unwrap();
    let path = dir.path().join("wgpu_roundtrip.mat");
    block_on(save_builtin(vec![
        Value::from(path.to_string_lossy().to_string()),
        Value::from("wgpu_tensor"),
    ]))
    .unwrap();

    let file = File::open(&path).unwrap();
    let mat = matfile::MatFile::parse(file).unwrap();
    let array = mat.find_by_name("wgpu_tensor").unwrap();
    match (provider.precision(), array.data()) {
        (
            runmat_accelerate_api::ProviderPrecision::F32,
            matfile::NumericData::Single { real, imag },
        ) => {
            assert_eq!(real, &[0.0_f32, 1.0, 2.0, 3.0]);
            assert!(imag.is_none());
        }
        (
            runmat_accelerate_api::ProviderPrecision::F64,
            matfile::NumericData::Double { real, imag },
        ) => {
            assert_eq!(real, &tensor.materialize_f64());
            assert!(imag.is_none());
        }
        (precision, data) => {
            panic!("expected MAT numeric class matching {precision:?}, got {data:?}")
        }
    }
}
