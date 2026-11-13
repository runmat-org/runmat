use super::*;
use crate::context::RuntimeContext;
use crate::execution::RuntimeExecutionService;
use futures::executor::block_on;
use runmat_execution::cancellation::CancellationSource;
use runmat_filesystem::{MemoryFsProvider, OpenOptions};
use runmat_value::Value;
use std::rc::Rc;

fn registered_file(runtime: Option<&RuntimeContext>) -> i32 {
    let path = PathBuf::from("/fid-session/source.txt");
    let filesystem = Arc::new(FileFilesystem::capture(&path).expect("capture exact owner"));
    let mut options = OpenOptions::new();
    options.read(true);
    let file = block_on(options.open_async_on(filesystem.provider.as_ref(), &filesystem.path))
        .expect("open exact owner");
    register_file_in(
        RegisteredFile {
            path,
            permission: "r".into(),
            access: FileAccess::READ,
            machinefmt: "native".into(),
            encoding: "UTF-8".into(),
            handle: Arc::new(StdMutex::new(Some(file))),
        },
        runtime,
        Some(filesystem),
    )
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn session_fids_do_not_cross_sessions_or_process_embedding() {
    let _registry = test_guard();
    let _filesystem = runmat_filesystem::provider_override_lock();
    reset_for_tests();
    let provider = Arc::new(MemoryFsProvider::with_current_dir("/fid-session"));
    provider
        .write_project_path("/fid-session/source.txt", b"first\nsecond\n")
        .expect("fixture");
    let _owner = runmat_filesystem::replace_provider(provider);
    let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let fid = registered_file(Some(&first));
    let rebound = first
        .clone()
        .with_cancellation_token(CancellationSource::new().token());
    assert!(info_for_in(fid, Some(&first)).is_some());
    assert!(info_for_in(fid, Some(&rebound)).is_some());
    assert!(info_for_in(fid, Some(&second)).is_none());
    assert!(info_for_in(fid, None).is_none());
    assert!(list_infos_in(Some(&first))
        .iter()
        .any(|info| info.id == fid));
    assert!(!list_infos_in(Some(&second))
        .iter()
        .any(|info| info.id == fid));
    assert!(!list_infos_in(None).iter().any(|info| info.id == fid));
    assert!(matches!(
        acquire_file_in(fid, Some(&second)),
        Err(FileOperationError::ForeignSession)
    ));
    assert!(matches!(
        acquire_file_in(fid, None),
        Err(FileOperationError::ForeignSession)
    ));
    assert!(block_on(close_async_in(fid, Some(&second)))
        .expect("foreign close protocol")
        .is_none());
    assert!(block_on(close_async_in(fid, None))
        .expect("process close protocol")
        .is_none());
    let foreign = block_on(crate::builtins::io::filetext::fgetl::evaluate_in(
        Some(&second),
        &Value::Num(fid as f64),
        crate::call::FunctionArgumentView::from_slice(&[]),
    ))
    .expect_err("foreign session cannot read the identifier");
    assert_eq!(foreign.identifier(), Some("RunMat:fgetl:InvalidIdentifier"));
    let replacement = Arc::new(MemoryFsProvider::with_current_dir("/fid-session"));
    replacement
        .write_project_path("/fid-session/source.txt", b"wrong owner\n")
        .expect("replacement fixture");
    let _replacement = runmat_filesystem::replace_provider(replacement);
    let mut retained = acquire_file_in(fid, Some(&rebound)).expect("same-session owner");
    let mut first_line = [0u8; 6];
    std::io::Read::read_exact(retained.file_mut(), &mut first_line).expect("read original owner");
    assert_eq!(&first_line, b"first\n");
    drop(retained);
    assert!(block_on(close_async_in(fid, Some(&first)))
        .expect("owner close")
        .is_some());
    let process_fid = registered_file(None);
    assert!(info_for_in(process_fid, None).is_some());
    assert!(info_for_in(process_fid, Some(&first)).is_none());
    assert!(block_on(close_async_in(process_fid, None))
        .expect("process close")
        .is_some());
    reset_for_tests();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn session_teardown_tombstones_checked_out_file_until_its_drop() {
    let _registry = test_guard();
    let _filesystem = runmat_filesystem::provider_override_lock();
    reset_for_tests();
    let provider = Arc::new(MemoryFsProvider::with_current_dir("/fid-session"));
    provider
        .write_project_path("/fid-session/source.txt", b"data")
        .expect("fixture");
    let _owner = runmat_filesystem::replace_provider(provider);
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let fid = registered_file(Some(&runtime));
    let operation = acquire_file_in(fid, Some(&runtime)).expect("checkout");
    drop(runtime);
    assert!(REGISTRY
        .lock()
        .expect("registry")
        .entries
        .get(&fid)
        .is_some_and(|entry| entry.retired));
    assert!(operation.check_authority(None).is_err());
    drop(operation);
    assert!(!REGISTRY
        .lock()
        .expect("registry")
        .entries
        .contains_key(&fid));
    reset_for_tests();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fopen_and_fgetl_share_only_the_issuing_session() {
    let _registry = test_guard();
    let _filesystem = runmat_filesystem::provider_override_lock();
    reset_for_tests();
    let provider = Arc::new(MemoryFsProvider::with_current_dir("/fid-session"));
    provider
        .write_project_path("/fid-session/source.txt", b"first\nsecond\n")
        .expect("fixture");
    let _owner = runmat_filesystem::replace_provider(provider);
    let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let opened = block_on(crate::builtins::io::filetext::fopen::evaluate_in(
        Some(&first),
        &[Value::from("/fid-session/source.txt"), Value::from("r")],
    ))
    .expect("open");
    let fid = opened.as_open().expect("open result").fid as i32;
    assert!(fid >= 3);
    let foreign = block_on(crate::builtins::io::filetext::fgetl::evaluate_in(
        Some(&second),
        &Value::Num(fid as f64),
        crate::call::FunctionArgumentView::from_slice(&[]),
    ))
    .expect_err("foreign read");
    assert_eq!(foreign.identifier(), Some("RunMat:fgetl:InvalidIdentifier"));
    let first_line = block_on(crate::builtins::io::filetext::fgetl::evaluate_in(
        Some(&first),
        &Value::Num(fid as f64),
        crate::call::FunctionArgumentView::from_slice(&[]),
    ))
    .expect("issuer read");
    let Value::CharArray(line) = first_line.first_output() else {
        panic!("expected character row")
    };
    assert_eq!(
        runmat_value::decode_utf16_code_units(line.data()).expect("valid UTF-16 fixture"),
        "first"
    );
    assert!(block_on(close_async_in(fid, Some(&first)))
        .expect("issuer close")
        .is_some());
    reset_for_tests();
}
