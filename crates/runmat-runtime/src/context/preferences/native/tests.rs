use super::*;
use crate::execution::RuntimeExecutionService;
use futures::executor::block_on;
use runmat_value::IntValue;
use std::future::Future;
use std::rc::Rc;
use std::task::{Context, Poll};

fn runtime() -> RuntimeContext {
    RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
}

fn driver() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("isolated test driver")
}

#[test]
fn new_sessions_observe_atomic_multi_name_commits_without_project_filesystem() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("preferences.json");
    let driver = driver();
    let first = NativePreferenceService::new(path.clone(), driver.handle().clone()).unwrap();
    let second = NativePreferenceService::new(path.clone(), driver.handle().clone()).unwrap();
    assert!(!path.exists(), "constructor performs no file effect");
    block_on(first.set_many(
        runtime(),
        "app".into(),
        vec![
            ("one".into(), Value::Int(IntValue::U64(u64::MAX))),
            ("two".into(), Value::Num(2.0)),
        ],
    ))
    .unwrap();
    let values = block_on(second.group_values(runtime(), "app".into()))
        .unwrap()
        .unwrap();
    assert_eq!(values["one"], Value::Int(IntValue::U64(u64::MAX)));
    assert_eq!(values["two"], Value::Num(2.0));
    let (left, right) = block_on(async {
        futures::join!(
            first.set_many(
                runtime(),
                "app".into(),
                vec![("left".into(), Value::Bool(true))]
            ),
            second.set_many(
                runtime(),
                "app".into(),
                vec![("right".into(), Value::Bool(false))]
            ),
        )
    });
    left.unwrap();
    right.unwrap();
    let values = block_on(first.group_values(runtime(), "app".into()))
        .unwrap()
        .unwrap();
    assert_eq!(values.len(), 4);
    assert_eq!(values["left"], Value::Bool(true));
    assert_eq!(values["right"], Value::Bool(false));
}

#[test]
fn missing_without_default_does_not_create_a_settings_file() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("preferences.json");
    let driver = driver();
    let service = NativePreferenceService::new(path.clone(), driver.handle().clone()).unwrap();
    let error = block_on(service.read_or_create_many(
        runtime(),
        "app".into(),
        vec!["missing".into()],
        None,
    ))
    .unwrap_err();
    assert!(matches!(error, PreferenceServiceError::Missing { .. }));
    assert!(!path.exists());
}

#[test]
fn abandoned_private_stage_and_corrupt_existing_document_never_replace() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("preferences.json");
    let store = NativeStore::new(path.clone()).unwrap();
    let mut write = store.begin_write().unwrap();
    write
        .document
        .groups
        .entry("app".into())
        .or_default()
        .insert(
            "value".into(),
            encode_durable_value(&Value::Num(1.0)).unwrap(),
        );
    drop(write.stage().unwrap());
    assert!(!path.exists(), "private staged bytes are not publication");
    std::fs::write(&path, b"{broken").unwrap();
    let driver = driver();
    let service = NativePreferenceService::new(path.clone(), driver.handle().clone()).unwrap();
    let error = block_on(service.set_many(
        runtime(),
        "app".into(),
        vec![("new".into(), Value::Num(2.0))],
    ))
    .unwrap_err();
    assert!(matches!(error, PreferenceServiceError::Document(_)));
    assert_eq!(std::fs::read(&path).unwrap(), b"{broken");
}

#[test]
fn actual_pending_mutation_drop_cannot_publish_after_lock_release() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("preferences.json");
    let store = NativeStore::new(path.clone()).unwrap();
    let held = store.begin_write().unwrap();
    let driver = driver();
    let service = NativePreferenceService::new(path.clone(), driver.handle().clone()).unwrap();
    let mut call = service.set_many(
        runtime(),
        "app".into(),
        vec![("discarded".into(), Value::Num(1.0))],
    );
    let mut context = Context::from_waker(futures::task::noop_waker_ref());
    assert!(matches!(
        Future::poll(call.as_mut(), &mut context),
        Poll::Pending
    ));
    drop(call); // Drops authorization sender even if worker is still waiting on the OS lock.
    drop(held);
    driver.shutdown_timeout(std::time::Duration::from_secs(2));
    assert!(
        !path.exists(),
        "abandoned authorization cannot publish staged bytes"
    );
}
