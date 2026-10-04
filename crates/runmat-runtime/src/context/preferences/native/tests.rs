use super::*;
use crate::execution::RuntimeExecutionService;
use futures::executor::block_on;
use runmat_value::IntValue;
use std::future::Future;
use std::io::BufRead;
use std::rc::Rc;
use std::task::{Context, Poll};

struct WakeSignal(std::sync::mpsc::Sender<()>);

impl futures::task::ArcWake for WakeSignal {
    fn wake_by_ref(arc_self: &std::sync::Arc<Self>) {
        let _ = arc_self.0.send(());
    }
}

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
fn user_path_and_preferences_share_one_atomic_document_without_key_collision() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("application.json");
    let driver = driver();
    let first = NativePreferenceService::new(path.clone(), driver.handle().clone()).unwrap();
    let second = NativePreferenceService::new(path, driver.handle().clone()).unwrap();
    assert_eq!(
        block_on(RuntimeUserPathService::read(&first, runtime()))
            .unwrap()
            .revision,
        0
    );
    block_on(first.set_many(
        runtime(),
        "app".into(),
        vec![("wide".into(), Value::Int(IntValue::U64(u64::MAX)))],
    ))
    .unwrap();
    let set = block_on(RuntimeUserPathService::commit(
        &first,
        runtime(),
        UserPathMutation::Set("Relative/MiXeD".into()),
    ))
    .unwrap();
    assert_eq!(set.revision, 1);
    assert_eq!(
        block_on(RuntimeUserPathService::read(&second, runtime())).unwrap(),
        set
    );
    assert_eq!(
        block_on(second.group_values(runtime(), "app".into()))
            .unwrap()
            .unwrap()["wide"],
        Value::Int(IntValue::U64(u64::MAX))
    );
    let cleared = block_on(RuntimeUserPathService::commit(
        &second,
        runtime(),
        UserPathMutation::Clear,
    ))
    .unwrap();
    assert_eq!(cleared.revision, 2);
    assert_eq!(cleared.setting, UserPathSetting::Cleared);
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
    let mut write = store.try_begin_write().unwrap().unwrap();
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
    let held = store.try_begin_write().unwrap().unwrap();
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

#[test]
fn dropped_outer_future_after_authorization_does_not_claim_rollback() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("preferences.json");
    let driver = driver();
    let (reached_tx, reached_rx) = std::sync::mpsc::channel();
    let (resume_tx, resume_rx) = std::sync::mpsc::channel();
    let (wake_tx, wake_rx) = std::sync::mpsc::channel();
    let gate = std::sync::Arc::new(store::CommitGate {
        reached: reached_tx,
        wake_test: wake_tx.clone(),
        resume: std::sync::Mutex::new(resume_rx),
    });
    let mut service = NativePreferenceService::new(path.clone(), driver.handle().clone()).unwrap();
    service.store = service.store.with_commit_gate(gate);
    let mut call = service.set_many(
        runtime(),
        "app".into(),
        vec![("authorized".into(), Value::Num(9.0))],
    );
    let waker = futures::task::waker(std::sync::Arc::new(WakeSignal(wake_tx)));
    let mut context = Context::from_waker(&waker);
    loop {
        assert!(matches!(
            Future::poll(call.as_mut(), &mut context),
            Poll::Pending
        ));
        if reached_rx.try_recv().is_ok() {
            break;
        }
        wake_rx
            .recv_timeout(std::time::Duration::from_secs(3))
            .expect("transaction must make progress to authorization");
    }
    assert!(!path.exists(), "private stage is not a published document");
    drop(call);
    resume_tx.send(()).unwrap();
    driver.shutdown_timeout(std::time::Duration::from_secs(3));
    let stored = NativeStore::new(path).unwrap().read().unwrap();
    assert_eq!(
        decode_durable_value(&stored.groups["app"]["authorized"]).unwrap(),
        Value::Num(9.0)
    );
}

#[test]
fn cross_process_child_entry() {
    let Some(path) = std::env::var_os("RUNMAT_PREF_TEST_CHILD_DOCUMENT") else {
        return;
    };
    let driver = driver();
    let service = NativePreferenceService::new(path.into(), driver.handle().clone()).unwrap();
    println!("RUNMAT_PREF_CHILD_READY");
    block_on(service.set_many(
        runtime(),
        "app".into(),
        vec![("child".into(), Value::Num(2.0))],
    ))
    .unwrap();
    println!("RUNMAT_PREF_CHILD_DONE");
}

struct ChildWitness {
    child: std::process::Child,
    reader: Option<std::thread::JoinHandle<()>>,
    reaped: bool,
}

impl Drop for ChildWitness {
    fn drop(&mut self) {
        if !self.reaped {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

#[test]
fn cross_process_lock_preserves_disjoint_updates() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("preferences.json");
    let store = NativeStore::new(path.clone()).unwrap();
    let mut held = store.try_begin_write().unwrap().unwrap();
    let executable = std::env::current_exe().unwrap();
    let mut child = std::process::Command::new(executable)
        .arg("cross_process_child_entry")
        .arg("--nocapture")
        .env("RUNMAT_PREF_TEST_CHILD_DOCUMENT", &path)
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let output = child.stdout.take().unwrap();
    let (marker_tx, marker_rx) = std::sync::mpsc::channel();
    let reader = std::thread::spawn(move || {
        for line in std::io::BufReader::new(output).lines() {
            let line = line.unwrap();
            if line.contains("RUNMAT_PREF_CHILD_READY") {
                let _ = marker_tx.send("ready");
            }
            if line.contains("RUNMAT_PREF_CHILD_DONE") {
                let _ = marker_tx.send("done");
            }
        }
    });
    let mut witness = ChildWitness {
        child,
        reader: Some(reader),
        reaped: false,
    };
    assert_eq!(
        marker_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("child entered before parent publication"),
        "ready"
    );
    held.document
        .groups
        .entry("app".into())
        .or_default()
        .insert(
            "parent".into(),
            encode_durable_value(&Value::Num(1.0)).unwrap(),
        );
    held.stage().unwrap().commit().unwrap();
    assert_eq!(
        marker_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("child completed after parent publication"),
        "done"
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let status = loop {
        if let Some(status) = witness.child.try_wait().unwrap() {
            break status;
        }
        assert!(std::time::Instant::now() < deadline, "child did not exit");
        std::thread::park_timeout(std::time::Duration::from_millis(10));
    };
    witness.reaped = true;
    assert!(status.success());
    let document = store.read().unwrap();
    assert_eq!(
        decode_durable_value(&document.groups["app"]["parent"]).unwrap(),
        Value::Num(1.0)
    );
    assert_eq!(
        decode_durable_value(&document.groups["app"]["child"]).unwrap(),
        Value::Num(2.0)
    );
}
