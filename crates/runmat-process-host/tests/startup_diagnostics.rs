use runmat_process_host::{ChildLifetime, HostCommand, ProcessOutputCompletion, StdioPolicy};
use std::io::Write;

const CHILD_ENV: &str = "RUNMAT_STARTUP_DIAGNOSTICS_TEST_CHILD";
const ORIGINAL: &[u8] = b"original \xce\xbb\xff startup";

#[test]
fn detached_startup_child() {
    let Some(marker) = std::env::var_os(CHILD_ENV) else {
        return;
    };
    std::io::stderr().write_all(ORIGINAL).unwrap();
    runmat_process_host::release_startup_stderr().unwrap();
    // The startup pipe is closed, but the detached service can still write diagnostics
    // without depending on its starter's pipe or runtime lifetime.
    std::thread::sleep(std::time::Duration::from_millis(100));
    std::io::stderr().write_all(b"after readiness").unwrap();
    // Publish completion after the full marker is written. A reader can observe
    // a newly created file before write_all has supplied its contents.
    let marker = std::path::PathBuf::from(marker);
    let pending = marker.with_extension("pending");
    std::fs::write(&pending, b"original child finished after handoff").unwrap();
    std::fs::rename(pending, marker).unwrap();
}

#[tokio::test]
async fn detached_startup_stream_closes_before_service_lifetime_ends() {
    let temp = tempfile::tempdir().unwrap();
    let marker = temp.path().join("original-child-finished");
    let mut spec = HostCommand::new(std::env::current_exe().unwrap());
    spec.arguments = vec![
        "--exact".into(),
        "detached_startup_child".into(),
        "--nocapture".into(),
    ];
    spec.environment
        .insert(CHILD_ENV.into(), marker.to_str().unwrap().into());
    spec.lifetime = ChildLifetime::Detached;
    spec.stdio = StdioPolicy::StartupDiagnostics;
    spec.max_stderr_bytes = 11;
    let mut child = spec.spawn().await.unwrap();
    assert!(child.take_stdio().is_err());
    let capture = child.captured_stderr();
    tokio::time::timeout(std::time::Duration::from_secs(5), capture.wait_closed())
        .await
        .unwrap();
    let snapshot = capture.snapshot();
    assert_eq!(snapshot.bytes.as_ref(), &ORIGINAL[..11]);
    assert_eq!(snapshot.observed_bytes, Some(ORIGINAL.len() as u64));
    assert!(snapshot.truncated);
    assert_eq!(snapshot.completion, ProcessOutputCompletion::EndOfStream);
    let id = child.id().unwrap();
    drop(child);
    drop(capture);
    let done = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while !marker.exists() {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await;
    if done.is_err() {
        let _ = runmat_process_host::terminate_process_tree(id).await;
    }
    done.unwrap();
    assert_eq!(
        std::fs::read(marker).unwrap(),
        b"original child finished after handoff"
    );
}
