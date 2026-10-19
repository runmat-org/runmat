use runmat_process_host::{terminate_process_tree, ChildLifetime, HostCommand, StdioPolicy};

const CHILD_MARKER: &str = "runmat-detached-child-finished";
const CHILD_ENV: &str = "RUNMAT_DETACHED_TEST_CHILD";

#[test]
fn detached_child_helper() {
    if std::env::var_os(CHILD_ENV).is_none() {
        return;
    }
    std::thread::sleep(std::time::Duration::from_millis(100));
    print!("{CHILD_MARKER}");
}

#[tokio::test]
async fn detached_file_backed_child_survives_its_host_handle() {
    let temp = tempfile::tempdir().unwrap();
    let stdout = temp.path().join("stdout.log");
    let stderr = temp.path().join("stderr.log");
    let mut spec = HostCommand::new(std::env::current_exe().expect("resolve test executable"));
    spec.arguments = vec![
        "--exact".into(),
        "detached_child_helper".into(),
        "--nocapture".into(),
    ];
    spec.environment.insert(CHILD_ENV.into(), "1".into());
    spec.lifetime = ChildLifetime::Detached;
    spec.stdio = StdioPolicy::Files {
        stdout: stdout.clone(),
        stderr: stderr.clone(),
    };
    let child = spec.spawn().await.unwrap();
    let process_id = child.id().unwrap();
    drop(child);

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    while tokio::time::Instant::now() < deadline {
        if std::fs::read_to_string(&stdout)
            .ok()
            .is_some_and(|contents| contents.contains(CHILD_MARKER))
        {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    let _ = terminate_process_tree(process_id).await;
    let stderr = std::fs::read_to_string(stderr).unwrap_or_default();
    panic!("detached child did not finish after its host handle was dropped; stderr: {stderr:?}");
}
