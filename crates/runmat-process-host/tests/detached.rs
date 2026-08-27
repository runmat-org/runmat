use runmat_process_host::{terminate_process_tree, ChildLifetime, HostCommand, StdioPolicy};

#[tokio::test]
async fn detached_file_backed_child_survives_its_host_handle() {
    let temp = tempfile::tempdir().unwrap();
    let stdout = temp.path().join("stdout.log");
    let stderr = temp.path().join("stderr.log");
    #[cfg(unix)]
    let mut spec = {
        let mut spec = HostCommand::new("/bin/sh");
        spec.arguments = vec!["-c".into(), "sleep 0.1; printf detached".into()];
        spec
    };
    #[cfg(windows)]
    let mut spec = {
        let mut spec = HostCommand::new("cmd.exe");
        spec.arguments = vec![
            "/D".into(),
            "/S".into(),
            "/C".into(),
            "ping -n 2 127.0.0.1 >NUL && <NUL set /p =detached".into(),
        ];
        spec
    };
    spec.lifetime = ChildLifetime::Detached;
    spec.stdio = StdioPolicy::Files {
        stdout: stdout.clone(),
        stderr,
    };
    let child = spec.spawn().await.unwrap();
    let process_id = child.id().unwrap();
    drop(child);

    for _ in 0..50 {
        if std::fs::read_to_string(&stdout).ok().as_deref() == Some("detached") {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    let _ = terminate_process_tree(process_id).await;
    panic!("detached child did not finish after its host handle was dropped");
}
