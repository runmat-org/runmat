#[cfg(unix)]
use runmat_process_host::{is_process_alive, HostCommand};
#[cfg(unix)]
use tokio::io::AsyncReadExt;

#[cfg(unix)]
#[tokio::test]
async fn terminate_kills_the_child_process_group() {
    for leader_exited in [false, true] {
        let mut spec = HostCommand::new("/bin/sh");
        let script = if leader_exited {
            "sleep 60 & echo $!; exit 0"
        } else {
            "sleep 60 & echo $!; wait"
        };
        spec.arguments = vec!["-c".into(), script.into()];
        let mut child = spec.spawn().await.unwrap();
        let mut stdio = child.take_stdio().unwrap();
        let mut pid_line = Vec::new();
        loop {
            let mut byte = [0];
            stdio.stdout.read_exact(&mut byte).await.unwrap();
            if byte[0] == b'\n' {
                break;
            }
            pid_line.push(byte[0]);
        }
        let descendant: i32 = String::from_utf8(pid_line).unwrap().parse().unwrap();
        if leader_exited {
            assert!(child.wait().await.unwrap().success);
        }
        child.terminate_tree().await.unwrap();
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
        while is_process_alive(descendant.try_into().unwrap()) {
            assert!(
                tokio::time::Instant::now() < deadline,
                "descendant process {descendant} survived process-tree termination (leader exited: {leader_exited})"
            );
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }
}

#[cfg(windows)]
#[tokio::test]
async fn failed_taskkill_does_not_advertise_tree_retirement_from_absent_leader() {
    let unknown_process = u32::MAX;
    assert!(!runmat_process_host::is_process_alive(unknown_process));
    assert!(runmat_process_host::terminate_process_tree(unknown_process)
        .await
        .is_err());
}
