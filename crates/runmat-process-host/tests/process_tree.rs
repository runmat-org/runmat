#[cfg(unix)]
use runmat_process_host::{is_process_alive, HostCommand};
#[cfg(unix)]
use tokio::io::AsyncReadExt;

#[cfg(unix)]
#[tokio::test]
async fn terminate_kills_the_child_process_group() {
    let mut spec = HostCommand::new("/bin/sh");
    spec.arguments = vec!["-c".into(), "sleep 60 & echo $!; wait".into()];
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
    child.terminate_tree().await.unwrap();
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
    while is_process_alive(descendant.try_into().unwrap()) {
        assert!(
            tokio::time::Instant::now() < deadline,
            "descendant process {descendant} survived process-tree termination"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
}
