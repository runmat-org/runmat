#[cfg(unix)]
use runmat_process_host::HostCommand;
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
    let result = unsafe { libc::kill(descendant, 0) };
    if result == 0 {
        panic!("descendant process {descendant} survived process-tree termination");
    }
}
