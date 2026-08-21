#[cfg(unix)]
use runmat_process_host::HostCommand;

#[cfg(unix)]
#[tokio::test]
async fn stderr_capture_is_bounded() {
    let mut spec = HostCommand::new("/bin/sh");
    spec.arguments = vec!["-c".into(), "printf '0123456789abcdef' >&2".into()];
    spec.max_stderr_bytes = 7;
    let mut child = spec.spawn().await.unwrap();
    let _stdio = child.take_stdio().unwrap();
    let capture = child.captured_stderr();
    let exit = child.wait().await.unwrap();
    assert!(exit.success);
    for _ in 0..20 {
        if capture.bytes().len() == 7 {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(capture.text(), "0123456");
}
