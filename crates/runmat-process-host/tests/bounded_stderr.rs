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
    capture.wait_closed().await;
    let snapshot = capture.snapshot();
    assert_eq!(
        snapshot.completion,
        runmat_process_host::ProcessOutputCompletion::EndOfStream
    );
    assert_eq!(snapshot.observed_bytes, Some(16));
    assert!(snapshot.truncated);
    assert_eq!(capture.text(), "0123456");
}

#[tokio::test]
async fn read_error_is_not_relabelled_as_eof() {
    struct Failed;
    impl tokio::io::AsyncRead for Failed {
        fn poll_read(
            self: std::pin::Pin<&mut Self>,
            _: &mut std::task::Context<'_>,
            _: &mut tokio::io::ReadBuf<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            std::task::Poll::Ready(Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "original read error",
            )))
        }
    }
    let capture = runmat_process_host::child::CapturedProcessOutput::new(7);
    capture.drain(Failed).unwrap();
    assert!(capture.drain(tokio::io::empty()).is_err());
    capture.wait_closed().await;
    assert!(capture.drain(tokio::io::empty()).is_err());
    assert_eq!(
        capture.snapshot().completion,
        runmat_process_host::ProcessOutputCompletion::ReadFailed(std::io::ErrorKind::BrokenPipe)
    );
    // Completion is retained even if it occurred before a later waiter registered.
    capture.wait_closed().await;

    let interrupted = runmat_process_host::child::CapturedProcessOutput::new(7);
    let mut producer = interrupted.begin_capture().unwrap();
    producer.observe(b"0123456789abcdef").unwrap();
    drop(producer);
    interrupted.wait_closed().await;
    let snapshot = interrupted.snapshot();
    assert_eq!(
        snapshot.completion,
        runmat_process_host::ProcessOutputCompletion::ReadFailed(std::io::ErrorKind::Interrupted)
    );
    assert_eq!(snapshot.observed_bytes, Some(16));
    assert!(snapshot.truncated);
    assert_eq!(snapshot.bytes.as_ref(), b"0123456");
    assert!(interrupted.begin_capture().is_err());
    let transferred = interrupted.take_closed_snapshot().unwrap();
    assert_eq!(transferred, snapshot);
    assert_eq!(
        interrupted.completion(),
        runmat_process_host::ProcessOutputCompletion::Transferred
    );
    assert!(interrupted.take_closed_snapshot().is_err());
    assert!(interrupted.bytes().is_empty());
}

#[tokio::test]
async fn unrequested_capture_does_not_claim_a_live_reader() {
    let capture = runmat_process_host::child::CapturedProcessOutput::new(7);
    assert_eq!(
        capture.completion(),
        runmat_process_host::ProcessOutputCompletion::NotCaptured
    );
    capture.wait_closed().await;
    assert!(capture.bytes().is_empty());
}
