use std::io::{Read, Write};

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::{PrivateControlEndpoint, PrivateControlListener};

#[tokio::test]
async fn separate_control_channels_round_trip_and_writer_shutdown_reaches_host() {
    let listener = PrivateControlListener::bind().expect("private listener");
    let endpoint = PrivateControlEndpoint::decode(&listener.endpoint().encode().unwrap()).unwrap();
    let host = std::thread::spawn(move || {
        let mut control = endpoint.connect_blocking().expect("host connects");
        let mut input = Vec::new();
        control.reader.read_to_end(&mut input).expect("driver EOF");
        assert_eq!(input, b"request");
        control.writer.write_all(b"response").unwrap();
        control.writer.flush().unwrap();
    });
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        let mut control = listener.accept().await.expect("parent accepts");
        control.writer.write_all(b"request").await.unwrap();
        control.writer.shutdown().await.unwrap();
        let mut output = Vec::new();
        control.reader.read_to_end(&mut output).await.unwrap();
        assert_eq!(output, b"response");
    })
    .await
    .expect("half-close must not wait for parent drop");
    host.join().unwrap();
}

#[test]
fn bootstrap_decode_rejects_a_shared_channel() {
    let encoded = r#"{"request":"same","response":"same"}"#;
    assert!(PrivateControlEndpoint::decode(encoded).is_err());
}

#[tokio::test]
async fn async_host_keeps_response_after_request_half_close() {
    let listener = PrivateControlListener::bind().unwrap();
    let endpoint = listener.endpoint().clone();
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        let host = async {
            let mut control = endpoint.connect().await.unwrap();
            let mut request = Vec::new();
            control.reader.read_to_end(&mut request).await.unwrap();
            assert_eq!(request, b"request");
            control.writer.write_all(b"response").await.unwrap();
            control.writer.shutdown().await.unwrap();
        };
        let driver = async {
            let mut control = listener.accept().await.unwrap();
            control.writer.write_all(b"request").await.unwrap();
            control.writer.shutdown().await.unwrap();
            let mut response = Vec::new();
            control.reader.read_to_end(&mut response).await.unwrap();
            assert_eq!(response, b"response");
        };
        tokio::join!(host, driver);
    })
    .await
    .expect("async half-close must not wait for the whole connection to drop");
}

#[tokio::test]
async fn application_stdout_and_stderr_do_not_enter_the_child_control_channel() {
    use crate::{HostCommand, StdioPolicy};

    let listener = PrivateControlListener::bind().unwrap();
    let mut command = HostCommand::new(std::env::current_exe().unwrap());
    command.arguments = vec![
        "--exact".into(),
        "ipc::private_control::tests::application_output_fixture".into(),
        "--nocapture".into(),
    ];
    command.stdio = StdioPolicy::PrivateControl(listener.endpoint().clone());
    let mut child = command.spawn().await.unwrap();
    let mut output = child.take_application_output().unwrap();
    let exchange = async {
        let mut control = listener.accept().await.unwrap();
        control.writer.write_all(b"request").await.unwrap();
        control.writer.shutdown().await.unwrap();
        let mut response = Vec::new();
        control.reader.read_to_end(&mut response).await.unwrap();
        assert_eq!(response, b"response");
    };
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let ((), stdout_read, stderr_read) =
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            tokio::join!(
                exchange,
                output.stdout.read_to_end(&mut stdout),
                output.stderr.read_to_end(&mut stderr)
            )
        })
        .await
        .expect("child application output must not block its control protocol");
    stdout_read.unwrap();
    stderr_read.unwrap();
    let diagnostics = child.captured_stderr();
    assert!(
        diagnostics.begin_capture().is_err(),
        "application pipe has one diagnostic producer"
    );
    output.stderr_diagnostics.observe(&stderr).unwrap();
    output.stderr_diagnostics.finish(Ok(()));
    assert_eq!(diagnostics.bytes(), stderr);
    assert_eq!(
        diagnostics.completion(),
        crate::ProcessOutputCompletion::EndOfStream
    );
    assert!(child.wait_and_retire_tree().await.unwrap().success);
    assert!(String::from_utf8(stdout)
        .unwrap()
        .contains("application stdout"));
    assert!(String::from_utf8(stderr)
        .unwrap()
        .contains("application stderr"));
}

#[test]
fn application_output_fixture() {
    if std::env::var_os(super::BOOTSTRAP_ENV).is_none() {
        return;
    }
    let mut control = super::connect_host_blocking().unwrap();
    println!("application stdout");
    eprintln!("application stderr");
    let mut request = Vec::new();
    control.reader.read_to_end(&mut request).unwrap();
    assert_eq!(request, b"request");
    control.writer.write_all(b"response").unwrap();
    control.writer.flush().unwrap();
}
