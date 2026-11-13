#[cfg(unix)]
#[tokio::test]
async fn dropping_server_releases_authority_with_silent_connections_and_live_runtime() {
    let temp = tempfile::tempdir().unwrap();
    let config = crate::supervisor::tests::test_config(&temp);
    let paths = config.paths.clone();
    let client = crate::supervisor::LocalSupervisorClient::new(config.clone());
    let server = tokio::spawn(crate::supervisor::run_local_supervisor(config.clone()));
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            if client.ping().await.is_ok() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let silent = tokio::net::UnixStream::connect(&paths.socket)
        .await
        .unwrap();
    // A real ping lets the listener accept and poll the silent connection too.
    client.ping().await.unwrap();
    server.abort();
    assert!(server.await.unwrap_err().is_cancelled());
    assert!(
        runmat_filesystem::ExclusiveFileLock::try_acquire(&paths.root.join("supervisor.lock"))
            .unwrap()
            .is_some(),
        "a detached service task retained original authority"
    );
    drop(silent);
    let reopened = crate::supervisor::LocalSupervisor::open(config).unwrap();
    drop(reopened);
}

#[tokio::test]
async fn oversized_response_delivers_a_bounded_failure_on_the_original_frame() {
    use super::*;
    let response = SupervisorResponse::Error {
        code: "original".into(),
        message: "x".repeat(SUPERVISOR_MAX_MESSAGE_BYTES as usize),
    };
    let payload = response_payload(&response).unwrap();
    assert!(payload.len() < 1024);
    let limits = FrameLimits {
        max_message_bytes: SUPERVISOR_MAX_MESSAGE_BYTES,
    };
    let (mut producer, mut consumer) = tokio::io::duplex(4096);
    write_payload(&mut producer, &payload, limits)
        .await
        .unwrap();
    let received = read_payload(&mut consumer, limits).await.unwrap();
    let decoded: SupervisorResponse = serde_json::from_slice(&received).unwrap();
    assert!(matches!(decoded, SupervisorResponse::Error { code, .. }
        if code == "response_encoding_failed"));
}
