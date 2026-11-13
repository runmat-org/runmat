use super::*;
use crate::supervisor::{auth, filesystem, SupervisorPaths};

async fn answered_peer(
    response: SupervisorResponse,
) -> (
    tempfile::TempDir,
    LocalSupervisorClient,
    tokio::task::JoinHandle<()>,
) {
    let temp = tempfile::tempdir().unwrap();
    let paths = SupervisorPaths::new(temp.path().to_path_buf()).unwrap();
    filesystem::secure_directory(&paths.root).unwrap();
    auth::load_or_create_token(&paths).unwrap();
    let mut execution = crate::NativeExecutionConfig::for_current_executable().unwrap();
    // Any attempted fallback spawn would fail and replace the original error.
    execution.executable = temp.path().join("must-not-launch");
    let config = LocalSupervisorConfig::new(execution, paths).unwrap();
    let identity = config.identity().unwrap();
    let listener = tokio::net::UnixListener::bind(&config.paths.socket).unwrap();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let limits = FrameLimits {
            max_message_bytes: SUPERVISOR_MAX_MESSAGE_BYTES,
        };
        let payload = read_payload(&mut stream, limits).await.unwrap();
        let request: SupervisorRequest = serde_json::from_slice(&payload).unwrap();
        assert_eq!(request.config_identity, identity);
        assert_eq!(request.protocol_version, SUPERVISOR_PROTOCOL_VERSION);
        write_payload(&mut stream, &serde_json::to_vec(&response).unwrap(), limits)
            .await
            .unwrap();
    });
    (temp, LocalSupervisorClient::new(config), server)
}

#[tokio::test]
async fn authenticated_rejection_is_preserved_without_bootstrap() {
    let (temp, client, server) = answered_peer(SupervisorResponse::Error {
        code: "authentication_failed".into(),
        message: "original rejection".into(),
    })
    .await;
    let result = client.connect_or_start().await;
    assert!(
        matches!(result, Err(NativeExecutionError::SupervisorRejected { code, message })
        if code == "authentication_failed" && message == "original rejection")
    );
    server.await.unwrap();
    assert!(!temp.path().join("supervisor-profile.json").exists());
}

#[tokio::test]
async fn foreign_execution_profile_cannot_trigger_replacement_bootstrap() {
    let (temp, client, server) = answered_peer(SupervisorResponse::Pong {
        config_identity: runmat_execution::Digest::sha256(b"foreign-profile"),
    })
    .await;
    assert!(matches!(
        client.connect_or_start().await,
        Err(NativeExecutionError::Configuration(_))
    ));
    server.await.unwrap();
    assert!(!temp.path().join("supervisor-profile.json").exists());
}

#[tokio::test]
async fn present_endpoint_with_malformed_credential_does_not_start_new_authority() {
    let temp = tempfile::tempdir().unwrap();
    let paths = SupervisorPaths::new(temp.path().to_path_buf()).unwrap();
    filesystem::secure_directory(&paths.root).unwrap();
    let token = auth::load_or_create_token(&paths).unwrap();
    std::fs::write(&paths.token, b"malformed").unwrap();
    let listener = tokio::net::UnixListener::bind(&paths.socket).unwrap();
    let mut execution = crate::NativeExecutionConfig::for_current_executable().unwrap();
    execution.executable = temp.path().join("must-not-launch");
    let config = LocalSupervisorConfig::new(execution, paths).unwrap();
    let client = LocalSupervisorClient::new(config);
    assert!(matches!(
        client.connect_or_start().await,
        Err(NativeExecutionError::Protocol(_))
    ));
    assert_ne!(
        std::fs::read(temp.path().join("auth-token")).unwrap(),
        token.as_bytes()
    );
    assert_eq!(
        std::fs::read(temp.path().join("auth-token")).unwrap(),
        b"malformed"
    );
    assert!(!temp.path().join("supervisor-profile.json").exists());
    drop(listener);
}

#[tokio::test]
async fn connected_silent_peer_cannot_extend_readiness_or_trigger_bootstrap() {
    let temp = tempfile::tempdir().unwrap();
    let paths = SupervisorPaths::new(temp.path().to_path_buf()).unwrap();
    filesystem::secure_directory(&paths.root).unwrap();
    auth::load_or_create_token(&paths).unwrap();
    let listener = tokio::net::UnixListener::bind(&paths.socket).unwrap();
    let config = LocalSupervisorConfig::new(
        crate::NativeExecutionConfig::for_current_executable().unwrap(),
        paths,
    )
    .unwrap();
    let peer = tokio::spawn(async move {
        let (_stream, _) = listener.accept().await.unwrap();
        std::future::pending::<()>().await;
    });
    let client = LocalSupervisorClient::new(config);
    let result = tokio::time::timeout(Duration::from_secs(7), client.connect_or_start())
        .await
        .expect("connected peer exceeded the readiness bound");
    peer.abort();
    assert!(
        matches!(result, Err(NativeExecutionError::SupervisorTransport { source })
        if source.kind() == std::io::ErrorKind::TimedOut)
    );
    assert!(!temp.path().join("supervisor-profile.json").exists());
}

#[tokio::test]
async fn over_limit_request_rejects_before_opening_a_channel() {
    let temp = tempfile::tempdir().unwrap();
    let paths = SupervisorPaths::new(temp.path().to_path_buf()).unwrap();
    let config = LocalSupervisorConfig::new(
        crate::NativeExecutionConfig::for_current_executable().unwrap(),
        paths,
    )
    .unwrap();
    let client = LocalSupervisorClient::new(config);
    let result = client
        .request(SupervisorCommand::ExecutionContext {
            idempotency_key: Some("x".repeat(SUPERVISOR_MAX_MESSAGE_BYTES as usize)),
        })
        .await;
    assert!(matches!(result, Err(NativeExecutionError::Encoding(error))
        if error.kind == runmat_canonical_codec::CanonicalCodecErrorKind::ResourceLimit));
    assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 0);
}

#[tokio::test]
async fn original_failed_startup_retains_exit_and_bounded_raw_stderr() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("original-supervisor");
    std::fs::write(
        &path,
        b"#!/bin/sh\nprintf 'original startup error with long suffix' >&2\ndd if=/dev/zero bs=1048576 count=1 1>&2 2>/dev/null\nexit 23\n",
    )
    .unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
    let mut execution = crate::NativeExecutionConfig::for_current_executable().unwrap();
    execution.executable = path;
    execution.max_stderr_bytes = 8;
    let config = LocalSupervisorConfig::new(
        execution,
        SupervisorPaths::new(temp.path().join("authority")).unwrap(),
    )
    .unwrap();
    let error = LocalSupervisorClient::new(config)
        .connect_or_start()
        .await
        .unwrap_err();
    let NativeExecutionError::SupervisorStartup {
        cause,
        exit,
        stderr,
    } = error
    else {
        panic!("lost original startup failure: {error}");
    };
    assert_eq!(
        exit.map(|status| status.code),
        Some(Some(23)),
        "cause={cause:?}, stderr={stderr:?}"
    );
    assert_eq!(stderr.bytes.as_ref(), b"original");
    assert_eq!(
        stderr.observed_bytes,
        Some(b"original startup error with long suffix".len() as u64 + 1024 * 1024)
    );
    assert!(stderr.truncated);
    assert_eq!(
        stderr.completion,
        runmat_process_host::ProcessOutputCompletion::EndOfStream
    );
}
