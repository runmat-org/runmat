use runmat_process_host::ipc::{
    authenticate_driver, authenticate_host, HostHandshake, SessionSecret,
};

#[tokio::test]
async fn both_stdio_peers_authenticate_and_negotiate_limits() {
    let secret = SessionSecret::generate();
    let host_secret = SessionSecret::from_hex(&secret.expose_hex()).unwrap();
    let (driver_stream, host_stream) = tokio::io::duplex(16 * 1024);
    let (mut driver_reader, mut driver_writer) = tokio::io::split(driver_stream);
    let (mut host_reader, mut host_writer) = tokio::io::split(host_stream);

    let driver = authenticate_driver(
        &mut driver_reader,
        &mut driver_writer,
        HostHandshake::new("runmat.extension", 1, 4096),
        &secret,
    );
    let host = authenticate_host(
        &mut host_reader,
        &mut host_writer,
        HostHandshake::new("runmat.extension", 1, 2048),
        &host_secret,
    );
    let (driver, host) = tokio::join!(driver, host);

    assert_eq!(driver.unwrap().limits.max_message_bytes, 2048);
    assert_eq!(host.unwrap().limits.max_message_bytes, 2048);
}

#[tokio::test]
async fn mismatched_secrets_fail_without_disclosing_secret_material() {
    let driver_secret = SessionSecret::generate();
    let host_secret = SessionSecret::generate();
    let (driver_stream, host_stream) = tokio::io::duplex(16 * 1024);
    let (mut driver_reader, mut driver_writer) = tokio::io::split(driver_stream);
    let (mut host_reader, mut host_writer) = tokio::io::split(host_stream);

    let driver = authenticate_driver(
        &mut driver_reader,
        &mut driver_writer,
        HostHandshake::new("runmat.extension", 1, 4096),
        &driver_secret,
    );
    let host = authenticate_host(
        &mut host_reader,
        &mut host_writer,
        HostHandshake::new("runmat.extension", 1, 4096),
        &host_secret,
    );
    let (driver, host) = tokio::join!(driver, host);

    let error = driver.unwrap_err().to_string();
    assert_eq!(
        error,
        "local IPC protocol error: IPC peer authentication failed"
    );
    assert!(host.is_err());
    assert_eq!(format!("{driver_secret:?}"), "SessionSecret([REDACTED])");
}

#[test]
fn encoded_secrets_are_exact_and_reject_malformed_input() {
    let secret = SessionSecret::generate();
    let encoded = secret.expose_hex();
    assert_eq!(encoded.len(), 64);
    assert_eq!(
        SessionSecret::from_hex(&encoded).unwrap().expose_hex(),
        encoded
    );
    assert!(SessionSecret::from_hex("short").is_err());
    assert!(SessionSecret::from_hex(&"z".repeat(64)).is_err());
}
