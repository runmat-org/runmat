//! Supervisor service futures own reconciliation and every connection lifetime.
//! Dropping the service drops them all; none can retain its kernel authority detached.
use super::auth::{constant_time_eq, load_or_create_token};
use super::filesystem::secure_directory;
use super::protocol::{
    SupervisorRequest, SupervisorResponse, SUPERVISOR_MAX_MESSAGE_BYTES,
    SUPERVISOR_PROTOCOL_VERSION,
};
use super::service::LocalSupervisor;
use super::{LocalSupervisorConfig, SupervisorPaths};
use crate::{NativeExecutionError, NativeExecutionResult};
use futures::StreamExt;
use runmat_process_host::ipc::{read_payload, write_payload, FrameLimits};
#[cfg(unix)]
use std::fs;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncWrite};

pub async fn run_local_supervisor(config: LocalSupervisorConfig) -> NativeExecutionResult<()> {
    secure_directory(&config.paths.root)?;
    let token = load_or_create_token(&config.paths)?;
    let supervisor = LocalSupervisor::open(config.clone())?;
    supervisor.reconcile().await?;
    let background = Arc::clone(&supervisor);
    let reconciliation = async move {
        // Startup already reconciled the recovered journal. Serve readiness
        // before starting the next background pass over those same records.
        let period = Duration::from_millis(250);
        let mut interval = tokio::time::interval_at(tokio::time::Instant::now() + period, period);
        let mut was_failing = false;
        loop {
            interval.tick().await;
            match background.reconcile().await {
                Err(error) => {
                    if !was_failing {
                        tracing::error!(error = %error, "local supervisor reconciliation failed");
                    }
                    was_failing = true;
                }
                Ok(()) => {
                    if was_failing {
                        tracing::info!("local supervisor reconciliation recovered");
                    }
                    was_failing = false;
                }
            }
            // An overdue interval can remain ready across every cycle. Reconciliation
            // must yield even then so IPC and cancellation can share this executor.
            tokio::task::yield_now().await;
        }
    };
    tokio::select! {
        result = run_listener(config.paths, supervisor, token) => result,
        _ = reconciliation => unreachable!("reconciliation loop cannot complete"),
    }
}

async fn handle_connection<S>(mut stream: S, supervisor: Arc<LocalSupervisor>, token: Arc<String>)
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let limits = FrameLimits {
        max_message_bytes: SUPERVISOR_MAX_MESSAGE_BYTES,
    };
    let response = match read_payload(&mut stream, limits).await {
        Ok(payload) => match serde_json::from_slice::<SupervisorRequest>(&payload) {
            Ok(request)
                if request.protocol_version == SUPERVISOR_PROTOCOL_VERSION
                    && supervisor
                        .config
                        .identity()
                        .is_ok_and(|identity| identity == request.config_identity)
                    && constant_time_eq(
                        request.authentication_token.as_bytes(),
                        token.as_bytes(),
                    ) =>
            {
                supervisor
                    .command(request.command, request.local_fea_organization)
                    .await
                    .unwrap_or_else(|error| {
                        SupervisorResponse::error("request_failed", error.to_string())
                    })
            }
            Ok(_) => SupervisorResponse::error(
                "authentication_failed",
                "local supervisor authentication or protocol negotiation failed",
            ),
            Err(error) => SupervisorResponse::error("malformed_request", error.to_string()),
        },
        Err(error) => SupervisorResponse::error("invalid_frame", error.to_string()),
    };
    let payload = match response_payload(&response) {
        Ok(payload) => payload,
        Err(error) => {
            tracing::error!(error = %error, "supervisor failure envelope could not be encoded");
            return;
        }
    };
    if let Err(error) = write_payload(&mut stream, &payload, limits).await {
        tracing::warn!(error = %error, "supervisor response could not be delivered");
    }
}

/// An oversized response becomes a bounded failure on the original channel,
/// rather than an unexplained disconnect or an unbounded temporary buffer.
fn response_payload(
    response: &SupervisorResponse,
) -> Result<Vec<u8>, runmat_canonical_codec::CanonicalCodecError> {
    match runmat_canonical_codec::encode_json_bounded(
        response,
        u64::from(SUPERVISOR_MAX_MESSAGE_BYTES),
    ) {
        Ok(payload) => Ok(payload),
        Err(error) => runmat_canonical_codec::encode_json_bounded(
            &SupervisorResponse::error("response_encoding_failed", error.to_string()),
            u64::from(SUPERVISOR_MAX_MESSAGE_BYTES),
        ),
    }
}

#[cfg(unix)]
async fn run_listener(
    paths: SupervisorPaths,
    supervisor: Arc<LocalSupervisor>,
    token: String,
) -> NativeExecutionResult<()> {
    use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};

    if paths.socket.exists() {
        fs::remove_file(&paths.socket).map_err(protocol_io)?;
    }
    let listener = tokio::net::UnixListener::bind(&paths.socket).map_err(protocol_io)?;
    fs::set_permissions(&paths.socket, fs::Permissions::from_mode(0o600)).map_err(protocol_io)?;
    let token_owner = fs::metadata(&paths.token).map_err(protocol_io)?.uid();
    let token = Arc::new(token);
    super::bootstrap::release_startup_diagnostics()?;
    let mut connections = futures::stream::FuturesUnordered::new();
    loop {
        let (stream, _) = tokio::select! {
            accepted = listener.accept() => accepted.map_err(protocol_io)?,
            _ = connections.next(), if !connections.is_empty() => continue,
        };
        let peer_owner = stream.peer_cred().map_err(protocol_io)?.uid();
        if peer_owner != token_owner {
            continue;
        }
        connections.push(handle_connection(
            stream,
            Arc::clone(&supervisor),
            Arc::clone(&token),
        ));
    }
}

#[cfg(windows)]
async fn run_listener(
    paths: SupervisorPaths,
    supervisor: Arc<LocalSupervisor>,
    token: String,
) -> NativeExecutionResult<()> {
    use tokio::net::windows::named_pipe::ServerOptions;

    let token = Arc::new(token);
    let mut first = true;
    let mut connections = futures::stream::FuturesUnordered::new();
    loop {
        let server = ServerOptions::new()
            .first_pipe_instance(first)
            .reject_remote_clients(true)
            .create(&paths.pipe)
            .map_err(protocol_io)?;
        if first {
            super::bootstrap::release_startup_diagnostics()?;
        }
        first = false;
        loop {
            tokio::select! {
                connected = server.connect() => { connected.map_err(protocol_io)?; break; },
                _ = connections.next(), if !connections.is_empty() => {},
            }
        }
        connections.push(handle_connection(
            server,
            Arc::clone(&supervisor),
            Arc::clone(&token),
        ));
    }
}

fn protocol_io(error: std::io::Error) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

#[cfg(all(test, unix))]
mod tests;
