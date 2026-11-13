use super::*;

const WORKER_SUBPROCESS_ROLE_ENV: &str = "RUNMAT_DRIVER_REPLACEMENT_WORKER_ROLE";

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct WorkerSubprocessRole {
    relay_url: String,
    allocation_id: String,
    inventory: ResourceInventory,
    run_key: [u8; 32],
    driver_fence: u64,
    held_attempt: Option<super::super::worker_execution::HeldWorkloadAttempt>,
    journal: super::super::worker_attempt_journal::WorkerAttemptJournalOwner,
}

/// The same named unit test is the real worker process under this typed role.
/// Normal test invocation never takes this branch; no ignored/no-op test is
/// used as an executable surrogate.
pub(super) fn enter_worker_subprocess_if_requested() {
    let Ok(encoded) = std::env::var(WORKER_SUBPROCESS_ROLE_ENV) else {
        return;
    };
    let role: WorkerSubprocessRole =
        serde_json::from_str(&encoded).expect("typed replacement worker subprocess role");
    let key = RunKeyMaterial::from_entropy(role.run_key).expect("worker run key");
    let result = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("replacement worker process runtime")
        .block_on(run_worker(
            role.relay_url,
            &role.allocation_id,
            &role.inventory,
            key,
            role.driver_fence,
            role.held_attempt,
            role.journal,
        ));
    match result {
        Ok(()) => std::process::exit(0),
        Err(error) => {
            eprintln!("replacement worker process ended: {error}");
            std::process::exit(2)
        }
    }
}

pub(super) async fn spawn_worker_process(
    test_name: &str,
    relay_url: String,
    allocation: &DriverWorkerAllocation,
    inventory: &ResourceInventory,
    run_key: &RunKeyMaterial,
    held_attempt: Option<super::super::worker_execution::HeldWorkloadAttempt>,
    journal: super::super::worker_attempt_journal::WorkerAttemptJournalOwner,
) -> runmat_process_host::ChildProcess {
    let role = WorkerSubprocessRole {
        relay_url,
        allocation_id: allocation.allocation_lease_id.clone(),
        inventory: inventory.clone(),
        run_key: *run_key.expose_for_recipient_envelope(),
        driver_fence: journal.driver_fencing_token,
        held_attempt,
        journal,
    };
    let mut command = runmat_process_host::HostCommand::new(std::env::current_exe().unwrap());
    command.arguments = vec!["--exact".into(), test_name.into(), "--nocapture".into()];
    command.environment_policy = runmat_process_host::environment::EnvironmentPolicy::Allow(
        runmat_process_host::environment::EnvironmentAllowlist::platform_runtime(),
    );
    command.environment.insert(
        WORKER_SUBPROCESS_ROLE_ENV.into(),
        serde_json::to_string(&role).unwrap(),
    );
    command.stdio = runmat_process_host::StdioPolicy::Null;
    command.lifetime = runmat_process_host::ChildLifetime::Owned;
    command
        .spawn()
        .await
        .expect("owned replacement worker OS process")
}

pub(super) struct RelayServer {
    listener: tokio::net::TcpListener,
    address: SocketAddr,
}

impl RelayServer {
    pub(super) async fn bind() -> Self {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        Self { listener, address }
    }

    pub(super) fn http_base(&self) -> String {
        format!("http://{}", self.address)
    }

    pub(super) fn ws_url(&self, path: &str) -> String {
        format!("ws://{}/{path}", self.address)
    }

    pub(super) async fn forward_pair(self) -> Result<(), String> {
        let (first, _) = self
            .listener
            .accept()
            .await
            .map_err(|error| error.to_string())?;
        let (second, _) = self
            .listener
            .accept()
            .await
            .map_err(|error| error.to_string())?;
        let first = accept_relay_socket(first).await?;
        let second = accept_relay_socket(second).await?;
        let (mut first_sink, mut first_source) = first.split();
        let (mut second_sink, mut second_source) = second.split();
        loop {
            tokio::select! {
                message = first_source.next() => match message {
                    Some(Ok(message)) => {
                        if let Err(error) = second_sink.send(message).await {
                            if relay_closed(&error) { break; }
                            return Err(error.to_string());
                        }
                    },
                    Some(Err(error)) if relay_closed(&error) => break,
                    Some(Err(error)) => return Err(error.to_string()),
                    None => break,
                },
                message = second_source.next() => match message {
                    Some(Ok(message)) => {
                        if let Err(error) = first_sink.send(message).await {
                            if relay_closed(&error) { break; }
                            return Err(error.to_string());
                        }
                    },
                    Some(Err(error)) if relay_closed(&error) => break,
                    Some(Err(error)) => return Err(error.to_string()),
                    None => break,
                },
            }
        }
        Ok(())
    }
}

async fn accept_relay_socket(
    stream: tokio::net::TcpStream,
) -> Result<tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>, String> {
    tokio_tungstenite::accept_hdr_async(
        stream,
        |request: &tokio_tungstenite::tungstenite::handshake::server::Request,
         mut response: tokio_tungstenite::tungstenite::handshake::server::Response| {
            if let Some(protocol) = request
                .headers()
                .get(tokio_tungstenite::tungstenite::http::header::SEC_WEBSOCKET_PROTOCOL)
            {
                response.headers_mut().insert(
                    tokio_tungstenite::tungstenite::http::header::SEC_WEBSOCKET_PROTOCOL,
                    protocol.clone(),
                );
            }
            Ok(response)
        },
    )
    .await
    .map_err(|error| error.to_string())
}

fn relay_closed(error: &tokio_tungstenite::tungstenite::Error) -> bool {
    matches!(
        error,
        tokio_tungstenite::tungstenite::Error::ConnectionClosed
            | tokio_tungstenite::tungstenite::Error::AlreadyClosed
            | tokio_tungstenite::tungstenite::Error::Protocol(
                tokio_tungstenite::tungstenite::error::ProtocolError::ResetWithoutClosingHandshake
            )
    )
}

pub(super) async fn drive_generation<R, W, D>(
    relay: R,
    worker: W,
    driver: D,
) -> crate::NativeExecutionResult<()>
where
    R: std::future::Future<Output = Result<(), String>>,
    W: std::future::Future<Output = crate::NativeExecutionResult<()>>,
    D: std::future::Future<Output = crate::NativeExecutionResult<()>>,
{
    tokio::pin!(relay);
    tokio::pin!(worker);
    tokio::pin!(driver);
    tokio::select! {
        result = &mut driver => result,
        result = &mut relay => panic!("relay generation ended before its driver: {result:?}"),
        result = &mut worker => panic!("worker generation ended before its driver: {result:?}"),
    }
}

pub(super) async fn interrupt_generation_after_commit<R, W, D, C>(
    relay: R,
    worker: W,
    driver: D,
    committed: C,
) where
    R: std::future::Future<Output = Result<(), String>>,
    W: std::future::Future<Output = crate::NativeExecutionResult<()>>,
    D: std::future::Future<Output = crate::NativeExecutionResult<()>>,
    C: std::future::Future<Output = ()>,
{
    tokio::pin!(relay);
    tokio::pin!(worker);
    tokio::pin!(driver);
    tokio::pin!(committed);
    tokio::select! {
        () = &mut committed => {}
        result = &mut driver => panic!("driver ended before its recovery checkpoint committed: {result:?}"),
        result = &mut relay => panic!("relay ended before its recovery checkpoint committed: {result:?}"),
        result = &mut worker => panic!("worker ended before its recovery checkpoint committed: {result:?}"),
    }
}
