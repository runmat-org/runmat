use super::registry::{self, Client, ClientIo, Registry, Resource, Server};
use crate::context::{
    RuntimeTcpBind, RuntimeTcpClient, RuntimeTcpClientStatus, RuntimeTcpConnect, RuntimeTcpIssuer,
    RuntimeTcpReadMode, RuntimeTcpReadPublication, RuntimeTcpReadResult, RuntimeTcpServer,
};
use crate::tcp::{TcpTransportError, TcpTransportErrorKind, TcpWriteProgress};
use runmat_value::ForeignRef;
use std::io;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn io_error(context: &'static str, source: io::Error) -> TcpTransportError {
    TcpTransportError::with_source(TcpTransportErrorKind::Io, context, source)
}

async fn timed<T>(
    duration: Option<Duration>,
    work: impl std::future::Future<Output = io::Result<T>>,
    context: &'static str,
) -> Result<T, TcpTransportError> {
    match duration {
        Some(duration) => tokio::time::timeout(duration, work)
            .await
            .map_err(|source| {
                TcpTransportError::with_source(TcpTransportErrorKind::Timeout, context, source)
            })?
            .map_err(|source| io_error(context, source)),
        None => work.await.map_err(|source| io_error(context, source)),
    }
}

async fn client_io<T>(
    client: &Client,
    work: impl std::future::Future<Output = io::Result<T>>,
    context: &'static str,
) -> Result<T, TcpTransportError> {
    let mut closed = client.closed.subscribe();
    if *closed.borrow() {
        return Err(TcpTransportError::new(
            TcpTransportErrorKind::Closed,
            "TCP client is closed",
        ));
    }
    tokio::select! {
        _ = closed.changed() => Err(TcpTransportError::new(TcpTransportErrorKind::Closed, "TCP client is closed")),
        result = timed(client.timeout, work, context) => {
            let value = result?;
            if *closed.borrow() {
                Err(TcpTransportError::new(TcpTransportErrorKind::Closed, "TCP client is closed"))
            } else {
                Ok(value)
            }
        },
    }
}

pub(super) async fn connect(
    registry: Arc<Mutex<Registry>>,
    issuer: RuntimeTcpIssuer,
    request: RuntimeTcpConnect,
) -> Result<RuntimeTcpClient, TcpTransportError> {
    let stream = timed(
        request.connect_timeout,
        tokio::net::TcpStream::connect((request.host.as_str(), request.port)),
        "TCP connection failed",
    )
    .await?;
    let local_addr = stream
        .local_addr()
        .map_err(|source| io_error("TCP local address unavailable", source))?;
    let peer_addr = stream
        .peer_addr()
        .map_err(|source| io_error("TCP peer address unavailable", source))?;
    let resource = Resource::Client(Arc::new(Client {
        io: Arc::new(tokio::sync::Mutex::new(ClientIo {
            stream,
            unread: Vec::new(),
            closed: false,
        })),
        timeout: request.io_timeout,
        byte_order: request.byte_order,
        closed: tokio::sync::watch::channel(false).0,
    }));
    let authority = registry::issue(&registry, issuer, resource, None)?;
    Ok(RuntimeTcpClient {
        authority,
        local_addr,
        peer_addr,
        byte_order: request.byte_order,
    })
}

pub(super) async fn bind(
    registry: Arc<Mutex<Registry>>,
    issuer: RuntimeTcpIssuer,
    request: RuntimeTcpBind,
) -> Result<RuntimeTcpServer, TcpTransportError> {
    let listener = tokio::net::TcpListener::bind((request.address.as_str(), request.port))
        .await
        .map_err(|source| io_error("TCP listener bind failed", source))?;
    let local_addr = listener
        .local_addr()
        .map_err(|source| io_error("TCP listener address unavailable", source))?;
    let resource = Resource::Server(Arc::new(Server {
        listener,
        timeout: request.io_timeout,
        byte_order: request.byte_order,
        closed: tokio::sync::watch::channel(false).0,
    }));
    let authority = registry::issue(&registry, issuer, resource, None)?;
    Ok(RuntimeTcpServer {
        authority,
        local_addr,
    })
}

pub(super) async fn accept(
    registry: Arc<Mutex<Registry>>,
    issuer: RuntimeTcpIssuer,
    authority: ForeignRef,
    override_timeout: Option<Duration>,
) -> Result<RuntimeTcpClient, TcpTransportError> {
    let server = registry::server(&registry, &issuer, &authority)?;
    let timeout = override_timeout.or(server.timeout);
    let mut closed = server.closed.subscribe();
    if *closed.borrow() {
        return Err(TcpTransportError::new(
            TcpTransportErrorKind::Closed,
            "TCP server is closed",
        ));
    }
    let (stream, peer_addr) = tokio::select! {
        _ = closed.changed() => return Err(TcpTransportError::new(TcpTransportErrorKind::Closed, "TCP server is closed")),
        result = timed(timeout, server.listener.accept(), "TCP accept failed") => result?,
    };
    // Close may have invalidated the server while the accept was pending.
    registry::server(&registry, &issuer, &authority)?;
    let local_addr = stream
        .local_addr()
        .map_err(|source| io_error("accepted TCP local address unavailable", source))?;
    let byte_order = server.byte_order;
    let resource = Resource::Client(Arc::new(Client {
        io: Arc::new(tokio::sync::Mutex::new(ClientIo {
            stream,
            unread: Vec::new(),
            closed: false,
        })),
        timeout,
        byte_order,
        closed: tokio::sync::watch::channel(false).0,
    }));
    let client = registry::issue(&registry, issuer, resource, Some(authority.handle))?;
    Ok(RuntimeTcpClient {
        authority: client,
        local_addr,
        peer_addr,
        byte_order,
    })
}

struct ReadReceipt {
    guard: tokio::sync::OwnedMutexGuard<ClientIo>,
}

pub(super) async fn describe_client(
    registry: Arc<Mutex<Registry>>,
    issuer: RuntimeTcpIssuer,
    authority: ForeignRef,
) -> Result<RuntimeTcpClientStatus, TcpTransportError> {
    let client = registry::client(&registry, &issuer, &authority)?;
    let mut io = Arc::clone(&client.io).lock_owned().await;
    let mut chunk = [0u8; 4096];
    while !io.closed && !*client.closed.borrow() {
        match io.stream.try_read(&mut chunk) {
            Ok(0) => {
                io.closed = true;
                break;
            }
            Ok(n) => {
                io.unread.try_reserve(n).map_err(|source| {
                    TcpTransportError::with_source(
                        TcpTransportErrorKind::Io,
                        "TCP available-byte buffer allocation failed",
                        source,
                    )
                })?;
                io.unread.extend_from_slice(&chunk[..n]);
            }
            Err(source) if source.kind() == io::ErrorKind::WouldBlock => break,
            Err(source) if source.kind() == io::ErrorKind::Interrupted => continue,
            Err(source) => return Err(io_error("TCP available-byte inspection failed", source)),
        }
        tokio::task::yield_now().await;
    }
    Ok(RuntimeTcpClientStatus {
        connected: !io.closed && !*client.closed.borrow(),
        bytes_available: io.unread.len(),
        byte_order: client.byte_order,
        timeout: client.timeout,
    })
}

impl RuntimeTcpReadPublication for ReadReceipt {
    fn rollback(mut self: Box<Self>, bytes: Vec<u8>) {
        if !bytes.is_empty() {
            let mut restored = bytes;
            restored.append(&mut self.guard.unread);
            self.guard.unread = restored;
        }
    }
}

pub(super) async fn read(
    registry: Arc<Mutex<Registry>>,
    issuer: RuntimeTcpIssuer,
    authority: ForeignRef,
    mode: RuntimeTcpReadMode,
) -> Result<RuntimeTcpReadResult, TcpTransportError> {
    let client = registry::client(&registry, &issuer, &authority)?;
    let mut io = Arc::clone(&client.io).lock_owned().await;
    let mut chunk = [0u8; 4096];
    match mode {
        RuntimeTcpReadMode::Count(count) => {
            while io.unread.len() < count && !io.closed {
                let n =
                    client_io(&client, io.stream.read(&mut chunk), "TCP read timed out").await?;
                if n == 0 {
                    io.closed = true;
                    break;
                }
                io.unread.try_reserve(n).map_err(|source| {
                    TcpTransportError::with_source(
                        TcpTransportErrorKind::Io,
                        "TCP read buffer allocation failed",
                        source,
                    )
                })?;
                io.unread.extend_from_slice(&chunk[..n]);
                tokio::task::yield_now().await;
            }
        }
        RuntimeTcpReadMode::Available => {
            if io.unread.is_empty() && !io.closed {
                let n =
                    client_io(&client, io.stream.read(&mut chunk), "TCP read timed out").await?;
                if n == 0 {
                    io.closed = true;
                } else {
                    io.unread.extend_from_slice(&chunk[..n]);
                }
            }
            while !io.closed {
                if *client.closed.borrow() {
                    return Err(TcpTransportError::new(
                        TcpTransportErrorKind::Closed,
                        "TCP client is closed",
                    ));
                }
                match io.stream.try_read(&mut chunk) {
                    Ok(0) => {
                        io.closed = true;
                        break;
                    }
                    Ok(n) => io.unread.extend_from_slice(&chunk[..n]),
                    Err(source) if source.kind() == io::ErrorKind::WouldBlock => break,
                    Err(source) if source.kind() == io::ErrorKind::Interrupted => continue,
                    Err(source) => return Err(io_error("TCP available-byte read failed", source)),
                }
                tokio::task::yield_now().await;
            }
        }
        RuntimeTcpReadMode::Line => {
            while !io.unread.contains(&b'\n') && !io.closed {
                let n = client_io(
                    &client,
                    io.stream.read(&mut chunk),
                    "TCP line read timed out",
                )
                .await?;
                if n == 0 {
                    io.closed = true;
                    break;
                }
                io.unread.extend_from_slice(&chunk[..n]);
                tokio::task::yield_now().await;
            }
        }
    }
    // A cloned output buffer is required while the exclusive receipt retains
    // authoritative bytes for cancellation rollback. The caller can publish
    // only after decoding and its final cancellation checkpoint.
    let available = io.unread.len();
    let count = match mode {
        RuntimeTcpReadMode::Count(count) => available.min(count),
        RuntimeTcpReadMode::Available => available,
        RuntimeTcpReadMode::Line => io
            .unread
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available, |index| index + 1),
    };
    let bytes = io.unread.drain(..count).collect();
    let closed = io.closed;
    Ok(RuntimeTcpReadResult::new(
        bytes,
        closed,
        Box::new(ReadReceipt { guard: io }),
    ))
}

pub(super) async fn write(
    registry: Arc<Mutex<Registry>>,
    issuer: RuntimeTcpIssuer,
    authority: ForeignRef,
    bytes: Vec<u8>,
) -> Result<usize, TcpTransportError> {
    let client = registry::client(&registry, &issuer, &authority)?;
    let mut io = Arc::clone(&client.io).lock_owned().await;
    if io.closed || *client.closed.borrow() {
        return Err(TcpTransportError::new(
            TcpTransportErrorKind::Closed,
            "TCP client is disconnected",
        )
        .with_write_progress(TcpWriteProgress::NoBytesSent));
    }
    let mut sent = 0usize;
    while sent < bytes.len() {
        let result = client_io(
            &client,
            io.stream.write(&bytes[sent..]),
            "TCP write timed out",
        )
        .await;
        match result {
            Ok(0) => {
                return Err(TcpTransportError::new(
                    TcpTransportErrorKind::Closed,
                    "TCP peer accepted zero write bytes",
                )
                .with_write_progress(TcpWriteProgress::PrefixSent(sent)));
            }
            Ok(n) => sent += n,
            Err(error) => {
                // The current write future may have reached the peer without
                // returning a byte count. A previous completed prefix is
                // known, but total delivery is not.
                return Err(error.with_write_progress(TcpWriteProgress::Indeterminate));
            }
        }
        tokio::task::yield_now().await;
    }
    Ok(sent)
}
