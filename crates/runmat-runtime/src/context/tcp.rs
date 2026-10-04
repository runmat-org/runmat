//! Session-owned TCP transport contract. A managed value grants access only
//! when the installed service recognizes its exact lease and issuer.

use super::{RuntimeContext, RuntimeContextLocalId, RuntimeServiceFuture};
use runmat_value::ForeignRef;
use std::net::SocketAddr;
use std::sync::Weak;
use std::time::Duration;

#[derive(Debug, Clone)]
pub enum RuntimeTcpIssuer {
    Session {
        identity: RuntimeContextLocalId,
        lease: Weak<()>,
    },
    Process,
}

impl RuntimeTcpIssuer {
    pub fn from_context(runtime: Option<&RuntimeContext>) -> Self {
        match runtime {
            Some(runtime) => Self::Session {
                identity: runtime.local_identity(),
                lease: runtime.session_lease(),
            },
            None => Self::Process,
        }
    }

    pub fn admits(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Session { identity, lease },
                Self::Session {
                    identity: other_identity,
                    lease: other_lease,
                },
            ) => {
                identity == other_identity
                    && lease.strong_count() != 0
                    && Weak::ptr_eq(lease, other_lease)
            }
            (Self::Process, Self::Process) => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeTcpByteOrder {
    Little,
    Big,
}

#[derive(Debug, Clone)]
pub struct RuntimeTcpConnect {
    pub host: String,
    pub port: u16,
    pub connect_timeout: Option<Duration>,
    pub io_timeout: Option<Duration>,
    pub byte_order: RuntimeTcpByteOrder,
}

#[derive(Debug, Clone)]
pub struct RuntimeTcpBind {
    pub address: String,
    pub port: u16,
    pub io_timeout: Option<Duration>,
    pub byte_order: RuntimeTcpByteOrder,
}

#[derive(Debug, Clone)]
pub struct RuntimeTcpClient {
    pub authority: ForeignRef,
    pub local_addr: SocketAddr,
    pub peer_addr: SocketAddr,
    pub byte_order: RuntimeTcpByteOrder,
}

#[derive(Debug, Clone)]
pub struct RuntimeTcpServer {
    pub authority: ForeignRef,
    pub local_addr: SocketAddr,
}

#[derive(Debug, Clone, Copy)]
pub struct RuntimeTcpClientStatus {
    pub connected: bool,
    pub bytes_available: usize,
    pub byte_order: RuntimeTcpByteOrder,
    pub timeout: Option<Duration>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeTcpReadMode {
    Available,
    Count(usize),
    Line,
}

pub struct RuntimeTcpReadResult {
    bytes: Vec<u8>,
    connection_closed: bool,
    publication: Option<Box<dyn RuntimeTcpReadPublication>>,
}

pub trait RuntimeTcpReadPublication: Send {
    fn rollback(self: Box<Self>, bytes: Vec<u8>);
}

impl RuntimeTcpReadResult {
    pub fn new(
        bytes: Vec<u8>,
        connection_closed: bool,
        publication: Box<dyn RuntimeTcpReadPublication>,
    ) -> Self {
        Self {
            bytes,
            connection_closed,
            publication: Some(publication),
        }
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn connection_closed(&self) -> bool {
        self.connection_closed
    }

    pub fn publish(mut self) -> Vec<u8> {
        self.publication.take();
        std::mem::take(&mut self.bytes)
    }
}

impl Drop for RuntimeTcpReadResult {
    fn drop(&mut self) {
        if let Some(publication) = self.publication.take() {
            publication.rollback(std::mem::take(&mut self.bytes));
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeTcpCloseScope {
    Clients,
    Servers,
    All,
}

/// The port owns all socket state. Dropping an uncompleted future must stop
/// further I/O; a write may already have sent a prefix and is never replayed.
/// Reads retain unpublished bytes through outer decode and the final
/// cancellation checkpoint; only `RuntimeTcpReadResult::publish` consumes the
/// exclusive receipt.
pub trait RuntimeTcpService {
    fn describe_client(
        &self,
        issuer: RuntimeTcpIssuer,
        client: ForeignRef,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpClientStatus, crate::tcp::TcpTransportError>>;

    fn connect(
        &self,
        issuer: RuntimeTcpIssuer,
        request: RuntimeTcpConnect,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpClient, crate::tcp::TcpTransportError>>;

    fn bind(
        &self,
        issuer: RuntimeTcpIssuer,
        request: RuntimeTcpBind,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpServer, crate::tcp::TcpTransportError>>;

    fn accept(
        &self,
        issuer: RuntimeTcpIssuer,
        server: ForeignRef,
        timeout: Option<Duration>,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpClient, crate::tcp::TcpTransportError>>;

    fn read(
        &self,
        issuer: RuntimeTcpIssuer,
        client: ForeignRef,
        mode: RuntimeTcpReadMode,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpReadResult, crate::tcp::TcpTransportError>>;

    fn write(
        &self,
        issuer: RuntimeTcpIssuer,
        client: ForeignRef,
        bytes: Vec<u8>,
    ) -> RuntimeServiceFuture<Result<usize, crate::tcp::TcpTransportError>>;

    fn close(
        &self,
        issuer: RuntimeTcpIssuer,
        endpoint: ForeignRef,
    ) -> RuntimeServiceFuture<Result<bool, crate::tcp::TcpTransportError>>;

    fn close_scope(
        &self,
        issuer: RuntimeTcpIssuer,
        scope: RuntimeTcpCloseScope,
    ) -> RuntimeServiceFuture<Result<usize, crate::tcp::TcpTransportError>>;
}
