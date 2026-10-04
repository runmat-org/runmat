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
    pub byte_order: RuntimeTcpByteOrder,
}

#[derive(Debug, Clone)]
pub struct RuntimeTcpClient {
    pub authority: ForeignRef,
    pub local_addr: SocketAddr,
    pub peer_addr: SocketAddr,
}

#[derive(Debug, Clone)]
pub struct RuntimeTcpServer {
    pub authority: ForeignRef,
    pub local_addr: SocketAddr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeTcpReadMode {
    Available,
    Count(usize),
    Line,
}

#[derive(Debug)]
pub struct RuntimeTcpReadResult {
    pub bytes: Vec<u8>,
    pub connection_closed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeTcpCloseScope {
    Clients,
    Servers,
    All,
}

/// The port owns all socket state. Dropping an uncompleted future must stop
/// further I/O; a write may already have sent a prefix and is never replayed.
/// Reads retain unpublished bytes in the service across cancellation/drop.
pub trait RuntimeTcpService {
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
        timeout: Option<Duration>,
    ) -> RuntimeServiceFuture<Result<RuntimeTcpReadResult, crate::tcp::TcpTransportError>>;

    fn write(
        &self,
        issuer: RuntimeTcpIssuer,
        client: ForeignRef,
        bytes: Vec<u8>,
        timeout: Option<Duration>,
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
