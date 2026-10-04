use crate::context::{RuntimeTcpByteOrder, RuntimeTcpCloseScope, RuntimeTcpIssuer};
use crate::tcp::{TcpTransportError, TcpTransportErrorKind};
use runmat_types::{ForeignAffinity, ForeignLifetime, ForeignOwnership, ForeignTypeIdentity};
use runmat_value::{ForeignRef, ForeignResourceKey, ForeignResourceRelease, WeakForeignRef};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream};

const HOST: &str = "runmat.tcp.session";
const GENERATION: u64 = 1;

pub(super) struct ClientIo {
    pub(super) stream: TcpStream,
    /// Bytes read from the socket but not yet published. A dropped pending
    /// read leaves them here for the next call rather than consuming them.
    pub(super) unread: Vec<u8>,
    pub(super) closed: bool,
}

pub(super) struct Client {
    pub(super) io: Arc<tokio::sync::Mutex<ClientIo>>,
    pub(super) timeout: Option<Duration>,
    pub(super) byte_order: RuntimeTcpByteOrder,
    pub(super) closed: tokio::sync::watch::Sender<bool>,
}

pub(super) struct Server {
    pub(super) listener: TcpListener,
    pub(super) timeout: Option<Duration>,
    pub(super) byte_order: RuntimeTcpByteOrder,
    pub(super) closed: tokio::sync::watch::Sender<bool>,
}

#[derive(Clone)]
pub(super) enum Resource {
    Client(Arc<Client>),
    Server(Arc<Server>),
}

impl Resource {
    fn name(&self) -> &'static str {
        match self {
            Self::Client(_) => "tcpclient",
            Self::Server(_) => "tcpserver",
        }
    }

    fn close(&self) {
        match self {
            Self::Client(client) => {
                client.closed.send_replace(true);
            }
            Self::Server(server) => {
                server.closed.send_replace(true);
            }
        }
    }
}

struct Record {
    resource: Resource,
    issuer: RuntimeTcpIssuer,
    lease: WeakForeignRef,
    parent_server: Option<u64>,
}

#[derive(Default)]
pub(super) struct Registry {
    next: u64,
    records: BTreeMap<u64, Record>,
}

struct Release(Weak<Mutex<Registry>>);

impl std::fmt::Debug for Release {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("TcpAuthorityRelease")
    }
}

impl ForeignResourceRelease for Release {
    fn release(&self, key: &ForeignResourceKey) {
        if key.host_identity != HOST || key.generation != GENERATION {
            return;
        }
        if let Some(registry) = self.0.upgrade() {
            if let Ok(mut registry) = registry.lock() {
                if let Some(record) = registry.records.remove(&key.handle) {
                    record.resource.close();
                }
            }
        }
    }
}

fn invalid() -> TcpTransportError {
    TcpTransportError::new(
        TcpTransportErrorKind::InvalidAuthority,
        "TCP value does not carry a live authority issued by this session",
    )
}

fn unavailable() -> TcpTransportError {
    TcpTransportError::new(
        TcpTransportErrorKind::Io,
        "TCP service registry unavailable",
    )
}

fn admitted(record: &Record, issuer: &RuntimeTcpIssuer, reference: &ForeignRef) -> bool {
    record.issuer.admits(issuer)
        && reference.is_managed()
        && reference.host_identity == HOST
        && reference.generation == GENERATION
        && reference.ownership == ForeignOwnership::Shared
        && reference.affinity == ForeignAffinity::OriginProcess
        && reference.type_identity.family == "runmat.tcp"
        && reference.type_identity.name == record.resource.name()
        && reference.type_identity.version == 1
        && record.lease.upgrade().is_some_and(|issued| {
            issued.shares_managed_lease(reference)
                && issued.resource_key() == reference.resource_key()
        })
}

pub(super) fn issue(
    registry: &Arc<Mutex<Registry>>,
    issuer: RuntimeTcpIssuer,
    resource: Resource,
    parent_server: Option<u64>,
) -> Result<ForeignRef, TcpTransportError> {
    let mut state = registry.lock().map_err(|_| unavailable())?;
    if let Some(parent) = parent_server {
        let live_parent = state.records.get(&parent).is_some_and(|record| {
            record.issuer.admits(&issuer)
                && matches!(&record.resource, Resource::Server(_))
                && record.lease.upgrade().is_some()
        });
        if !live_parent {
            return Err(invalid());
        }
    }
    let handle = state.next.checked_add(1).ok_or_else(|| {
        TcpTransportError::new(TcpTransportErrorKind::Io, "TCP handle space exhausted")
    })?;
    state.next = handle;
    let lifetime = match &issuer {
        RuntimeTcpIssuer::Session { lease, .. } if lease.strong_count() != 0 => {
            ForeignLifetime::Session
        }
        RuntimeTcpIssuer::Process => ForeignLifetime::Persistent,
        RuntimeTcpIssuer::Session { .. } => return Err(invalid()),
    };
    let reference = ForeignRef::managed(
        ForeignResourceKey {
            host_identity: HOST.into(),
            handle,
            generation: GENERATION,
        },
        ForeignTypeIdentity {
            family: "runmat.tcp".into(),
            name: resource.name().into(),
            version: 1,
        },
        ForeignOwnership::Shared,
        ForeignAffinity::OriginProcess,
        lifetime,
        Arc::new(Release(Arc::downgrade(registry))),
    );
    let lease = reference
        .downgrade()
        .expect("new TCP authority has a managed lease");
    state.records.insert(
        handle,
        Record {
            resource,
            issuer,
            lease,
            parent_server,
        },
    );
    Ok(reference)
}

pub(super) fn client(
    registry: &Arc<Mutex<Registry>>,
    issuer: &RuntimeTcpIssuer,
    reference: &ForeignRef,
) -> Result<Arc<Client>, TcpTransportError> {
    let state = registry.lock().map_err(|_| unavailable())?;
    let record = state.records.get(&reference.handle).ok_or_else(invalid)?;
    if !admitted(record, issuer, reference) {
        return Err(invalid());
    }
    match &record.resource {
        Resource::Client(client) => Ok(Arc::clone(client)),
        Resource::Server(_) => Err(invalid()),
    }
}

pub(super) fn server(
    registry: &Arc<Mutex<Registry>>,
    issuer: &RuntimeTcpIssuer,
    reference: &ForeignRef,
) -> Result<Arc<Server>, TcpTransportError> {
    let state = registry.lock().map_err(|_| unavailable())?;
    let record = state.records.get(&reference.handle).ok_or_else(invalid)?;
    if !admitted(record, issuer, reference) {
        return Err(invalid());
    }
    match &record.resource {
        Resource::Server(server) => Ok(Arc::clone(server)),
        Resource::Client(_) => Err(invalid()),
    }
}

pub(super) fn close(
    registry: &Arc<Mutex<Registry>>,
    issuer: &RuntimeTcpIssuer,
    reference: &ForeignRef,
) -> Result<bool, TcpTransportError> {
    let mut state = registry.lock().map_err(|_| unavailable())?;
    let record = state.records.get(&reference.handle).ok_or_else(invalid)?;
    if !admitted(record, issuer, reference) {
        return Err(invalid());
    }
    let server = matches!(&record.resource, Resource::Server(_));
    if let Some(record) = state.records.remove(&reference.handle) {
        record.resource.close();
    }
    if server {
        state.records.retain(|_, record| {
            let retain = record.parent_server != Some(reference.handle);
            if !retain {
                record.resource.close();
            }
            retain
        });
    }
    Ok(true)
}

pub(super) fn close_scope(
    registry: &Arc<Mutex<Registry>>,
    issuer: &RuntimeTcpIssuer,
    scope: RuntimeTcpCloseScope,
) -> Result<usize, TcpTransportError> {
    let mut state = registry.lock().map_err(|_| unavailable())?;
    let before = state.records.len();
    state.records.retain(|_, record| {
        if !record.issuer.admits(issuer) {
            return true;
        }
        let retain = !matches!(
            (scope, &record.resource),
            (RuntimeTcpCloseScope::All, _)
                | (RuntimeTcpCloseScope::Clients, Resource::Client(_))
                | (RuntimeTcpCloseScope::Servers, Resource::Server(_))
        );
        if !retain {
            record.resource.close();
        }
        retain
    });
    Ok(before - state.records.len())
}

pub(super) fn shutdown_all(registry: &Arc<Mutex<Registry>>) {
    if let Ok(mut registry) = registry.lock() {
        for record in registry.records.values() {
            record.resource.close();
        }
        registry.records.clear();
    }
}
