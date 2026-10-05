use crate::context::{
    RuntimeContext, RuntimeContextLocalId, RuntimeTcpByteOrder, RuntimeTcpCloseScope,
    RuntimeTcpIssuer,
};
use crate::tcp::{TcpTransportError, TcpTransportErrorKind};
use runmat_types::{ForeignAffinity, ForeignLifetime, ForeignOwnership, ForeignTypeIdentity};
use runmat_value::{ForeignRef, ForeignResourceKey, ForeignResourceRelease, WeakForeignRef};
use std::collections::{BTreeMap, BTreeSet, HashSet};
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

/// A weak, issuer-bound proof for an explicitly closed managed value. This
/// remembers no socket owner and is removed when the last value lease drops.
struct ClosedRecord {
    issuer: RuntimeTcpIssuer,
    lease: WeakForeignRef,
    name: &'static str,
}

impl ClosedRecord {
    fn from_record(record: &Record) -> Self {
        Self {
            issuer: record.issuer.clone(),
            lease: record.lease.clone(),
            name: record.resource.name(),
        }
    }
}

#[derive(Default)]
pub(super) struct Registry {
    next: u64,
    records: BTreeMap<u64, Record>,
    closed: BTreeMap<u64, ClosedRecord>,
    registered_sessions: HashSet<RuntimeContextLocalId>,
}

pub(super) fn register_session_cleanup(
    registry: &Arc<Mutex<Registry>>,
    runtime: &RuntimeContext,
) -> Result<(), TcpTransportError> {
    let identity = runtime.local_identity();
    let issuer = RuntimeTcpIssuer::from_context(Some(runtime));
    {
        let mut state = registry.lock().map_err(|_| unavailable())?;
        if !state.registered_sessions.insert(identity) {
            return Ok(());
        }
    }
    let weak = Arc::downgrade(registry);
    runtime.register_session_resource_cleanup(move || {
        if let Some(registry) = weak.upgrade() {
            // The context still owns its lease while its cleanup callbacks
            // run. Resource close/watch notification happens outside locks.
            let _ = close_scope(&registry, &issuer, RuntimeTcpCloseScope::All);
            if let Ok(mut state) = registry.lock() {
                state.registered_sessions.remove(&identity);
            }
        }
    });
    Ok(())
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
            let removed = registry.lock().ok().map(|mut state| {
                state.closed.remove(&key.handle);
                let mut removed = Vec::new();
                if let Some(record) = state.records.remove(&key.handle) {
                    let server = matches!(&record.resource, Resource::Server(_));
                    removed.push(record);
                    if server {
                        let children: Vec<u64> = state
                            .records
                            .iter()
                            .filter_map(|(id, record)| {
                                (record.parent_server == Some(key.handle)).then_some(*id)
                            })
                            .collect();
                        for child in children {
                            if let Some(record) = state.records.remove(&child) {
                                state
                                    .closed
                                    .insert(child, ClosedRecord::from_record(&record));
                                removed.push(record);
                            }
                        }
                    }
                }
                removed
            });
            for record in removed.into_iter().flatten() {
                record.resource.close();
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
    admitted_lease(
        &record.issuer,
        &record.lease,
        record.resource.name(),
        issuer,
        reference,
    )
}

fn admitted_lease(
    owner: &RuntimeTcpIssuer,
    lease: &WeakForeignRef,
    name: &'static str,
    issuer: &RuntimeTcpIssuer,
    reference: &ForeignRef,
) -> bool {
    owner.admits(issuer)
        && reference.is_managed()
        && reference.host_identity == HOST
        && reference.generation == GENERATION
        && reference.ownership == ForeignOwnership::Shared
        && reference.affinity == ForeignAffinity::OriginProcess
        && reference.type_identity.family == "runmat.tcp"
        && reference.type_identity.name == name
        && reference.type_identity.version == 1
        && lease.upgrade().is_some_and(|issued| {
            issued.shares_managed_lease(reference)
                && issued.resource_key() == reference.resource_key()
                && issued.lifetime == reference.lifetime
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
    let removed = {
        let mut state = registry.lock().map_err(|_| unavailable())?;
        let Some(record) = state.records.get(&reference.handle) else {
            return if state.closed.get(&reference.handle).is_some_and(|closed| {
                admitted_lease(
                    &closed.issuer,
                    &closed.lease,
                    closed.name,
                    issuer,
                    reference,
                )
            }) {
                Ok(false)
            } else {
                Err(invalid())
            };
        };
        if !admitted(record, issuer, reference) {
            return Err(invalid());
        }
        let server = matches!(&record.resource, Resource::Server(_));
        let mut removed = Vec::new();
        if let Some(record) = state.records.remove(&reference.handle) {
            state
                .closed
                .insert(reference.handle, ClosedRecord::from_record(&record));
            removed.push(record);
        }
        if server {
            let children: Vec<u64> = state
                .records
                .iter()
                .filter_map(|(id, record)| {
                    (record.parent_server == Some(reference.handle)).then_some(*id)
                })
                .collect();
            for child in children {
                if let Some(record) = state.records.remove(&child) {
                    state
                        .closed
                        .insert(child, ClosedRecord::from_record(&record));
                    removed.push(record);
                }
            }
        }
        removed
    };
    for record in removed {
        record.resource.close();
    }
    Ok(true)
}

pub(super) fn close_scope(
    registry: &Arc<Mutex<Registry>>,
    issuer: &RuntimeTcpIssuer,
    scope: RuntimeTcpCloseScope,
) -> Result<usize, TcpTransportError> {
    let removed = {
        let mut state = registry.lock().map_err(|_| unavailable())?;
        let mut selected: BTreeSet<u64> = state
            .records
            .iter()
            .filter_map(|(id, record)| {
                (record.issuer.admits(issuer)
                    && matches!(
                        (scope, &record.resource),
                        (RuntimeTcpCloseScope::All, _)
                            | (RuntimeTcpCloseScope::Clients, Resource::Client(_))
                            | (RuntimeTcpCloseScope::Servers, Resource::Server(_))
                    ))
                .then_some(*id)
            })
            .collect();
        if matches!(
            scope,
            RuntimeTcpCloseScope::Servers | RuntimeTcpCloseScope::All
        ) {
            let children: Vec<u64> = state
                .records
                .iter()
                .filter_map(|(id, record)| {
                    record
                        .parent_server
                        .filter(|parent| selected.contains(parent) && record.issuer.admits(issuer))
                        .map(|_| *id)
                })
                .collect();
            selected.extend(children);
        }
        selected
            .into_iter()
            .filter_map(|id| {
                state.records.remove(&id).inspect(|record| {
                    state.closed.insert(id, ClosedRecord::from_record(record));
                })
            })
            .collect::<Vec<_>>()
    };
    let count = removed.len();
    for record in removed {
        record.resource.close();
    }
    Ok(count)
}

pub(super) fn shutdown_all(registry: &Arc<Mutex<Registry>>) {
    let removed = registry
        .lock()
        .ok()
        .map(|mut state| std::mem::take(&mut state.records));
    if let Some(removed) = removed {
        for record in removed.into_values() {
            record.resource.close();
        }
    }
}

#[cfg(test)]
pub(super) fn closed_count(registry: &Arc<Mutex<Registry>>) -> usize {
    registry.lock().expect("TCP test registry").closed.len()
}

#[cfg(test)]
pub(super) fn active_count(registry: &Arc<Mutex<Registry>>) -> usize {
    registry.lock().expect("TCP test registry").records.len()
}
