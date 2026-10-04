//! Shared live TCP server state for construction, acceptance, and closure.

use once_cell::sync::OnceCell;
use std::collections::HashMap;
use std::net::{SocketAddr, TcpListener};
use std::sync::{Arc, Mutex};

pub(crate) const DEFAULT_TIMEOUT_SECONDS: f64 = 10.0;
pub(crate) const HANDLE_ID_FIELD: &str = "__tcpserver_id";

pub(crate) type SharedTcpServer = Arc<Mutex<TcpServerState>>;

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct TcpServerState {
    pub(crate) id: u64,
    pub(crate) listener: TcpListener,
    pub(crate) requested_address: String,
    pub(crate) local_addr: SocketAddr,
    pub(crate) timeout: f64,
    pub(crate) name: Option<String>,
    pub(crate) byte_order: String,
}

#[derive(Default)]
struct TcpServerRegistry {
    next_id: u64,
    servers: HashMap<u64, SharedTcpServer>,
}

static TCP_SERVER_REGISTRY: OnceCell<Mutex<TcpServerRegistry>> = OnceCell::new();

fn registry() -> &'static Mutex<TcpServerRegistry> {
    TCP_SERVER_REGISTRY.get_or_init(|| Mutex::new(TcpServerRegistry::default()))
}

pub(crate) fn insert_server(
    listener: TcpListener,
    requested_address: String,
    local_addr: SocketAddr,
    timeout: f64,
    name: Option<String>,
    byte_order: String,
) -> u64 {
    let mut guard = registry()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    guard.next_id = guard.next_id.wrapping_add(1);
    let id = guard.next_id;
    let state = TcpServerState {
        id,
        listener,
        requested_address,
        local_addr,
        timeout,
        name,
        byte_order,
    };
    guard.servers.insert(id, Arc::new(Mutex::new(state)));
    id
}

#[allow(dead_code)]
pub(crate) fn server_handle(id: u64) -> Option<SharedTcpServer> {
    registry()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .servers
        .get(&id)
        .cloned()
}

pub(crate) fn close_server(id: u64) -> bool {
    let entry = {
        let mut guard = registry()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        guard.servers.remove(&id)
    };
    entry.is_some()
}

pub(crate) fn close_all_servers() -> usize {
    let entries = {
        let mut guard = registry()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        guard.servers.drain().collect::<Vec<_>>()
    };
    entries.len()
}

#[cfg(test)]
pub(crate) fn remove_server_for_test(id: u64) {
    if let Some(entry) = registry()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .servers
        .remove(&id)
    {
        drop(entry);
    }
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn clear_registry_for_test() {
    let mut guard = registry()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    guard.servers.clear();
    guard.next_id = 0;
}
