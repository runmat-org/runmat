//! Shared live TCP client state used by constructors and I/O operations.

use once_cell::sync::OnceCell;
use std::collections::HashMap;
use std::net::{Shutdown, SocketAddr, TcpStream};
use std::sync::{Arc, Mutex};

pub(crate) const CLIENT_HANDLE_FIELD: &str = "__tcpclient_id";

pub(crate) type SharedTcpClient = Arc<Mutex<TcpClientState>>;

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct TcpClientState {
    pub(crate) id: u64,
    pub(crate) server_id: u64,
    pub(crate) stream: TcpStream,
    pub(crate) peer_addr: SocketAddr,
    pub(crate) timeout: f64,
    pub(crate) byte_order: String,
    pub(crate) connected: bool,
    pub(crate) readline_buffer: Vec<u8>,
}

#[derive(Default)]
struct TcpClientRegistry {
    next_id: u64,
    clients: HashMap<u64, SharedTcpClient>,
}

static TCP_CLIENT_REGISTRY: OnceCell<Mutex<TcpClientRegistry>> = OnceCell::new();

#[cfg(test)]
static TCP_CLIENT_TEST_GUARD: OnceCell<Mutex<()>> = OnceCell::new();

fn client_registry() -> &'static Mutex<TcpClientRegistry> {
    TCP_CLIENT_REGISTRY.get_or_init(|| Mutex::new(TcpClientRegistry::default()))
}

#[cfg(test)]
pub(crate) fn test_guard() -> std::sync::MutexGuard<'static, ()> {
    TCP_CLIENT_TEST_GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}

pub(crate) fn insert_client(
    stream: TcpStream,
    server_id: u64,
    peer_addr: SocketAddr,
    timeout: f64,
    byte_order: String,
) -> u64 {
    let mut guard = client_registry()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    guard.next_id = guard.next_id.wrapping_add(1);
    let id = guard.next_id;
    let state = TcpClientState {
        id,
        server_id,
        stream,
        peer_addr,
        timeout,
        byte_order,
        connected: true,
        readline_buffer: Vec::new(),
    };
    guard.clients.insert(id, Arc::new(Mutex::new(state)));
    id
}

#[allow(dead_code)]
pub(crate) fn client_handle(id: u64) -> Option<SharedTcpClient> {
    client_registry()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .clients
        .get(&id)
        .cloned()
}

pub(crate) fn close_client(id: u64) -> bool {
    let entry = {
        let mut guard = client_registry()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        guard.clients.remove(&id)
    };
    if let Some(client) = entry {
        close_client_state(&client);
        true
    } else {
        false
    }
}

pub(crate) fn close_clients_for_server(server_id: u64) -> usize {
    let mut guard = client_registry()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut to_close: Vec<(u64, SharedTcpClient)> = Vec::new();
    for (id, client) in guard.clients.iter() {
        if let Ok(state) = client.lock() {
            if state.server_id == server_id {
                to_close.push((*id, client.clone()));
            }
        }
    }
    for (id, _) in &to_close {
        guard.clients.remove(id);
    }
    drop(guard);
    for (_, client) in &to_close {
        close_client_state(client);
    }
    to_close.len()
}

pub(crate) fn close_all_clients() -> usize {
    let entries = {
        let mut guard = client_registry()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        guard.clients.drain().collect::<Vec<_>>()
    };
    for (_, client) in &entries {
        close_client_state(client);
    }
    entries.len()
}

fn close_client_state(client: &SharedTcpClient) {
    if let Ok(mut state) = client.lock() {
        if state.connected {
            let _ = state.stream.shutdown(Shutdown::Both);
            state.connected = false;
        }
    }
}

#[cfg(test)]
pub(crate) fn remove_client_for_test(id: u64) {
    if let Some(entry) = client_registry()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .clients
        .remove(&id)
    {
        drop(entry);
    }
}
