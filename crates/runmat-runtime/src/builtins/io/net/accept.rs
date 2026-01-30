//! MATLAB-compatible `accept` builtin for RunMat networking.

use once_cell::sync::OnceCell;
use runmat_builtins::{IntValue, StructValue, Value};
use runmat_macros::runtime_builtin;

use super::tcpserver::{default_user_data, server_handle, TcpServerState, HANDLE_ID_FIELD};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};
use thiserror::Error;

use runmat_time::Instant;
use std::collections::HashMap;
use std::io::{self, ErrorKind};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const MESSAGE_ID_INVALID_SERVER: &str = "MATLAB:accept:InvalidTcpServer";
const MESSAGE_ID_TIMEOUT: &str = "MATLAB:accept:Timeout";
const MESSAGE_ID_INVALID_NAME_VALUE: &str = "MATLAB:accept:InvalidNameValue";
const MESSAGE_ID_INTERNAL: &str = "MATLAB:accept:InternalError";
const MESSAGE_ID_ACCEPT_FAILED: &str = "MATLAB:accept:AcceptFailed";

pub(crate) const CLIENT_HANDLE_FIELD: &str = "__tcpclient_id";

type SharedTcpClient = Arc<Mutex<TcpClientState>>;

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
        .unwrap()
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
    let shared = Arc::new(Mutex::new(state));
    guard.clients.insert(id, shared);
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
pub(super) fn remove_client_for_test(id: u64) {
    if let Some(entry) = client_registry()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .clients
        .remove(&id)
    {
        drop(entry);
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::net::accept")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "accept",
    op_kind: GpuOpKind::Custom("network"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host-only networking builtin; GPU inputs are gathered to CPU before accepting clients.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::net::accept")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "accept",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Networking builtin executed eagerly on the CPU.",
};

#[runtime_builtin(
    name = "accept",
    category = "io/net",
    summary = "Accept a pending client connection on a TCP server.",
    keywords = "accept,tcpserver,tcpclient",
    builtin_path = "crate::builtins::io::net::accept"
)]
pub(crate) async fn accept_builtin(server: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let server = gather_if_needed_async(&server).await?;
    let server_id = extract_server_id(&server)?;

    let options = parse_accept_options(rest).await?;

    let shared_server = server_handle(server_id).ok_or_else(|| {
        accept_error(
            MESSAGE_ID_INVALID_SERVER,
            "accept: tcpserver handle is no longer valid",
        )
    })?;

    let server_guard = shared_server
        .lock()
        .map_err(|_| accept_error(MESSAGE_ID_INTERNAL, "accept: server lock poisoned"))?;

    let timeout = options.timeout.unwrap_or(server_guard.timeout);
    validate_timeout(timeout)?;

    match accept_with_timeout(&server_guard.listener, timeout) {
        Ok((stream, peer_addr)) => {
            if let Err(err) = configure_stream(&stream, timeout) {
                drop(server_guard);
                return Err(accept_error(
                    MESSAGE_ID_INTERNAL,
                    format!("accept: failed to configure stream timeouts ({err})"),
                ));
            }
            let byte_order = server_guard.byte_order.clone();
            let client_id = insert_client(
                stream,
                server_guard.id,
                peer_addr,
                timeout,
                byte_order.clone(),
            );
            let client_value =
                build_tcpclient_value(client_id, &server_guard, peer_addr, timeout, byte_order);
            drop(server_guard);
            Ok(client_value)
        }
        Err(err) => {
            drop(server_guard);
            let message = match err.kind() {
                ErrorKind::WouldBlock => accept_error(
                    MESSAGE_ID_TIMEOUT,
                    format!(
                        "accept: timed out waiting for a client connection after {:.3} seconds",
                        timeout
                    ),
                ),
                _ => accept_error(
                    MESSAGE_ID_ACCEPT_FAILED,
                    format!("accept: failed to accept client ({err})"),
                ),
            };
            Err(message)
        }
    }
}

fn extract_server_id(value: &Value) -> BuiltinResult<u64> {
    match value {
        Value::Struct(struct_value) => {
            let id_value = struct_value.fields.get(HANDLE_ID_FIELD).ok_or_else(|| {
                accept_error(
                    MESSAGE_ID_INVALID_SERVER,
                    "accept: tcpserver struct missing internal identifier",
                )
            })?;
            let id = match id_value {
                Value::Int(IntValue::U64(id)) => *id,
                Value::Int(iv) => iv.to_i64() as u64,
                other => {
                    return Err(accept_error(
                        MESSAGE_ID_INVALID_SERVER,
                        format!("accept: expected numeric tcpserver identifier, got {other:?}"),
                    ));
                }
            };
            Ok(id)
        }
        _ => Err(accept_error(
            MESSAGE_ID_INVALID_SERVER,
            "accept: first argument must be the struct returned by tcpserver",
        )),
    }
}

#[derive(Default)]
struct AcceptOptions {
    timeout: Option<f64>,
}

async fn parse_accept_options(rest: Vec<Value>) -> BuiltinResult<AcceptOptions> {
    if rest.is_empty() {
        return Ok(AcceptOptions::default());
    }
    if !rest.len().is_multiple_of(2) {
        return Err(accept_error(
            MESSAGE_ID_INVALID_NAME_VALUE,
            "accept: name-value arguments must appear in pairs",
        ));
    }

    let mut options = AcceptOptions::default();
    let mut iter = rest.into_iter();
    while let Some(name_raw) = iter.next() {
        let value_raw = iter
            .next()
            .expect("paired iteration guarantees value exists");
        let name_value = gather_if_needed_async(&name_raw).await?;
        let name = match name_value {
            Value::String(ref s) => s.clone(),
            Value::CharArray(ref ca) if ca.rows == 1 => ca.data.iter().collect(),
            Value::StringArray(ref sa) if sa.data.len() == 1 => sa.data[0].clone(),
            other => {
                return Err(accept_error(
                    MESSAGE_ID_INVALID_NAME_VALUE,
                    format!("accept: invalid option name ({other:?})"),
                ));
            }
        };
        let lower = name.to_ascii_lowercase();
        match lower.as_str() {
            "timeout" => {
                let gathered = gather_if_needed_async(&value_raw).await?;
                let timeout = parse_timeout_value(&gathered).map_err(|msg| {
                    accept_error(
                        MESSAGE_ID_INVALID_NAME_VALUE,
                        format!("accept: invalid Timeout value: {msg}"),
                    )
                })?;
                options.timeout = Some(timeout);
            }
            _ => {
                return Err(accept_error(
                    MESSAGE_ID_INVALID_NAME_VALUE,
                    format!("accept: unsupported option '{name}'"),
                ));
            }
        }
    }
    Ok(options)
}

#[derive(Debug, Error)]
pub(crate) enum TimeoutParseError {
    #[error("Timeout must be a scalar")]
    NonScalar,
    #[error("Timeout must be numeric")]
    NonNumeric,
    #[error("Timeout must be finite or Inf")]
    NonFinite,
    #[error("Timeout must be non-negative")]
    Negative,
}

pub(crate) fn parse_timeout_value(value: &Value) -> Result<f64, TimeoutParseError> {
    let timeout = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        Value::Tensor(t) if t.data.len() == 1 => t.data[0],
        Value::Tensor(_) => {
            return Err(TimeoutParseError::NonScalar);
        }
        _ => return Err(TimeoutParseError::NonNumeric),
    };
    if !timeout.is_finite() && !timeout.is_infinite() {
        return Err(TimeoutParseError::NonFinite);
    }
    if timeout.is_sign_negative() {
        return Err(TimeoutParseError::Negative);
    }
    Ok(timeout)
}

fn validate_timeout(timeout: f64) -> BuiltinResult<()> {
    if timeout.is_nan() {
        return Err(accept_error(
            MESSAGE_ID_INVALID_NAME_VALUE,
            "accept: Timeout must not be NaN",
        ));
    }
    if timeout.is_sign_negative() {
        return Err(accept_error(
            MESSAGE_ID_INVALID_NAME_VALUE,
            "accept: Timeout must be non-negative",
        ));
    }
    Ok(())
}

fn accept_with_timeout(
    listener: &TcpListener,
    timeout: f64,
) -> io::Result<(TcpStream, SocketAddr)> {
    if timeout.is_infinite() {
        return listener.accept();
    }
    listener.set_nonblocking(true)?;
    let start = Instant::now();
    let deadline = Duration::from_secs_f64(timeout);
    loop {
        match listener.accept() {
            Ok((stream, addr)) => {
                let _ = listener.set_nonblocking(false);
                return Ok((stream, addr));
            }
            Err(err) if err.kind() == ErrorKind::WouldBlock => {
                if start.elapsed() >= deadline {
                    let _ = listener.set_nonblocking(false);
                    return Err(io::Error::new(ErrorKind::WouldBlock, "accept timeout"));
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(err) => {
                let _ = listener.set_nonblocking(false);
                return Err(err);
            }
        }
    }
}

pub(crate) fn configure_stream(stream: &TcpStream, timeout: f64) -> io::Result<()> {
    let opt = if timeout.is_infinite() || timeout == 0.0 {
        None
    } else {
        Some(Duration::from_secs_f64(timeout))
    };
    stream.set_read_timeout(opt)?;
    stream.set_write_timeout(opt)?;
    Ok(())
}

fn build_tcpclient_value(
    client_id: u64,
    server_state: &TcpServerState,
    peer_addr: SocketAddr,
    timeout: f64,
    byte_order: String,
) -> Value {
    let mut st = StructValue::new();
    st.fields
        .insert("Type".to_string(), Value::String("tcpclient".to_string()));
    st.fields.insert(
        "Address".to_string(),
        Value::String(peer_addr.ip().to_string()),
    );
    st.fields.insert(
        "Port".to_string(),
        Value::Int(IntValue::U16(peer_addr.port())),
    );
    st.fields.insert(
        "ServerAddress".to_string(),
        Value::String(server_state.local_addr.ip().to_string()),
    );
    st.fields.insert(
        "ServerPort".to_string(),
        Value::Int(IntValue::U16(server_state.local_addr.port())),
    );
    st.fields.insert("Connected".to_string(), Value::Bool(true));
    st.fields
        .insert("Status".to_string(), Value::String("connected".to_string()));
    st.fields.insert(
        "NumBytesAvailable".to_string(),
        Value::Int(IntValue::I32(0)),
    );
    st.fields
        .insert("BytesAvailableFcn".to_string(), default_user_data());
    st.fields.insert(
        "BytesAvailableFcnMode".to_string(),
        Value::String("byte".to_string()),
    );
    st.fields.insert(
        "BytesAvailableFcnCount".to_string(),
        Value::Int(IntValue::I32(1)),
    );
    st.fields
        .insert("ByteOrder".to_string(), Value::String(byte_order));
    st.fields.insert(
        "Timeout".to_string(),
        Value::Num(if timeout.is_infinite() {
            f64::INFINITY
        } else {
            timeout
        }),
    );
    st.fields
        .insert("UserData".to_string(), default_user_data());
    st.fields.insert(
        CLIENT_HANDLE_FIELD.to_string(),
        Value::Int(IntValue::U64(client_id)),
    );
    st.fields.insert(
        HANDLE_ID_FIELD.to_string(),
        Value::Int(IntValue::U64(server_state.id)),
    );
    Value::Struct(st)
}

fn accept_error(message_id: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier(message_id)
        .with_builtin("accept")
        .build()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::tcpserver::{
        remove_server_for_test, tcpserver_builtin, HANDLE_ID_FIELD as SERVER_FIELD,
    };
    use super::*;
    use runmat_builtins::Value;
    use std::net::TcpStream;
    use std::thread;
    use std::time::Duration;

    fn struct_field<'a>(value: &'a Value, name: &str) -> &'a Value {
        match value {
            Value::Struct(st) => st
                .fields
                .get(name)
                .unwrap_or_else(|| panic!("missing field {name}")),
            _ => panic!("expected struct"),
        }
    }

    fn client_id(value: &Value) -> u64 {
        match struct_field(value, CLIENT_HANDLE_FIELD) {
            Value::Int(IntValue::U64(id)) => *id,
            Value::Int(iv) => iv.to_i64() as u64,
            other => panic!("expected id int, got {other:?}"),
        }
    }

    fn assert_error_identifier(err: RuntimeError, expected: &str) {
        assert_eq!(err.identifier(), Some(expected));
    }

    fn run_accept(server: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(accept_builtin(server, rest))
    }

    fn run_tcpserver(address: Value, port: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(tcpserver_builtin(address, port, rest))
    }

    fn server_id(value: &Value) -> u64 {
        match struct_field(value, SERVER_FIELD) {
            Value::Int(IntValue::U64(id)) => *id,
            Value::Int(iv) => iv.to_i64() as u64,
            other => panic!("expected server id int, got {other:?}"),
        }
    }

    fn net_guard() -> std::sync::MutexGuard<'static, ()> {
        super::test_guard()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn accept_rejects_non_struct() {
        let _guard = net_guard();
        let err = run_accept(Value::Num(1.0), Vec::new()).unwrap_err();
        assert_error_identifier(err, MESSAGE_ID_INVALID_SERVER);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn accept_establishes_client_connection() {
        let _guard = net_guard();
        let server_value = run_tcpserver(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(0)),
            Vec::new(),
        )
        .expect("tcpserver");
        let port = match struct_field(&server_value, "ServerPort") {
            Value::Int(iv) => iv.to_i64() as u16,
            other => panic!("expected ServerPort int, got {other:?}"),
        };

        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            TcpStream::connect(("127.0.0.1", port)).expect("connect")
        });

        let client = run_accept(server_value.clone(), Vec::new()).expect("accept");
        let stream = handle.join().expect("client thread");
        drop(stream);

        match struct_field(&client, "Connected") {
            Value::Bool(flag) => assert!(*flag),
            other => panic!("expected Connected bool, got {other:?}"),
        }
        match struct_field(&client, "Address") {
            Value::String(addr) => assert_eq!(addr, "127.0.0.1"),
            other => panic!("expected Address string, got {other:?}"),
        }
        match struct_field(&client, "Timeout") {
            Value::Num(n) => assert_eq!(*n, 10.0),
            other => panic!("expected Timeout numeric, got {other:?}"),
        }

        let cid = client_id(&client);
        remove_client_for_test(cid);
        remove_server_for_test(server_id(&server_value));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn accept_times_out_when_no_client_connects() {
        let _guard = net_guard();
        let server_value = run_tcpserver(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(0)),
            Vec::new(),
        )
        .expect("tcpserver");
        let err = run_accept(
            server_value.clone(),
            vec![Value::from("Timeout"), Value::Num(0.05)],
        )
        .unwrap_err();
        assert_error_identifier(err, MESSAGE_ID_TIMEOUT);
        remove_server_for_test(server_id(&server_value));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn accept_rejects_invalid_timeout_name_value() {
        let _guard = net_guard();
        let server_value = run_tcpserver(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(0)),
            Vec::new(),
        )
        .expect("tcpserver");
        let err = run_accept(
            server_value.clone(),
            vec![Value::from("Timeout"), Value::Num(-1.0)],
        )
        .unwrap_err();
        assert_error_identifier(err, MESSAGE_ID_INVALID_NAME_VALUE);
        remove_server_for_test(server_id(&server_value));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn accept_respects_per_call_timeout_override() {
        let _guard = net_guard();
        let server_value = run_tcpserver(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(0)),
            Vec::new(),
        )
        .expect("tcpserver");
        let port = match struct_field(&server_value, "ServerPort") {
            Value::Int(iv) => iv.to_i64() as u16,
            other => panic!("expected ServerPort int, got {other:?}"),
        };

        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            TcpStream::connect(("127.0.0.1", port)).expect("connect")
        });

        let client = run_accept(
            server_value.clone(),
            vec![Value::from("Timeout"), Value::Num(1.0)],
        )
        .expect("accept");
        handle.join().expect("join");
        let timeout_val = match struct_field(&client, "Timeout") {
            Value::Num(n) => *n,
            other => panic!("expected Timeout numeric, got {other:?}"),
        };
        assert_eq!(timeout_val, 1.0);
        let cid = client_id(&client);
        remove_client_for_test(cid);
        remove_server_for_test(server_id(&server_value));
    }
}
