//! MATLAB-compatible `accept` builtin for RunMat networking.

use once_cell::sync::OnceCell;
use runmat_builtins::{IntValue, StructValue, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::gather_if_needed;
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};

use super::tcpserver::{default_user_data, server_handle, TcpServerState, HANDLE_ID_FIELD};

use std::collections::HashMap;
use std::io::{self, ErrorKind};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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

fn client_registry() -> &'static Mutex<TcpClientRegistry> {
    TCP_CLIENT_REGISTRY.get_or_init(|| Mutex::new(TcpClientRegistry::default()))
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

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "accept"
category: "io/net"
keywords: ["accept", "tcpserver", "tcpclient", "socket", "networking"]
summary: "Accept a pending client connection on a TCP server."
references:
  - https://www.mathworks.com/help/matlab/ref/accept.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Networking runs on the host CPU. GPU-resident metadata is gathered automatically before accepting connections."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::net::accept::tests"
  integration: "builtins::io::net::accept::tests::accept_establishes_client_connection"
---

# What does `accept` do in MATLAB / RunMat?
`accept(server)` waits for a pending TCP connection on the socket created by `tcpserver`. When a client connects, the builtin returns a MATLAB-compatible struct that mirrors the `tcpclient` object. The struct tracks connection metadata (remote address, port, byte order, and timeout settings) and holds an opaque identifier that other RunMat networking builtins use to operate on the live socket.

## How does `accept` behave in MATLAB / RunMat?
- The first argument must be the struct returned from `tcpserver`. RunMat validates that the struct contains a `__tcpserver_id` field and raises `MATLAB:accept:InvalidTcpServer` when the identifier is missing or no longer points to an active listener.
- By default, `accept` uses the timeout configured when the server was created (`Timeout` name-value pair on `tcpserver`). You can override it per call with `accept(server, "Timeout", seconds)`. The timeout must be non-negative; the builtin raises `MATLAB:accept:InvalidNameValue` when the value is NaN, negative, or non-scalar.
- Successful calls return immediately with a struct whose fields mirror MATLAB’s `tcpclient` properties (`Address`, `Port`, `NumBytesAvailable`, `BytesAvailableFcn`, `ByteOrder`, `Timeout`, `UserData`, and `Connected`). The struct also contains hidden fields `__tcpserver_id` and `__tcpclient_id` so higher-level builtins can operate on the live socket.
- If no client connects before the timeout expires, the builtin raises `MATLAB:accept:Timeout`.
- When the underlying OS reports an accept failure (for example, because the socket closed), the builtin raises `MATLAB:accept:AcceptFailed` with the platform error message.
- Networking occurs on the host CPU. If the server struct or timeout value lives on the GPU, RunMat gathers it to the host automatically before waiting for connections.

## `accept` Function GPU Execution Behaviour
`accept` does not involve the GPU. Any inputs that originate on the GPU are gathered before validation to make sure socket operations run on the host. The returned struct is always CPU-resident. No acceleration-provider hooks are required for this builtin, and future GPU-aware networking features will continue to gather metadata automatically while keeping sockets on the host.

## Examples of using the `accept` function in MATLAB / RunMat

### Accepting a localhost client connection
```matlab
srv = tcpserver("127.0.0.1", 0);
port = srv.ServerPort;
% In another MATLAB/RunMat session (or any TCP client script), connect to the port above.
client = accept(srv);
disp(client.Address)
disp(client.Port)
```

Expected output:
```matlab
127.0.0.1
55000    % varies per run
```

### Overriding the timeout for a single accept call
```matlab
srv = tcpserver("0.0.0.0", 40000, "Timeout", 10);
try
    client = accept(srv, "Timeout", 0.25);
catch err
    disp(err.identifier)
end
```

Expected output:
```matlab
MATLAB:accept:Timeout
```

### Inspecting connection metadata after accepting a client
```matlab
srv = tcpserver("::1", 45000);
client = accept(srv);
fprintf("Remote peer %s:%d\\n", client.Address, client.Port);
fprintf("Byte order: %s\\n", client.ByteOrder);
```

Expected output:
```matlab
Remote peer ::1:51432
Byte order: little-endian
```

### Handling multiple queued clients sequentially
```matlab
srv = tcpserver("127.0.0.1", 47000);
% Connect two clients (for example, two tcpclient calls) before running accept twice.
client1 = accept(srv);
client2 = accept(srv);
fprintf("First connection from %s\\n", client1.Address);
fprintf("Second connection from %s\\n", client2.Address);
```

Expected output:
```matlab
First connection from 127.0.0.1
Second connection from 127.0.0.1
```

### Using the returned identifier with other networking builtins
```matlab
srv = tcpserver("127.0.0.1", 52000);
% Connect a client from another process, then accept it.
client = accept(srv);
clientId = client.__tcpclient_id;
fprintf("Opaque client identifier: %d\\n", clientId);
```

Expected output:
```matlab
Opaque client identifier: 42   % identifier value varies per run
```

## GPU residency in RunMat (Do I need `gpuArray`?)
No. TCP sockets run on the host, and `accept` gathers any GPU-resident scalars or structs before waiting for a connection. Keeping metadata on the GPU offers no benefit, and the builtin always returns CPU-resident structs with identifiers that reference host networking resources.

## FAQ
### What happens if the server struct is invalid or already closed?
RunMat reports `MATLAB:accept:InvalidTcpServer`. Ensure you pass the struct returned by `tcpserver` and that the server is still active.

### Can I accept multiple clients with the same server?
Yes. Call `accept` repeatedly; each successful call registers a new client and returns its own struct while keeping the listener active.

### How do I change the timeout globally?
Configure it when creating the server (`tcpserver(..., "Timeout", seconds)`). You can override it per call with `accept(srv, "Timeout", value)` if needed.

### Does RunMat support IPv6 clients?
Yes. The builtin accepts IPv4 or IPv6 clients and records their string representation in the returned struct’s `Address` field.

### Is there a queue limit for pending connections?
The OS backlog applies. If the queue is full, new clients may be refused before `accept` sees them. Increase the backlog with OS-level tuning if needed.

### Can I use `accept` in parallel workers?
Yes. The listener must exist in the worker where you call `accept`, just like MATLAB. Future high-level helpers will coordinate cross-worker sharing.

### How do I close the accepted client?
Use forthcoming networking close helpers (or invoke platform APIs directly). Dropping the struct does not close the socket automatically; RunMat networking builtins reference the opaque identifier.

### What does `NumBytesAvailable` represent?
It mirrors MATLAB: the number of bytes buffered and ready to read. Initially zero; reading functions update it as data arrives.

### Does the builtin support TLS?
Not yet. TLS will be layered on top of the same client identifier once RunMat’s TLS provider lands.

## See also
[tcpserver](./tcpserver)

## Source & Feedback
- Source: `crates/runmat-runtime/src/builtins/io/net/accept.rs`
- Bugs & feature requests: https://github.com/runmat-org/runmat/issues/new/choose
"#;

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

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "accept",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Networking builtin executed eagerly on the CPU.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("accept", DOC_MD);

#[runtime_builtin(
    name = "accept",
    category = "io/net",
    summary = "Accept a pending client connection on a TCP server.",
    keywords = "accept,tcpserver,tcpclient"
)]
pub(crate) fn accept_builtin(server: Value, rest: Vec<Value>) -> Result<Value, String> {
    let server = gather_if_needed(&server)?;
    let server_id = extract_server_id(&server)?;

    let options = parse_accept_options(rest)?;

    let shared_server = server_handle(server_id).ok_or_else(|| {
        runtime_error(
            MESSAGE_ID_INVALID_SERVER,
            "accept: tcpserver handle is no longer valid".to_string(),
        )
    })?;

    let server_guard = shared_server
        .lock()
        .map_err(|_| runtime_error(MESSAGE_ID_INTERNAL, "accept: server lock poisoned".into()))?;

    let timeout = options.timeout.unwrap_or(server_guard.timeout);
    validate_timeout(timeout)?;

    match accept_with_timeout(&server_guard.listener, timeout) {
        Ok((stream, peer_addr)) => {
            if let Err(err) = configure_stream(&stream, timeout) {
                drop(server_guard);
                return Err(runtime_error(
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
            Err(match err.kind() {
                ErrorKind::WouldBlock => runtime_error(
                    MESSAGE_ID_TIMEOUT,
                    format!(
                        "accept: timed out waiting for a client connection after {:.3} seconds",
                        timeout
                    ),
                ),
                _ => runtime_error(
                    MESSAGE_ID_ACCEPT_FAILED,
                    format!("accept: failed to accept client ({err})"),
                ),
            })
        }
    }
}

fn extract_server_id(value: &Value) -> Result<u64, String> {
    match value {
        Value::Struct(struct_value) => {
            let id_value = struct_value.fields.get(HANDLE_ID_FIELD).ok_or_else(|| {
                runtime_error(
                    MESSAGE_ID_INVALID_SERVER,
                    "accept: tcpserver struct missing internal identifier".to_string(),
                )
            })?;
            let id = match id_value {
                Value::Int(IntValue::U64(id)) => *id,
                Value::Int(iv) => iv.to_i64() as u64,
                other => {
                    return Err(runtime_error(
                        MESSAGE_ID_INVALID_SERVER,
                        format!("accept: expected numeric tcpserver identifier, got {other:?}"),
                    ))
                }
            };
            Ok(id)
        }
        _ => Err(runtime_error(
            MESSAGE_ID_INVALID_SERVER,
            "accept: first argument must be the struct returned by tcpserver".to_string(),
        )),
    }
}

#[derive(Default)]
struct AcceptOptions {
    timeout: Option<f64>,
}

fn parse_accept_options(rest: Vec<Value>) -> Result<AcceptOptions, String> {
    if rest.is_empty() {
        return Ok(AcceptOptions::default());
    }
    if rest.len() % 2 != 0 {
        return Err(runtime_error(
            MESSAGE_ID_INVALID_NAME_VALUE,
            "accept: name-value arguments must appear in pairs".to_string(),
        ));
    }

    let mut options = AcceptOptions::default();
    let mut iter = rest.into_iter();
    while let Some(name_raw) = iter.next() {
        let value_raw = iter
            .next()
            .expect("paired iteration guarantees value exists");
        let name_value = gather_if_needed(&name_raw)?;
        let name = match name_value {
            Value::String(ref s) => s.clone(),
            Value::CharArray(ref ca) if ca.rows == 1 => ca.data.iter().collect(),
            Value::StringArray(ref sa) if sa.data.len() == 1 => sa.data[0].clone(),
            other => {
                return Err(runtime_error(
                    MESSAGE_ID_INVALID_NAME_VALUE,
                    format!("accept: invalid option name ({other:?})"),
                ))
            }
        };
        let lower = name.to_ascii_lowercase();
        match lower.as_str() {
            "timeout" => {
                let gathered = gather_if_needed(&value_raw)?;
                let timeout = parse_timeout_value(&gathered).map_err(|msg| {
                    runtime_error(
                        MESSAGE_ID_INVALID_NAME_VALUE,
                        format!("accept: invalid Timeout value: {msg}"),
                    )
                })?;
                options.timeout = Some(timeout);
            }
            _ => {
                return Err(runtime_error(
                    MESSAGE_ID_INVALID_NAME_VALUE,
                    format!("accept: unsupported option '{name}'"),
                ))
            }
        }
    }
    Ok(options)
}

pub(crate) fn parse_timeout_value(value: &Value) -> Result<f64, String> {
    let timeout = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        Value::Tensor(t) if t.data.len() == 1 => t.data[0],
        Value::Tensor(_) => {
            return Err("Timeout must be a scalar".to_string());
        }
        _ => return Err("Timeout must be numeric".to_string()),
    };
    if !timeout.is_finite() && !timeout.is_infinite() {
        return Err("Timeout must be finite or Inf".to_string());
    }
    if timeout.is_sign_negative() {
        return Err("Timeout must be non-negative".to_string());
    }
    Ok(timeout)
}

fn validate_timeout(timeout: f64) -> Result<(), String> {
    if timeout.is_nan() {
        return Err(runtime_error(
            MESSAGE_ID_INVALID_NAME_VALUE,
            "accept: Timeout must not be NaN".to_string(),
        ));
    }
    if timeout.is_sign_negative() {
        return Err(runtime_error(
            MESSAGE_ID_INVALID_NAME_VALUE,
            "accept: Timeout must be non-negative".to_string(),
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

fn runtime_error(message_id: &'static str, message: String) -> String {
    format!("{message_id}: {message}")
}

#[cfg(test)]
mod tests {
    use super::super::tcpserver::{
        remove_server_for_test, tcpserver_builtin, HANDLE_ID_FIELD as SERVER_FIELD,
    };
    use super::*;
    #[cfg(feature = "doc_export")]
    use crate::builtins::common::test_support;
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

    fn server_id(value: &Value) -> u64 {
        match struct_field(value, SERVER_FIELD) {
            Value::Int(IntValue::U64(id)) => *id,
            Value::Int(iv) => iv.to_i64() as u64,
            other => panic!("expected server id int, got {other:?}"),
        }
    }

    #[test]
    fn accept_rejects_non_struct() {
        let err = accept_builtin(Value::Num(1.0), Vec::new()).unwrap_err();
        assert!(err.starts_with(MESSAGE_ID_INVALID_SERVER));
    }

    #[test]
    fn accept_establishes_client_connection() {
        let server_value = tcpserver_builtin(
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

        let client = accept_builtin(server_value.clone(), Vec::new()).expect("accept");
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

    #[test]
    fn accept_times_out_when_no_client_connects() {
        let server_value = tcpserver_builtin(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(0)),
            Vec::new(),
        )
        .expect("tcpserver");
        let err = accept_builtin(
            server_value.clone(),
            vec![Value::from("Timeout"), Value::Num(0.05)],
        )
        .unwrap_err();
        assert!(err.starts_with(MESSAGE_ID_TIMEOUT));
        remove_server_for_test(server_id(&server_value));
    }

    #[test]
    fn accept_rejects_invalid_timeout_name_value() {
        let server_value = tcpserver_builtin(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(0)),
            Vec::new(),
        )
        .expect("tcpserver");
        let err = accept_builtin(
            server_value.clone(),
            vec![Value::from("Timeout"), Value::Num(-1.0)],
        )
        .unwrap_err();
        assert!(err.starts_with(MESSAGE_ID_INVALID_NAME_VALUE));
        remove_server_for_test(server_id(&server_value));
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn accept_respects_per_call_timeout_override() {
        let server_value = tcpserver_builtin(
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

        let client = accept_builtin(
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
