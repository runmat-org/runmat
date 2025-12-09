//! MATLAB-compatible `close` builtin for networking resources in RunMat.

use runmat_builtins::{StructValue, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::gather_if_needed;

use super::accept::{
    close_all_clients, close_client, close_clients_for_server, CLIENT_HANDLE_FIELD,
};
use super::tcpserver::{close_all_servers, close_server, HANDLE_ID_FIELD};

const MESSAGE_ID_INVALID_ARGUMENT: &str = "MATLAB:close:InvalidArgument";
const MESSAGE_ID_INVALID_HANDLE: &str = "MATLAB:close:InvalidHandle";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(name = "close")
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "close"
category: "io/net"
keywords: ["close", "tcpclient", "tcpserver", "socket", "networking"]
summary: "Close TCP clients or servers that were created by tcpclient, tcpserver, or accept."
references:
  - https://www.mathworks.com/help/matlab/ref/tcpclient.html
  - https://www.mathworks.com/help/matlab/ref/tcpserver.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Networking runs on the host CPU. GPU-resident structs are gathered automatically before the close operation executes."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 4
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::net::close::tests"
  integration: "builtins::io::net::close::tests::close_tcpclient_releases_handle"
---

# What does the `close` function do in MATLAB / RunMat?
`close` releases TCP clients and servers that were created with `tcpclient`, `tcpserver`, or
`accept`. It mirrors MATLAB semantics for network connections while adopting the familiar
`status = close(obj)` pattern used elsewhere in MATLAB: the function returns `1` when one or more
handles are closed and `0` when nothing needed to be released.

## How does the `close` function behave in MATLAB / RunMat?
- `close(t)` closes the TCP client struct `t` that was previously returned by `tcpclient` or
  `accept`. Subsequent socket operations on that client raise MATLAB-style “not connected”
  diagnostics.
- `close(s)` closes the TCP server struct `s` that was returned by `tcpserver`. Any clients that were
  accepted from that server are also disconnected to match MATLAB’s lifecycle rules.
- `close('clients')` closes every registered TCP client (including those produced by `accept`).
  Likewise, `close('servers')` closes every TCP server, and `close('all')` closes both clients and
  servers.
- Multiple inputs are processed from left to right. The return value is `1` when at least one handle
  was closed and `0` otherwise.
- Invalid arguments raise `MATLAB:close:InvalidArgument`. Structs that do not contain the hidden
  RunMat networking identifiers raise `MATLAB:close:InvalidHandle`.
- Networking happens on the CPU. If the arguments live on the GPU, RunMat gathers them
  automatically before touching the host-side registries.
- Arguments wrapped in cell arrays (possibly nested) or scalar string arrays are supported. `close`
  walks each element in-order and applies the usual rules to every value it discovers.

## Examples of using the `close` function in MATLAB / RunMat

### Close a TCP client after finishing I/O
```matlab
client = tcpclient("127.0.0.1", 50000);
status = close(client);
```
Expected output:
```matlab
status = 1
```

### Close a TCP server and any accepted clients
```matlab
srv = tcpserver("127.0.0.1", 0);
client = accept(srv);             % accept a pending connection
status = close(srv);              % closes the server and the accepted client
```
Expected output:
```matlab
status = 1
```

### Close every open TCP client
```matlab
tcpclient("localhost", 40000);
tcpclient("localhost", 40001);
status = close("clients");
```
Expected output:
```matlab
status = 1
```

### Close all networking resources at once
```matlab
client = tcpclient("localhost", 40000);
srv = tcpserver("localhost", 0);
status = close("all");
```
Expected output:
```matlab
status = 1
```

### Calling close on an already-closed client
```matlab
client = tcpclient("127.0.0.1", 50000);
close(client);
status = close(client);   % nothing left to close
```
Expected output:
```matlab
status = 0
```

### Close handles stored in a cell array
```matlab
client = tcpclient("localhost", 40000);
srv = tcpserver("localhost", 0);
handles = {client, srv};
status = close(handles);
```
Expected output:
```matlab
status = 1
```

## `close` Function GPU Execution Behaviour
`close` performs only CPU-side bookkeeping. When network structs or option strings reside on the
GPU, RunMat gathers them to host memory before inspecting their hidden identifiers. No provider
hooks participate, and GPU residency is irrelevant to the close operation.

## FAQ

### What does the return value represent?
The return value is `1` when at least one client or server was closed and `0` otherwise. Use it to
detect whether `close` released any networking resources.

### Can I pass multiple clients or servers at once?
Yes. Pass them as separate arguments (`close(client1, client2)`) or wrap them in a cell array. The
function closes each handle in order and returns `1` when any of the handles required closing.

### Does closing a server also close accepted clients?
Yes. RunMat mirrors MATLAB by closing every client that originated from the server before releasing
the listener itself.

### What happens if I pass a non-network struct?
The builtin raises `MATLAB:close:InvalidHandle`. Only structs produced by the RunMat networking
builtins (`tcpclient`, `tcpserver`, `accept`) carry the hidden identifiers that `close` recognises.

### Do GPU arrays require special handling?
No. RunMat gathers GPU-resident values automatically before inspecting them, and networking always
runs on the CPU.

### Is it safe to call close twice?
Yes. The second call simply returns `0`, indicating that nothing needed to be closed.

## See Also
[tcpclient](./tcpclient), [tcpserver](./tcpserver), [accept](./accept), [read](./read), [write](./write)

## Source & Feedback
- Implementation: `crates/runmat-runtime/src/builtins/io/net/close.rs`
- Found an issue? [Open a ticket](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "close",
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
    notes:
        "Networking resources are host-only; the builtin gathers GPU values before closing handles.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "close",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Networking builtins execute eagerly on the CPU; close participates only in host bookkeeping.",
};

#[runtime_builtin(
    name = "close",
    category = "io/net",
    summary = "Close TCP clients or servers created by tcpclient, tcpserver, or accept.",
    keywords = "close,tcpclient,tcpserver,networking"
)]
fn close_builtin(args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        let closed = close_everything();
        return Ok(Value::Num(if closed { 1.0 } else { 0.0 }));
    }

    let mut any_closed = false;
    for raw in args {
        let gathered = gather_if_needed(&raw)
            .map_err(|err| runtime_error(MESSAGE_ID_INVALID_ARGUMENT, err))?;
        any_closed |= close_value(&gathered)?;
    }

    Ok(Value::Num(if any_closed { 1.0 } else { 0.0 }))
}

fn close_value(value: &Value) -> Result<bool, String> {
    match value {
        Value::Struct(st) => close_struct(st),
        Value::String(text) => close_command(text),
        Value::CharArray(chars) => {
            if chars.data.is_empty() {
                Ok(false)
            } else if chars.rows == 1 {
                let text: String = chars.data.iter().collect();
                close_command(&text)
            } else {
                Err(runtime_error(
                    MESSAGE_ID_INVALID_ARGUMENT,
                    "close: character arrays must be a single row of text",
                ))
            }
        }
        Value::StringArray(sa) => match sa.data.len() {
            0 => Ok(false),
            1 => close_command(&sa.data[0]),
            _ => Err(runtime_error(
                MESSAGE_ID_INVALID_ARGUMENT,
                "close: string array inputs must be scalar",
            )),
        },
        Value::Cell(cell) => {
            let mut closed = false;
            for element in &cell.data {
                let inner = unsafe { &*element.as_raw() };
                closed |= close_value(inner)?;
            }
            Ok(closed)
        }
        Value::Tensor(tensor) if tensor.data.is_empty() => Ok(false),
        Value::LogicalArray(logical) if logical.is_empty() => Ok(false),
        _ => Err(runtime_error(
            MESSAGE_ID_INVALID_ARGUMENT,
            format!("close: unsupported argument {value:?}"),
        )),
    }
}

fn close_struct(struct_value: &StructValue) -> Result<bool, String> {
    if let Some(id_value) = struct_value.fields.get(CLIENT_HANDLE_FIELD) {
        let id = value_to_u64(id_value).ok_or_else(|| {
            runtime_error(
                MESSAGE_ID_INVALID_HANDLE,
                "close: tcpclient identifier is missing or invalid",
            )
        })?;
        return Ok(close_client(id));
    }

    if let Some(id_value) = struct_value.fields.get(HANDLE_ID_FIELD) {
        let id = value_to_u64(id_value).ok_or_else(|| {
            runtime_error(
                MESSAGE_ID_INVALID_HANDLE,
                "close: tcpserver identifier is missing or invalid",
            )
        })?;
        let clients_closed = close_clients_for_server(id);
        let server_closed = close_server(id);
        return Ok(server_closed || clients_closed > 0);
    }

    Err(runtime_error(
        MESSAGE_ID_INVALID_HANDLE,
        "close: expected tcpclient or tcpserver struct",
    ))
}

fn close_command(raw: &str) -> Result<bool, String> {
    let token = raw.trim().to_ascii_lowercase();
    match token.as_str() {
        "all" => Ok(close_everything()),
        "clients" | "client" => Ok(close_all_clients() > 0),
        "servers" | "server" => Ok(close_all_servers() > 0),
        "" => Ok(false),
        _ => Err(runtime_error(
            MESSAGE_ID_INVALID_ARGUMENT,
            format!("close: unrecognised option '{raw}'"),
        )),
    }
}

fn close_everything() -> bool {
    let clients = close_all_clients();
    let servers = close_all_servers();
    clients > 0 || servers > 0
}

fn value_to_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Int(int) => {
            let raw = int.to_i64();
            if raw >= 0 {
                Some(raw as u64)
            } else {
                None
            }
        }
        Value::Num(num) => {
            if num.is_finite() && *num >= 0.0 && num.fract() == 0.0 {
                Some(*num as u64)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn runtime_error(id: &'static str, message: impl Into<String>) -> String {
    format!("{id}: {}", message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::builtins::io::net::accept::{accept_builtin, client_handle};
    use crate::builtins::io::net::tcpclient::tcpclient_builtin;
    use crate::builtins::io::net::tcpserver::{server_handle, tcpserver_builtin};
    use once_cell::sync::Lazy;
    use runmat_builtins::{
        CellArray, CharArray, IntValue, StringArray, StructValue, Tensor, Value,
    };
    use std::net::{TcpListener, TcpStream};
    use std::sync::Mutex;
    use std::thread;
    use std::time::Duration;

    static TEST_GUARD: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    fn client_id(value: &Value) -> u64 {
        match value {
            Value::Struct(st) => match st.fields.get(CLIENT_HANDLE_FIELD) {
                Some(Value::Int(iv)) => iv.to_i64() as u64,
                Some(Value::Num(n)) => *n as u64,
                other => panic!("unexpected client id {other:?}"),
            },
            other => panic!("expected tcpclient struct, got {other:?}"),
        }
    }

    fn server_id(value: &Value) -> u64 {
        match value {
            Value::Struct(st) => match st.fields.get(HANDLE_ID_FIELD) {
                Some(Value::Int(iv)) => iv.to_i64() as u64,
                Some(Value::Num(n)) => *n as u64,
                other => panic!("unexpected server id {other:?}"),
            },
            other => panic!("expected tcpserver struct, got {other:?}"),
        }
    }

    fn server_address(value: &Value) -> (String, u16) {
        match value {
            Value::Struct(st) => {
                let address = match st.fields.get("ServerAddress") {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::CharArray(ca)) if ca.rows == 1 => ca.data.iter().collect(),
                    other => panic!("unexpected ServerAddress {other:?}"),
                };
                let port = match st.fields.get("ServerPort") {
                    Some(Value::Int(iv)) => iv.to_i64() as u16,
                    other => panic!("unexpected ServerPort {other:?}"),
                };
                (address, port)
            }
            other => panic!("expected tcpserver struct, got {other:?}"),
        }
    }

    fn spawn_loopback_client() -> Value {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind loopback");
        let port = listener.local_addr().expect("local addr").port();
        let accept_thread = thread::spawn(move || {
            let (_stream, _) = listener.accept().expect("accept");
            thread::sleep(Duration::from_millis(10));
        });

        let client = tcpclient_builtin(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(port as i32)),
            Vec::new(),
        )
        .expect("tcpclient");

        accept_thread.join().expect("join accept thread");
        client
    }

    fn spawn_tcp_server() -> Value {
        tcpserver_builtin(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(0)),
            Vec::new(),
        )
        .expect("tcpserver")
    }

    fn accept_from_server(server: &Value) -> Value {
        let (host, port) = server_address(server);
        let connector = thread::spawn(move || {
            let _stream = TcpStream::connect((host.as_str(), port)).expect("connect client");
        });
        let accepted = accept_builtin(server.clone(), Vec::new()).expect("accept client");
        connector.join().expect("join connector");
        accepted
    }

    #[test]
    fn close_tcpclient_releases_handle() {
        let _lock = TEST_GUARD.lock().unwrap();

        let client = spawn_loopback_client();

        let cid = client_id(&client);
        let status = close_builtin(vec![client.clone()]).expect("close");
        assert_eq!(status, Value::Num(1.0));
        assert!(client_handle(cid).is_none());

        let second = close_builtin(vec![client]).expect("close again");
        assert_eq!(second, Value::Num(0.0));
    }

    #[test]
    fn close_tcpserver_releases_listener_and_clients() {
        let _lock = TEST_GUARD.lock().unwrap();

        let server = spawn_tcp_server();
        let sid = server_id(&server);
        let accepted = accept_from_server(&server);
        let accepted_id = client_id(&accepted);

        let status = close_builtin(vec![server.clone()]).expect("close server");
        assert_eq!(status, Value::Num(1.0));
        assert!(client_handle(accepted_id).is_none());
        assert!(server_handle(sid).is_none());

        let second = close_builtin(vec![server]).expect("close server again");
        assert_eq!(second, Value::Num(0.0));
    }

    #[test]
    fn close_multiple_handles_in_single_call() {
        let _lock = TEST_GUARD.lock().unwrap();

        let standalone_client = spawn_loopback_client();
        let standalone_id = client_id(&standalone_client);

        let server = spawn_tcp_server();
        let sid = server_id(&server);
        let accepted = accept_from_server(&server);
        let accepted_id = client_id(&accepted);

        let status = close_builtin(vec![
            standalone_client.clone(),
            accepted.clone(),
            server.clone(),
        ])
        .expect("close resources");
        assert_eq!(status, Value::Num(1.0));
        assert!(client_handle(standalone_id).is_none());
        assert!(client_handle(accepted_id).is_none());
        assert!(server_handle(sid).is_none());

        let second = close_builtin(vec![standalone_client, accepted, server])
            .expect("close resources again");
        assert_eq!(second, Value::Num(0.0));
    }

    #[test]
    fn close_returns_zero_when_no_resources() {
        let _lock = TEST_GUARD.lock().unwrap();

        let _ = close_builtin(Vec::new()).expect("initial cleanup");
        let status = close_builtin(Vec::new()).expect("close without resources");
        assert_eq!(status, Value::Num(0.0));
    }

    #[test]
    fn close_string_clients_option() {
        let _lock = TEST_GUARD.lock().unwrap();

        let client_a = spawn_loopback_client();
        let client_b = spawn_loopback_client();
        let id_a = client_id(&client_a);
        let id_b = client_id(&client_b);

        let status = close_builtin(vec![Value::from("clients")]).expect("close clients");
        assert_eq!(status, Value::Num(1.0));
        assert!(client_handle(id_a).is_none());
        assert!(client_handle(id_b).is_none());

        let second = close_builtin(vec![Value::from("clients")]).expect("close clients again");
        assert_eq!(second, Value::Num(0.0));
    }

    #[test]
    fn close_string_servers_option() {
        let _lock = TEST_GUARD.lock().unwrap();

        let server = spawn_tcp_server();
        let sid = server_id(&server);

        let status = close_builtin(vec![Value::from("servers")]).expect("close servers");
        assert_eq!(status, Value::Num(1.0));
        assert!(server_handle(sid).is_none());

        let second = close_builtin(vec![Value::from("servers")]).expect("close servers again");
        assert_eq!(second, Value::Num(0.0));
    }

    #[test]
    fn close_string_all_option() {
        let _lock = TEST_GUARD.lock().unwrap();

        let standalone = spawn_loopback_client();
        let standalone_id = client_id(&standalone);

        let server = spawn_tcp_server();
        let sid = server_id(&server);
        let accepted = accept_from_server(&server);
        let accepted_id = client_id(&accepted);

        let status = close_builtin(vec![Value::from("all")]).expect("close all");
        assert_eq!(status, Value::Num(1.0));
        assert!(client_handle(standalone_id).is_none());
        assert!(client_handle(accepted_id).is_none());
        assert!(server_handle(sid).is_none());

        let second = close_builtin(vec![Value::from("all")]).expect("close all again");
        assert_eq!(second, Value::Num(0.0));
    }

    #[test]
    fn close_char_array_command() {
        let _lock = TEST_GUARD.lock().unwrap();

        let client = spawn_loopback_client();
        let cid = client_id(&client);
        let status = close_builtin(vec![Value::CharArray(CharArray::new_row("clients"))])
            .expect("close char command");
        assert_eq!(status, Value::Num(1.0));
        assert!(client_handle(cid).is_none());
    }

    #[test]
    fn close_cell_array_arguments() {
        let _lock = TEST_GUARD.lock().unwrap();

        let client = spawn_loopback_client();
        let client_id_value = client_id(&client);
        let server = spawn_tcp_server();
        let server_id_value = server_id(&server);
        let accepted = accept_from_server(&server);
        let accepted_id = client_id(&accepted);

        let cell =
            CellArray::new(vec![client, accepted, server, Value::from("clients")], 1, 4).unwrap();
        let status = close_builtin(vec![Value::Cell(cell)]).expect("close cell inputs");
        assert_eq!(status, Value::Num(1.0));
        assert!(client_handle(client_id_value).is_none());
        assert!(client_handle(accepted_id).is_none());
        assert!(server_handle(server_id_value).is_none());
    }

    #[test]
    fn close_invalid_argument_errors() {
        let _lock = TEST_GUARD.lock().unwrap();

        let err = close_builtin(vec![Value::Num(13.0)]).unwrap_err();
        assert!(
            err.starts_with(MESSAGE_ID_INVALID_ARGUMENT),
            "error did not include id: {err}"
        );

        let multi = CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).unwrap();
        let err = close_builtin(vec![Value::CharArray(multi)]).unwrap_err();
        assert!(
            err.starts_with(MESSAGE_ID_INVALID_ARGUMENT),
            "char array error missing id: {err}"
        );

        let strings =
            StringArray::new(vec!["clients".to_string(), "servers".to_string()], vec![2]).unwrap();
        let err = close_builtin(vec![Value::StringArray(strings)]).unwrap_err();
        assert!(
            err.starts_with(MESSAGE_ID_INVALID_ARGUMENT),
            "string array error missing id: {err}"
        );
    }

    #[test]
    fn close_invalid_struct_errors() {
        let _lock = TEST_GUARD.lock().unwrap();

        let st = StructValue::new();
        let err = close_builtin(vec![Value::Struct(st)]).unwrap_err();
        assert!(
            err.starts_with(MESSAGE_ID_INVALID_HANDLE),
            "error did not include id: {err}"
        );
    }

    #[test]
    fn close_empty_array_argument_returns_zero() {
        let _lock = TEST_GUARD.lock().unwrap();

        let empty = Tensor::new(vec![], vec![0]).expect("empty tensor");
        let status = close_builtin(vec![Value::Tensor(empty)]).expect("close empty tensor");
        assert_eq!(status, Value::Num(0.0));
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn close_with_wgpu_provider_active() {
        let _lock = TEST_GUARD.lock().unwrap();

        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

        let client = spawn_loopback_client();
        let cid = client_id(&client);

        let status = close_builtin(vec![client]).expect("close with provider active");
        assert_eq!(status, Value::Num(1.0));
        assert!(client_handle(cid).is_none());
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
