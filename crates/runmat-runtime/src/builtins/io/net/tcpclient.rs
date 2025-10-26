//! MATLAB-compatible `tcpclient` builtin for RunMat.

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

use super::accept::{configure_stream, insert_client, parse_timeout_value, CLIENT_HANDLE_FIELD};
use super::tcpserver::{
    canonicalize_byte_order, default_user_data, parse_port, string_scalar, DEFAULT_TIMEOUT_SECONDS,
    HANDLE_ID_FIELD,
};

use std::collections::HashMap;
use std::io::{self, ErrorKind};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

const MESSAGE_ID_INVALID_ADDRESS: &str = "MATLAB:tcpclient:InvalidAddress";
const MESSAGE_ID_INVALID_PORT: &str = "MATLAB:tcpclient:InvalidPort";
const MESSAGE_ID_INVALID_NAME_VALUE: &str = "MATLAB:tcpclient:InvalidNameValue";
const MESSAGE_ID_CONNECT_FAILED: &str = "MATLAB:tcpclient:ConnectionFailed";
const MESSAGE_ID_INTERNAL: &str = "MATLAB:tcpclient:InternalError";

const DEFAULT_BUFFER_SIZE: usize = 8192;

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "tcpclient"
category: "io/net"
keywords: ["tcpclient", "tcp", "socket", "networking", "client"]
summary: "Open a TCP client socket that connects to MATLAB-compatible servers."
references:
  - https://www.mathworks.com/help/matlab/ref/tcpclient.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "All networking executes on the host CPU. GPU-resident scalars are gathered automatically before connecting."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::net::tcpclient::tests"
  integration: "builtins::io::net::tcpclient::tests::tcpclient_connects_to_loopback_server"
---

# What does the `tcpclient` function do in MATLAB / RunMat?
`tcpclient(host, port)` opens a TCP/IP connection to a remote server and returns a MATLAB-compatible struct that mirrors the `tcpclient` object. The struct tracks connection metadata (remote address and port, byte order, timeouts, and callback configuration) and carries an opaque identifier that other RunMat networking builtins use to operate on the live socket.

## How does the `tcpclient` function behave in MATLAB / RunMat?
- `tcpclient(host, port)` resolves the hostname (IPv4, IPv6, or DNS) and connects using the default 10 second `ConnectTimeout`. Ports must lie in the range `0–65535`.
- Name-value pairs mirror MATLAB defaults: `Timeout` (non-negative seconds, determines read/write timeouts), `ConnectTimeout` (non-negative seconds, controls how long connection establishment waits), `ByteOrder` (`"little-endian"` or `"big-endian"`), `InputBufferSize`, `OutputBufferSize`, `UserData`, and `Name`. Unknown options raise `MATLAB:tcpclient:InvalidNameValue`.
- Successful calls return a struct whose fields match MATLAB’s `tcpclient` object, including callback placeholders (`BytesAvailableFcn`, `BytesAvailableFcnMode`, `BytesAvailableFcnCount`), connection metadata (`Address`, `Port`, `ServerAddress`, `ServerPort`, `LocalAddress`, `LocalPort`), and configuration (`Timeout`, `ConnectTimeout`, buffer sizes, `ByteOrder`). Hidden fields `__tcpclient_id` and `__tcpserver_id` retain the live socket handle for companion networking builtins.
- Read and write timeouts are enforced using the `Timeout` value. Passing `inf` keeps operations blocking. The returned struct reports the configured timeout verbatim.
- Connection failures raise `MATLAB:tcpclient:ConnectionFailed` with the OS error message. Invalid addresses, ports, or name-value arguments raise the corresponding MATLAB-style diagnostics.

## `tcpclient` Function GPU Execution Behaviour
Networking always happens on the host CPU. If `host`, `port`, or name-value arguments reside on the GPU, RunMat gathers them automatically before the socket is created. The returned struct is CPU-resident, and no acceleration-provider hooks are required.

## Examples of using the `tcpclient` function in MATLAB / RunMat

### Connecting to a loopback server for local testing
```matlab
client = tcpclient("127.0.0.1", 55000);
disp(client.Address)
disp(client.Port)
```

Expected output:
```matlab
127.0.0.1
55000
```

### Customizing tcpclient timeouts and byte order
```matlab
client = tcpclient("localhost", 60000, "Timeout", 5, "ConnectTimeout", 2, "ByteOrder", "big-endian");
disp(client.Timeout)
disp(client.ConnectTimeout)
disp(client.ByteOrder)
```

Expected output:
```matlab
5
2
big-endian
```

### Storing session metadata in `UserData`
```matlab
meta = struct("session", "demo", "started", "2024-01-01T00:00:00Z");
client = tcpclient("example.com", 80, "UserData", meta);
disp(client.UserData.session)
```

Expected output:
```matlab
demo
```

### Detecting connection failures with a shorter connect timeout
```matlab
try
    client = tcpclient("192.0.2.20", 65530, "ConnectTimeout", 0.2);
catch err
    disp(err.identifier)
end
```

Expected output:
```matlab
MATLAB:tcpclient:ConnectionFailed
```

### Keeping a streaming connection open with infinite timeouts
```matlab
client = tcpclient("data.example.com", 50000, "Timeout", inf, "ConnectTimeout", inf);
disp(client.Timeout)
disp(client.ConnectTimeout)
```

Expected output:
```matlab
Inf
Inf
```

## GPU residency in RunMat (Do I need `gpuArray`?)
No. RunMat automatically gathers GPU scalars before opening sockets. The returned struct—and all networking operations—run on the CPU, so `gpuArray` offers no benefit for `tcpclient`.

## FAQ
- **Which byte orders are supported?** `"little-endian"` (default) and `"big-endian"`. Any other string raises `MATLAB:tcpclient:InvalidNameValue`.
- **Can I pass `inf` for `Timeout` or `ConnectTimeout`?** Yes. `Timeout = inf` keeps I/O blocking, and `ConnectTimeout = inf` waits indefinitely for a connection.
- **How do I close the client?** A companion builtin will release the socket. Until then, tests can use internal helpers to drop clients when finished.
- **Where do buffer sizes apply?** `InputBufferSize` and `OutputBufferSize` store the desired limits for future buffered I/O builtins. The current implementation records the values for compatibility.
- **Does the builtin support IPv6?** Yes. Pass an IPv6 literal (for example `"::1"`) or a hostname that resolves to IPv6. The returned struct reports the chosen address family.
- **What happens when the server rejects the connection?** `tcpclient` raises `MATLAB:tcpclient:ConnectionFailed` with the OS error (such as “connection refused”).

## See also
[tcpserver](./tcpserver), [accept](./accept), [fread](../filetext/fread), [fwrite](../filetext/fwrite)

## Source & feedback
- Source: `crates/runmat-runtime/src/builtins/io/net/tcpclient.rs`
- Bugs & feature requests: https://github.com/runmat-org/runmat/issues/new/choose
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "tcpclient",
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
    notes: "Host networking only. Inputs backed by GPU memory are gathered before connecting.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "tcpclient",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Networking builtin executed eagerly on the CPU.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("tcpclient", DOC_MD);

#[runtime_builtin(
    name = "tcpclient",
    category = "io/net",
    summary = "Open a TCP client socket that connects to MATLAB-compatible servers.",
    keywords = "tcpclient,tcp,network,client"
)]
pub(crate) fn tcpclient_builtin(
    host: Value,
    port: Value,
    rest: Vec<Value>,
) -> Result<Value, String> {
    let host = gather_if_needed(&host)?;
    let port = gather_if_needed(&port)?;

    let host_text = string_scalar(&host, "tcpclient host").map_err(|msg| {
        runtime_error(
            MESSAGE_ID_INVALID_ADDRESS,
            format!("tcpclient: invalid host argument ({msg})"),
        )
    })?;
    let port_num = parse_port(&port).map_err(|msg| {
        runtime_error(
            MESSAGE_ID_INVALID_PORT,
            format!("tcpclient: invalid port argument ({msg})"),
        )
    })?;

    let options = parse_name_value_pairs(rest)?;

    let (stream, resolved_addr) =
        connect_with_timeout(&host_text, port_num, options.connect_timeout).map_err(|err| {
            runtime_error(
                MESSAGE_ID_CONNECT_FAILED,
                format!("tcpclient: unable to connect to {host_text}:{port_num} ({err})"),
            )
        })?;

    if let Err(err) = configure_stream(&stream, options.timeout) {
        return Err(runtime_error(
            MESSAGE_ID_INTERNAL,
            format!("tcpclient: failed to configure stream timeouts ({err})"),
        ));
    }

    let peer_addr = stream.peer_addr().map_err(|err| {
        runtime_error(
            MESSAGE_ID_INTERNAL,
            format!("tcpclient: failed to query peer address for {resolved_addr} ({err})"),
        )
    })?;
    let local_addr = stream
        .local_addr()
        .map_err(|err| runtime_error(MESSAGE_ID_INTERNAL, format!("tcpclient: {err}")))?;

    let client_id = insert_client(
        stream,
        0,
        peer_addr,
        options.timeout,
        options.byte_order.clone(),
    );

    Ok(build_tcpclient_struct(
        client_id, &host_text, peer_addr, local_addr, &options,
    ))
}

#[derive(Clone)]
struct TcpClientOptions {
    timeout: f64,
    connect_timeout: f64,
    byte_order: String,
    user_data: Value,
    name: Option<String>,
    input_buffer_size: i32,
    output_buffer_size: i32,
}

impl Default for TcpClientOptions {
    fn default() -> Self {
        Self {
            timeout: DEFAULT_TIMEOUT_SECONDS,
            connect_timeout: DEFAULT_TIMEOUT_SECONDS,
            byte_order: "little-endian".to_string(),
            user_data: default_user_data(),
            name: None,
            input_buffer_size: DEFAULT_BUFFER_SIZE as i32,
            output_buffer_size: DEFAULT_BUFFER_SIZE as i32,
        }
    }
}

fn parse_name_value_pairs(rest: Vec<Value>) -> Result<TcpClientOptions, String> {
    if rest.is_empty() {
        return Ok(TcpClientOptions::default());
    }
    if rest.len() % 2 != 0 {
        return Err(runtime_error(
            MESSAGE_ID_INVALID_NAME_VALUE,
            "tcpclient: name-value arguments must appear in pairs".to_string(),
        ));
    }

    let mut options = TcpClientOptions::default();
    let mut iter = rest.into_iter();
    while let Some(name_raw) = iter.next() {
        let value_raw = iter
            .next()
            .expect("even-length vec ensures paired name/value");
        let name_value = gather_if_needed(&name_raw)?;
        let option_name = string_scalar(&name_value, "OptionName").map_err(|msg| {
            runtime_error(
                MESSAGE_ID_INVALID_NAME_VALUE,
                format!("tcpclient: invalid option name ({msg})"),
            )
        })?;
        let lower = option_name.to_ascii_lowercase();
        match lower.as_str() {
            "timeout" => {
                let timeout_value = gather_if_needed(&value_raw)?;
                options.timeout = parse_timeout_value(&timeout_value).map_err(|msg| {
                    runtime_error(
                        MESSAGE_ID_INVALID_NAME_VALUE,
                        format!("tcpclient: invalid Timeout value ({msg})"),
                    )
                })?;
            }
            "connecttimeout" => {
                let connect_value = gather_if_needed(&value_raw)?;
                options.connect_timeout = parse_timeout_value(&connect_value).map_err(|msg| {
                    runtime_error(
                        MESSAGE_ID_INVALID_NAME_VALUE,
                        format!("tcpclient: invalid ConnectTimeout value ({msg})"),
                    )
                })?;
            }
            "byteorder" => {
                let order_value = gather_if_needed(&value_raw)?;
                let raw_order = string_scalar(&order_value, "ByteOrder").map_err(|msg| {
                    runtime_error(
                        MESSAGE_ID_INVALID_NAME_VALUE,
                        format!("tcpclient: invalid ByteOrder value ({msg})"),
                    )
                })?;
                let canon = canonicalize_byte_order(&raw_order).ok_or_else(|| {
                    runtime_error(
                        MESSAGE_ID_INVALID_NAME_VALUE,
                        format!("tcpclient: unsupported ByteOrder '{raw_order}'"),
                    )
                })?;
                options.byte_order = canon.to_string();
            }
            "userdata" => options.user_data = value_raw,
            "name" => {
                let name_value = gather_if_needed(&value_raw)?;
                let text = string_scalar(&name_value, "Name").map_err(|msg| {
                    runtime_error(
                        MESSAGE_ID_INVALID_NAME_VALUE,
                        format!("tcpclient: invalid Name value ({msg})"),
                    )
                })?;
                options.name = Some(text);
            }
            "inputbuffersize" => {
                let gathered = gather_if_needed(&value_raw)?;
                options.input_buffer_size = parse_buffer_size(&gathered, "InputBufferSize")?;
            }
            "outputbuffersize" => {
                let gathered = gather_if_needed(&value_raw)?;
                options.output_buffer_size = parse_buffer_size(&gathered, "OutputBufferSize")?;
            }
            _ => {
                return Err(runtime_error(
                    MESSAGE_ID_INVALID_NAME_VALUE,
                    format!("tcpclient: unsupported option '{option_name}'"),
                ));
            }
        }
    }

    Ok(options)
}

fn parse_buffer_size(value: &Value, label: &str) -> Result<i32, String> {
    let raw = match value {
        Value::Int(i) => i.to_i64(),
        Value::Num(n) => {
            if !n.is_finite() || n.fract() != 0.0 {
                return Err(runtime_error(
                    MESSAGE_ID_INVALID_NAME_VALUE,
                    format!("tcpclient: {label} must be a finite integer"),
                ));
            }
            *n as i64
        }
        Value::Tensor(t) if t.data.len() == 1 => {
            let n = t.data[0];
            if !n.is_finite() || n.fract() != 0.0 {
                return Err(runtime_error(
                    MESSAGE_ID_INVALID_NAME_VALUE,
                    format!("tcpclient: {label} must be a finite integer"),
                ));
            }
            n as i64
        }
        _ => {
            return Err(runtime_error(
                MESSAGE_ID_INVALID_NAME_VALUE,
                format!("tcpclient: {label} must be a numeric scalar"),
            ));
        }
    };

    if raw <= 0 || raw > i32::MAX as i64 {
        return Err(runtime_error(
            MESSAGE_ID_INVALID_NAME_VALUE,
            format!("tcpclient: {label} must lie in 1..{}", i32::MAX),
        ));
    }
    Ok(raw as i32)
}

fn connect_with_timeout(
    host: &str,
    port: u16,
    timeout: f64,
) -> io::Result<(TcpStream, SocketAddr)> {
    let mut addrs = (host, port).to_socket_addrs()?;
    let mut last_err: Option<io::Error> = None;
    while let Some(addr) = addrs.next() {
        let attempt = if timeout.is_infinite() {
            TcpStream::connect(addr)
        } else {
            let duration = Duration::from_secs_f64(timeout);
            TcpStream::connect_timeout(&addr, duration)
        };
        match attempt {
            Ok(stream) => return Ok((stream, addr)),
            Err(err) => last_err = Some(err),
        }
    }

    match last_err {
        Some(err) => Err(err),
        None => Err(io::Error::new(
            ErrorKind::NotFound,
            "tcpclient: no addresses resolved",
        )),
    }
}

fn build_tcpclient_struct(
    client_id: u64,
    requested_host: &str,
    peer_addr: SocketAddr,
    local_addr: SocketAddr,
    options: &TcpClientOptions,
) -> Value {
    let mut fields = HashMap::new();
    let remote_addr = peer_addr.ip().to_string();
    let remote_port = peer_addr.port();
    let local_address = local_addr.ip().to_string();
    let local_port = local_addr.port();

    let name = options
        .name
        .clone()
        .unwrap_or_else(|| format!("tcpclient:{requested_host}:{remote_port}"));

    fields.insert("Type".to_string(), Value::String("tcpclient".to_string()));
    fields.insert("Name".to_string(), Value::String(name));
    fields.insert("Address".to_string(), Value::String(remote_addr.clone()));
    fields.insert("Port".to_string(), Value::Int(IntValue::U16(remote_port)));
    fields.insert(
        "ServerAddress".to_string(),
        Value::String(remote_addr.clone()),
    );
    fields.insert(
        "ServerPort".to_string(),
        Value::Int(IntValue::U16(remote_port)),
    );
    fields.insert("LocalAddress".to_string(), Value::String(local_address));
    fields.insert(
        "LocalPort".to_string(),
        Value::Int(IntValue::U16(local_port)),
    );
    fields.insert(
        "RequestedAddress".to_string(),
        Value::String(requested_host.to_string()),
    );
    fields.insert("Connected".to_string(), Value::Bool(true));
    fields.insert("Status".to_string(), Value::String("connected".to_string()));
    fields.insert(
        "NumBytesAvailable".to_string(),
        Value::Int(IntValue::I32(0)),
    );
    fields.insert("BytesAvailableFcn".to_string(), default_user_data());
    fields.insert(
        "BytesAvailableFcnMode".to_string(),
        Value::String("byte".to_string()),
    );
    fields.insert(
        "BytesAvailableFcnCount".to_string(),
        Value::Int(IntValue::I32(1)),
    );
    fields.insert(
        "ByteOrder".to_string(),
        Value::String(options.byte_order.clone()),
    );
    fields.insert("Timeout".to_string(), Value::Num(options.timeout));
    fields.insert(
        "ConnectTimeout".to_string(),
        Value::Num(options.connect_timeout),
    );
    fields.insert(
        "InputBufferSize".to_string(),
        Value::Int(IntValue::I32(options.input_buffer_size)),
    );
    fields.insert(
        "OutputBufferSize".to_string(),
        Value::Int(IntValue::I32(options.output_buffer_size)),
    );
    fields.insert("UserData".to_string(), options.user_data.clone());
    fields.insert(
        CLIENT_HANDLE_FIELD.to_string(),
        Value::Int(IntValue::U64(client_id)),
    );
    fields.insert(HANDLE_ID_FIELD.to_string(), Value::Int(IntValue::I32(0)));

    Value::Struct(StructValue { fields })
}

fn runtime_error(message_id: &'static str, message: String) -> String {
    format!("{message_id}: {message}")
}

#[cfg(test)]
mod tests {
    use super::super::accept::remove_client_for_test;
    use super::*;
    #[cfg(feature = "doc_export")]
    use crate::builtins::common::test_support;
    use runmat_builtins::Value;
    use std::net::TcpListener;
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
            other => panic!("expected client id, got {other:?}"),
        }
    }

    #[test]
    fn tcpclient_connects_to_loopback_server() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind loopback");
        let port = listener.local_addr().expect("local addr").port();

        let handle = thread::spawn(move || {
            let (_stream, _) = listener.accept().expect("accept");
            thread::sleep(Duration::from_millis(20));
        });

        let client = tcpclient_builtin(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(port as i32)),
            Vec::new(),
        )
        .expect("tcpclient");

        handle.join().expect("join listener thread");

        match struct_field(&client, "Connected") {
            Value::Bool(flag) => assert!(*flag),
            other => panic!("expected Connected bool, got {other:?}"),
        }
        match struct_field(&client, "Address") {
            Value::String(addr) => assert_eq!(addr, "127.0.0.1"),
            other => panic!("expected Address string, got {other:?}"),
        }

        let cid = client_id(&client);
        remove_client_for_test(cid);
    }

    #[test]
    fn tcpclient_applies_name_value_options() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind loopback");
        let port = listener.local_addr().expect("local addr").port();
        let handle = thread::spawn(move || {
            let (_stream, _) = listener.accept().expect("accept");
        });

        let args = vec![
            Value::from("Timeout"),
            Value::Num(5.5),
            Value::from("ConnectTimeout"),
            Value::Num(1.0),
            Value::from("ByteOrder"),
            Value::from("big-endian"),
            Value::from("InputBufferSize"),
            Value::Int(IntValue::I32(4096)),
            Value::from("OutputBufferSize"),
            Value::Int(IntValue::I32(16384)),
            Value::from("UserData"),
            Value::Num(42.0),
            Value::from("Name"),
            Value::from("CustomClient"),
        ];

        let client = tcpclient_builtin(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(port as i32)),
            args,
        )
        .expect("tcpclient");

        handle.join().expect("join listener thread");

        match struct_field(&client, "Name") {
            Value::String(name) => assert_eq!(name, "CustomClient"),
            other => panic!("expected Name string, got {other:?}"),
        }
        match struct_field(&client, "Timeout") {
            Value::Num(n) => assert_eq!(*n, 5.5),
            other => panic!("expected Timeout numeric, got {other:?}"),
        }
        match struct_field(&client, "ByteOrder") {
            Value::String(order) => assert_eq!(order, "big-endian"),
            other => panic!("expected ByteOrder string, got {other:?}"),
        }
        match struct_field(&client, "InputBufferSize") {
            Value::Int(iv) => assert_eq!(iv.to_i64(), 4096),
            other => panic!("expected InputBufferSize int, got {other:?}"),
        }
        match struct_field(&client, "UserData") {
            Value::Num(n) => assert_eq!(*n, 42.0),
            other => panic!("expected UserData numeric, got {other:?}"),
        }

        let cid = client_id(&client);
        remove_client_for_test(cid);
    }

    #[test]
    fn tcpclient_rejects_invalid_port() {
        let err = tcpclient_builtin(
            Value::from("localhost"),
            Value::Int(IntValue::I32(70000)),
            Vec::new(),
        )
        .unwrap_err();
        assert!(err.starts_with(MESSAGE_ID_INVALID_PORT));
    }

    #[test]
    fn tcpclient_reports_connection_failure() {
        // Assume nothing listens on port 65000.
        let err = tcpclient_builtin(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(65000)),
            vec![Value::from("ConnectTimeout"), Value::Num(0.05)],
        )
        .unwrap_err();
        assert!(err.starts_with(MESSAGE_ID_CONNECT_FAILED));
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
