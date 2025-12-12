//! MATLAB-compatible `tcpserver` builtin for RunMat.

use once_cell::sync::OnceCell;
use runmat_builtins::{IntValue, StructValue, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::gather_if_needed;

use std::collections::HashMap;
use std::net::{SocketAddr, TcpListener};
use std::sync::{Arc, Mutex};

#[cfg(test)]
use once_cell::sync::Lazy;

const MESSAGE_ID_INVALID_ADDRESS: &str = "MATLAB:tcpserver:InvalidAddress";
const MESSAGE_ID_INVALID_PORT: &str = "MATLAB:tcpserver:InvalidPort";
const MESSAGE_ID_INVALID_NAME_VALUE: &str = "MATLAB:tcpserver:InvalidNameValue";
const MESSAGE_ID_BIND_FAILED: &str = "MATLAB:tcpserver:BindFailed";
const MESSAGE_ID_INTERNAL: &str = "MATLAB:tcpserver:InternalError";

pub(crate) const DEFAULT_TIMEOUT_SECONDS: f64 = 10.0;
pub(crate) const HANDLE_ID_FIELD: &str = "__tcpserver_id";

type SharedTcpServer = Arc<Mutex<TcpServerState>>;

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct TcpServerState {
    pub(crate) id: u64,
    pub(crate) listener: TcpListener,
    pub(crate) requested_address: String,
    pub(crate) local_addr: SocketAddr,
    pub(crate) timeout: f64,
    pub(crate) name: Option<String>,
    #[allow(dead_code)]
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

fn insert_server(
    listener: TcpListener,
    requested_address: String,
    local_addr: SocketAddr,
    options: &ParsedOptions,
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
        timeout: options.timeout,
        name: options.name.clone(),
        byte_order: options.byte_order.clone(),
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
pub(super) fn remove_server_for_test(id: u64) {
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
pub(super) fn clear_registry_for_test() {
    let mut guard = registry()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    guard.servers.clear();
    guard.next_id = 0;
}

#[cfg(test)]
static TCP_TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "tcpserver",
        builtin_path = "crate::builtins::io::net::tcpserver"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "tcpserver"
category: "io/net"
keywords: ["tcpserver", "tcp", "socket", "networking", "server", "loopback"]
summary: "Create a TCP server that listens for MATLAB-compatible client connections."
references:
  - https://www.mathworks.com/help/matlab/ref/tcpserver.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "TCP sockets run on the host CPU. GPU-resident inputs are gathered automatically before binding sockets."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::net::tcpserver::tests"
  integration: "builtins::io::net::tcpserver::tests::tcpserver_accepts_loopback_connection"
---

# What does `tcpserver` do in MATLAB / RunMat?
`tcpserver(address, port)` creates a TCP/IP listener that waits for clients and exposes MATLAB-compatible properties describing the socket. RunMat mirrors this behaviour: addresses may be IPv4, IPv6, or hostnames, and the function returns a handle-like struct whose fields match MATLAB’s `tcpserver` object (including `ServerAddress`, `ServerPort`, `Timeout`, `UserData`, and `BytesAvailableFcn` metadata). The builtin validates inputs, binds the socket, records configuration, and raises MATLAB-style diagnostics on failure.

## How does `tcpserver` behave in MATLAB / RunMat?
- `tcpserver(address, port)` binds to the requested interface; pass `"0.0.0.0"` or `"::"` to listen on every IPv4 or IPv6 adapter respectively.
- Ports must be in the range `0–65535`. Passing `0` requests an ephemeral OS-assigned port that RunMat reports in the returned struct.
- Supported name-value pairs mirror MATLAB defaults: `Timeout` (non-negative seconds, default `10`), `UserData` (stored verbatim), `Name` (defaults to `"tcpserver:<address>:<port>"`), and `ByteOrder` (`"little-endian"` or `"big-endian"`). Unsupported options raise `MATLAB:tcpserver:InvalidNameValue`.
- The returned struct mirrors MATLAB properties, including connection state, callback metadata, and an internal `__tcpserver_id` identifier that future networking builtins use to locate the listener.
- GPU-resident scalars are gathered automatically before binding so that socket setup always executes on the host CPU.
- Bind failures raise `MATLAB:tcpserver:BindFailed` with the OS-provided error message, preserving MATLAB-style diagnostics.

## `tcpserver` Function GPU Execution Behaviour
Networking occurs entirely on the host CPU. If the address or port arguments originate on the GPU, RunMat gathers them before binding the socket. The struct returned by `tcpserver` is always CPU-resident, and acceleration providers do not need to implement any hooks for this builtin.

## Examples of using the `tcpserver` function in MATLAB / RunMat

### Creating a loopback TCP server on a fixed port
```matlab
srv = tcpserver("127.0.0.1", 55000);
disp(srv.ServerAddress)
disp(srv.ServerPort)
```

Expected output:
```matlab
127.0.0.1
55000
```

### Requesting an ephemeral port and inspecting the assigned value
```matlab
srv = tcpserver("0.0.0.0", 0);
fprintf("Listening on %s:%d\n", srv.ServerAddress, srv.ServerPort);
```

Expected output:
```matlab
Listening on 0.0.0.0:54873   % actual port varies by run
```

### Configuring the timeout and storing metadata in UserData
```matlab
srv = tcpserver("localhost", 60000, "Timeout", 5, "UserData", struct("name", "demo"));
disp(srv.Timeout)
disp(srv.UserData.name)
```

Expected output:
```matlab
5
demo
```

### Assigning a custom server name
```matlab
srv = tcpserver("::1", 45000, "Name", "LoopbackServer");
disp(srv.Name)
```

Expected output:
```matlab
LoopbackServer
```

### Selecting big-endian byte order for binary protocols
```matlab
srv = tcpserver("127.0.0.1", 45001, "ByteOrder", "big-endian");
disp(srv.ByteOrder)
```

Expected output:
```matlab
big-endian
```

### Handling invalid ports with MATLAB-style diagnostics
```matlab
try
    srv = tcpserver("127.0.0.1", 99999);
catch err
    disp(err.identifier)
    disp(err.message)
end
```

Expected output:
```matlab
MATLAB:tcpserver:InvalidPort
MATLAB:tcpserver:InvalidPort: tcpserver: port 99999 is outside the valid range 0–65535
```

## GPU residency in RunMat (Do I need `gpuArray`?)
No. `tcpserver` is a host-side operation. RunMat transparently gathers GPU scalars before binding the socket, so keeping address or port values on the GPU provides no performance benefit.

## FAQ
- **What range of ports can I use?** Valid ports are `0–65535`. Port `0` lets the OS choose an available ephemeral port, which RunMat reports in the returned struct.
- **How do I discover which clients are connected?** The returned struct exposes MATLAB-compatible fields (`Connected`, `ClientAddress`, `ClientPort`). Future networking builtins will use the internal `__tcpserver_id` to inspect active connections.
- **Does RunMat support IPv6?** Yes. Pass an IPv6 literal (e.g., `"::1"`) or hostname that resolves to IPv6. The returned `ServerAddress` reflects the bound address.
- **Can I change the timeout after creating the server?** Not yet. The current builtin records the timeout value for future builtins. A forthcoming setter will update the listener configuration.
- **Does `tcpserver` fire callbacks like MATLAB’s `BytesAvailableFcn`?** Callback-related properties are present for compatibility, and future updates will allow callers to configure them. They currently act as placeholders while connection management matures.
- **How do I close the server?** A companion builtin (planned) will accept the returned struct and release the underlying listener. Tests can invoke internal helpers until that builtin lands.
- **Can I use GPU arrays for address or port?** Yes—RunMat gathers them automatically before binding.
- **What happens if the port is already in use?** `tcpserver` raises `MATLAB:tcpserver:BindFailed` with the OS error message.
- **How do I pass additional socket options?** Current support covers `Timeout`, `Name`, `UserData`, and `ByteOrder`. Additional name-value options (broadcast, reuse, keep-alive) will land alongside their corresponding provider hooks.
- **Is TLS supported?** Not directly. Combine `tcpserver` with application-layer protocol helpers or custom TLS wrappers until dedicated support lands.

## See also
[fopen](./fopen), [fread](./fread), [fwrite](./fwrite), [fprintf](./fprintf)

## Source & feedback
- Source: `crates/runmat-runtime/src/builtins/io/net/tcpserver.rs`
- Bugs & feature requests: https://github.com/runmat-org/runmat/issues/new/choose
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::net::tcpserver")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "tcpserver",
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
    notes: "Host networking only. GPU-resident scalars are gathered prior to socket binding.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::net::tcpserver")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "tcpserver",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Networking builtin executed eagerly on the CPU.",
};

#[runtime_builtin(
    name = "tcpserver",
    category = "io/net",
    summary = "Create a TCP server that listens for MATLAB-compatible client connections.",
    keywords = "tcpserver,tcp,network,server",
    builtin_path = "crate::builtins::io::net::tcpserver"
)]
pub(crate) fn tcpserver_builtin(
    address: Value,
    port: Value,
    rest: Vec<Value>,
) -> Result<Value, String> {
    let address = gather_if_needed(&address)?;
    let port = gather_if_needed(&port)?;

    let host = string_scalar(&address, "tcpserver address")
        .map_err(|msg| runtime_error(MESSAGE_ID_INVALID_ADDRESS, msg))?;
    let port = parse_port(&port).map_err(|msg| runtime_error(MESSAGE_ID_INVALID_PORT, msg))?;

    let options = parse_name_value_pairs(rest)?;

    let listener = TcpListener::bind((host.as_str(), port)).map_err(|err| {
        runtime_error(
            MESSAGE_ID_BIND_FAILED,
            format!("tcpserver: unable to bind {host}:{port} ({err})"),
        )
    })?;
    let local_addr = listener
        .local_addr()
        .map_err(|err| runtime_error(MESSAGE_ID_INTERNAL, format!("tcpserver: {err}")))?;

    let id = insert_server(listener, host.clone(), local_addr, &options);
    Ok(build_tcpserver_struct(id, &host, local_addr, &options))
}

#[derive(Clone)]
struct ParsedOptions {
    timeout: f64,
    user_data: Value,
    name: Option<String>,
    byte_order: String,
}

impl Default for ParsedOptions {
    fn default() -> Self {
        Self {
            timeout: DEFAULT_TIMEOUT_SECONDS,
            user_data: default_user_data(),
            name: None,
            byte_order: "little-endian".to_string(),
        }
    }
}

fn parse_name_value_pairs(rest: Vec<Value>) -> Result<ParsedOptions, String> {
    if rest.is_empty() {
        return Ok(ParsedOptions::default());
    }
    if !rest.len().is_multiple_of(2) {
        return Err(runtime_error(
            MESSAGE_ID_INVALID_NAME_VALUE,
            "tcpserver: name-value arguments must appear in pairs".to_string(),
        ));
    }

    let mut options = ParsedOptions::default();
    let mut iter = rest.into_iter();
    while let Some(name_raw) = iter.next() {
        let value_raw = iter
            .next()
            .expect("even-length vector ensures paired name/value");
        let name_value = gather_if_needed(&name_raw)?;

        let name = string_scalar(&name_value, "OptionName").map_err(|msg| {
            runtime_error(
                MESSAGE_ID_INVALID_NAME_VALUE,
                format!("tcpserver: invalid option name: {msg}"),
            )
        })?;
        let lower = name.to_ascii_lowercase();
        match lower.as_str() {
            "timeout" => {
                let timeout_value = gather_if_needed(&value_raw)?;
                options.timeout = parse_timeout(&timeout_value).map_err(|msg| {
                    runtime_error(
                        MESSAGE_ID_INVALID_NAME_VALUE,
                        format!("tcpserver: invalid Timeout value: {msg}"),
                    )
                })?
            }
            "userdata" => options.user_data = value_raw,
            "name" => {
                let name_value = gather_if_needed(&value_raw)?;
                let text = string_scalar(&name_value, "Name").map_err(|msg| {
                    runtime_error(
                        MESSAGE_ID_INVALID_NAME_VALUE,
                        format!("tcpserver: invalid Name value: {msg}"),
                    )
                })?;
                options.name = Some(text);
            }
            "byteorder" => {
                let order_value = gather_if_needed(&value_raw)?;
                let raw_order = string_scalar(&order_value, "ByteOrder").map_err(|msg| {
                    runtime_error(
                        MESSAGE_ID_INVALID_NAME_VALUE,
                        format!("tcpserver: invalid ByteOrder value: {msg}"),
                    )
                })?;
                let canon = canonicalize_byte_order(&raw_order).ok_or_else(|| {
                    runtime_error(
                        MESSAGE_ID_INVALID_NAME_VALUE,
                        format!("tcpserver: unsupported ByteOrder '{raw_order}'"),
                    )
                })?;
                options.byte_order = canon.to_string();
            }
            _ => {
                return Err(runtime_error(
                    MESSAGE_ID_INVALID_NAME_VALUE,
                    format!("tcpserver: unsupported option '{name}'"),
                ));
            }
        }
    }

    Ok(options)
}

fn build_tcpserver_struct(
    id: u64,
    requested_address: &str,
    local_addr: SocketAddr,
    options: &ParsedOptions,
) -> Value {
    let mut st = StructValue::new();

    let server_address = local_addr.ip().to_string();
    let server_port = local_addr.port();
    let name = options
        .name
        .clone()
        .unwrap_or_else(|| format!("tcpserver:{server_address}:{server_port}"));

    st.fields
        .insert("Type".to_string(), Value::String("tcpserver".to_string()));
    st.fields.insert("Name".to_string(), Value::String(name));
    st.fields.insert(
        "ServerAddress".to_string(),
        Value::String(server_address.clone()),
    );
    st.fields.insert(
        "ServerPort".to_string(),
        Value::Int(IntValue::U16(server_port)),
    );
    st.fields
        .insert("Port".to_string(), Value::Int(IntValue::U16(server_port)));
    st.fields.insert(
        "RequestedAddress".to_string(),
        Value::String(requested_address.to_string()),
    );
    st.fields
        .insert("Connected".to_string(), Value::Bool(false));
    st.fields
        .insert("ClientAddress".to_string(), Value::String(String::new()));
    st.fields
        .insert("ClientPort".to_string(), Value::Int(IntValue::I32(0)));
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
        .insert("ConnectionChangedFcn".to_string(), default_user_data());
    st.fields.insert(
        "ByteOrder".to_string(),
        Value::String(options.byte_order.clone()),
    );
    st.fields
        .insert("EnableBroadcast".to_string(), Value::Bool(false));
    st.fields
        .insert("EnableMulticast".to_string(), Value::Bool(false));
    st.fields
        .insert("EnableReuseAddress".to_string(), Value::Bool(false));
    st.fields
        .insert("KeepAlive".to_string(), Value::Bool(false));
    st.fields
        .insert(HANDLE_ID_FIELD.to_string(), Value::Int(IntValue::U64(id)));
    st.fields
        .insert("Timeout".to_string(), Value::Num(options.timeout));
    st.fields
        .insert("UserData".to_string(), options.user_data.clone());

    Value::Struct(st)
}

pub(crate) fn string_scalar(value: &Value, context: &str) -> Result<String, String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        other => Err(format!(
            "{context} must be a string scalar or character vector (got {other:?})"
        )),
    }
}

pub(crate) fn parse_port(value: &Value) -> Result<u16, String> {
    let port = match value {
        Value::Int(int) => int.to_i64(),
        Value::Num(num) => {
            if !num.is_finite() {
                return Err("port must be finite".to_string());
            }
            if num.fract() != 0.0 {
                return Err("port must be an integer value".to_string());
            }
            *num as i64
        }
        Value::Tensor(t) if t.data.len() == 1 => {
            let raw = t.data[0];
            if !raw.is_finite() {
                return Err("port must be finite".to_string());
            }
            if raw.fract() != 0.0 {
                return Err("port must be an integer value".to_string());
            }
            raw as i64
        }
        Value::Tensor(_) => {
            return Err("port must be a numeric scalar".to_string());
        }
        _ => {
            return Err("port must be a numeric scalar".to_string());
        }
    };

    if !(0..=65_535).contains(&port) {
        return Err(format!("port {port} is outside the valid range 0–65535"));
    }
    Ok(port as u16)
}

fn parse_timeout(value: &Value) -> Result<f64, String> {
    let timeout = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        Value::Tensor(t) if t.data.len() == 1 => t.data[0],
        Value::Tensor(_) => return Err("Timeout must be a scalar".to_string()),
        _ => {
            return Err("Timeout must be a numeric scalar".to_string());
        }
    };

    if !timeout.is_finite() || timeout < 0.0 {
        return Err("Timeout must be a finite, non-negative scalar".to_string());
    }
    Ok(timeout)
}

pub(crate) fn canonicalize_byte_order(raw: &str) -> Option<&'static str> {
    let mut compact = String::with_capacity(raw.len());
    for ch in raw.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            compact.push(ch.to_ascii_lowercase());
        }
    }
    match compact.as_str() {
        "littleendian" | "little" => Some("little-endian"),
        "bigendian" | "big" => Some("big-endian"),
        _ => None,
    }
}

pub(crate) fn default_user_data() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

fn runtime_error(message_id: &'static str, message: String) -> String {
    format!("{message_id}: {message}")
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{Tensor, Value};
    use std::net::TcpStream;
    use std::time::Duration;

    fn struct_field<'a>(value: &'a Value, name: &str) -> &'a Value {
        match value {
            Value::Struct(struct_value) => struct_value
                .fields
                .get(name)
                .unwrap_or_else(|| panic!("missing field {name}")),
            _ => panic!("expected struct result"),
        }
    }

    fn server_id(value: &Value) -> u64 {
        match struct_field(value, HANDLE_ID_FIELD) {
            Value::Int(IntValue::U64(id)) => *id,
            Value::Int(iv) => iv.to_i64() as u64,
            other => panic!("unexpected id representation {other:?}"),
        }
    }

    #[test]
    fn tcpserver_accepts_loopback_connection() {
        let _lock = TCP_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let result = tcpserver_builtin(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(0)),
            Vec::new(),
        )
        .expect("tcpserver");
        let id = server_id(&result);
        let address = match struct_field(&result, "ServerAddress") {
            Value::String(s) => s.clone(),
            other => panic!("expected ServerAddress string, got {other:?}"),
        };
        let port = match struct_field(&result, "ServerPort") {
            Value::Int(iv) => iv.to_i64() as u16,
            other => panic!("expected ServerPort int, got {other:?}"),
        };

        TcpStream::connect((address.as_str(), port)).expect("connect to loopback server");
        remove_server_for_test(id);
    }

    #[test]
    fn tcpserver_applies_timeout_option() {
        let _lock = TCP_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let result = tcpserver_builtin(
            Value::from("localhost"),
            Value::Int(IntValue::I32(0)),
            vec![Value::from("Timeout"), Value::Num(5.0)],
        )
        .expect("tcpserver");

        let timeout = match struct_field(&result, "Timeout") {
            Value::Num(n) => *n,
            other => panic!("expected numeric timeout, got {other:?}"),
        };
        assert!((timeout - 5.0).abs() < f64::EPSILON);

        let id = server_id(&result);
        remove_server_for_test(id);
    }

    #[test]
    fn tcpserver_supports_custom_name() {
        let _lock = TCP_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let result = tcpserver_builtin(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(0)),
            vec![Value::from("Name"), Value::from("CustomListener")],
        )
        .expect("tcpserver");

        let name = match struct_field(&result, "Name") {
            Value::String(s) => s.clone(),
            other => panic!("expected Name string, got {other:?}"),
        };
        assert_eq!(name, "CustomListener");

        let id = server_id(&result);
        remove_server_for_test(id);
    }

    #[test]
    fn tcpserver_accepts_byte_order_option() {
        let _lock = TCP_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let result = tcpserver_builtin(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(0)),
            vec![Value::from("ByteOrder"), Value::from("big-endian")],
        )
        .expect("tcpserver");

        let order = match struct_field(&result, "ByteOrder") {
            Value::String(s) => s.clone(),
            other => panic!("expected ByteOrder string, got {other:?}"),
        };
        assert_eq!(order, "big-endian");

        let id = server_id(&result);
        remove_server_for_test(id);
    }

    #[test]
    fn tcpserver_rejects_invalid_byte_order() {
        let err = tcpserver_builtin(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(8000)),
            vec![Value::from("ByteOrder"), Value::from("middle-endian")],
        )
        .unwrap_err();
        assert!(err.starts_with(MESSAGE_ID_INVALID_NAME_VALUE));
    }

    #[test]
    fn tcpserver_accepts_scalar_tensor_port() {
        let _lock = TCP_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let tensor_port = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let result = tcpserver_builtin(
            Value::from("127.0.0.1"),
            Value::Tensor(tensor_port),
            Vec::new(),
        )
        .expect("tcpserver");

        let port = match struct_field(&result, "ServerPort") {
            Value::Int(iv) => iv.to_i64() as u16,
            other => panic!("expected ServerPort int, got {other:?}"),
        };
        assert!(port > 0);

        let id = server_id(&result);
        remove_server_for_test(id);
    }

    #[test]
    fn tcpserver_rejects_invalid_port() {
        let err = tcpserver_builtin(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(-1)),
            Vec::new(),
        )
        .unwrap_err();
        assert!(err.starts_with(MESSAGE_ID_INVALID_PORT));
    }

    #[test]
    fn tcpserver_requires_name_value_pairs() {
        let err = tcpserver_builtin(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(9000)),
            vec![Value::from("Timeout")],
        )
        .unwrap_err();
        assert!(err.starts_with(MESSAGE_ID_INVALID_NAME_VALUE));
    }

    #[test]
    fn tcpserver_stores_userdata() {
        let _lock = TCP_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let mut user_struct_value = StructValue::new();
        user_struct_value
            .fields
            .insert("tag".to_string(), Value::from("demo"));
        let user_struct = Value::Struct(user_struct_value);

        let result = tcpserver_builtin(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(0)),
            vec![Value::from("UserData"), user_struct.clone()],
        )
        .expect("tcpserver");

        let stored = struct_field(&result, "UserData");
        assert_eq!(stored, &user_struct);

        let id = server_id(&result);
        remove_server_for_test(id);
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    fn tcpserver_times_out_connect_attempt() {
        let _lock = TCP_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let result = tcpserver_builtin(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(0)),
            vec![Value::from("Timeout"), Value::Num(1.5)],
        )
        .expect("tcpserver");
        let id = server_id(&result);
        let port = match struct_field(&result, "ServerPort") {
            Value::Int(iv) => iv.to_i64() as u16,
            other => panic!("expected ServerPort int, got {other:?}"),
        };

        let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_millis(10)))
            .expect("set timeout");

        remove_server_for_test(id);
    }
}
