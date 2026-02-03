//! MATLAB-compatible `tcpclient` builtin for RunMat.

use runmat_builtins::{IntValue, StructValue, Value};
use runmat_macros::runtime_builtin;

use super::accept::{configure_stream, insert_client, parse_timeout_value, CLIENT_HANDLE_FIELD};
use super::tcpserver::{
    canonicalize_byte_order, default_user_data, parse_port, string_scalar, DEFAULT_TIMEOUT_SECONDS,
    HANDLE_ID_FIELD,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

use std::io::{self, ErrorKind};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

const MESSAGE_ID_INVALID_ADDRESS: &str = "MATLAB:tcpclient:InvalidAddress";
const MESSAGE_ID_INVALID_PORT: &str = "MATLAB:tcpclient:InvalidPort";
const MESSAGE_ID_INVALID_NAME_VALUE: &str = "MATLAB:tcpclient:InvalidNameValue";
const MESSAGE_ID_CONNECT_FAILED: &str = "MATLAB:tcpclient:ConnectionFailed";
const MESSAGE_ID_INTERNAL: &str = "MATLAB:tcpclient:InternalError";

const DEFAULT_BUFFER_SIZE: usize = 8192;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::net::tcpclient")]
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

fn tcpclient_error(message_id: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier(message_id)
        .with_builtin("tcpclient")
        .build()
}

fn tcpclient_flow(message_id: &'static str, message: impl Into<String>) -> RuntimeError {
    tcpclient_error(message_id, message)
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::net::tcpclient")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "tcpclient",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Networking builtin executed eagerly on the CPU.",
};

#[runtime_builtin(
    name = "tcpclient",
    category = "io/net",
    summary = "Open a TCP client socket that connects to MATLAB-compatible servers.",
    keywords = "tcpclient,tcp,network,client",
    type_resolver(crate::builtins::io::type_resolvers::tcpclient_type),
    builtin_path = "crate::builtins::io::net::tcpclient"
)]
pub(crate) async fn tcpclient_builtin(
    host: Value,
    port: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let host = gather_if_needed_async(&host).await?;
    let port = gather_if_needed_async(&port).await?;

    let host_text = string_scalar(&host, "tcpclient host").map_err(|err| {
        tcpclient_flow(
            MESSAGE_ID_INVALID_ADDRESS,
            format!("tcpclient: invalid host argument ({err})"),
        )
    })?;
    let port_num = parse_port(&port).map_err(|err| {
        tcpclient_flow(
            MESSAGE_ID_INVALID_PORT,
            format!("tcpclient: invalid port argument ({err})"),
        )
    })?;

    let options = parse_name_value_pairs(rest).await?;

    let (stream, resolved_addr) =
        connect_with_timeout(&host_text, port_num, options.connect_timeout).map_err(|err| {
            tcpclient_flow(
                MESSAGE_ID_CONNECT_FAILED,
                format!("tcpclient: unable to connect to {host_text}:{port_num} ({err})"),
            )
        })?;

    if let Err(err) = configure_stream(&stream, options.timeout) {
        return Err(tcpclient_flow(
            MESSAGE_ID_INTERNAL,
            format!("tcpclient: failed to configure stream timeouts ({err})"),
        ));
    }

    let peer_addr = stream.peer_addr().map_err(|err| {
        tcpclient_flow(
            MESSAGE_ID_INTERNAL,
            format!("tcpclient: failed to query peer address for {resolved_addr} ({err})"),
        )
    })?;
    let local_addr = stream
        .local_addr()
        .map_err(|err| tcpclient_flow(MESSAGE_ID_INTERNAL, format!("tcpclient: {err}")))?;

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

async fn parse_name_value_pairs(rest: Vec<Value>) -> BuiltinResult<TcpClientOptions> {
    if rest.is_empty() {
        return Ok(TcpClientOptions::default());
    }
    if !rest.len().is_multiple_of(2) {
        return Err(tcpclient_flow(
            MESSAGE_ID_INVALID_NAME_VALUE,
            "tcpclient: name-value arguments must appear in pairs",
        ));
    }

    let mut options = TcpClientOptions::default();
    let mut iter = rest.into_iter();
    while let Some(name_raw) = iter.next() {
        let value_raw = iter
            .next()
            .expect("even-length vec ensures paired name/value");
        let name_value = gather_if_needed_async(&name_raw).await?;
        let option_name = string_scalar(&name_value, "OptionName").map_err(|err| {
            tcpclient_flow(
                MESSAGE_ID_INVALID_NAME_VALUE,
                format!("tcpclient: invalid option name ({err})"),
            )
        })?;
        let lower = option_name.to_ascii_lowercase();
        match lower.as_str() {
            "timeout" => {
                let timeout_value = gather_if_needed_async(&value_raw).await?;
                options.timeout = parse_timeout_value(&timeout_value).map_err(|err| {
                    tcpclient_flow(
                        MESSAGE_ID_INVALID_NAME_VALUE,
                        format!("tcpclient: invalid Timeout value ({err})"),
                    )
                })?;
            }
            "connecttimeout" => {
                let connect_value = gather_if_needed_async(&value_raw).await?;
                options.connect_timeout = parse_timeout_value(&connect_value).map_err(|err| {
                    tcpclient_flow(
                        MESSAGE_ID_INVALID_NAME_VALUE,
                        format!("tcpclient: invalid ConnectTimeout value ({err})"),
                    )
                })?;
            }
            "byteorder" => {
                let order_value = gather_if_needed_async(&value_raw).await?;
                let raw_order = string_scalar(&order_value, "ByteOrder").map_err(|err| {
                    tcpclient_flow(
                        MESSAGE_ID_INVALID_NAME_VALUE,
                        format!("tcpclient: invalid ByteOrder value ({err})"),
                    )
                })?;
                let canon = canonicalize_byte_order(&raw_order).ok_or_else(|| {
                    tcpclient_flow(
                        MESSAGE_ID_INVALID_NAME_VALUE,
                        format!("tcpclient: unsupported ByteOrder '{raw_order}'"),
                    )
                })?;
                options.byte_order = canon.to_string();
            }
            "userdata" => options.user_data = value_raw,
            "name" => {
                let name_value = gather_if_needed_async(&value_raw).await?;
                let text = string_scalar(&name_value, "Name").map_err(|err| {
                    tcpclient_flow(
                        MESSAGE_ID_INVALID_NAME_VALUE,
                        format!("tcpclient: invalid Name value ({err})"),
                    )
                })?;
                options.name = Some(text);
            }
            "inputbuffersize" => {
                let gathered = gather_if_needed_async(&value_raw).await?;
                options.input_buffer_size = parse_buffer_size(&gathered, "InputBufferSize")?;
            }
            "outputbuffersize" => {
                let gathered = gather_if_needed_async(&value_raw).await?;
                options.output_buffer_size = parse_buffer_size(&gathered, "OutputBufferSize")?;
            }
            _ => {
                return Err(tcpclient_flow(
                    MESSAGE_ID_INVALID_NAME_VALUE,
                    format!("tcpclient: unsupported option '{option_name}'"),
                ));
            }
        }
    }

    Ok(options)
}

fn parse_buffer_size(value: &Value, label: &str) -> BuiltinResult<i32> {
    let raw = match value {
        Value::Int(i) => i.to_i64(),
        Value::Num(n) => {
            if !n.is_finite() || n.fract() != 0.0 {
                return Err(tcpclient_flow(
                    MESSAGE_ID_INVALID_NAME_VALUE,
                    format!("tcpclient: {label} must be a finite integer"),
                ));
            }
            *n as i64
        }
        Value::Tensor(t) if t.data.len() == 1 => {
            let n = t.data[0];
            if !n.is_finite() || n.fract() != 0.0 {
                return Err(tcpclient_flow(
                    MESSAGE_ID_INVALID_NAME_VALUE,
                    format!("tcpclient: {label} must be a finite integer"),
                ));
            }
            n as i64
        }
        _ => {
            return Err(tcpclient_flow(
                MESSAGE_ID_INVALID_NAME_VALUE,
                format!("tcpclient: {label} must be a numeric scalar"),
            ));
        }
    };

    if raw <= 0 || raw > i32::MAX as i64 {
        return Err(tcpclient_flow(
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
    let mut last_err: Option<io::Error> = None;
    for addr in (host, port).to_socket_addrs()? {
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
    let mut st = StructValue::new();
    let remote_addr = peer_addr.ip().to_string();
    let remote_port = peer_addr.port();
    let local_address = local_addr.ip().to_string();
    let local_port = local_addr.port();

    let name = options
        .name
        .clone()
        .unwrap_or_else(|| format!("tcpclient:{requested_host}:{remote_port}"));

    st.fields
        .insert("Type".to_string(), Value::String("tcpclient".to_string()));
    st.fields.insert("Name".to_string(), Value::String(name));
    st.fields
        .insert("Address".to_string(), Value::String(remote_addr.clone()));
    st.fields
        .insert("Port".to_string(), Value::Int(IntValue::U16(remote_port)));
    st.fields.insert(
        "ServerAddress".to_string(),
        Value::String(remote_addr.clone()),
    );
    st.fields.insert(
        "ServerPort".to_string(),
        Value::Int(IntValue::U16(remote_port)),
    );
    st.fields
        .insert("LocalAddress".to_string(), Value::String(local_address));
    st.fields.insert(
        "LocalPort".to_string(),
        Value::Int(IntValue::U16(local_port)),
    );
    st.fields.insert(
        "RequestedAddress".to_string(),
        Value::String(requested_host.to_string()),
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
    st.fields.insert(
        "ByteOrder".to_string(),
        Value::String(options.byte_order.clone()),
    );
    st.fields
        .insert("Timeout".to_string(), Value::Num(options.timeout));
    st.fields.insert(
        "ConnectTimeout".to_string(),
        Value::Num(options.connect_timeout),
    );
    st.fields.insert(
        "InputBufferSize".to_string(),
        Value::Int(IntValue::I32(options.input_buffer_size)),
    );
    st.fields.insert(
        "OutputBufferSize".to_string(),
        Value::Int(IntValue::I32(options.output_buffer_size)),
    );
    st.fields
        .insert("UserData".to_string(), options.user_data.clone());
    st.fields.insert(
        CLIENT_HANDLE_FIELD.to_string(),
        Value::Int(IntValue::U64(client_id)),
    );
    st.fields
        .insert(HANDLE_ID_FIELD.to_string(), Value::Int(IntValue::I32(0)));

    Value::Struct(st)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::accept::remove_client_for_test;
    use super::*;
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

    fn assert_error_identifier(err: RuntimeError, expected: &str) {
        assert_eq!(err.identifier(), Some(expected));
    }

    fn run_tcpclient(host: Value, port: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(tcpclient_builtin(host, port, rest))
    }

    fn net_guard() -> std::sync::MutexGuard<'static, ()> {
        crate::builtins::io::net::accept::test_guard()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpclient_connects_to_loopback_server() {
        let _guard = net_guard();
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind loopback");
        let port = listener.local_addr().expect("local addr").port();

        let handle = thread::spawn(move || {
            let (_stream, _) = listener.accept().expect("accept");
            thread::sleep(Duration::from_millis(20));
        });

        let client = run_tcpclient(
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpclient_applies_name_value_options() {
        let _guard = net_guard();
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

        let client = run_tcpclient(
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpclient_rejects_invalid_port() {
        let _guard = net_guard();
        let err = run_tcpclient(
            Value::from("localhost"),
            Value::Int(IntValue::I32(70000)),
            Vec::new(),
        )
        .unwrap_err();
        assert_error_identifier(err, MESSAGE_ID_INVALID_PORT);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpclient_reports_connection_failure() {
        let _guard = net_guard();
        // Assume nothing listens on port 65000.
        let err = run_tcpclient(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(65000)),
            vec![Value::from("ConnectTimeout"), Value::Num(0.05)],
        )
        .unwrap_err();
        assert_error_identifier(err, MESSAGE_ID_CONNECT_FAILED);
    }
}
