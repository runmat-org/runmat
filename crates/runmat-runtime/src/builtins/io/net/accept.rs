//! MATLAB-compatible `accept` builtin for RunMat networking.

use runmat_builtins::{
    BuiltinErrorDescriptor, NetConstructorOption, ACCEPT_ERROR_ACCEPT_FAILED,
    ACCEPT_ERROR_INTERNAL, ACCEPT_ERROR_INVALID_NAME_VALUE, ACCEPT_ERROR_INVALID_SERVER,
    ACCEPT_ERROR_TIMEOUT,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, StructValue, Value};

use super::client_state::{insert_client, CLIENT_HANDLE_FIELD};
use super::control::{configure_stream, default_user_data, parse_timeout_value};
use super::server_state::{server_handle, TcpServerState, HANDLE_ID_FIELD};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

use runmat_time::Instant;
use std::io::{self, ErrorKind};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::time::Duration;

const BUILTIN_NAME: &str = "accept";

#[runtime_builtin(
    name = "accept",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::net::accept"
)]
pub(crate) async fn accept_builtin(server: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if !matches!(server, Value::GpuTensor(_)) {
        extract_server_id(&server)?;
    }
    let prepared_options = preflight_accept_options(&rest)?;
    super::transport::require_native_tcp(BUILTIN_NAME)?;
    let server = gather_if_needed_async(&server).await.map_err(|err| {
        super::error::with_source(
            BUILTIN_NAME,
            &ACCEPT_ERROR_INTERNAL,
            format!("accept: {}", err.message()),
            err,
        )
    })?;
    let server_id = extract_server_id(&server)?;

    let options = parse_accept_options(rest, prepared_options).await?;

    let shared_server = server_handle(server_id).ok_or_else(|| {
        accept_flow(
            &ACCEPT_ERROR_INVALID_SERVER,
            "accept: tcpserver handle is no longer valid",
        )
    })?;

    let server_guard = shared_server
        .lock()
        .map_err(|_| accept_flow(&ACCEPT_ERROR_INTERNAL, "accept: server lock poisoned"))?;

    let timeout = options.timeout.unwrap_or(server_guard.timeout);
    validate_timeout(timeout)?;

    match accept_with_timeout(&server_guard.listener, timeout) {
        Ok((stream, peer_addr)) => {
            if let Err(err) = configure_stream(&stream, timeout) {
                drop(server_guard);
                return Err(super::error::with_source(
                    BUILTIN_NAME,
                    &ACCEPT_ERROR_INTERNAL,
                    format!("accept: failed to configure stream timeouts ({err})"),
                    err,
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
                ErrorKind::WouldBlock => accept_flow(
                    &ACCEPT_ERROR_TIMEOUT,
                    format!(
                        "accept: timed out waiting for a client connection after {:.3} seconds",
                        timeout
                    ),
                ),
                _ => super::error::with_source(
                    BUILTIN_NAME,
                    &ACCEPT_ERROR_ACCEPT_FAILED,
                    format!("accept: failed to accept client ({err})"),
                    err,
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
                accept_flow(
                    &ACCEPT_ERROR_INVALID_SERVER,
                    "accept: tcpserver struct missing internal identifier",
                )
            })?;
            let id = match id_value {
                Value::Int(iv) => iv.try_to_u64().ok_or_else(|| {
                    accept_flow(
                        &ACCEPT_ERROR_INVALID_SERVER,
                        "accept: tcpserver struct has invalid internal identifier",
                    )
                })?,
                other => {
                    return Err(accept_flow(
                        &ACCEPT_ERROR_INVALID_SERVER,
                        format!("accept: expected numeric tcpserver identifier, got {other:?}"),
                    ));
                }
            };
            Ok(id)
        }
        _ => Err(accept_flow(
            &ACCEPT_ERROR_INVALID_SERVER,
            "accept: first argument must be the struct returned by tcpserver",
        )),
    }
}

#[derive(Default)]
struct AcceptOptions {
    timeout: Option<f64>,
}

fn preflight_accept_options(rest: &[Value]) -> BuiltinResult<Vec<Option<NetConstructorOption>>> {
    if !rest.len().is_multiple_of(2) {
        return Err(accept_flow(
            &ACCEPT_ERROR_INVALID_NAME_VALUE,
            "accept: name-value arguments must appear in pairs",
        ));
    }
    let mut prepared = Vec::with_capacity(rest.len() / 2);
    for pair in rest.chunks_exact(2) {
        if matches!(&pair[0], Value::GpuTensor(_)) {
            prepared.push(None);
            continue;
        }
        let name = crate::builtins::common::validation::value_text_scalar_string(&pair[0])
            .ok_or_else(|| {
                accept_flow(
                    &ACCEPT_ERROR_INVALID_NAME_VALUE,
                    format!("accept: invalid option name ({:?})", pair[0]),
                )
            })?;
        let option = NetConstructorOption::accept(&name).ok_or_else(|| {
            accept_flow(
                &ACCEPT_ERROR_INVALID_NAME_VALUE,
                format!("accept: unsupported option '{name}'"),
            )
        })?;
        if !matches!(pair[1], Value::GpuTensor(_)) {
            parse_timeout_value(&pair[1]).map_err(|err| {
                accept_flow(
                    &ACCEPT_ERROR_INVALID_NAME_VALUE,
                    format!("accept: invalid Timeout value: {err}"),
                )
            })?;
        }
        prepared.push(Some(option));
    }
    Ok(prepared)
}

async fn parse_accept_options(
    rest: Vec<Value>,
    prepared: Vec<Option<NetConstructorOption>>,
) -> BuiltinResult<AcceptOptions> {
    if rest.is_empty() {
        return Ok(AcceptOptions::default());
    }
    if !rest.len().is_multiple_of(2) {
        return Err(accept_flow(
            &ACCEPT_ERROR_INVALID_NAME_VALUE,
            "accept: name-value arguments must appear in pairs",
        ));
    }

    let mut options = AcceptOptions::default();
    let mut iter = rest.into_iter();
    for prepared_option in prepared {
        let name_raw = iter.next().expect("preflight retained paired option name");
        let value_raw = iter
            .next()
            .expect("paired iteration guarantees value exists");
        let option = match prepared_option {
            Some(option) => option,
            None => {
                let name_value = gather_if_needed_async(&name_raw)
                    .await
                    .map_err(accept_internal_source)?;
                let name =
                    crate::builtins::common::validation::value_text_scalar_string(&name_value)
                        .ok_or_else(|| {
                            accept_flow(
                                &ACCEPT_ERROR_INVALID_NAME_VALUE,
                                format!("accept: invalid option name ({name_value:?})"),
                            )
                        })?;
                NetConstructorOption::accept(&name).ok_or_else(|| {
                    accept_flow(
                        &ACCEPT_ERROR_INVALID_NAME_VALUE,
                        format!("accept: unsupported option '{name}'"),
                    )
                })?
            }
        };
        match option {
            NetConstructorOption::Timeout => {
                let gathered = gather_if_needed_async(&value_raw)
                    .await
                    .map_err(accept_internal_source)?;
                let timeout = parse_timeout_value(&gathered).map_err(|msg| {
                    accept_flow(
                        &ACCEPT_ERROR_INVALID_NAME_VALUE,
                        format!("accept: invalid Timeout value: {msg}"),
                    )
                })?;
                options.timeout = Some(timeout);
            }
            _ => {
                return Err(accept_flow(
                    &ACCEPT_ERROR_INVALID_NAME_VALUE,
                    "accept: unsupported option",
                ));
            }
        }
    }
    Ok(options)
}

fn validate_timeout(timeout: f64) -> BuiltinResult<()> {
    if timeout.is_nan() {
        return Err(accept_flow(
            &ACCEPT_ERROR_INVALID_NAME_VALUE,
            "accept: Timeout must not be NaN",
        ));
    }
    if timeout.is_sign_negative() {
        return Err(accept_flow(
            &ACCEPT_ERROR_INVALID_NAME_VALUE,
            "accept: Timeout must be non-negative",
        ));
    }
    if timeout.is_finite() && Duration::try_from_secs_f64(timeout).is_err() {
        return Err(accept_flow(
            &ACCEPT_ERROR_INVALID_NAME_VALUE,
            "accept: Timeout exceeds the supported host duration range",
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

fn accept_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn accept_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let detail = detail.strip_prefix("accept: ").unwrap_or(detail);
    accept_error_with_message(format!("{}: {}", error.message, detail), error)
}

fn accept_flow(error: &'static BuiltinErrorDescriptor, message: impl AsRef<str>) -> RuntimeError {
    accept_error_with_detail(error, message)
}

fn accept_internal_source(err: RuntimeError) -> RuntimeError {
    let detail = err.message().to_owned();
    super::error::with_source(BUILTIN_NAME, &ACCEPT_ERROR_INTERNAL, detail, err)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) mod tests {
    use super::super::client_state::remove_client_for_test;
    use super::super::server_state::{remove_server_for_test, HANDLE_ID_FIELD as SERVER_FIELD};
    use super::super::tcpserver::tcpserver_builtin;
    use super::*;
    use crate::builtins::io::net::test_support::CountingDownloadProvider;
    use runmat_builtins::ACCEPT_DESCRIPTOR;
    use runmat_value::{IntegerStorage, Tensor, Value};
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

    #[test]
    fn typed_negative_server_handle_is_rejected() {
        let mut server = StructValue::new();
        server
            .fields
            .insert(SERVER_FIELD.to_string(), Value::Int(IntValue::I8(-1)));
        let err = extract_server_id(&Value::Struct(server)).unwrap_err();
        assert_error_identifier(err, ACCEPT_ERROR_INVALID_SERVER.identifier.unwrap());
    }

    fn run_accept(server: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(accept_builtin(server, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn known_invalid_timeout_rejects_before_resident_server_download() {
        let _lock = crate::builtins::common::test_support::accel_test_lock();
        let owner = Box::leak(Box::new(CountingDownloadProvider::new()));
        let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(owner));
        let server = Value::GpuTensor(owner.resident());
        let err = run_accept(
            server.clone(),
            vec![Value::from("Timeout"), Value::Num(-1.0)],
        )
        .expect_err("invalid timeout rejects before resident server download");
        assert_eq!(err.identifier(), ACCEPT_ERROR_INVALID_NAME_VALUE.identifier);
        let err = run_accept(server, vec![Value::from("Backlog"), Value::Num(1.0)])
            .expect_err("unknown option rejects before resident server download");
        assert_eq!(err.identifier(), ACCEPT_ERROR_INVALID_NAME_VALUE.identifier);
        assert_eq!(owner.downloads(), 0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn accept_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = ACCEPT_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"client = accept(server)"));
        assert!(labels.contains(&"client = accept(server, \"Timeout\", timeout)"));
        assert!(labels.contains(&"client = accept(server, Name, Value, ...)"));
        assert_eq!(ACCEPT_INTEGER_CAPABILITIES.len(), 1);
        assert_eq!(
            ACCEPT_INTEGER_CAPABILITIES[0].inputs[0].classes,
            crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES
        );
    }

    #[test]
    fn typed_timeout_parser_accepts_all_integer_classes_and_rejects_bounds() {
        for value in [
            IntValue::I8(30),
            IntValue::I16(30),
            IntValue::I32(30),
            IntValue::I64(30),
            IntValue::U8(30),
            IntValue::U16(30),
            IntValue::U32(30),
            IntValue::U64(30),
        ] {
            assert_eq!(
                parse_timeout_value(&Value::Int(value)).expect("integer timeout"),
                30.0
            );
        }
        for storage in [
            IntegerStorage::I8(vec![30]),
            IntegerStorage::I16(vec![30]),
            IntegerStorage::I32(vec![30]),
            IntegerStorage::I64(vec![30]),
            IntegerStorage::U8(vec![30]),
            IntegerStorage::U16(vec![30]),
            IntegerStorage::U32(vec![30]),
            IntegerStorage::U64(vec![30]),
        ] {
            let typed = Tensor::new_integer(storage, vec![1, 1]).expect("timeout");
            assert_eq!(
                parse_timeout_value(&Value::Tensor(typed)).expect("tensor timeout"),
                30.0
            );
        }
        assert!(matches!(
            parse_timeout_value(&Value::Int(IntValue::I16(-1))),
            Err(TimeoutParseError::Negative)
        ));
        assert!(matches!(
            parse_timeout_value(&Value::Int(IntValue::U64(u64::MAX))),
            Err(TimeoutParseError::OutOfRange)
        ));
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
        super::super::client_state::test_guard()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn accept_rejects_non_struct() {
        let _guard = net_guard();
        let err = run_accept(Value::Num(1.0), Vec::new()).unwrap_err();
        assert_error_identifier(err, ACCEPT_ERROR_INVALID_SERVER.identifier.unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn accept_establishes_client_connection() {
        let _guard = net_guard();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let server_value = run_tcpserver(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(0)),
            Vec::new(),
        )
        .expect("tcpserver");
        let port = match struct_field(&server_value, "ServerPort") {
            Value::Int(iv) => iv.to_i64() as u16,
            Value::Num(value) => *value as u16,
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
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
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
        assert_error_identifier(err, ACCEPT_ERROR_TIMEOUT.identifier.unwrap());
        remove_server_for_test(server_id(&server_value));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn accept_rejects_invalid_timeout_name_value() {
        let _guard = net_guard();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
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
        assert_error_identifier(err, ACCEPT_ERROR_INVALID_NAME_VALUE.identifier.unwrap());
        remove_server_for_test(server_id(&server_value));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn accept_respects_per_call_timeout_override() {
        let _guard = net_guard();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let server_value = run_tcpserver(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(0)),
            Vec::new(),
        )
        .expect("tcpserver");
        let port = match struct_field(&server_value, "ServerPort") {
            Value::Int(iv) => iv.to_i64() as u16,
            Value::Num(value) => *value as u16,
            other => panic!("expected ServerPort int, got {other:?}"),
        };

        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            TcpStream::connect(("127.0.0.1", port)).expect("connect")
        });

        let client = run_accept(
            server_value.clone(),
            vec![Value::from("Timeout"), Value::Int(IntValue::U8(1))],
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
