//! MATLAB-compatible `tcpclient` builtin for RunMat.

use runmat_builtins::{
    BuiltinErrorDescriptor, BuiltinExtensionDescriptor, NetConstructorOption,
    TCPCLIENT_ERROR_CONNECT_FAILED, TCPCLIENT_ERROR_INTERNAL, TCPCLIENT_ERROR_INVALID_ADDRESS,
    TCPCLIENT_ERROR_INVALID_NAME_VALUE, TCPCLIENT_ERROR_INVALID_PORT,
    TCPCLIENT_EXPLICIT_GPU_INPUT_EXTENSION as EXPLICIT_GPU_INPUT_EXTENSION,
    TCPCLIENT_INTEGER_CONNECT_TIMEOUT_EXTENSION as INTEGER_CONNECT_TIMEOUT_EXTENSION,
    TCPCLIENT_INTEGER_PORT_EXTENSION as INTEGER_PORT_EXTENSION,
    TCPCLIENT_INTEGER_TIMEOUT_EXTENSION as INTEGER_TIMEOUT_EXTENSION,
    TCPCLIENT_LEGACY_OPTIONS_EXTENSION as LEGACY_OPTIONS_EXTENSION,
    TCPCLIENT_ZERO_PORT_EXTENSION as ZERO_PORT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, StructValue, Value};

use super::client_state::{insert_client, CLIENT_HANDLE_FIELD};
use super::control::{
    canonicalize_byte_order, configure_stream, default_user_data, parse_port, parse_timeout_value,
    string_scalar,
};
use super::server_state::{DEFAULT_TIMEOUT_SECONDS, HANDLE_ID_FIELD};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

use std::io::{self, ErrorKind};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

const BUILTIN_NAME: &str = "tcpclient";

const DEFAULT_BUFFER_SIZE: usize = 8192;

fn tcpclient_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn tcpclient_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let detail = detail.strip_prefix("tcpclient: ").unwrap_or(detail);
    tcpclient_error_with_message(format!("{}: {}", error.message, detail), error)
}

fn tcpclient_flow(
    error: &'static BuiltinErrorDescriptor,
    message: impl AsRef<str>,
) -> RuntimeError {
    tcpclient_error_with_detail(error, message)
}

#[runtime_builtin(
    name = "tcpclient",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::net::tcpclient"
)]
pub(crate) async fn tcpclient_builtin(
    host: Value,
    port: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    if std::iter::once(&host)
        .chain(std::iter::once(&port))
        .chain(rest.iter())
        .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &EXPLICIT_GPU_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    ensure_integer_extension(&port, &INTEGER_PORT_EXTENSION)?;
    preflight_name_value_extensions(&rest).await?;
    super::transport::require_native_tcp(BUILTIN_NAME)?;
    let host = gather_if_needed_async(&host)
        .await
        .map_err(|err| tcpclient_flow(&TCPCLIENT_ERROR_INTERNAL, err.message()))?;
    let port = gather_if_needed_async(&port)
        .await
        .map_err(|err| tcpclient_flow(&TCPCLIENT_ERROR_INTERNAL, err.message()))?;

    let host_text = string_scalar(&host, "tcpclient host").map_err(|err| {
        tcpclient_flow(
            &TCPCLIENT_ERROR_INVALID_ADDRESS,
            format!("tcpclient: invalid host argument ({err})"),
        )
    })?;
    let port_num = parse_port(&port).map_err(|err| {
        tcpclient_flow(
            &TCPCLIENT_ERROR_INVALID_PORT,
            format!("tcpclient: invalid port argument ({err})"),
        )
    })?;
    if port_num == 0 {
        crate::compatibility::ensure_builtin_extension_enabled(&ZERO_PORT_EXTENSION, BUILTIN_NAME)?;
    }

    let options = parse_name_value_pairs(rest).await?;

    let (stream, resolved_addr) =
        connect_with_timeout(&host_text, port_num, options.connect_timeout).map_err(|err| {
            tcpclient_flow(
                &TCPCLIENT_ERROR_CONNECT_FAILED,
                format!("tcpclient: unable to connect to {host_text}:{port_num} ({err})"),
            )
        })?;

    if let Err(err) = configure_stream(&stream, options.timeout) {
        return Err(tcpclient_flow(
            &TCPCLIENT_ERROR_INTERNAL,
            format!("tcpclient: failed to configure stream timeouts ({err})"),
        ));
    }

    let peer_addr = stream.peer_addr().map_err(|err| {
        tcpclient_flow(
            &TCPCLIENT_ERROR_INTERNAL,
            format!("tcpclient: failed to query peer address for {resolved_addr} ({err})"),
        )
    })?;
    let local_addr = stream
        .local_addr()
        .map_err(|err| tcpclient_flow(&TCPCLIENT_ERROR_INTERNAL, format!("tcpclient: {err}")))?;

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

async fn preflight_name_value_extensions(rest: &[Value]) -> BuiltinResult<()> {
    if !rest.len().is_multiple_of(2) {
        return Err(tcpclient_flow(
            &TCPCLIENT_ERROR_INVALID_NAME_VALUE,
            "tcpclient: name-value arguments must appear in pairs",
        ));
    }
    let mut known = Vec::with_capacity(rest.len() / 2);
    for pair in rest.chunks_exact(2) {
        let option = match &pair[0] {
            Value::GpuTensor(_) => None,
            value => {
                let name = string_scalar(value, "OptionName").map_err(|err| {
                    tcpclient_flow(
                        &TCPCLIENT_ERROR_INVALID_NAME_VALUE,
                        format!("tcpclient: invalid option name ({err})"),
                    )
                })?;
                Some(NetConstructorOption::tcpclient(&name).ok_or_else(|| {
                    tcpclient_flow(
                        &TCPCLIENT_ERROR_INVALID_NAME_VALUE,
                        format!("tcpclient: unsupported option '{name}'"),
                    )
                })?)
            }
        };
        if let Some(option) = option {
            preflight_known_option_value(option, &pair[1])?;
        }
        known.push((option, &pair[1]));
    }
    for (option, value) in known {
        match option {
            Some(NetConstructorOption::Timeout) => {
                crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
                    value,
                    &INTEGER_TIMEOUT_EXTENSION,
                    BUILTIN_NAME,
                    "Timeout",
                )
                .await?;
            }
            Some(NetConstructorOption::ConnectTimeout) => {
                crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
                    value,
                    &INTEGER_CONNECT_TIMEOUT_EXTENSION,
                    BUILTIN_NAME,
                    "ConnectTimeout",
                )
                .await?;
            }
            Some(
                NetConstructorOption::ByteOrder
                | NetConstructorOption::UserData
                | NetConstructorOption::Name
                | NetConstructorOption::InputBufferSize
                | NetConstructorOption::OutputBufferSize,
            ) => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &LEGACY_OPTIONS_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            None => {}
        }
    }
    Ok(())
}

fn preflight_known_option_value(option: NetConstructorOption, value: &Value) -> BuiltinResult<()> {
    if matches!(value, Value::GpuTensor(_)) {
        return Ok(());
    }
    match option {
        NetConstructorOption::Timeout | NetConstructorOption::ConnectTimeout => {
            parse_timeout_value(value).map_err(|err| {
                tcpclient_flow(
                    &TCPCLIENT_ERROR_INVALID_NAME_VALUE,
                    format!("tcpclient: invalid timeout value ({err})"),
                )
            })?;
        }
        NetConstructorOption::ByteOrder => {
            let text = string_scalar(value, "ByteOrder").map_err(|err| {
                tcpclient_flow(
                    &TCPCLIENT_ERROR_INVALID_NAME_VALUE,
                    format!("tcpclient: invalid ByteOrder value ({err})"),
                )
            })?;
            if canonicalize_byte_order(&text).is_none() {
                return Err(tcpclient_flow(
                    &TCPCLIENT_ERROR_INVALID_NAME_VALUE,
                    format!("tcpclient: unsupported ByteOrder '{text}'"),
                ));
            }
        }
        NetConstructorOption::Name => {
            string_scalar(value, "Name").map_err(|err| {
                tcpclient_flow(
                    &TCPCLIENT_ERROR_INVALID_NAME_VALUE,
                    format!("tcpclient: invalid Name value ({err})"),
                )
            })?;
        }
        NetConstructorOption::InputBufferSize => {
            parse_buffer_size(value, "InputBufferSize")?;
        }
        NetConstructorOption::OutputBufferSize => {
            parse_buffer_size(value, "OutputBufferSize")?;
        }
        NetConstructorOption::UserData => {}
    }
    Ok(())
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
            &TCPCLIENT_ERROR_INVALID_NAME_VALUE,
            "tcpclient: name-value arguments must appear in pairs",
        ));
    }

    let mut options = TcpClientOptions::default();
    let mut iter = rest.into_iter();
    while let Some(name_raw) = iter.next() {
        let value_raw = iter
            .next()
            .expect("even-length vec ensures paired name/value");
        let name_value = gather_if_needed_async(&name_raw)
            .await
            .map_err(|err| tcpclient_flow(&TCPCLIENT_ERROR_INTERNAL, err.message()))?;
        let option_name = string_scalar(&name_value, "OptionName").map_err(|err| {
            tcpclient_flow(
                &TCPCLIENT_ERROR_INVALID_NAME_VALUE,
                format!("tcpclient: invalid option name ({err})"),
            )
        })?;
        match NetConstructorOption::tcpclient(&option_name) {
            Some(NetConstructorOption::Timeout) => {
                crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
                    &value_raw,
                    &INTEGER_TIMEOUT_EXTENSION,
                    BUILTIN_NAME,
                    "Timeout",
                )
                .await?;
                let timeout_value = gather_if_needed_async(&value_raw)
                    .await
                    .map_err(|err| tcpclient_flow(&TCPCLIENT_ERROR_INTERNAL, err.message()))?;
                options.timeout = parse_timeout_value(&timeout_value).map_err(|err| {
                    tcpclient_flow(
                        &TCPCLIENT_ERROR_INVALID_NAME_VALUE,
                        format!("tcpclient: invalid Timeout value ({err})"),
                    )
                })?;
            }
            Some(NetConstructorOption::ConnectTimeout) => {
                crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
                    &value_raw,
                    &INTEGER_CONNECT_TIMEOUT_EXTENSION,
                    BUILTIN_NAME,
                    "ConnectTimeout",
                )
                .await?;
                let connect_value = gather_if_needed_async(&value_raw)
                    .await
                    .map_err(|err| tcpclient_flow(&TCPCLIENT_ERROR_INTERNAL, err.message()))?;
                options.connect_timeout = parse_timeout_value(&connect_value).map_err(|err| {
                    tcpclient_flow(
                        &TCPCLIENT_ERROR_INVALID_NAME_VALUE,
                        format!("tcpclient: invalid ConnectTimeout value ({err})"),
                    )
                })?;
            }
            Some(NetConstructorOption::ByteOrder) => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &LEGACY_OPTIONS_EXTENSION,
                    BUILTIN_NAME,
                )?;
                let order_value = gather_if_needed_async(&value_raw)
                    .await
                    .map_err(|err| tcpclient_flow(&TCPCLIENT_ERROR_INTERNAL, err.message()))?;
                let raw_order = string_scalar(&order_value, "ByteOrder").map_err(|err| {
                    tcpclient_flow(
                        &TCPCLIENT_ERROR_INVALID_NAME_VALUE,
                        format!("tcpclient: invalid ByteOrder value ({err})"),
                    )
                })?;
                let canon = canonicalize_byte_order(&raw_order).ok_or_else(|| {
                    tcpclient_flow(
                        &TCPCLIENT_ERROR_INVALID_NAME_VALUE,
                        format!("tcpclient: unsupported ByteOrder '{raw_order}'"),
                    )
                })?;
                options.byte_order = canon.to_string();
            }
            Some(NetConstructorOption::UserData) => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &LEGACY_OPTIONS_EXTENSION,
                    BUILTIN_NAME,
                )?;
                options.user_data = value_raw;
            }
            Some(NetConstructorOption::Name) => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &LEGACY_OPTIONS_EXTENSION,
                    BUILTIN_NAME,
                )?;
                let name_value = gather_if_needed_async(&value_raw)
                    .await
                    .map_err(|err| tcpclient_flow(&TCPCLIENT_ERROR_INTERNAL, err.message()))?;
                let text = string_scalar(&name_value, "Name").map_err(|err| {
                    tcpclient_flow(
                        &TCPCLIENT_ERROR_INVALID_NAME_VALUE,
                        format!("tcpclient: invalid Name value ({err})"),
                    )
                })?;
                options.name = Some(text);
            }
            Some(NetConstructorOption::InputBufferSize) => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &LEGACY_OPTIONS_EXTENSION,
                    BUILTIN_NAME,
                )?;
                let gathered = gather_if_needed_async(&value_raw)
                    .await
                    .map_err(|err| tcpclient_flow(&TCPCLIENT_ERROR_INTERNAL, err.message()))?;
                options.input_buffer_size = parse_buffer_size(&gathered, "InputBufferSize")?;
            }
            Some(NetConstructorOption::OutputBufferSize) => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &LEGACY_OPTIONS_EXTENSION,
                    BUILTIN_NAME,
                )?;
                let gathered = gather_if_needed_async(&value_raw)
                    .await
                    .map_err(|err| tcpclient_flow(&TCPCLIENT_ERROR_INTERNAL, err.message()))?;
                options.output_buffer_size = parse_buffer_size(&gathered, "OutputBufferSize")?;
            }
            None => {
                return Err(tcpclient_flow(
                    &TCPCLIENT_ERROR_INVALID_NAME_VALUE,
                    format!("tcpclient: unsupported option '{option_name}'"),
                ));
            }
        }
    }

    Ok(options)
}

fn ensure_integer_extension(
    value: &Value,
    extension: &'static BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    if crate::builtins::common::validation::value_has_native_integer_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(extension, BUILTIN_NAME)?;
    }
    Ok(())
}

fn parse_buffer_size(value: &Value, label: &str) -> BuiltinResult<i32> {
    let raw = match value {
        Value::Int(i) => i.try_to_i64().ok_or_else(|| {
            tcpclient_flow(
                &TCPCLIENT_ERROR_INVALID_NAME_VALUE,
                format!("tcpclient: {label} must lie in 1..{}", i32::MAX),
            )
        })?,
        Value::Num(n) => {
            if !n.is_finite() || n.fract() != 0.0 {
                return Err(tcpclient_flow(
                    &TCPCLIENT_ERROR_INVALID_NAME_VALUE,
                    format!("tcpclient: {label} must be a finite integer"),
                ));
            }
            *n as i64
        }
        Value::Tensor(t) if crate::builtins::common::tensor::is_scalar_tensor(t) => {
            if let Some(int) = t.integer_storage().and_then(|storage| storage.value_at(0)) {
                int.try_to_i64().ok_or_else(|| {
                    tcpclient_flow(
                        &TCPCLIENT_ERROR_INVALID_NAME_VALUE,
                        format!("tcpclient: {label} must lie in 1..{}", i32::MAX),
                    )
                })?
            } else {
                let n = crate::builtins::common::tensor::tensor_value_f64(t, 0);
                if !n.is_finite() || n.fract() != 0.0 {
                    return Err(tcpclient_flow(
                        &TCPCLIENT_ERROR_INVALID_NAME_VALUE,
                        format!("tcpclient: {label} must be a finite integer"),
                    ));
                }
                n as i64
            }
        }
        _ => {
            return Err(tcpclient_flow(
                &TCPCLIENT_ERROR_INVALID_NAME_VALUE,
                format!("tcpclient: {label} must be a numeric scalar"),
            ));
        }
    };

    if raw <= 0 || raw > i32::MAX as i64 {
        return Err(tcpclient_flow(
            &TCPCLIENT_ERROR_INVALID_NAME_VALUE,
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
        .insert("Port".to_string(), Value::Num(f64::from(remote_port)));
    st.fields.insert(
        "ServerAddress".to_string(),
        Value::String(remote_addr.clone()),
    );
    st.fields
        .insert("ServerPort".to_string(), Value::Num(f64::from(remote_port)));
    st.fields
        .insert("LocalAddress".to_string(), Value::String(local_address));
    st.fields
        .insert("LocalPort".to_string(), Value::Num(f64::from(local_port)));
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

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) mod tests {
    use super::super::client_state::remove_client_for_test;
    use super::*;
    use crate::builtins::io::net::test_support::CountingDownloadProvider;
    use runmat_builtins::TCPCLIENT_DESCRIPTOR;
    use runmat_value::{IntegerStorage, Tensor, Value};
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn known_invalid_options_reject_before_resident_endpoint_download() {
        let _lock = crate::builtins::common::test_support::accel_test_lock();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let owner = Box::leak(Box::new(CountingDownloadProvider::new()));
        let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(owner));
        let host = Value::GpuTensor(owner.resident());
        let err = run_tcpclient(
            host.clone(),
            Value::Num(80.0),
            vec![Value::from("EnableTransferDelay"), Value::Bool(true)],
        )
        .expect_err("known invalid option rejects before endpoint download");
        assert_eq!(
            err.identifier(),
            TCPCLIENT_ERROR_INVALID_NAME_VALUE.identifier
        );
        let err = run_tcpclient(host, Value::Num(80.0), vec![Value::from("Timeout")])
            .expect_err("incomplete option pair rejects before endpoint download");
        assert_eq!(
            err.identifier(),
            TCPCLIENT_ERROR_INVALID_NAME_VALUE.identifier
        );
        assert_eq!(owner.downloads(), 0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpclient_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = TCPCLIENT_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"client = tcpclient(host, port)"));
        assert!(labels.contains(&"client = tcpclient(host, port, Name, Value, ...)"));
    }

    #[test]
    fn typed_buffer_size_parser_preserves_range_boundaries() {
        assert_eq!(
            parse_buffer_size(&Value::Int(IntValue::U16(512)), "InputBufferSize").unwrap(),
            512
        );
        assert!(parse_buffer_size(&Value::Int(IntValue::I8(-1)), "InputBufferSize").is_err());
        assert!(
            parse_buffer_size(&Value::Int(IntValue::U64(u64::MAX)), "InputBufferSize").is_err()
        );

        let typed = Tensor::new_integer(IntegerStorage::U64(vec![i32::MAX as u64]), vec![1, 1])
            .expect("typed buffer size");
        assert_eq!(
            parse_buffer_size(&Value::Tensor(typed), "InputBufferSize").unwrap(),
            i32::MAX
        );

        let typed_too_large =
            Tensor::new_integer(IntegerStorage::U64(vec![i32::MAX as u64 + 1]), vec![1, 1])
                .expect("typed buffer size");
        assert!(parse_buffer_size(&Value::Tensor(typed_too_large), "InputBufferSize").is_err());

        let typed_negative = Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1])
            .expect("typed buffer size");
        assert!(parse_buffer_size(&Value::Tensor(typed_negative), "InputBufferSize").is_err());
    }

    fn net_guard() -> std::sync::MutexGuard<'static, ()> {
        crate::builtins::io::net::client_state::test_guard()
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
            Value::Num(port as f64),
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
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
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

        let client = run_tcpclient(Value::from("127.0.0.1"), Value::Num(port as f64), args)
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
        let err =
            run_tcpclient(Value::from("localhost"), Value::Num(70000.0), Vec::new()).unwrap_err();
        assert_error_identifier(err, TCPCLIENT_ERROR_INVALID_PORT.identifier.unwrap());
    }

    #[test]
    fn tcpclient_extensions_reject_before_socket_access_in_compatibility_mode() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer_port = run_tcpclient(
            Value::from("127.0.0.1"),
            Value::Int(IntValue::U16(1)),
            Vec::new(),
        )
        .unwrap_err();
        assert_eq!(
            integer_port.identifier(),
            INTEGER_PORT_EXTENSION.error_identifier
        );

        let zero_port =
            run_tcpclient(Value::from("127.0.0.1"), Value::Num(0.0), Vec::new()).unwrap_err();
        assert_eq!(zero_port.identifier(), ZERO_PORT_EXTENSION.error_identifier);

        let integer_timeout = run_tcpclient(
            Value::from("127.0.0.1"),
            Value::Num(1.0),
            vec![Value::from("Timeout"), Value::Int(IntValue::U8(5))],
        )
        .unwrap_err();
        assert_eq!(
            integer_timeout.identifier(),
            INTEGER_TIMEOUT_EXTENSION.error_identifier
        );

        let legacy_option = run_tcpclient(
            Value::from("127.0.0.1"),
            Value::Num(1.0),
            vec![Value::from("Name"), Value::from("client")],
        )
        .unwrap_err();
        assert_eq!(
            legacy_option.identifier(),
            LEGACY_OPTIONS_EXTENSION.error_identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpclient_reports_connection_failure() {
        let _guard = net_guard();
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind probe listener");
        let port = listener.local_addr().expect("probe local addr").port();
        drop(listener);

        let err = run_tcpclient(
            Value::from("127.0.0.1"),
            Value::Num(port as f64),
            vec![Value::from("ConnectTimeout"), Value::Num(0.05)],
        )
        .unwrap_err();
        assert_error_identifier(err, TCPCLIENT_ERROR_CONNECT_FAILED.identifier.unwrap());
    }
}
