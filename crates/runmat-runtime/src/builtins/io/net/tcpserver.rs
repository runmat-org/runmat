//! MATLAB-compatible `tcpserver` builtin for RunMat.

use super::control::{
    canonicalize_byte_order, default_user_data, parse_port, string_scalar, DEFAULT_TIMEOUT_SECONDS,
};
use crate::context::{RuntimeTcpBind, RuntimeTcpByteOrder, RuntimeTcpInvocation};
use runmat_builtins::{
    BuiltinErrorDescriptor, NetConstructorOption, TCPSERVER_ERROR_BIND_FAILED,
    TCPSERVER_ERROR_INTERNAL, TCPSERVER_ERROR_INVALID_ADDRESS, TCPSERVER_ERROR_INVALID_NAME_VALUE,
    TCPSERVER_ERROR_INVALID_PORT,
    TCPSERVER_EXPLICIT_GPU_INPUT_EXTENSION as EXPLICIT_GPU_INPUT_EXTENSION,
    TCPSERVER_INTEGER_PORT_EXTENSION as INTEGER_PORT_EXTENSION,
    TCPSERVER_INTEGER_TIMEOUT_EXTENSION as INTEGER_TIMEOUT_EXTENSION,
    TCPSERVER_LEGACY_OPTIONS_EXTENSION as LEGACY_OPTIONS_EXTENSION,
    TCPSERVER_ZERO_PORT_EXTENSION as ZERO_PORT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ForeignRef, IntValue, Tensor, Value};

use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};
use thiserror::Error;

use std::net::SocketAddr;

const BUILTIN_NAME: &str = "tcpserver";

fn tcpserver_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn tcpserver_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let detail = detail.strip_prefix("tcpserver: ").unwrap_or(detail);
    tcpserver_error_with_message(format!("{}: {}", error.message, detail), error)
}

fn tcpserver_flow(
    error: &'static BuiltinErrorDescriptor,
    message: impl AsRef<str>,
) -> RuntimeError {
    tcpserver_error_with_detail(error, message)
}

fn tcpserver_internal_source(err: RuntimeError) -> RuntimeError {
    let detail = err.message().to_owned();
    super::error::with_source(BUILTIN_NAME, &TCPSERVER_ERROR_INTERNAL, detail, err)
}

#[runtime_builtin(
    name = "tcpserver",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::net::tcpserver"
)]
pub(crate) async fn tcpserver_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    address: Value,
    port: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    if std::iter::once(&address)
        .chain(std::iter::once(&port))
        .chain(rest.iter())
        .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &EXPLICIT_GPU_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if crate::builtins::common::validation::value_has_native_integer_class(&port) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTEGER_PORT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let prepared_options = preflight_name_value_extensions(rest.view()).await?;
    let known_address = if matches!(&address, Value::GpuTensor(_)) {
        None
    } else {
        Some(parse_address(&address)?)
    };
    let known_port = if matches!(&port, Value::GpuTensor(_)) {
        None
    } else {
        Some(parse_endpoint_port(&port)?)
    };
    let service = super::transport::service(runtime.as_ref(), BUILTIN_NAME)?;
    let mut work = super::transport::NetCooperation::new(runtime.as_ref(), BUILTIN_NAME);
    let address = gather_if_needed_async(&address).await.map_err(|err| {
        super::error::with_source(
            BUILTIN_NAME,
            &TCPSERVER_ERROR_INTERNAL,
            format!("tcpserver: {}", err.message()),
            err,
        )
    })?;
    let port = gather_if_needed_async(&port).await.map_err(|err| {
        super::error::with_source(
            BUILTIN_NAME,
            &TCPSERVER_ERROR_INTERNAL,
            format!("tcpserver: {}", err.message()),
            err,
        )
    })?;

    let host = match known_address {
        Some(address) => address,
        None => parse_address(&address)?,
    };
    let port = match known_port {
        Some(port) => port,
        None => parse_endpoint_port(&port)?,
    };

    let options = parse_name_value_pairs(rest, prepared_options).await?;

    let endpoint = work
        .await_service(
            service.bind(
                RuntimeTcpInvocation::from_context(runtime.as_ref()),
                RuntimeTcpBind {
                    address: host.clone(),
                    port,
                    io_timeout: (options.timeout != 0.0)
                        .then(|| std::time::Duration::from_secs_f64(options.timeout)),
                    byte_order: if options.byte_order == "big-endian" {
                        RuntimeTcpByteOrder::Big
                    } else {
                        RuntimeTcpByteOrder::Little
                    },
                },
            ),
        )
        .await
        .map_err(|failure| failure.into_runtime_error(BUILTIN_NAME))?
        .map_err(|err| {
            super::error::with_source(
                BUILTIN_NAME,
                &TCPSERVER_ERROR_BIND_FAILED,
                format!("tcpserver: unable to bind {host}:{port} ({err})"),
                err,
            )
        })?;
    work.completed(0).await?;
    build_tcpserver_object(
        runtime.as_ref(),
        endpoint.authority,
        &host,
        endpoint.local_addr,
        &options,
    )
}

fn parse_address(address: &Value) -> BuiltinResult<String> {
    string_scalar(address, "tcpserver address").map_err(|err| {
        super::error::with_source(
            BUILTIN_NAME,
            &TCPSERVER_ERROR_INVALID_ADDRESS,
            err.to_string(),
            err,
        )
    })
}

fn parse_endpoint_port(port: &Value) -> BuiltinResult<u16> {
    let port_num = parse_port(port).map_err(|err| {
        super::error::with_source(
            BUILTIN_NAME,
            &TCPSERVER_ERROR_INVALID_PORT,
            err.to_string(),
            err,
        )
    })?;
    if port_num == 0 {
        crate::compatibility::ensure_builtin_extension_enabled(&ZERO_PORT_EXTENSION, BUILTIN_NAME)?;
    }
    Ok(port_num)
}

async fn preflight_name_value_extensions(
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Vec<Option<NetConstructorOption>>> {
    if !rest.len().is_multiple_of(2) {
        return Err(tcpserver_flow(
            &TCPSERVER_ERROR_INVALID_NAME_VALUE,
            "tcpserver: name-value arguments must appear in pairs",
        ));
    }
    let mut known = Vec::with_capacity(rest.len() / 2);
    for pair in rest.chunks_exact(2) {
        let option = match &pair[0] {
            Value::GpuTensor(_) => None,
            value => {
                let name = string_scalar(value, "OptionName").map_err(|err| {
                    tcpserver_flow(
                        &TCPSERVER_ERROR_INVALID_NAME_VALUE,
                        format!("tcpserver: invalid option name: {err}"),
                    )
                })?;
                Some(NetConstructorOption::tcpserver(&name).ok_or_else(|| {
                    tcpserver_flow(
                        &TCPSERVER_ERROR_INVALID_NAME_VALUE,
                        format!("tcpserver: unsupported option '{name}'"),
                    )
                })?)
            }
        };
        if let Some(option) = option {
            preflight_known_option_value(option, &pair[1])?;
        }
        known.push(option);
    }
    for (option, pair) in known.iter().copied().zip(rest.chunks_exact(2)) {
        let value = &pair[1];
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
            Some(NetConstructorOption::UserData | NetConstructorOption::Name) => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &LEGACY_OPTIONS_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            Some(NetConstructorOption::ByteOrder) => {}
            None => {}
            Some(_) => {
                return Err(tcpserver_flow(
                    &TCPSERVER_ERROR_INVALID_NAME_VALUE,
                    "tcpserver: unsupported option",
                ));
            }
        }
    }
    Ok(known)
}

fn preflight_known_option_value(option: NetConstructorOption, value: &Value) -> BuiltinResult<()> {
    if matches!(value, Value::GpuTensor(_)) {
        return Ok(());
    }
    match option {
        NetConstructorOption::Timeout => {
            parse_timeout(value).map_err(|err| {
                tcpserver_flow(
                    &TCPSERVER_ERROR_INVALID_NAME_VALUE,
                    format!("tcpserver: invalid Timeout value: {err}"),
                )
            })?;
        }
        NetConstructorOption::ByteOrder => {
            let text = string_scalar(value, "ByteOrder").map_err(|err| {
                tcpserver_flow(
                    &TCPSERVER_ERROR_INVALID_NAME_VALUE,
                    format!("tcpserver: invalid ByteOrder value: {err}"),
                )
            })?;
            if canonicalize_byte_order(&text).is_none() {
                return Err(tcpserver_flow(
                    &TCPSERVER_ERROR_INVALID_NAME_VALUE,
                    format!("tcpserver: unsupported ByteOrder '{text}'"),
                ));
            }
        }
        NetConstructorOption::Name => {
            string_scalar(value, "Name").map_err(|err| {
                tcpserver_flow(
                    &TCPSERVER_ERROR_INVALID_NAME_VALUE,
                    format!("tcpserver: invalid Name value: {err}"),
                )
            })?;
        }
        NetConstructorOption::UserData => {}
        _ => {
            return Err(tcpserver_flow(
                &TCPSERVER_ERROR_INVALID_NAME_VALUE,
                "tcpserver: unsupported option",
            ));
        }
    }
    Ok(())
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

async fn parse_name_value_pairs(
    rest: crate::call::BuiltinArguments,
    prepared: Vec<Option<NetConstructorOption>>,
) -> BuiltinResult<ParsedOptions> {
    if rest.is_empty() {
        return Ok(ParsedOptions::default());
    }
    if !rest.len().is_multiple_of(2) {
        return Err(tcpserver_flow(
            &TCPSERVER_ERROR_INVALID_NAME_VALUE,
            "tcpserver: name-value arguments must appear in pairs",
        ));
    }

    let mut options = ParsedOptions::default();
    let mut iter = rest.into_iter();
    for prepared_option in prepared {
        let name_raw = iter.next().expect("preflight retained paired option name");
        let value_raw = iter
            .next()
            .expect("even-length vector ensures paired name/value");
        let option = match prepared_option {
            Some(option) => option,
            None => {
                let name_value = gather_if_needed_async(&name_raw)
                    .await
                    .map_err(tcpserver_internal_source)?;
                let name = string_scalar(&name_value, "OptionName").map_err(|err| {
                    tcpserver_flow(
                        &TCPSERVER_ERROR_INVALID_NAME_VALUE,
                        format!("tcpserver: invalid option name: {err}"),
                    )
                })?;
                NetConstructorOption::tcpserver(&name).ok_or_else(|| {
                    tcpserver_flow(
                        &TCPSERVER_ERROR_INVALID_NAME_VALUE,
                        format!("tcpserver: unsupported option '{name}'"),
                    )
                })?
            }
        };
        match option {
            NetConstructorOption::Timeout => {
                crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
                    &value_raw,
                    &INTEGER_TIMEOUT_EXTENSION,
                    BUILTIN_NAME,
                    "Timeout",
                )
                .await?;
                let timeout_value = gather_if_needed_async(&value_raw)
                    .await
                    .map_err(tcpserver_internal_source)?;
                options.timeout = parse_timeout(&timeout_value).map_err(|err| {
                    tcpserver_flow(
                        &TCPSERVER_ERROR_INVALID_NAME_VALUE,
                        format!("tcpserver: invalid Timeout value: {err}"),
                    )
                })?
            }
            NetConstructorOption::UserData => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &LEGACY_OPTIONS_EXTENSION,
                    BUILTIN_NAME,
                )?;
                options.user_data = value_raw;
            }
            NetConstructorOption::Name => {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &LEGACY_OPTIONS_EXTENSION,
                    BUILTIN_NAME,
                )?;
                let name_value = gather_if_needed_async(&value_raw)
                    .await
                    .map_err(tcpserver_internal_source)?;
                let text = string_scalar(&name_value, "Name").map_err(|err| {
                    tcpserver_flow(
                        &TCPSERVER_ERROR_INVALID_NAME_VALUE,
                        format!("tcpserver: invalid Name value: {err}"),
                    )
                })?;
                options.name = Some(text);
            }
            NetConstructorOption::ByteOrder => {
                let order_value = gather_if_needed_async(&value_raw)
                    .await
                    .map_err(tcpserver_internal_source)?;
                let raw_order = string_scalar(&order_value, "ByteOrder").map_err(|err| {
                    tcpserver_flow(
                        &TCPSERVER_ERROR_INVALID_NAME_VALUE,
                        format!("tcpserver: invalid ByteOrder value: {err}"),
                    )
                })?;
                let canon = canonicalize_byte_order(&raw_order).ok_or_else(|| {
                    tcpserver_flow(
                        &TCPSERVER_ERROR_INVALID_NAME_VALUE,
                        format!("tcpserver: unsupported ByteOrder '{raw_order}'"),
                    )
                })?;
                options.byte_order = canon.to_string();
            }
            _ => {
                return Err(tcpserver_flow(
                    &TCPSERVER_ERROR_INVALID_NAME_VALUE,
                    "tcpserver: unsupported option",
                ));
            }
        }
    }

    Ok(options)
}

fn build_tcpserver_object(
    runtime: Option<&crate::context::RuntimeContext>,
    authority: ForeignRef,
    requested_address: &str,
    local_addr: SocketAddr,
    options: &ParsedOptions,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    let server_address = local_addr.ip().to_string();
    let server_port = local_addr.port();
    let name = options
        .name
        .clone()
        .unwrap_or_else(|| format!("tcpserver:{server_address}:{server_port}"));

    let properties = [
        ("Type", Value::string("tcpserver".to_string())),
        ("Name", Value::string(name)),
        ("ServerAddress", Value::string(server_address.clone())),
        ("ServerPort", Value::Num(f64::from(server_port))),
        ("Port", Value::Num(f64::from(server_port))),
        (
            "RequestedAddress",
            Value::string(requested_address.to_string()),
        ),
        ("Connected", Value::Bool(false)),
        ("ClientAddress", Value::string(String::new())),
        (
            "ClientPort",
            Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty client port")),
        ),
        ("NumBytesAvailable", Value::Int(IntValue::I32(0))),
        ("BytesAvailableFcn", default_user_data()),
        ("BytesAvailableFcnMode", Value::string("byte".to_string())),
        ("BytesAvailableFcnCount", Value::Int(IntValue::I32(1))),
        ("ConnectionChangedFcn", default_user_data()),
        ("ByteOrder", Value::string(options.byte_order.clone())),
        ("EnableBroadcast", Value::Bool(false)),
        ("EnableMulticast", Value::Bool(false)),
        ("EnableReuseAddress", Value::Bool(false)),
        ("KeepAlive", Value::Bool(false)),
        ("Timeout", Value::Num(options.timeout)),
        ("UserData", options.user_data.clone()),
    ];

    crate::execution::object::construct_object_from_fields_with_native(
        runtime,
        runmat_builtins::TCPSERVER_CLASS.display_name(),
        properties.iter().map(|(name, value)| (*name, value)),
        Some(crate::execution::object::ObjectNativeInitialization::TcpAuthority(authority)),
    )
}

#[derive(Debug, Error)]
enum TimeoutParseError {
    #[error("Timeout must be a scalar")]
    NonScalar,
    #[error("Timeout must be a numeric scalar")]
    NonNumeric,
    #[error("Timeout must be a finite, non-negative scalar")]
    NonFinite,
}

fn parse_timeout(value: &Value) -> Result<f64, TimeoutParseError> {
    let timeout = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        Value::Tensor(t) if crate::builtins::common::tensor::is_scalar_tensor(t) => {
            if let Some(int) = t.integer_storage().and_then(|storage| storage.value_at(0)) {
                int.to_f64()
            } else {
                crate::builtins::common::tensor::tensor_value_f64(t, 0)
            }
        }
        Value::Tensor(_) => return Err(TimeoutParseError::NonScalar),
        _ => {
            return Err(TimeoutParseError::NonNumeric);
        }
    };

    if !timeout.is_finite() || timeout < 0.0 {
        return Err(TimeoutParseError::NonFinite);
    }
    Ok(timeout)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::io::net::test_support::CountingDownloadProvider;
    use crate::context::{RuntimeContext, RuntimeServicePorts};
    use crate::tcp::NativeTcpService;
    use runmat_builtins::TCPSERVER_DESCRIPTOR;
    use runmat_value::{IntegerStorage, Tensor, Value};
    use std::error::Error;
    use std::net::{TcpListener, TcpStream};
    use std::rc::Rc;
    use std::time::Duration;

    fn struct_field<'a>(value: &'a Value, name: &str) -> &'a Value {
        match value {
            Value::Object(object) => object
                .properties()
                .get(name)
                .unwrap_or_else(|| panic!("missing field {name}")),
            _ => panic!("expected TCP server object"),
        }
    }

    struct TestServer {
        value: crate::execution::RootedValue,
        _runtime: RuntimeContext,
        _driver: tokio::runtime::Runtime,
    }

    impl std::fmt::Debug for TestServer {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("TestTcpServer")
        }
    }

    impl std::ops::Deref for TestServer {
        type Target = Value;

        fn deref(&self) -> &Self::Target {
            &self.value
        }
    }

    fn assert_error_identifier(err: RuntimeError, expected: &str) {
        assert_eq!(err.identifier(), Some(expected));
    }

    fn run_tcpserver(address: Value, port: Value, rest: Vec<Value>) -> BuiltinResult<TestServer> {
        let driver = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("isolated TCP server driver");
        let service = Rc::new(NativeTcpService::new(driver.handle().clone()));
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(RuntimeServicePorts::default().with_tcp(service.clone()));
        let value = driver.block_on(tcpserver_builtin(
            Some(runtime.clone()),
            address,
            port,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))?;
        Ok(TestServer {
            value,
            _runtime: runtime,
            _driver: driver,
        })
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn known_invalid_options_reject_before_resident_bind_address_download() {
        let _lock = crate::builtins::common::test_support::accel_test_lock();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let owner = Box::leak(Box::new(CountingDownloadProvider::new()));
        let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(owner));
        let address = Value::GpuTensor(owner.resident());
        let err = run_tcpserver(
            address.clone(),
            Value::Num(80.0),
            vec![Value::from("EnableTransferDelay"), Value::Bool(true)],
        )
        .expect_err("known invalid option rejects before address download");
        assert_eq!(
            err.identifier(),
            TCPSERVER_ERROR_INVALID_NAME_VALUE.identifier
        );
        let err = run_tcpserver(address, Value::Num(80.0), vec![Value::from("Timeout")])
            .expect_err("incomplete option pair rejects before address download");
        assert_eq!(
            err.identifier(),
            TCPSERVER_ERROR_INVALID_NAME_VALUE.identifier
        );
        assert_eq!(owner.downloads(), 0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpserver_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = TCPSERVER_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"server = tcpserver(address, port)"));
        assert!(labels.contains(&"server = tcpserver(address, port, Name, Value, ...)"));
    }

    #[test]
    fn typed_port_parser_preserves_range_boundaries() {
        assert_eq!(
            parse_port(&Value::Int(IntValue::U16(u16::MAX))).unwrap(),
            u16::MAX
        );
        assert!(parse_port(&Value::Int(IntValue::I8(-1))).is_err());
        assert!(parse_port(&Value::Int(IntValue::U32(u16::MAX as u32 + 1))).is_err());
        assert!(parse_port(&Value::Int(IntValue::U64(u64::MAX))).is_err());

        let typed_max = Tensor::new_integer(IntegerStorage::U64(vec![u16::MAX as u64]), vec![1, 1])
            .expect("typed port");
        assert_eq!(parse_port(&Value::Tensor(typed_max)).unwrap(), u16::MAX);

        let typed_too_large =
            Tensor::new_integer(IntegerStorage::U64(vec![u16::MAX as u64 + 1]), vec![1, 1])
                .expect("typed port");
        assert!(parse_port(&Value::Tensor(typed_too_large)).is_err());

        let typed_negative =
            Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("typed port");
        assert!(parse_port(&Value::Tensor(typed_negative)).is_err());
    }

    #[test]
    fn typed_timeout_parser_reads_integer_storage_exactly() {
        let timeout =
            Tensor::new_integer(IntegerStorage::U16(vec![30]), vec![1, 1]).expect("timeout");

        assert_eq!(parse_timeout(&Value::Tensor(timeout)).unwrap(), 30.0);
    }

    fn net_guard() -> std::sync::MutexGuard<'static, ()> {
        crate::builtins::io::net::test_support::test_guard()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpserver_accepts_loopback_connection() {
        let _guard = net_guard();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);

        let result = run_tcpserver(Value::from("127.0.0.1"), Value::Num(0.0), Vec::new())
            .expect("tcpserver");
        let address = match struct_field(&result, "ServerAddress") {
            Value::String(runmat_value::StringElement::Text(s)) => s.clone(),
            other => panic!("expected ServerAddress string, got {other:?}"),
        };
        let port = match struct_field(&result, "ServerPort") {
            Value::Num(value) => *value as u16,
            other => panic!("expected double ServerPort, got {other:?}"),
        };

        TcpStream::connect((address.as_str(), port)).expect("connect to loopback server");
        drop(result);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpserver_applies_timeout_option() {
        let _guard = net_guard();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);

        let result = run_tcpserver(
            Value::from("localhost"),
            Value::Num(0.0),
            vec![Value::from("Timeout"), Value::Num(5.0)],
        )
        .expect("tcpserver");

        let timeout = match struct_field(&result, "Timeout") {
            Value::Num(n) => *n,
            other => panic!("expected numeric timeout, got {other:?}"),
        };
        assert!((timeout - 5.0).abs() < f64::EPSILON);

        drop(result);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpserver_supports_custom_name() {
        let _guard = net_guard();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);

        let result = run_tcpserver(
            Value::from("127.0.0.1"),
            Value::Num(0.0),
            vec![Value::from("Name"), Value::from("CustomListener")],
        )
        .expect("tcpserver");

        let name = match struct_field(&result, "Name") {
            Value::String(runmat_value::StringElement::Text(s)) => s.clone(),
            other => panic!("expected Name string, got {other:?}"),
        };
        assert_eq!(name, "CustomListener");

        drop(result);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpserver_accepts_byte_order_option() {
        let _guard = net_guard();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);

        let result = run_tcpserver(
            Value::from("127.0.0.1"),
            Value::Num(0.0),
            vec![Value::from("ByteOrder"), Value::from("big-endian")],
        )
        .expect("tcpserver");

        let order = match struct_field(&result, "ByteOrder") {
            Value::String(runmat_value::StringElement::Text(s)) => s.clone(),
            other => panic!("expected ByteOrder string, got {other:?}"),
        };
        assert_eq!(order, "big-endian");

        drop(result);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpserver_rejects_invalid_byte_order() {
        let _guard = net_guard();
        let err = run_tcpserver(
            Value::from("127.0.0.1"),
            Value::Num(8000.0),
            vec![Value::from("ByteOrder"), Value::from("middle-endian")],
        )
        .unwrap_err();
        assert_error_identifier(err, TCPSERVER_ERROR_INVALID_NAME_VALUE.identifier.unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpserver_accepts_scalar_tensor_port() {
        let _guard = net_guard();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);

        let tensor_port = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let result = run_tcpserver(
            Value::from("127.0.0.1"),
            Value::Tensor(tensor_port),
            Vec::new(),
        )
        .expect("tcpserver");

        let port = match struct_field(&result, "ServerPort") {
            Value::Num(value) => *value as u16,
            other => panic!("expected double ServerPort, got {other:?}"),
        };
        assert!(port > 0);

        drop(result);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpserver_rejects_invalid_port() {
        let err =
            run_tcpserver(Value::from("127.0.0.1"), Value::Num(-1.0), Vec::new()).unwrap_err();
        assert_error_identifier(err, TCPSERVER_ERROR_INVALID_PORT.identifier.unwrap());
    }

    #[test]
    fn tcpserver_extensions_reject_before_socket_access_in_compatibility_mode() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer_port = run_tcpserver(
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
            run_tcpserver(Value::from("127.0.0.1"), Value::Num(0.0), Vec::new()).unwrap_err();
        assert_eq!(zero_port.identifier(), ZERO_PORT_EXTENSION.error_identifier);

        let integer_timeout = run_tcpserver(
            Value::from("127.0.0.1"),
            Value::Num(1.0),
            vec![Value::from("Timeout"), Value::Int(IntValue::U8(5))],
        )
        .unwrap_err();
        assert_eq!(
            integer_timeout.identifier(),
            INTEGER_TIMEOUT_EXTENSION.error_identifier
        );

        let legacy_option = run_tcpserver(
            Value::from("127.0.0.1"),
            Value::Num(1.0),
            vec![Value::from("Name"), Value::from("server")],
        )
        .unwrap_err();
        assert_eq!(
            legacy_option.identifier(),
            LEGACY_OPTIONS_EXTENSION.error_identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpserver_requires_name_value_pairs() {
        let _guard = net_guard();
        let err = run_tcpserver(
            Value::from("127.0.0.1"),
            Value::Num(9000.0),
            vec![Value::from("Timeout")],
        )
        .unwrap_err();
        assert_error_identifier(err, TCPSERVER_ERROR_INVALID_NAME_VALUE.identifier.unwrap());
    }

    #[test]
    fn occupied_port_bind_failure_retains_io_cause() {
        let _guard = net_guard();
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("reserve loopback port");
        let port = listener.local_addr().expect("local port").port();
        let error = run_tcpserver(
            Value::from("127.0.0.1"),
            Value::Num(f64::from(port)),
            vec![],
        )
        .expect_err("occupied port rejects bind");
        assert_eq!(error.identifier(), TCPSERVER_ERROR_BIND_FAILED.identifier);
        let transport = Error::source(&error)
            .and_then(|source| source.downcast_ref::<crate::tcp::TcpTransportError>())
            .expect("bind failure retains typed transport cause");
        assert_eq!(transport.kind(), crate::tcp::TcpTransportErrorKind::Io);
        assert!(Error::source(transport)
            .and_then(|source| source.downcast_ref::<std::io::Error>())
            .is_some());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpserver_stores_userdata() {
        let _guard = net_guard();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);

        let mut user_struct_value = runmat_value::UnboundedStructValueBuilder::new();
        user_struct_value
            .fields
            .insert("tag".to_string(), Value::from("demo"));
        let user_struct = Value::Struct(user_struct_value.finish());

        let result = run_tcpserver(
            Value::from("127.0.0.1"),
            Value::Num(0.0),
            vec![Value::from("UserData"), user_struct.clone()],
        )
        .expect("tcpserver");

        let stored = struct_field(&result, "UserData");
        assert_eq!(stored, &user_struct);

        drop(result);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tcpserver_times_out_connect_attempt() {
        let _guard = net_guard();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);

        let result = run_tcpserver(
            Value::from("127.0.0.1"),
            Value::Num(0.0),
            vec![Value::from("Timeout"), Value::Num(1.5)],
        )
        .expect("tcpserver");
        let port = match struct_field(&result, "ServerPort") {
            Value::Num(value) => *value as u16,
            other => panic!("expected double ServerPort, got {other:?}"),
        };

        let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_millis(10)))
            .expect("set timeout");

        drop(result);
    }
}
