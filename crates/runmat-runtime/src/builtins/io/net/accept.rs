//! MATLAB-compatible `accept` builtin for RunMat networking.

use runmat_builtins::{
    BuiltinErrorDescriptor, NetConstructorOption, ACCEPT_ERROR_ACCEPT_FAILED,
    ACCEPT_ERROR_INTERNAL, ACCEPT_ERROR_INVALID_NAME_VALUE, ACCEPT_ERROR_INVALID_SERVER,
    ACCEPT_ERROR_TIMEOUT,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ForeignRef, IntValue, Value};

use super::control::{default_user_data, parse_timeout_value};
use crate::context::{RuntimeTcpByteOrder, RuntimeTcpInvocation};
use crate::tcp::TcpTransportErrorKind;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

use std::net::SocketAddr;
use std::time::Duration;

const BUILTIN_NAME: &str = "accept";

#[runtime_builtin(
    name = "accept",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::net::accept"
)]
pub(crate) async fn accept_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    server: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    let server_authority = super::receiver::authority(&server, runmat_builtins::TCPSERVER_CLASS)
        .ok_or_else(|| {
            accept_flow(
                &ACCEPT_ERROR_INVALID_SERVER,
                "accept: expected a live tcpserver object",
            )
        })?;
    let prepared_options = preflight_accept_options(rest.view())?;
    let service = super::transport::service(runtime.as_ref(), BUILTIN_NAME)?;
    let mut work = super::transport::NetCooperation::new(runtime.as_ref(), BUILTIN_NAME);
    let options = parse_accept_options(rest, prepared_options).await?;
    if let Some(timeout) = options.timeout {
        validate_timeout(timeout)?;
    }
    let endpoint = work
        .await_service(service.accept(
            RuntimeTcpInvocation::from_context(runtime.as_ref()),
            server_authority,
            options.timeout.and_then(|seconds| {
                seconds
                    .is_finite()
                    .then(|| Duration::from_secs_f64(seconds))
            }),
        ))
        .await
        .map_err(|failure| failure.into_runtime_error(BUILTIN_NAME))?
        .map_err(|err| {
            if err.kind() == TcpTransportErrorKind::Timeout {
                super::error::with_source_message(
                    BUILTIN_NAME,
                    &ACCEPT_ERROR_TIMEOUT,
                    "accept: timed out waiting for a client connection",
                    err,
                )
            } else {
                super::error::with_source(
                    BUILTIN_NAME,
                    &ACCEPT_ERROR_ACCEPT_FAILED,
                    format!("accept: {err}"),
                    err,
                )
            }
        })?;
    work.completed(0).await?;
    build_tcpclient_value(
        runtime.as_ref(),
        endpoint.authority,
        endpoint.local_addr,
        endpoint.peer_addr,
        endpoint.io_timeout,
        endpoint.byte_order,
    )
}

#[derive(Default)]
struct AcceptOptions {
    timeout: Option<f64>,
}

fn preflight_accept_options(
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Vec<Option<NetConstructorOption>>> {
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
    rest: crate::call::BuiltinArguments,
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

fn build_tcpclient_value(
    runtime: Option<&crate::context::RuntimeContext>,
    authority: ForeignRef,
    local_addr: SocketAddr,
    peer_addr: SocketAddr,
    timeout: Option<Duration>,
    byte_order: RuntimeTcpByteOrder,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    let properties = [
        ("Type", Value::string("tcpclient".to_string())),
        ("Address", Value::string(peer_addr.ip().to_string())),
        ("Port", Value::Int(IntValue::U16(peer_addr.port()))),
        ("ServerAddress", Value::string(local_addr.ip().to_string())),
        ("ServerPort", Value::Int(IntValue::U16(local_addr.port()))),
        ("Connected", Value::Bool(true)),
        ("Status", Value::string("connected".to_string())),
        ("NumBytesAvailable", Value::Int(IntValue::I32(0))),
        ("BytesAvailableFcn", default_user_data()),
        ("BytesAvailableFcnMode", Value::string("byte".to_string())),
        ("BytesAvailableFcnCount", Value::Int(IntValue::I32(1))),
        (
            "ByteOrder",
            Value::String(
                match byte_order {
                    RuntimeTcpByteOrder::Little => "little-endian",
                    RuntimeTcpByteOrder::Big => "big-endian",
                }
                .into(),
            ),
        ),
        (
            "Timeout",
            Value::Num(timeout.map_or(f64::INFINITY, |duration| duration.as_secs_f64())),
        ),
        ("UserData", default_user_data()),
    ];
    crate::execution::object::construct_object_from_fields_with_native(
        runtime,
        runmat_builtins::TCPCLIENT_CLASS.display_name(),
        properties.iter().map(|(name, value)| (*name, value)),
        Some(crate::execution::object::ObjectNativeInitialization::TcpAuthority(authority)),
    )
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
    use super::super::control::TimeoutParseError;
    use super::super::tcpserver::tcpserver_builtin;
    use super::*;
    use crate::builtins::io::net::test_support::CountingDownloadProvider;
    use crate::context::{RuntimeContext, RuntimeServicePorts};
    use crate::tcp::NativeTcpService;
    use runmat_builtins::{ACCEPT_DESCRIPTOR, ACCEPT_INTEGER_CAPABILITIES};
    use runmat_value::{IntegerStorage, ObjectInstance, Tensor, Value};
    use std::net::TcpStream;
    use std::rc::Rc;
    use std::thread;
    use std::time::Duration;

    fn struct_field<'a>(value: &'a Value, name: &str) -> &'a Value {
        match value {
            Value::Object(object) => object
                .properties()
                .get(name)
                .unwrap_or_else(|| panic!("missing field {name}")),
            _ => panic!("expected TCP object"),
        }
    }

    fn assert_error_identifier(err: RuntimeError, expected: &str) {
        assert_eq!(err.identifier(), Some(expected));
    }

    #[test]
    fn detached_server_object_is_rejected() {
        let (runtime, _driver) = test_runtime();
        let server = Value::Object(ObjectInstance::new(runmat_builtins::TCPSERVER_CLASS));
        let err = run_accept(&runtime, server, Vec::new()).unwrap_err();
        assert_error_identifier(err, ACCEPT_ERROR_INVALID_SERVER.identifier.unwrap());
    }

    fn test_runtime() -> (RuntimeContext, tokio::runtime::Runtime) {
        let driver = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("isolated TCP host driver");
        let service = Rc::new(NativeTcpService::new(driver.handle().clone()));
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(RuntimeServicePorts::default().with_tcp(service));
        (runtime, driver)
    }

    fn run_accept(
        runtime: &RuntimeContext,
        server: Value,
        rest: Vec<Value>,
    ) -> BuiltinResult<crate::execution::RootedValue> {
        futures::executor::block_on(accept_builtin(
            Some(runtime.clone()),
            server,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn known_invalid_timeout_rejects_before_resident_server_download() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let (runtime, _driver) = test_runtime();
        let _lock = crate::builtins::common::test_support::accel_test_lock();
        let owner = Box::leak(Box::new(CountingDownloadProvider::new()));
        let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(owner));
        let server = Value::GpuTensor(owner.resident());
        let err = run_accept(
            &runtime,
            server.clone(),
            vec![Value::from("Timeout"), Value::Num(-1.0)],
        )
        .expect_err("invalid timeout rejects before resident server download");
        assert_eq!(err.identifier(), ACCEPT_ERROR_INVALID_SERVER.identifier);
        let err = run_accept(
            &runtime,
            server,
            vec![Value::from("Backlog"), Value::Num(1.0)],
        )
        .expect_err("unknown option rejects before resident server download");
        assert_eq!(err.identifier(), ACCEPT_ERROR_INVALID_SERVER.identifier);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let server = _driver
                .block_on(tcpserver_builtin(
                    Some(runtime.clone()),
                    Value::from("127.0.0.1"),
                    Value::Num(0.0),
                    crate::builtins::common::test_support::builtin_arguments(vec![]),
                ))
                .expect("live server authority");
            for options in [
                vec![Value::from("Timeout"), Value::Num(-1.0)],
                vec![Value::from("Backlog"), Value::Num(1.0)],
            ] {
                let err = run_accept(&runtime, server.value().clone(), options)
                    .expect_err("invalid controls reject on valid server authority");
                assert_eq!(err.identifier(), ACCEPT_ERROR_INVALID_NAME_VALUE.identifier);
            }
        }
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

    fn run_tcpserver(
        runtime: &RuntimeContext,
        address: Value,
        port: Value,
        rest: Vec<Value>,
    ) -> BuiltinResult<crate::execution::RootedValue> {
        futures::executor::block_on(tcpserver_builtin(
            Some(runtime.clone()),
            address,
            port,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }

    fn net_guard() -> std::sync::MutexGuard<'static, ()> {
        super::super::test_support::test_guard()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn accept_rejects_non_struct() {
        let (runtime, _driver) = test_runtime();
        let _guard = net_guard();
        let err = run_accept(&runtime, Value::Num(1.0), Vec::new()).unwrap_err();
        assert_error_identifier(err, ACCEPT_ERROR_INVALID_SERVER.identifier.unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn accept_establishes_client_connection() {
        let (runtime, _driver) = test_runtime();
        let _guard = net_guard();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let server_value = run_tcpserver(
            &runtime,
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

        let client =
            run_accept(&runtime, server_value.value().clone(), Vec::new()).expect("accept");
        let stream = handle.join().expect("client thread");
        drop(stream);

        match struct_field(&client, "Connected") {
            Value::Bool(flag) => assert!(*flag),
            other => panic!("expected Connected bool, got {other:?}"),
        }
        match struct_field(&client, "Address") {
            Value::String(runmat_value::StringElement::Text(addr)) => assert_eq!(addr, "127.0.0.1"),
            other => panic!("expected Address string, got {other:?}"),
        }
        match struct_field(&client, "Timeout") {
            Value::Num(n) => assert_eq!(*n, 10.0),
            other => panic!("expected Timeout numeric, got {other:?}"),
        }

        drop(client);
        drop(server_value);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn accept_times_out_when_no_client_connects() {
        let (runtime, _driver) = test_runtime();
        let _guard = net_guard();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let server_value = run_tcpserver(
            &runtime,
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(0)),
            Vec::new(),
        )
        .expect("tcpserver");
        let err = run_accept(
            &runtime,
            server_value.value().clone(),
            vec![Value::from("Timeout"), Value::Num(0.05)],
        )
        .unwrap_err();
        assert_eq!(err.identifier(), ACCEPT_ERROR_TIMEOUT.identifier);
        assert!(err
            .message()
            .contains("timed out waiting for a client connection"));
        assert_eq!(err.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        assert!(
            std::error::Error::source(&err).is_some(),
            "timeout retains its typed transport cause"
        );
        drop(server_value);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn accept_rejects_invalid_timeout_name_value() {
        let (runtime, _driver) = test_runtime();
        let _guard = net_guard();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let server_value = run_tcpserver(
            &runtime,
            Value::from("127.0.0.1"),
            Value::Int(IntValue::I32(0)),
            Vec::new(),
        )
        .expect("tcpserver");
        let err = run_accept(
            &runtime,
            server_value.value().clone(),
            vec![Value::from("Timeout"), Value::Num(-1.0)],
        )
        .unwrap_err();
        assert_error_identifier(err, ACCEPT_ERROR_INVALID_NAME_VALUE.identifier.unwrap());
        drop(server_value);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn accept_respects_per_call_timeout_override() {
        let (runtime, _driver) = test_runtime();
        let _guard = net_guard();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let server_value = run_tcpserver(
            &runtime,
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
            &runtime,
            server_value.value().clone(),
            vec![Value::from("Timeout"), Value::Int(IntValue::U8(1))],
        )
        .expect("accept");
        handle.join().expect("join");
        let timeout_val = match struct_field(&client, "Timeout") {
            Value::Num(n) => *n,
            other => panic!("expected Timeout numeric, got {other:?}"),
        };
        assert_eq!(timeout_val, 1.0);
        drop(client);
        drop(server_value);
    }
}
