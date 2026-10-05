//! TCP close admission. Public object properties never authorize a socket.

use crate::context::{RuntimeContext, RuntimeTcpCloseScope, RuntimeTcpIssuer};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{CloseNetworkSelector, TCPCLIENT_CLASS, TCPSERVER_CLASS};
use runmat_value::{ForeignRef, Value};

#[derive(Clone)]
enum Target {
    Endpoint(ForeignRef),
    Scope(RuntimeTcpCloseScope),
}

fn invalid_handle() -> RuntimeError {
    build_runtime_error("close: invalid TCP handle")
        .with_builtin("close")
        .with_identifier("RunMat:close:InvalidHandle")
        .build()
}

fn invalid_argument() -> RuntimeError {
    build_runtime_error("close: invalid networking target")
        .with_builtin("close")
        .with_identifier("RunMat:close:InvalidArgument")
        .build()
}

fn selector(text: &str) -> Option<RuntimeTcpCloseScope> {
    match CloseNetworkSelector::parse(text)? {
        CloseNetworkSelector::Clients => Some(RuntimeTcpCloseScope::Clients),
        CloseNetworkSelector::Servers => Some(RuntimeTcpCloseScope::Servers),
    }
}

fn is_network_target(value: &Value) -> bool {
    match value {
        Value::Object(object) => {
            object.class_name.is(TCPCLIENT_CLASS) || object.class_name.is(TCPSERVER_CLASS)
        }
        Value::Struct(_) => true,
        Value::String(text) => selector(text).is_some(),
        Value::CharArray(chars) if chars.rows == 1 => {
            selector(&chars.data.iter().collect::<String>()).is_some()
        }
        Value::StringArray(strings) if strings.data.len() == 1 => {
            selector(&strings.data[0]).is_some()
        }
        Value::Cell(cell) => !cell.data.is_empty() && cell.data.iter().all(is_network_target),
        _ => false,
    }
}

fn collect(value: &Value, targets: &mut Vec<Target>) -> BuiltinResult<()> {
    match value {
        Value::Object(object)
            if object.class_name.is(TCPCLIENT_CLASS) || object.class_name.is(TCPSERVER_CLASS) =>
        {
            let authority = object.tcp_authority().cloned().ok_or_else(invalid_handle)?;
            targets.push(Target::Endpoint(authority));
        }
        Value::Struct(_) => return Err(invalid_handle()),
        Value::String(text) => {
            targets.push(Target::Scope(selector(text).ok_or_else(invalid_argument)?));
        }
        Value::CharArray(chars) if chars.rows == 1 => {
            let text: String = chars.data.iter().collect();
            targets.push(Target::Scope(selector(&text).ok_or_else(invalid_argument)?));
        }
        Value::StringArray(strings) if strings.data.len() == 1 => {
            targets.push(Target::Scope(
                selector(&strings.data[0]).ok_or_else(invalid_argument)?,
            ));
        }
        Value::Cell(cell) if !cell.data.is_empty() => {
            for value in &cell.data {
                collect(value, targets)?;
            }
        }
        _ => return Err(invalid_argument()),
    }
    Ok(())
}

pub(crate) async fn close_if_network_targets(
    runtime: Option<&RuntimeContext>,
    args: &[Value],
) -> BuiltinResult<Option<f64>> {
    if args.is_empty() || !args.iter().all(is_network_target) {
        return Ok(None);
    }
    let mut targets = Vec::new();
    for value in args {
        collect(value, &mut targets)?;
    }
    let service = super::transport::service(runtime, "close")?;
    let issuer = RuntimeTcpIssuer::from_context(runtime);
    let mut any_closed = false;
    for target in targets {
        if let Some(runtime) = runtime {
            runtime.cpu_cooperation().completed(0).await?;
        }
        let result = match target {
            Target::Endpoint(endpoint) => {
                let future = service.close(issuer.clone(), endpoint);
                let outcome = if let Some(runtime) = runtime {
                    runtime
                        .cpu_cooperation()
                        .await_cancellable(future)
                        .await
                        .map_err(|source| source.into_runtime_error())?
                } else {
                    future.await
                };
                outcome.map_err(transport_error)?
            }
            Target::Scope(scope) => {
                let future = service.close_scope(issuer.clone(), scope);
                let outcome = if let Some(runtime) = runtime {
                    runtime
                        .cpu_cooperation()
                        .await_cancellable(future)
                        .await
                        .map_err(|source| source.into_runtime_error())?
                } else {
                    future.await
                };
                outcome.map_err(transport_error)? != 0
            }
        };
        any_closed |= result;
        if let Some(runtime) = runtime {
            runtime.cpu_cooperation().completed(0).await?;
        }
    }
    Ok(Some(if any_closed { 1.0 } else { 0.0 }))
}

fn transport_error(source: crate::tcp::TcpTransportError) -> RuntimeError {
    build_runtime_error(format!("close: TCP close failed ({source})"))
        .with_builtin("close")
        .with_source(source)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{CellArray, ObjectInstance, StructValue};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn graphics_all_and_bare_close_never_claim_tcp_scope() {
        assert!(
            futures::executor::block_on(close_if_network_targets(None, &[]))
                .unwrap()
                .is_none()
        );
        assert!(futures::executor::block_on(close_if_network_targets(
            None,
            &[Value::String("all".into())]
        ))
        .unwrap()
        .is_none());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn forged_tcp_class_and_legacy_struct_reject_before_service() {
        let forged = Value::Object(ObjectInstance::new(TCPCLIENT_CLASS));
        let error =
            futures::executor::block_on(close_if_network_targets(None, &[forged])).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:close:InvalidHandle"));
        let legacy = Value::Struct(StructValue::new());
        let error =
            futures::executor::block_on(close_if_network_targets(None, &[legacy])).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:close:InvalidHandle"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nested_invalid_tcp_container_has_no_partial_close() {
        let cell = CellArray::new(
            vec![
                Value::String("clients".into()),
                Value::Object(ObjectInstance::new(TCPSERVER_CLASS)),
            ],
            1,
            2,
        )
        .unwrap();
        let error =
            futures::executor::block_on(close_if_network_targets(None, &[Value::Cell(cell)]))
                .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:close:InvalidHandle"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_service_closes_exact_session_objects_and_preserves_second_status() {
        use crate::context::{
            RuntimeServicePorts, RuntimeTcpBind, RuntimeTcpByteOrder, RuntimeTcpConnect,
            RuntimeTcpService,
        };
        use crate::tcp::NativeTcpService;
        use std::rc::Rc;

        let driver = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("isolated TCP driver");
        let service = Rc::new(NativeTcpService::new(driver.handle().clone()));
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(RuntimeServicePorts::default().with_tcp(service.clone()));
        let issuer = RuntimeTcpIssuer::from_context(Some(&runtime));
        let server = driver
            .block_on(service.bind(
                issuer.clone(),
                RuntimeTcpBind {
                    address: "127.0.0.1".into(),
                    port: 0,
                    io_timeout: None,
                    byte_order: RuntimeTcpByteOrder::Little,
                },
            ))
            .expect("bind");
        let client = driver
            .block_on(service.connect(
                issuer,
                RuntimeTcpConnect {
                    host: "127.0.0.1".into(),
                    port: server.local_addr.port(),
                    connect_timeout: Some(std::time::Duration::from_secs(2)),
                    io_timeout: None,
                    byte_order: RuntimeTcpByteOrder::Little,
                },
            ))
            .expect("connect");
        let client = Value::Object(
            ObjectInstance::new(TCPCLIENT_CLASS).with_tcp_authority(client.authority),
        );
        let other_session =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(RuntimeServicePorts::default().with_tcp(service));
        assert!(driver
            .block_on(close_if_network_targets(
                Some(&other_session),
                &[client.clone()]
            ))
            .is_err());
        assert_eq!(
            driver
                .block_on(close_if_network_targets(Some(&runtime), &[client.clone()]))
                .unwrap(),
            Some(1.0)
        );
        assert_eq!(
            driver
                .block_on(close_if_network_targets(Some(&runtime), &[client]))
                .unwrap(),
            Some(0.0)
        );
    }
}
