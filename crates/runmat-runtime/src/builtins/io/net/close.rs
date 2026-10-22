//! TCP close admission. Public object properties never authorize a socket.

use crate::context::{RuntimeContext, RuntimeTcpCloseScope, RuntimeTcpIssuer};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{
    CloseNetworkSelector, CLOSE_NETWORK_SELECTOR_EXTENSION, CLOSE_TCP_CELL_EXTENSION,
    TCPCLIENT_CLASS, TCPSERVER_CLASS,
};
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

fn selector_value(value: &Value) -> Option<RuntimeTcpCloseScope> {
    let text = crate::builtins::common::validation::value_text_scalar_string(value)?;
    selector(&text)
}

fn is_network_target(value: &Value) -> bool {
    match value {
        Value::Object(object) => {
            object.class_name().is(TCPCLIENT_CLASS) || object.class_name().is(TCPSERVER_CLASS)
        }
        Value::Struct(_) => true,
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => {
            selector_value(value).is_some()
        }
        Value::Cell(cell) => !cell.data().is_empty() && cell.data().iter().all(is_network_target),
        _ => false,
    }
}

fn collect(value: &Value, targets: &mut Vec<Target>) -> BuiltinResult<()> {
    match value {
        Value::Object(object)
            if object.class_name().is(TCPCLIENT_CLASS)
                || object.class_name().is(TCPSERVER_CLASS) =>
        {
            let authority = object.tcp_authority().cloned().ok_or_else(invalid_handle)?;
            targets.push(Target::Endpoint(authority));
        }
        Value::Struct(_) => return Err(invalid_handle()),
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => {
            let scope = selector_value(value).ok_or_else(invalid_argument)?;
            crate::compatibility::ensure_builtin_extension_enabled(
                &CLOSE_NETWORK_SELECTOR_EXTENSION,
                "close",
            )?;
            targets.push(Target::Scope(scope));
        }
        Value::Cell(cell) if !cell.data().is_empty() => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &CLOSE_TCP_CELL_EXTENSION,
                "close",
            )?;
            for value in cell.data() {
                collect(value, targets)?;
            }
        }
        _ => return Err(invalid_argument()),
    }
    Ok(())
}

pub(crate) async fn close_if_network_targets(
    runtime: Option<&RuntimeContext>,
    args: crate::call::FunctionArgumentView<'_>,
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
    let mut work = super::transport::NetCooperation::new(runtime, "close");
    let mut any_closed = false;
    for target in targets {
        work.completed(0).await?;
        let result = match target {
            Target::Endpoint(endpoint) => {
                let future = service.close(issuer.clone(), endpoint);
                let outcome = work
                    .await_service(future)
                    .await
                    .map_err(|failure| failure.into_runtime_error("close"))?;
                outcome.map_err(transport_error)?
            }
            Target::Scope(scope) => {
                let future = service.close_scope(issuer.clone(), scope);
                let outcome = work
                    .await_service(future)
                    .await
                    .map_err(|failure| failure.into_runtime_error("close"))?;
                outcome.map_err(transport_error)? != 0
            }
        };
        any_closed |= result;
        work.completed(1).await?;
    }
    Ok(Some(if any_closed { 1.0 } else { 0.0 }))
}

fn transport_error(source: crate::tcp::TcpTransportError) -> RuntimeError {
    build_runtime_error(format!("close: TCP close failed ({source})"))
        .with_builtin("close")
        .with_source(source)
        .build()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "close/admission_tests.rs"]
mod admission_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{CellArray, CharArray, ObjectInstance, StructValue};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn graphics_all_and_bare_close_never_claim_tcp_scope() {
        assert!(futures::executor::block_on(close_if_network_targets(
            None,
            crate::call::FunctionArgumentView::from_slice(&[])
        ))
        .unwrap()
        .is_none());
        assert!(futures::executor::block_on(close_if_network_targets(
            None,
            crate::call::FunctionArgumentView::from_slice(&[Value::String("all".into())])
        ))
        .unwrap()
        .is_none());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn scalar_selector_compatibility_precedes_browser_target_rejection() {
        let trailing = Value::CharArray(
            CharArray::from_unicode_scalars_with_shape("clients".chars().collect(), vec![1, 7, 1])
                .expect("trailing singleton"),
        );
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = futures::executor::block_on(close_if_network_targets(
                None,
                crate::call::FunctionArgumentView::from_slice(std::slice::from_ref(&trailing)),
            ))
            .expect_err("selector extension is rejected before target service");
            assert_eq!(
                error.identifier(),
                CLOSE_NETWORK_SELECTOR_EXTENSION.error_identifier
            );
        }
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        #[cfg(target_arch = "wasm32")]
        {
            let error = futures::executor::block_on(close_if_network_targets(None, &[trailing]))
                .expect_err("native TCP is unavailable in browser");
            assert_eq!(
                error.identifier(),
                runmat_builtins::NET_ERROR_UNSUPPORTED_TARGET.identifier
            );
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = trailing;
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn forged_tcp_class_and_legacy_struct_reject_before_service() {
        let forged = Value::Object(ObjectInstance::new(TCPCLIENT_CLASS));
        let error = futures::executor::block_on(close_if_network_targets(
            None,
            crate::call::FunctionArgumentView::from_slice(&[forged]),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:close:InvalidHandle"));
        let legacy = Value::Struct(StructValue::new());
        let error = futures::executor::block_on(close_if_network_targets(
            None,
            crate::call::FunctionArgumentView::from_slice(&[legacy]),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:close:InvalidHandle"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nested_invalid_tcp_container_has_no_partial_close() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let cell = CellArray::new(
            vec![
                Value::String("clients".into()),
                Value::Object(ObjectInstance::new(TCPSERVER_CLASS)),
            ],
            1,
            2,
        )
        .unwrap();
        let error = futures::executor::block_on(close_if_network_targets(
            None,
            crate::call::FunctionArgumentView::from_slice(&[Value::Cell(cell)]),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:close:InvalidHandle"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_service_closes_exact_session_objects_and_preserves_second_status() {
        use crate::context::{
            RuntimeServicePorts, RuntimeTcpBind, RuntimeTcpByteOrder, RuntimeTcpConnect,
            RuntimeTcpInvocation, RuntimeTcpService,
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
        let server = driver
            .block_on(service.bind(
                RuntimeTcpInvocation::from_context(Some(&runtime)),
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
                RuntimeTcpInvocation::from_context(Some(&runtime)),
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
                .with_service_ports(RuntimeServicePorts::default().with_tcp(service.clone()));
        assert!(driver
            .block_on(close_if_network_targets(
                Some(&other_session),
                crate::call::FunctionArgumentView::from_slice(std::slice::from_ref(&client))
            ))
            .is_err());
        let foreign = driver
            .block_on(service.connect(
                RuntimeTcpInvocation::from_context(Some(&other_session)),
                RuntimeTcpConnect {
                    host: "127.0.0.1".into(),
                    port: server.local_addr.port(),
                    connect_timeout: Some(std::time::Duration::from_secs(2)),
                    io_timeout: None,
                    byte_order: RuntimeTcpByteOrder::Little,
                },
            ))
            .expect("second session connects");
        let foreign = Value::Object(
            ObjectInstance::new(TCPCLIENT_CLASS).with_tcp_authority(foreign.authority),
        );
        let error = driver
            .block_on(close_if_network_targets(
                Some(&runtime),
                crate::call::FunctionArgumentView::from_slice(&[client.clone(), foreign.clone()]),
            ))
            .expect_err("later foreign target must not replay or roll back earlier close");
        assert!(error.message().contains("TCP close failed"));
        assert_eq!(
            driver
                .block_on(close_if_network_targets(
                    Some(&runtime),
                    crate::call::FunctionArgumentView::from_slice(std::slice::from_ref(&client))
                ))
                .unwrap(),
            Some(0.0),
        );
        assert_eq!(
            driver
                .block_on(close_if_network_targets(
                    Some(&runtime),
                    crate::call::FunctionArgumentView::from_slice(&[client])
                ))
                .unwrap(),
            Some(0.0)
        );
        assert_eq!(
            driver
                .block_on(close_if_network_targets(
                    Some(&other_session),
                    crate::call::FunctionArgumentView::from_slice(&[foreign])
                ))
                .unwrap(),
            Some(1.0),
        );
    }
}
